import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


TARGET_EXTENSIONS = frozenset({".mp4"})
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
USING_CUDA = True
THRESHOLD_PSNR = 40.0
THRESHOLD_SSIM = 0.93
COMPRESS_RATIOS = (0.6, 0.7, 0.8, 0.9)
RESULT_COLUMNS = (
    "file_name", "status", "reason", "psnr_avg", "psnr_y", "ssim_all", "ssim_y",
    "orig_file_size", "trans_file_size", "ratio", "orig_video_bitrate",
    "target_video_bitrate", "trans_video_bitrate", "attempts",
)


@dataclass(frozen=True)
class QualityMetrics:
    psnr_avg: float
    psnr_y: float
    ssim_all: float
    ssim_y: float

    @classmethod
    def empty(cls) -> "QualityMetrics":
        # Invalid reports must fail even when the thresholds are zero.
        return cls(math.nan, math.nan, math.nan, math.nan)

    def meets(self, psnr_threshold: float, ssim_threshold: float) -> bool:
        return (
            self.psnr_avg >= psnr_threshold
            and self.psnr_y >= psnr_threshold
            and 0 <= self.ssim_all <= 1
            and 0 <= self.ssim_y <= 1
            and self.ssim_all >= ssim_threshold
            and self.ssim_y >= ssim_threshold
        )


@dataclass(frozen=True)
class VideoInfo:
    bitrate: int
    width: int
    height: int
    pixel_format: str
    frames: int
    duration: float
    stream_types: tuple[str, ...]
    video_properties: tuple[tuple[str, str], ...] = ()
    hdr_metadata: tuple[str, ...] = ()


class ContentTranscoding:
    def __init__(
        self, args: argparse.Namespace, *, output_directory: Path | None = None,
        event_callback: Callable[[dict[str, object]], None] | None = None,
    ):
        if args is None or not getattr(args, "path", None):
            raise ValueError("A target path is required")
        self.target_path = Path(args.path).resolve()
        self.psnr_threshold = float(getattr(args, "psnr", THRESHOLD_PSNR))
        self.ssim_threshold = float(getattr(args, "ssim", THRESHOLD_SSIM))
        self.ratios = tuple(getattr(args, "ratios", COMPRESS_RATIOS))
        self.encoder_mode = getattr(args, "encoder", None) or (
            "cuda" if getattr(args, "use_cuda", USING_CUDA) else "cpu"
        )
        self.use_cuda = self.encoder_mode != "cpu"
        if not math.isfinite(self.psnr_threshold) or self.psnr_threshold < 0:
            raise ValueError("PSNR must be finite and nonnegative")
        if not math.isfinite(self.ssim_threshold) or not 0 <= self.ssim_threshold <= 1:
            raise ValueError("SSIM must be between 0 and 1")
        if not self.ratios or any(not math.isfinite(r) or not 0 < r < 1 for r in self.ratios):
            raise ValueError("Bitrate ratios must be between 0 and 1 (exclusive)")
        if tuple(sorted(set(self.ratios))) != self.ratios:
            raise ValueError("Bitrate ratios must be unique and strictly increasing")
        if self.encoder_mode not in {"auto", "cpu", "cuda"}:
            raise ValueError("Encoder must be auto, cpu or cuda")
        self.output_root = Path(output_directory).resolve() if output_directory else self.target_path
        self.temp_path = self.output_root / "temporary"
        self.done_path = self.output_root / "done"
        self.event_callback = event_callback
        self._source_files: dict[Path, Path] = {}
        self._results: list[dict[str, object]] = []
        self._measured_metrics: dict[tuple[Path, Path], QualityMetrics] = {}

    def _emit(self, event: str, **data: object) -> None:
        if self.event_callback is not None:
            self.event_callback({"event": event, **data})

    def _prepare_directories(self) -> None:
        if not self.target_path.is_dir():
            raise ValueError(f"Invalid target directory: {self.target_path}")
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.done_path.mkdir(parents=True, exist_ok=True)
        for path in (self.temp_path, self.done_path):
            if path.is_symlink() or path.resolve().parent != self.output_root:
                raise ValueError(f"Output directory must be inside the target: {path}")

    def _gather_target_files(self) -> list[Path]:
        return sorted(
            path for path in self.target_path.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in TARGET_EXTENSIONS
        )

    def _build_transcode_command(
        self, target_file: Path, transcoded_file: Path, video_bitrate: int
    ) -> list[str]:
        command = [
            FFMPEG, "-nostdin", "-y", "-v", "error", "-xerror",
            "-noautorotate", "-i", str(target_file),
            "-map", "0", "-map_metadata", "0", "-map_chapters", "0", "-c", "copy",
            "-c:v:0", "hevc_nvenc" if self.use_cuda else "libx265",
            "-b:v:0", str(video_bitrate), "-preset:v:0", "p5" if self.use_cuda else "medium",
            "-fps_mode:v:0", "passthrough", "-enc_time_base:v:0", "demux",
            "-tag:v:0", "hvc1", "-movflags", "+faststart",
        ]
        if self.use_cuda:
            command.extend(("-rc:v:0", "vbr", "-multipass:v:0", "fullres"))
        command.append(str(transcoded_file))
        return command

    def _transcode(self, target_file: Path, video_bitrate: int) -> Path | None:
        transcoded_file = self.temp_path / target_file.name
        for attempt in range(2):
            transcoded_file.unlink(missing_ok=True)
            try:
                subprocess.run(
                    self._build_transcode_command(target_file, transcoded_file, video_bitrate),
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                if self._is_nonempty_file(transcoded_file):
                    return transcoded_file
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"[Error] Failed to transcode {target_file}: {self._error_text(error)}")
            if attempt == 0 and self.encoder_mode == "auto" and self.use_cuda:
                self.use_cuda = False
                self._emit("encoder_changed", encoder="cpu")
                print("[Fallback] Retrying with CPU libx265")
            else:
                break
        transcoded_file.unlink(missing_ok=True)
        return None

    @staticmethod
    def _error_text(error: Exception) -> str:
        stderr = getattr(error, "stderr", None)
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return str(stderr or error).strip()

    def _metric_report_paths(self, target_file: Path) -> tuple[Path, Path]:
        # Keep user filenames out of FFmpeg's filter syntax.
        key = hashlib.sha256(target_file.name.encode("utf-8")).hexdigest()
        return self.temp_path / f"{key}_psnr.txt", self.temp_path / f"{key}_ssim.txt"

    def _measure(
        self, anchor_file: Path, target_file: Path | None
    ) -> tuple[Path | None, Path | None]:
        if target_file is None:
            return None, None
        self._emit("phase", phase="measuring", file_name=target_file.name)
        psnr_report, ssim_report = self._metric_report_paths(target_file)
        self._measured_metrics.pop((psnr_report, ssim_report), None)
        self._remove_files((psnr_report, ssim_report))
        # Align by frame index, independent of container start times/timebases.
        # Frame counts and duration are checked separately before acceptance.
        filter_graph = (
            "[0:v:0]settb=AVTB,setpts=N[ref];"
            "[1:v:0]settb=AVTB,setpts=N,split=2[dist1][dist2];"
            "[ref]split=2[ref1][ref2];"
            f"[dist1][ref1]psnr=f={psnr_report.name}:stats_version=2:output_max=1:"
            "shortest=1:repeatlast=0[p];"
            f"[dist2][ref2]ssim=f={ssim_report.name}:shortest=1:repeatlast=0[s]"
        )
        command = [
            FFMPEG, "-nostdin", "-nostats", "-v", "info", "-xerror",
            "-noautorotate", "-i", str(anchor_file.resolve()),
            "-noautorotate", "-i", str(target_file.resolve()),
            "-filter_complex", filter_graph, "-map", "[p]", "-map", "[s]",
            "-fps_mode", "passthrough", "-an", "-f", "null", "-",
        ]
        try:
            completed = subprocess.run(command, check=True, cwd=self.temp_path,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as error:
            self._remove_files((psnr_report, ssim_report))
            print(f"[Error] Failed to measure {target_file}: {self._error_text(error)}")
            return None, None
        if not all(self._is_nonempty_file(p) for p in (psnr_report, ssim_report)):
            self._remove_files((psnr_report, ssim_report))
            return None, None
        # FFmpeg's final summaries use accumulated, unrounded MSE. Per-frame
        # logs round PSNR to two decimals and are used to verify frame coverage.
        log = completed.stderr.decode("utf-8", errors="replace")
        psnr = re.search(r"\bPSNR y:(\S+).*?\baverage:(\S+)", log)
        ssim = re.search(r"\bSSIM Y:(\S+).*?\bAll:(\S+)", log)
        if psnr is None or ssim is None:
            self._remove_files((psnr_report, ssim_report))
            print(f"[Error] Missing final quality summaries for {target_file}")
            return None, None
        self._measured_metrics[(psnr_report, ssim_report)] = QualityMetrics(
            float(psnr[2]), float(psnr[1]), float(ssim[2]), float(ssim[1])
        )
        return psnr_report, ssim_report

    @staticmethod
    def _parse_report(
        report: Path, field_names: Sequence[str], expected_frames: int | None = None
    ) -> tuple[float, ...]:
        totals = [0.0] * len(field_names)
        count = 0
        with report.open("r", encoding="utf-8") as file:
            for line in file:
                fields = dict(part.split(":", 1) for part in line.split() if ":" in part)
                if fields.get("psnr_log_version") or not line.strip():
                    continue
                count += 1
                if int(fields.get("n", "0")) != count:
                    raise ValueError(f"Missing/out-of-order frame in {report}")
                for index, name in enumerate(field_names):
                    value = float(fields[name])
                    if name.startswith("psnr_"):
                        if math.isnan(value) or value < 0:
                            raise ValueError(f"Invalid PSNR in {report}")
                        # Average normalized MSE, not logarithmic dB. A lossless
                        # frame contributes zero error instead of infinite mean PSNR.
                        value = 10 ** (-value / 10)
                    elif not math.isfinite(value) or not -1 <= value <= 1:
                        raise ValueError(f"Invalid SSIM in {report}")
                    totals[index] += value
        if not count or (expected_frames is not None and count != expected_frames):
            raise ValueError(f"Incomplete metric data in {report}: {count} frames")
        return tuple(
            (math.inf if total == 0 else -10 * math.log10(total / count))
            if name.startswith("psnr_") else total / count
            for name, total in zip(field_names, totals)
        )

    def _parse_metrics(
        self, psnr_report: Path, ssim_report: Path, expected_frames: int | None = None
    ) -> QualityMetrics:
        try:
            psnr_avg, psnr_y = self._parse_report(psnr_report, ("psnr_avg", "psnr_y"), expected_frames)
            ssim_all, ssim_y = self._parse_report(ssim_report, ("All", "Y"), expected_frames)
            return self._measured_metrics.get(
                (psnr_report, ssim_report), QualityMetrics(psnr_avg, psnr_y, ssim_all, ssim_y)
            )
        except (OSError, ValueError, KeyError, OverflowError) as error:
            print(f"[Error] Failed to parse metric reports: {error}")
            return QualityMetrics.empty()

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    @staticmethod
    def _remove_files(target_files: Iterable[Path]) -> None:
        for path in target_files:
            path.unlink(missing_ok=True)

    def list_already_measured_files(self) -> set[str]:
        # Informational only: cached reports never authorize acceptance.
        if not self.temp_path.is_dir():
            return set()
        return {
            path.name for path in self.temp_path.iterdir()
            if path.suffix.lower() in TARGET_EXTENSIONS and self._is_nonempty_file(path)
            and all(self._is_nonempty_file(p) for p in self._metric_report_paths(path))
        }

    def list_up_already_measured_files(self) -> list[str]:
        return sorted(self.list_already_measured_files())

    @staticmethod
    def _probe(target_file: Path) -> dict:
        return json.loads(subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(target_file)],
            stderr=subprocess.PIPE,
        ))

    @staticmethod
    def _positive_number(value: object) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) and number > 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _get_video_bitrate(cls, target_file: Path, probe: dict | None = None) -> int:
        probe = cls._probe(target_file) if probe is None else probe
        stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
        if stream is None:
            raise ValueError("No video stream")
        bitrate = cls._positive_number(stream.get("bit_rate"))
        if bitrate >= 1:
            return round(bitrate)
        # Container bitrate includes audio/other tracks. Stream video packets
        # instead of buffering packet data for an entire long video.
        size = 0
        start, end = math.inf, -math.inf
        with tempfile.TemporaryFile() as errors:
            with subprocess.Popen(
                [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_packets",
                 "-show_entries", "packet=size,pts_time,duration_time", "-of", "compact=p=0:nk=0", str(target_file)],
                stdout=subprocess.PIPE, stderr=errors, text=True, encoding="utf-8",
            ) as process:
                try:
                    for line in process.stdout:
                        fields = dict(p.split("=", 1) for p in line.strip().split("|") if "=" in p)
                        size += int(fields["size"])
                        pts = float(fields.get("pts_time", "nan"))
                        duration = cls._positive_number(fields.get("duration_time"))
                        if math.isfinite(pts):
                            start = min(start, pts)
                            end = max(end, pts + duration)
                    if process.wait() != 0:
                        errors.seek(0)
                        raise subprocess.CalledProcessError(process.returncode, process.args, stderr=errors.read())
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
        duration = cls._positive_number(stream.get("duration")) or (end - start)
        if not math.isfinite(duration) or duration <= 0 or size <= 0:
            raise ValueError("Cannot determine video bitrate from packets")
        return max(1, round(size * 8 / duration))

    @classmethod
    def _get_video_info(cls, target_file: Path) -> VideoInfo:
        probe = cls._probe(target_file)
        streams = probe.get("streams", [])
        stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if stream is None or stream.get("disposition", {}).get("attached_pic"):
            raise ValueError("First video stream is missing or is a cover image")
        # Container metadata alone can describe a truncated file. Count decoded frames.
        frame_output = subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(target_file)],
            stderr=subprocess.PIPE,
        )
        frames = int(frame_output.strip())
        duration = cls._positive_number(stream.get("duration"))
        if not duration:
            duration = cls._positive_number(probe.get("format", {}).get("duration"))
        if frames <= 0 or duration <= 0:
            raise ValueError("Video has no decoded frames or valid duration")
        properties = {
            key: str(stream[key]) for key in (
                "sample_aspect_ratio", "color_range", "color_space", "color_transfer",
                "color_primaries", "chroma_location", "field_order",
            ) if stream.get(key) not in (None, "unknown", "unspecified", "N/A", "0:1")
        }
        rotation = next((float(s["rotation"]) for s in stream.get("side_data_list", [])
                         if "rotation" in s), 0.0)
        properties["rotation"] = str(rotation % 360)
        hdr_metadata: set[str] = set()
        side_data = stream.get("side_data_list", [])
        if any("DOVI" in s.get("side_data_type", "") for s in side_data):
            raise ValueError("Dolby Vision encoding is not supported; original retained")
        # HDR SEI can exist even when the container omits color-transfer tags.
        frame_data = json.loads(subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+1",
             "-show_frames", "-show_entries", "frame=side_data_list", "-of", "json", str(target_file)],
            stderr=subprocess.PIPE,
        ))
        for frame in frame_data.get("frames", []):
            for data in frame.get("side_data_list", []):
                data_type = data.get("side_data_type", "")
                if "Dynamic" in data_type or "Dolby Vision" in data_type:
                    raise ValueError("Dynamic HDR encoding is not supported; original retained")
                if data_type in {"Mastering display metadata", "Content light level metadata"}:
                    hdr_metadata.add(json.dumps(data, sort_keys=True))
        return VideoInfo(
            cls._get_video_bitrate(target_file, probe), int(stream["width"]), int(stream["height"]),
            stream["pix_fmt"], frames, duration, tuple(s.get("codec_type", "unknown") for s in streams),
            tuple(sorted(properties.items())),
            tuple(sorted(hdr_metadata)),
        )

    @staticmethod
    def _compatible(source: VideoInfo, candidate: VideoInfo) -> bool:
        return (
            source.width == candidate.width and source.height == candidate.height
            and source.pixel_format == candidate.pixel_format
            and source.frames == candidate.frames and source.stream_types == candidate.stream_types
            and abs(source.duration - candidate.duration) <= max(0.01, source.duration / source.frames)
            and all(dict(candidate.video_properties).get(key) == value
                    for key, value in source.video_properties)
            and source.hdr_metadata == candidate.hdr_metadata
        )

    def _print_quality_result(self, metrics: QualityMetrics, accepted: bool) -> None:
        print(
            f"[{'Done' if accepted else 'Retry'}] PSNR={metrics.psnr_avg:.4f}, "
            f"PSNR Y={metrics.psnr_y:.4f}, SSIM={metrics.ssim_all:.6f}, SSIM Y={metrics.ssim_y:.6f}"
        )

    def _process_file(self, target_file: Path, position: int, total: int) -> dict[str, object]:
        record: dict[str, object] = dict.fromkeys(RESULT_COLUMNS)
        record.update(file_name=target_file.name, status="failed", reason="", attempts=0)
        try:
            source_stat = target_file.stat()
            record["orig_file_size"] = source_stat.st_size
        except OSError as error:
            record["reason"] = str(error)
            return record
        if (self.done_path / target_file.name).exists():
            record["reason"] = "Output already exists; it was not overwritten"
            return record
        try:
            self._emit("phase", phase="analyzing", file_name=target_file.name)
            source = self._get_video_info(target_file)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            record["reason"] = self._error_text(error)
            return record
        record["orig_video_bitrate"] = source.bitrate
        had_error = False
        last_error = ""
        for attempt, ratio in enumerate(self.ratios, start=1):
            bitrate = max(1, round(source.bitrate * ratio))
            for key in ("psnr_avg", "psnr_y", "ssim_all", "ssim_y", "trans_file_size", "ratio", "trans_video_bitrate"):
                record[key] = None
            record.update(attempts=attempt, target_video_bitrate=bitrate)
            print(f"[{position}/{total}] {target_file.name}: {source.bitrate} -> {bitrate} bps")
            self._emit("phase", phase="encoding", file_name=target_file.name,
                       attempt=attempt, max_attempts=len(self.ratios), bitrate=bitrate,
                       encoder="cuda" if self.use_cuda else "cpu")
            transcoded = self._transcode(target_file, bitrate)
            if transcoded is None:
                had_error = True
                last_error = "Encoding failed"
                break
            reports = self._metric_report_paths(transcoded)
            try:
                self._emit("phase", phase="validating", file_name=target_file.name)
                candidate = self._get_video_info(transcoded)
                if not self._compatible(source, candidate):
                    raise ValueError("Frame count, duration, format, streams or video metadata changed")
                measured = self._measure(target_file, transcoded)
                if None in measured:
                    raise ValueError("Quality measurement failed")
                metrics = self._parse_metrics(*reports, expected_frames=source.frames)
                if any(math.isnan(v) for v in (metrics.psnr_avg, metrics.psnr_y, metrics.ssim_all, metrics.ssim_y)):
                    raise ValueError("Invalid or incomplete quality reports")
                size = transcoded.stat().st_size
                current_stat = target_file.stat()
                if (current_stat.st_size, current_stat.st_mtime_ns) != (
                    source_stat.st_size, source_stat.st_mtime_ns
                ):
                    raise ValueError("Source changed during transcoding")
                record.update(
                    psnr_avg=metrics.psnr_avg, psnr_y=metrics.psnr_y,
                    ssim_all=metrics.ssim_all, ssim_y=metrics.ssim_y,
                    trans_file_size=size, ratio=size / record["orig_file_size"],
                    trans_video_bitrate=candidate.bitrate,
                )
                accepted = metrics.meets(self.psnr_threshold, self.ssim_threshold)
                self._print_quality_result(metrics, accepted)
                self._emit("metrics", **record)
                if accepted and size < source_stat.st_size and candidate.bitrate < source.bitrate:
                    record["status"] = "accepted"
                    self._source_files[transcoded] = target_file
                    return record
                if accepted:
                    print("[Keep original] Quality passed, but size/bitrate did not decrease")
                    break
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
                had_error = True
                last_error = self._error_text(error)
                print(f"[Error] {target_file.name}: {self._error_text(error)}")
                break
            finally:
                if transcoded not in self._source_files:
                    self._remove_files((transcoded, *reports))
        record.update(
            status="failed" if had_error else "unchanged",
            reason=f"{last_error}; original retained" if had_error
            else "No candidate met both quality and reduction requirements; original retained",
        )
        return record

    def _run_transcoding(self, target_files: Sequence[Path]) -> None:
        for position, target in enumerate(target_files, start=1):
            self._emit("file_started", position=position, total=len(target_files),
                       file_name=target.name, source_path=str(target))
            try:
                record = self._process_file(target, position, len(target_files))
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                self._source_files.pop(self.temp_path / target.name, None)
                record = dict.fromkeys(RESULT_COLUMNS)
                record.update(file_name=target.name, status="failed", attempts=0,
                              reason=self._error_text(error))
            self._results.append(record)
            self._emit("file_completed", position=position, record=record)
            if record["status"] != "accepted":
                print(f"[{record['status']}] {target.name}: {record['reason']}")

    def _completed_transcoded_files(self) -> list[Path]:
        return sorted(path for path in self._source_files if self._is_nonempty_file(path))

    @staticmethod
    def _publish_file(source: Path, destination: Path) -> None:
        # Hard links atomically refuse existing destinations. Exclusive copying
        # supports filesystems without hard links as well.
        try:
            os.link(source, destination)
        except FileExistsError:
            raise
        except OSError:
            created = False
            try:
                with destination.open("xb") as output:
                    created = True
                    with source.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
            except BaseException:
                if created:
                    destination.unlink(missing_ok=True)
                raise
        source.unlink()

    def _move_transcoded_files(self) -> bool:
        succeeded = True
        for path in self._source_files:
            try:
                self._publish_file(path, self.done_path / path.name)
            except OSError as error:
                succeeded = False
                for record in self._results:
                    if record["file_name"] == path.name:
                        record.update(status="failed", reason=f"Could not publish output: {error}")
                print(f"[Error] Failed to publish {path}: {error}")
        return succeeded

    def _next_result_paths(self) -> tuple[Path, Path]:
        index = 0
        while True:
            suffix = "" if index == 0 else f"_{index}"
            csv_path = self.done_path / f"measured_data{suffix}.csv"
            json_path = self.done_path / f"measured_data{suffix}.json"
            if not csv_path.exists() and not json_path.exists():
                return csv_path, json_path
            index += 1

    def _gather_measured_data(self) -> bool:
        csv_path, json_path = self._next_result_paths()
        created = []
        # JSON has no Infinity/NaN numbers. Lossless PSNR is the string "inf";
        # unavailable metrics are null. CSV uses the same convention.
        records = [
            {key: ("inf" if value == math.inf else None) if isinstance(value, float)
             and not math.isfinite(value) else value for key, value in record.items()}
            for record in self._results
        ]
        try:
            with csv_path.open("x", encoding="utf-8", newline="") as file:
                created.append(csv_path)
                writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
                writer.writeheader()
                writer.writerows(records)
            with json_path.open("x", encoding="utf-8") as file:
                created.append(json_path)
                json.dump(records, file, indent=4, ensure_ascii=False, allow_nan=False)
        except (OSError, ValueError, TypeError) as error:
            self._remove_files(created)
            print(f"[Error] Failed to save measured data: {error}")
            return False
        print(f"[Success] Saved measured data to {csv_path} and {json_path}")
        return True

    def run(self, target_files: Sequence[Path] | None = None) -> bool:
        self._prepare_directories()
        for executable in (FFMPEG, FFPROBE):
            if shutil.which(executable) is None:
                raise ValueError(f"Required executable not found in PATH: {executable}")
        targets = self._gather_target_files() if target_files is None else [Path(p).absolute() for p in target_files]
        if len(set(targets)) != len(targets):
            raise ValueError("Duplicate target files")
        for target in targets:
            if (target.parent != self.target_path or target.is_symlink() or not target.is_file()
                    or target.suffix.lower() not in TARGET_EXTENSIONS):
                raise ValueError(f"Invalid selected video: {target}")
        if not targets:
            print("[Info] No MP4 files found")
            return True
        self._source_files.clear()
        self._results.clear()
        self._measured_metrics.clear()
        base_temp_path = self.temp_path
        # Isolate concurrent runs and stale/untrusted reports.
        self.temp_path = Path(tempfile.mkdtemp(prefix="run-", dir=base_temp_path))
        try:
            self._run_transcoding(targets)
            moved = self._move_transcoded_files()
            saved = self._gather_measured_data()
            if moved and saved:
                if self.temp_path.resolve().parent != base_temp_path.resolve():
                    raise ValueError("Unexpected temporary directory location")
                shutil.rmtree(self.temp_path)
            else:
                print(f"[Recovery] Retained temporary artifacts in {self.temp_path}")
            counts = {status: sum(r["status"] == status for r in self._results)
                      for status in ("accepted", "unchanged", "failed")}
            print(f"[Summary] {counts}")
            return moved and saved and counts["failed"] == 0
        finally:
            self.temp_path = base_temp_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce video bitrate while maintaining PSNR and SSIM thresholds.")
    parser.add_argument("-p", "--path", help="Directory containing MP4 files")
    parser.add_argument("--psnr", default=THRESHOLD_PSNR, type=float, help="Minimum PSNR (dB)")
    parser.add_argument("--ssim", default=THRESHOLD_SSIM, type=float, help="Minimum SSIM (0 to 1)")
    parser.add_argument("--ratios", nargs="+", type=float, default=COMPRESS_RATIOS,
                        help="Increasing bitrate ratios to try (default: 0.6 0.7 0.8 0.9)")
    parser.add_argument("--encoder", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto: try NVIDIA NVENC, fall back to CPU libx265")
    parser.add_argument("-t", "--threshold", action="store_true", help="Show configured quality thresholds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.threshold:
        print(f"Threshold of PSNR: {args.psnr}, Threshold of SSIM: {args.ssim}")
        return 0
    if args.path is None:
        parser.print_help()
        return 0
    try:
        return 0 if ContentTranscoding(args).run() else 1
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:
        print(f"[Error] {error}")
        return 1
    except KeyboardInterrupt:
        print("[Interrupted] Original files retained")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
