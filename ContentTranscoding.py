import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import pandas as pd


TARGET_EXTENSIONS = frozenset({".mp4"})
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
USING_CUDA = True

THRESHOLD_PSNR = 40.0
THRESHOLD_SSIM = 0.93
COMPRESS_RATIOS = (0.6, 0.7, 0.8, 0.9)
RESULT_COLUMNS = (
    "file_name",
    "psnr_avg",
    "psnr_y",
    "ssim_all",
    "ssim_y",
    "orig_file_size",
    "trans_file_size",
    "ratio",
)


@dataclass(frozen=True)
class QualityMetrics:
    psnr_avg: float
    psnr_y: float
    ssim_all: float
    ssim_y: float

    @classmethod
    def empty(cls) -> "QualityMetrics":
        return cls(0.0, 0.0, 0.0, 0.0)

    def meets(self, psnr_threshold: float, ssim_threshold: float) -> bool:
        return (
            self.psnr_avg > psnr_threshold
            and self.psnr_y > psnr_threshold
            and self.ssim_all > ssim_threshold
            and self.ssim_y > ssim_threshold
        )


class ContentTranscoding:
    def __init__(self, args: argparse.Namespace):
        if args is None or not getattr(args, "path", None):
            raise ValueError("A target path is required")

        self.target_path = Path(args.path)
        self.psnr_threshold = float(getattr(args, "psnr", THRESHOLD_PSNR))
        self.ssim_threshold = float(getattr(args, "ssim", THRESHOLD_SSIM))
        self.use_cuda = bool(getattr(args, "use_cuda", USING_CUDA))
        self.temp_path = self.target_path / "temporary"
        self.done_path = self.target_path / "done"
        self._source_files: dict[Path, Path] = {}

    def _prepare_directories(self) -> None:
        if not self.target_path.is_dir():
            raise ValueError(f"Invalid target directory: {self.target_path}")
        self.temp_path.mkdir(parents=True, exist_ok=True)
        self.done_path.mkdir(parents=True, exist_ok=True)

    def _gather_target_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.target_path.iterdir()
            if path.is_file() and path.suffix.lower() in TARGET_EXTENSIONS
        )

    def _build_transcode_command(
        self, target_file: Path, transcoded_file: Path, video_bitrate: int
    ) -> list[str]:
        command = [FFMPEG, "-y", "-loglevel", "error"]
        if self.use_cuda:
            command.extend(("-hwaccel", "cuda"))

        encoder = "hevc_nvenc" if self.use_cuda else "libx265"
        command.extend(
            (
                "-i",
                str(target_file),
                "-b:v",
                str(video_bitrate),
                "-c:v",
                encoder,
                "-c:a",
                "copy",
                str(transcoded_file),
            )
        )
        return command

    def _transcode(self, target_file: Path, video_bitrate: int) -> Path | None:
        transcoded_file = self.temp_path / target_file.name
        transcoded_file.unlink(missing_ok=True)
        command = self._build_transcode_command(
            target_file, transcoded_file, video_bitrate
        )

        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            transcoded_file.unlink(missing_ok=True)
            print(
                "[Error] Failed to transcode "
                f"{target_file} at bitrate {video_bitrate}: {error}"
            )
            return None

        if not self._is_nonempty_file(transcoded_file):
            transcoded_file.unlink(missing_ok=True)
            return None
        return transcoded_file

    def _metric_report_paths(self, target_file: Path) -> tuple[Path, Path]:
        return (
            self.temp_path / f"{target_file.name}_psnr.txt",
            self.temp_path / f"{target_file.name}_ssim.txt",
        )

    def _measure(
        self, anchor_file: Path, target_file: Path | None
    ) -> tuple[Path | None, Path | None]:
        if target_file is None:
            return None, None

        psnr_report, ssim_report = self._metric_report_paths(target_file)
        self._remove_files((psnr_report, ssim_report))
        filter_graph = (
            "[1:v:0]split=2[ref1][ref2];"
            f"[0:v:0][ref1]psnr=f={psnr_report.name}[v_pass];"
            f"[v_pass][ref2]ssim=f={ssim_report.name}"
        )
        command = [
            FFMPEG,
            "-loglevel",
            "error",
            "-i",
            str(anchor_file.resolve()),
            "-i",
            str(target_file.resolve()),
            "-filter_complex",
            filter_graph,
            "-f",
            "null",
            "-",
        ]

        try:
            subprocess.run(command, check=True, cwd=self.temp_path)
        except (OSError, subprocess.CalledProcessError) as error:
            self._remove_files((psnr_report, ssim_report))
            print(f"[Error] Failed to measure {target_file}: {error}")
            return None, None

        if not all(self._is_nonempty_file(path) for path in (psnr_report, ssim_report)):
            self._remove_files((psnr_report, ssim_report))
            return None, None
        return psnr_report, ssim_report

    @staticmethod
    def _parse_report(report: Path, field_names: Sequence[str]) -> tuple[float, ...]:
        values = {field_name: [] for field_name in field_names}
        with report.open("r", encoding="utf-8") as file:
            for line in file:
                fields = {}
                for part in line.split():
                    key, separator, value = part.partition(":")
                    if separator:
                        fields[key] = value

                if all(field_name in fields for field_name in field_names):
                    for field_name in field_names:
                        values[field_name].append(float(fields[field_name]))

        if not all(values.values()):
            raise ValueError(f"No metric data found in {report}")
        return tuple(fmean(values[field_name]) for field_name in field_names)

    def _parse_metrics(self, psnr_report: Path, ssim_report: Path) -> QualityMetrics:
        try:
            psnr_avg, psnr_y = self._parse_report(
                psnr_report, ("psnr_avg", "psnr_y")
            )
            ssim_all, ssim_y = self._parse_report(ssim_report, ("All", "Y"))
        except (OSError, ValueError) as error:
            print(f"[Error] Failed to parse metric reports: {error}")
            return QualityMetrics.empty()

        return QualityMetrics(
            psnr_avg=round(psnr_avg, 3),
            psnr_y=round(psnr_y, 3),
            ssim_all=round(ssim_all, 6),
            ssim_y=round(ssim_y, 6),
        )

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    def _remove_empty_files(self) -> None:
        for path in self.temp_path.iterdir():
            if path.is_file() and path.stat().st_size == 0:
                path.unlink(missing_ok=True)

    def list_already_measured_files(self) -> set[str]:
        self._remove_empty_files()
        measured_files = set()
        for transcoded_file in self.temp_path.iterdir():
            if not (
                transcoded_file.is_file()
                and transcoded_file.suffix.lower() in TARGET_EXTENSIONS
            ):
                continue
            reports = self._metric_report_paths(transcoded_file)
            if all(self._is_nonempty_file(report) for report in reports):
                measured_files.add(transcoded_file.name)
        return measured_files

    # Backward-compatible alias for callers using the previous public name.
    def list_up_already_measured_files(self) -> list[str]:
        return sorted(self.list_already_measured_files())

    @staticmethod
    def _remove_files(target_files: Iterable[Path]) -> None:
        for path in target_files:
            path.unlink(missing_ok=True)

    def _get_file_sizes(self, transcoded_file: Path) -> tuple[int, int]:
        original_file = self._source_files.get(transcoded_file)
        if original_file is None:
            raise ValueError(f"Original file not registered for {transcoded_file}")
        return original_file.stat().st_size, transcoded_file.stat().st_size

    def _next_result_paths(self) -> tuple[Path, Path]:
        index = 0
        while True:
            suffix = "" if index == 0 else f"_{index}"
            csv_path = self.done_path / f"measured_data{suffix}.csv"
            json_path = self.done_path / f"measured_data{suffix}.json"
            if not csv_path.exists() and not json_path.exists():
                return csv_path, json_path
            index += 1

    def _save_results(self, dataframe: pd.DataFrame) -> bool:
        csv_path, json_path = self._next_result_paths()
        try:
            dataframe.to_csv(csv_path, index=False)
            dataframe.to_json(
                json_path, orient="records", indent=4, force_ascii=False
            )
        except (OSError, ValueError, TypeError) as error:
            self._remove_files((csv_path, json_path))
            print(f"[Error] Failed to save measured data: {error}")
            return False

        print(f"[Success] Saved measured data to {csv_path} and {json_path}")
        return True

    def _result_record(self, transcoded_file: Path) -> dict[str, object]:
        psnr_file, ssim_file = self._metric_report_paths(transcoded_file)
        metrics = self._parse_metrics(psnr_file, ssim_file)
        original_size, transcoded_size = self._get_file_sizes(transcoded_file)
        return {
            "file_name": transcoded_file.name,
            "psnr_avg": metrics.psnr_avg,
            "psnr_y": metrics.psnr_y,
            "ssim_all": metrics.ssim_all,
            "ssim_y": metrics.ssim_y,
            "orig_file_size": original_size,
            "trans_file_size": transcoded_size,
            "ratio": round(transcoded_size / original_size, 2),
        }

    def _completed_transcoded_files(self) -> list[Path]:
        return sorted(
            path for path in self._source_files if self._is_nonempty_file(path)
        )

    def _gather_measured_data(self) -> bool:
        transcoded_files = self._completed_transcoded_files()
        records = [self._result_record(path) for path in transcoded_files]
        dataframe = pd.DataFrame(records, columns=RESULT_COLUMNS)
        if not self._save_results(dataframe):
            return False

        reports = (
            report
            for transcoded_file in transcoded_files
            for report in self._metric_report_paths(transcoded_file)
        )
        self._remove_files(reports)
        return True

    def _move_transcoded_files(self) -> None:
        transcoded_files = self._completed_transcoded_files()
        failed_files = []
        for path in transcoded_files:
            try:
                shutil.move(str(path), self.done_path / path.name)
            except (OSError, shutil.Error) as error:
                print(f"[Error] Failed to move {path} to {self.done_path}: {error}")
                failed_files.append(path)

        if failed_files:
            print(f"[Error] Failed to move {len(failed_files)} transcoded file(s)")
        else:
            print(f"[Success] Moved all transcoded files to {self.done_path}")

        if not any(self.temp_path.iterdir()):
            self.temp_path.rmdir()

    @staticmethod
    def _get_video_bitrate(target_file: Path) -> int:
        command = [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(target_file),
        ]
        output = subprocess.check_output(command, text=True).strip()
        bitrate = int(output)
        if bitrate <= 0:
            raise ValueError(f"Invalid video bitrate: {output!r}")
        return bitrate

    def _print_quality_result(self, metrics: QualityMetrics, accepted: bool) -> None:
        if accepted:
            print(
                "[Done] Quality accepted: "
                f"PSNR={metrics.psnr_avg}, PSNR Y={metrics.psnr_y}, "
                f"SSIM={metrics.ssim_all}, SSIM Y={metrics.ssim_y}"
            )
            return

        print(
            "[Retry] Substandard quality: "
            f"PSNR={metrics.psnr_avg}/{self.psnr_threshold}, "
            f"PSNR Y={metrics.psnr_y}/{self.psnr_threshold}, "
            f"SSIM={metrics.ssim_all}/{self.ssim_threshold}, "
            f"SSIM Y={metrics.ssim_y}/{self.ssim_threshold}"
        )

    def _process_file(self, target_file: Path, position: int, total: int) -> None:
        try:
            original_bitrate = self._get_video_bitrate(target_file)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"[Error] Failed to get bitrate for {target_file}: {error}")
            return

        last_artifacts: tuple[Path, ...] = ()
        for ratio in COMPRESS_RATIOS:
            video_bitrate = round(original_bitrate * ratio)
            print(
                f"[{position}/{total}] Transcoding {target_file} "
                f"from {original_bitrate} to {video_bitrate} bps"
            )
            transcoded_file = self._transcode(target_file, video_bitrate)
            if transcoded_file is None:
                continue

            psnr_report, ssim_report = self._measure(target_file, transcoded_file)
            if psnr_report is None or ssim_report is None:
                transcoded_file.unlink(missing_ok=True)
                continue

            last_artifacts = (transcoded_file, psnr_report, ssim_report)
            metrics = self._parse_metrics(psnr_report, ssim_report)
            accepted = metrics.meets(self.psnr_threshold, self.ssim_threshold)
            self._print_quality_result(metrics, accepted)
            if accepted:
                self._source_files[transcoded_file] = target_file
                return

        self._remove_files(last_artifacts)

    def _run_transcoding(self, target_files: Sequence[Path]) -> None:
        already_measured = self.list_already_measured_files()
        total = len(target_files)
        for position, target_file in enumerate(target_files, start=1):
            transcoded_file = self.temp_path / target_file.name
            if target_file.name in already_measured:
                print(f"[Skip] {target_file.name}: already measured")
                self._source_files[transcoded_file] = target_file
                continue
            self._process_file(target_file, position, total)

    def run(self) -> bool:
        self._prepare_directories()
        self._source_files.clear()
        target_files = self._gather_target_files()
        self._run_transcoding(target_files)
        if not self._gather_measured_data():
            return False
        self._move_transcoded_files()
        return True


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcode videos while maintaining PSNR and SSIM thresholds."
    )
    parser.add_argument("-p", "--path", help="Directory containing video files")
    parser.add_argument(
        "--psnr", default=THRESHOLD_PSNR, type=float, help="PSNR threshold"
    )
    parser.add_argument(
        "--ssim", default=THRESHOLD_SSIM, type=float, help="SSIM threshold"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store_true",
        help="Show the default PSNR and SSIM thresholds",
    )
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
        succeeded = ContentTranscoding(args).run()
    except ValueError as error:
        parser.error(str(error))
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
