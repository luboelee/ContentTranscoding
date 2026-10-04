import argparse
import io
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ContentTranscoding import RESULT_COLUMNS, ContentTranscoding, QualityMetrics, VideoInfo, build_argument_parser, main


class ContentTranscodingTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.target_path = Path(self.temporary_directory.name)
        args = argparse.Namespace(
            path=str(self.target_path),
            psnr=40.0,
            ssim=0.93,
            use_cuda=False,
        )
        self.transcoder = ContentTranscoding(args)
        self.transcoder._prepare_directories()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_gather_target_files_is_sorted_and_case_insensitive(self):
        (self.target_path / "b.MP4").write_bytes(b"video")
        (self.target_path / "a.mp4").write_bytes(b"video")
        (self.target_path / "notes.txt").write_text("ignore", encoding="utf-8")

        result = self.transcoder._gather_target_files()

        self.assertEqual([path.name for path in result], ["a.mp4", "b.MP4"])

    def test_build_transcode_command_uses_argument_list_and_cpu_encoder(self):
        source = self.target_path / "source file.mp4"
        destination = self.transcoder.temp_path / source.name

        command = self.transcoder._build_transcode_command(
            source, destination, 1_500_000
        )

        self.assertEqual(command[0], "ffmpeg")
        self.assertNotIn("-hwaccel", command)
        self.assertEqual(command[command.index("-c:v:0") + 1], "libx265")
        self.assertEqual(command[command.index("-map") + 1], "0")
        self.assertEqual(command[command.index("-c") + 1], "copy")
        self.assertEqual(command[command.index("-i") + 1], str(source))
        self.assertIn(str(destination), command)

    def test_parse_metrics_averages_error_instead_of_psnr_decibels(self):
        psnr_report = self.transcoder.temp_path / "video.mp4_psnr.txt"
        ssim_report = self.transcoder.temp_path / "video.mp4_ssim.txt"
        psnr_report.write_text(
            "n:1 psnr_y:41.0 psnr_avg:40.0\n"
            "n:2 psnr_y:43.0 psnr_avg:44.0\n",
            encoding="utf-8",
        )
        ssim_report.write_text(
            "n:1 Y:0.94 All:0.95\n"
            "n:2 Y:0.96 All:0.97\n",
            encoding="utf-8",
        )

        metrics = self.transcoder._parse_metrics(psnr_report, ssim_report)

        self.assertAlmostEqual(metrics.psnr_avg, -10 * math.log10((1e-4 + 10 ** -4.4) / 2))
        self.assertAlmostEqual(metrics.psnr_y, -10 * math.log10((10 ** -4.1 + 10 ** -4.3) / 2))
        self.assertAlmostEqual(metrics.ssim_all, 0.96)
        self.assertAlmostEqual(metrics.ssim_y, 0.95)

    def test_parse_metrics_returns_empty_metrics_for_missing_data(self):
        missing = self.transcoder.temp_path / "missing.txt"

        with redirect_stdout(io.StringIO()):
            metrics = self.transcoder._parse_metrics(missing, missing)

        self.assertFalse(metrics.meets(0, 0))

    def test_already_measured_requires_both_nonempty_reports(self):
        complete = self.transcoder.temp_path / "complete.mp4"
        incomplete = self.transcoder.temp_path / "incomplete.mp4"
        complete.write_bytes(b"video")
        incomplete.write_bytes(b"video")
        for report in self.transcoder._metric_report_paths(complete):
            report.write_text("metrics", encoding="utf-8")
        incomplete_psnr, incomplete_ssim = self.transcoder._metric_report_paths(
            incomplete
        )
        incomplete_psnr.write_text("metrics", encoding="utf-8")
        incomplete_ssim.touch()

        result = self.transcoder.list_already_measured_files()

        self.assertEqual(result, {"complete.mp4"})
        self.assertTrue(incomplete_ssim.exists())  # Informational API never deletes user artifacts.

    def test_next_result_paths_keeps_csv_and_json_suffixes_together(self):
        (self.transcoder.done_path / "measured_data.csv").touch()
        (self.transcoder.done_path / "measured_data_1.json").touch()

        csv_path, json_path = self.transcoder._next_result_paths()

        self.assertEqual(csv_path.name, "measured_data_2.csv")
        self.assertEqual(json_path.name, "measured_data_2.json")

    def test_completed_files_excludes_unregistered_temporary_videos(self):
        source = self.target_path / "source.mp4"
        registered = self.transcoder.temp_path / "registered.mp4"
        orphan = self.transcoder.temp_path / "orphan.mp4"
        source.write_bytes(b"original")
        registered.write_bytes(b"transcoded")
        orphan.write_bytes(b"stale")
        self.transcoder._source_files[registered] = source

        result = self.transcoder._completed_transcoded_files()

        self.assertEqual(result, [registered])

    def test_process_file_stops_after_first_accepted_result(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original")
        transcoded = self.transcoder.temp_path / source.name
        transcoded.write_bytes(b"small")
        reports = self.transcoder._metric_report_paths(transcoded)
        accepted_metrics = QualityMetrics(41.0, 42.0, 0.94, 0.95)

        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=[
                VideoInfo(1000, 320, 240, "yuv420p", 2, 1, ("video",)),
                VideoInfo(600, 320, 240, "yuv420p", 2, 1, ("video",)),
            ]),
            patch.object(self.transcoder, "_transcode", return_value=transcoded) as transcode,
            patch.object(self.transcoder, "_measure", return_value=reports),
            patch.object(self.transcoder, "_parse_metrics", return_value=accepted_metrics),
            patch.object(self.transcoder, "_print_quality_result"),
        ):
            self.transcoder._process_file(source, 1, 1)

        transcode.assert_called_once_with(source, 600)
        self.assertEqual(self.transcoder._source_files[transcoded], source)

    def test_thresholds_are_inclusive_and_do_not_round_metrics(self):
        self.assertTrue(QualityMetrics(40, 40, 0.93, 0.93).meets(40, 0.93))
        self.assertFalse(QualityMetrics(39.9999, 41, 0.94, 0.94).meets(40, 0.93))
        self.assertFalse(QualityMetrics(41, 41, 0.9299999, 0.94).meets(40, 0.93))

    def test_compatibility_allows_only_data_stream_removal(self):
        source = VideoInfo(1000, 320, 240, "yuv420p", 2, 1, ("video", "audio", "data", "video", "subtitle"))
        output = replace(source, bitrate=600, stream_types=("video", "audio", "video", "subtitle"))
        self.assertTrue(self.transcoder._compatible(source, output))
        for types in (("video", "video", "subtitle"), ("video", "audio", "subtitle"),
                      ("video", "audio", "video"), ("video", "audio", "data", "video", "subtitle")):
            with self.subTest(types=types):
                self.assertFalse(self.transcoder._compatible(source, replace(output, stream_types=types)))
        self.assertFalse(self.transcoder._compatible(source, replace(output, frames=1)))
        self.assertFalse(self.transcoder._compatible(source, replace(output, pixel_format="yuv420p10le")))

    def test_excluded_camera_data_is_reported_without_reducing_quality_checks(self):
        source = self.target_path / "camera.mp4"
        source.write_bytes(b"original video")
        candidate = self.transcoder.temp_path / source.name
        candidate.write_bytes(b"small")
        source_info = VideoInfo(1000, 320, 240, "yuv420p", 2, 1, ("video", "audio", "data", "data"),
                                excluded_streams=("#2 (djmd)", "#3 (tmcd)"))
        output_info = replace(source_info, bitrate=600, stream_types=("video", "audio"), excluded_streams=())
        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=[source_info, output_info]),
            patch.object(self.transcoder, "_transcode", return_value=candidate),
            patch.object(self.transcoder, "_measure", return_value=self.transcoder._metric_report_paths(candidate)),
            patch.object(self.transcoder, "_parse_metrics", return_value=QualityMetrics(41, 41, 0.94, 0.94)),
            redirect_stdout(io.StringIO()),
        ):
            record = self.transcoder._process_file(source, 1, 1)
            self.assertEqual(record["status"], "accepted")
            self.assertIn("djmd", record["warnings"])
            self.assertIn("tmcd", record["warnings"])
            self.transcoder._results = [record]
            self.assertTrue(self.transcoder._gather_measured_data())
        reported = json.loads((self.transcoder.done_path / "measured_data.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(reported["warnings"], record["warnings"])
        self.assertEqual(source.read_bytes(), b"original video")

    def test_lossless_frame_does_not_hide_low_quality_frame(self):
        report = self.transcoder.temp_path / "psnr.txt"
        report.write_text("n:1 psnr_avg:inf psnr_y:inf\nn:2 psnr_avg:20 psnr_y:20\n")
        average, y = self.transcoder._parse_report(report, ("psnr_avg", "psnr_y"), 2)
        self.assertAlmostEqual(average, 23.0102999566)
        self.assertAlmostEqual(y, average)
        self.assertLess(average, 40)

    def test_incomplete_or_invalid_report_cannot_pass(self):
        psnr, ssim = self.transcoder._metric_report_paths(Path("video.mp4"))
        for content in ("n:1 psnr_avg:41 psnr_y:41\n", "n:2 psnr_avg:41 psnr_y:41\n",
                        "n:1 psnr_avg:nan psnr_y:41\n", "garbage\n"):
            with self.subTest(content=content), redirect_stdout(io.StringIO()):
                psnr.write_text(content)
                ssim.write_text("n:1 All:0.99 Y:0.99\n")
                self.assertFalse(self.transcoder._parse_metrics(psnr, ssim, 2).meets(0, 0))

    def test_retry_increases_bitrate_and_only_keeps_accepted_candidate(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original video")
        candidate = self.transcoder.temp_path / source.name
        source_info = VideoInfo(1000, 320, 240, "yuv420p", 2, 1, ("video",))
        candidate_info = VideoInfo(700, 320, 240, "yuv420p", 2, 1, ("video",))

        def encode(*args):
            candidate.write_bytes(b"small")
            return candidate

        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=[source_info, candidate_info, candidate_info]),
            patch.object(self.transcoder, "_transcode", side_effect=encode) as encode_mock,
            patch.object(self.transcoder, "_measure", return_value=self.transcoder._metric_report_paths(candidate)),
            patch.object(self.transcoder, "_parse_metrics", side_effect=[
                QualityMetrics(39, 39, 0.92, 0.92), QualityMetrics(41, 41, 0.94, 0.94)]),
            redirect_stdout(io.StringIO()),
        ):
            record = self.transcoder._process_file(source, 1, 1)
        self.assertEqual([c.args[1] for c in encode_mock.call_args_list], [600, 700])
        self.assertEqual(record["status"], "accepted")
        self.assertTrue(candidate.exists())

    def test_quality_pass_with_larger_file_is_rejected(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"small")
        candidate = self.transcoder.temp_path / source.name
        candidate.write_bytes(b"larger encoded video")
        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=[
                VideoInfo(1000, 320, 240, "yuv420p", 2, 1, ("video",)),
                VideoInfo(600, 320, 240, "yuv420p", 2, 1, ("video",)),
            ]),
            patch.object(self.transcoder, "_transcode", return_value=candidate),
            patch.object(self.transcoder, "_measure", return_value=self.transcoder._metric_report_paths(candidate)),
            patch.object(self.transcoder, "_parse_metrics", return_value=QualityMetrics(45, 45, 0.99, 0.99)),
            redirect_stdout(io.StringIO()),
        ):
            record = self.transcoder._process_file(source, 1, 1)
        self.assertEqual(record["status"], "unchanged")
        self.assertFalse(candidate.exists())
        self.assertEqual(source.read_bytes(), b"small")

    def test_all_quality_failures_preserve_original_and_remove_candidates(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original")
        candidate = self.transcoder.temp_path / source.name
        reports = self.transcoder._metric_report_paths(candidate)

        def encode(*args):
            candidate.write_bytes(b"new")
            for report in reports:
                report.write_text("temporary metrics")
            return candidate

        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=[
                VideoInfo(1000, 320, 240, "yuv420p", 2, 1, ("video",)),
                *[VideoInfo(600, 320, 240, "yuv420p", 2, 1, ("video",))] * 4,
            ]),
            patch.object(self.transcoder, "_transcode", side_effect=encode) as encode_mock,
            patch.object(self.transcoder, "_measure", return_value=reports),
            patch.object(self.transcoder, "_parse_metrics", return_value=QualityMetrics(30, 30, 0.8, 0.8)),
            redirect_stdout(io.StringIO()),
        ):
            record = self.transcoder._process_file(source, 1, 1)
        self.assertEqual([c.args[1] for c in encode_mock.call_args_list], [600, 700, 800, 900])
        self.assertEqual(record["status"], "unchanged")
        self.assertFalse(any(p.exists() for p in (candidate, *reports)))
        self.assertEqual(source.read_bytes(), b"original")

    def test_truncated_video_is_rejected_before_metrics(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original")
        candidate = self.transcoder.temp_path / source.name
        candidate.write_bytes(b"short")
        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=[
                VideoInfo(1000, 320, 240, "yuv420p", 100, 4, ("video", "audio")),
                VideoInfo(600, 320, 240, "yuv420p", 50, 2, ("video", "audio")),
            ]),
            patch.object(self.transcoder, "_transcode", return_value=candidate),
            patch.object(self.transcoder, "_measure") as measure,
            redirect_stdout(io.StringIO()),
        ):
            record = self.transcoder._process_file(source, 1, 1)
        self.assertEqual(record["status"], "failed")
        measure.assert_not_called()
        self.assertFalse(candidate.exists())

    def test_existing_output_is_never_overwritten(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original")
        output = self.transcoder.done_path / source.name
        output.write_bytes(b"existing")
        with patch.object(self.transcoder, "_transcode") as encode:
            record = self.transcoder._process_file(source, 1, 1)
        self.assertEqual(record["status"], "failed")
        encode.assert_not_called()
        self.assertEqual(output.read_bytes(), b"existing")
        with self.assertRaises(FileExistsError):
            self.transcoder._publish_file(source, output)
        self.assertEqual(source.read_bytes(), b"original")

    def test_auto_encoder_falls_back_at_same_bitrate(self):
        self.transcoder.encoder_mode = "auto"
        self.transcoder.use_cuda = True
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original")
        candidate = self.transcoder.temp_path / source.name

        def run(command, **kwargs):
            if command[command.index("-c:v:0") + 1] == "hevc_nvenc":
                candidate.write_bytes(b"partial")
                raise subprocess.CalledProcessError(1, command, stderr=b"No CUDA device")
            candidate.write_bytes(b"cpu")

        with patch("ContentTranscoding.subprocess.run", side_effect=run) as runner, redirect_stdout(io.StringIO()):
            self.assertEqual(self.transcoder._transcode(source, 600), candidate)
        self.assertEqual(runner.call_count, 2)
        self.assertFalse(self.transcoder.use_cuda)
        self.assertEqual(candidate.read_bytes(), b"cpu")

    def test_stale_reports_do_not_skip_processing(self):
        source = self.target_path / "video.mp4"
        source.write_bytes(b"original")
        candidate = self.transcoder.temp_path / source.name
        candidate.write_bytes(b"stale")
        for report in self.transcoder._metric_report_paths(candidate):
            report.write_text("n:1 invalid:1\n")
        with patch.object(self.transcoder, "_process_file", return_value={"status": "accepted"}) as process:
            self.transcoder._run_transcoding([source])
        process.assert_called_once_with(source, 1, 1)

    def test_failed_batch_returns_false_and_records_failure(self):
        (self.target_path / "bad.mp4").write_bytes(b"invalid mp4")
        with (
            patch.object(self.transcoder, "_get_video_info", side_effect=ValueError("bad video")),
            patch("ContentTranscoding.shutil.which", return_value="executable"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertFalse(self.transcoder.run())
        records = json.loads((self.transcoder.done_path / "measured_data.json").read_text())
        self.assertEqual(records[0]["status"], "failed")
        self.assertIn("bad video", records[0]["reason"])

    def test_save_failure_preserves_raw_reports(self):
        (self.target_path / "video.mp4").write_bytes(b"original")

        def process(targets):
            (self.transcoder.temp_path / "raw_report.txt").write_text("metrics")
            self.transcoder._results.append({"status": "unchanged"})

        with (
            patch.object(self.transcoder, "_run_transcoding", side_effect=process),
            patch.object(self.transcoder, "_gather_measured_data", return_value=False),
            patch("ContentTranscoding.shutil.which", return_value="executable"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertFalse(self.transcoder.run())
        self.assertEqual(len(list(self.transcoder.temp_path.glob("run-*/raw_report.txt"))), 1)

    def test_json_output_supports_lossless_psnr(self):
        self.transcoder._results = [dict.fromkeys(RESULT_COLUMNS)]
        self.transcoder._results[0].update(psnr_avg=math.inf, psnr_y=math.inf)
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.transcoder._gather_measured_data())
        raw = (self.transcoder.done_path / "measured_data.json").read_text()
        self.assertEqual(json.loads(raw)[0]["psnr_avg"], "inf")
        self.assertNotIn("Infinity", raw)

    def test_partial_publish_copy_is_removed_on_error(self):
        source = self.transcoder.temp_path / "video.mp4"
        destination = self.transcoder.done_path / source.name
        source.write_bytes(b"complete candidate")

        def fail_copy(input_file, output):
            output.write(b"partial")
            raise OSError("disk full")

        with (
            patch("ContentTranscoding.os.link", side_effect=OSError("unsupported")),
            patch("ContentTranscoding.shutil.copyfileobj", side_effect=fail_copy),
            self.assertRaises(OSError),
        ):
            self.transcoder._publish_file(source, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(source.read_bytes(), b"complete candidate")

    def test_publish_error_marks_failure_and_preserves_candidate(self):
        original = self.target_path / "video.mp4"
        candidate = self.transcoder.temp_path / original.name
        original.write_bytes(b"original")
        candidate.write_bytes(b"candidate")
        self.transcoder._source_files[candidate] = original
        self.transcoder._results = [{"file_name": original.name, "status": "accepted"}]
        with patch.object(self.transcoder, "_publish_file", side_effect=OSError("disk full")), redirect_stdout(io.StringIO()):
            self.assertFalse(self.transcoder._move_transcoded_files())
        self.assertEqual(self.transcoder._results[0]["status"], "failed")
        self.assertTrue(candidate.exists())

    def test_per_file_io_failure_does_not_abort_batch(self):
        first, second = self.target_path / "a.mp4", self.target_path / "b.mp4"
        with (
            patch.object(self.transcoder, "_process_file", side_effect=[
                OSError("permission denied"), {"file_name": "b.mp4", "status": "accepted"}]),
            redirect_stdout(io.StringIO()),
        ):
            self.transcoder._run_transcoding([first, second])
        self.assertEqual([r["status"] for r in self.transcoder._results], ["failed", "accepted"])


class CommandLineTestCase(unittest.TestCase):
    def test_invalid_quality_thresholds_and_ratios_are_rejected(self):
        parser = build_argument_parser()
        for options in (["--psnr", "nan"], ["--psnr", "-1"], ["--ssim", "1.1"],
                        ["--ratios", "0.9", "0.6"], ["--ratios", "0"], ["--ratios", "1"],
                        ["--ratios", "0.6", "0.6"]):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ContentTranscoding(parser.parse_args(["--path", ".", *options]))

    def test_threshold_command_accepts_overrides(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--threshold", "--psnr", "42", "--ssim", "0.95"])

        self.assertEqual(exit_code, 0)
        self.assertIn("PSNR: 42.0", output.getvalue())
        self.assertIn("SSIM: 0.95", output.getvalue())


if __name__ == "__main__":
    unittest.main()
