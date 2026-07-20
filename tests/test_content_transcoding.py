import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ContentTranscoding import ContentTranscoding, QualityMetrics, main


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
        self.assertEqual(command[command.index("-c:v") + 1], "libx265")
        self.assertEqual(command[command.index("-i") + 1], str(source))
        self.assertIn(str(destination), command)

    def test_parse_metrics_averages_psnr_and_ssim_reports(self):
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

        self.assertEqual(metrics, QualityMetrics(42.0, 42.0, 0.96, 0.95))

    def test_parse_metrics_returns_empty_metrics_for_missing_data(self):
        missing = self.transcoder.temp_path / "missing.txt"

        with redirect_stdout(io.StringIO()):
            metrics = self.transcoder._parse_metrics(missing, missing)

        self.assertEqual(metrics, QualityMetrics.empty())

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
        self.assertFalse(incomplete_ssim.exists())

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
        reports = self.transcoder._metric_report_paths(transcoded)
        accepted_metrics = QualityMetrics(41.0, 42.0, 0.94, 0.95)

        with (
            patch.object(self.transcoder, "_get_video_bitrate", return_value=1000),
            patch.object(self.transcoder, "_transcode", return_value=transcoded) as transcode,
            patch.object(self.transcoder, "_measure", return_value=reports),
            patch.object(self.transcoder, "_parse_metrics", return_value=accepted_metrics),
            patch.object(self.transcoder, "_print_quality_result"),
        ):
            self.transcoder._process_file(source, 1, 1)

        transcode.assert_called_once_with(source, 600)
        self.assertEqual(self.transcoder._source_files[transcoded], source)


class CommandLineTestCase(unittest.TestCase):
    def test_threshold_command_accepts_overrides(self):
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--threshold", "--psnr", "42", "--ssim", "0.95"])

        self.assertEqual(exit_code, 0)
        self.assertIn("PSNR: 42.0", output.getvalue())
        self.assertIn("SSIM: 0.95", output.getvalue())


if __name__ == "__main__":
    unittest.main()
