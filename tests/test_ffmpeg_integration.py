"""Real encode/decode checks; skipped when FFmpeg/ffprobe are not installed."""
import argparse
import hashlib
import io
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ContentTranscoding import ContentTranscoding


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class FFmpegIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.directory.name)
        self.source = self.root / "한글 [a], quote'.MP4"
        self.transcoder = ContentTranscoding(argparse.Namespace(
            path=str(self.root), psnr=40.0, ssim=0.93, encoder="cpu",
        ))
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
            "-t", "2", "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", str(self.source),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def audio_hashes(path):
        data = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a", "-show_packets", "-show_data_hash", "sha256",
            "-show_entries", "packet=stream_index,data_hash", "-of", "json", str(path),
        ])
        return json.loads(data)["packets"]

    @staticmethod
    def frame_times(path):
        data = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
        ])
        return [float(frame["best_effort_timestamp_time"]) for frame in json.loads(data)["frames"]]

    def test_real_transcode_meets_defaults_and_preserves_both_audio_tracks(self):
        original_hash = hashlib.sha256(self.source.read_bytes()).digest()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.transcoder.run())
        output = self.root / "done" / self.source.name
        self.assertTrue(output.is_file())
        self.assertLess(output.stat().st_size, self.source.stat().st_size)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), original_hash)
        self.assertEqual(self.audio_hashes(self.source), self.audio_hashes(output))
        self.assertTrue(self.transcoder._compatible(
            self.transcoder._get_video_info(self.source), self.transcoder._get_video_info(output)))
        record = json.loads((self.root / "done" / "measured_data.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(record["status"], "accepted")
        self.assertGreaterEqual(record["psnr_avg"], 40)
        self.assertGreaterEqual(record["psnr_y"], 40)
        self.assertGreaterEqual(record["ssim_all"], 0.93)
        self.assertGreaterEqual(record["ssim_y"], 0.93)
        self.assertLess(record["trans_video_bitrate"], record["orig_video_bitrate"])

    def test_identical_frames_produce_lossless_psnr_and_full_coverage(self):
        self.transcoder._prepare_directories()
        reports = self.transcoder._measure(self.source, self.source)
        self.assertNotIn(None, reports)
        metrics = self.transcoder._parse_metrics(*reports, expected_frames=48)
        self.assertEqual(metrics.psnr_avg, math.inf)
        self.assertEqual(metrics.ssim_all, 1.0)

    def test_camera_data_and_timecode_exclusion_preserves_audio_and_cover(self):
        base = self.root / "timecode-base.mp4"
        cover = self.root / "cover.jpg"
        camera = self.root / "camera.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(self.source),
            "-map", "0", "-c", "copy", "-timecode", "02:41:34:10", str(base),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=red:size=160x90",
            "-frames:v", "1", "-update", "1", str(cover),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        preparation = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(base), "-i", str(cover),
            "-map", "0", "-map", "-0:d", "-map", "1:v", "-c", "copy",
            "-disposition:v:1", "attached_pic", "-timecode", "02:41:34:10", "-write_tmcd", "1", str(camera),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.assertEqual(preparation.returncode, 0, preparation.stderr.decode(errors="replace"))
        camera_bytes = camera.read_bytes()
        for tag in ("tmcd", "djmd"):
            with self.subTest(tag=tag):
                # Camera manufacturers use private sample-entry tags that FFmpeg
                # reads as codec none. Change only the tiny generated fixture.
                selected = self.root / f"camera-{tag}.mp4"
                selected.write_bytes(camera_bytes.replace(b"tmcd", tag.encode("ascii")))
                original_hash = hashlib.sha256(selected.read_bytes()).digest()
                source_probe = self.transcoder._probe(selected)
                data = [s for s in source_probe["streams"] if s["codec_type"] == "data"]
                self.assertGreaterEqual(len(data), 1)
                self.assertTrue(all(stream["codec_tag_string"] == tag for stream in data))
                with redirect_stdout(io.StringIO()) as log:
                    self.assertTrue(self.transcoder.run([selected]), log.getvalue())
                output = self.root / "done" / selected.name
                self.assertTrue(output.is_file(), log.getvalue())
                self.assertEqual(hashlib.sha256(selected.read_bytes()).digest(), original_hash)
                self.assertEqual(self.audio_hashes(selected), self.audio_hashes(output))
                output_probe = self.transcoder._probe(output)
                self.assertEqual([s["codec_type"] for s in output_probe["streams"]], ["video", "audio", "audio", "video"])
                self.assertEqual(output_probe["streams"][-1]["disposition"]["attached_pic"], 1)
                record = self.transcoder._results[0]
                self.assertEqual(record["status"], "accepted")
                self.assertIn(tag, record["warnings"])
                self.assertGreaterEqual(record["psnr_avg"], 40)
                self.assertGreaterEqual(record["ssim_all"], 0.93)

    def test_default_auto_encoder_produces_acceptable_result(self):
        self.transcoder.encoder_mode = "auto"
        self.transcoder.use_cuda = True
        with redirect_stdout(io.StringIO()) as log:
            self.assertTrue(self.transcoder.run(), log.getvalue())
        output = self.root / "done" / self.source.name
        self.assertTrue(output.exists())
        record = json.loads((self.root / "done" / "measured_data.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(record["status"], "accepted")
        self.assertGreaterEqual(record["psnr_avg"], 40)
        self.assertGreaterEqual(record["ssim_all"], 0.93)
        self.assertLess(record["ratio"], 1)

    def test_packet_fallback_measures_video_without_audio(self):
        probe = self.transcoder._probe(self.source)
        stream = probe["streams"][0]
        expected = int(stream.pop("bit_rate"))
        bitrate = self.transcoder._get_video_bitrate(self.source, probe)
        self.assertAlmostEqual(bitrate / expected, 1.0, places=3)

    def test_unreachable_threshold_retains_original_and_no_output(self):
        self.transcoder.psnr_threshold = 100
        original_hash = hashlib.sha256(self.source.read_bytes()).digest()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.transcoder.run())
        self.assertFalse((self.root / "done" / self.source.name).exists())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), original_hash)
        record = json.loads((self.root / "done" / "measured_data.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(record["status"], "unchanged")
        self.assertEqual(record["attempts"], 4)

    def test_ten_bit_variable_frame_rate_and_rotation_are_preserved(self):
        self.source.unlink()
        raw = self.root / "raw.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24",
            "-t", "2", "-vf", "select='if(lt(n,24),1,not(mod(n,2)))',settb=1/1000,setpts='PTS+if(gte(N,24),13,0)',format=yuv420p10le",
            "-fps_mode", "vfr", "-enc_time_base", "demux", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(raw),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-display_rotation", "90", "-i", str(raw), "-c", "copy",
            str(self.source),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        raw.unlink()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(self.transcoder.run())
        output = self.root / "done" / self.source.name
        self.assertTrue(output.exists())
        source_info = self.transcoder._get_video_info(self.source)
        output_info = self.transcoder._get_video_info(output)
        self.assertEqual(source_info.frames, 36)
        self.assertEqual(dict(source_info.video_properties)["rotation"], "90.0")
        self.assertEqual(output_info.pixel_format, "yuv420p10le")
        self.assertEqual(dict(output_info.video_properties)["rotation"], "90.0")
        self.assertTrue(self.transcoder._compatible(source_info, output_info))
        for source_time, output_time in zip(self.frame_times(self.source), self.frame_times(output)):
            self.assertAlmostEqual(source_time, output_time, places=5)

    def test_static_hdr_metadata_is_preserved(self):
        self.source.unlink()
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24",
            "-t", "2", "-pix_fmt", "yuv420p10le", "-c:v", "libx265",
            "-x265-params", "lossless=1:pools=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
            str(self.source),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        source_info = self.transcoder._get_video_info(self.source)
        self.assertTrue(source_info.hdr_metadata)
        with redirect_stdout(io.StringIO()) as log:
            result = self.transcoder.run()
        output = self.root / "done" / self.source.name
        if output.exists():
            self.assertTrue(result)
            self.assertEqual(source_info.hdr_metadata, self.transcoder._get_video_info(output).hdr_metadata)
        else:
            # Rejecting unsupported metadata is preferable to silent HDR loss.
            self.assertFalse(result)
            self.assertIn("metadata changed", log.getvalue())


if __name__ == "__main__":
    unittest.main()
