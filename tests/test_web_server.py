import hashlib
import http.client
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from web_server import InstanceLock, JobManager, PathAccess, TranscodingServer


class WebServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.directory.name)
        self.media = self.root / "media"
        self.media.mkdir()
        self.manager = JobManager(self.root / "jobs")
        self.server = TranscodingServer(("127.0.0.1", 0), PathAccess([self.media]), self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.directory.cleanup()

    def request(self, path, data=None, headers=None, method=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        request_headers = headers or {}
        if data is not None:
            request_headers = {"Content-Type": "application/json", "X-CSRF-Token": self.server.token, **request_headers}
        connection.request(method or ("POST" if data is not None else "GET"), path,
                           body=json.dumps(data).encode() if data is not None else None, headers=request_headers)
        response = connection.getresponse()
        body = response.read()
        content_type = response.getheader("Content-Type")
        status = response.status
        connection.close()
        return status, json.loads(body) if "application/json" in content_type else body

    def test_static_page_and_config(self):
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("화질은 지키고".encode(), page)
        status, config = self.request("/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(config["token"], self.server.token)
        self.assertEqual(config["initial_path"], str(self.media))

    def test_folder_selection_excludes_outputs_and_supports_recursion(self):
        (self.media / "a.MP4").write_bytes(b"a")
        child = self.media / "nested"
        child.mkdir()
        (child / "b.mp4").write_bytes(b"bb")
        done = self.media / "done"
        done.mkdir()
        (done / "old.mp4").write_bytes(b"old")
        status, preview = self.request("/api/selection", {"paths": [str(self.media)], "recursive": False})
        self.assertEqual(status, 200)
        self.assertEqual([f["name"] for f in preview["files"]], ["a.MP4"])
        status, preview = self.request("/api/selection", {"paths": [str(self.media)], "recursive": True})
        self.assertEqual(status, 200)
        self.assertEqual([f["name"] for f in preview["files"]], ["a.MP4", "b.mp4"])
        self.assertEqual(preview["total_size"], 3)

    def test_root_escape_and_cross_origin_requests_are_denied(self):
        status, _ = self.request("/api/browse?path=" + str(self.root))
        self.assertEqual(status, 403)
        status, _ = self.request("/api/config", headers={"Origin": "https://untrusted.example"})
        self.assertEqual(status, 403)
        status, _ = self.request("/api/config", headers={"Host": f"untrusted.example:{self.server.server_port}"})
        self.assertEqual(status, 403)
        status, _ = self.request("/api/config", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(status, 403)

    def test_csrf_required_and_invalid_selection_is_rejected(self):
        status, _ = self.request("/api/selection", {"paths": [str(self.media)]}, {"X-CSRF-Token": "invalid"})
        self.assertEqual(status, 403)
        status, _ = self.request("/api/selection", {"paths": []})
        self.assertEqual(status, 400)
        text = self.media / "notes.txt"
        text.write_text("not a video")
        status, _ = self.request("/api/selection", {"paths": [str(text)]})
        self.assertEqual(status, 400)

    def test_history_marks_unfinished_jobs_as_interrupted(self):
        job_id = "a" * 32
        directory = self.root / "history" / job_id
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(json.dumps({"id": job_id, "status": "running"}))
        manager = JobManager(directory.parent)
        self.assertEqual(manager.jobs[job_id]["status"], "interrupted")
        self.assertIsNotNone(manager.jobs[job_id]["finished_at"])

    def test_artifact_cannot_download_outside_job_directory(self):
        job_id = "b" * 32
        (self.manager.directory / job_id).mkdir()
        source = self.media / "source.mp4"
        source.write_bytes(b"original")
        self.manager.jobs[job_id] = {"records": [{"output_path": str(source)}]}
        with self.assertRaises(PermissionError):
            self.manager.artifact(job_id, "files/0")
        with self.assertRaises(KeyError):
            self.manager.artifact(job_id, "files/-1")

    def test_duplicate_files_are_deduplicated(self):
        source = self.media / "a.mp4"
        source.write_bytes(b"video")
        files = self.server.access.select([str(self.media), str(source)])
        self.assertEqual(len(files), 1)

    def test_invalid_settings_are_rejected_without_starting_worker(self):
        source = self.media / "a.mp4"
        source.write_bytes(b"video")
        for settings in ({"psnr": float("nan")}, {"ssim": 1.1}, {"ratios": [0.9, 0.6]},
                         {"ratios": [True]}, {"encoder": None}, {"psnr": True}):
            with self.subTest(settings=settings):
                status, _ = self.request("/api/jobs", {"paths": [str(source)], "settings": settings})
                self.assertEqual(status, 400)
        self.assertFalse(self.manager.jobs)

    def test_second_active_job_is_rejected(self):
        source = self.media / "a.mp4"
        source.write_bytes(b"video")
        self.manager.running_job_id = "existing-job"
        with patch("web_server.shutil.which", return_value="executable"):
            status, message = self.request("/api/jobs", {"paths": [str(source)]})
        self.assertEqual(status, 409)
        self.assertIn("실행 중", message["error"])

    def test_instance_lock_prevents_duplicate_servers_and_releases_on_close(self):
        first = InstanceLock(self.root / "lock-test")
        try:
            with self.assertRaises(RuntimeError):
                InstanceLock(self.root / "lock-test")
        finally:
            first.close()
        second = InstanceLock(self.root / "lock-test")
        second.close()

    def make_history(self, job_id, status="completed"):
        directory = self.manager.directory / job_id
        directory.mkdir()
        job = {"id": job_id, "status": status, "created_at": "2026-10-05T00:00:00+00:00",
               "finished_at": None, "total": 1, "completed": 1, "label": "sample.mp4",
               "summary": {}, "records": []}
        self.manager.jobs[job_id] = job
        self.manager._persist(job)
        return directory

    def test_delete_history_persists_without_removing_sources_or_results(self):
        job_id = "c" * 32
        directory = self.make_history(job_id)
        source = self.media / "original.mp4"
        source.write_bytes(b"original")
        output = directory / "output.mp4"
        output.write_bytes(b"compressed")
        report = directory / "results.json"
        report.write_text("[]")
        status, response = self.request(f"/api/jobs/{job_id}", {}, method="DELETE")
        self.assertEqual(status, 200)
        self.assertEqual(response["deleted_ids"], [job_id])
        self.assertFalse((directory / "job.json").exists())
        self.assertEqual(source.read_bytes(), b"original")
        self.assertEqual(output.read_bytes(), b"compressed")
        self.assertEqual(report.read_text(), "[]")
        self.assertEqual(self.request(f"/api/jobs/{job_id}")[0], 404)
        self.assertEqual(self.request(f"/api/jobs/{job_id}/report.json")[0], 404)
        self.assertNotIn(job_id, JobManager(self.manager.directory).jobs)
        self.assertEqual(self.request(f"/api/jobs/{job_id}", {}, method="DELETE")[0], 404)

    def test_delete_running_history_is_rejected(self):
        for status in ("queued", "running", "completed"):
            with self.subTest(status=status):
                job_id = {"queued": "d", "running": "e", "completed": "f"}[status] * 32
                directory = self.make_history(job_id, status)
                self.manager.running_job_id = job_id
                self.assertEqual(self.request(f"/api/jobs/{job_id}", {}, method="DELETE")[0], 409)
                self.assertTrue((directory / "job.json").exists())

    def test_clear_history_excludes_active_jobs_and_is_repeatable(self):
        removed = ["1" * 32, "2" * 32, "3" * 32]
        for job_id, status in zip(removed, ("completed", "failed", "interrupted")):
            self.make_history(job_id, status)
        for job_id, status in (("4" * 32, "running"), ("5" * 32, "queued"), ("6" * 32, "completed")):
            self.make_history(job_id, status)
        self.manager.running_job_id = "6" * 32
        status, response = self.request("/api/jobs", {}, method="DELETE")
        self.assertEqual(status, 200)
        self.assertEqual(response["deleted_ids"], removed)
        self.assertEqual(set(self.manager.jobs), {"4" * 32, "5" * 32, "6" * 32})
        for job_id in self.manager.jobs:
            self.assertTrue((self.manager.directory / job_id / "job.json").exists())
        self.assertEqual(self.request("/api/jobs", {}, method="DELETE")[1]["deleted_ids"], [])

    def test_delete_requires_token_and_trusted_origin(self):
        job_id = "7" * 32
        directory = self.make_history(job_id)
        for headers in ({"X-CSRF-Token": "invalid"}, {"Origin": "https://untrusted.example"},
                        {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(f"/api/jobs/{job_id}", {}, headers, method="DELETE")[0], 403)
                self.assertTrue((directory / "job.json").exists())

    def test_delete_rejects_invalid_persisted_job_path(self):
        self.manager.jobs[".."] = {"status": "completed"}
        outside = self.manager.directory.parent / "job.json"
        outside.write_text("must remain")
        status, _ = self.request("/api/jobs/..", {}, method="DELETE")
        self.assertEqual(status, 403)
        self.assertEqual(outside.read_text(), "must remain")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_web_job_transcodes_only_selected_files_and_exposes_results(self):
        source = self.media / "selected [한글].mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24",
            "-t", "1", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", str(source),
        ], check=True, stderr=subprocess.PIPE)
        unrelated = self.media / "unselected.mp4"
        unrelated.write_bytes(b"must not be processed")
        original_hash = hashlib.sha256(source.read_bytes()).digest()
        status, job = self.request("/api/jobs", {"paths": [str(source)], "settings": {"encoder": "cpu"}})
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status, job = self.request(f"/api/jobs/{job['id']}")
            if job["status"] not in {"running", "queued"}:
                break
            time.sleep(0.1)
        self.assertEqual(job["status"], "completed", job.get("error") or job.get("logs"))
        self.assertEqual(job["total"], 1)
        self.assertEqual(job["records"][0]["status"], "accepted")
        self.assertEqual(job["summary"]["accepted"], 1)
        self.assertIsNotNone(job["finished_at"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), original_hash)
        self.assertEqual(unrelated.read_bytes(), b"must not be processed")
        self.assertFalse((self.media / "done").exists())
        status, result = self.request(job["records"][0]["download_url"])
        self.assertEqual(status, 200)
        self.assertLess(len(result), source.stat().st_size)
        status, report = self.request(job["reports"]["json"])
        self.assertEqual(status, 200)
        self.assertEqual(report[0]["file_name"], source.name)
        status, history = self.request("/api/jobs")
        self.assertEqual(history["jobs"][0]["id"], job["id"])
        reloaded = JobManager(self.manager.directory)
        self.assertEqual(reloaded.jobs[job["id"]]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
