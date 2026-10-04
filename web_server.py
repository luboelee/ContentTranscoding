"""Local web interface for ContentTranscoding. No third-party packages required."""
import argparse
import copy
import json
import mimetypes
import os
import secrets
import stat
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from ContentTranscoding import ContentTranscoding, COMPRESS_RATIOS, THRESHOLD_PSNR, THRESHOLD_SSIM


APP_DIRECTORY = Path(__file__).resolve().parent
STATIC_DIRECTORY = APP_DIRECTORY / "web"
MAX_FILES = 5000
MAX_BODY = 2 * 1024 * 1024


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def is_link(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        return True
    if os.name == "nt":
        try:
            return getattr(path.lstat(), "st_reparse_tag", 0) == stat.IO_REPARSE_TAG_MOUNT_POINT
        except OSError:
            return False
    return False


class InstanceLock:
    """An OS-held lock prevents two servers from changing the same job history."""
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / "server.lock").open("a+b")
        if os.fstat(self.file.fileno()).st_size == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("같은 작업 폴더를 사용하는 웹서버가 이미 실행 중입니다.") from None

    def close(self):
        self.file.close()


class PathAccess:
    def __init__(self, roots):
        self.roots = tuple(dict.fromkeys(Path(root).resolve() for root in roots))
        if not self.roots or any(not root.is_dir() for root in self.roots):
            raise ValueError("Browse roots must be existing directories")

    def resolve(self, raw):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("파일 또는 폴더 경로를 선택해 주세요.")
        path = Path(raw).expanduser().resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.roots):
            raise PermissionError("선택 가능한 폴더 범위를 벗어난 경로입니다.")
        return path

    def browse(self, raw):
        path = self.resolve(raw)
        if not path.is_dir():
            raise ValueError("폴더 경로를 입력해 주세요.")
        entries = []
        for child in path.iterdir():
            if child.name.startswith(".") or is_link(child):
                continue
            try:
                if child.is_dir():
                    entries.append({"name": child.name, "path": str(child), "type": "folder"})
                elif child.suffix.lower() == ".mp4" and child.is_file():
                    entries.append({"name": child.name, "path": str(child), "type": "file", "size": child.stat().st_size})
            except OSError:
                continue
        entries.sort(key=lambda entry: (entry["type"] != "folder", entry["name"].casefold()))
        parent = path.parent
        return {"path": str(path), "parent": str(parent) if parent != path and any(parent.is_relative_to(r) for r in self.roots) else None,
                "entries": entries[:2000], "truncated": len(entries) > 2000}

    def select(self, paths, recursive=False):
        if not isinstance(paths, list) or not 0 < len(paths) <= MAX_FILES:
            raise ValueError("하나 이상의 파일 또는 폴더를 선택해 주세요.")
        if not isinstance(recursive, bool):
            raise ValueError("하위 폴더 포함 설정이 올바르지 않습니다.")
        files = {}
        excluded = {"done", "temporary", ".web-data", ".git", "node_modules"}
        visited = 0

        def add(path):
            nonlocal visited
            visited += 1
            if visited > 100000:
                raise ValueError("폴더가 너무 큽니다. 더 작은 폴더를 선택해 주세요.")
            if is_link(path) or path.suffix.lower() != ".mp4" or not path.is_file():
                return
            resolved = self.resolve(str(path))
            if resolved not in files:
                files[resolved] = {"path": str(resolved), "name": resolved.name, "size": resolved.stat().st_size}
            if len(files) > MAX_FILES:
                raise ValueError(f"한 번에 최대 {MAX_FILES:,}개 영상까지 선택할 수 있습니다.")

        for raw in paths:
            # Reject symbolic links before resolving them to real files.
            if isinstance(raw, str) and is_link(Path(raw).expanduser()):
                raise ValueError("링크 대신 원본 파일 또는 폴더를 선택해 주세요.")
            path = self.resolve(raw)
            if path.is_file():
                if path.suffix.lower() != ".mp4":
                    raise ValueError("현재 MP4 영상만 지원합니다.")
                add(path)
            elif recursive:
                def walk_error(error):
                    raise error
                for directory, folders, names in os.walk(path, followlinks=False, onerror=walk_error):
                    folders[:] = [name for name in folders if name not in excluded and not name.startswith(".")
                                  and not is_link(Path(directory) / name)]
                    for name in names:
                        add(Path(directory) / name)
            else:
                for child in path.iterdir():
                    add(child)
        if not files:
            raise ValueError("선택한 위치에 MP4 영상이 없습니다.")
        return sorted(files.values(), key=lambda entry: entry["path"].casefold())


class JobManager:
    def __init__(self, data_directory):
        self.directory = Path(data_directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active_process = None
        self.running_job_id = None
        self.replacing_jobs = set()
        self.idle = threading.Event()
        self.idle.set()
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if path.parent.name != job["id"] or len(job["id"]) != 32:
                    continue
                if job["status"] in {"queued", "running"}:
                    job.update(status="interrupted", finished_at=timestamp(), phase="interrupted",
                               error="서버가 종료되어 작업이 중단되었습니다. 새 작업으로 다시 실행해 주세요.")
                    write_json(path, job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue

    def _persist(self, job):
        write_json(self.directory / job["id"] / "job.json", job)

    def list(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item["created_at"], reverse=True)
            return [{key: copy.deepcopy(job[key]) for key in (
                "id", "status", "created_at", "finished_at", "total", "completed", "label", "summary"
            )} for job in jobs]

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("작업을 찾을 수 없습니다.")
            job = copy.deepcopy(self.jobs[job_id])
        for index, record in enumerate(job["records"]):
            record["download_url"] = f"/api/jobs/{job_id}/files/{index}" if record.get("output_path") else None
        job["reports"] = {extension: f"/api/jobs/{job_id}/report.{extension}" for extension in ("csv", "json")
                          if (self.directory / job_id / f"results.{extension}").is_file()}
        job["replacing"] = job_id in self.replacing_jobs
        job["replaceable_count"] = sum(record.get("status") == "accepted" and bool(record.get("output_path"))
                                       and not record.get("replaced_at") for record in job["records"])
        return job

    @staticmethod
    def _copy_atomic(source, destination, before_replace=None):
        """Finish copying on the destination volume before touching the original."""
        descriptor, name = tempfile.mkstemp(prefix=".transcoding-", suffix=".tmp", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != source.stat().st_size:
                raise OSError("복사한 파일 크기가 결과 파일과 다릅니다.")
            if before_replace:
                before_replace()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def replace_originals(self, job_id, access):
        with self.lock:
            directory = self.directory / job_id
            if (len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id)
                    or is_link(directory) or directory.resolve().parent != self.directory):
                raise PermissionError("허용되지 않은 작업 경로입니다.")
            job = self.get(job_id)
            if self.replacing_jobs:
                raise RuntimeError("원본 파일을 대체하는 중입니다.")
            records = [(index, record) for index, record in enumerate(job["records"])
                       if record["status"] == "accepted" and record.get("output_path")
                       and not record.get("replaced_at")]
            if not records:
                raise RuntimeError("대체할 수 있는 성공한 결과 파일이 없습니다.")
            selected = {entry["path"] for entry in job.get("files", [])}
            busy_sources = {entry["path"] for key, other in self.jobs.items()
                            if key != job_id and other["status"] in {"queued", "running"}
                            for entry in other.get("files", [])}
            self.replacing_jobs.add(job_id)
        replaced, errors, reports = [], [], set()
        try:
            for index, record in records:
                try:
                    raw = record.get("source_path")
                    if raw not in selected or is_link(Path(raw)):
                        raise PermissionError("이 작업에서 선택한 원본 파일만 대체할 수 있습니다.")
                    if raw in busy_sources:
                        raise RuntimeError("다른 작업에서 사용 중인 원본입니다. 작업 완료 후 다시 시도해 주세요.")
                    source = access.resolve(raw)
                    if not source.is_file() or source.suffix.lower() != ".mp4":
                        raise ValueError("원본 MP4 파일을 찾을 수 없습니다.")
                    output = self.artifact(job_id, f"files/{index}")
                    size = output.stat().st_size
                    if size <= 0 or size != record.get("trans_file_size") or size >= record.get("orig_file_size", 0):
                        raise ValueError("결과 파일의 크기가 검증된 압축 결과와 다릅니다.")
                    original_stat = source.stat()
                    if original_stat.st_size != record.get("orig_file_size") or (
                        record.get("source_mtime_ns") is not None
                        and original_stat.st_mtime_ns != record["source_mtime_ns"]
                    ):
                        raise RuntimeError("트랜스코딩 이후 원본이 변경되어 대체하지 않았습니다.")

                    # Each output group belongs to one original folder. Copy its
                    # reports even when other groups are still being transcoded.
                    for extension in ("json", "csv"):
                        report = output.parent / f"measured_data.{extension}"
                        resolved = report.resolve(strict=True)
                        if not resolved.is_relative_to(self.directory / job_id) or not resolved.is_file():
                            raise PermissionError("허용되지 않은 보고서 경로입니다.")
                        destination = source.parent / f"measured_data_{job_id}.{extension}"
                        if destination not in reports:
                            if is_link(destination):
                                raise PermissionError("보고서 대상이 링크 파일입니다.")
                            self._copy_atomic(resolved, destination)
                            reports.add(destination)

                    def verify_original():
                        if is_link(Path(raw)) or access.resolve(raw) != source:
                            raise PermissionError("원본 파일 경로가 변경되었습니다.")
                        current = source.stat()
                        if (current.st_size, current.st_mtime_ns, current.st_ino) != (
                            original_stat.st_size, original_stat.st_mtime_ns, original_stat.st_ino
                        ):
                            raise RuntimeError("복사 중 원본이 변경되어 대체하지 않았습니다.")

                    self._copy_atomic(output, source, verify_original)
                    with self.lock:
                        current_record = self.jobs[job_id]["records"][index]
                        current_record.update(replaced_at=timestamp(), replacement_error=None)
                        self._persist(self.jobs[job_id])
                    replaced.append(str(source))
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    errors.append({"file_name": record["file_name"], "error": str(error)})
                    with self.lock:
                        self.jobs[job_id]["records"][index]["replacement_error"] = str(error)
                        self._persist(self.jobs[job_id])
            return {"replaced_count": len(replaced), "replaced_paths": replaced,
                    "report_paths": sorted(str(path) for path in reports), "errors": errors}
        finally:
            with self.lock:
                self.replacing_jobs.discard(job_id)

    def delete_history(self, job_id=None):
        """Remove persisted history only; leave sources, videos and reports intact."""
        with self.lock:
            if job_id is not None:
                if job_id not in self.jobs:
                    raise KeyError("작업을 찾을 수 없습니다.")
                if (self.jobs[job_id]["status"] in {"queued", "running"} or job_id == self.running_job_id
                        or job_id in self.replacing_jobs):
                    raise RuntimeError("실행 중인 작업의 기록은 삭제할 수 없습니다.")
                selected = [job_id]
            else:
                selected = [key for key, job in self.jobs.items()
                            if job["status"] not in {"queued", "running"} and key != self.running_job_id
                            and key not in self.replacing_jobs]
            deleted = []
            for key in selected:
                directory = self.directory / key
                if (len(key) != 32 or any(character not in "0123456789abcdef" for character in key)
                        or is_link(directory) or directory.resolve().parent != self.directory):
                    raise PermissionError("허용되지 않은 작업 기록 경로입니다.")
                (directory / "job.json").unlink(missing_ok=True)
                del self.jobs[key]
                deleted.append(key)
            return {"deleted_ids": deleted, "preserved_directory": str(self.directory)}

    @staticmethod
    def summary(records):
        accepted = [record for record in records if record["status"] == "accepted" and record.get("output_path")]
        original = sum(record["orig_file_size"] for record in accepted)
        compressed = sum(record["trans_file_size"] for record in accepted)
        return {"accepted": len(accepted), "unchanged": sum(r["status"] == "unchanged" for r in records),
                "failed": sum(r["status"] == "failed" for r in records),
                "original_size": original, "output_size": compressed, "saved_size": original - compressed,
                "saved_percent": (1 - compressed / original) * 100 if original else 0}

    def start(self, files, settings):
        if not isinstance(settings, dict):
            raise ValueError("트랜스코딩 설정이 올바르지 않습니다.")
        settings = {"psnr": settings.get("psnr", THRESHOLD_PSNR), "ssim": settings.get("ssim", THRESHOLD_SSIM),
                    "ratios": settings.get("ratios", list(COMPRESS_RATIOS)), "encoder": settings.get("encoder", "auto")}
        if (any(isinstance(settings[key], bool) or not isinstance(settings[key], (int, float)) for key in ("psnr", "ssim"))
                or not isinstance(settings["ratios"], list)
                or any(isinstance(r, bool) or not isinstance(r, (int, float)) for r in settings["ratios"])
                or len(settings["ratios"]) > 20):
            raise ValueError("숫자 형식의 화질 기준과 비트레이트 비율을 입력해 주세요.")
        if not isinstance(settings["encoder"], str) or settings["encoder"] not in {"auto", "cpu", "cuda"}:
            raise ValueError("인코딩 방식은 자동, CPU, NVIDIA GPU 중에서 선택해 주세요.")
        ContentTranscoding(argparse.Namespace(path=str(Path(files[0]["path"]).parent), **settings))
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ValueError("FFmpeg와 ffprobe를 설치한 뒤 서버를 다시 실행해 주세요.")
        with self.lock:
            if self.running_job_id or self.replacing_jobs or any(job["status"] in {"queued", "running"} for job in self.jobs.values()):
                raise RuntimeError("실행 중인 작업이 있습니다. 완료된 후 새 작업을 시작해 주세요.")
            job_id = uuid.uuid4().hex
            directory = self.directory / job_id
            directory.mkdir()
            job = {"id": job_id, "status": "queued", "created_at": timestamp(), "started_at": None,
                   "finished_at": None, "label": files[0]["name"] + (f" 외 {len(files)-1}개" if len(files) > 1 else ""),
                   "total": len(files), "completed": 0, "current_file": None, "phase": "queued",
                   "attempt": None, "max_attempts": len(settings["ratios"]), "encoder": settings["encoder"],
                   "bitrate": None, "metrics": None, "settings": settings, "files": files,
                   "records": [], "logs": [], "error": None, "summary": self.summary([])}
            event_token = secrets.token_hex(16)
            write_json(directory / "request.json", {"files": files, "settings": settings, "event_token": event_token})
            self.jobs[job_id] = job
            self._persist(job)
            self.running_job_id = job_id
            self.idle.clear()
            threading.Thread(target=self._run, args=(job_id, event_token), daemon=False, name=f"transcode-{job_id[:8]}").start()
            return self.get(job_id)

    def _run(self, job_id, event_token):
        directory = self.directory / job_id
        prefix = f"@frame-event/{event_token}:"
        try:
            with self.lock:
                job = self.jobs[job_id]
                job.update(status="running", started_at=timestamp(), phase="analyzing")
                self._persist(job)
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with subprocess.Popen(
                [sys.executable, str(APP_DIRECTORY / "web_worker.py"), str(directory)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                env=environment, cwd=APP_DIRECTORY, creationflags=flags,
            ) as process, (directory / "worker.log").open("w", encoding="utf-8") as log:
                with self.lock:
                    self.active_process = process
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        job = self.jobs[job_id]
                        if line.startswith(prefix):
                            self._event(job, json.loads(line[len(prefix):]))
                        else:
                            job["logs"].append(line.rstrip()[:2000])
                            job["logs"] = job["logs"][-200:]
                        self._persist(job)
                code = process.wait()
            with self.lock:
                job = self.jobs[job_id]
                final_status = job.pop("final_status", "failed")
                job["status"] = final_status if code == 0 else "failed"
                if job["status"] == "failed":
                    job["error"] = job["error"] or "일부 파일의 처리가 실패했습니다. 결과와 로그를 확인해 주세요."
                job.update(finished_at=timestamp(), phase=job["status"])
                self._persist(job)
        except Exception as error:
            with self.lock:
                job = self.jobs[job_id]
                job.update(status="failed", phase="failed", error=str(error), finished_at=timestamp())
                self._persist(job)
        finally:
            with self.lock:
                self.active_process = None
                self.running_job_id = None
                self.idle.set()

    def _event(self, job, event):
        kind = event["event"]
        if kind in {"results", "finished"}:
            previous = {record.get("source_path"): record for record in job["records"]}
            for record in event["records"]:
                saved = previous.get(record.get("source_path"), {})
                for key in ("replaced_at", "replacement_error"):
                    if key in saved:
                        record[key] = saved[key]
        if kind == "file_started":
            job.update(current_file=event["file_name"], current_path=event["source_path"],
                       position=event["position"], metrics=None, attempt=None, phase="analyzing")
        elif kind == "phase":
            for key in ("phase", "attempt", "max_attempts", "bitrate", "encoder"):
                if key in event:
                    job[key] = event[key]
        elif kind == "encoder_changed":
            job["encoder"] = event["encoder"]
        elif kind == "metrics":
            job["metrics"] = {key: event.get(key) for key in ("psnr_avg", "psnr_y", "ssim_all", "ssim_y")}
        elif kind == "file_completed":
            job["completed"] = event["position"]
            record = {**event["record"], "source_path": job.get("current_path"), "output_path": None}
            job["records"].append(record)
            job["summary"] = self.summary(job["records"])
        elif kind == "results":
            job.update(records=event["records"], completed=event["completed"], summary=self.summary(event["records"]))
        elif kind == "finished":
            job.update(records=event["records"], final_status=event["status"], summary=self.summary(event["records"]))
        elif kind == "error":
            job["error"] = event["message"]

    def artifact(self, job_id, name):
        job = self.get(job_id)
        directory = self.directory / job_id
        if name in {"report.csv", "report.json"}:
            path = directory / name.replace("report", "results")
        elif name.startswith("files/"):
            try:
                index = int(name.split("/")[1])
                if index < 0:
                    raise ValueError
                record = job["records"][index]
                if not record.get("output_path"):
                    raise ValueError
                path = Path(record["output_path"])
            except (IndexError, ValueError):
                raise KeyError("결과 파일을 찾을 수 없습니다.") from None
        else:
            raise KeyError("결과 파일을 찾을 수 없습니다.")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(directory) or not resolved.is_file():
            raise PermissionError("허용되지 않은 다운로드 경로입니다.")
        return resolved


class TranscodingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, access, manager):
        super().__init__(address, Handler)
        self.access, self.manager = access, manager
        self.token = secrets.token_urlsafe(32)
        self.allowed_hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "FrameStudio/1.0"

    def log_message(self, format, *args):
        if args and str(args[1] if len(args) > 1 else "").startswith("4"):
            super().log_message(format, *args)

    def _trusted(self):
        if self.headers.get("Host") not in self.server.allowed_hosts:
            raise PermissionError("허용되지 않은 호스트입니다.")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in self.server.allowed_hosts}:
            raise PermissionError("동일한 웹페이지에서만 요청할 수 있습니다.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("외부 페이지 요청은 허용되지 않습니다.")

    def _headers(self, status, content_type, size, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _file(self, path, attachment=False):
        with path.open("rb") as file:
            extra = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(path.name)}"} if attachment else None
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in {".html", ".css", ".js"}:
                mime += "; charset=utf-8"
            self._headers(200, mime, os.fstat(file.fileno()).st_size, extra)
            shutil.copyfileobj(file, self.wfile)

    def _dispatch(self, method):
        try:
            body = None
            if method in {"POST", "DELETE"}:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("요청 크기가 올바르지 않습니다.")
                self.connection.settimeout(15)
                # Drain a bounded request before returning an authorization error.
                # Closing with unread data can reset the socket on Windows.
                body = self.rfile.read(length)
            self._trusted()
            url = urlsplit(self.path)
            path = url.path
            if method == "GET":
                if path in {"/", "/app.js", "/styles.css", "/favicon.svg"}:
                    self._file(STATIC_DIRECTORY / ("index.html" if path == "/" else path[1:]))
                elif path == "/api/config":
                    locations = [Path.cwd(), Path.home(), Path.home() / "Videos", *self.server.access.roots]
                    locations = list(dict.fromkeys(p.resolve() for p in locations if p.is_dir()
                        and any(p.resolve().is_relative_to(r) for r in self.server.access.roots)))
                    self._json(200, {"token": self.server.token, "locations": [
                        {"name": "프로젝트 폴더" if p == Path.cwd() else p.name or str(p), "path": str(p)} for p in locations],
                        "initial_path": str(locations[0]), "history_directory": str(self.server.manager.directory),
                        "ffmpeg_ready": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
                        "defaults": {"psnr": THRESHOLD_PSNR, "ssim": THRESHOLD_SSIM, "ratios": COMPRESS_RATIOS, "encoder": "auto"}})
                elif path == "/api/browse":
                    query = parse_qs(url.query)
                    self._json(200, self.server.access.browse(query.get("path", [""])[0]))
                elif path == "/api/jobs":
                    self._json(200, {"jobs": self.server.manager.list()})
                elif path.startswith("/api/jobs/"):
                    parts = path.strip("/").split("/")
                    if len(parts) == 3:
                        self._json(200, self.server.manager.get(parts[2]))
                    else:
                        self._file(self.server.manager.artifact(parts[2], "/".join(parts[3:])), attachment=True)
                else:
                    self._json(404, {"error": "페이지를 찾을 수 없습니다."})
            else:
                if not secrets.compare_digest(self.headers.get("X-CSRF-Token", ""), self.server.token):
                    raise PermissionError("페이지를 새로고침한 뒤 다시 시도해 주세요.")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON 형식의 요청이 필요합니다.")
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("요청 형식이 올바르지 않습니다.")
                if method == "DELETE":
                    parts = path.strip("/").split("/")
                    if path == "/api/jobs":
                        self._json(200, self.server.manager.delete_history())
                    elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                        self._json(200, self.server.manager.delete_history(parts[2]))
                    else:
                        self._json(404, {"error": "API를 찾을 수 없습니다."})
                    return
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "replace":
                    self._json(200, self.server.manager.replace_originals(parts[2], self.server.access))
                    return
                files = self.server.access.select(data.get("paths"), data.get("recursive", False))
                if path == "/api/selection":
                    self._json(200, {"files": files, "total_size": sum(f["size"] for f in files)})
                elif path == "/api/jobs":
                    self._json(202, self.server.manager.start(files, data.get("settings", {})))
                else:
                    self._json(404, {"error": "API를 찾을 수 없습니다."})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PermissionError as error:
            self._json(403, {"error": str(error)})
        except (FileNotFoundError, KeyError) as error:
            self._json(404, {"error": str(error).strip("'")})
        except RuntimeError as error:
            self._json(409, {"error": str(error)})
        except (ValueError, TypeError) as error:
            self._json(400, {"error": str(error)})
        except OSError as error:
            self._json(400, {"error": f"파일에 접근할 수 없습니다: {error}"})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


def default_roots():
    if os.name == "nt":
        return [Path(f"{letter}:/") for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{letter}:/").is_dir()]
    return [Path.home(), Path.cwd()]


def main():
    parser = argparse.ArgumentParser(description="Local video transcoding web server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", action="append", help="Limit the file browser to this directory (repeatable)")
    parser.add_argument("--data-dir", default=str(APP_DIRECTORY / ".web-data"), help="Job history and outputs directory")
    parser.add_argument("--open", action="store_true", help="Open the web interface in the default browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    instance = None
    try:
        data_directory = Path(args.data_dir).resolve()
        instance = InstanceLock(data_directory)
        server = TranscodingServer(("127.0.0.1", args.port), PathAccess(args.root or default_roots()),
                                  JobManager(data_directory / "jobs"))
    except (OSError, ValueError, RuntimeError) as error:
        if instance is not None:
            instance.close()
        parser.error(str(error))
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"Frame Studio: {address}", flush=True)
    print("Press Ctrl+C to stop. Originals are preserved until Replace is used.", flush=True)
    if args.open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping the web server...", flush=True)
        # Non-daemon manager threads keep draining the child and persisting results.
        with server.manager.lock:
            process = server.manager.active_process
        if process is not None:
            print("Waiting for the active transcode to finish and save its results...", flush=True)
    finally:
        server.server_close()
        server.manager.idle.wait()
        instance.close()


if __name__ == "__main__":
    main()
