#!/usr/bin/env python3
"""按需 CUDA Whisper ASR 网关：POST /api/asr，返回 {"text": "..."}。"""

import argparse
import cgi
import json
import os
import socket
import subprocess
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[3]
TEMP_DIR = PROJECT_DIR / "data" / "asr" / "tmp"
DEFAULT_BINARY = PROJECT_DIR / "third_party" / "whisper.cpp" / "build" / "bin" / "whisper-server"
DEFAULT_MODEL = PROJECT_DIR / "data" / "asr" / "whispercpp" / "ggml-base.bin"
FALLBACK_MODEL_DIR = PROJECT_DIR / "data" / "asr" / "models"
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class ASRRuntime:
    def __init__(self, binary, model_path, language, worker_port, fallback_model_name):
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.binary = Path(binary)
        self.model_path = Path(model_path)
        self.language = language
        self.fallback_model_name = fallback_model_name
        self.worker_port = worker_port
        self.worker = None
        self.fallback_model = None
        self.last_request = time.monotonic()
        self._lock = threading.Lock()

    def touch(self):
        self.last_request = time.monotonic()

    def _start_worker(self):
        if self.worker and self.worker.poll() is None:
            return
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK) or not self.model_path.is_file():
            return False
        command = [
            str(self.binary), "--host", "127.0.0.1", "--port", str(self.worker_port),
            "-m", str(self.model_path), "-t", "4", "-fa", "-sns", "-nt",
        ]
        if self.language != "auto":
            command.extend(("-l", self.language))
        self.worker = subprocess.Popen(command)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.worker.poll() is not None:
                raise RuntimeError("CUDA Whisper 服务启动失败，请查看 journalctl -u gemma4-asr.service")
            try:
                with socket.create_connection(("127.0.0.1", self.worker_port), timeout=0.3):
                    return
            except OSError:
                time.sleep(0.2)
        self.stop_worker()
        raise RuntimeError("CUDA Whisper 服务启动超时")

    def transcribe(self, audio_path):
        with self._lock:
            if self._start_worker() is False:
                # 首次 CUDA 编译尚未结束时保持原 tiny CPU 服务可用；无需额外安装。
                if self.fallback_model is None:
                    import whisper
                    FALLBACK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    self.fallback_model = whisper.load_model(self.fallback_model_name, device="cpu", download_root=str(FALLBACK_MODEL_DIR))
                options = {"fp16": False, "verbose": False, "temperature": 0, "condition_on_previous_text": False}
                if self.language != "auto":
                    options["language"] = self.language
                result = self.fallback_model.transcribe(str(audio_path), **options)
                return self.to_simplified(result.get("text") or "")
            boundary = "----gemma4asrboundary"
            audio = audio_path.read_bytes()
            body = b"".join((
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n',
                b"Content-Type: audio/wav\r\n\r\n", audio, b"\r\n",
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n',
                f"--{boundary}--\r\n".encode(),
            ))
            request = Request(
                f"http://127.0.0.1:{self.worker_port}/inference", data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST",
            )
            try:
                with urlopen(request, timeout=90) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(f"Whisper 推理请求失败：{error}") from error
            return self.to_simplified(result.get("text") or "")

    @staticmethod
    def to_simplified(text):
        """使用 OpenCC 台湾繁体到大陆简体词典统一输出；不可用时不阻断识别。"""
        text = text.strip()
        if not text or not shutil.which("opencc"):
            return text
        try:
            converted = subprocess.run(
                ["opencc", "-c", "tw2sp.json"], input=text, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=3, check=True,
            ).stdout.strip()
            return converted or text
        except (OSError, subprocess.SubprocessError):
            return text

    def stop_worker(self):
        if self.worker and self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
        self.worker = None


class Handler(BaseHTTPRequestHandler):
    server_version = "Gemma4ASR/2.0"

    def log_message(self, _format, *_args):
        return

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json(200, {})

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json(200, {"status": "ok", "backend": "whisper.cpp-cuda"})
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/asr":
            self._json(404, {"detail": "not found"})
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(415, {"detail": "请使用 multipart/form-data 上传 file"})
            return
        if int(self.headers.get("Content-Length", "0")) > MAX_AUDIO_BYTES:
            self._json(413, {"detail": "音频超过 25 MB 限制"})
            return
        try:
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type})
            if "file" not in form or not getattr(form["file"], "file", None):
                self._json(400, {"detail": "缺少音频 file"})
                return
            item = form["file"]
            payload = item.file.read(MAX_AUDIO_BYTES + 1)
            if len(payload) > MAX_AUDIO_BYTES:
                self._json(413, {"detail": "音频超过 25 MB 限制"})
                return
            with tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=".wav", delete=False) as output:
                output.write(payload)
                audio_path = Path(output.name)
            try:
                self.server.runtime.touch()
                text = self.server.runtime.transcribe(audio_path)
            finally:
                audio_path.unlink(missing_ok=True)
            self._json(200, {"text": text})
        except Exception as error:
            self._json(500, {"detail": f"ASR 识别失败：{error}"})


def make_server(args, runtime):
    if not args.socket_activation:
        server = ThreadingHTTPServer((args.bind, args.port), Handler)
    else:
        listen_pid = int(os.environ.get("LISTEN_PID", "0"))
        listen_fds = int(os.environ.get("LISTEN_FDS", "0"))
        if listen_pid != os.getpid() or listen_fds < 1:
            raise RuntimeError("systemd socket activation 环境变量无效")
        activated_socket = socket.socket(fileno=3)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
        server.socket.close()
        server.socket = activated_socket
        server.server_address = activated_socket.getsockname()
    server.timeout = 1
    server.runtime = runtime
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-activation", action="store_true")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--idle-seconds", type=int, default=180)
    parser.add_argument("--binary", default=os.environ.get("ASR_BINARY", str(DEFAULT_BINARY)))
    parser.add_argument("--model-path", default=os.environ.get("ASR_MODEL_PATH", str(DEFAULT_MODEL)))
    parser.add_argument("--language", default=os.environ.get("ASR_LANGUAGE", "auto"))
    parser.add_argument("--fallback-model", default=os.environ.get("ASR_FALLBACK_MODEL", "base"))
    parser.add_argument("--worker-port", type=int, default=int(os.environ.get("ASR_WORKER_PORT", "8002")))
    args = parser.parse_args()
    runtime = ASRRuntime(args.binary, args.model_path, args.language, args.worker_port, args.fallback_model)
    server = make_server(args, runtime)
    try:
        while time.monotonic() - runtime.last_request < args.idle_seconds:
            server.handle_request()
    finally:
        runtime.stop_worker()
        server.server_close()


if __name__ == "__main__":
    main()
