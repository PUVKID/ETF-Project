"""외부 공개·실제 주문 없이 loopback에서만 실행하는 대시보드 서버입니다."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit

from src.runtime import ENV_ROOT, PRIVATE, ROOT, STOP, is_busy, load_state, update_state
from src.core import read_json
from src.paper import ACCOUNT
from src.autotrading import account_view, trade_page


class JobManager:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()

    def launch(self, action):
        with self.lock:
            if action == "stop":
                PRIVATE.mkdir(parents=True, exist_ok=True)
                STOP.write_text("사용자 중지 요청\n", encoding="utf-8")
                return True, "중지 요청을 전달했습니다. 현재 학습 단계 저장 후 멈춥니다."
            if (self.process and self.process.poll() is None) or is_busy():
                return False, "수집·학습 작업이 이미 실행 중입니다. 중복 실행하지 않습니다."
            PRIVATE.mkdir(parents=True, exist_ok=True)
            STOP.unlink(missing_ok=True)
            executable = ENV_ROOT / ".venv" / "Scripts" / "python.exe"
            if not executable.exists():
                return False, "프로젝트 .venv가 없습니다. requirements.txt에 따라 설치해 주세요."
            update_state(job={"status": "starting", "action": action, "continuous": action == "start",
                              "message": "로컬 작업을 시작하고 있습니다", "next_run_at": None})
            with (PRIVATE / "작업출력.log").open("a", encoding="utf-8") as output:
                self.process = subprocess.Popen([str(executable), "-B", "-X", "utf8", str(ROOT / "run_pipeline.py"), action],
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True, "계속 학습·모의운용을 시작했습니다" if action == "start" else "작업을 시작했습니다"


def make_server(port=8879):
    manager = JobManager()
    web = (ROOT / "web").resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalETF/1.0"

        def log_message(self, fmt, *args):
            # API 폴링을 매초 디스크에 기록하지 않습니다.
            return

        def send_bytes(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def reply(self, status, payload):
            self.send_bytes(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, {"error": "로컬 Host만 허용합니다"})
                return
            route = unquote(urlsplit(self.path).path)
            if route == "/api/state":
                state = load_state()
                account = read_json(ACCOUNT)
                if account is not None:
                    state["paper"] = account_view(account)
                state["worker_alive"] = is_busy()
                self.reply(200, state)
                return
            if route == "/api/trades":
                try:
                    query = parse_qs(urlsplit(self.path).query, keep_blank_values=True, strict_parsing=True)
                    if set(query) - {"offset", "limit", "side"} or any(len(values) != 1 for values in query.values()):
                        raise ValueError("중복 또는 지원하지 않는 조회 항목입니다")
                    offset = int(query.get("offset", ["0"])[0])
                    limit = int(query.get("limit", ["50"])[0])
                    side = query.get("side", ["all"])[0]
                    self.reply(200, trade_page(read_json(ACCOUNT, {"trades": [], "positions": []}), offset, limit, side))
                except (ValueError, TypeError) as error:
                    self.reply(400, {"error": str(error)})
                return
            if route.startswith("/api/"):
                self.reply(404, {"error": "지원하지 않는 API입니다"})
                return
            target = (web / ("index.html" if route == "/" else route.lstrip("/"))).resolve()
            try:
                target.relative_to(web)
            except ValueError:
                self.reply(404, {"error": "파일을 찾을 수 없습니다"})
                return
            if not target.is_file():
                self.reply(404, {"error": "파일을 찾을 수 없습니다"})
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or target.suffix == ".js":
                content_type += "; charset=utf-8"
            self.send_bytes(200, target.read_bytes(), content_type)

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Local-Request") != "1":
                self.reply(403, {"error": "로컬 화면의 요청만 허용합니다"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.reply(403, {"error": "다른 출처의 변경 요청은 허용하지 않습니다"})
                return
            if urlsplit(self.path).path != "/api/jobs":
                self.reply(404, {"error": "지원하지 않는 API입니다"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("요청 크기가 유효하지 않습니다")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON 요청이 필요합니다")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or set(payload) != {"action"} or payload["action"] not in ("refresh", "train", "stop", "start"):
                    raise ValueError("지원하는 작업은 refresh/train/stop/start입니다")
            except (ValueError, UnicodeError) as error:
                self.reply(400, {"error": str(error)})
                return
            accepted, message = manager.launch(payload["action"])
            self.reply(202 if accepted else 409, {"accepted": accepted, "message": message, "state": load_state()})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.manager = manager
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"로컬 대시보드 http://127.0.0.1:{server.server_port}", flush=True)
    if args.continuous:
        print(server.manager.launch("start")[1], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STOP.write_text("서버 종료에 따른 작업 중지\n", encoding="utf-8")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
