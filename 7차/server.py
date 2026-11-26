"""새 화면과 기존 실행을 분리하는 127.0.0.1 전용 대시보드입니다."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import mimetypes
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
PROJECT = 'etf'
SOURCE = Path('E:/2026/ETF Project/6차')
WEB = ROOT / 'web'
MAX_BYTES = 8 * 1024 * 1024


def source_module(name, path):
    """원본의 읽기 어댑터만 적재하며 main·서비스 시작 함수를 호출하지 않습니다."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def make_server(port=8880, upstream_port=8879, reader=None):
    web = WEB.resolve()
    if reader is None and PROJECT == 'upbit':
        sys.path.insert(0, str(SOURCE/'tools'))
        original = source_module('astryx_original_upbit_gateway', SOURCE/'tools/docker_control.py')
        def reader(path):
            result = original.relay(path)
            status = result['status']
            if type(status) is not int or not 200 <= status <= 599:
                raise ValueError('원본 응답 상태가 유효하지 않습니다')
            raw = base64.b64decode(result['body_base64'], validate=True)
            if len(raw) > MAX_BYTES:
                raise ValueError('원본 응답 상한을 넘었습니다')
            return status, raw
    elif reader is None and PROJECT == 'face':
        sys.path.insert(0, str(SOURCE))
        from benchmark.status import read_public_status, log_tail, json_bytes
        state_dir = SOURCE.parent/'work/stage1_revision2'
        def reader(path):
            parsed = urlsplit(path)
            if parsed.path == '/api/status':
                if parsed.query:
                    return 400, json_bytes({'error': '상태 조회는 입력을 받지 않습니다'})
                return 200, json_bytes(read_public_status(state_dir))
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
            if set(query)-{'id', 'limit'} or any(len(values) != 1 for values in query.values()):
                raise ValueError('로그 조회 인자가 유효하지 않습니다')
            try:
                return 200, json_bytes(log_tail(state_dir, query.get('id', ['acquisition'])[0], int(query.get('limit', ['100'])[0])))
            except (ValueError, TypeError):
                return 400, json_bytes({'error': '허용된 로그와 줄 수만 조회할 수 있습니다'})

    class Handler(BaseHTTPRequestHandler):
        server_version = 'AstryxLocalDashboard/1'
        sys_version = ''

        def log_message(self, *args):
            # URL·인증값을 접근 로그에 복제하지 않습니다.
            pass

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def send(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def reply(self, status, value):
            self.send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))

        def relay_etf(self, body=None):
            headers = {'Accept': 'application/json'}
            if body is not None:
                headers.update({'Content-Type': 'application/json', 'X-Local-Request': '1', 'Origin': f'http://127.0.0.1:{upstream_port}'})
            request = Request(f'http://127.0.0.1:{upstream_port}'+self.path, data=body, headers=headers)
            try:
                response = urlopen(request, timeout=10)
            except HTTPError as error:
                response = error
            with response:
                raw = response.read(MAX_BYTES+1)
                if len(raw) > MAX_BYTES:
                    raise ValueError('원본 응답 상한을 넘었습니다')
                self.send(response.status, raw)

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {'error': '로컬 Host만 허용합니다'})
            parsed = urlsplit(self.path)
            route = unquote(parsed.path)
            allowed = {'etf': {'/api/state', '/api/trades'}, 'upbit': {'/api/status'}, 'face': {'/api/status', '/api/logs'}}[PROJECT]
            if route.startswith('/api'):
                if route not in allowed:
                    return self.reply(404, {'error': '없는 API입니다'})
                try:
                    if PROJECT == 'etf':
                        return self.relay_etf()
                    if PROJECT == 'upbit' and parsed.query:
                        return self.reply(400, {'error': '상태 조회는 입력을 받지 않습니다'})
                    status, body = reader(self.path)
                    if len(body) > MAX_BYTES:
                        raise ValueError('응답 상한을 넘었습니다')
                    return self.send(status, body)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                    print(f'기존 상태 연결 실패: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
                    return self.reply(503, {'error': '기존 상태에 연결하지 못했습니다. 실행 상태를 원래 프로젝트에서 확인해 주세요.'})
            if parsed.query or '\\' in route or any(part.startswith('.') for part in Path(route).parts):
                return self.reply(403, {'error': '정적 경로를 거부했습니다'})
            target = web/(route.lstrip('/') or 'index.html')
            if not target.resolve().is_relative_to(web) or any(parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()) for parent in [target, *target.parents] if parent != web.parent):
                return self.reply(403, {'error': '화면 폴더 밖의 경로입니다'})
            if not target.is_file():
                return self.reply(404, {'error': '화면 파일이 없습니다'})
            if target.stat().st_size > 10*1024*1024:
                return self.reply(413, {'error': '정적 출력 상한을 넘었습니다'})
            content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
            if content_type.startswith('text/') or target.suffix == '.js':
                content_type += '; charset=utf-8'
            self.send(200, target.read_bytes(), content_type)

        def do_POST(self):
            if not self.valid_host():
                return self.reply(403, {'error': '로컬 Host만 허용합니다'})
            if PROJECT != 'etf':
                return self.reply(405, {'error': '읽기 전용 화면입니다'})
            if self.path != '/api/jobs':
                return self.reply(404, {'error': '없는 API입니다'})
            if self.headers.get('X-Local-Request') != '1' or self.headers.get('Origin') not in {None, f'http://{self.headers.get("Host")}'}:
                return self.reply(403, {'error': '현재 로컬 화면의 요청만 허용합니다'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('유효한 JSON 요청이 필요합니다')
                body = self.rfile.read(length)
                value = json.loads(body.decode('utf-8'))
                if not isinstance(value, dict) or set(value) != {'action'} or value['action'] not in {'refresh', 'train', 'stop', 'start'}:
                    raise ValueError('원래 화면에서 지원하는 작업만 허용합니다')
            except (ValueError, UnicodeError, TypeError) as error:
                return self.reply(400, {'error': str(error)})
            try:
                # 새 작업자를 만들지 않고 원래 서버의 중복 실행 방어를 사용합니다.
                self.relay_etf(body)
            except (OSError, ValueError) as error:
                print(f'원래 작업 요청 연결 실패: {error}', file=sys.stderr, flush=True)
                self.reply(503, {'error': '원래 서버에 요청을 전달하지 못했습니다. 자동 재시도하지 않습니다.'})

        def reject(self):
            self.reply(405, {'error': '지원하지 않는 요청입니다'})

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = reject

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8880)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f'새 차수 화면 http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
