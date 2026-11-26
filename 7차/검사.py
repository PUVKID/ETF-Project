"""실제 엔진을 시작하지 않고 화면 연결·보존·요청 경계를 검사합니다."""
import ast
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import re
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('astryx_checked_dashboard', ROOT/'server.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


@contextmanager
def running(server):
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def fetch(server,path='/',data=None,headers=None):
    request=Request(f'http://127.0.0.1:{server.server_port}'+path,data=data,headers=headers or {})
    try:
        response=urlopen(request,timeout=3)
    except HTTPError as error:
        response=error
    with response:
        return response.status,response.read(),response.headers


class ScreenContract(unittest.TestCase):
    def test_01_original_sources_are_unchanged(self):
        manifest=json.loads((ROOT/'work/화면원본_고정목록_비공개.json').read_text(encoding='utf-8'))
        for item in manifest:
            with self.subTest(source=item['source']):
                raw=Path(item['source']).read_bytes()
                self.assertEqual(len(raw),item['size'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])

    def test_02_strict_text_and_python_syntax(self):
        paths=[ROOT/'server.py',ROOT/'검사.py',ROOT/'테마생성.py']
        front=ROOT/('dashboard/frontend/src' if app.PROJECT=='face' else 'web')
        paths += [p for p in front.rglob('*') if p.suffix in {'.js','.jsx','.css','.html'}]
        for path in paths:
            with self.subTest(path=str(path)):
                text=path.read_text(encoding='utf-8',errors='strict')
                self.assertNotIn('\ufffd',text)
                if path.suffix=='.py':ast.parse(text,filename=str(path))

    def test_03_generated_theme_is_connected(self):
        front=ROOT/('dashboard/frontend/src' if app.PROJECT=='face' else 'web')
        theme=(front/'astryx-theme.css').read_text(encoding='utf-8')
        adapter=(front/'astryx-adapter.css').read_text(encoding='utf-8')
        self.assertIn('Rust에서 생성',theme)
        self.assertIn('[data-theme="dark"]',theme)
        for token in ['--color-primary','--color-background-surface','--color-text-primary','--radius-element','--astryx-control-height']:
            self.assertIn(token,theme)
            self.assertIn('var('+token+')',adapter)
        self.assertTrue((front/'astryx-official.css').stat().st_size>100)
        entry=(front/'main.jsx' if app.PROJECT=='face' else front/'index.html').read_text(encoding='utf-8')
        for name in ['astryx-official.css','astryx-theme.css','astryx-adapter.css']:
            self.assertIn(name,entry)

    def test_04_controls_and_data_logic_are_preserved(self):
        manifest=json.loads((ROOT/'work/화면원본_고정목록_비공개.json').read_text(encoding='utf-8'))
        for item in manifest:
            if app.PROJECT!='face' and item['target'].endswith(('.js',)):
                self.assertEqual(hashlib.sha256((ROOT/item['target']).read_bytes()).hexdigest(),item['sha256'])
        if app.PROJECT!='face':
            original=(app.SOURCE/'web/index.html').read_text(encoding='utf-8')
            current=(ROOT/'web/index.html').read_text(encoding='utf-8')
            self.assertEqual(re.findall(r'\bid="([^"]+)"',original),re.findall(r'\bid="([^"]+)"',current))
        else:
            for name in ['Panels.jsx','Workflow.jsx','format.js','useStatus.js']:
                self.assertEqual((ROOT/'dashboard/frontend/src'/name).read_bytes(),(app.SOURCE/'dashboard/frontend/src'/name).read_bytes())

    def test_05_static_path_and_host_boundaries(self):
        with running(app.make_server(0,reader=lambda p:(200,b'{}'))) as server:
            self.assertEqual(fetch(server)[0],200)
            self.assertEqual(fetch(server,'/missing.css')[0],404)
            self.assertEqual(fetch(server,'/%2e%2e/server.py')[0],403)
            self.assertEqual(fetch(server,'/.codex/routing-checks.json')[0],403)
            self.assertEqual(fetch(server,headers={'Host':'untrusted.example'})[0],403)
            self.assertEqual(fetch(server,'/api/unknown')[0],404)

    def test_06_read_only_and_csrf(self):
        with running(app.make_server(0,reader=lambda p:(200,b'{}'))) as server:
            path='/api/jobs' if app.PROJECT=='etf' else '/api/status'
            self.assertEqual(fetch(server,path,b'{}')[0],403 if app.PROJECT=='etf' else 405)
            if app.PROJECT=='etf':
                self.assertEqual(fetch(server,path,b'{"action":"start"}',{'X-Local-Request':'1','Origin':'https://untrusted.example','Content-Type':'application/json'})[0],403)
                self.assertEqual(fetch(server,path,b'{"action":"unknown"}',{'X-Local-Request':'1','Content-Type':'application/json'})[0],400)

    def test_07_api_data_and_failure_are_visible(self):
        if app.PROJECT=='etf':return
        with running(app.make_server(0,reader=lambda p:(200,b'{"actual":0,"unknown":null}'))) as server:
            self.assertEqual(json.loads(fetch(server,'/api/status')[1]),{'actual':0,'unknown':None})
        def unavailable(path):raise OSError('격리된 연결 실패')
        with running(app.make_server(0,reader=unavailable)) as server:
            status,body,_=fetch(server,'/api/status')
            self.assertEqual(status,503)
            self.assertIn('기존 상태',json.loads(body)['error'])

    def test_08_etf_relays_to_one_original_manager(self):
        if app.PROJECT!='etf':return
        calls=[]
        class Original(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(('GET',self.path))
                body=b'{"worker_alive":true,"actual":0}'
                self.send_response(200);self.end_headers();self.wfile.write(body)
            def do_POST(self):
                calls.append(('POST',self.path,self.rfile.read(int(self.headers['Content-Length'])),self.headers.get('Origin'),self.headers.get('X-Local-Request')))
                self.send_response(409);self.end_headers();self.wfile.write(b'{"accepted":false}')
            def log_message(self,*args):pass
        with running(ThreadingHTTPServer(('127.0.0.1',0),Original)) as original:
            with running(app.make_server(0,upstream_port=original.server_port)) as server:
                self.assertEqual(json.loads(fetch(server,'/api/state')[1])['actual'],0)
                status,_,_=fetch(server,'/api/jobs',b'{"action":"start"}',{'Content-Type':'application/json','X-Local-Request':'1','Origin':f'http://127.0.0.1:{server.server_port}'})
                self.assertEqual(status,409)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1][1],'/api/jobs')
        self.assertEqual(calls[1][2],b'{"action":"start"}')
        self.assertEqual(calls[1][4],'1')

if __name__=='__main__':unittest.main(verbosity=2)
