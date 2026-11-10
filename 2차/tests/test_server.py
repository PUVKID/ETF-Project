import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ServerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("server"), "로컬 서버 구현 전")
        from server import make_server
        self.server = make_server(0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        if hasattr(self, "server"):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(3)

    def response(self, path, data=None, headers=None):
        request = Request(self.base + path, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=3) as result:
                return result.status, result.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_second_revision_serves_its_own_ui_and_reuses_first_engine(self):
        import server
        self.assertEqual(server.ROOT.resolve(), (Path(__file__).resolve().parents[2] / "1차").resolve())
        status, body = self.response("/")
        self.assertEqual(status, 200)
        self.assertEqual(body, (Path(__file__).resolve().parents[1] / "web/index.html").read_bytes())

    def test_state_is_readable_on_loopback(self):
        status, body = self.response("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["schema_version"], 1)
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_state_remains_readable_while_a_worker_holds_the_real_file_lock(self):
        from src.runtime import worker_lock
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.runtime.LOCK", Path(folder) / "작업잠금"):
                with worker_lock():
                    status, body = self.response("/api/state")
                    self.assertEqual(status, 200)
                    self.assertTrue(json.loads(body)["worker_alive"])

    def test_external_origin_and_missing_header_cannot_start_work(self):
        payload = b'{"action":"train"}'
        self.assertEqual(self.response("/api/jobs", payload)[0], 403)
        headers = {"X-Local-Request": "1", "Origin": "https://external.example", "Content-Type": "application/json"}
        self.assertEqual(self.response("/api/jobs", payload, headers)[0], 403)

    def test_rebinding_and_private_file_access_are_blocked(self):
        self.assertEqual(self.response("/api/state", headers={"Host": "external.example"})[0], 403)
        for path in ("/work/비공개/개인설정.json", "/..%2Fwork%2Fsecret.json", "/server.py"):
            encoded = path.replace("비공개", "%EB%B9%84%EA%B3%B5%EA%B0%9C").replace("개인설정", "%EA%B0%9C%EC%9D%B8%EC%84%A4%EC%A0%95")
            self.assertEqual(self.response(encoded)[0], 404)

    def test_real_order_and_invalid_action_have_no_route(self):
        headers = {"X-Local-Request": "1", "Content-Type": "application/json"}
        self.assertEqual(self.response("/api/jobs", b'{"action":"order"}', headers)[0], 400)
        self.assertEqual(self.response("/api/orders", b'{}', headers)[0], 404)
        self.assertEqual(self.response("/api/jobs", b'[]', headers)[0], 400)


if __name__ == "__main__":
    unittest.main()
