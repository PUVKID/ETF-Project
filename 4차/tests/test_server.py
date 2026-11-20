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

    def test_trade_page_filters_real_ledger_without_mutating_it(self):
        from src.core import save_json
        with tempfile.TemporaryDirectory() as folder:
            account = Path(folder) / "모의계좌.json"
            original = {"positions": [], "trades": [
                {"symbol": "VTI", "side": "매수", "quantity": 1},
                {"symbol": "GLD", "side": "매도", "quantity": 2},
                {"symbol": "BND", "side": "매수", "quantity": 3}]}
            save_json(account, original)
            before = account.read_bytes()
            with patch("server.ACCOUNT", account, create=True):
                status, body = self.response("/api/trades?offset=1&limit=1&side=buy")
                self.assertEqual(status, 200)
                payload = json.loads(body)
                self.assertEqual(payload["total"], 2)
                self.assertEqual(payload["trades"][0]["symbol"], "VTI")
                self.assertIsNone(payload["trades"][0]["realized_pnl_krw"])
            self.assertEqual(account.read_bytes(), before)

    def test_trade_page_rejects_ambiguous_invalid_and_private_queries(self):
        for query in ("offset=-1", "offset=1.5", "limit=0", "limit=101", "side=all&side=buy", "side=unknown", "path=secret", "offset="):
            with self.subTest(query=query):
                self.assertEqual(self.response("/api/trades?" + query)[0], 400)
        self.assertEqual(self.response("/api/trades/..%2Fwork%2Fsecret")[0], 404)
        headers = {"X-Local-Request": "1", "Content-Type": "application/json"}
        self.assertEqual(self.response("/api/trades", b'{}', headers)[0], 404)


if __name__ == "__main__":
    unittest.main()
