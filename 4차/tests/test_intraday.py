"""공급자 거래시간과 완결 시각을 보존하는 단기 시세 검사입니다."""

from datetime import datetime, timezone
import importlib
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


class IntradayTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("src.intraday"), "단기 관측 구현 전")
        return importlib.import_module("src.intraday")

    def chart(self):
        base = int(datetime(2026, 11, 9, 20, 58, tzinfo=timezone.utc).timestamp())
        return {"meta": {"symbol": "VTI", "currency": "USD", "instrumentType": "ETF", "dataGranularity": "1m",
                         "tradingPeriods": [[{"start": base - 23280, "end": base + 120}]]},
                "timestamp": [base, base + 60, base + 120],
                "indicators": {"quote": [{"close": [100, 101, 102], "volume": [3, 2, 1]}]}}

    def test_incomplete_trailing_and_after_close_bars_are_excluded(self):
        module = self.module()
        quote = module.parse_quote("VTI", self.chart(), "2026-11-09T20:59:30+00:00", "fixture")
        self.assertEqual(quote["price"], 100)
        self.assertEqual(quote["bar_end"], "2026-11-09T20:59:00+00:00")
        final = module.parse_quote("VTI", self.chart(), "2026-11-09T21:01:00+00:00", "fixture")
        self.assertEqual(final["price"], 101)
        self.assertEqual(final["session_close"], "2026-11-09T21:00:00+00:00")

    def test_missing_calendar_and_wrong_symbol_never_become_trade_quotes(self):
        module = self.module()
        chart = self.chart()
        chart["meta"].pop("tradingPeriods")
        with self.assertRaises(ValueError):
            module.parse_quote("VTI", chart, "2026-11-09T21:01:00+00:00", "fixture")
        with self.assertRaises(ValueError):
            module.parse_quote("GLD", self.chart(), "2026-11-09T21:01:00+00:00", "fixture")

    def test_one_unavailable_etf_does_not_fabricate_or_discard_other_quotes(self):
        module = self.module()
        def fetch(symbol):
            if symbol == "BND":
                raise RuntimeError("검사용 공급자 수집 실패")
            return self.chart(), "fixture"
        result = module.collect_quotes(["VTI", "BND"], "2026-11-09T21:01:00+00:00", fetch)
        self.assertEqual(set(result["quotes"]), {"VTI"})
        self.assertIn("수집 실패", result["errors"]["BND"])

    def test_stop_during_network_collection_does_not_publish_or_fill_afterward(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            stop, quotes = Path(folder) / "중지", Path(folder) / "시세.json"
            finished = threading.Event()
            def collected(*args, **kwargs):
                stop.write_text("사용자 중지", encoding="utf-8")
                finished.set()
                return {"quotes": {}, "errors": {}, "error_details": {}, "status": "ready"}
            with patch.object(module, "STOP", stop), patch.object(module, "QUOTES", quotes), \
                 patch.object(module, "load_config", return_value={"symbols": ["VTI"], "autotrading": {"quote_poll_seconds": 60}}), \
                 patch.object(module, "collect_quotes", side_effect=collected), patch("src.paper.reconcile_paper"):
                module.monitor_loop(finished)
            self.assertFalse(quotes.exists(), "중지 요청 후 뒤늦게 도착한 시세를 발행하지 않아야 합니다")


if __name__ == "__main__":
    unittest.main()
