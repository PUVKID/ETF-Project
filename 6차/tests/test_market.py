from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.core import read_json, save_json
from src import market


class MarketFailureTests(unittest.TestCase):
    def test_network_failure_does_not_replace_previous_real_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "시세.json"
            old = {"series": {"A": [{"date": "2024-01-02", "open": 10, "close": 11}]}, "metadata": {}}
            save_json(path, old)
            with patch.object(market, "MARKET", path), patch.object(market, "load_config", return_value={"symbols": ["A"]}), \
                 patch.object(market, "update_state"), patch.object(market, "fetch_chart", side_effect=ConnectionError("검사용 연결 실패")):
                with self.assertRaisesRegex(ConnectionError, "검사용 연결 실패"):
                    market.refresh_market()
            self.assertEqual(read_json(path), old)


if __name__ == "__main__":
    unittest.main()
