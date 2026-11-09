import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


class CoreTests(unittest.TestCase):
    def core(self):
        self.assertIsNotNone(importlib.util.find_spec("src.core"), "핵심 계산 모듈 구현 전")
        from src import core
        return core

    def test_duplicate_dates_are_not_counted_twice(self):
        c = self.core()
        row = {"date": "2024-01-02", "open": 10, "close": 11}
        self.assertEqual(len(c.validate_rows([row, dict(row)])), 1)
        with self.assertRaises(ValueError):
            c.validate_rows([row, {**row, "close": 12}])

    def test_invalid_and_incomplete_prices_are_rejected(self):
        c = self.core()
        for price in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                c.validate_rows([{"date": "2024-01-02", "open": 1, "close": price}])
        self.assertFalse(c.session_completed("2024-01-02", "2024-01-02T20:00:00+00:00"))
        self.assertTrue(c.session_completed("2024-01-02", "2024-01-02T22:00:00+00:00"))

    def test_splits_are_chronological_and_disjoint(self):
        c = self.core()
        dates = [f"2024-01-{i:02d}" for i in range(1, 31)]
        split = c.make_split(dates)
        self.assertEqual(split["train_end"], "2024-01-18")
        self.assertEqual(split["validation_end"], "2024-01-24")
        self.assertEqual(split["test_start"], "2024-01-25")
        with self.assertRaises(ValueError):
            c.make_split(dates[:4])

    def test_a_signal_cannot_capture_its_own_close_to_next_open_jump(self):
        c = self.core()
        series = {"A": [
            {"date": "2024-01-02", "open": 100, "close": 100},
            {"date": "2024-01-03", "open": 200, "close": 200},
            {"date": "2024-01-04", "open": 200, "close": 400},
        ]}
        result = c.backtest(series, fee_bps=0)
        self.assertEqual(result["history"][1]["value"], 100)
        self.assertEqual(result["history"][2]["value"], 200)
        self.assertEqual(result["trades"][0]["signal_date"], "2024-01-02")
        self.assertEqual(result["trades"][0]["execution_date"], "2024-01-03")

    def test_future_rows_do_not_change_past_execution(self):
        c = self.core()
        rows = [{"date": f"2024-01-0{i}", "open": 10, "close": 10 + i} for i in range(2, 6)]
        before = c.backtest({"A": rows}, fee_bps=10)
        changed = [dict(row) for row in rows]
        changed[-1]["close"] = 900
        after = c.backtest({"A": changed}, fee_bps=10)
        self.assertEqual(before["history"][:-1], after["history"][:-1])

    def test_costs_lower_returns_and_rebalance_is_next_session(self):
        c = self.core()
        rows = [{"date": d, "open": 100, "close": 100} for d in ("2024-01-30", "2024-01-31", "2024-02-01")]
        free = c.backtest({"A": rows}, fee_bps=0)
        paid = c.backtest({"A": rows}, fee_bps=10, rebalance=True)
        self.assertLess(paid["history"][-1]["value"], free["history"][-1]["value"])
        self.assertAlmostEqual(paid["costs"], 100 - paid["history"][-1]["value"], places=7)
        self.assertEqual(paid["trades"][0]["execution_date"], "2024-01-31")

    def test_prediction_metrics_have_known_answer(self):
        c = self.core()
        metrics = c.forecast_metrics([100, 200], [110, 180], [90, 210])
        self.assertEqual(metrics["mae"], 15)
        self.assertAlmostEqual(metrics["mape"], 0.1)
        self.assertEqual(metrics["direction_accuracy"], 1)
        with self.assertRaises(ValueError):
            c.forecast_metrics([100], [math.nan], [90])

    def test_atomic_json_does_not_leave_temporary_files(self):
        c = self.core()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "상태.json"
            c.save_json(target, {"메시지": "실측"})
            self.assertEqual(c.read_json(target)["메시지"], "실측")
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
