"""새 주문 뒤 관측만 사용하는 자동 모의매매의 경계 검사입니다."""

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


FX = {"rate": 1000, "as_of": "2026-11-09", "source": "검사용 환율"}
NOW = "2026-11-09T15:05:10+00:00"


class AutoTradingTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("src.autotrading"), "자동 매매 구현 전")
        return importlib.import_module("src.autotrading")

    def account(self):
        from src.paper import create_account
        return create_account(FX, "2026-11-06", spread_bps=0)

    def quote(self, **changes):
        value = {"symbol": "VTI", "price": 100.0, "bar_start": "2026-11-09T15:04:00+00:00",
                 "bar_end": "2026-11-09T15:05:00+00:00", "fetched_at": NOW,
                 "session_open": "2026-11-09T14:30:00+00:00", "session_close": "2026-11-09T21:00:00+00:00",
                 "interval_seconds": 60, "source": "검사용 완결 1분봉", "regular_session": True}
        return {**value, **changes}

    def signal(self, weights=None, stamp="2026-11-09T15:03:00+00:00", key="new"):
        return {"id": key, "as_of": "2026-11-06", "created_at": stamp, "policy_version": 3,
                "universe_hash": "fixture", "weights": weights or {"VTI": 0.5},
                "reasons": {"VTI": "모델 전망이 비용 기준을 넘었습니다"}}

    def test_policy_selects_new_etfs_and_can_keep_all_cash(self):
        module = self.module()
        forecasts = [{"symbol": s, "adjusted_price": p, "last_adjusted_price": 100}
                     for s, p in (("QQQM", 99), ("VTI", 103), ("GLD", 102), ("BND", 101))]
        allocation = module.select_allocation(forecasts, 1000)
        self.assertGreater(allocation["weights"]["VTI"], 0)
        self.assertEqual(allocation["weights"]["QQQM"], 0)
        self.assertLessEqual(sum(v > 0 for v in allocation["weights"].values()), 3)
        self.assertLessEqual(sum(allocation["weights"].values()), 0.9)
        self.assertAlmostEqual(sum(allocation["weights"].values()) + allocation["cash_weight"], 1)
        down = module.select_allocation([{**row, "adjusted_price": 99} for row in forecasts], 1000)
        self.assertEqual(down["cash_weight"], 1)

    def test_post_order_completed_quote_fills_once_without_cash_overrun(self):
        module = self.module()
        account = self.account()
        module.queue_orders(account, self.signal(), {"VTI": 100})
        self.assertEqual(account["orders"][0]["status"], "pending")
        module.fill_orders(account, {"VTI": self.quote()}, FX, NOW)
        self.assertEqual(len(account["trades"]), 1)
        self.assertEqual(account["positions"][0]["quantity"], 5)
        self.assertAlmostEqual(account["positions"][0]["cost_basis_usd"], 500.5)
        self.assertAlmostEqual(account["cash_usd"], 499.5)
        module.fill_orders(account, {"VTI": self.quote()}, FX, NOW)
        self.assertEqual(len(account["trades"]), 1)
        self.assertEqual(account["orders"][0]["status"], "filled")

    def test_quotes_before_order_future_stale_outside_session_and_incomplete_wait(self):
        module = self.module()
        bad = [self.quote(bar_start="2026-11-09T15:02:00+00:00", bar_end="2026-11-09T15:03:00+00:00"),
               self.quote(bar_end="2026-11-09T15:06:00+00:00"),
               self.quote(bar_start="2026-11-09T14:40:00+00:00", bar_end="2026-11-09T14:41:00+00:00"),
               self.quote(regular_session=False), self.quote(interval_seconds=86400)]
        for quote in bad:
            with self.subTest(quote=quote):
                account = self.account()
                module.queue_orders(account, self.signal(stamp="2026-11-09T14:31:00+00:00"), {"VTI": 100})
                if quote["bar_start"] == "2026-11-09T15:02:00+00:00":
                    account["orders"][0]["created_at"] = "2026-11-09T15:03:00+00:00"
                module.fill_orders(account, {"VTI": quote}, FX, NOW)
                self.assertEqual(account["trades"], [])
                self.assertEqual(account["orders"][0]["status"], "pending")
                self.assertTrue(account["orders"][0]["waiting_reason"])
        account = self.account()
        module.queue_orders(account, self.signal(), {"VTI": 100})
        module.fill_orders(account, {"VTI": self.quote()}, FX, "2026-11-09T22:00:00+00:00")
        self.assertEqual(account["trades"], [])

    def test_partial_sale_releases_cost_and_full_sale_removes_position(self):
        module = self.module()
        account = self.account()
        module.queue_orders(account, self.signal(), {"VTI": 100})
        module.fill_orders(account, {"VTI": self.quote()}, FX, NOW)
        account["orders"].append({"id": "sale-1", "symbol": "VTI", "side": "매도", "quantity": 2,
                                  "created_at": "2026-11-09T15:03:00+00:00", "status": "pending", "reason": "배분 축소"})
        module.fill_orders(account, {"VTI": self.quote(price=120)}, FX, NOW)
        self.assertAlmostEqual(account["positions"][0]["cost_basis_usd"], 300.3)
        self.assertAlmostEqual(account["trades"][-1]["realized_pnl_krw"], 39550.0)
        account["orders"].append({"id": "sale-2", "symbol": "VTI", "side": "매도", "quantity": 3,
                                  "created_at": "2026-11-09T15:03:00+00:00", "status": "pending", "reason": "현금 보유"})
        module.fill_orders(account, {"VTI": self.quote(price=120)}, FX, NOW)
        self.assertEqual(account["positions"], [])
        self.assertGreaterEqual(account["cash_usd"], 0)

    def test_missing_fx_and_missing_symbol_quote_never_fill(self):
        module = self.module()
        for quotes, fx in (({}, FX), ({"VTI": self.quote()}, None)):
            account = self.account()
            module.queue_orders(account, self.signal(), {"VTI": 100})
            module.fill_orders(account, quotes, fx, NOW)
            self.assertEqual(account["trades"], [])

    def test_migration_preserves_cash_receipt_and_unknown_position_cost(self):
        module = self.module()
        account = self.account()
        receipt = deepcopy(account["fx_receipt"])
        account["positions"] = [{"symbol": "VTI", "quantity": 2, "price_usd": 100}]
        module.migrate_account(account)
        self.assertEqual(account["cash_usd"], 1000)
        self.assertEqual(account["fx_receipt"], receipt)
        self.assertIsNone(account["positions"][0]["cost_basis_usd"])
        self.assertIsNone(account["positions"][0]["average_cost_usd"])
        self.assertEqual(len(account["cash_movements"]), 1)
        module.migrate_account(account)
        self.assertEqual(len(account["cash_movements"]), 1)

    def test_new_policy_cancels_pending_old_policy_without_erasing_history(self):
        module = self.module()
        account = self.account()
        account["orders"] = [{"id": "legacy", "symbol": "QQQM", "side": "매수", "status": "pending"}]
        module.queue_orders(account, self.signal(), {"VTI": 100})
        self.assertEqual(account["orders"][0]["status"], "cancelled")
        self.assertTrue(account["orders"][0]["cancel_reason"])
        count = len(account["orders"])
        module.queue_orders(account, self.signal(), {"VTI": 100})
        self.assertEqual(len(account["orders"]), count)

    def test_split_preserves_total_cost_and_changes_average_cost(self):
        module = self.module()
        account = self.account()
        account["positions"] = [{"symbol": "VTI", "quantity": 2, "price_usd": 100,
                                  "cost_basis_usd": 200, "cost_basis_krw": 200000, "average_cost_usd": 100}]
        module.apply_corporate_actions(account, {"VTI": [{"date": "2026-11-09", "split": 2, "dividend": 0}]})
        self.assertEqual(account["positions"][0]["quantity"], 4)
        self.assertEqual(account["positions"][0]["average_cost_usd"], 50)
        self.assertEqual(account["positions"][0]["cost_basis_krw"], 200000)
        module.apply_corporate_actions(account, {"VTI": [{"date": "2026-11-09", "split": 2, "dividend": 0}]})
        self.assertEqual(account["positions"][0]["quantity"], 4)

    def test_expansion_training_excludes_initial_holdout_dates(self):
        from src import modeling
        self.assertTrue(callable(getattr(modeling, "build_training_extension", None)), "누출 방어 확장 구현 전")
        rows = [{"date": day, "close": value} for day, value in
                (("2024-06-02", 1), ("2024-06-03", 2), ("2024-06-04", 999), ("2026-11-09", 888))]
        original = {"QQQM": rows[:2]}
        result = modeling.build_training_extension({"QQQM": rows, "VTI": rows}, original, "2024-06-03")
        self.assertEqual([row["close"] for row in result["VTI"]], [1, 2])
        self.assertNotIn("QQQM", result)
        self.assertEqual(original, {"QQQM": rows[:2]})

    def test_later_fx_funds_the_same_account_once(self):
        module = self.module()
        self.assertTrue(callable(getattr(module, "fund_account", None)), "환율 대기 계좌의 보존 환전 구현 전")
        from src.paper import create_account
        account = create_account(None, "2026-11-06")
        account["marker"] = "기존 원장 보존"
        module.fund_account(account, FX, 50)
        self.assertAlmostEqual(account["cash_usd"], 995.0248756218906)
        self.assertEqual(account["marker"], "기존 원장 보존")
        receipt = deepcopy(account["fx_receipt"])
        module.fund_account(account, FX, 50)
        self.assertEqual(account["fx_receipt"], receipt)
        self.assertEqual(len(account["cash_movements"]), 1)

    def test_new_forecast_scope_preserves_legacy_first_forecast(self):
        from src import modeling, paper
        from src.core import read_json, save_json
        from src.universe import UNIVERSE_HASH
        with tempfile.TemporaryDirectory() as folder:
            private = Path(folder)
            legacy = {"symbol": "QQQM", "as_of": "2026-11-06", "created_at": "2026-11-09T15:00:00+00:00", "actual": None}
            save_json(private / "전진예측원장.json", [legacy])
            save_json(private / "환율.json", FX)
            rows = {s: [{"date": "2026-11-06", "close": 100}] for s in ("QQQM", "VTI")}
            with patch.object(modeling, "PRIVATE", private), patch.object(modeling, "STOP", private / "중지요청"), patch.object(paper, "FX", private / "환율.json"), \
                 patch.object(paper, "ACCOUNT", private / "모의계좌.json"), \
                 patch.object(modeling, "predict_values", return_value=[103, 104]), \
                 patch.object(modeling, "load_config", return_value={"autotrading": {}}), \
                 patch.object(modeling, "update_state"), patch.object(modeling, "reconcile_forward"), patch.object(paper, "reconcile_paper"):
                modeling.publish_forecasts(None, rows, 32, "검사용 실제 함수 경로")
                modeling.publish_forecasts(None, rows, 32, "검사용 실제 함수 경로")
            ledger = read_json(private / "전진예측원장.json")
            self.assertEqual(ledger[0], legacy)
            self.assertEqual(len(ledger), 3)
            self.assertEqual(sum(row.get("universe_hash") == UNIVERSE_HASH for row in ledger), 2)
            self.assertEqual(len(read_json(private / "모의신호.json")), 1)

    def test_stop_during_prediction_never_publishes_or_reconciles(self):
        from src import modeling, paper
        with tempfile.TemporaryDirectory() as folder:
            private = Path(folder)
            stop = private / "중지요청"
            def predict(*args):
                stop.write_text("예측 계산 중 사용자 중지", encoding="utf-8")
                return [103]
            with patch.object(modeling, "PRIVATE", private), patch.object(modeling, "STOP", stop), \
                 patch.object(modeling, "predict_values", side_effect=predict), \
                 patch.object(modeling, "update_state") as updating, \
                 patch.object(paper, "reconcile_paper") as reconcile:
                with self.assertRaises(InterruptedError):
                    modeling.publish_forecasts(None, {"VTI": [{"date": "2026-11-06", "close": 100}]}, 32, "격리")
            self.assertFalse((private / "모의신호.json").exists())
            self.assertFalse((private / "전진예측원장.json").exists())
            updating.assert_not_called()
            reconcile.assert_not_called()

    def test_forward_metrics_do_not_mix_legacy_and_new_universe(self):
        from src import modeling
        from src.core import save_json
        from src.universe import UNIVERSE_HASH
        with tempfile.TemporaryDirectory() as folder:
            private = Path(folder)
            base = {"as_of": "2026-11-06", "actual": 102, "last_adjusted_price": 100, "adjusted_price": 103}
            save_json(private / "전진예측원장.json", [{**base, "symbol": "QQQM"},
                {**base, "symbol": "VTI", "universe_hash": UNIVERSE_HASH}])
            with patch.object(modeling, "PRIVATE", private), patch.object(modeling, "update_state"):
                result = modeling.reconcile_forward({})
            self.assertIsInstance(result, dict)
            self.assertEqual(result["n"], 1)
            self.assertEqual(result["previous_scope"]["n"], 1)
            self.assertEqual(result["coverage_symbols"], ["VTI"])


if __name__ == "__main__":
    unittest.main()
