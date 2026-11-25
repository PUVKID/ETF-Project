import importlib.util
import unittest


class PaperAccountTests(unittest.TestCase):
    def paper(self):
        self.assertIsNotNone(importlib.util.find_spec("src.paper"), "전진 모의계좌 구현 전")
        from src import paper
        return paper

    def test_missing_fx_keeps_the_original_won_without_fake_conversion(self):
        p = self.paper()
        account = p.create_account(None, "2024-01-02")
        self.assertEqual(account["cash_krw"], 1000000)
        self.assertIsNone(account["cash_usd"])
        self.assertEqual(account["positions"], [])

    def test_conversion_charge_and_daily_currency_move_are_included(self):
        p = self.paper()
        fx = {"rate": 1000, "as_of": "2024-01-02", "source": "검사용 고정 환율"}
        account = p.create_account(fx, "2024-01-02", spread_bps=10)
        self.assertEqual(account["cash_krw"], 0)
        self.assertAlmostEqual(account["cash_usd"], 999.0009990009991)
        self.assertAlmostEqual(account["exchange_cost_krw"], 999.0009990008548)
        p.mark_account(account, {}, {**fx, "rate": 1100})
        self.assertAlmostEqual(account["current_balance_krw"], 1098901.098901099)
        self.assertAlmostEqual(account["pnl_krw"], 98901.098901099)

    def test_signal_created_after_open_cannot_fill_at_that_past_open(self):
        p = self.paper()
        fx = {"rate": 1000, "as_of": "2024-01-02", "source": "검사용 고정 환율"}
        account = p.create_account(fx, "2024-01-02", spread_bps=0)
        rows = [{"date": day, "open": price, "close": price, "raw_open": price, "raw_close": price}
                for day, price in (("2024-01-02", 10), ("2024-01-03", 10), ("2024-01-04", 20))]
        signal = {"as_of": "2024-01-02", "created_at": "2024-01-03T16:00:00+00:00", "weights": {"A": 1}}
        p.advance_account(account, {"A": rows[:2]}, fx, [signal])
        self.assertEqual(account["trades"], [])
        p.advance_account(account, {"A": rows}, fx, [signal])
        self.assertEqual(account["trades"][0]["execution_date"], "2024-01-04")
        self.assertEqual(account["positions"][0]["quantity"], 49.9505)
        self.assertGreaterEqual(account["cash_usd"], 0)
        self.assertLessEqual(account["current_balance_krw"], 1000000)
        before = list(account["trades"])
        p.advance_account(account, {"A": rows}, fx, [signal])
        self.assertEqual(account["trades"], before)

    def test_short_and_leveraged_weights_are_rejected(self):
        p = self.paper()
        for weights in ({"A": -1}, {"A": 2}, {"A": float("nan")}):
            with self.assertRaises(ValueError):
                p.validate_weights(weights)

    def test_opening_uses_new_york_daylight_saving_time(self):
        p = self.paper()
        self.assertEqual(p.session_open("2024-01-03").isoformat(), "2024-01-03T14:30:00+00:00")
        self.assertEqual(p.session_open("2024-07-03").isoformat(), "2024-07-03T13:30:00+00:00")

    def test_future_fx_is_not_applied_to_earlier_accounting_days(self):
        p = self.paper()
        fx = {"rate": 1500, "as_of": "2024-01-04", "source": "검사용 고정 환율",
              "rates": [{"date": "2024-01-02", "rate": 1000}, {"date": "2024-01-04", "rate": 1500}]}
        self.assertEqual(p.fx_for_day(fx, "2024-01-03")["rate"], 1000)
        self.assertIsNone(p.fx_for_day(fx, "2024-01-01"))


if __name__ == "__main__":
    unittest.main()
