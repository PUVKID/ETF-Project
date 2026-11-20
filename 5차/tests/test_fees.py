import importlib.util
import unittest


class TossPolicyTests(unittest.TestCase):
    def fees(self):
        self.assertIsNotNone(importlib.util.find_spec("src.fees"), "토스 공시 기준 비용 모듈 구현 전")
        from src import fees
        return fees

    def test_small_orders_are_exempt_from_commission_only(self):
        f = self.fees()
        self.assertEqual(f.trade_fees(10, "buy")["commission_usd"], 0)
        self.assertEqual(f.trade_fees(10, "sell")["sec_usd"], 0.01)
        self.assertEqual(f.trade_fees(1000, "buy")["commission_usd"], 1)
        self.assertEqual(f.trade_fees(123.456, "buy")["commission_usd"], 0.12)
        self.assertEqual(f.trade_fees(1000, "sell")["sec_usd"], 0.02)
        self.assertEqual(f.trade_fees(1214, "sell")["sec_usd"], 0.03)

    def test_exchange_uses_the_offer_rate_instead_of_subtracting_a_flat_fee(self):
        f = self.fees()
        result = f.convert_won(1000000, 1000, 50)
        self.assertAlmostEqual(result["received_usd"], 995.0248756218906)
        self.assertAlmostEqual(result["cost_krw"], 4975.124378109351)

    def test_unknown_korean_holiday_does_not_get_an_unverified_discount(self):
        f = self.fees()
        self.assertEqual(f.fx_spread_bps("2026-11-10T10:00:00+09:00", business_day=None), 50)
        self.assertEqual(f.fx_spread_bps("2026-11-10T10:00:00+09:00", business_day=True), 5)
        self.assertEqual(f.fx_spread_bps("2026-11-10T03:00:00+09:00", business_day=True), 50)

    def test_fractional_order_never_exceeds_cash(self):
        f = self.fees()
        quantity = f.affordable_quantity(100, 31, max_quantity=10)
        amount = quantity * 31
        self.assertLessEqual(amount + f.trade_fees(amount, "buy")["total_usd"], 100)
        self.assertEqual(round(quantity, 6), quantity)


if __name__ == "__main__":
    unittest.main()
