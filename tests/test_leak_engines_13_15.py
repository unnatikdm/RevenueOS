import unittest

from backend.functions.leak_engine.stockout_detector import detect_stockout_leak
from backend.functions.leak_engine.return_detector import detect_return_spike_leak
from backend.functions.leak_engine.funnel_detector import detect_funnel_friction_leak


class TestLeakEngines13To15(unittest.TestCase):
    def test_stockout_leak_detection(self):
        # 30-day sales history with 24 units/day average, 7-day stockout on SKU-104, ASP 2499.00
        daily_sales = [24] * 30
        res = detect_stockout_leak(
            product_id="PROD-104",
            sku="SKU-104",
            daily_sales=daily_sales,
            stockout_days=7,
            asp=2499.00,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["leak_type"], "stockout")
        self.assertEqual(res["entity_id"], "SKU-104")
        self.assertEqual(res["evidence"]["estimated_missed_units"], 168)  # 24 * 7
        self.assertEqual(res["impact_amount"], 419832.00)  # 168 * 2499
        self.assertGreaterEqual(res["confidence"], 0.95)

    def test_stockout_leak_no_leak_when_demand_too_low(self):
        daily_sales = [0] * 30
        res = detect_stockout_leak("PROD-99", "SKU-99", daily_sales, 7, 500.0)
        self.assertIsNone(res)

    def test_return_spike_leak_detection(self):
        # SKU-208: 60-day baseline returns: 40 on 1000 sales (4.0%)
        # 14-day evaluation: 142 returns on 1000 sales (14.2%)
        # ASP: 1899.00, Dominant reason: "Size Too Small / Fit Issue"
        res = detect_return_spike_leak(
            sku="SKU-208",
            baseline_returns=40,
            baseline_sales=1000,
            eval_returns=142,
            eval_sales=1000,
            asp=1899.00,
            top_reason="Size Too Small / Fit Issue",
            reason_ratio=0.88,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["leak_type"], "return_spike")
        self.assertEqual(res["entity_id"], "SKU-208")
        self.assertEqual(res["evidence"]["top_correlated_reason"], "Size Too Small / Fit Issue")
        self.assertAlmostEqual(res["evidence"]["current_return_rate"], 0.142, places=3)
        self.assertGreater(res["impact_amount"], 100000.00)

    def test_checkout_funnel_friction_detection(self):
        # Baseline: 5000 checkouts, 2450 orders (CR = 49%)
        # Eval: 1200 checkouts, 456 orders (CR = 38%, drop > 15%)
        # AOV = 2150.00
        res = detect_funnel_friction_leak(
            device_or_channel="mobile",
            baseline_checkouts=5000,
            baseline_orders=2450,
            eval_checkouts=1200,
            eval_orders=456,
            aov=2150.00,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["leak_type"], "checkout_friction")
        self.assertEqual(res["entity_id"], "mobile")
        self.assertGreater(res["evidence"]["lost_orders"], 100)
        self.assertGreater(res["impact_amount"], 250000.00)
        self.assertGreaterEqual(res["confidence"], 0.90)


if __name__ == "__main__":
    unittest.main()
