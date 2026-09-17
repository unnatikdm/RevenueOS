import math
import unittest

from backend.shared.calculations.aggregations import (
    compute_daily_demand_rate,
    compute_return_rate_metrics,
    compute_checkout_funnel_dropoff,
    compute_price_elasticity,
    compute_cohort_retention_metrics,
)


class TestAggregations(unittest.TestCase):
    def test_daily_demand_rate(self):
        # 10 days with 5 units/day = 5.0
        sales = [5] * 10
        ddr = compute_daily_demand_rate(sales, window_days=30)
        self.assertEqual(ddr, 5.0)

        # Empty sales list returns 0.0
        self.assertEqual(compute_daily_demand_rate([]), 0.0)

        # 30-day window slice check
        sales_long = [2] * 20 + [10] * 30
        ddr_30 = compute_daily_demand_rate(sales_long, window_days=30)
        self.assertEqual(ddr_30, 10.0)

    def test_return_rate_metrics(self):
        reasons = {
            "Size Too Small / Fit Issue": 85,
            "Customer Changed Mind": 10,
            "Defective Stitching": 5,
        }
        res = compute_return_rate_metrics(sales_units=1000, return_units=100, reason_counts=reasons)
        self.assertEqual(res["return_rate"], 0.10)
        self.assertEqual(res["top_reason"], "Size Too Small / Fit Issue")
        self.assertEqual(res["top_reason_ratio"], 0.85)

    def test_checkout_funnel_dropoff(self):
        # Normal baseline: 500 checkouts, 245 orders = 49% CR
        normal = compute_checkout_funnel_dropoff(checkouts=500, orders=245, baseline_cr=0.49)
        self.assertFalse(normal["is_anomaly"])
        self.assertEqual(normal["current_cr"], 0.49)

        # Anomaly drop: 500 checkouts, 180 orders = 36% CR (>15% drop below 49%)
        anomaly = compute_checkout_funnel_dropoff(checkouts=500, orders=180, baseline_cr=0.49)
        self.assertTrue(anomaly["is_anomaly"])
        self.assertAlmostEqual(anomaly["dropoff_cr"], 0.13, places=2)

    def test_price_elasticity(self):
        # Price increases from 1299 to 1699 (+30.79%)
        # Quantity decreases from 100 to 58 (-42.0%)
        # Elasticity = -0.42 / 0.3079 = -1.364 (|epsilon| > 1.2 elastic demand)
        p1, q1 = 1299.0, 100
        p2, q2 = 1699.0, 58
        eps = compute_price_elasticity(p1, q1, p2, q2)
        self.assertLess(eps, -1.2)

    def test_cohort_retention_metrics(self):
        # Cohort of 1000 new customers, historical baseline repeat = 28% (280 customers)
        # Actual repeat customers = 190 (shortfall = 90 customers)
        # Repeat AOV = 2499.00 -> Lost revenue = 90 * 2499 = 224,910.00
        res = compute_cohort_retention_metrics(
            cohort_size=1000,
            repeat_customers=190,
            baseline_repeat_rate=0.28,
            repeat_aov=2499.00,
        )
        self.assertEqual(res["missed_customers"], 90)
        self.assertEqual(res["lost_revenue"], 224910.00)
        self.assertEqual(res["current_retention_rate"], 0.19)


if __name__ == "__main__":
    unittest.main()
