import csv
import os
import unittest
from decimal import Decimal


class TestSyntheticRetailData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.sample_dir = os.path.join(cls.base_dir, "data", "sample")
        cls.orders_path = os.path.join(cls.sample_dir, "raw_orders_export.csv")
        cls.inventory_path = os.path.join(cls.sample_dir, "raw_inventory_export.csv")
        cls.returns_path = os.path.join(cls.sample_dir, "raw_returns_export.csv")
        cls.funnel_path = os.path.join(cls.sample_dir, "checkout_funnel_metrics.csv")

    def test_files_exist(self):
        self.assertTrue(os.path.exists(self.orders_path))
        self.assertTrue(os.path.exists(self.inventory_path))
        self.assertTrue(os.path.exists(self.returns_path))
        self.assertTrue(os.path.exists(self.funnel_path))

    def test_orders_and_skus_count(self):
        order_ids = set()
        skus = set()
        total_gross = Decimal("0.00")

        with open(self.orders_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                order_ids.add(row["order_id"])
                skus.add(row["sku"])
                total_gross += Decimal(row["gross_amount"])

        self.assertEqual(len(order_ids), 18432, "Must contain exactly 18,432 unique orders")
        self.assertGreaterEqual(len(skus), 240, "Must contain at least 240 distinct catalog SKUs")
        # Ensure substantial GMV > ₹50 Lakhs
        self.assertGreater(total_gross, Decimal("5000000.00"))

    def test_anomaly_1_stockout_on_top_sku(self):
        # SKU-104 should have 0 inventory for days 40-46
        zero_inventory_dates = []
        with open(self.inventory_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sku"] == "SKU-104" and int(row["available_units"]) == 0:
                    zero_inventory_dates.append(row["snapshot_date"])

        self.assertEqual(len(zero_inventory_dates), 7, "SKU-104 must have exactly 7 consecutive zero-inventory days")

    def test_anomaly_2_return_spike(self):
        # SKU-208 should have return spike with top reason 'Size Too Small / Fit Issue'
        sku_208_returns = []
        with open(self.returns_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sku"] == "SKU-208":
                    sku_208_returns.append(row)

        self.assertGreater(len(sku_208_returns), 20)
        fit_reasons = [r for r in sku_208_returns if "Fit Issue" in r["return_reason"]]
        self.assertGreater(len(fit_reasons), len(sku_208_returns) * 0.5)

    def test_anomaly_3_checkout_funnel_friction(self):
        # Days 65-75 should flag friction anomaly with completion rate < 0.45
        flagged_days = 0
        with open(self.funnel_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["friction_anomaly_flag"] == "YES":
                    flagged_days += 1
                    self.assertLess(float(row["completion_rate"]), 0.45)
        self.assertEqual(flagged_days, 11)


if __name__ == "__main__":
    unittest.main()
