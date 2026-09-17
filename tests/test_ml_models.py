import os
import shutil
import tempfile
import unittest

from backend.ml.anomaly_detector import TransactionAnomalyDetector
from backend.ml.customer_clustering import CustomerRFMClusterer
from backend.ml.demand_forecaster import SKUDemandForecaster
from backend.ml.return_propensity import ReturnPropensityClassifier
from backend.ml.pipeline import RevenueOSMLSuite, get_ml_suite
from backend.functions.leak_engine.stockout_detector import detect_stockout_leak
from backend.functions.leak_engine.return_detector import detect_return_spike_leak
from backend.functions.leak_engine.pricing_detector import detect_price_elasticity_leak
from backend.functions.leak_engine.retention_detector import detect_retention_decay_leak
from backend.functions.dashboard.api_handlers import lambda_handler


class TestMachineLearningSuite(unittest.TestCase):
    def setUp(self):
        self.sample_transactions = [
            {"sku": "SKU-TEST-1", "unit_price": 1000.0, "quantity": 2, "gross_amount": 2000.0, "discount": 0.0, "hour": 14, "day_of_week": 1},
            {"sku": "SKU-TEST-1", "unit_price": 950.0, "quantity": 1, "gross_amount": 950.0, "discount": 0.0, "hour": 15, "day_of_week": 1},
            {"sku": "SKU-TEST-1", "unit_price": 1050.0, "quantity": 3, "gross_amount": 3150.0, "discount": 50.0, "hour": 12, "day_of_week": 2},
            {"sku": "SKU-TEST-1", "unit_price": 1000.0, "quantity": 2, "gross_amount": 2000.0, "discount": 0.0, "hour": 16, "day_of_week": 3},
            {"sku": "SKU-TEST-1", "unit_price": 980.0, "quantity": 2, "gross_amount": 1960.0, "discount": 0.0, "hour": 11, "day_of_week": 4},
            {"sku": "SKU-TEST-2", "unit_price": 450.0, "quantity": 1, "gross_amount": 450.0, "discount": 0.0, "hour": 10, "day_of_week": 1},
            {"sku": "SKU-TEST-2", "unit_price": 450.0, "quantity": 2, "gross_amount": 900.0, "discount": 0.0, "hour": 13, "day_of_week": 2},
            {"sku": "SKU-TEST-2", "unit_price": 480.0, "quantity": 1, "gross_amount": 480.0, "discount": 0.0, "hour": 15, "day_of_week": 3},
            {"sku": "SKU-TEST-2", "unit_price": 450.0, "quantity": 1, "gross_amount": 450.0, "discount": 0.0, "hour": 17, "day_of_week": 4},
            {"sku": "SKU-TEST-2", "unit_price": 460.0, "quantity": 2, "gross_amount": 920.0, "discount": 0.0, "hour": 9, "day_of_week": 5},
        ] * 10  # 100 records

        self.sample_sales_series = [
            18.0, 22.0, 19.0, 25.0, 30.0, 32.0, 20.0,
            21.0, 23.0, 20.0, 27.0, 31.0, 33.0, 22.0,
            19.0, 24.0, 21.0, 26.0, 30.0, 35.0, 24.0,
            20.0, 22.0, 21.0, 28.0, 32.0, 34.0, 23.0,
        ]  # 28 days

    def test_anomaly_detector_training_and_inference(self):
        detector = TransactionAnomalyDetector(contamination=0.05, n_estimators=30, random_state=42)
        detector.fit(self.sample_transactions)
        self.assertTrue(detector.is_fitted)

        # Inlier evaluation
        normal_item = {
            "sku": "SKU-TEST-1",
            "unit_price": 1000.0,
            "quantity": 2,
            "gross_amount": 2000.0,
            "discount": 0.0,
        }
        res_normal = detector.predict(normal_item)
        self.assertIn("is_anomaly", res_normal)
        self.assertIn("anomaly_score", res_normal)
        self.assertGreaterEqual(res_normal["anomaly_score"], 0.0)
        self.assertLessEqual(res_normal["anomaly_score"], 1.0)

        # Severe discount abuse outlier
        outlier_item = {
            "sku": "SKU-TEST-1",
            "unit_price": 1000.0,
            "quantity": 2,
            "gross_amount": 2000.0,
            "discount": 1600.0,  # 80% discount
        }
        res_outlier = detector.predict(outlier_item)
        self.assertIn("anomaly_type", res_outlier)
        self.assertGreater(res_outlier["exposure_estimate"], 0.0)

    def test_anomaly_detector_fallback_behavior(self):
        detector = TransactionAnomalyDetector()
        self.assertFalse(detector.is_fitted)
        out = detector.predict({"sku": "SKU-FALLBACK", "unit_price": 500.0, "quantity": 100, "gross_amount": 50000.0})
        self.assertEqual(out["model_status"], "FALLBACK_HEURISTIC")
        self.assertTrue(out["is_anomaly"])

    def test_demand_forecaster_autoregressive(self):
        forecaster = SKUDemandForecaster(n_estimators=30, random_state=42)
        forecaster.fit(self.sample_sales_series)
        self.assertTrue(forecaster.is_fitted)

        forecast = forecaster.forecast_demand(
            historical_sales=self.sample_sales_series,
            current_inventory=50,
            asp=1499.0,
            horizon_days=7,
        )
        self.assertEqual(len(forecast["daily_forecasts"]), 7)
        self.assertGreater(forecast["projected_daily_velocity"], 0.0)
        self.assertIn(forecast["stockout_risk_tier"], ["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        self.assertIn("estimated_uncaptured_revenue", forecast)

    def test_demand_forecaster_fallback(self):
        forecaster = SKUDemandForecaster()
        short_series = [15.0, 18.0, 20.0]
        forecast = forecaster.forecast_demand(short_series, current_inventory=10, asp=1000.0)
        self.assertEqual(forecast["model_engine"], "MOVING_AVERAGE_FALLBACK")
        self.assertEqual(len(forecast["daily_forecasts"]), 7)

    def test_return_propensity_classifier(self):
        classifier = ReturnPropensityClassifier(n_estimators=30, random_state=42)
        records = [
            {"sku": "SKU-A", "unit_price": 800.0, "quantity": 1, "is_returned": 0},
            {"sku": "SKU-A", "unit_price": 800.0, "quantity": 1, "is_returned": 0},
            {"sku": "SKU-A", "unit_price": 800.0, "quantity": 1, "is_returned": 0},
            {"sku": "SKU-B", "unit_price": 3200.0, "quantity": 3, "is_returned": 1},
            {"sku": "SKU-B", "unit_price": 3200.0, "quantity": 2, "is_returned": 1},
        ] * 10
        classifier.fit(records)
        self.assertTrue(classifier.is_fitted)

        pred_low = classifier.predict_propensity({"sku": "SKU-A", "unit_price": 800.0, "quantity": 1})
        self.assertGreaterEqual(pred_low["return_probability"], 0.0)
        self.assertLessEqual(pred_low["return_probability"], 1.0)
        self.assertIn(pred_low["risk_tier"], ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

        pred_high = classifier.predict_propensity({"sku": "SKU-B", "unit_price": 3500.0, "quantity": 4})
        self.assertGreater(pred_high["projected_refund_exposure"], 0.0)
        self.assertTrue(len(pred_high["contributing_factors"]) > 0)

    def test_customer_rfm_clustering(self):
        clusterer = CustomerRFMClusterer(n_clusters=4, random_state=42)
        customers = [
            {"customer_id": f"CUST-VIP-{i}", "recency_days": 5, "frequency_orders": 12, "monetary_spend": 28000.0}
            for i in range(15)
        ] + [
            {"customer_id": f"CUST-CHURN-{i}", "recency_days": 110, "frequency_orders": 8, "monetary_spend": 19000.0}
            for i in range(15)
        ] + [
            {"customer_id": f"CUST-CORE-{i}", "recency_days": 25, "frequency_orders": 4, "monetary_spend": 6000.0}
            for i in range(15)
        ] + [
            {"customer_id": f"CUST-ONE-{i}", "recency_days": 140, "frequency_orders": 1, "monetary_spend": 800.0}
            for i in range(15)
        ]

        clusterer.fit(customers)
        self.assertTrue(clusterer.is_fitted)

        seg_res = clusterer.predict_segment({
            "customer_id": "CUST-TEST",
            "recency_days": 85.0,
            "frequency_orders": 6.0,
            "monetary_spend": 15000.0,
        })
        self.assertIn("segment_label", seg_res)
        self.assertIn("churn_risk_score", seg_res)
        self.assertIn("projected_ltv_loss", seg_res)
        self.assertIn("recommended_action", seg_res)

    def test_ml_suite_serialization_and_reloading(self):
        temp_dir = tempfile.mkdtemp()
        try:
            suite = RevenueOSMLSuite(models_dir=temp_dir)
            suite.anomaly_detector.fit(self.sample_transactions)
            suite.demand_forecaster.fit(self.sample_sales_series)
            suite.save_models()

            # Reload into new suite instance
            reloaded_suite = RevenueOSMLSuite(models_dir=temp_dir)
            self.assertTrue(reloaded_suite.anomaly_detector.is_fitted)
            self.assertTrue(reloaded_suite.demand_forecaster.is_fitted)
            status = reloaded_suite.get_health_status()
            self.assertEqual(status["status"], "HEALTHY")
            self.assertTrue(status["models"]["anomaly_detector"]["is_fitted"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_leak_engine_ml_augmentations(self):
        # 1. Stockout Detector with ML
        stockout_res = detect_stockout_leak(
            product_id="PROD-104",
            sku="SKU-104",
            daily_sales=[20, 25, 22, 24, 28, 30, 26, 25, 27, 29],
            stockout_days=7,
            asp=2499.0,
        )
        self.assertIsNotNone(stockout_res)
        self.assertIn("ml_forecast", stockout_res["evidence"])
        self.assertIn("projected_daily_velocity", stockout_res["evidence"]["ml_forecast"])

        # 2. Return Detector with ML
        return_res = detect_return_spike_leak(
            sku="SKU-208",
            baseline_returns=40,
            baseline_sales=1000,
            eval_returns=142,
            eval_sales=1000,
            asp=1899.0,
            top_reason="Size Too Small",
            reason_ratio=0.88,
        )
        self.assertIsNotNone(return_res)
        self.assertIn("ml_return_propensity", return_res["evidence"])
        self.assertIn("return_probability", return_res["evidence"]["ml_return_propensity"])

        # 3. Pricing Detector with ML
        price_res = detect_price_elasticity_leak(
            sku="SKU-305",
            p1=1299.0,
            q1=100,
            p2=1699.0,
            q2=58,
            unit_cost=750.0,
        )
        self.assertIsNotNone(price_res)
        self.assertIn("ml_anomaly_detection", price_res["evidence"])
        self.assertIn("is_anomaly", price_res["evidence"]["ml_anomaly_detection"])

        # 4. Retention Detector with ML
        ret_res = detect_retention_decay_leak(
            cohort_id="COHORT-2026-01",
            cohort_size=1000,
            actual_repeat_customers=190,
            baseline_repeat_rate=0.28,
            repeat_aov=2499.0,
        )
        self.assertIsNotNone(ret_res)
        self.assertIn("ml_rfm_segmentation", ret_res["evidence"])
        self.assertIn("churn_risk_score", ret_res["evidence"]["ml_rfm_segmentation"])

    def test_dashboard_ml_status_api_endpoint(self):
        event = {
            "requestContext": {
                "http": {"method": "GET"},
                "authorizer": {
                    "jwt": {"claims": {"custom:tenant_id": "tenant_apex_fashion"}}
                },
            },
            "rawPath": "/ml/status",
        }
        res = lambda_handler(event, None)
        self.assertEqual(res["statusCode"], 200)
        body = res["body"]
        self.assertIn("HEALTHY", body)
        self.assertIn("IsolationForest", body)


if __name__ == "__main__":
    unittest.main()
