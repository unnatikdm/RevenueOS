from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence
import joblib

from backend.ml.anomaly_detector import TransactionAnomalyDetector
from backend.ml.customer_clustering import CustomerRFMClusterer
from backend.ml.demand_forecaster import SKUDemandForecaster
from backend.ml.return_propensity import ReturnPropensityClassifier

logger = logging.getLogger("revenueos.ml.pipeline")

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "external", "online_retail_raw.csv")
SAMPLE_ORDERS_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "raw_orders_export.csv")


class RevenueOSMLSuite:
    """
    Unified Machine Learning Intelligence Suite for RevenueOS.
    Manages the lifecycle, persistent loading, real-time inference, and fallback
    mechanisms for the 4 production ML models:
    1. Isolation Forest Transaction Anomaly Detector
    2. XGBoost Time-Series Demand Forecaster
    3. Random Forest Return Propensity Classifier
    4. K-Means Customer RFM Behavioral Segmentation Engine
    """

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.anomaly_detector = TransactionAnomalyDetector()
        self.demand_forecaster = SKUDemandForecaster()
        self.return_classifier = ReturnPropensityClassifier()
        self.rfm_clusterer = CustomerRFMClusterer()
        self.is_loaded: bool = False
        self._load_or_initialize()

    def _model_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)

    def _load_or_initialize(self) -> None:
        """Loads serialized models if present on disk, otherwise keeps fast fallbacks ready."""
        os.makedirs(self.models_dir, exist_ok=True)
        anomaly_file = self._model_path("anomaly_detector.joblib")
        forecaster_file = self._model_path("demand_forecaster.joblib")
        return_file = self._model_path("return_propensity.joblib")
        rfm_file = self._model_path("customer_rfm.joblib")

        try:
            if os.path.exists(anomaly_file):
                self.anomaly_detector = joblib.load(anomaly_file)
            if os.path.exists(forecaster_file):
                self.demand_forecaster = joblib.load(forecaster_file)
            if os.path.exists(return_file):
                self.return_classifier = joblib.load(return_file)
            if os.path.exists(rfm_file):
                self.rfm_clusterer = joblib.load(rfm_file)
            self.is_loaded = any([
                self.anomaly_detector.is_fitted,
                self.demand_forecaster.is_fitted,
                self.return_classifier.is_fitted,
                self.rfm_clusterer.is_fitted,
            ])
        except Exception as e:
            logger.warning("Error loading serialized ML models (%s); operating with resilient fallbacks.", e)

    def save_models(self) -> None:
        """Serializes all fitted models to disk."""
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.anomaly_detector, self._model_path("anomaly_detector.joblib"))
        joblib.dump(self.demand_forecaster, self._model_path("demand_forecaster.joblib"))
        joblib.dump(self.return_classifier, self._model_path("return_propensity.joblib"))
        joblib.dump(self.rfm_clusterer, self._model_path("customer_rfm.joblib"))
        logger.info("Saved all ML models to %s", self.models_dir)

    def evaluate_transaction_anomaly(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Inference wrapper for Isolation Forest anomaly detection."""
        return self.anomaly_detector.predict(record)

    def forecast_sku_demand(
        self,
        historical_sales: Sequence[float],
        current_inventory: int = 100,
        asp: float = 1999.0,
        horizon_days: int = 7,
    ) -> Dict[str, Any]:
        """Inference wrapper for XGBoost demand forecasting."""
        return self.demand_forecaster.forecast_demand(
            historical_sales=historical_sales,
            current_inventory=current_inventory,
            asp=asp,
            horizon_days=horizon_days,
        )

    def predict_return_risk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Inference wrapper for Random Forest return propensity."""
        return self.return_classifier.predict_propensity(item)

    def segment_customer_rfm(self, customer_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Inference wrapper for K-Means RFM clustering & retention churn risk."""
        return self.rfm_clusterer.predict_segment(customer_stats)

    def get_health_status(self) -> Dict[str, Any]:
        """Returns readiness, model versions, and fitted flags for all ML engines."""
        return {
            "status": "HEALTHY",
            "models_loaded": self.is_loaded,
            "models": {
                "anomaly_detector": {
                    "algorithm": "IsolationForest",
                    "is_fitted": self.anomaly_detector.is_fitted,
                    "contamination": self.anomaly_detector.contamination,
                },
                "demand_forecaster": {
                    "algorithm": "XGBRegressor",
                    "is_fitted": self.demand_forecaster.is_fitted,
                    "r2_score": getattr(self.demand_forecaster, "r2_score", 0.0),
                },
                "return_classifier": {
                    "algorithm": "RandomForestClassifier",
                    "is_fitted": self.return_classifier.is_fitted,
                },
                "rfm_clusterer": {
                    "algorithm": "KMeans_RFM",
                    "is_fitted": self.rfm_clusterer.is_fitted,
                    "clusters": self.rfm_clusterer.n_clusters,
                },
            },
        }


# Global shared instance
_GLOBAL_ML_SUITE: Optional[RevenueOSMLSuite] = None


def get_ml_suite() -> RevenueOSMLSuite:
    global _GLOBAL_ML_SUITE
    if _GLOBAL_ML_SUITE is None:
        _GLOBAL_ML_SUITE = RevenueOSMLSuite()
    return _GLOBAL_ML_SUITE


def train_models_from_dataset(
    raw_csv_path: Optional[str] = None,
    max_rows: int = 40000,
    save_to_disk: bool = True,
) -> RevenueOSMLSuite:
    """
    Reads the authentic commercial retail dataset (or sample export)
    and trains all 4 production ML models.
    """
    csv_file = raw_csv_path or (EXTERNAL_DATA_PATH if os.path.exists(EXTERNAL_DATA_PATH) else SAMPLE_ORDERS_PATH)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Training dataset not found at: {csv_file}")

    print(f"Loading training data from: {csv_file} (max {max_rows:,} rows)...")

    transactions: List[Dict[str, Any]] = []
    daily_sales_map: Dict[str, Dict[str, float]] = {}
    return_samples: List[Dict[str, Any]] = []
    customers: Dict[str, Dict[str, Any]] = {}

    ref_date = datetime(2011, 12, 10, tzinfo=timezone.utc)

    with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break

            invoice = row.get("InvoiceNo", row.get("order_id", "")).strip()
            stock = row.get("StockCode", row.get("sku", "")).strip()
            qty_str = row.get("Quantity", row.get("quantity", "1")).strip()
            price_str = row.get("UnitPrice", row.get("unit_price", "0.0")).strip()
            cust_id = row.get("CustomerID", row.get("customer_id", "")).strip() or "GUEST"
            date_str = row.get("InvoiceDate", row.get("created_at", "")).strip()

            try:
                qty = int(float(qty_str))
                price = float(price_str) * 105.0  # Convert to INR currency
            except (ValueError, TypeError):
                continue

            if price <= 0:
                continue

            is_return = invoice.startswith("C") or qty < 0
            abs_qty = abs(qty)
            gross = round(price * abs_qty, 2)

            dt = None
            for fmt in ("%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                    break
                except Exception:
                    pass

            dt = dt or ref_date
            day_key = dt.strftime("%Y-%m-%d")

            # 1. Transaction Anomaly records
            if not is_return:
                transactions.append({
                    "sku": stock,
                    "unit_price": price,
                    "quantity": abs_qty,
                    "gross_amount": gross,
                    "discount": 0.0,
                    "hour": dt.hour,
                    "day_of_week": dt.weekday(),
                })

                # 2. Daily SKU aggregation for demand forecasting
                if stock not in daily_sales_map:
                    daily_sales_map[stock] = {}
                daily_sales_map[stock][day_key] = daily_sales_map[stock].get(day_key, 0.0) + abs_qty

                # 4. Customer RFM metrics
                if cust_id != "GUEST":
                    if cust_id not in customers:
                        customers[cust_id] = {
                            "customer_id": cust_id,
                            "last_date": dt,
                            "frequency": 0,
                            "monetary": 0.0,
                        }
                    customers[cust_id]["frequency"] += 1
                    customers[cust_id]["monetary"] += gross
                    if dt > customers[cust_id]["last_date"]:
                        customers[cust_id]["last_date"] = dt

            # 3. Return samples
            return_samples.append({
                "sku": stock,
                "unit_price": price,
                "quantity": abs_qty,
                "gross_amount": gross,
                "is_returned": is_return,
                "order_hour": dt.hour,
            })

    print(f"Extracted {len(transactions):,} transactions, {len(return_samples):,} return samples, {len(customers):,} customer profiles.")

    # Instantiate Suite
    suite = RevenueOSMLSuite()

    # Train Model 1: Anomaly Detector
    print("Training IsolationForest Multi-Dimensional Anomaly Detector...")
    suite.anomaly_detector.fit(transactions[:15000])

    # Train Model 2: Demand Forecaster on Top Volume SKU
    top_sku = max(daily_sales_map.keys(), key=lambda s: sum(daily_sales_map[s].values()))
    sorted_days = sorted(daily_sales_map[top_sku].keys())
    sales_series = [daily_sales_map[top_sku][d] for d in sorted_days]
    print(f"Training XGBoost Demand Forecaster on primary SKU {top_sku} ({len(sales_series)} daily time-steps)...")
    suite.demand_forecaster.fit(sales_series)

    # Train Model 3: Return Propensity Classifier
    print("Training RandomForest Return Propensity Classifier...")
    suite.return_classifier.fit(return_samples[:15000])

    # Train Model 4: Customer RFM Clustering
    customer_list = []
    for c_id, c_data in customers.items():
        recency = max(0, (ref_date - c_data["last_date"]).days)
        customer_list.append({
            "customer_id": c_id,
            "recency_days": recency,
            "frequency_orders": c_data["frequency"],
            "monetary_spend": round(c_data["monetary"], 2),
        })

    print(f"Training KMeans RFM Clustering Engine on {len(customer_list):,} customers...")
    suite.rfm_clusterer.fit(customer_list[:8000])

    suite.is_loaded = True

    if save_to_disk:
        suite.save_models()

    print("All 4 RevenueOS Machine Learning Models trained successfully!")
    return suite


if __name__ == "__main__":
    train_models_from_dataset()
