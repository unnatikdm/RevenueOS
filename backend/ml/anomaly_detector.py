from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("revenueos.ml.anomaly_detector")


class TransactionAnomalyDetector:
    """
    Unsupervised Multi-Dimensional Leak & Discount Anomaly Detector.
    Leverages scikit-learn IsolationForest to identify anomalous multi-variable
    transaction vectors (price, quantity, gross amount, discount velocity, and temporal features)
    without requiring labeled fraud or leak data.
    """

    FEATURE_NAMES = [
        "unit_price",
        "quantity",
        "gross_amount",
        "discount_ratio",
        "price_dev_sku",
        "qty_dev_sku",
        "hour_of_day",
        "day_of_week",
    ]

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False
        self._sku_medians: Dict[str, Dict[str, float]] = {}

    def _extract_feature_vector(
        self, record: Dict[str, Any], sku_medians: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[float]:
        medians = sku_medians or self._sku_medians
        sku = str(record.get("sku", "UNKNOWN"))
        price = float(record.get("unit_price", record.get("price", 0.0)))
        qty = float(record.get("quantity", record.get("qty", 1.0)))
        gross = float(record.get("gross_amount", price * qty))
        discount = float(record.get("discount", record.get("total_discount", 0.0)))
        discount_ratio = (discount / gross) if gross > 0 else 0.0

        sku_stats = medians.get(sku, {"median_price": price, "median_qty": qty})
        price_dev = abs(price - sku_stats.get("median_price", price))
        qty_dev = abs(qty - sku_stats.get("median_qty", qty))

        hour = float(record.get("hour", record.get("hour_of_day", 12.0)))
        day_of_week = float(record.get("day_of_week", 2.0))

        return [
            max(0.0, price),
            max(0.0, qty),
            max(0.0, gross),
            min(1.0, max(0.0, discount_ratio)),
            price_dev,
            qty_dev,
            hour % 24.0,
            day_of_week % 7.0,
        ]

    def fit(self, records: Sequence[Dict[str, Any]]) -> TransactionAnomalyDetector:
        """Fits the Isolation Forest on a batch of raw transaction records."""
        if not records:
            logger.warning("Empty records provided to TransactionAnomalyDetector.fit")
            return self

        # Calculate SKU baseline medians
        sku_groups: Dict[str, Dict[str, List[float]]] = {}
        for r in records:
            sku = str(r.get("sku", "UNKNOWN"))
            if sku not in sku_groups:
                sku_groups[sku] = {"prices": [], "qtys": []}
            sku_groups[sku]["prices"].append(float(r.get("unit_price", r.get("price", 0.0))))
            sku_groups[sku]["qtys"].append(float(r.get("quantity", r.get("qty", 1.0))))

        self._sku_medians = {
            sku: {
                "median_price": float(np.median(data["prices"])) if data["prices"] else 0.0,
                "median_qty": float(np.median(data["qtys"])) if data["qtys"] else 1.0,
            }
            for sku, data in sku_groups.items()
        }

        X = np.array([self._extract_feature_vector(r, self._sku_medians) for r in records], dtype=np.float64)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(X)
        self.is_fitted = True
        logger.info("TransactionAnomalyDetector successfully fitted on %d records.", len(records))
        return self

    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a transaction vector for leakage anomalies.
        Returns anomaly diagnosis, calibrated score, and financial exposure.
        """
        price = float(record.get("unit_price", record.get("price", 0.0)))
        qty = float(record.get("quantity", record.get("qty", 1.0)))
        gross = float(record.get("gross_amount", price * qty))
        discount = float(record.get("discount", record.get("total_discount", 0.0)))
        sku = str(record.get("sku", "UNKNOWN"))

        if not self.is_fitted or self.model is None:
            # Fallback deterministic cold-start heuristic
            discount_ratio = (discount / gross) if gross > 0 else 0.0
            is_anomaly = discount_ratio > 0.40 or (price > 0 and qty > 50)
            score = 0.85 if is_anomaly else 0.15
            sub_type = "DISCOUNT_ABUSE" if discount_ratio > 0.40 else ("BULK_QUANTITY_SPIKE" if is_anomaly else "NORMAL")
            exposure = round(gross * discount_ratio if discount_ratio > 0.40 else (gross * 0.20 if is_anomaly else 0.0), 2)
            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": score,
                "anomaly_type": sub_type,
                "exposure_estimate": exposure,
                "model_status": "FALLBACK_HEURISTIC",
                "sku": sku,
            }

        feat_vec = np.array([self._extract_feature_vector(record)], dtype=np.float64)
        decision_val = float(self.model.decision_function(feat_vec)[0])  # lower = more anomalous
        raw_pred = -1 if decision_val < 0.0 else 1
        normalized_score = float(np.clip(1.0 / (1.0 + np.exp(decision_val * 8.0)), 0.0, 1.0))
        discount_ratio = (discount / gross) if gross > 0 else 0.0

        is_anomaly = bool(raw_pred == -1 or normalized_score >= 0.58 or discount_ratio > 0.35)
        sku_stats = self._sku_medians.get(sku, {"median_price": price, "median_qty": qty})
        median_price = sku_stats.get("median_price", price)
        median_qty = sku_stats.get("median_qty", qty)

        if is_anomaly:
            if discount_ratio > 0.35:
                anomaly_type = "DISCOUNT_ABUSE"
                exposure = round(discount, 2)
            elif price < (median_price * 0.60) and median_price > 0:
                anomaly_type = "UNDERPRICING_MARGIN_EROSION"
                exposure = round((median_price - price) * qty, 2)
            elif qty > (median_qty * 4.0):
                anomaly_type = "ABNORMAL_BULK_OUTFLOW"
                exposure = round(gross * 0.15, 2)
            else:
                anomaly_type = "SUSPICIOUS_ORDER_BURST"
                exposure = round(gross * 0.10, 2)
        else:
            anomaly_type = "NORMAL"
            exposure = 0.0

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(normalized_score, 4),
            "raw_decision_score": round(float(decision_val), 4),
            "anomaly_type": anomaly_type,
            "exposure_estimate": round(max(0.0, exposure), 2),
            "model_status": "ISOLATION_FOREST_VERIFIED",
            "sku": sku,
        }
