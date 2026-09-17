from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("revenueos.ml.return_propensity")


class ReturnPropensityClassifier:
    """
    Supervised Return Propensity Classifier.
    Employs a Random Forest Classifier trained on order items, SKU return defect histories,
    price tiers, and transaction parameters to predict the probability P(Return)
    of an order/SKU and quantify downstream refund liabilities.
    """

    FEATURE_NAMES = [
        "unit_price",
        "quantity",
        "gross_amount",
        "sku_return_rate",
        "customer_return_rate",
        "price_tier",
        "order_hour",
    ]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.is_fitted: bool = False
        self._sku_return_baselines: Dict[str, float] = {}

    def _get_price_tier(self, price: float) -> float:
        if price < 750.0:
            return 0.0  # Budget
        elif price < 2500.0:
            return 1.0  # Core
        else:
            return 2.0  # Premium / High-Value

    def _extract_feature_vector(self, item: Dict[str, Any]) -> List[float]:
        price = float(item.get("unit_price", item.get("price", 0.0)))
        qty = float(item.get("quantity", item.get("qty", 1.0)))
        gross = float(item.get("gross_amount", price * qty))
        sku = str(item.get("sku", "UNKNOWN"))

        sku_rate = float(item.get("sku_return_rate", self._sku_return_baselines.get(sku, 0.06)))
        cust_rate = float(item.get("customer_return_rate", 0.05))
        price_tier = self._get_price_tier(price)
        hour = float(item.get("order_hour", item.get("hour", 14.0))) % 24.0

        return [
            price,
            qty,
            gross,
            min(1.0, max(0.0, sku_rate)),
            min(1.0, max(0.0, cust_rate)),
            price_tier,
            hour,
        ]

    def fit(self, training_records: Sequence[Dict[str, Any]]) -> ReturnPropensityClassifier:
        """
        Trains the Random Forest model on historical order items labeled with is_returned (0 or 1).
        """
        if not training_records:
            logger.warning("Empty records provided to ReturnPropensityClassifier.fit")
            return self

        # Precompute SKU return baselines
        sku_counts: Dict[str, int] = {}
        sku_returns: Dict[str, int] = {}
        for r in training_records:
            sku = str(r.get("sku", "UNKNOWN"))
            is_ret = 1 if r.get("is_returned", False) else 0
            sku_counts[sku] = sku_counts.get(sku, 0) + 1
            sku_returns[sku] = sku_returns.get(sku, 0) + is_ret

        self._sku_return_baselines = {
            sku: (sku_returns[sku] / sku_counts[sku])
            for sku in sku_counts
            if sku_counts[sku] >= 5
        }

        X_list: List[List[float]] = []
        y_list: List[int] = []

        for r in training_records:
            X_list.append(self._extract_feature_vector(r))
            y_list.append(1 if r.get("is_returned", False) else 0)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)

        # Check for class diversity
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.info("Single class in return training set; initializing calibrated fallback.")
            self.is_fitted = False
            return self

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("ReturnPropensityClassifier fitted on %d samples.", len(training_records))
        return self

    def predict_propensity(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes forward-looking return probability for an item or SKU cohort.
        """
        price = float(item.get("unit_price", item.get("price", 0.0)))
        qty = float(item.get("quantity", item.get("qty", 1.0)))
        gross = float(item.get("gross_amount", price * qty))
        sku = str(item.get("sku", "UNKNOWN"))
        sku_rate = self._sku_return_baselines.get(sku, float(item.get("sku_return_rate", 0.06)))

        if not self.is_fitted or self.model is None:
            # Deterministic empirical Bayesian fallback
            prob = float(np.clip(sku_rate * (1.2 if price > 2000 else 1.0), 0.02, 0.85))
        else:
            feat_vec = np.array([self._extract_feature_vector(item)], dtype=np.float64)
            prob = float(self.model.predict_proba(feat_vec)[0][1])

        # Assign risk tier
        if prob >= 0.50:
            risk_tier = "CRITICAL"
        elif prob >= 0.30:
            risk_tier = "HIGH"
        elif prob >= 0.15:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        # Diagnose contributing risk factors
        factors: List[str] = []
        if sku_rate > 0.12:
            factors.append(f"Historical SKU defect rate ({round(sku_rate * 100, 1)}%) exceeds category baseline")
        if price > 2500.0:
            factors.append("High-ticket price tier correlates with increased buyer remorse/scrutiny")
        if qty >= 3:
            factors.append("Multi-quantity order indicates bracket purchasing (ordering multiple sizes)")
        if not factors:
            factors.append("Standard baseline return propensity")

        projected_refund = round(gross * prob, 2)

        return {
            "return_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "projected_refund_exposure": projected_refund,
            "contributing_factors": factors,
            "sku_baseline_return_rate": round(sku_rate, 4),
            "model_status": "RANDOM_FOREST_PROBABILITY" if self.is_fitted else "EMPIRICAL_BAYESIAN_FALLBACK",
            "sku": sku,
        }
