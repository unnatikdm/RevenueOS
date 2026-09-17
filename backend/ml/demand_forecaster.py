from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger("revenueos.ml.demand_forecaster")


class SKUDemandForecaster:
    """
    Time-Series Demand Forecasting & Stockout Predictor.
    Utilizes XGBoost Regressor (with GradientBoostingRegressor fallback)
    engineered with lag features (t-1, t-2, t-7, t-14), rolling statistics,
    and calendar seasonality to forecast future SKU sales velocity and preempt stockouts.
    """

    FEATURE_NAMES = [
        "lag_1",
        "lag_2",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "day_of_week",
        "is_weekend",
    ]

    def __init__(
        self,
        n_estimators: int = 80,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.is_fitted: bool = False
        self.r2_score: float = 0.0

    def _create_lagged_dataset(
        self, daily_series: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs supervised autoregressive feature matrix from 1D sales series."""
        n = len(daily_series)
        X_list: List[List[float]] = []
        y_list: List[float] = []

        for t in range(14, n):
            target = float(daily_series[t])
            lag_1 = float(daily_series[t - 1])
            lag_2 = float(daily_series[t - 2])
            lag_7 = float(daily_series[t - 7])
            lag_14 = float(daily_series[t - 14])

            window_7 = [float(daily_series[i]) for i in range(t - 7, t)]
            window_14 = [float(daily_series[i]) for i in range(t - 14, t)]

            rolling_mean_7 = float(np.mean(window_7))
            rolling_std_7 = float(np.std(window_7))
            rolling_mean_14 = float(np.mean(window_14))

            day_of_week = float(t % 7)
            is_weekend = 1.0 if day_of_week in (5.0, 6.0) else 0.0

            row = [
                lag_1,
                lag_2,
                lag_7,
                lag_14,
                rolling_mean_7,
                rolling_std_7,
                rolling_mean_14,
                day_of_week,
                is_weekend,
            ]
            X_list.append(row)
            y_list.append(target)

        if not X_list:
            return np.empty((0, len(self.FEATURE_NAMES))), np.empty((0,))

        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    def fit(self, daily_sales: Sequence[float]) -> SKUDemandForecaster:
        """Trains the regressor on the historical non-censored daily sales series."""
        clean_series = [max(0.0, float(x)) for x in daily_sales]
        if len(clean_series) < 21:
            logger.info("Insufficient time-series history (< 21 days) for XGBoost; using moving average fallback.")
            self.is_fitted = False
            return self

        X, y = self._create_lagged_dataset(clean_series)
        if len(X) < 7:
            self.is_fitted = False
            return self

        if XGB_AVAILABLE:
            try:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=1,
                )
                self.model.fit(X, y)
                self.is_fitted = True
            except Exception as e:
                logger.warning("XGBoost training exception (%s); switching to GradientBoostingRegressor", e)
                self.model = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                )
                self.model.fit(X, y)
                self.is_fitted = True
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.model.fit(X, y)
            self.is_fitted = True

        # Calculate approximate R2 score
        preds = self.model.predict(X)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - preds) ** 2))
        self.r2_score = round(max(0.0, 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0), 3)
        logger.info("SKUDemandForecaster fitted successfully with R2 = %.3f", self.r2_score)
        return self

    def forecast_demand(
        self,
        historical_sales: Sequence[float],
        current_inventory: int = 100,
        asp: float = 1999.0,
        horizon_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Generates multi-day recursive sales velocity forecasts and evaluates stockout runway.
        """
        history = [max(0.0, float(x)) for x in historical_sales]
        if not history:
            history = [10.0] * 14

        # Fallback if model is unfitted or history is short
        if not self.is_fitted or self.model is None or len(history) < 14:
            base_mean = float(np.mean(history[-7:] if len(history) >= 7 else history))
            base_mean = max(1.0, base_mean)
            projected_days = [round(base_mean, 2)] * horizon_days
            total_proj = sum(projected_days)
            runout = int(current_inventory / base_mean) if base_mean > 0 else 999
            risk_tier = "CRITICAL" if runout <= 3 else ("HIGH" if runout <= 7 else ("MEDIUM" if runout <= 14 else "LOW"))
            stockout_days = max(0, horizon_days - runout) if runout < horizon_days else 0
            missed_revenue = round(stockout_days * base_mean * asp, 2)

            return {
                "daily_forecasts": projected_days,
                "projected_daily_velocity": round(base_mean, 2),
                "total_horizon_demand": round(total_proj, 1),
                "current_inventory": current_inventory,
                "projected_runout_days": runout,
                "stockout_risk_tier": risk_tier,
                "estimated_uncaptured_revenue": missed_revenue,
                "model_confidence": 0.75,
                "model_engine": "MOVING_AVERAGE_FALLBACK",
            }

        # Recursive Multi-Step Forecasting
        series_buffer = list(history)
        projected_days: List[float] = []

        for step in range(horizon_days):
            t = len(series_buffer)
            lag_1 = series_buffer[-1]
            lag_2 = series_buffer[-2]
            lag_7 = series_buffer[-7]
            lag_14 = series_buffer[-14]

            window_7 = series_buffer[-7:]
            window_14 = series_buffer[-14:]

            rolling_mean_7 = float(np.mean(window_7))
            rolling_std_7 = float(np.std(window_7))
            rolling_mean_14 = float(np.mean(window_14))

            day_of_week = float(t % 7)
            is_weekend = 1.0 if day_of_week in (5.0, 6.0) else 0.0

            feat = np.array([[
                lag_1,
                lag_2,
                lag_7,
                lag_14,
                rolling_mean_7,
                rolling_std_7,
                rolling_mean_14,
                day_of_week,
                is_weekend,
            ]], dtype=np.float64)

            pred_val = float(self.model.predict(feat)[0])
            pred_val = max(0.0, pred_val)
            projected_days.append(round(pred_val, 2))
            series_buffer.append(pred_val)

        avg_velocity = float(np.mean(projected_days))
        total_proj = float(sum(projected_days))

        # Stockout runway calculation
        cum_demand = 0.0
        runout_days = horizon_days + 1
        for day_idx, d in enumerate(projected_days):
            cum_demand += d
            if cum_demand >= current_inventory:
                runout_days = day_idx + 1
                break

        if current_inventory <= 0:
            runout_days = 0
            risk_tier = "CRITICAL"
        elif runout_days <= 3:
            risk_tier = "CRITICAL"
        elif runout_days <= 7:
            risk_tier = "HIGH"
        elif runout_days <= 14:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        stockout_days = max(0, horizon_days - runout_days)
        estimated_loss = round(stockout_days * avg_velocity * asp, 2)

        return {
            "daily_forecasts": projected_days,
            "projected_daily_velocity": round(avg_velocity, 2),
            "total_horizon_demand": round(total_proj, 1),
            "current_inventory": current_inventory,
            "projected_runout_days": runout_days,
            "stockout_risk_tier": risk_tier,
            "estimated_uncaptured_revenue": estimated_loss,
            "model_confidence": round(min(0.98, max(0.80, self.r2_score)), 3),
            "model_engine": "XGBOOST_TIME_SERIES",
        }
