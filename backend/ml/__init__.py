from __future__ import annotations

from backend.ml.anomaly_detector import TransactionAnomalyDetector
from backend.ml.demand_forecaster import SKUDemandForecaster
from backend.ml.return_propensity import ReturnPropensityClassifier
from backend.ml.customer_clustering import CustomerRFMClusterer
from backend.ml.pipeline import RevenueOSMLSuite, get_ml_suite, train_models_from_dataset

__all__ = [
    "TransactionAnomalyDetector",
    "SKUDemandForecaster",
    "ReturnPropensityClassifier",
    "CustomerRFMClusterer",
    "RevenueOSMLSuite",
    "get_ml_suite",
    "train_models_from_dataset",
]
