from decimal import Decimal
from typing import Any, Dict, Optional

from backend.shared.calculations.engines import detect_return_spike_leak as _detect_return_spike_leak


def _legacy_numbers(payload: Dict[str, Any]) -> Dict[str, Any]:
    converted = dict(payload)
    converted["impact_amount"] = float(converted["impact_amount"])
    converted["confidence"] = float(converted["confidence"])
    converted["evidence"] = {
        key: float(value) if isinstance(value, Decimal) else value
        for key, value in converted.get("evidence", {}).items()
    }
    return converted


def detect_return_spike_leak(
    sku: str,
    baseline_returns: int,
    baseline_sales: int,
    eval_returns: int,
    eval_sales: int,
    asp: float,
    top_reason: str,
    reason_ratio: float,
) -> Optional[Dict[str, Any]]:
    result = _detect_return_spike_leak(
        sku=sku,
        baseline_returns=baseline_returns,
        baseline_sales=baseline_sales,
        eval_returns=eval_returns,
        eval_sales=eval_sales,
        asp=asp,
        top_reason=top_reason,
        reason_ratio=reason_ratio,
    )
    if result.get("status") == "INSUFFICIENT_DATA":
        return None
    output = _legacy_numbers(result)

    # Augment with ML Return Propensity
    try:
        from backend.ml.pipeline import get_ml_suite
        ml_return = get_ml_suite().predict_return_risk({
            "sku": sku,
            "unit_price": float(asp),
            "quantity": 1,
            "sku_return_rate": float(eval_returns / eval_sales) if eval_sales > 0 else 0.05,
        })
        output["evidence"]["ml_return_propensity"] = {
            "return_probability": ml_return["return_probability"],
            "risk_tier": ml_return["risk_tier"],
            "contributing_factors": ml_return["contributing_factors"],
            "model_status": ml_return["model_status"],
        }
    except Exception:
        pass

    return output
