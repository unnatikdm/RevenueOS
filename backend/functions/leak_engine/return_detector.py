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
    return _legacy_numbers(result)
