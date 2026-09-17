from decimal import Decimal
from typing import Any, Dict, Optional

from backend.shared.calculations.engines import detect_pricing_elasticity_leak as _detect_pricing_elasticity_leak


def _legacy_numbers(payload: Dict[str, Any]) -> Dict[str, Any]:
    converted = dict(payload)
    converted["impact_amount"] = float(converted["impact_amount"])
    converted["confidence"] = float(converted["confidence"])
    converted["evidence"] = {
        key: float(value) if isinstance(value, Decimal) else value
        for key, value in converted.get("evidence", {}).items()
    }
    return converted


def detect_price_elasticity_leak(
    sku: str,
    p1: float,
    q1: int,
    p2: float,
    q2: int,
    unit_cost: float = 0.0,
) -> Optional[Dict[str, Any]]:
    result = _detect_pricing_elasticity_leak(
        sku=sku,
        p1=p1,
        q1=q1,
        p2=p2,
        q2=q2,
        unit_cost=unit_cost,
    )
    if result.get("status") == "INSUFFICIENT_DATA":
        return None
    return _legacy_numbers(result)
