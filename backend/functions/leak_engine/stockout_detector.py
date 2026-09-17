from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from backend.shared.calculations.engines import detect_stockout_leak as _detect_stockout_leak


def _legacy_numbers(payload: Dict[str, Any]) -> Dict[str, Any]:
    converted = dict(payload)
    converted["impact_amount"] = float(converted["impact_amount"])
    converted["confidence"] = float(converted["confidence"])
    evidence = {}
    for key, value in converted.get("evidence", {}).items():
        evidence[key] = float(value) if isinstance(value, Decimal) else value
    converted["evidence"] = evidence
    return converted


def detect_stockout_leak(
    product_id: str,
    sku: str,
    daily_sales: List[int],
    stockout_days: int,
    asp: float,
    tenant_id: Optional[str] = None,
    inventory_units: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    result = _detect_stockout_leak(
        product_id=product_id,
        sku=sku,
        daily_sales=daily_sales,
        stockout_days=stockout_days,
        asp=asp,
        inventory_units=inventory_units,
        tenant_id=tenant_id,
    )
    if result.get("status") == "INSUFFICIENT_DATA":
        return None
    output = _legacy_numbers(result)

    # Augment with ML Demand Forecaster
    try:
        from backend.ml.pipeline import get_ml_suite
        ml_forecast = get_ml_suite().forecast_sku_demand(
            historical_sales=daily_sales,
            current_inventory=0,
            asp=float(asp),
            horizon_days=max(7, stockout_days),
        )
        output["evidence"]["ml_forecast"] = {
            "projected_daily_velocity": ml_forecast["projected_daily_velocity"],
            "stockout_risk_tier": ml_forecast["stockout_risk_tier"],
            "estimated_uncaptured_revenue": ml_forecast["estimated_uncaptured_revenue"],
            "model_confidence": ml_forecast["model_confidence"],
            "model_engine": ml_forecast["model_engine"],
        }
    except Exception:
        pass

    return output
