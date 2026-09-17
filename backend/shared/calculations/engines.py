from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

getcontext().prec = 28

Z_ONE_TAILED_95 = Decimal("1.645")
Z_TWO_TAILED_95 = Decimal("1.96")
MIN_STOCKOUT_DEMAND_DAYS = 7
MIN_RETURN_WINDOW_N = 30
MIN_PRICE_WINDOW_UNITS = 30


def _d(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _rate(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _insufficient(leak_type: str, entity_id: str, reason: str, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "leak_type": leak_type,
        "entity_id": entity_id,
        "status": "INSUFFICIENT_DATA",
        "reason": reason,
        "impact_amount": Decimal("0.00"),
        "confidence": Decimal("0.00"),
        "evidence": metrics or {},
    }


def _as_active_demand_samples(
    daily_sales: Sequence[Any],
    inventory_units: Optional[Sequence[Any]],
) -> Tuple[List[Decimal], int]:
    if inventory_units is None:
        return [_d(units) for units in daily_sales if _d(units) >= 0], 0

    active: List[Decimal] = []
    censored_days = 0
    for units, inventory in zip(daily_sales, inventory_units):
        if _d(inventory) <= 0:
            censored_days += 1
            continue
        units_decimal = _d(units)
        if units_decimal >= 0:
            active.append(units_decimal)
    return active, censored_days


def _sample_variance(samples: Sequence[Decimal], mean: Decimal) -> Decimal:
    if len(samples) < 2:
        return Decimal("0")
    numerator = sum((sample - mean) ** 2 for sample in samples)
    return numerator / Decimal(len(samples) - 1)


def detect_stockout_leak(
    product_id: str,
    sku: str,
    daily_sales: Sequence[Any],
    stockout_days: int,
    asp: Any,
    inventory_units: Optional[Sequence[Any]] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    active_samples, censored_days = _as_active_demand_samples(daily_sales, inventory_units)
    asp_decimal = _d(asp)

    if stockout_days <= 0 or asp_decimal <= 0:
        return _insufficient("stockout", sku, "stockout_days and asp must be positive")
    if len(active_samples) < MIN_STOCKOUT_DEMAND_DAYS:
        return _insufficient(
            "stockout",
            sku,
            "not enough non-censored selling days to estimate baseline demand",
            {"active_demand_days": len(active_samples), "censored_zero_inventory_days": censored_days},
        )

    n = Decimal(len(active_samples))
    mean = sum(active_samples) / n
    if mean <= 0:
        return _insufficient("stockout", sku, "baseline demand is zero after censoring stockout intervals")

    variance = _sample_variance(active_samples, mean)
    std_dev = Decimal(str(math.sqrt(float(variance))))
    margin = Z_TWO_TAILED_95 * (std_dev / Decimal(str(math.sqrt(float(n)))))
    ci_low = max(Decimal("0"), mean - margin)
    ci_high = mean + margin
    missed_units_decimal = mean * Decimal(stockout_days)
    missed_units = int(missed_units_decimal.to_integral_value(rounding=ROUND_HALF_UP))
    impact = _money(Decimal(missed_units) * asp_decimal)
    cv = std_dev / mean if mean else Decimal("0")
    confidence = min(Decimal("0.98"), max(Decimal("0.60"), Decimal("0.76") + Decimal("0.01") * min(n, Decimal("20")) - min(cv, Decimal("1.0")) * Decimal("0.08")))

    return {
        "leak_type": "stockout",
        "entity_id": sku,
        "status": "ACTIVE",
        "impact_amount": impact,
        "confidence": _rate(confidence),
        "evidence": {
            "product_id": product_id,
            "daily_demand_rate": _rate(mean),
            "demand_sample_size": len(active_samples),
            "demand_sample_variance": _rate(variance),
            "ddr_confidence_interval_low": _rate(ci_low),
            "ddr_confidence_interval_high": _rate(ci_high),
            "stockout_days": stockout_days,
            "average_selling_price": _money(asp_decimal),
            "estimated_missed_units": missed_units,
            "censored_zero_inventory_days": censored_days,
        },
    }


def _proportion_sample_ok(successes: int, trials: int) -> bool:
    if trials <= 0:
        return False
    p = Decimal(successes) / Decimal(trials)
    return Decimal(trials) * p >= 5 and Decimal(trials) * (Decimal("1") - p) >= 5


def detect_return_spike_leak(
    sku: str,
    baseline_returns: int,
    baseline_sales: int,
    eval_returns: int,
    eval_sales: int,
    asp: Any,
    top_reason: Optional[str] = None,
    reason_ratio: Any = Decimal("0"),
    reverse_logistics_cost: Any = Decimal("120.00"),
    salvage_loss_rate: Any = Decimal("0.70"),
) -> Dict[str, Any]:
    asp_decimal = _d(asp)
    if baseline_sales < MIN_RETURN_WINDOW_N or eval_sales < MIN_RETURN_WINDOW_N or asp_decimal <= 0:
        return _insufficient("return_spike", sku, "baseline and evaluation windows do not meet minimum sample size")
    if not _proportion_sample_ok(baseline_returns, baseline_sales) or not _proportion_sample_ok(eval_returns, eval_sales):
        return _insufficient("return_spike", sku, "two-proportion z-test normal approximation criteria are not satisfied")

    p_base = Decimal(baseline_returns) / Decimal(baseline_sales)
    p_eval = Decimal(eval_returns) / Decimal(eval_sales)
    pooled = Decimal(baseline_returns + eval_returns) / Decimal(baseline_sales + eval_sales)
    se = Decimal(str(math.sqrt(float(pooled * (Decimal("1") - pooled) * ((Decimal("1") / Decimal(eval_sales)) + (Decimal("1") / Decimal(baseline_sales)))))))
    z_score = (p_eval - p_base) / se if se > 0 else Decimal("0")

    if z_score < Z_ONE_TAILED_95:
        return _insufficient(
            "return_spike",
            sku,
            "return rate increase is not statistically significant at alpha=0.05 one-tailed",
            {"z_score": _rate(z_score), "baseline_return_rate": _rate(p_base), "current_return_rate": _rate(p_eval)},
        )

    excess_units = int((Decimal(eval_sales) * (p_eval - p_base)).to_integral_value(rounding=ROUND_HALF_UP))
    unit_drag = asp_decimal * _d(salvage_loss_rate) + _d(reverse_logistics_cost)
    impact = _money(Decimal(max(0, excess_units)) * unit_drag)
    percentage_increase = ((p_eval - p_base) / p_base * Decimal("100")) if p_base > 0 else Decimal("0")
    confidence = min(Decimal("0.99"), Decimal("0.80") + min(z_score, Decimal("4.0")) * Decimal("0.04"))

    return {
        "leak_type": "return_spike",
        "entity_id": sku,
        "status": "ACTIVE",
        "impact_amount": impact,
        "confidence": _rate(confidence),
        "evidence": {
            "baseline_return_rate": _rate(p_base),
            "current_return_rate": _rate(p_eval),
            "percentage_increase": _rate(percentage_increase),
            "z_score": _rate(z_score),
            "p_value_alpha": Decimal("0.05"),
            "excess_units": excess_units,
            "unit_cost_drag": _money(unit_drag),
            "top_correlated_reason": top_reason,
            "reason_prevalence": _rate(_d(reason_ratio)),
        },
    }


def detect_pricing_elasticity_leak(
    sku: str,
    p1: Any,
    q1: int,
    p2: Any,
    q2: int,
    unit_cost: Any = Decimal("0.00"),
    min_units_per_window: int = MIN_PRICE_WINDOW_UNITS,
) -> Dict[str, Any]:
    price_1 = _d(p1)
    price_2 = _d(p2)
    cost = _d(unit_cost)

    if price_1 <= 0 or price_2 <= 0 or q1 < min_units_per_window or q2 < min_units_per_window:
        return _insufficient("pricing", sku, "both price windows must meet minimum transaction volume")
    if price_1 == price_2 or q1 + q2 == 0:
        return _insufficient("pricing", sku, "price did not change or quantity observations are empty")

    quantity_delta_ratio = (Decimal(q2 - q1) / Decimal(q2 + q1))
    price_delta_ratio = ((price_2 - price_1) / (price_2 + price_1))
    if price_delta_ratio == 0:
        return _insufficient("pricing", sku, "midpoint price denominator is zero")

    elasticity = quantity_delta_ratio / price_delta_ratio
    gross_revenue_1 = price_1 * Decimal(q1)
    gross_revenue_2 = price_2 * Decimal(q2)
    lost_revenue = max(Decimal("0.00"), gross_revenue_1 - gross_revenue_2)

    if price_2 <= price_1 or elasticity >= Decimal("-1.0") or lost_revenue <= 0:
        return _insufficient(
            "pricing",
            sku,
            "price change did not create statistically actionable elastic revenue contraction",
            {"price_elasticity": _rate(elasticity), "gross_revenue_delta": _money(gross_revenue_2 - gross_revenue_1)},
        )

    margin_loss = None
    if cost > 0:
        margin_loss = _money(((price_1 - cost) * Decimal(q1)) - ((price_2 - cost) * Decimal(q2)))

    confidence = min(Decimal("0.97"), Decimal("0.72") + Decimal(min(q1, q2, 300)) / Decimal("3000") + min(abs(elasticity), Decimal("3")) * Decimal("0.04"))

    return {
        "leak_type": "pricing",
        "entity_id": sku,
        "status": "ACTIVE",
        "impact_amount": _money(lost_revenue),
        "confidence": _rate(confidence),
        "evidence": {
            "baseline_price": _money(price_1),
            "post_change_price": _money(price_2),
            "baseline_units": q1,
            "post_change_units": q2,
            "price_elasticity": _rate(elasticity),
            "gross_revenue_loss": _money(lost_revenue),
            "gross_margin_loss": margin_loss,
        },
    }
