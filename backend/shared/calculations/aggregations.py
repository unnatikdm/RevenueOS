"""
TASK-012: Rolling Time-Series Metric Aggregator
Target: backend/shared/calculations/aggregations.py

Input: DynamoDB item collections or normalized canonical records
Vectorized/deterministic statistical calculation routines:
1. compute_daily_demand_rate(daily_units: List[int], window_days: int = 30) -> float
2. compute_return_rate_metrics(sales_units: int, return_units: int, reason_counts: Dict[str, int]) -> Dict[str, Any]
3. compute_checkout_funnel_dropoff(checkouts: int, orders: int, baseline_cr: float) -> Dict[str, Any]
4. compute_price_elasticity(p1: float, q1: int, p2: float, q2: int) -> float
5. compute_cohort_retention_metrics(cohort_size: int, repeat_customers: int, baseline_repeat_rate: float, repeat_aov: float) -> Dict[str, Any]
6. aggregate_rolling_metrics(orders: List[Dict], returns: List[Dict], inventory: List[Dict], funnel: List[Dict]) -> Dict[str, Any]
"""

import math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


def compute_daily_demand_rate(daily_units: List[int], window_days: int = 30) -> float:
    """
    Computes the Daily Demand Rate (DDR) over active selling days.
    DDR_i = (1 / |T_active|) * sum(Q_{i,t}) for t in T_active
    """
    if not daily_units:
        return 0.0
    active_days_sales = [u for u in daily_units[-window_days:] if u >= 0]
    if not active_days_sales:
        return 0.0
    return round(sum(active_days_sales) / len(active_days_sales), 4)


def compute_return_rate_metrics(
    sales_units: int,
    return_units: int,
    reason_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Computes return rates and reason distributions per SKU.
    R = Return_Units / Sales_Units
    """
    if sales_units <= 0:
        return {
            "sales_units": 0,
            "return_units": return_units,
            "return_rate": 0.0,
            "top_reason": "None",
            "top_reason_ratio": 0.0,
            "reason_distribution": {},
        }

    return_rate = round(return_units / sales_units, 4)
    reasons = reason_counts or {}
    total_reasons = sum(reasons.values())
    if total_reasons > 0:
        sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        top_reason, top_count = sorted_reasons[0]
        top_ratio = round(top_count / total_reasons, 4)
    else:
        top_reason = "Customer Changed Mind"
        top_ratio = 1.0

    return {
        "sales_units": sales_units,
        "return_units": return_units,
        "return_rate": return_rate,
        "top_reason": top_reason,
        "top_reason_ratio": top_ratio,
        "reason_distribution": reasons,
    }


def compute_checkout_funnel_dropoff(
    checkouts: int,
    orders: int,
    baseline_cr: float,
) -> Dict[str, Any]:
    """
    Computes checkout funnel completion rate and relative drop-off vs baseline:
    CR_curr = Orders / Checkouts
    Dropoff = max(0, baseline_cr - CR_curr)
    """
    if checkouts <= 0:
        return {
            "checkouts": 0,
            "orders": 0,
            "current_cr": 0.0,
            "baseline_cr": baseline_cr,
            "dropoff_cr": 0.0,
            "is_anomaly": False,
        }

    current_cr = round(orders / checkouts, 4)
    delta_cr = max(0.0, baseline_cr - current_cr)
    # Trigger anomaly alert when completion rates drop more than 15% below baseline
    is_anomaly = (current_cr < (baseline_cr * 0.85)) and (delta_cr > 0.05)

    return {
        "checkouts": checkouts,
        "orders": orders,
        "current_cr": current_cr,
        "baseline_cr": baseline_cr,
        "dropoff_cr": round(delta_cr, 4),
        "is_anomaly": is_anomaly,
    }


def compute_price_elasticity(p1: float, q1: int, p2: float, q2: int) -> float:
    """
    Price elasticity of demand:
    epsilon = ((Q2 - Q1) / Q1) / ((P2 - P1) / P1)
    """
    if q1 <= 0 or p1 <= 0.0 or (p2 - p1) == 0:
        return 0.0

    pct_delta_q = (q2 - q1) / q1
    pct_delta_p = (p2 - p1) / p1

    return round(pct_delta_q / pct_delta_p, 4)


def compute_cohort_retention_metrics(
    cohort_size: int,
    repeat_customers: int,
    baseline_repeat_rate: float,
    repeat_aov: float,
) -> Dict[str, Any]:
    """
    Evaluates customer retention decay across 60-day customer cohorts.
    Missed Repeat Customers = max(0, cohort_size * baseline_repeat_rate - repeat_customers)
    Impact = Missed Customers * repeat_aov
    """
    if cohort_size <= 0:
        return {
            "cohort_size": 0,
            "repeat_customers": 0,
            "current_retention_rate": 0.0,
            "baseline_retention_rate": baseline_repeat_rate,
            "missed_customers": 0,
            "lost_revenue": 0.0,
        }

    current_rr = round(repeat_customers / cohort_size, 4)
    expected_repeat = math.ceil(cohort_size * baseline_repeat_rate)
    shortfall = max(0, expected_repeat - repeat_customers)
    lost_rev = round(shortfall * repeat_aov, 2)

    return {
        "cohort_size": cohort_size,
        "repeat_customers": repeat_customers,
        "current_retention_rate": current_rr,
        "baseline_retention_rate": baseline_repeat_rate,
        "missed_customers": shortfall,
        "lost_revenue": lost_rev,
    }
