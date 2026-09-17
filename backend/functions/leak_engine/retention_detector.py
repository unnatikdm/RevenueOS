"""
TASK-017: Customer Retention Decay Engine (Advanced Enterprise Level)
Target: backend/functions/leak_engine/retention_detector.py

Deterministic Algorithm:
- 60-day customer cohort repeat purchase evaluation.
- Compares actual repeat order customer counts against historical baseline repeat ratios:
    Expected Repeat Customers = ceil(N_cohort * RR_base)
    Missed Repeat Customers = max(0, Expected Repeat Customers - N_actual)
    Impact = Missed Repeat Customers * AOV_repeat
- Calculates statistical binomial significance (Z-score) of retention decay:
    SE_retention = sqrt(RR_base * (1 - RR_base) / N_cohort)
    Z_decay = (RR_base - RR_actual) / SE_retention
"""

import math
from typing import Any, Dict, Optional


def detect_retention_decay_leak(
    cohort_id: str,
    cohort_size: int,
    actual_repeat_customers: int,
    baseline_repeat_rate: float,
    repeat_aov: float,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates customer retention decay across customer cohorts.
    - cohort_size: Total new customers in cohort acquisition window (e.g. Month 0)
    - actual_repeat_customers: Number of customers who repurchased within 60 days
    - baseline_repeat_rate: Historical benchmark repeat purchase rate (e.g. 0.28)
    - repeat_aov: Average Order Value on repeat purchases
    """
    if cohort_size < 50 or baseline_repeat_rate <= 0.0 or repeat_aov <= 0.0:
        return None

    actual_repeat_rate = actual_repeat_customers / cohort_size

    # Flag anomaly if actual repeat rate drops by >= 15% relative to baseline
    if actual_repeat_rate >= (baseline_repeat_rate * 0.85):
        return None

    expected_repeat_customers = math.ceil(cohort_size * baseline_repeat_rate)
    missed_customers = max(0, expected_repeat_customers - actual_repeat_customers)

    if missed_customers <= 0:
        return None

    lost_revenue = round(missed_customers * repeat_aov, 2)

    # Binomial Z-score of retention decay
    se = math.sqrt(baseline_repeat_rate * (1.0 - baseline_repeat_rate) / cohort_size)
    z_score = (baseline_repeat_rate - actual_repeat_rate) / se if se > 0 else 0.0

    retention_drop_pct = round(((baseline_repeat_rate - actual_repeat_rate) / baseline_repeat_rate) * 100, 1)

    # Confidence scaling based on cohort size and z-score
    confidence = min(0.98, max(0.72, 0.78 + (0.04 * min(z_score, 4.0))))

    return {
        "leak_type": "retention",
        "entity_id": cohort_id,
        "impact_amount": lost_revenue,
        "confidence": round(confidence, 2),
        "evidence": {
            "cohort_size": cohort_size,
            "baseline_repeat_rate": round(baseline_repeat_rate, 4),
            "actual_repeat_rate": round(actual_repeat_rate, 4),
            "retention_drop_percentage": retention_drop_pct,
            "expected_repeat_customers": expected_repeat_customers,
            "actual_repeat_customers": actual_repeat_customers,
            "missed_repeat_customers": missed_customers,
            "repeat_average_order_value": round(repeat_aov, 2),
            "z_score": round(z_score, 2),
        },
    }
