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

    evidence_dict: Dict[str, Any] = {
        "cohort_size": cohort_size,
        "baseline_repeat_rate": round(baseline_repeat_rate, 4),
        "actual_repeat_rate": round(actual_repeat_rate, 4),
        "retention_drop_percentage": retention_drop_pct,
        "expected_repeat_customers": expected_repeat_customers,
        "actual_repeat_customers": actual_repeat_customers,
        "missed_repeat_customers": missed_customers,
        "repeat_average_order_value": round(repeat_aov, 2),
        "z_score": round(z_score, 2),
    }

    # Augment with K-Means Customer RFM Segment Telemetry
    try:
        from backend.ml.pipeline import get_ml_suite
        ml_rfm = get_ml_suite().segment_customer_rfm({
            "customer_id": cohort_id,
            "recency_days": 60.0,
            "frequency_orders": max(1, int(actual_repeat_customers / max(1, cohort_size * 0.1))),
            "monetary_spend": float(actual_repeat_customers * repeat_aov),
        })
        evidence_dict["ml_rfm_segmentation"] = {
            "segment_label": ml_rfm["segment_label"],
            "churn_risk_score": ml_rfm["churn_risk_score"],
            "projected_ltv_loss": ml_rfm["projected_ltv_loss"],
            "recommended_action": ml_rfm["recommended_action"],
            "model_status": ml_rfm["model_status"],
        }
    except Exception:
        pass

    return {
        "leak_type": "retention",
        "entity_id": cohort_id,
        "impact_amount": lost_revenue,
        "confidence": round(confidence, 2),
        "evidence": evidence_dict,
    }
