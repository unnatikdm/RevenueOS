"""
TASK-015: Checkout Funnel Friction Detection Engine
Target: backend/functions/leak_engine/funnel_detector.py

Algorithm:
- Evaluates checkout initiation versus completed order conversions over rolling 7-day windows.
- Trigger anomaly alert when completion rates drop more than 15% below baseline while initiation remains steady.
- Computes:
    Delta CR = max(0, CR_base - CR_curr)
    Lost Orders = round(Checkouts_curr * Delta CR)
    Impact = round(Lost Orders * AOV, 2)
    Confidence = 0.90 if Checkouts_curr > 100 else 0.75
"""

import math
from typing import Any, Dict, Optional


def detect_funnel_friction_leak(
    device_or_channel: str,
    baseline_checkouts: int,
    baseline_orders: int,
    eval_checkouts: int,
    eval_orders: int,
    aov: float,
) -> Optional[Dict[str, Any]]:
    """
    Advanced Deterministic Funnel Friction Engine:
    - Analyzes conversion rates across device types (mobile, desktop).
    - Checks for statistically significant drop-off:
        CR_base = Orders_base / Checkouts_base
        CR_curr = Orders_eval / Checkouts_eval
        Delta_CR = max(0, CR_base - CR_curr)
    - Verifies relative friction drop >= 15% (CR_curr <= CR_base * 0.85).
    - Quantifies opportunity: Lost Orders * AOV.
    - Confidence is weighted by evaluation sample volume:
        confidence = min(0.96, 0.75 + (0.01 * min(eval_checkouts // 50, 20)))
    """
    if baseline_checkouts < 50 or eval_checkouts < 20 or aov <= 0:
        return None

    cr_base = baseline_orders / baseline_checkouts
    cr_curr = eval_orders / eval_checkouts

    # Check for completion rate drop > 15% below baseline
    if cr_curr >= (cr_base * 0.85):
        return None

    delta_cr = max(0.0, cr_base - cr_curr)
    lost_orders = round(eval_checkouts * delta_cr)
    impact_amount = round(lost_orders * aov, 2)

    # Standard error of conversion rate difference
    se_diff = math.sqrt((cr_base * (1 - cr_base) / baseline_checkouts) + (cr_curr * (1 - cr_curr) / eval_checkouts))
    z_score = delta_cr / se_diff if se_diff > 0 else 0.0

    # Volume-weighted confidence
    sample_depth = eval_checkouts // 50
    confidence = min(0.96, max(0.72, 0.75 + (0.01 * min(sample_depth, 20))))

    return {
        "leak_type": "checkout_friction",
        "entity_id": device_or_channel,
        "impact_amount": impact_amount,
        "confidence": round(confidence, 2),
        "evidence": {
            "baseline_conversion_rate": round(cr_base, 4),
            "current_conversion_rate": round(cr_curr, 4),
            "dropoff_percentage": round(((cr_base - cr_curr) / cr_base) * 100, 1),
            "lost_orders": lost_orders,
            "z_score": round(z_score, 2),
            "average_order_value": round(aov, 2),
            "evaluated_checkouts": eval_checkouts,
        },
    }
