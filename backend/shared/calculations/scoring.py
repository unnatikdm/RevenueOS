"""
TASK-018: Unified Revenue Opportunity Scoring Engine (Advanced Enterprise Level)
Target: backend/shared/calculations/scoring.py

Composite Opportunity Scoring Formula (bounded strictly 0 to 100):
Opportunity Score = min(100, (
    ((log10(max(1000, Impact)) - 3) / 3 * 40) +
    (Confidence * 30) +
    (Urgency * 20) +
    (Evidence_Strength * 10)
))

Parameters:
1. Financial Scale (0-40 pts, logarithmic): Scaled from ₹1,000 to ₹10,00,000
2. Algorithmic Confidence (0-30 pts): Derived from sample depth, stability, z-score
3. Urgency Weight (0-20 pts): Anomaly trend velocity and active rate of capital bleed
4. Evidence Strength (0-10 pts): Data richness, transaction counts, and correlation strength
"""

import math
from typing import Any, Dict, List, Optional


def compute_opportunity_score(
    impact_amount: float,
    confidence: float,
    urgency_weight: float = 0.8,
    evidence_strength: float = 0.9,
) -> float:
    """
    Computes normalized Opportunity Score (0-100) using logarithmic financial scaling.
    """
    if impact_amount <= 0:
        return 0.0

    # 1. Financial Impact: scaled logarithmically from ₹1,000 (log10=3) to ₹10,00,000 (log10=6)
    clamped_impact = max(1000.0, float(impact_amount))
    log_val = math.log10(clamped_impact)
    financial_pts = ((log_val - 3.0) / 3.0) * 40.0
    financial_pts = max(0.0, min(40.0, financial_pts))

    # 2. Algorithmic Confidence (0 to 30 pts)
    clamped_conf = max(0.0, min(1.0, float(confidence)))
    confidence_pts = clamped_conf * 30.0

    # 3. Urgency / Velocity Weight (0 to 20 pts)
    clamped_urgency = max(0.0, min(1.0, float(urgency_weight)))
    urgency_pts = clamped_urgency * 20.0

    # 4. Evidence Sample Strength (0 to 10 pts)
    clamped_evidence = max(0.0, min(1.0, float(evidence_strength)))
    evidence_pts = clamped_evidence * 10.0

    total_score = financial_pts + confidence_pts + urgency_pts + evidence_pts
    return round(max(0.0, min(100.0, total_score)), 1)


def rank_and_score_leaks(leaks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Annotates each leak candidate with its calculated Opportunity Score
    and returns leaks sorted in descending order of financial priority.
    """
    scored_leaks = []
    for leak in leaks:
        impact = float(leak.get("impact_amount", 0.0))
        conf = float(leak.get("confidence", 0.80))
        evidence = leak.get("evidence", {})

        # Derive dynamic urgency from leak type and metrics
        leak_type = leak.get("leak_type", "")
        if leak_type == "stockout":
            urgency = 1.0 if evidence.get("stockout_days", 0) >= 5 else 0.8
            evidence_str = 0.95 if evidence.get("daily_demand_rate", 0) > 2.0 else 0.80
        elif leak_type == "return_spike":
            urgency = 0.9 if evidence.get("percentage_increase", 0) > 100 else 0.75
            evidence_str = min(1.0, 0.7 + (evidence.get("reason_prevalence", 0.5) * 0.3))
        elif leak_type == "checkout_friction":
            urgency = 0.95
            evidence_str = 0.90
        elif leak_type == "pricing":
            urgency = 0.75
            evidence_str = 0.85
        else:
            urgency = 0.70
            evidence_str = 0.80

        opp_score = compute_opportunity_score(impact, conf, urgency, evidence_str)
        leak_copy = dict(leak)
        leak_copy["opp_score"] = opp_score
        leak_copy["urgency_weight"] = urgency
        leak_copy["evidence_strength"] = evidence_str
        scored_leaks.append(leak_copy)

    # Sort descending: primary recommendation first
    return sorted(scored_leaks, key=lambda x: (x["opp_score"], x["impact_amount"]), reverse=True)
