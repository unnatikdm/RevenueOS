from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("revenueos.ml.customer_clustering")


class CustomerRFMClusterer:
    """
    Customer RFM Behavioral Segmentation & Churn Risk Engine.
    Employs K-Means Clustering on normalized Recency, Frequency, Monetary (RFM),
    and Average Order Value (AOV) vectors to discover behavioral customer cohorts
    and preempt high-value churn revenue leaks.
    """

    FEATURE_NAMES = ["recency_days", "frequency_orders", "monetary_spend", "average_order_value"]

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.is_fitted: bool = False
        self._cluster_map: Dict[int, str] = {}

    def _extract_rfm_vector(self, customer_stats: Dict[str, Any]) -> List[float]:
        recency = float(customer_stats.get("recency_days", customer_stats.get("recency", 30.0)))
        freq = float(customer_stats.get("frequency_orders", customer_stats.get("frequency", 2.0)))
        monetary = float(customer_stats.get("monetary_spend", customer_stats.get("monetary", 2500.0)))
        aov = (monetary / freq) if freq > 0 else monetary

        # Log transform skewed features for cluster stability
        return [
            max(0.0, recency),
            float(np.log1p(max(0.0, freq))),
            float(np.log1p(max(0.0, monetary))),
            float(np.log1p(max(0.0, aov))),
        ]

    def fit(self, customer_records: Sequence[Dict[str, Any]]) -> CustomerRFMClusterer:
        """Fits the StandardScaler and KMeans model on a cohort of customer RFM profiles."""
        if len(customer_records) < self.n_clusters:
            logger.warning("Insufficient customer samples (%d) for KMeans clustering.", len(customer_records))
            self.is_fitted = False
            return self

        raw_vectors = np.array(
            [self._extract_rfm_vector(c) for c in customer_records],
            dtype=np.float64,
        )

        scaled_X = self.scaler.fit_transform(raw_vectors)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.kmeans.fit(scaled_X)

        # Interpret cluster centers to assign intuitive labels
        # Centers are in scaled space: [recency, log_freq, log_monetary, log_aov]
        centers = self.kmeans.cluster_centers_
        cluster_labels = {}
        for idx, center in enumerate(centers):
            r_val, f_val, m_val, aov_val = center
            if r_val > 0.4 and m_val > 0.2:
                cluster_labels[idx] = "AT_RISK_HIGH_VALUE"
            elif r_val <= 0.0 and m_val > 0.3:
                cluster_labels[idx] = "CHAMPIONS"
            elif r_val <= 0.2 and f_val >= 0.0:
                cluster_labels[idx] = "LOYAL_CORE"
            else:
                cluster_labels[idx] = "DORMANT_OCCASIONAL"

        self._cluster_map = cluster_labels
        self.is_fitted = True
        logger.info("CustomerRFMClusterer fitted on %d customers across %d clusters.", len(customer_records), self.n_clusters)
        return self

    def predict_segment(self, customer_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies a customer or cohort into an RFM segment and computes churn revenue risk.
        """
        recency = float(customer_stats.get("recency_days", 30.0))
        freq = float(customer_stats.get("frequency_orders", 2.0))
        monetary = float(customer_stats.get("monetary_spend", 2500.0))
        aov = round((monetary / freq) if freq > 0 else monetary, 2)
        customer_id = str(customer_stats.get("customer_id", "CUST-UNKNOWN"))

        if not self.is_fitted or self.kmeans is None:
            # Deterministic RFM rule fallback
            if recency > 60 and monetary > 5000:
                seg = "AT_RISK_HIGH_VALUE"
                churn_score = 0.82
            elif recency <= 21 and monetary > 4000:
                seg = "CHAMPIONS"
                churn_score = 0.10
            elif recency <= 45:
                seg = "LOYAL_CORE"
                churn_score = 0.35
            else:
                seg = "DORMANT_OCCASIONAL"
                churn_score = 0.70
            cluster_id = 0
        else:
            raw_vec = np.array([self._extract_rfm_vector(customer_stats)], dtype=np.float64)
            scaled_vec = self.scaler.transform(raw_vec)
            cluster_id = int(self.kmeans.predict(scaled_vec)[0])
            seg = self._cluster_map.get(cluster_id, "LOYAL_CORE")

            # Calculate continuous churn risk score based on recency vs frequency
            churn_score = float(np.clip((recency / 90.0) * (0.8 if seg == "AT_RISK_HIGH_VALUE" else 0.5), 0.05, 0.95))

        # Financial exposure: estimated lost future LTV if churn is realized
        if seg == "AT_RISK_HIGH_VALUE":
            projected_ltv_loss = round(monetary * 0.50, 2)
            action = "Trigger personalized high-touch VIP win-back campaign with exclusive incentive."
        elif seg == "DORMANT_OCCASIONAL":
            projected_ltv_loss = round(aov * 1.0, 2)
            action = "Re-engage via automated reactivation email sequence."
        elif seg == "CHAMPIONS":
            projected_ltv_loss = 0.0
            action = "Enroll in loyalty tier and early-access previews to maximize retention."
        else:
            projected_ltv_loss = round(aov * 0.25, 2)
            action = "Standard replenishment prompts based on expected order cycle."

        return {
            "customer_id": customer_id,
            "cluster_id": cluster_id,
            "segment_label": seg,
            "churn_risk_score": round(churn_score, 3),
            "projected_ltv_loss": projected_ltv_loss,
            "recommended_action": action,
            "rfm_summary": {
                "recency_days": round(recency, 1),
                "frequency_orders": int(freq),
                "monetary_spend": round(monetary, 2),
                "average_order_value": aov,
            },
            "model_status": "KMEANS_CLUSTERING" if self.is_fitted else "DETERMINISTIC_RFM_FALLBACK",
        }
