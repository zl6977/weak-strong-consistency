"""
Legacy confidence fusion retained from earlier score-based OOD experiments.

The paper's main analysis uses black-box top-label agreement instead of this
derived confidence score path.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FusionConfig:
    """Configuration for confidence fusion."""
    agreement_metrics_weights: Dict[str, float] = field(
        default_factory=lambda: {"js_divergence": 0.4, "cosine_similarity": 0.3, "top_k_overlap": 0.3}
    )
    agreement_weight: float = 0.5
    confidence_weight: float = 0.5
    gamma: float = 1.0
    strategy: Optional[str] = None


def compute_agreement_factor(
    agreement_metrics: Dict[str, float],
    config: FusionConfig,
) -> float:
    """Compute a single disagreement factor in [0, 1].

    Higher factor = more disagreement = stronger OOD signal.

    JS divergence is already higher-disagreement.
    Cosine and overlap are higher-agreement, so inverted: 1 - value.
    """
    ln2 = float(np.log(2))
    metrics = {}
    weights = {}

    if "js_divergence" in config.agreement_metrics_weights:
        metrics["js_divergence"] = min(agreement_metrics.get("js_divergence", 0) / ln2, 1.0)
        weights["js_divergence"] = config.agreement_metrics_weights["js_divergence"]

    if "cosine_similarity" in config.agreement_metrics_weights:
        metrics["cosine_similarity"] = 1.0 - agreement_metrics.get("cosine_similarity", 1.0)
        weights["cosine_similarity"] = config.agreement_metrics_weights["cosine_similarity"]

    if "top_k_overlap" in config.agreement_metrics_weights:
        metrics["top_k_overlap"] = 1.0 - agreement_metrics.get("top_k_overlap", 1.0)
        weights["top_k_overlap"] = config.agreement_metrics_weights["top_k_overlap"]

    if not metrics:
        return 0.0

    total_weight = sum(weights.values())
    agreement_factor = sum(weights[k] * metrics[k] for k in metrics) / total_weight
    return float(np.clip(agreement_factor, 0.0, 1.0))


def fuse_confidence(
    id_conf_A: float,
    id_conf_B: float,
    agreement_metrics: Dict[str, float],
    config: FusionConfig,
) -> Dict[str, float]:
    """Fuse per-model ID confidence with inter-model agreement into OOD score.

    Returns dict with ood_score, id_conf_A, id_conf_B, id_conf_avg,
    agreement_factor, and the full agreement_metrics.
    """
    id_conf_avg = float((id_conf_A + id_conf_B) / 2.0)

    agreement_factor = compute_agreement_factor(agreement_metrics, config)

    ood_from_conf = float(np.clip(1.0 - (id_conf_avg ** config.gamma), 0.0, 1.0))

    ood_score = (
        config.confidence_weight * ood_from_conf
        + config.agreement_weight * agreement_factor
    )
    ood_score = float(np.clip(ood_score, 0.0, 1.0))

    return {
        "ood_score": ood_score,
        "id_conf_A": id_conf_A,
        "id_conf_B": id_conf_B,
        "id_conf_avg": id_conf_avg,
        "agreement_factor": agreement_factor,
        "agreement_metrics": agreement_metrics,
    }
