"""
Agreement metrics for weak-strong consistency analysis.

Computes similarity/disagreement between two sparse top-logprob score maps.
The paper's main analysis uses top-label agreement; these distributional
metrics are kept for auxiliary and legacy analyses.
"""

import numpy as np
from typing import Dict, Tuple

EPSILON = 1e-10


def align_distributions(
    pA: Dict[str, float],
    pB: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two probability distributions to a common label space.

    Missing labels get EPSILON probability, then renormalize.
    """
    all_labels = sorted(set(pA.keys()) | set(pB.keys()))
    pa = np.array([pA.get(lbl, EPSILON) for lbl in all_labels])
    pb = np.array([pB.get(lbl, EPSILON) for lbl in all_labels])
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return pa, pb


def js_divergence(
    pA: Dict[str, float],
    pB: Dict[str, float],
) -> float:
    """Jensen-Shannon Divergence between two distributions.

    JS(pA||pB) = 0.5*KL(pA||M) + 0.5*KL(pB||M), M = 0.5*(pA+pB)
    Range: [0, ln(2)]. Higher = more disagreement.
    """
    pa, pb = align_distributions(pA, pB)
    m = 0.5 * (pa + pb)
    kl_pa = np.sum(np.where(pa > 0, pa * np.log(pa / (m + EPSILON)), 0.0))
    kl_pb = np.sum(np.where(pb > 0, pb * np.log(pb / (m + EPSILON)), 0.0))
    return float(0.5 * kl_pa + 0.5 * kl_pb)


def cosine_similarity(
    pA: Dict[str, float],
    pB: Dict[str, float],
) -> float:
    """Cosine similarity between two distributions as vectors.

    Range: [0, 1]. Higher = more agreement.
    """
    pa, pb = align_distributions(pA, pB)
    dot_product = np.dot(pa, pb)
    norm_a = np.linalg.norm(pa)
    norm_b = np.linalg.norm(pb)
    return float(dot_product / (norm_a * norm_b + EPSILON))


def top_k_overlap(
    pA: Dict[str, float],
    pB: Dict[str, float],
    k: int = 3,
) -> float:
    """Overlap between top-K labels of two distributions.

    overlap = |topK(pA) ∩ topK(pB)| / K
    Range: [0, 1]. Higher = more agreement.
    """
    topk_a = sorted(pA.keys(), key=lambda x: pA.get(x, 0), reverse=True)[:k]
    topk_b = sorted(pB.keys(), key=lambda x: pB.get(x, 0), reverse=True)[:k]
    intersection = len(set(topk_a) & set(topk_b))
    return float(intersection) / k


def compute_agreement_metrics(
    pA: Dict[str, float],
    pB: Dict[str, float],
    top_k: int = 3,
) -> Dict[str, float]:
    """Compute all agreement metrics at once.

    Returns dict with js_divergence, cosine_similarity,
    top_k_overlap, top_k_rank_agreement.
    """
    return {
        "js_divergence": js_divergence(pA, pB),
        "cosine_similarity": cosine_similarity(pA, pB),
        "top_k_overlap": top_k_overlap(pA, pB, k=top_k),
    }
