"""
Legacy evaluation suite for score-based OOD experiments.

Computes full metric suite: ROC-AUC, PR-AUC, EPOC, FPR95,
per-threshold PR, per-class analysis, and generates plots.
"""

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .. import configs as cfg


def compute_epoc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    """EPOC: Extended Area Under the Primary Curve (Huang et al. 2021).

    Primary curve: at threshold t, classify as ID if score < t, as OOD if score >= t.
    TPR = P(score < t | ID), FPR = P(score < t | OOD).
    EPOC = integral of (1-FPR) d(TPR). Higher is better (1.0 = perfect).
    """
    all_scores = np.concatenate([id_scores, ood_scores])
    thresholds = np.sort(np.unique(all_scores))
    if len(thresholds) < 2:
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)

    fprs = []
    tprs = []

    for t in thresholds:
        tpr = (id_scores < t).sum() / max(len(id_scores), 1)
        fpr = (ood_scores < t).sum() / max(len(ood_scores), 1)
        tprs.append(tpr)
        fprs.append(fpr)

    if len(fprs) < 2:
        return 0.5

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    sort_idx = np.argsort(tprs)
    fprs = fprs[sort_idx]
    tprs = tprs[sort_idx]

    epoc = float(np.trapezoid(1 - fprs, tprs))
    return float(np.clip(epoc, 0.0, 1.0))


def evaluate_ood_detection(
    records: List[Dict],
    ground_truths: List[str],
    ood_label: str = "oos",
    ood_score_key: str = "fusion.ood_score",
) -> Dict[str, float]:
    """Full evaluation suite for legacy score-based OOD experiments.

    Args:
        records: Result dicts from agreement_runner.
        ground_truths: Ground truth labels.
        ood_label: Label that indicates OOD samples.
        ood_score_key: Dot-separated key to extract score (e.g., "fusion.ood_score").

    Returns dict of metrics.
    """
    ood_scores = []
    for r in records:
        score = r
        for part in ood_score_key.split("."):
            score = score[part]
        ood_scores.append(score)
    ood_scores = np.array(ood_scores)

    # 1=OOD, 0=ID (positive class = OOD)
    labels_binary = np.array([1 if gt == ood_label else 0 for gt in ground_truths])

    ood_mask = labels_binary == 1
    id_mask = labels_binary == 0
    ood_scores_arr = ood_scores[ood_mask]
    id_scores_arr = ood_scores[id_mask]

    metrics = {}

    # ROC-AUC
    if len(np.unique(labels_binary)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels_binary, ood_scores))
    else:
        metrics["roc_auc"] = float("nan")

    # PR-AUC
    metrics["pr_auc"] = float(average_precision_score(labels_binary, ood_scores))

    # EPOC
    metrics["epoc"] = compute_epoc(id_scores_arr, ood_scores_arr)

    # FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(labels_binary, ood_scores)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    metrics["fpr95"] = float(fpr[idx_95])

    # Accuracy at median threshold
    median_threshold = np.median(ood_scores)
    predictions = (ood_scores >= median_threshold).astype(int)  # >= threshold -> OOD (1)
    metrics["accuracy"] = float((predictions == labels_binary).mean())

    id_preds = predictions[id_mask]
    ood_preds = predictions[ood_mask]
    metrics["id_accuracy"] = float((id_preds == 0).mean()) if len(id_preds) > 0 else 0.0
    metrics["ood_accuracy"] = float((ood_preds == 1).mean()) if len(ood_preds) > 0 else 0.0

    # Per-threshold PR
    precision, recall, pr_thresholds = precision_recall_curve(labels_binary, ood_scores)
    pr_data = []
    for i in range(len(pr_thresholds)):
        f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)
        pr_data.append({
            "threshold": float(pr_thresholds[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1),
        })
    metrics["per_threshold_pr"] = pr_data

    # Confidence distribution stats
    metrics["id_conf_mean"] = float(np.mean(id_scores_arr))
    metrics["id_conf_std"] = float(np.std(id_scores_arr))
    metrics["ood_conf_mean"] = float(np.mean(ood_scores_arr))
    metrics["ood_conf_std"] = float(np.std(ood_scores_arr))

    return metrics


def evaluate_per_class(
    records: List[Dict],
    ground_truths: List[str],
    ood_label: str = "oos",
    ood_score_key: str = "fusion.ood_score",
) -> Dict[str, Dict]:
    """Per-class legacy OOD score performance.

    For each ID class, treat its samples as positive and all others as negative.
    """
    ood_scores = []
    for r in records:
        score = r
        for part in ood_score_key.split("."):
            score = score[part]
        ood_scores.append(score)
    ood_scores = np.array(ood_scores)

    id_classes = sorted(set(gt for gt in ground_truths if gt != ood_label))
    per_class = {}

    for cls in id_classes:
        binary = np.array([1 if gt == cls else 0 for gt in ground_truths])
        if len(np.unique(binary)) < 2:
            continue
        per_class[cls] = {
            "roc_auc": float(roc_auc_score(binary, ood_scores)),
            "pr_auc": float(average_precision_score(binary, ood_scores)),
            "n_samples": int(binary.sum()),
            "mean_score": float(np.mean(ood_scores[binary == 1])),
        }

    return per_class


def generate_plots(
    records: List[Dict],
    ground_truths: List[str],
    ood_label: str = "oos",
    ood_score_key: str = "fusion.ood_score",
    output_dir: str | None = None,
) -> List[str]:
    """Generate evaluation plots.

    Returns list of file paths created.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.plot_dir, "evaluation_plots")
    os.makedirs(output_dir, exist_ok=True)

    ood_scores = []
    for r in records:
        score = r
        for part in ood_score_key.split("."):
            score = score[part]
        ood_scores.append(score)
    ood_scores = np.array(ood_scores)
    labels_binary = np.array([1 if gt == ood_label else 0 for gt in ground_truths])

    files = []

    # 1. ROC Curve
    if len(np.unique(labels_binary)) > 1:
        fpr, tpr, _ = roc_curve(labels_binary, ood_scores)
        roc_auc = roc_auc_score(labels_binary, ood_scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        path = os.path.join(output_dir, "roc_curve.png")
        fig.savefig(path)
        plt.close(fig)
        files.append(path)

    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(labels_binary, ood_scores)
    pr_auc = average_precision_score(labels_binary, ood_scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    path = os.path.join(output_dir, "pr_curve.png")
    fig.savefig(path)
    plt.close(fig)
    files.append(path)

    # 3. Score distribution (ID vs OOD)
    id_scores = ood_scores[labels_binary == 0]
    ood_scores_arr = ood_scores[labels_binary == 1]
    fig, ax = plt.subplots()
    ax.hist(id_scores, bins=50, alpha=0.5, label="ID", density=True)
    ax.hist(ood_scores_arr, bins=50, alpha=0.5, label="OOD", density=True)
    ax.set_xlabel("OOD Score")
    ax.set_ylabel("Density")
    ax.legend()
    path = os.path.join(output_dir, "score_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    files.append(path)

    # 4. Calibration reliability diagram
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_predicted = []
    bin_actual = []
    for i in range(n_bins):
        mask = (ood_scores >= bin_edges[i]) & (ood_scores < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_predicted.append(bin_edges[i + 1])
            bin_actual.append(labels_binary[mask].mean())
    fig, ax = plt.subplots()
    ax.bar(bin_centers, bin_actual, width=1 / n_bins, alpha=0.6, label="Actual OOD rate")
    ax.plot(bin_centers, bin_predicted, "r-o", label="Predicted OOD score")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("OOD Score Bin")
    ax.set_ylabel("OOD Rate")
    ax.legend()
    path = os.path.join(output_dir, "calibration.png")
    fig.savefig(path)
    plt.close(fig)
    files.append(path)

    return files
