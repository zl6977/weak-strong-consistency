"""Reproducible consistency analysis from cached LLM runs.

The empirical role assignment used here is:
strong1 = Gemma 31B, strong2 = Qwen 27B, middle = Qwen 9B, weak = Qwen 35B.
The strong1 model remains the predictor for agreement-gate metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover - sklearn is in pyproject, but keep script portable.
    average_precision_score = None
    roc_auc_score = None

from src.agreement_metrics import compute_agreement_metrics
from src.run_task import _load_banking77_data, _load_clinc_oos_data


DATASET_CONFIGS = {
    "clinc": {
        "loader": lambda: _load_clinc_oos_data(split="plus"),
        "cache_root": Path("results/dual_cache/clinc"),
        "cache_file": "cache_plus.json",
        "output_dir": Path("results/analysis/clinc_metrics"),
        "role_runs": {
            "strong1": "gemma-4-31B-run0-all",
            "strong2": "Qwen3.6-27B-run0-all",
            "middle": "qwen3.5-9b-clinc-run0-all",
            "weak": "Qwen3.6-35B-run0-all",
        },
        "description": "CLINC plus test",
    },
    "bank77": {
        "loader": _load_banking77_data,
        "cache_root": Path("results/dual_cache/bank77"),
        "cache_file": "cache_banking77_test.json",
        "output_dir": Path("results/analysis/bank77_metrics"),
        "role_runs": {
            "strong1": "gemma-4-31B-banking77-run0",
            "strong2": "Qwen3.6-27B-banking77-run0",
            "middle": "Qwen3.5-9B-banking77-run0",
            "weak": "Qwen3.6-35B-A3B-banking77-run0",
        },
        "description": "BANKING77 test",
    },
}

PAIR_ORDER = [
    ("strong1", "strong2"),
    ("strong1", "middle"),
    ("strong1", "weak"),
]


def load_cache(cache_root: Path, run_dir: str, cache_file: str) -> dict[str, Any]:
    path = cache_root / run_dir / cache_file
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", data if isinstance(data, list) else [])
    if not isinstance(results, list):
        raise ValueError(f"Unexpected cache format in {path}")
    return {
        "run_dir": run_dir,
        "model": data.get("model", run_dir) if isinstance(data, dict) else run_dir,
        "results": results,
        "path": str(path),
    }


def top_label(result: dict[str, Any]) -> str | None:
    probs = result.get("probs") or {}
    if not probs:
        return None
    return max(probs, key=probs.get)


def top_confidence(result: dict[str, Any]) -> float:
    probs = result.get("probs") or {}
    if not probs:
        return float("nan")
    return float(max(probs.values()))


def safe_mean(values: list[bool] | np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def model_accuracy_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
    oos_label: str,
) -> list[dict[str, Any]]:
    rows = []
    gold_arr = np.array(gold)
    id_mask = gold_arr != oos_label
    oos_mask = gold_arr == oos_label

    for role, run in runs.items():
        preds = np.array([top_label(r) for r in run["results"]], dtype=object)
        correct = preds == gold_arr
        pred_oos = preds == oos_label
        tp_oos = int(np.sum(pred_oos & oos_mask))
        fp_oos = int(np.sum(pred_oos & id_mask))
        fn_oos = int(np.sum(~pred_oos & oos_mask))
        rows.append(
            {
                "role": role,
                "run": run["run_dir"],
                "model": run["model"],
                "n": len(gold),
                "accuracy": safe_mean(correct),
                "id_accuracy": safe_mean(correct[id_mask]),
                "oos_accuracy_recall": safe_mean(correct[oos_mask]),
                "oos_precision": safe_div(tp_oos, tp_oos + fp_oos),
                "oos_recall": safe_div(tp_oos, tp_oos + fn_oos),
                "pred_oos_rate": safe_mean(pred_oos),
            }
        )
    return rows


def model_accuracy_split_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
    oos_label: str,
) -> list[dict[str, Any]]:
    rows = []
    gold_arr = np.array(gold)
    split_masks = {
        "overall": np.ones(len(gold_arr), dtype=bool),
        "id": gold_arr != oos_label,
        "ood": gold_arr == oos_label,
    }

    for role, run in runs.items():
        preds = np.array([top_label(r) for r in run["results"]], dtype=object)
        correct = preds == gold_arr
        pred_oos = preds == oos_label
        for split, mask in split_masks.items():
            rows.append(
                {
                    "role": role,
                    "split": split,
                    "run": run["run_dir"],
                    "model": run["model"],
                    "n": int(np.sum(mask)),
                    "accuracy": safe_mean(correct[mask]),
                    "pred_ood_rate": safe_mean(pred_oos[mask]),
                }
            )
    return rows


def agreement_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
    oos_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reliability_rows = []
    split_rows = []
    score_rows = []

    gold_arr = np.array(gold)
    id_mask = gold_arr != oos_label
    oos_mask = gold_arr == oos_label

    strong1 = runs["strong1"]
    strong1_preds = np.array([top_label(r) for r in strong1["results"]], dtype=object)
    strong1_correct = strong1_preds == gold_arr
    strong1_conf = np.array([float(r.get("id_conf", float("nan"))) for r in strong1["results"]])

    for primary_role, validator_role in PAIR_ORDER:
        primary = runs[primary_role]
        validator = runs[validator_role]
        primary_preds = np.array([top_label(r) for r in primary["results"]], dtype=object)
        validator_preds = np.array([top_label(r) for r in validator["results"]], dtype=object)
        correct = primary_preds == gold_arr
        agree = primary_preds == validator_preds
        disagree = ~agree

        reliability_rows.append(
            {
                "pair": f"{primary_role}-{validator_role}",
                "primary_run": primary["run_dir"],
                "validator_run": validator["run_dir"],
                "coverage": safe_mean(agree),
                "acc_at_agree": safe_mean(correct[agree]),
                "acc_at_disagree": safe_mean(correct[disagree]),
                "reliability_gap": safe_mean(correct[agree]) - safe_mean(correct[disagree]),
                "error_rate_at_agree": 1.0 - safe_mean(correct[agree]),
                "error_rate_at_disagree": 1.0 - safe_mean(correct[disagree]),
                "error_capture_at_disagree": safe_div(
                    float(np.sum(disagree & ~correct)),
                    float(np.sum(~correct)),
                ),
                "n_agree": int(np.sum(agree)),
                "n_disagree": int(np.sum(disagree)),
            }
        )

        for split_name, split_mask in [("overall", np.ones_like(id_mask, dtype=bool)), ("id", id_mask), ("ood", oos_mask)]:
            split_agree = agree & split_mask
            split_disagree = disagree & split_mask
            split_correct = correct & split_mask
            split_rows.append(
                {
                    "pair": f"{primary_role}-{validator_role}",
                    "split": split_name,
                    "n": int(np.sum(split_mask)),
                    "coverage": safe_div(float(np.sum(split_agree)), float(np.sum(split_mask))),
                    "acc_at_agree": safe_mean(correct[split_agree]),
                    "acc_at_disagree": safe_mean(correct[split_disagree]),
                    "error_capture_at_disagree": safe_div(
                        float(np.sum(split_disagree & ~split_correct)),
                        float(np.sum(split_mask & ~correct)),
                    ),
                }
            )

        score = agree.astype(float)
        correctness_binary = strong1_correct.astype(int)
        score_rows.append(score_metric_row(f"{primary_role}-{validator_role}", "agreement_binary", correctness_binary, score))

        if np.all(np.isfinite(strong1_conf)):
            score_rows.append(score_metric_row(f"{primary_role}-{validator_role}", "strong1_id_conf", correctness_binary, strong1_conf))
            combined = 0.5 * strong1_conf + 0.5 * score
            score_rows.append(score_metric_row(f"{primary_role}-{validator_role}", "0.5_conf_plus_0.5_agree", correctness_binary, combined))

        js_scores = []
        cosine_scores = []
        top3_scores = []
        for p_result, v_result in zip(primary["results"], validator["results"], strict=True):
            metrics = compute_agreement_metrics(
                p_result.get("probs") or {},
                v_result.get("probs") or {},
                top_k=3,
            )
            js_scores.append(1.0 - min(metrics["js_divergence"] / math.log(2), 1.0))
            cosine_scores.append(metrics["cosine_similarity"])
            top3_scores.append(metrics["top_k_overlap"])
        score_rows.append(score_metric_row(f"{primary_role}-{validator_role}", "1-normalized_js", correctness_binary, np.array(js_scores)))
        score_rows.append(score_metric_row(f"{primary_role}-{validator_role}", "cosine_similarity", correctness_binary, np.array(cosine_scores)))
        score_rows.append(score_metric_row(f"{primary_role}-{validator_role}", "top3_overlap", correctness_binary, np.array(top3_scores)))

    return reliability_rows, split_rows, score_rows


def score_metric_row(pair: str, score_name: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    row = {"pair": pair, "score": score_name, "auroc_correctness": float("nan"), "auprc_correctness": float("nan")}
    if roc_auc_score is None or average_precision_score is None:
        return row
    if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
        return row
    row["auroc_correctness"] = float(roc_auc_score(labels, scores))
    row["auprc_correctness"] = float(average_precision_score(labels, scores))
    return row


def msp_baseline_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
) -> list[dict[str, Any]]:
    """For each coverage implied by an agreement gate, report MSP accepted accuracy.

    MSP gate: keep the top-K highest-confidence samples and report their accuracy.
    This produces a fair baseline for comparing against agreement gates at the
    same coverage.
    """
    strong1 = runs["strong1"]
    strong1_preds = np.array([top_label(r) for r in strong1["results"]], dtype=object)
    strong1_conf = np.array([top_confidence(r) for r in strong1["results"]])
    correct = strong1_preds == np.array(gold)

    # sort by confidence descending
    sorted_idx = np.argsort(-strong1_conf)
    sorted_correct = correct[sorted_idx]
    n = len(gold)

    # collect coverages from the agreement gates
    covs: list[float] = []
    for primary_role, validator_role in PAIR_ORDER:
        validator = runs[validator_role]
        validator_preds = np.array([top_label(r) for r in validator["results"]], dtype=object)
        agree = strong1_preds == validator_preds
        cov = float(np.mean(agree))
        covs.append(cov)
    covs = sorted(set(round(c, 4) for c in covs))

    rows = []
    for cov in covs:
        k = max(1, int(round(n * cov)))
        accepted_correct = sorted_correct[:k]
        rows.append(
            {
                "method": "msp",
                "coverage": cov,
                "accepted_accuracy": safe_mean(accepted_correct),
                "risk": 1.0 - safe_mean(accepted_correct),
                "n_accepted": k,
            }
        )
    return rows


def confidence_quadrant_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
    oos_label: str,
    confidence_quantile: float,
) -> list[dict[str, Any]]:
    rows = []
    gold_arr = np.array(gold)
    split_masks = {
        "overall": np.ones(len(gold_arr), dtype=bool),
        "id": gold_arr != oos_label,
        "ood": gold_arr == oos_label,
    }

    strong1 = runs["strong1"]
    strong1_preds = np.array([top_label(r) for r in strong1["results"]], dtype=object)
    strong1_correct = strong1_preds == gold_arr
    strong1_conf = np.array([float(r.get("id_conf", float("nan"))) for r in strong1["results"]])
    threshold = float(np.quantile(strong1_conf, confidence_quantile))
    high_conf = strong1_conf >= threshold

    for primary_role, validator_role in PAIR_ORDER:
        validator = runs[validator_role]
        validator_preds = np.array([top_label(r) for r in validator["results"]], dtype=object)
        agree = strong1_preds == validator_preds

        groups = [
            ("high", "agree", high_conf & agree),
            ("high", "disagree", high_conf & ~agree),
            ("low", "agree", ~high_conf & agree),
            ("low", "disagree", ~high_conf & ~agree),
        ]
        for split, split_mask in split_masks.items():
            for conf_group, agreement_group, mask in groups:
                group_mask = mask & split_mask
                accuracy = safe_mean(strong1_correct[group_mask])
                rows.append(
                    {
                        "pair": f"{primary_role}-{validator_role}",
                        "split": split,
                        "confidence_threshold_quantile": confidence_quantile,
                        "confidence_threshold": threshold,
                        "confidence": conf_group,
                        "agreement": agreement_group,
                        "n": int(np.sum(group_mask)),
                        "coverage": safe_div(float(np.sum(group_mask)), float(np.sum(split_mask))),
                        "accuracy": accuracy,
                        "error_rate": 1.0 - accuracy if not math.isnan(accuracy) else float("nan"),
                    }
                )
    return rows


def confidence_penalty_rows(quadrant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key = {}
    for row in quadrant_rows:
        key = (row["pair"], row["split"], row["confidence"])
        by_key[(key, row["agreement"])] = row

    keys = sorted({key for key, agreement in by_key if agreement in {"agree", "disagree"}})
    for pair, split, confidence in keys:
        agree = by_key.get(((pair, split, confidence), "agree"))
        disagree = by_key.get(((pair, split, confidence), "disagree"))
        if agree is None or disagree is None:
            continue
        rows.append(
            {
                "pair": pair,
                "split": split,
                "confidence": confidence,
                "agree_accuracy": agree["accuracy"],
                "disagree_accuracy": disagree["accuracy"],
                "accuracy_penalty": agree["accuracy"] - disagree["accuracy"],
                "agree_error_rate": agree["error_rate"],
                "disagree_error_rate": disagree["error_rate"],
                "error_rate_increase": disagree["error_rate"] - agree["error_rate"],
                "agree_coverage": agree["coverage"],
                "disagree_coverage": disagree["coverage"],
                "agree_n": agree["n"],
                "disagree_n": disagree["n"],
            }
        )
    return rows


def _build_agreement_vs_msp_table(
    reliability_rows: list[dict[str, Any]],
    msp_rows: list[dict[str, Any]],
) -> str:
    """Build a side-by-side comparison: agreement gate vs MSP at the same coverage."""
    msp_by_cov = {row["coverage"]: row["accepted_accuracy"] for row in msp_rows}
    cols = ["pair", "coverage", "acc_at_agree", "msp_acc", "delta"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in reliability_rows:
        cov = round(row["coverage"], 4)
        msp_acc = msp_by_cov.get(cov)
        delta = row["acc_at_agree"] - msp_acc if msp_acc is not None else float("nan")
        body.append("| " + " | ".join([
            row["pair"],
            fmt(cov),
            fmt(row["acc_at_agree"]),
            fmt(msp_acc),
            fmt(delta),
        ]) + " |")
    return "\n".join([header, sep, *body])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(fmt(row.get(col)) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_markdown_report(
    path: Path,
    dataset: str,
    dataset_description: str,
    model_rows: list[dict[str, Any]],
    model_split_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    quadrant_rows: list[dict[str, Any]],
    penalty_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    msp_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# {dataset.upper()} Consistency Metrics",
        "",
        f"Dataset: {dataset_description}.",
        "",
        "Role assignment: strong1 = Gemma31B, strong2 = Qwen27B, middle = Qwen9B, weak = Qwen35B.",
        "",
        "## Single Model Accuracy",
        markdown_table(
            model_rows,
            ["role", "run", "accuracy", "id_accuracy", "oos_accuracy_recall", "oos_precision", "pred_oos_rate"],
        ),
        "",
        "## Single Model Accuracy by Split",
        markdown_table(
            model_split_rows,
            ["role", "split", "n", "accuracy", "pred_ood_rate", "run"],
        ),
        "",
        "## Agreement Reliability",
        markdown_table(
            reliability_rows,
            [
                "pair",
                "coverage",
                "acc_at_agree",
                "acc_at_disagree",
                "reliability_gap",
                "error_capture_at_disagree",
                "n_agree",
                "n_disagree",
            ],
        ),
        "",
        "## ID/OOD Decomposition",
        markdown_table(
            split_rows,
            ["pair", "split", "n", "coverage", "acc_at_agree", "acc_at_disagree", "error_capture_at_disagree"],
        ),
        "",
        "## Confidence x Agreement",
        markdown_table(
            [row for row in quadrant_rows if row["split"] == "overall"],
            ["pair", "split", "confidence", "agreement", "n", "coverage", "accuracy", "error_rate"],
        ),
        "",
        "## Confidence x Agreement by Split",
        markdown_table(
            quadrant_rows,
            ["pair", "split", "confidence", "agreement", "n", "coverage", "accuracy", "error_rate"],
        ),
        "",
        "## Confidence Agreement Penalty",
        markdown_table(
            penalty_rows,
            [
                "pair",
                "split",
                "confidence",
                "agree_accuracy",
                "disagree_accuracy",
                "accuracy_penalty",
                "error_rate_increase",
                "agree_n",
                "disagree_n",
            ],
        ),
        "",
        "## Correctness Ranking Scores",
        markdown_table(
            score_rows,
            ["pair", "score", "auroc_correctness", "auprc_correctness"],
        ),
        "",
        "## MSP Baseline (Selective Prediction)",
        "For each coverage point used by an agreement gate, MSP reports the accepted accuracy of a confidence-only gate.",
        "",
        markdown_table(
            msp_rows,
            ["method", "coverage", "accepted_accuracy", "risk", "n_accepted"],
        ),
        "",
        "## Agreement vs MSP Comparison",
        _build_agreement_vs_msp_table(reliability_rows, msp_rows),
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), default="clinc")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--oos-label", default="oos")
    parser.add_argument(
        "--confidence-quantile",
        type=float,
        default=0.5,
        help="Strong-model id_conf quantile used to split high/low confidence.",
    )
    return parser.parse_args()


def _msp_comparison_table(reliability_rows: list[dict[str, Any]], msp_rows: list[dict[str, Any]]) -> str:
    """Console-friendly comparison of agreement vs MSP at the same coverage."""
    msp_by_cov = {row["coverage"]: row["accepted_accuracy"] for row in msp_rows}
    cols = ["pair", "coverage", "acc@agree", "msp_acc", "delta"]
    return markdown_table(
        [
            {
                "pair": r["pair"],
                "coverage": round(r["coverage"], 4),
                "acc@agree": r["acc_at_agree"],
                "msp_acc": msp_by_cov.get(round(r["coverage"], 4)),
                "delta": r["acc_at_agree"] - msp_by_cov.get(round(r["coverage"], 4), float("nan")),
            }
            for r in reliability_rows
        ],
        cols,
    )


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    cache_root = args.cache_root or config["cache_root"]
    cache_file = args.cache_file or config["cache_file"]
    output_dir = args.output_dir or config["output_dir"]

    _, gold, _, _, _ = config["loader"]()
    runs = {
        role: load_cache(cache_root, run_dir, cache_file)
        for role, run_dir in config["role_runs"].items()
    }

    for role, run in runs.items():
        if len(run["results"]) != len(gold):
            raise ValueError(
                f"Length mismatch for {role} ({run['run_dir']}): "
                f"{len(run['results'])} cache rows vs {len(gold)} gold labels"
            )

    model_rows = model_accuracy_rows(runs, gold, args.oos_label)
    model_split_rows = model_accuracy_split_rows(runs, gold, args.oos_label)
    reliability_rows, split_rows, score_rows = agreement_rows(runs, gold, args.oos_label)
    quadrant_rows = confidence_quadrant_rows(runs, gold, args.oos_label, args.confidence_quantile)
    penalty_rows = confidence_penalty_rows(quadrant_rows)
    msp_rows = msp_baseline_rows(runs, gold)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "single_model_accuracy.csv", model_rows)
    write_csv(output_dir / "single_model_accuracy_by_split.csv", model_split_rows)
    write_csv(output_dir / "agreement_reliability.csv", reliability_rows)
    write_csv(output_dir / "agreement_reliability_by_split.csv", split_rows)
    write_csv(output_dir / "agreement_id_oos_decomposition.csv", split_rows)
    write_csv(output_dir / "confidence_agreement_quadrants.csv", quadrant_rows)
    write_csv(output_dir / "confidence_agreement_quadrants_by_split.csv", quadrant_rows)
    write_csv(output_dir / "confidence_agreement_penalty.csv", penalty_rows)
    write_csv(output_dir / "correctness_ranking_scores.csv", score_rows)
    write_csv(output_dir / "msp_baseline.csv", msp_rows)
    write_markdown_report(
        output_dir / "report.md",
        args.dataset,
        config["description"],
        model_rows,
        model_split_rows,
        reliability_rows,
        split_rows,
        quadrant_rows,
        penalty_rows,
        score_rows,
        msp_rows,
    )

    print("\nSingle model accuracy")
    print(markdown_table(model_rows, ["role", "run", "accuracy", "id_accuracy", "oos_accuracy_recall", "oos_precision"]))
    print("\nAgreement reliability")
    print(markdown_table(reliability_rows, ["pair", "coverage", "acc_at_agree", "acc_at_disagree", "reliability_gap", "error_capture_at_disagree"]))
    print("\nAgreement reliability by split")
    print(markdown_table(split_rows, ["pair", "split", "n", "coverage", "acc_at_agree", "acc_at_disagree", "error_capture_at_disagree"]))
    print("\nRQ3 high-confidence agree/disagree penalty")
    high_penalty_rows = [row for row in penalty_rows if row["confidence"] == "high" and row["split"] == "overall"]
    print(markdown_table(high_penalty_rows, ["pair", "split", "agree_accuracy", "disagree_accuracy", "accuracy_penalty", "error_rate_increase", "agree_n", "disagree_n"]))
    print(f"\nWrote outputs to {output_dir}")
    print("\nAgreement vs MSP (same coverage)")
    print(_msp_comparison_table(reliability_rows, msp_rows))


if __name__ == "__main__":
    main()
