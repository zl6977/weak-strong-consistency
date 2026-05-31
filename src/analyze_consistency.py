"""Reproduce the paper analysis from cached LLM top-label runs.

The empirical role assignment used here is:
strong1 = Gemma 31B, strong2 = Qwen 27B, middle = Qwen 9B, weak = Qwen 35B.
The strong1 model remains the predictor; the other models are validators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src import configs as cfg
from src.run_task import _load_banking77_data, _load_clinc_oos_data


DATASET_CONFIGS = {
    "clinc": {
        "loader": lambda: _load_clinc_oos_data(split="plus"),
        "cache_root": Path(cfg.dual_cache_dir) / "clinc",
        "cache_file": "cache_plus.json",
        "output_dir": Path(cfg.analysis_dir) / "clinc_metrics",
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
        "cache_root": Path(cfg.dual_cache_dir) / "bank77",
        "cache_file": "cache_banking77_test.json",
        "output_dir": Path(cfg.analysis_dir) / "bank77_metrics",
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


def agreement_by_split_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
    oos_label: str,
) -> list[dict[str, Any]]:
    rows = []
    gold_arr = np.array(gold)
    id_mask = gold_arr != oos_label
    oos_mask = gold_arr == oos_label

    for primary_role, validator_role in PAIR_ORDER:
        primary = runs[primary_role]
        validator = runs[validator_role]
        primary_preds = np.array([top_label(r) for r in primary["results"]], dtype=object)
        validator_preds = np.array([top_label(r) for r in validator["results"]], dtype=object)
        correct = primary_preds == gold_arr
        agree = primary_preds == validator_preds

        for split_name, split_mask in [
            ("overall", np.ones_like(id_mask, dtype=bool)),
            ("id", id_mask),
            ("ood", oos_mask),
        ]:
            split_agree = agree & split_mask
            split_disagree = ~agree & split_mask
            split_correct = correct & split_mask
            acc_at_agree = safe_mean(correct[split_agree])
            acc_at_disagree = safe_mean(correct[split_disagree])
            rows.append(
                {
                    "pair": f"{primary_role}-{validator_role}",
                    "split": split_name,
                    "n": int(np.sum(split_mask)),
                    "coverage": safe_div(float(np.sum(split_agree)), float(np.sum(split_mask))),
                    "acc_at_agree": acc_at_agree,
                    "acc_at_disagree": acc_at_disagree,
                    "reliability_gap": acc_at_agree - acc_at_disagree,
                    "error_capture_at_disagree": safe_div(
                        float(np.sum(split_disagree & ~split_correct)),
                        float(np.sum(split_mask & ~correct)),
                    ),
                    "n_agree": int(np.sum(split_agree)),
                    "n_disagree": int(np.sum(split_disagree)),
                }
            )
    return rows


def high_confidence_agreement_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
    confidence_quantile: float,
) -> list[dict[str, Any]]:
    """BANKING77 RQ3 table: agreement inside the high-confidence subset.

    Confidence is the cached top-label score: max(result["probs"].values()).
    This is a gray-box diagnostic; the main agreement results remain black-box
    top-label agreement results.
    """
    gold_arr = np.array(gold)
    strong1 = runs["strong1"]
    strong1_preds = np.array([top_label(r) for r in strong1["results"]], dtype=object)
    strong1_correct = strong1_preds == gold_arr
    strong1_conf = np.array([top_confidence(r) for r in strong1["results"]])
    threshold = float(np.quantile(strong1_conf, confidence_quantile))
    high_conf = strong1_conf >= threshold

    rows = []
    for primary_role, validator_role in PAIR_ORDER:
        validator = runs[validator_role]
        validator_preds = np.array([top_label(r) for r in validator["results"]], dtype=object)
        agree = strong1_preds == validator_preds
        agree_mask = high_conf & agree
        disagree_mask = high_conf & ~agree
        agree_accuracy = safe_mean(strong1_correct[agree_mask])
        disagree_accuracy = safe_mean(strong1_correct[disagree_mask])
        rows.append(
            {
                "pair": f"{primary_role}-{validator_role}",
                "confidence_subset": "high",
                "confidence_threshold_quantile": confidence_quantile,
                "confidence_score": "top_label_score",
                "confidence_threshold": threshold,
                "agree_accuracy": agree_accuracy,
                "disagree_accuracy": disagree_accuracy,
                "accuracy_penalty": agree_accuracy - disagree_accuracy,
                "agree_n": int(np.sum(agree_mask)),
                "disagree_n": int(np.sum(disagree_mask)),
            }
        )
    return rows


def msp_baseline_rows(
    runs: dict[str, dict[str, Any]],
    gold: list[str],
) -> list[dict[str, Any]]:
    """BANKING77 diagnostic baseline: MSP accepted accuracy at agreement coverages."""
    strong1 = runs["strong1"]
    strong1_preds = np.array([top_label(r) for r in strong1["results"]], dtype=object)
    strong1_conf = np.array([top_confidence(r) for r in strong1["results"]])
    correct = strong1_preds == np.array(gold)

    sorted_idx = np.argsort(-strong1_conf)
    sorted_correct = correct[sorted_idx]
    n = len(gold)

    coverages = []
    for _, validator_role in PAIR_ORDER:
        validator = runs[validator_role]
        validator_preds = np.array([top_label(r) for r in validator["results"]], dtype=object)
        coverages.append(float(np.mean(strong1_preds == validator_preds)))

    rows = []
    for coverage in sorted(set(round(c, 4) for c in coverages)):
        k = max(1, int(round(n * coverage)))
        accepted_accuracy = safe_mean(sorted_correct[:k])
        rows.append(
            {
                "method": "msp",
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "risk": 1.0 - accepted_accuracy,
                "n_accepted": k,
            }
        )
    return rows


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
    agreement_rows: list[dict[str, Any]],
    high_conf_rows: list[dict[str, Any]],
    msp_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# {dataset.upper()} Paper Metrics",
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
        "## Agreement-Conditioned Accuracy by Split",
        markdown_table(
            agreement_rows,
            [
                "pair",
                "split",
                "n",
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
    ]
    if high_conf_rows:
        parts.extend(
            [
                "## BANKING77 High-Confidence Complementarity",
                markdown_table(
                    high_conf_rows,
                    ["pair", "agree_accuracy", "disagree_accuracy", "agree_n", "disagree_n"],
                ),
                "",
            ]
        )
    if msp_rows:
        parts.extend(
            [
                "## BANKING77 MSP Baseline",
                markdown_table(
                    msp_rows,
                    ["method", "coverage", "accepted_accuracy", "risk", "n_accepted"],
                ),
                "",
            ]
        )
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
        help="Strong-model top-label score quantile used for the BANKING77 high-confidence diagnostic.",
    )
    return parser.parse_args()


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("*.csv"):
        path.unlink()
    report = output_dir / "report.md"
    if report.exists():
        report.unlink()


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
    agreement_rows = agreement_by_split_rows(runs, gold, args.oos_label)
    high_conf_rows = []
    msp_rows = []
    if args.dataset == "bank77":
        high_conf_rows = high_confidence_agreement_rows(runs, gold, args.confidence_quantile)
        msp_rows = msp_baseline_rows(runs, gold)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)
    write_csv(output_dir / "single_model_accuracy.csv", model_rows)
    write_csv(output_dir / "agreement_reliability_by_split.csv", agreement_rows)
    write_csv(output_dir / "confidence_agreement_penalty.csv", high_conf_rows)
    write_csv(output_dir / "msp_baseline.csv", msp_rows)
    write_markdown_report(
        output_dir / "report.md",
        args.dataset,
        config["description"],
        model_rows,
        agreement_rows,
        high_conf_rows,
        msp_rows,
    )

    print("\nSingle model accuracy")
    print(markdown_table(model_rows, ["role", "run", "accuracy", "id_accuracy", "oos_accuracy_recall", "oos_precision"]))
    print("\nAgreement-conditioned accuracy by split")
    print(markdown_table(agreement_rows, ["pair", "split", "n", "coverage", "acc_at_agree", "acc_at_disagree", "error_capture_at_disagree"]))
    if high_conf_rows:
        print("\nBANKING77 high-confidence complementarity")
        print(markdown_table(high_conf_rows, ["pair", "agree_accuracy", "disagree_accuracy", "agree_n", "disagree_n"]))
    if msp_rows:
        print("\nBANKING77 MSP baseline")
        print(markdown_table(msp_rows, ["method", "coverage", "accepted_accuracy", "risk", "n_accepted"]))
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
