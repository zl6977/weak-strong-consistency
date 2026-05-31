"""Runner for cached LLM classification and legacy agreement scoring."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from . import configs as cfg
from . import logprob_classifier as lc
from .agreement_config import StrategyConfig
from .agreement_metrics import compute_agreement_metrics
from .legacy.confidence_fusion import fuse_confidence, FusionConfig
from .legacy.ood_evaluator import evaluate_ood_detection, evaluate_per_class, generate_plots

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _run_dir(output_label: str) -> str:
    """Directory for a single run's output (cache + logs)."""
    return os.path.join(cfg.dual_cache_dir, output_label)


def _model_cache_path(output_label: str, split: str = "plus") -> str:
    """Cache file inside the run directory."""
    d = _run_dir(output_label)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"cache_{split}.json")


def _next_output_label(model: str, split: str = "plus") -> str:
    """Find the next available output label for a model. e.g. Qwen3.6-27B-run0"""
    import glob as g
    short = model.split("/")[-1].split(".")[0]  # e.g. "Qwen3.6-27B"
    pattern = os.path.join(cfg.dual_cache_dir, f"{short}-run*", f"cache_{split}.json")
    existing = []
    for fp in g.glob(pattern):
        dirname = os.path.basename(os.path.dirname(fp))
        try:
            idx_str = dirname.split("-run")[1]
            existing.append(int(idx_str))
        except (IndexError, ValueError):
            pass
    return f"{short}-run{max(existing, default=-1) + 1}"


def _load_cached_run(output_label: str, split: str = "plus") -> tuple[str, List[Dict]]:
    """Load cached run, returning (model_name, results_list)."""
    cache_path = _model_cache_path(output_label, split)
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        model = data.get("model", "")
        results = data.get("results", [])
    else:
        model = ""
        results = data
    return model, results


def ranked_to_flat_probs(ranked_options: List[Dict]) -> Dict[str, float]:
    """Convert [{label: {average, ...}}] to {label: average}."""
    flat = {}
    for item in ranked_options:
        for label, data in item.items():
            flat[label] = data["average"]
    return flat


# -------------------------------------------------------------------
# single-sample inference for ONE model
# -------------------------------------------------------------------

def run_single_model_sample(
    sample: str,
    all_labels: List[str],
    label_descriptions: Dict[str, str],
    config: StrategyConfig,
    model: str,
    prompt_template: Optional[str] = None,
) -> Dict:
    if prompt_template is None:
        prompt_template = lc.PROMPT_TEMPLATES["Rank"]["ranking"]

    ranked = lc.rank_options_by_llm_probabilities(
        prompt_template=prompt_template,
        options=all_labels,
        options_placeholder=config.options_placeholder,
        option_explanations=label_descriptions,
        option_explanations_placeholder=config.option_explanations_placeholder,
        kvPairs={"<sample>": sample},
        num_runs=config.num_runs,
        model=model,
    )
    probs = ranked_to_flat_probs(ranked)
    id_conf = lc.compute_id_score(
        matching_scores=probs,
        need_softmax=False,
        weights=config.id_score_weights,
    )
    return {"ranked": ranked, "probs": probs, "id_conf": id_conf}


# -------------------------------------------------------------------
# phase runner (all samples, with checkpointing)
# -------------------------------------------------------------------

def run_phase(
    model: str,
    sample_list: List[str],
    all_labels: List[str],
    label_descriptions: Dict[str, str],
    config: StrategyConfig,
    split: str = "plus",
    prompt_template: Optional[str] = None,
    output_label: str = "",
) -> List[Dict]:
    """Run a single model on every sample, with save/resume checkpointing."""
    if not output_label:
        output_label = _next_output_label(model, split)
    cache_path = _model_cache_path(output_label, split)
    total = len(sample_list)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        # Support both old format (list) and new format (dict with model + results)
        if isinstance(cached_data, dict):
            cached = cached_data.get("results", cached_data.get("run_results", []))
        else:
            cached = cached_data
        completed = len(cached)
        logger.info(f"Resuming [{output_label}]: {completed}/{total} done")
    else:
        cached = []
        completed = 0

    results = list(cached)

    for i in range(completed, total):
        sample = sample_list[i]
        logger.info(f"[{i + 1}/{total}]: {sample}")
        entry = run_single_model_sample(
            sample=sample,
            all_labels=all_labels,
            label_descriptions=label_descriptions,
            config=config,
            model=model,
            prompt_template=prompt_template,
        )
        results.append(entry)

        if (i + 1) % 10 == 0 or i + 1 == total:
            cache = {"model": model, "results": results}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            logger.info(f"  checkpoint at {i + 1}/{total}")

    return results


# -------------------------------------------------------------------
# legacy score fusion + evaluation
# -------------------------------------------------------------------

def fuse_and_evaluate(
    sample_list: List[str],
    intent_list: List[str],
    results_A: List[Dict],
    results_B: List[Dict],
    config: StrategyConfig,
    split: str = "plus",
) -> Dict:
    """Combine cached runs for the legacy score-based evaluation path."""
    assert len(results_A) == len(results_B) == len(sample_list)

    records = []
    for i in range(len(sample_list)):
        agreement = compute_agreement_metrics(
            results_A[i]["probs"], results_B[i]["probs"],
            top_k=config.agreement_top_k,
        )
        fusion_cfg = FusionConfig(
            agreement_metrics_weights=config.agreement_metrics_weights,
            agreement_weight=config.agreement_weight,
            confidence_weight=config.confidence_weight,
            gamma=config.gamma,
            strategy=config.strategy,
        )
        fusion = fuse_confidence(
            results_A[i]["id_conf"], results_B[i]["id_conf"],
            agreement, fusion_cfg,
        )
        predicted_intent = max(results_A[i]["probs"], key=results_A[i]["probs"].get)

        records.append({
            "sample": sample_list[i],
            "ground_truth_intent": intent_list[i],
            "model_A": {"model_name": config.model_A, **results_A[i]},
            "model_B": {"model_name": config.model_B, **results_B[i]},
            "agreement": agreement,
            "fusion": fusion,
            "predicted_intent": predicted_intent,
        })

    has_ood = any(gt == config.ood_label for gt in intent_list)
    if has_ood:
        metrics = evaluate_ood_detection(
            records=records, ground_truths=intent_list,
            ood_label=config.ood_label, ood_score_key="fusion.ood_score",
        )
        per_class = evaluate_per_class(
            records=records, ground_truths=intent_list, ood_label=config.ood_label,
        )

        plot_dir = os.path.join(cfg.plot_dir, f"{config.strategy}_eval")
        plots = generate_plots(
            records=records, ground_truths=intent_list,
            ood_label=config.ood_label, output_dir=plot_dir,
        )
    else:
        metrics = {
            "skipped": True,
            "reason": f"No OOD label '{config.ood_label}' found in ground truth labels.",
            "per_threshold_pr": [],
        }
        per_class = {}
        plots = []

    logger.info("=" * 60)
    logger.info(f"Strategy: {config.strategy}")
    logger.info(f"Model A:  {config.model_A}")
    logger.info(f"Model B:  {config.model_B}")
    if metrics.get("skipped"):
        logger.info(f"OOD evaluation skipped: {metrics['reason']}")
    else:
        logger.info(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:   {metrics['pr_auc']:.4f}")
        logger.info(f"EPOC:     {metrics['epoc']:.4f}")
        logger.info(f"FPR95:    {metrics['fpr95']:.4f}")
    logger.info("=" * 60)

    return _save_results(records, metrics, per_class, plots, config)


def _save_results(records, metrics, per_class, plots, config):
    output = {
        "strategy": config.strategy,
        "model_A": config.model_A,
        "model_B": config.model_B,
        "metrics": {k: v for k, v in metrics.items() if k != "per_threshold_pr"},
        "per_threshold_pr": metrics["per_threshold_pr"],
        "per_class": per_class,
        "plot_files": plots,
        "records": records,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.fusion_dir, f"{config.strategy}_{timestamp}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {out_path}")
    return metrics
