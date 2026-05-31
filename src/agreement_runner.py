"""Runner for cached LLM classification."""

import json
import logging
import os
from typing import Dict, List, Optional

from . import configs as cfg
from . import logprob_classifier as lc
from .agreement_config import StrategyConfig

logger = logging.getLogger(__name__)


def _run_dir(output_label: str) -> str:
    """Directory for a single run's output cache and logs."""
    return os.path.join(cfg.dual_cache_dir, output_label)


def _model_cache_path(output_label: str, split: str = "plus") -> str:
    """Cache file inside the run directory."""
    run_dir = _run_dir(output_label)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, f"cache_{split}.json")


def _next_output_label(model: str, split: str = "plus") -> str:
    """Find the next available output label, e.g. Qwen3.6-27B-run0."""
    import glob

    short = model.split("/")[-1].split(".")[0]
    pattern = os.path.join(cfg.dual_cache_dir, f"{short}-run*", f"cache_{split}.json")
    existing = []
    for path in glob.glob(pattern):
        dirname = os.path.basename(os.path.dirname(path))
        try:
            existing.append(int(dirname.split("-run")[1]))
        except (IndexError, ValueError):
            pass
    return f"{short}-run{max(existing, default=-1) + 1}"


def ranked_to_flat_probs(ranked_options: List[Dict]) -> Dict[str, float]:
    """Convert [{label: {average, ...}}] to {label: average}."""
    flat = {}
    for item in ranked_options:
        for label, data in item.items():
            flat[label] = data["average"]
    return flat


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
    """Run one model on every sample, with save/resume checkpointing."""
    if not output_label:
        output_label = _next_output_label(model, split)
    cache_path = _model_cache_path(output_label, split)
    total = len(sample_list)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        cached = cached_data.get("results", cached_data.get("run_results", [])) if isinstance(cached_data, dict) else cached_data
        completed = len(cached)
        logger.info(f"Resuming [{output_label}]: {completed}/{total} done")
    else:
        cached = []
        completed = 0

    results = list(cached)
    for i in range(completed, total):
        sample = sample_list[i]
        logger.info(f"[{i + 1}/{total}]: {sample}")
        results.append(
            run_single_model_sample(
                sample=sample,
                all_labels=all_labels,
                label_descriptions=label_descriptions,
                config=config,
                model=model,
                prompt_template=prompt_template,
            )
        )

        if (i + 1) % 10 == 0 or i + 1 == total:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"model": model, "results": results}, f, ensure_ascii=False)
            logger.info(f"  checkpoint at {i + 1}/{total}")

    return results
