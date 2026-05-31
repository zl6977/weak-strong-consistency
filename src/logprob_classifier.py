"""Logprob-based classifier: extracts probability distributions from LLM logprobs."""

import copy
import logging
import math
import os
import random
import string
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tiktoken
import yaml

from . import llm_api as llm

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

PROMPT_TEMPLATES: Dict[str, Any] = {}
with open(os.path.join(_PROJECT_ROOT, "prompt_resources", "templates_generic.yaml"), "r") as file:
    PROMPT_TEMPLATES = yaml.safe_load(file)


# -------------------------------------------------------------------
# prompt assembly
# -------------------------------------------------------------------

def assemble_prompt(prompt_template: str, kvPairs: dict) -> str:
    """Replace placeholders in a prompt template with actual values."""
    prompt = prompt_template
    for keyword, value in kvPairs.items():
        if value is not None and len(value) > 0:
            prompt = prompt.replace(keyword, str(value))
    return prompt


# -------------------------------------------------------------------
# tokenizer helpers
# -------------------------------------------------------------------

_TOKENIZERS_CACHE: Dict[str, Any] = {}


def _get_encoding() -> tiktoken.Encoding:
    if "_cl100k_" not in _TOKENIZERS_CACHE:
        _TOKENIZERS_CACHE["_cl100k_"] = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZERS_CACHE["_cl100k_"]


def get_token_id(characters: str, model: str = llm.DEFAULT_MODEL) -> list:
    """Encode text to token IDs using cl100k_base."""
    return _get_encoding().encode(characters)


def check_is_single_token(text: str, model: str = llm.DEFAULT_MODEL) -> bool:
    """Check if a string encodes as exactly one token."""
    try:
        return len(get_token_id(text, model)) == 1
    except Exception:
        return False


def generate_single_token_letters(num_needed: int, model: str = llm.DEFAULT_MODEL) -> list:
    """Generate single-token letter combos (A..Z, AA..ZZ, AAA..) for label mapping."""
    letters: list[str] = []
    for first in string.ascii_uppercase:
        if check_is_single_token(first, model):
            letters.append(first)
        if len(letters) >= num_needed:
            return letters[:num_needed]
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            combo = first + second
            if check_is_single_token(combo, model):
                letters.append(combo)
            if len(letters) >= num_needed:
                return letters[:num_needed]
    for first in string.ascii_uppercase:
        for second in string.ascii_uppercase:
            for third in string.ascii_uppercase:
                combo = first + second + third
                if check_is_single_token(combo, model):
                    letters.append(combo)
                if len(letters) >= num_needed:
                    return letters[:num_needed]
    raise ValueError(f"Could not generate {num_needed} single-token letter combos for {model}")


def create_letter_mapping(candidate_list: list, model: str = llm.DEFAULT_MODEL, candidate_limit: int = 300) -> tuple:
    """Map candidates to single-token letter tokens. Returns (letter_map, reverse_map)."""
    if len(candidate_list) > candidate_limit:
        logger.warning(f"Candidate list exceeded {candidate_limit}, truncating.")
        candidate_list = candidate_list[:candidate_limit]
    candidate_letters = generate_single_token_letters(len(candidate_list), model)
    letter_map = {cand: candidate_letters[i] for i, cand in enumerate(candidate_list)}
    reverse_map = {v: k for k, v in letter_map.items()}
    return letter_map, reverse_map


def shuffle_candidates_unique(candidate_list: list, previous_shuffles: list) -> tuple:
    """Shuffle candidates, ensuring the order hasn't appeared before."""
    shuffled = copy.deepcopy(candidate_list)
    for _ in range(10):
        random.shuffle(shuffled)
        if shuffled not in previous_shuffles:
            return True, shuffled
    return False, shuffled


# -------------------------------------------------------------------
# probability extraction from one LLM call
# -------------------------------------------------------------------

def query_llm_for_probability_distribution(
    prompt_template: str,
    options: Union[list, tuple],
    options_placeholder: str,
    option_explanations: dict,
    option_explanations_placeholder: str,
    kvPairs: dict,
    model: Optional[str] = None,
    user_config: Optional[dict] = None,
) -> dict:
    """Query the LLM once and return a {label: probability} dict via logprobs.

    Labels are mapped to single-token letters so the model emits one token,
    and we read the top-k logprobs to get a distribution.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    letter_map, reverse_map = create_letter_mapping(options, model=model)
    options_labeled = [f"{letter}: {cand}" for cand, letter in letter_map.items()]

    kvPairs = kvPairs.copy()
    kvPairs[options_placeholder] = "\n".join(options_labeled)
    kvPairs[option_explanations_placeholder] = "\n".join(
        f"{key}: {value}" for key, value in option_explanations.items()
    )
    prompt = assemble_prompt(prompt_template, kvPairs)
    logger.debug(f"Prompt: {prompt}")

    config = user_config.copy() if user_config else {}
    config.update({
        "logprobs": True,
        "temperature": 0.0,
        "top_logprobs": 5,
        "max_completion_tokens": 1,
    })

    response = llm.chat_with_llm(prompt=prompt, model=model, user_config=config)
    result = llm.extract_response(response, model=model)

    token_list = result["logprobs"].content
    assert len(token_list) == 1, "LLM returned more than one token."

    top_logprobs = token_list[0].top_logprobs
    probs = {
        reverse_map[item.token]: math.exp(item.logprob)
        for item in top_logprobs
        if item.token in reverse_map
    }
    logger.debug(f"probs_distribution: {probs}")
    return probs


# -------------------------------------------------------------------
# multi-run aggregation (shuffles + averaging)
# -------------------------------------------------------------------

def _aggregate_probability_distributions(
    probs_list: list, sort_flag: bool = True
) -> list:
    """Average multiple probability dicts into [{label: {average, history}}]."""
    history_dict: Dict[str, list] = {}
    for entry in probs_list:
        for key, value in entry.items():
            history_dict.setdefault(key, []).append(value)

    agg = [
        {key: {"average": sum(values) / len(values), "history": values}}
        for key, values in history_dict.items()
    ]
    if sort_flag:
        agg.sort(key=lambda x: list(x.values())[0]["average"], reverse=True)
    return agg


def rank_options_by_llm_probabilities(
    prompt_template: str,
    options: Union[list, tuple],
    options_placeholder: str,
    option_explanations: dict,
    option_explanations_placeholder: str,
    kvPairs: dict,
    num_runs: int = 3,
    model: Optional[str] = None,
    user_config: Optional[dict] = None,
) -> list:
    """Rank options by querying the LLM multiple times with shuffled labels.

    Returns aggregated [{label: {average, history}}] sorted by average descending.
    """
    if model is None:
        model = llm.DEFAULT_MODEL

    previous_shuffles: list = []
    probs_list: list = []

    for run_num in range(num_runs):
        is_unique, shuffled = shuffle_candidates_unique(list(options), previous_shuffles)
        if not is_unique:
            logger.warning(f"Run {run_num + 1}: failed to find unique shuffle.")
            continue
        previous_shuffles.append(shuffled)

        probs = query_llm_for_probability_distribution(
            prompt_template=prompt_template,
            options=shuffled,
            options_placeholder=options_placeholder,
            option_explanations=option_explanations,
            option_explanations_placeholder=option_explanations_placeholder,
            kvPairs=kvPairs,
            model=model,
            user_config=user_config,
        )
        probs_list.append(probs)
        logger.debug(f"Run {run_num + 1} distribution: {probs}")

    return _aggregate_probability_distributions(probs_list)


# -------------------------------------------------------------------
# confidence scoring
# -------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def compute_id_score(
    matching_scores: dict,
    need_softmax: bool = True,
    non_id_labels: set | None = None,
    weights: list = [0.7, 0.1, 0.2],
) -> float:
    """Compute in-distribution confidence from a label->score dict.

    Confidence is a weighted geometric mean of MSP, Margin, and Sharpness
    computed over the ID subset of labels (excluding oos/uncertain/etc).
    """
    if len(weights) != 3:
        raise ValueError("weights must be length 3: (w_msp, w_margin, w_sharp)")

    w = np.asarray(weights, dtype=float)
    w = w / (np.sum(w) + 1e-12)
    w_msp, w_margin, w_sharp = map(float, w)

    labels = list(matching_scores.keys())
    vals = np.array(list(matching_scores.values()), dtype=float)

    p = _softmax(vals) if need_softmax else (vals / vals.sum() if abs(vals.sum() - 1.0) > 0.001 else vals)

    if non_id_labels is None:
        nonid_set = {"OUT-OF-SET", "UNCERTAIN", "OOS"}
    else:
        nonid_set = {str(lbl).upper() for lbl in non_id_labels}

    id_mask = np.array([str(lbl).upper() not in nonid_set for lbl in labels], dtype=bool)
    p_id = p[id_mask]
    K = int(np.sum(id_mask))

    # MSP
    msp = float(np.max(p_id)) if K > 0 else 0.0

    # Margin (top-1 - top-2)
    if K <= 1:
        margin = 1.0 if K == 1 else 0.0
    else:
        top2 = np.sort(p_id)[-2:]
        margin = float(top2[1] - top2[0])

    # Sharpness (1 - normalized entropy)
    if K <= 1 or p_id.sum() <= 1e-12:
        sharpness = 1.0
    else:
        pid_cond = p_id / (p_id.sum() + 1e-12)
        H = -np.sum(np.where(pid_cond > 0, pid_cond * np.log(pid_cond), 0.0))
        H_norm = float(H / (np.log(K + 1e-12) + 1e-12))
        sharpness = 1.0 - H_norm

    eps = 1e-9
    id_conf = float(np.clip(
        np.exp(w_msp * np.log(max(msp, eps)) + w_margin * np.log(max(margin, eps)) + w_sharp * np.log(max(sharpness, eps))),
        0.0, 1.0,
    ))
    return id_conf
