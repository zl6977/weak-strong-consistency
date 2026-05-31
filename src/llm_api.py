"""OpenAI-compatible LLM API client for the project's LLM server."""

import logging
import os
from typing import Dict, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"

LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8000/v1")

LLM_MODELS = [
    "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4",
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "cyankiwi/Qwen3.5-9B-AWQ-4bit",
]

LLM_MODEL_PROVIDERS = {
    "cyankiwi/gemma-4-31B-it-AWQ-4bit": LLM_SERVER_URL,
    "cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4": LLM_SERVER_URL,
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": LLM_SERVER_URL,
    "cyankiwi/Qwen3.5-9B-AWQ-4bit": LLM_SERVER_URL,
}

_CLIENTS: Dict[str, OpenAI] = {}


def _get_client(model: str) -> OpenAI:
    base_url = LLM_MODEL_PROVIDERS[model]
    if base_url in _CLIENTS:
        return _CLIENTS[base_url]

    _CLIENTS[base_url] = OpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=2,
        timeout=600,
    )
    return _CLIENTS[base_url]


def extract_response(response_data, model: str) -> dict:
    """Extract content and logprobs from an OpenAI-style chat completion response."""
    if response_data is None:
        return {}

    model_name = getattr(response_data, "model", None)
    usage = getattr(response_data, "usage", None)

    choices = getattr(response_data, "choices", None) or []
    choice = choices[0] if choices else None
    if choice is None:
        return {"model": model_name, "content": "", "logprobs": None}

    message = getattr(choice, "message", None)
    content_text = ""
    if message is not None:
        content = getattr(message, "content", None)
        content_text = content if isinstance(content, str) else ""

    return {
        "model": model_name,
        "content": content_text,
        "logprobs": getattr(choice, "logprobs", None),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }


def chat_with_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    user_config: Optional[dict] = None,
):
    """Send a chat completion request to the LLM server.

    Args:
        prompt: User message text.
        model: Model name (must be in LLM_MODELS).
        user_config: Extra kwargs passed to the API call (e.g. logprobs, temperature).

    Returns:
        Raw OpenAI chat completion response object.
    """
    if model not in LLM_MODELS:
        logger.error("Model %s is not registered in LLM_MODELS.", model)
        return None

    client = _get_client(model)

    data_config = {"temperature": 0.01}
    if user_config is not None:
        data_config.update(user_config)

    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        **data_config,
    )
