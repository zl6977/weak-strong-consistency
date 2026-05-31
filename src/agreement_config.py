"""Configuration for weak-strong agreement inference."""

from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """Settings needed to generate cached model predictions."""

    model_A: str = "Qwen/Qwen3.6-27B"
    model_B: str = "Qwen/Qwen3.5-9B"
    num_runs: int = 5
    options_placeholder: str = "<labels>"
    option_explanations_placeholder: str = "<descriptions>"
    id_score_weights: list = field(default_factory=lambda: [0.7, 0.1, 0.2])
