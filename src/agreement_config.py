"""Configuration for weak-strong agreement experiments."""

from dataclasses import dataclass, field
@dataclass
class StrategyConfig:
    """Configuration shared by cached inference and legacy scoring utilities."""

    strategy: str = "high_low"
    model_A: str = "Qwen/Qwen3.6-27B-Instruct"
    model_B: str = "Qwen/Qwen3.5-9B-Instruct"

    num_runs: int = 5

    options_placeholder: str = "<labels>"
    option_explanations_placeholder: str = "<descriptions>"

    id_score_weights: list = field(default_factory=lambda: [0.7, 0.1, 0.2])

    gamma: float = 1.0

    agreement_top_k: int = 3
    agreement_metrics_weights: dict = field(
        default_factory=lambda: {
            "js_divergence": 0.4,
            "cosine_similarity": 0.3,
            "top_k_overlap": 0.3,
        }
    )

    agreement_weight: float = 0.5
    confidence_weight: float = 0.5

    ood_label: str = "oos"

    @classmethod
    def same_model(cls, model: str = "Qwen/Qwen3.5-9B-Instruct", **kwargs) -> "StrategyConfig":
        return cls(
            strategy="same_model",
            model_A=model,
            model_B=model,
            agreement_weight=kwargs.get("agreement_weight", 0.4),
            confidence_weight=kwargs.get("confidence_weight", 0.6),
            **{k: v for k, v in kwargs.items() if k not in {"agreement_weight", "confidence_weight"}},
        )

    @classmethod
    def high_low(cls, strong: str = "Qwen/Qwen3.6-27B-Instruct", weak: str = "Qwen/Qwen3.5-9B-Instruct", **kwargs) -> "StrategyConfig":
        return cls(
            strategy="high_low",
            model_A=strong,
            model_B=weak,
            agreement_weight=kwargs.get("agreement_weight", 0.6),
            confidence_weight=kwargs.get("confidence_weight", 0.4),
            **{k: v for k, v in kwargs.items() if k not in {"agreement_weight", "confidence_weight"}},
        )

    @classmethod
    def two_high(cls, strong_a: str = "Qwen/Qwen3.6-27B-Instruct", strong_b: str = "google/Gemma-4-31B-Instruct", **kwargs) -> "StrategyConfig":
        return cls(
            strategy="two_high",
            model_A=strong_a,
            model_B=strong_b,
            agreement_weight=kwargs.get("agreement_weight", 0.4),
            confidence_weight=kwargs.get("confidence_weight", 0.6),
            **{k: v for k, v in kwargs.items() if k not in {"agreement_weight", "confidence_weight"}},
        )
