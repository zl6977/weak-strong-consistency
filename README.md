# Weak-Strong Consistency as an Easiness Probe for LLM Text Classification

Code and cached outputs for the paper "Weak Models Reveal Easy Samples: Weak--Strong Consistency as an Easiness Probe for Large Language Model Text Classification".

The experiments ask whether agreement between a strong LLM classifier and weaker LLM classifiers can identify easy, reliable samples. Across CLINC-OOS and BANKING77, agreement is used as a selective prediction signal: samples where the strong model agrees with another model are treated as easier, while disagreement marks higher-risk examples.

## Key Results

Agreement with a weaker model gives a simple reliability gate for the strong model.

| Dataset   | Pair            | Coverage | Accuracy when agreeing | Accuracy when disagreeing |    Gap |
| --------- | --------------- | -------: | ---------------------: | ------------------------: | -----: |
| CLINC-OOS | strong1-strong2 |   0.7767 |                 0.9228 |                    0.6482 | 0.2745 |
| CLINC-OOS | strong1-middle  |   0.6745 |                 0.9197 |                    0.7408 | 0.1789 |
| CLINC-OOS | strong1-weak    |   0.6115 |                 0.9260 |                    0.7599 | 0.1660 |
| BANKING77 | strong1-strong2 |   0.8825 |                 0.8536 |                    0.4088 | 0.4447 |
| BANKING77 | strong1-middle  |   0.6416 |                 0.8639 |                    0.6893 | 0.1746 |
| BANKING77 | strong1-weak    |   0.6211 |                 0.8714 |                    0.6864 | 0.1850 |

Cached model outputs and regenerated analysis tables are included under `results/`.

## Quick Start

```bash
uv sync

# Reproduce CLINC-OOS analysis from cached model outputs.
uv run python -m src.analyze_consistency --dataset clinc

# Reproduce BANKING77 analysis from cached model outputs.
uv run python -m src.analyze_consistency --dataset bank77
```

The generated CSV files and Markdown reports are written to:

- `results/analysis/clinc_metrics/`
- `results/analysis/bank77_metrics/`

See `REPRODUCTION.md` for full inference and analysis commands.

## Repository Layout

```text
src/
  analyze_consistency.py    Rebuilds paper analysis tables from cached runs
  run_task.py               CLI for cached inference and legacy score fusion
  agreement_runner.py       Checkpointed inference and legacy score fusion
  agreement_config.py       Model-pair and agreement settings
  agreement_metrics.py      Auxiliary agreement metrics between score maps
  llm_api.py                OpenAI-compatible local LLM client
  legacy/                   Score-based OOD utilities from earlier experiments
dataset/
  clinc_oos/                Local CLINC-OOS parquet files
  banking77/                Local BANKING77 parquet files
results/
  dual_cache/               Cached LLM inference results and run logs
  analysis/                 Reproduced CSV metrics and Markdown reports
  fusion/                   Legacy score-fusion JSON outputs
  plots/                    Legacy score-fusion plots
latex/                      Paper source and conference template files
prompt_resources/           Prompt templates
```

## Running New Inference

Inference expects an OpenAI-compatible chat completions server. By default, the client uses:

```text
http://127.0.0.1:8000/v1
```

Override it with:

```bash
export LLM_SERVER_URL=http://127.0.0.1:8000/v1
```

Example:

```bash
uv run python -m src.run_task \
  --dataset clinc_oos \
  --model cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4 \
  --output Qwen3.6-27B-run1 \
  --quick 2
```

New run caches are written to `results/dual_cache/<output>/`.

## Citation

```bibtex
@misc{weak_strong_consistency_2026,
  title = {Weak Models Reveal Easy Samples: Weak--Strong Consistency as an Easiness Probe for Large Language Model Text Classification},
  year = {2026},
  note = {Code release}
}
```
