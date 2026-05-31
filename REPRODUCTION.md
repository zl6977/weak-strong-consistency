# Reproduction Guide

This repository supports two reproduction paths:

1. Rebuild the paper analysis tables from included cached model outputs.
2. Run new inference against a local OpenAI-compatible LLM server, then analyze the new caches.

## Environment

```bash
uv sync
```

Python 3.12 or newer is required.

## Data

The repository includes local parquet copies of the CLINC-OOS and BANKING77 splits used by the scripts. Dataset provenance and license notes are in `dataset/README.md`.

## Reproduce Analysis from Cached Outputs

```bash
uv run python -m src.analyze_consistency --dataset clinc
uv run python -m src.analyze_consistency --dataset bank77
```

Expected outputs:

```text
results/analysis/clinc_metrics/
results/analysis/bank77_metrics/
```

Each directory contains CSV metric tables and a `report.md` summary.

## Run New Inference

Start an OpenAI-compatible local server that exposes `/v1/chat/completions`. The default base URL is:

```text
http://127.0.0.1:8000/v1
```

To use a different server:

```bash
export LLM_SERVER_URL=http://127.0.0.1:8000/v1
```

Example single-model cache run:

```bash
uv run python -m src.run_task \
  --dataset clinc_oos \
  --model cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4 \
  --output Qwen3.6-27B-run1
```

For a quick smoke test:

```bash
uv run python -m src.run_task \
  --dataset clinc_oos \
  --model cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4 \
  --output Qwen3.6-27B-smoke \
  --quick 2
```

Run caches and logs are written to:

```text
results/dual_cache/<output>/
```

## Legacy Score Fusion for Two Cached Runs

```bash
uv run python -m src.run_task \
  --dataset clinc_oos \
  --fuse high_low \
  --run-A Qwen3.6-27B-run1 \
  --run-B Qwen3.6-35B-run1
```

Legacy score-fusion summaries are written under `results/fusion/`, evaluation plots are written under `results/plots/`, and any run-specific logs are kept beside the cache directories under `results/dual_cache/`. This path is not required for the paper's black-box top-label agreement analysis.

## Included Cached Runs

The included paper caches are organized by dataset:

```text
results/dual_cache/clinc/
results/dual_cache/bank77/
```

Each cached record includes `ranked`, `probs`, and `id_conf`. The `id_conf` field is a legacy score-based OOD confidence used only by `src/legacy/`; the paper analysis uses black-box top labels derived from `ranked`/`probs` and is not affected by `id_conf`.

The analysis script assigns model roles as:

```text
strong1 = Gemma 31B
strong2 = Qwen 27B
middle  = Qwen 9B
weak    = Qwen 35B
```
