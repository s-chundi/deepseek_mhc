# MHC Qwen3

Custom Qwen3 model with hyperconnection parameters for residual stream weighting.

## Overview

This project extends `Qwen/Qwen3-0.6B` with additional learnable parameters (`residual_stream_weights`) that control how information flows through the residual stream. The current implementation initializes these weights to preserve the original model behavior.

## Setup

```bash
uv sync
```

## Quickstart

1. Save the model with custom parameters:

```bash
cd src
uv run python -m model.train # Just saves the model at this stage, doesn't actually train anything
```

This loads `Qwen/Qwen3-0.6B`, initializes the custom parameters, and saves to `./checkpoints/initial_model`.

2. Evaluate on GSM8K:

```bash
uv run python -m model.eval_gsm8k ./checkpoints/initial_model
```

Options:
- `--batch-size N` - Batch size (default: 4)
- `--num-fewshot 0` - Few-shot examples (default: 0)
- `--limit N` - Limit examples for debugging
- `--device DEVICE` - cuda, mps, or cpu (default: mps)

## Project Structure

```
src/model/
  configuration_mhc_q3.py  # Custom Qwen3Config with hyperconnection_dim
  modeling_mhc_q3.py       # Custom Qwen3 model with residual stream weights
  utils.py                 # Model loading and dataset utilities
  train.py                 # Save initial model checkpoint
  eval_gsm8k.py            # Evaluate on GSM8K benchmark
```
