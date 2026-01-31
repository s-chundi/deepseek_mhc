# MHC Qwen3

Custom Qwen3 model with hyperconnection parameters for residual stream weighting.

## Overview

This project extends `Qwen/Qwen3-0.6B` with additional learnable parameters (`residual_stream_weights`) that control how information flows through the residual stream. The current implementation initializes these weights to preserve the original model behavior.

## Setup

```bash
uv sync
```

## Quickstart

1. SFT the model (needed to initialize the custom parameters):

```bash
uv run src/model/train.py
```

This loads `Qwen/Qwen3-0.6B`, initializes the custom parameters, and saves based on `config.yaml`

2. Evaluate on GSM8K:

```bash
uv run src/model/eval_gsm8k.py [path to model checkpoint]
```

## Project Structure

```
src/model/
  configuration_mhc_q3.py  # Custom Qwen3Config with hyperconnection_dim
  modeling_mhc_q3.py       # Custom Qwen3 model with residual stream weights
  utils.py                 # Model loading and dataset utilities
  train.py                 # SFT the model
  posttrain.py             # Posttrain the model
  eval_gsm8k.py            # Evaluate on GSM8K benchmark
```
