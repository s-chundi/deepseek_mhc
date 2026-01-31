"""Evaluate the MHC model on GSM8K using lm-evaluation-harness."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7"

import argparse

import lm_eval
from lm_eval.models.huggingface import HFLM

from model.utils import get_qwen_model, get_best_device


def evaluate_gsm8k(
    model_path: str,
    batch_size: int = 4,
    num_fewshot: int = 0,
    limit: int | None = None,
    device: str | None = None,
):
    """
    Evaluate the model on GSM8K.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model ID.
        batch_size: Batch size for evaluation.
        num_fewshot: Number of few-shot examples (GSM8K standard is 5 or 8).
        limit: Limit number of examples (for debugging). None = full dataset.
        device: Device to run on. If None, automatically selects best available.
    """
    if device is None:
        device = get_best_device()

    tokenizer, model = get_qwen_model(checkpoint_path=model_path)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        max_length=2048, 
        gen_kwargs="temperature=0,max_gen_toks=1024"
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        num_fewshot=num_fewshot,
        limit=limit,
    )

    print("\n" + "=" * 60)
    print("GSM8K Results")
    print("=" * 60)

    gsm8k_results = results["results"]["gsm8k"]
    for metric, value in gsm8k_results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MHC model on GSM8K")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for debugging)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda, mps, cpu). Auto-detects if not specified.",
    )
    args = parser.parse_args()

    evaluate_gsm8k(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
