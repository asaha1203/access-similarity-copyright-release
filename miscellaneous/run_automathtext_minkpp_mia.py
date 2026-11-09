#!/usr/bin/env python
"""Run MinK%++ MIA for a trained AutoMathText LoRA checkpoint.

This is a convenience wrapper around the text MIA pipeline. It scores
AutoMathText members from train_full and nonmembers from
mi_reserve_not_in_training, then runs the standard MIA analyzer.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(SRC_ROOT))

from copyright_pipeline.constants import PROCESSED_DATA_ROOT  # noqa: E402
from copyright_pipeline.inference_tasks import run_analyze_mia, run_score_mia_text  # noqa: E402


def _resolve_model_dir(path: Optional[str], dataset: str, prefix: str) -> Path:
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if not (candidate / "training_config.yaml").exists():
            raise FileNotFoundError(f"Expected training_config.yaml under model directory: {candidate}")
        return candidate

    run_root = PROJECT_ROOT / "runs" / dataset
    matches = [
        item
        for item in run_root.glob(f"{prefix}*")
        if item.is_dir() and (item / "training_config.yaml").exists()
    ]
    if not matches:
        raise FileNotFoundError(f"No run matching {run_root / (prefix + '*')}")
    return max(matches, key=lambda item: (item / "training_config.yaml").stat().st_mtime)


def _default_limit(dataset: str, member_split: str, nonmember_split: str) -> int:
    member_ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset / member_split))
    nonmember_ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset / nonmember_split))
    return max(len(member_ds), len(nonmember_ds))


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MinK%++ membership inference against an AutoMathText text LoRA run."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=(
            "Trained run directory, or an adapter file inside it. Defaults to the latest "
            "runs/automathtext/baseline_full_lora* directory."
        ),
    )
    parser.add_argument("--dataset", type=str, default="automathtext")
    parser.add_argument("--member-split", type=str, default="train_full")
    parser.add_argument("--nonmember-split", type=str, default="mi_reserve_not_in_training")
    parser.add_argument("--run-prefix", type=str, default="baseline_full_lora")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Samples per split. Defaults to all available data.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for text MIA scoring.")
    parser.add_argument(
        "--max-sequence-tokens",
        type=int,
        default=2048,
        help="Maximum tokenized length per member/nonmember sequence before MIA scoring.",
    )
    parser.add_argument(
        "--equalize-class-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score equal numbers of member and nonmember samples (default: true).",
    )
    parser.add_argument(
        "--text-scoring-mode",
        type=str,
        default="prompt_target",
        choices=["prompt_target", "full_text_chunked"],
        help="How to score text for MIA; prompt_target keeps per-token MinK++ artifacts in records.",
    )
    parser.add_argument(
        "--chunk-token-size",
        type=int,
        default=2048,
        help="Token chunk size for full_text_chunked scoring.",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=1,
        help="Number of text chunks to score per forward pass in full_text_chunked mode.",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default=None,
        choices=["fp16", "bf16"],
        help="Convenience mixed-precision mode; overrides --text-dtype when set.",
    )
    parser.add_argument(
        "--text-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Torch dtype for loading text models on CUDA.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--k-percent-grid", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    parser.add_argument("--default-k-percent", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--fpr-grid", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_text_dtype = str(args.precision_mode or args.text_dtype)
    if args.default_k_percent not in args.k_percent_grid:
        raise ValueError("--default-k-percent must be one of --k-percent-grid values.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be >= 1")
    if int(args.max_sequence_tokens) < 2:
        raise ValueError("--max-sequence-tokens must be >= 2")
    if int(args.chunk_token_size) < 2:
        raise ValueError("--chunk-token-size must be >= 2")
    if int(args.chunk_batch_size) <= 0:
        raise ValueError("--chunk-batch-size must be >= 1")

    model_dir = _resolve_model_dir(args.model_dir, args.dataset, args.run_prefix)
    limit = args.limit or _default_limit(args.dataset, args.member_split, args.nonmember_split)

    score_config = {
        "project": {
            "run_name": f"minkpp_mia_{model_dir.name}",
            "output_root": str(args.output_root),
            "seed": int(args.seed),
        },
        "dataset": {
            "name": args.dataset,
            "modality": "text",
            "member_split": args.member_split,
            "nonmember_split": args.nonmember_split,
        },
        "models": {
            "target": {
                "checkpoint_dir": str(model_dir),
            },
        },
        "cpk": {
            "enabled": False,
        },
        "mia": {
            "limit": int(limit),
            "batch_size": int(args.batch_size),
            "max_sequence_tokens": int(args.max_sequence_tokens),
            "text_scoring_mode": str(args.text_scoring_mode),
            "chunk_token_size": int(args.chunk_token_size),
            "chunk_batch_size": int(args.chunk_batch_size),
            "equalize_class_counts": bool(args.equalize_class_counts),
        },
        "minkpp": {
            "k_percent_grid": [int(k) for k in args.k_percent_grid],
            "default_k_percent": int(args.default_k_percent),
            "epsilon": float(args.epsilon),
        },
        "runtime": {
            "device": args.device,
            "text_dtype": runtime_text_dtype,
        },
    }

    config_dir = PROJECT_ROOT / "miscellaneous" / "generated_configs"
    score_config_path = config_dir / f"{args.dataset}_minkpp_mia_score.yaml"
    _write_yaml(score_config_path, score_config)

    score_dir = run_score_mia_text(score_config)

    analyze_config = {
        "project": {
            "run_name": f"analyze_{score_dir.parent.name}",
            "dataset_name": args.dataset,
            "output_root": str(args.output_root),
        },
        "inputs": {
            "records_path": str(score_dir / "records.jsonl"),
        },
        "mia_analysis": {
            "fpr_grid": [float(x) for x in args.fpr_grid],
        },
        "cpk": {
            "enabled": False,
            "alpha_grid": [1.0],
        },
    }
    analyze_config_path = config_dir / f"{args.dataset}_minkpp_mia_analyze.yaml"
    _write_yaml(analyze_config_path, analyze_config)

    analyze_dir = run_analyze_mia(analyze_config)

    print(f"Model directory: {model_dir}")
    print(f"Scoring config: {score_config_path}")
    print(f"Scored records: {score_dir / 'records.jsonl'}")
    print(f"Analysis config: {analyze_config_path}")
    print(f"Analysis metrics: {analyze_dir / 'metrics' / 'mia_summary.json'}")


if __name__ == "__main__":
    main()
