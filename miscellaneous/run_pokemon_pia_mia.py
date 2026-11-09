#!/usr/bin/env python
"""Run PIA-style diffusion MIA for a trained Pokemon LoRA checkpoint.

This is a convenience wrapper around the project's inference MIA pipeline. It
scores Pokemon member images from train_full and nonmembers from
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
from copyright_pipeline.inference_tasks import run_analyze_mia, run_score_mia_diffusion  # noqa: E402


def _infer_member_split(run_name: str) -> str:
    name = run_name.lower()
    if "shard1" in name:
        return "shard1"
    if "shard2" in name:
        return "shard2"
    return "train_full"


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


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def _resolve_checkpoint_dir(model_dir: Path, checkpoint_index: Optional[int]) -> Path:
    if checkpoint_index is None:
        return model_dir

    checkpoints = [
        child
        for child in model_dir.glob("checkpoint-*")
        if child.is_dir() and child.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found under: {model_dir}")

    checkpoints = sorted(checkpoints, key=_checkpoint_step)
    num_checkpoints = len(checkpoints)
    if checkpoint_index < -num_checkpoints or checkpoint_index >= num_checkpoints:
        raise IndexError(
            f"checkpoint_index={checkpoint_index} out of range for {num_checkpoints} checkpoints "
            f"(valid: {-num_checkpoints}..{num_checkpoints - 1})"
        )
    return checkpoints[checkpoint_index]


def _default_limit(dataset: str, member_split: str, nonmember_split: str) -> int:
    member_ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset / member_split))
    nonmember_ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset / nonmember_split))
    return max(len(member_ds), len(nonmember_ds))


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PIA-style membership inference against a Pokemon diffusion LoRA run."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=(
            "Trained run directory, or an adapter file inside it. Defaults to the latest "
            "runs/pokemon/baseline_full_lora* directory."
        ),
    )
    parser.add_argument("--dataset", type=str, default="pokemon")
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        default=None,
        help=(
            "Optional index into sorted checkpoint-* directories under --model-dir "
            "(ascending by step number; supports negative indexing like -1 for latest)."
        ),
    )
    parser.add_argument(
        "--member-split",
        type=str,
        default=None,
        help="Override member split. Defaults by model: shard1->shard1, shard2->shard2, else train_full.",
    )
    parser.add_argument("--nonmember-split", type=str, default="mi_reserve_not_in_training")
    parser.add_argument("--run-prefix", type=str, default="baseline_full_lora")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Samples per split. Defaults to all available data.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--timestep-grid", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--norm-p", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="pipeline",
        choices=["pipeline", "reference_linear"],
        help=(
            "PIA noising schedule. 'pipeline' uses the loaded scheduler's training "
            "alphas/add_noise; 'reference_linear' keeps the legacy hard-coded "
            "linear beta schedule."
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--precision-mode",
        type=str,
        default=None,
        choices=["fp16", "bf16"],
        help="Convenience mixed-precision mode; overrides --diffusion-dtype when set.",
    )
    parser.add_argument("--diffusion-dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--strict-adapter-load", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_diffusion_dtype = str(args.precision_mode or args.diffusion_dtype)
    run_dir = _resolve_model_dir(args.model_dir, args.dataset, args.run_prefix)
    model_dir = _resolve_checkpoint_dir(run_dir, args.checkpoint_index)
    member_split = args.member_split or _infer_member_split(run_dir.name)
    limit = args.limit or _default_limit(args.dataset, member_split, args.nonmember_split)
    run_name_suffix = model_dir.name if model_dir != run_dir else run_dir.name

    score_config = {
        "project": {
            "run_name": f"pia_mia_{run_name_suffix}",
            "output_root": str(args.output_root),
        },
        "dataset": {
            "name": args.dataset,
            "modality": "image",
            "member_split": member_split,
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
            "limit": limit,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
        },
        "pia": {
            "norm_p": args.norm_p,
            "timestep_grid": args.timestep_grid,
            "default_timestep": args.timestep,
            "guidance_scale": args.guidance_scale,
            "beta_schedule": args.beta_schedule,
        },
        "runtime": {
            "device": args.device,
            "strict_adapter_load": args.strict_adapter_load,
            "amp": bool(args.amp),
            "diffusion_dtype": runtime_diffusion_dtype,
        },
    }

    config_dir = PROJECT_ROOT / "miscellaneous" / "generated_configs"
    score_config_path = config_dir / f"{args.dataset}_pia_mia_score.yaml"
    _write_yaml(score_config_path, score_config)

    score_dir = run_score_mia_diffusion(score_config)

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
            "fpr_grid": [0.01, 0.05, 0.1],
        },
        "cpk": {
            "enabled": False,
            "alpha_grid": [1.0],
        },
    }
    analyze_config_path = config_dir / f"{args.dataset}_pia_mia_analyze.yaml"
    _write_yaml(analyze_config_path, analyze_config)

    analyze_dir = run_analyze_mia(analyze_config)

    if model_dir != run_dir:
        print(f"Run directory: {run_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Member split: {member_split}")
    print(f"Scoring config: {score_config_path}")
    print(f"Scored records: {score_dir / 'records.jsonl'}")
    print(f"Analysis config: {analyze_config_path}")
    print(f"Analysis metrics: {analyze_dir / 'metrics' / 'mia_summary.json'}")


if __name__ == "__main__":
    main()
