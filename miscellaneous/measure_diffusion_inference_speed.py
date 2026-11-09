#!/usr/bin/env python
"""Benchmark diffusion-model inference speed for a trained run directory."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_diffusion_model  # noqa: E402


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(device: str, dtype_name: str) -> Optional[torch.dtype]:
    name = dtype_name.lower()
    if name == "auto":
        if device.startswith("cuda"):
            return torch.float16
        return None
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --dtype value: {dtype_name}")


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return (1.0 - frac) * sorted_vals[lo] + frac * sorted_vals[hi]


def _load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompt_file is not None:
        raw_lines = args.prompt_file.read_text(encoding="utf-8").splitlines()
        prompts.extend([line.strip() for line in raw_lines if line.strip()])
    if args.prompt:
        prompts.extend([p.strip() for p in args.prompt if p.strip()])
    if not prompts:
        prompts = ["a high quality photo of a mountain at sunrise"]
    return prompts


def _build_batch_prompts(prompts: List[str], batch_size: int, batch_idx: int) -> List[str]:
    out: List[str] = []
    start = batch_idx * batch_size
    for j in range(batch_size):
        out.append(prompts[(start + j) % len(prompts)])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure diffusion-model inference speed.")
    parser.add_argument(
        "--model-run-dir",
        type=Path,
        required=True,
        help="Run directory containing training_config.yaml (+ adapter_config.json or unet/ if finetuned).",
    )
    parser.add_argument("--device", type=str, default=_default_device(), help="Torch device (e.g. cuda, cuda:0, cpu).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        help="Compute dtype for the loaded diffusion pipeline.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts/images per generation batch.")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of timed benchmark batches.")
    parser.add_argument("--warmup-batches", type=int, default=3, help="Warmup batches excluded from timing stats.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=None, help="Generation height. Defaults to training resolution.")
    parser.add_argument("--width", type=int, default=None, help="Generation width. Defaults to training resolution.")
    parser.add_argument(
        "--output-type",
        type=str,
        default="pil",
        choices=["pil", "latent"],
        help="Use latent to exclude VAE decode from speed measurement.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Base RNG seed.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file containing one prompt per line.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt string (repeat flag to provide multiple prompts).",
    )
    parser.add_argument("--strict-adapter-load", action="store_true", help="Fail if adapter loading fails.")
    parser.add_argument(
        "--compile-unet",
        action="store_true",
        help="Compile UNet with torch.compile for potentially faster steady-state inference.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode used when --compile-unet is enabled.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path for full benchmark report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be >= 1")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be >= 0")

    run_dir = args.model_run_dir.expanduser()
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find --model-run-dir: {run_dir}")

    prompts = _load_prompts(args)
    device = torch.device(args.device)
    torch_dtype = _resolve_torch_dtype(str(device), args.dtype)
    loaded = load_diffusion_model(
        run_dir=run_dir,
        device=device,
        strict_adapter_load=bool(args.strict_adapter_load),
        torch_dtype=torch_dtype,
    )
    compile_enabled = False
    supports_cudagraph_mark_step = bool(
        hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    )
    if args.compile_unet:
        if not hasattr(torch, "compile"):
            print("torch.compile is not available in this PyTorch build; continuing without compilation.")
        else:
            try:
                # Compiling UNet usually gives the best speedup for diffusion pipelines.
                loaded.pipe.unet = torch.compile(loaded.pipe.unet, mode=args.compile_mode)
                compile_enabled = True
                print(f"Enabled torch.compile on UNet (mode={args.compile_mode}).")
            except Exception as exc:
                print(f"torch.compile failed; continuing without compilation: {exc}")
    loaded.pipe.set_progress_bar_config(disable=True)

    default_resolution = int(loaded.training_config.get("resolution", 512))
    height = int(args.height) if args.height is not None else default_resolution
    width = int(args.width) if args.width is not None else default_resolution

    total_batches = args.warmup_batches + args.num_batches
    per_batch_seconds: List[float] = []
    per_batch_images_per_sec: List[float] = []
    total_timed_images = 0
    total_timed_seconds = 0.0

    for i in range(total_batches):
        batch_prompts = _build_batch_prompts(prompts, args.batch_size, i)
        batch_seeds = [args.seed + i * args.batch_size + j for j in range(args.batch_size)]
        generators = [torch.Generator(device=str(device)).manual_seed(seed) for seed in batch_seeds]

        # Compiled CUDA graph execution needs explicit iteration boundaries in some pipelines.
        if compile_enabled and device.type == "cuda" and supports_cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = loaded.pipe(
                prompt=batch_prompts,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.guidance_scale),
                generator=generators,
                height=height,
                width=width,
                output_type=args.output_type,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        stage = "warmup" if i < args.warmup_batches else "timed"
        imgs_per_sec = float(args.batch_size / max(elapsed, 1e-9))
        print(
            f"[{stage}] batch={i + 1:03d}/{total_batches:03d} "
            f"time_s={elapsed:.4f} imgs_per_sec={imgs_per_sec:.3f}"
        )

        if stage == "timed":
            per_batch_seconds.append(float(elapsed))
            per_batch_images_per_sec.append(imgs_per_sec)
            total_timed_seconds += float(elapsed)
            total_timed_images += int(args.batch_size)

    summary = {
        "model_run_dir": str(run_dir),
        "pipeline_name": loaded.name,
        "device": str(device),
        "dtype": str(torch_dtype) if torch_dtype is not None else "none",
        "num_prompts_loaded": len(prompts),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "warmup_batches": int(args.warmup_batches),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "height": int(height),
        "width": int(width),
        "output_type": args.output_type,
        "compile_unet": bool(compile_enabled),
        "compile_mode": args.compile_mode if compile_enabled else None,
        "timed_total_images": int(total_timed_images),
        "timed_total_seconds": float(total_timed_seconds),
        "timed_images_per_second": float(total_timed_images / max(total_timed_seconds, 1e-9)),
        "latency_seconds": {
            "mean": float(statistics.mean(per_batch_seconds)),
            "median": float(statistics.median(per_batch_seconds)),
            "std": float(statistics.pstdev(per_batch_seconds) if len(per_batch_seconds) > 1 else 0.0),
            "p90": float(_quantile(per_batch_seconds, 0.90)),
            "p95": float(_quantile(per_batch_seconds, 0.95)),
            "min": float(min(per_batch_seconds)),
            "max": float(max(per_batch_seconds)),
        },
        "throughput_images_per_second": {
            "mean": float(statistics.mean(per_batch_images_per_sec)),
            "median": float(statistics.median(per_batch_images_per_sec)),
            "p10": float(_quantile(per_batch_images_per_sec, 0.10)),
            "p05": float(_quantile(per_batch_images_per_sec, 0.05)),
            "min": float(min(per_batch_images_per_sec)),
            "max": float(max(per_batch_images_per_sec)),
        },
    }

    print("")
    print("=== Benchmark Summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json is not None:
        output_path = args.output_json.expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote benchmark report to: {output_path}")


if __name__ == "__main__":
    main()
