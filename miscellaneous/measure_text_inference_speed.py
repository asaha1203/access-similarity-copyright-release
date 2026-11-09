#!/usr/bin/env python
"""Benchmark language-model inference speed for a trained run directory."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_text_model  # noqa: E402


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
        prompts = ["Write a short paragraph about machine learning."]
    return prompts


def _build_batch_prompts(prompts: List[str], batch_size: int, batch_idx: int) -> List[str]:
    out: List[str] = []
    start = batch_idx * batch_size
    for j in range(batch_size):
        out.append(prompts[(start + j) % len(prompts)])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure language-model inference speed.")
    parser.add_argument(
        "--model-run-dir",
        type=Path,
        required=True,
        help="Run directory containing training_config.yaml (+ adapter weights if finetuned).",
    )
    parser.add_argument("--device", type=str, default=_default_device(), help="Torch device (e.g. cuda, cuda:0, cpu).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        help="Compute dtype for the loaded language model.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per generation batch.")
    parser.add_argument("--num-batches", type=int, default=30, help="Number of timed benchmark batches.")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Warmup batches excluded from timing stats.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=512, help="Tokenizer truncation length for prompt inputs.")
    parser.add_argument("--pad-to-multiple-of", type=int, default=None, help="Pad prompt inputs to this token multiple.")
    parser.add_argument(
        "--prompt-length-buckets",
        type=int,
        nargs="*",
        default=[],
        help="Optional fixed prompt lengths; each batch pads to the smallest bucket that fits.",
    )
    parser.add_argument(
        "--cache-implementation",
        type=str,
        default=None,
        help="Optional Transformers generation cache implementation, e.g. dynamic or static.",
    )
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling in generation.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
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
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile the language model with torch.compile for potentially faster steady-state inference.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode used when --compile-model is enabled.",
    )
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Keep PEFT LoRA adapters unmerged during inference for comparison.",
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
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens must be >= 1")

    run_dir = args.model_run_dir.expanduser()
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find --model-run-dir: {run_dir}")

    prompts = _load_prompts(args)
    device = torch.device(args.device)
    torch_dtype = _resolve_torch_dtype(str(device), args.dtype)
    loaded = load_text_model(
        run_dir=run_dir,
        device=device,
        torch_dtype=torch_dtype,
        compile_model=False,
        merge_lora=not bool(args.no_merge_lora),
    )

    compile_enabled = False
    supports_cudagraph_mark_step = bool(
        hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")
    )
    if args.compile_model:
        if not hasattr(torch, "compile"):
            print("torch.compile is not available in this PyTorch build; continuing without compilation.")
        else:
            try:
                loaded.model = torch.compile(loaded.model, mode=args.compile_mode)
                compile_enabled = True
                print(f"Enabled torch.compile on language model (mode={args.compile_mode}).")
            except Exception as exc:
                print(f"torch.compile failed; continuing without compilation: {exc}")

    total_batches = args.warmup_batches + args.num_batches
    per_batch_seconds: List[float] = []
    per_batch_reqs_per_sec: List[float] = []
    per_batch_tokens_per_sec: List[float] = []
    per_batch_new_tokens: List[int] = []

    total_timed_requests = 0
    total_timed_new_tokens = 0
    total_timed_seconds = 0.0

    pad_token_id = loaded.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = loaded.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    for i in range(total_batches):
        batch_prompts = _build_batch_prompts(prompts, args.batch_size, i)
        batch_seed = args.seed + i
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": int(args.max_prompt_tokens),
        }
        if args.pad_to_multiple_of is not None:
            tokenizer_kwargs["pad_to_multiple_of"] = int(args.pad_to_multiple_of)
        if args.prompt_length_buckets:
            prompt_lengths = [
                len(
                    loaded.tokenizer.encode(
                        prompt,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=int(args.max_prompt_tokens),
                    )
                )
                for prompt in batch_prompts
            ]
            longest_prompt = min(max(prompt_lengths) if prompt_lengths else 0, int(args.max_prompt_tokens))
            buckets = sorted(set(int(bucket) for bucket in args.prompt_length_buckets))
            bucket_max = next((bucket for bucket in buckets if bucket >= longest_prompt), buckets[-1])
            if args.pad_to_multiple_of is not None and bucket_max % int(args.pad_to_multiple_of) != 0:
                multiple = int(args.pad_to_multiple_of)
                bucket_max = ((bucket_max + multiple - 1) // multiple) * multiple
            tokenizer_kwargs["padding"] = "max_length"
            tokenizer_kwargs["max_length"] = max(1, min(bucket_max, int(args.max_prompt_tokens)))
        enc = loaded.tokenizer(batch_prompts, **tokenizer_kwargs).to(device)

        if compile_enabled and device.type == "cuda" and supports_cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        generate_kwargs = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "pad_token_id": int(pad_token_id),
            "eos_token_id": loaded.tokenizer.eos_token_id,
        }
        if args.do_sample:
            # Sampling-only knobs; avoid passing them during greedy decoding.
            if float(args.temperature) != 1.0:
                generate_kwargs["temperature"] = float(args.temperature)
            generate_kwargs["top_p"] = float(args.top_p)
            generate_kwargs["top_k"] = int(args.top_k)
        if args.cache_implementation is not None:
            generate_kwargs["cache_implementation"] = str(args.cache_implementation)
        # Some model wrappers reject `generator`; global seeding is more portable.
        torch.manual_seed(int(batch_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(batch_seed))
        with torch.inference_mode():
            generated = loaded.model.generate(
                **enc,
                **generate_kwargs,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        prompt_lens = enc["attention_mask"].sum(dim=1)
        nonpad_lens = (generated != int(pad_token_id)).sum(dim=1)
        new_token_count = int((nonpad_lens - prompt_lens).clamp(min=0).sum().item())

        stage = "warmup" if i < args.warmup_batches else "timed"
        reqs_per_sec = float(args.batch_size / max(elapsed, 1e-9))
        toks_per_sec = float(new_token_count / max(elapsed, 1e-9))
        print(
            f"[{stage}] batch={i + 1:03d}/{total_batches:03d} "
            f"time_s={elapsed:.4f} reqs_per_sec={reqs_per_sec:.3f} toks_per_sec={toks_per_sec:.1f} "
            f"new_tokens={new_token_count}"
        )

        if stage == "timed":
            per_batch_seconds.append(float(elapsed))
            per_batch_reqs_per_sec.append(reqs_per_sec)
            per_batch_tokens_per_sec.append(toks_per_sec)
            per_batch_new_tokens.append(new_token_count)
            total_timed_seconds += float(elapsed)
            total_timed_requests += int(args.batch_size)
            total_timed_new_tokens += int(new_token_count)

    summary = {
        "model_run_dir": str(run_dir),
        "model_name": loaded.name,
        "device": str(device),
        "dtype": str(torch_dtype) if torch_dtype is not None else "none",
        "num_prompts_loaded": len(prompts),
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "warmup_batches": int(args.warmup_batches),
        "max_new_tokens": int(args.max_new_tokens),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "pad_to_multiple_of": int(args.pad_to_multiple_of) if args.pad_to_multiple_of is not None else None,
        "prompt_length_buckets": [int(bucket) for bucket in args.prompt_length_buckets],
        "cache_implementation": args.cache_implementation,
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "compile_model": bool(compile_enabled),
        "compile_mode": args.compile_mode if compile_enabled else None,
        "merge_lora": not bool(args.no_merge_lora),
        "timed_total_requests": int(total_timed_requests),
        "timed_total_new_tokens": int(total_timed_new_tokens),
        "timed_total_seconds": float(total_timed_seconds),
        "timed_requests_per_second": float(total_timed_requests / max(total_timed_seconds, 1e-9)),
        "timed_tokens_per_second": float(total_timed_new_tokens / max(total_timed_seconds, 1e-9)),
        "new_tokens_per_request": float(total_timed_new_tokens / max(total_timed_requests, 1)),
        "latency_seconds": {
            "mean": float(statistics.mean(per_batch_seconds)),
            "median": float(statistics.median(per_batch_seconds)),
            "std": float(statistics.pstdev(per_batch_seconds) if len(per_batch_seconds) > 1 else 0.0),
            "p90": float(_quantile(per_batch_seconds, 0.90)),
            "p95": float(_quantile(per_batch_seconds, 0.95)),
            "min": float(min(per_batch_seconds)),
            "max": float(max(per_batch_seconds)),
        },
        "throughput_requests_per_second": {
            "mean": float(statistics.mean(per_batch_reqs_per_sec)),
            "median": float(statistics.median(per_batch_reqs_per_sec)),
            "p10": float(_quantile(per_batch_reqs_per_sec, 0.10)),
            "p05": float(_quantile(per_batch_reqs_per_sec, 0.05)),
            "min": float(min(per_batch_reqs_per_sec)),
            "max": float(max(per_batch_reqs_per_sec)),
        },
        "throughput_tokens_per_second": {
            "mean": float(statistics.mean(per_batch_tokens_per_sec)),
            "median": float(statistics.median(per_batch_tokens_per_sec)),
            "p10": float(_quantile(per_batch_tokens_per_sec, 0.10)),
            "p05": float(_quantile(per_batch_tokens_per_sec, 0.05)),
            "min": float(min(per_batch_tokens_per_sec)),
            "max": float(max(per_batch_tokens_per_sec)),
        },
        "new_tokens_per_batch": {
            "mean": float(statistics.mean(per_batch_new_tokens)),
            "median": float(statistics.median(per_batch_new_tokens)),
            "min": int(min(per_batch_new_tokens)),
            "max": int(max(per_batch_new_tokens)),
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
