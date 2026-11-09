#!/usr/bin/env python
import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.constants import PROJECT_ROOT  # noqa: E402
from copyright_pipeline.inference_common import (  # noqa: E402
    bleu1,
    collect_text_prompts,
    dump_csv,
    dump_jsonl,
    load_text_model,
    normalized_levenshtein,
    resolve_run_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prompt-level dataset reconstruction by generating multiple continuations "
            "and scoring each generation against the intended target reconstruction."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the trained checkpoint directory (or run name under runs/).",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Dataset key under data/processed (defaults to training_config.yaml dataset_key).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_full",
        help="Dataset split to draw prompt/target pairs from (default: train_full).",
    )
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to evaluate.")
    parser.add_argument("--num-generations", type=int, default=10, help="Generations per prompt.")
    parser.add_argument(
        "--num-gens-per-call",
        type=int,
        default=2,
        help=(
            "How many sampled continuations to request per prompt in one generate() call. "
            "Higher can be faster but uses more GPU memory."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens per generation (hard-capped to 2048).",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling-based decoding (default: enabled).",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling parameter.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter.")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompt generations per forward pass.",
    )
    parser.add_argument(
        "--text-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Text model dtype on CUDA (default: fp16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string: auto, cuda, cuda:0, or cpu (default: auto).",
    )
    parser.add_argument(
        "--infringement-lev-threshold",
        type=float,
        default=0.2,
        help="Mark a generation as infringement when normalized Levenshtein <= threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference/reconstruction_eval",
        help="Root output directory for metrics/artifacts.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_torch_dtype(device: torch.device, text_dtype: str) -> torch.dtype | None:
    if not str(device).startswith("cuda"):
        return None
    if text_dtype == "fp16":
        return torch.float16
    if text_dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if values else 0.0


def main() -> None:
    args = parse_args()

    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be >= 1")
    if args.num_generations <= 0:
        raise ValueError("--num-generations must be >= 1")
    if args.num_gens_per_call <= 0:
        raise ValueError("--num-gens-per-call must be >= 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.infringement_lev_threshold < 0.0 or args.infringement_lev_threshold > 1.0:
        raise ValueError("--infringement-lev-threshold must be in [0, 1]")

    run_dir = resolve_run_dir(args.model_dir)
    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(device, args.text_dtype)
    loaded = load_text_model(run_dir, device=device, torch_dtype=torch_dtype)
    loaded.tokenizer.padding_side = "left"
    loaded.model.config.use_cache = True
    # Prevent repeated HF warnings when generation_config defines max_length and
    # we explicitly use max_new_tokens for reconstruction.
    if getattr(loaded.model, "generation_config", None) is not None:
        loaded.model.generation_config.max_length = None
    if hasattr(loaded.model, "base_model") and getattr(loaded.model.base_model, "generation_config", None) is not None:
        loaded.model.base_model.generation_config.max_length = None

    dataset_key = args.dataset_key or str(loaded.training_config.get("dataset_key", "")).strip()
    if not dataset_key:
        raise ValueError(
            "Could not determine dataset key. Pass --dataset-key explicitly, "
            "or include dataset_key in training_config.yaml."
        )

    samples = collect_text_prompts(dataset_key, args.split, args.num_prompts)
    if not samples:
        raise RuntimeError(f"No prompt/target samples found for dataset={dataset_key}, split={args.split}.")

    max_new_tokens = min(int(args.max_new_tokens), 2048)
    if max_new_tokens < args.max_new_tokens:
        print(f"[warning] --max-new-tokens capped to {max_new_tokens} (requested {args.max_new_tokens}).")
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_dir / dataset_key / f"{run_dir.name}__{timestamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    bleu_values: List[float] = []
    lev_values: List[float] = []
    exact_count = 0
    infringement_count = 0
    total_steps = len(samples) * int(args.num_generations)
    pbar = tqdm(total=total_steps, desc="Reconstruction generation", unit="sample")
    prompts = [str(s["prompt"]) for s in samples]
    targets = [str(s["target_text"]) for s in samples]
    sample_ids = [int(s["sample_id"]) for s in samples]

    for gen_start in range(0, int(args.num_generations), int(args.num_gens_per_call)):
        cur_num_gens = min(int(args.num_gens_per_call), int(args.num_generations) - gen_start)
        for start in range(0, len(prompts), int(args.batch_size)):
            end = min(start + int(args.batch_size), len(prompts))
            prompt_batch = prompts[start:end]
            target_batch = targets[start:end]
            sample_id_batch = sample_ids[start:end]

            batch_seed = int(args.seed) + gen_start * 10_000 + start
            torch.manual_seed(batch_seed)
            if str(device).startswith("cuda"):
                torch.cuda.manual_seed_all(batch_seed)

            enc = loaded.tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(device)
            generate_kwargs = {
                **enc,
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(args.do_sample),
                "num_return_sequences": int(cur_num_gens),
                "pad_token_id": loaded.tokenizer.eos_token_id,
            }
            if bool(args.do_sample):
                generate_kwargs.update(
                    {
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                        "top_k": int(args.top_k),
                    }
                )
            with torch.inference_mode():
                generated = loaded.model.generate(**generate_kwargs)

            prompt_lengths = enc["attention_mask"].sum(dim=1).tolist()
            for local_idx, prompt_len in enumerate(prompt_lengths):
                for rep_idx in range(cur_num_gens):
                    gen_idx = gen_start + rep_idx
                    prompt_idx = start + local_idx
                    prompt = prompt_batch[local_idx]
                    target = target_batch[local_idx]
                    sample_id = int(sample_id_batch[local_idx])
                    sample_seed = int(args.seed) + prompt_idx * 10_000 + gen_idx
                    output_row_idx = local_idx * cur_num_gens + rep_idx

                    gen_ids = generated[output_row_idx, int(prompt_len) :]
                    continuation = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    exact_match = continuation == target
                    bleu = bleu1(target, continuation)
                    nlev = normalized_levenshtein(target, continuation)
                    infringement = nlev <= float(args.infringement_lev_threshold)

                    exact_count += int(exact_match)
                    infringement_count += int(infringement)
                    bleu_values.append(float(bleu))
                    lev_values.append(float(nlev))

                    rows.append(
                        {
                            "sample_id": sample_id,
                            "prompt_index": prompt_idx,
                            "generation_index": gen_idx,
                            "seed": sample_seed,
                            "prompt": prompt,
                            "target_text": target,
                            "generated_text": continuation,
                            "generated_token_count": int(gen_ids.numel()),
                            "exact_match": bool(exact_match),
                            "infringement": bool(infringement),
                            "bleu": float(bleu),
                            "normalized_levenshtein": float(nlev),
                        }
                    )
                    pbar.update(1)
            pbar.set_postfix(gen=f"{gen_start + 1}-{gen_start + cur_num_gens}", batch=f"{start}:{end}")

    pbar.close()

    total = len(rows)
    by_prompt: List[Dict[str, Any]] = []
    for prompt_idx in range(len(samples)):
        subset = [r for r in rows if int(r["prompt_index"]) == prompt_idx]
        subset_total = len(subset)
        subset_exact = sum(int(r["exact_match"]) for r in subset)
        subset_infr = sum(int(r["infringement"]) for r in subset)
        subset_bleu = [float(r["bleu"]) for r in subset]
        subset_lev = [float(r["normalized_levenshtein"]) for r in subset]
        by_prompt.append(
            {
                "prompt_index": prompt_idx,
                "sample_id": int(samples[prompt_idx]["sample_id"]),
                "num_generations": subset_total,
                "exact_match_count": subset_exact,
                "exact_match_rate": float(subset_exact / max(1, subset_total)),
                "infringement_count": subset_infr,
                "infringement_rate": float(subset_infr / max(1, subset_total)),
                "bleu_mean": _mean(subset_bleu),
                "bleu_std": _stdev(subset_bleu),
                "normalized_levenshtein_mean": _mean(subset_lev),
                "normalized_levenshtein_std": _stdev(subset_lev),
            }
        )

    summary = {
        "model_dir": str(run_dir),
        "dataset_key": dataset_key,
        "split": args.split,
        "device": str(device),
        "num_prompts": len(samples),
        "num_generations_per_prompt": int(args.num_generations),
        "num_gens_per_call": int(args.num_gens_per_call),
        "total_generations": total,
        "max_new_tokens": int(max_new_tokens),
        "decode_strategy": "sampling" if bool(args.do_sample) else "greedy",
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature) if bool(args.do_sample) else None,
        "top_p": float(args.top_p) if bool(args.do_sample) else None,
        "top_k": int(args.top_k) if bool(args.do_sample) else None,
        "base_seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "text_dtype": args.text_dtype,
        "effective_torch_dtype": str(torch_dtype) if torch_dtype is not None else "default",
        "infringement_lev_threshold": float(args.infringement_lev_threshold),
        "exact_match_count": int(exact_count),
        "exact_match_rate": float(exact_count / max(1, total)),
        "infringement_count": int(infringement_count),
        "infringement_rate": float(infringement_count / max(1, total)),
        "bleu_mean": _mean(bleu_values),
        "bleu_std": _stdev(bleu_values),
        "normalized_levenshtein_mean": _mean(lev_values),
        "normalized_levenshtein_std": _stdev(lev_values),
    }

    dump_jsonl(out_dir / "reconstruction_records.jsonl", rows)
    dump_csv(out_dir / "reconstruction_records.csv", rows)
    write_json(out_dir / "reconstruction_summary.json", summary)
    write_json(out_dir / "reconstruction_by_prompt.json", {"rows": by_prompt})

    print("Reconstruction evaluation complete.")
    print(f"Model: {run_dir}")
    print(f"Dataset split: {dataset_key}/{args.split}")
    print(f"Device: {device}")
    print(f"Generations: {total} ({len(samples)} prompts x {args.num_generations})")
    print(f"Exact match count/rate: {exact_count}/{summary['exact_match_rate']:.4f}")
    print(f"Infringement count/rate: {infringement_count}/{summary['infringement_rate']:.4f}")
    print(f"BLEU mean: {summary['bleu_mean']:.4f}")
    print(f"Normalized Levenshtein mean: {summary['normalized_levenshtein_mean']:.4f}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
