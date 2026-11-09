#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.constants import PROCESSED_DATA_ROOT  # noqa: E402
from copyright_pipeline.inference_common import load_yaml, resolve_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate causal-LM perplexity for a saved text checkpoint/run directory."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help=(
            "Run directory, checkpoint directory, or run name under runs/. "
            "LoRA adapters are auto-loaded if adapter_config.json exists."
        ),
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Processed dataset key. Defaults to the checkpoint training_config.yaml dataset_key.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Processed split to evaluate. Defaults to the checkpoint training_config.yaml split_name.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Tokenizer truncation length. Defaults to training_config.yaml max_seq_len.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of examples to score.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model load dtype. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the metrics JSON.",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Progress print interval in batches.")
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return None


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def resolve_training_run_dir(path: Path) -> Path:
    if (path / "training_config.yaml").exists():
        return path
    if (path.parent / "training_config.yaml").exists():
        return path.parent
    raise FileNotFoundError(f"Could not find training_config.yaml in {path} or its parent.")


def resolve_model_dir(path: Path) -> Path:
    if (path / "adapter_config.json").exists():
        return path

    checkpoints = [
        child
        for child in path.glob("checkpoint-*")
        if child.is_dir() and (child / "adapter_config.json").exists()
    ]
    if checkpoints:
        return max(checkpoints, key=checkpoint_step)

    return path


def load_tokenizer(model_dir: Path, base_model: str):
    tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_dir: Path, base_model: str, device: torch.device, torch_dtype: Optional[torch.dtype]):
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    adapter_cfg = model_dir / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel  # pylint: disable=import-outside-toplevel

        model = PeftModel.from_pretrained(model, str(model_dir), device_map={"": str(device)})
    else:
        print(f"Warning: no adapter_config.json found under {model_dir}; evaluating base model.", flush=True)

    model.to(device)
    model.eval()
    return model


def tokenized_dataset(dataset, tokenizer, max_seq_len: int):
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


def shifted_label_token_count(labels: torch.Tensor) -> int:
    if labels.size(1) <= 1:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


def main() -> None:
    args = parse_args()
    input_dir = resolve_run_dir(args.checkpoint_dir)
    run_dir = resolve_training_run_dir(input_dir)
    model_dir = resolve_model_dir(input_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = resolve_dtype(args.dtype, device)

    training_cfg = load_yaml(run_dir / "training_config.yaml")
    base_model = training_cfg["model_name_or_path"]
    tokenizer = load_tokenizer(model_dir, base_model)
    model = load_model(model_dir, base_model, device=device, torch_dtype=torch_dtype)

    dataset_key = args.dataset_key or training_cfg["dataset_key"]
    split_name = args.split_name or training_cfg["split_name"]
    max_seq_len = int(args.max_seq_len or training_cfg["max_seq_len"])
    split_path = PROCESSED_DATA_ROOT / dataset_key / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Processed split not found: {split_path}")

    dataset = load_from_disk(str(split_path))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(int(args.max_samples), len(dataset))))

    tokenized = tokenized_dataset(dataset, tokenizer, max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(tokenized, batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)

    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            token_count = shifted_label_token_count(batch["labels"])
            if token_count == 0:
                continue
            outputs = model(**batch)
            total_nll += float(outputs.loss.detach().cpu()) * token_count
            total_tokens += token_count
            if args.log_every > 0 and batch_idx % int(args.log_every) == 0:
                running_loss = total_nll / max(total_tokens, 1)
                print(
                    json.dumps(
                        {
                            "batches": batch_idx,
                            "tokens": total_tokens,
                            "nll": running_loss,
                            "perplexity": math.exp(running_loss),
                        }
                    ),
                    flush=True,
                )

    if total_tokens == 0:
        raise RuntimeError("No target tokens were available for perplexity estimation.")

    mean_nll = total_nll / total_tokens
    metrics = {
        "checkpoint_dir": str(input_dir),
        "model_dir": str(model_dir),
        "run_dir": str(run_dir),
        "dataset_key": dataset_key,
        "split_name": split_name,
        "num_examples": len(dataset),
        "max_seq_len": max_seq_len,
        "batch_size": int(args.batch_size),
        "target_tokens": total_tokens,
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
