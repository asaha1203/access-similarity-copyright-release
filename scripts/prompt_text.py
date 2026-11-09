#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import (  # noqa: E402
    load_text_model,
    realized_token_logprobs,
    resolve_run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt a (LoRA) text checkpoint and print raw outputs.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Run directory (or name under runs/) containing training_config.yaml; LoRA is auto-loaded if adapter_config.json exists.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to feed the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device string (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)

    parser.add_argument("--show-token-ids", action="store_true", help="Print generated token IDs.")
    parser.add_argument(
        "--show-token-logprobs",
        action="store_true",
        help="Score the generated continuation and print per-token logprobs (and seq/mean).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.checkpoint_dir)
    device = torch.device(args.device)

    torch.manual_seed(int(args.seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(args.seed))

    loaded = load_text_model(run_dir, device=device)
    prompt = str(args.prompt)

    enc = loaded.tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = loaded.model.generate(
            **enc,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            pad_token_id=loaded.tokenizer.eos_token_id,
        )

    gen_ids = generated[0, enc["input_ids"].shape[1] :]
    continuation = loaded.tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("=== PROMPT ===")
    print(prompt)
    print("\n=== COMPLETION ===")
    print(continuation)

    if args.show_token_ids:
        print("\n=== GENERATED TOKEN IDS ===")
        print(gen_ids.detach().cpu().tolist())

    if args.show_token_logprobs:
        scores = realized_token_logprobs(loaded, prompt, continuation, score_prompt_tokens=False)
        print("\n=== TOKEN LOGPROBS (GENERATED TOKENS) ===")
        print(scores["token_logprobs"])
        print("\n=== SUMMARY ===")
        print(
            {
                "generated_token_count": scores["scored_token_count"],
                "seq_logprob": scores["seq_logprob"],
                "mean_logprob": scores["mean_logprob"],
            }
        )


if __name__ == "__main__":
    main()

