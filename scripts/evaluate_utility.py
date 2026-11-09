#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_evaluate_utility


def main() -> None:
    parser = argparse.ArgumentParser(description="Run utility score+analysis from an existing generation cache.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input .pt generation cache path.")
    parser.add_argument("--output", type=str, default=None, help="Optional .pt utility features cache output path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache output.")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    print(f"[Evaluate utility] config={args.config}")
    print(f"[Evaluate utility] source_generation_cache={args.input}")
    cfg.setdefault("inputs", {})
    cfg["inputs"]["generation_cache_path"] = args.input
    out = run_evaluate_utility(cfg, output_path=args.output, overwrite=args.overwrite)
    print(f"Utility evaluation completed: {out}")


if __name__ == "__main__":
    main()

