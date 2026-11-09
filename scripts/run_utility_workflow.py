#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_evaluate_utility


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache-first utility score+analysis workflow.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input .pt generation cache path.")
    parser.add_argument("--output", type=str, default=None, help="Optional .pt utility features cache output path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    print(f"[Utility workflow] config={args.config}")
    print(f"[Utility workflow] source_generation_cache={args.input}")
    cfg.setdefault("inputs", {})
    cfg["inputs"]["generation_cache_path"] = args.input
    analysis_dir = run_evaluate_utility(cfg, output_path=args.output, overwrite=args.overwrite)
    print(f"Utility workflow completed: {analysis_dir}")


if __name__ == "__main__":
    main()
