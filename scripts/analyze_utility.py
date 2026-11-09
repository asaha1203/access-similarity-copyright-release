#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_analyze_utility_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze utility features cache with optional CP-k quantile slicing.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input .pt utility features cache path.")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    out = run_analyze_utility_cache(cfg, args.input)
    print(f"Utility cache analysis completed: {out}")


if __name__ == "__main__":
    main()
