#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_analyze_mia, run_analyze_mia_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MIA records.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, default=None, help="Optional .pt MIA or MIA+CP-k cache input.")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    out = run_analyze_mia_cache(cfg, args.input) if args.input else run_analyze_mia(cfg)
    print(f"MIA analysis completed: {out}")


if __name__ == "__main__":
    main()

