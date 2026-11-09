#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_score_cpk_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CP-k components for a MIA or DRA cache.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input .pt cache path.")
    parser.add_argument("--output", type=str, default=None, help="Output .pt CP-k cache path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache output.")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    out = run_score_cpk_cache(cfg, input_path=args.input, output_path=args.output, overwrite=args.overwrite)
    print(f"CP-k scoring completed: {out}")


if __name__ == "__main__":
    main()
