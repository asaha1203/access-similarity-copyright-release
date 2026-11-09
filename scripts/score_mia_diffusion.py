#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_score_mia_diffusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Score diffusion MIA.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Optional .pt MIA cache output path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache output.")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    out = run_score_mia_diffusion(cfg, output_path=args.output, overwrite=args.overwrite)
    print(f"Diffusion MIA scoring completed: {out}")


if __name__ == "__main__":
    main()

