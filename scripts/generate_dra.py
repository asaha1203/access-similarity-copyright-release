#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_generate_dra


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DRA candidates.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Optional .pt DRA generation cache output path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache output.")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    out = run_generate_dra(cfg, output_path=args.output, overwrite=args.overwrite)
    print(f"DRA generation completed: {out}")


if __name__ == "__main__":
    main()

