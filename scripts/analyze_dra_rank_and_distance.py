#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_common import load_yaml
from copyright_pipeline.inference_tasks import run_analyze_dra_rank_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DRA ranking and distance.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    out = run_analyze_dra_rank_distance(cfg)
    print(f"DRA rank+distance analysis completed: {out}")


if __name__ == "__main__":
    main()

