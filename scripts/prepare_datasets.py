#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.constants import AUTOMATH_DEFAULT_CONFIG, DEFAULT_SPLIT_SEED
from copyright_pipeline.data_prep import DatasetPreparer, PrepareOptions


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare deterministic local datasets and splits.")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--automath-config",
        type=str,
        default=AUTOMATH_DEFAULT_CONFIG,
        help="Config name for math-ai/AutoMathText.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help=(
            "Optional dataset keys to prepare "
            "(automathtext writingprompts pokemon laion_mi). "
            "Text datasets are encoded as prompt[SEP]target."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    options = PrepareOptions(
        split_seed=args.split_seed,
        overwrite=args.overwrite,
        automath_config=args.automath_config,
    )
    preparer = DatasetPreparer(options)
    if args.datasets:
        preparer.prepare_selected(args.datasets)
    else:
        preparer.prepare_all()
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()

