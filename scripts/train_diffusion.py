#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.diffusion_trainer import DiffusionRunConfig, run_diffusion_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion LoRA or full-UNet fine-tuning runs.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = DiffusionRunConfig(**payload)
    max_steps_override = 1 if args.dry_run else None
    if args.dry_run:
        cfg.num_train_epochs = 1
    run_dir = run_diffusion_training(cfg, max_steps_override=max_steps_override)
    print(f"Diffusion run completed: {run_dir}")


if __name__ == "__main__":
    main()

