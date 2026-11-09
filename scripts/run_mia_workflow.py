#!/usr/bin/env python
import argparse
import traceback
from pathlib import Path
import sys
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_cache import infer_modality, resolve_output_path
from copyright_pipeline.inference_common import load_yaml, write_json
from copyright_pipeline.inference_tasks import (
    run_analyze_mia_cache,
    run_score_cpk_cache,
    run_score_mia_diffusion,
    run_score_mia_text,
)


def _resolve_wandb_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    wandb_cfg = dict(cfg.get("wandb", {}))
    enabled = bool(wandb_cfg)
    if "enabled" in wandb_cfg:
        enabled = bool(wandb_cfg.get("enabled"))
    wandb_cfg["enabled"] = enabled
    merged_tags = [str(tag) for tag in wandb_cfg.get("tags", [])]
    merged_tags.append("final_mia")
    # Preserve order while removing duplicates.
    wandb_cfg["tags"] = list(dict.fromkeys(merged_tags))
    return wandb_cfg


def _start_wandb_run(
    cfg: Dict[str, Any],
    wandb_cfg: Dict[str, Any],
    config_path: Path,
    overwrite: bool,
) -> Optional[Any]:
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb  # pylint: disable=import-outside-toplevel

        run_name = cfg.get("project", {}).get("run_name") or config_path.stem
        run = wandb.init(
            project=wandb_cfg.get("project", "copyright-paper"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name", run_name),
            tags=wandb_cfg.get("tags", []),
            config=cfg,
            reinit=True,
        )
        run.config.update(
            {
                "workflow": "mia",
                "workflow_config_path": str(config_path),
                "workflow_overwrite": bool(overwrite),
            },
            allow_val_change=True,
        )
        run.log({"workflow/status": "started"})
        return run
    except Exception as exc:
        print(f"[warn] Failed to initialize wandb for MIA workflow: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache-first MIA workflow end to end.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    wandb_cfg = _resolve_wandb_config(cfg)
    wandb_run = _start_wandb_run(cfg, wandb_cfg, config_path=config_path, overwrite=args.overwrite)

    try:
        mia_cache = resolve_output_path(cfg, "mia_cache")
        cpk_cache = resolve_output_path(cfg, "mia_cpk_cache")
        modality = infer_modality(cfg, default=cfg.get("dataset", {}).get("modality", "text"))
        cpk_enabled = bool(cfg.get("cpk", {}).get("enabled", False))

        if args.overwrite or not mia_cache.exists():
            if modality == "diffusion":
                run_score_mia_diffusion(cfg, output_path=str(mia_cache), overwrite=args.overwrite)
            else:
                run_score_mia_text(cfg, output_path=str(mia_cache), overwrite=args.overwrite)

        analysis_input = mia_cache
        if cpk_enabled:
            if args.overwrite or not cpk_cache.exists():
                run_score_cpk_cache(cfg, input_path=str(mia_cache), output_path=str(cpk_cache), overwrite=args.overwrite)
            analysis_input = cpk_cache

        analysis_dir = run_analyze_mia_cache(cfg, str(analysis_input))
        manifest = {
            "workflow": "mia",
            "mia_cache": str(mia_cache),
            "cpk_cache": str(cpk_cache) if cpk_enabled else None,
            "analysis_dir": str(analysis_dir),
            "config_path": str(config_path),
        }
        write_json(analysis_dir / "workflow_manifest.json", manifest)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "workflow/status": "completed",
                    "workflow/cpk_enabled": bool(cpk_enabled),
                    "workflow/modality": str(modality),
                }
            )
            wandb_run.summary["mia_cache_path"] = str(mia_cache)
            wandb_run.summary["cpk_cache_path"] = str(cpk_cache) if cpk_enabled else ""
            wandb_run.summary["analysis_dir"] = str(analysis_dir)
            try:
                wandb_run.save(str(analysis_dir / "workflow_manifest.json"), policy="now")
            except Exception:
                pass
        print(f"MIA workflow completed: {analysis_dir}")
    except Exception:
        if wandb_run is not None:
            err = traceback.format_exc()
            wandb_run.log({"workflow/status": "failed"})
            wandb_run.summary["error"] = err
        raise
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
