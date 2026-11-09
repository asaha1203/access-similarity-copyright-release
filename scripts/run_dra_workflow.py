#!/usr/bin/env python
import argparse
import time
import traceback
from pathlib import Path
import sys
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_cache import resolve_output_path
from copyright_pipeline.inference_common import load_yaml, write_json
from copyright_pipeline.inference_tasks import (
    run_analyze_dra_cache,
    run_generate_dra,
    run_score_cpk_cache,
    run_score_dra_mia_cache,
)


def _monitor_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime_cfg = dict(cfg.get("runtime", {}))
    monitor_cfg_raw = runtime_cfg.get("monitor", cfg.get("monitor", {}))
    monitor_cfg = dict(monitor_cfg_raw) if isinstance(monitor_cfg_raw, dict) else {}
    return {
        "stage_prints": bool(monitor_cfg.get("stage_prints", True)),
    }


def _log_stage_event(wandb_run: Optional[Any], stage: str, status: str, payload: Optional[Dict[str, Any]] = None) -> None:
    if wandb_run is None:
        return
    event = {
        "workflow/stage": stage,
        "workflow/stage_status": status,
        f"workflow/stage/{stage}/status": status,
    }
    if payload:
        for key, value in payload.items():
            event[f"workflow/stage/{stage}/{key}"] = value
    wandb_run.log(event)


def _merge_wandb_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    wandb_cfg = dict(cfg.get("wandb", {}))
    enabled = bool(wandb_cfg.get("enabled", False))
    if args.wandb:
        enabled = True
    wandb_cfg["enabled"] = enabled
    if args.wandb_project:
        wandb_cfg["project"] = args.wandb_project
    if args.wandb_entity:
        wandb_cfg["entity"] = args.wandb_entity
    cli_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] if args.wandb_tags else []
    merged_tags = [str(tag) for tag in wandb_cfg.get("tags", [])] + cli_tags
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
                "workflow": "dra",
                "workflow_config_path": str(config_path),
                "workflow_overwrite": bool(overwrite),
            },
            allow_val_change=True,
        )
        run.log({"workflow/status": "started"})
        return run
    except Exception as exc:
        print(f"[warn] Failed to initialize wandb for DRA workflow: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache-first DRA workflow end to end.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for this workflow run.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override wandb project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Override wandb entity/team.")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated extra wandb tags.")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    monitor_cfg = _monitor_config(cfg)
    wandb_cfg = _merge_wandb_config(cfg, args)
    wandb_run = _start_wandb_run(cfg, wandb_cfg, config_path=config_path, overwrite=args.overwrite)

    try:
        generation_cache = resolve_output_path(cfg, "dra_generation_cache")
        scored_cache = resolve_output_path(cfg, "dra_scored_cache")
        cpk_cache = resolve_output_path(cfg, "dra_cpk_cache")
        cpk_enabled = bool(cfg.get("cpk", {}).get("enabled", False))
        modality = str(cfg.get("dataset", {}).get("modality", "text")).lower()

        def _run_stage(stage: str, fn):
            if monitor_cfg["stage_prints"]:
                print(f"[DRA workflow] stage_start={stage}")
            _log_stage_event(wandb_run, stage, "started")
            stage_start = time.perf_counter()
            result = fn()
            elapsed = time.perf_counter() - stage_start
            if monitor_cfg["stage_prints"]:
                print(f"[DRA workflow] stage_done={stage} elapsed_s={elapsed:.2f}")
            _log_stage_event(wandb_run, stage, "completed", payload={"duration_seconds": float(elapsed)})
            return result

        if args.overwrite or not generation_cache.exists():
            _run_stage(
                "generate_dra",
                lambda: run_generate_dra(cfg, output_path=str(generation_cache), overwrite=args.overwrite),
            )
        else:
            if monitor_cfg["stage_prints"]:
                print(f"[DRA workflow] stage_skip=generate_dra cache={generation_cache}")
            _log_stage_event(wandb_run, "generate_dra", "skipped", payload={"cache_path": str(generation_cache)})
        if args.overwrite or not scored_cache.exists():
            _run_stage(
                "score_dra_mia",
                lambda: run_score_dra_mia_cache(
                    cfg,
                    input_path=str(generation_cache),
                    output_path=str(scored_cache),
                    overwrite=args.overwrite,
                ),
            )
        else:
            if monitor_cfg["stage_prints"]:
                print(f"[DRA workflow] stage_skip=score_dra_mia cache={scored_cache}")
            _log_stage_event(wandb_run, "score_dra_mia", "skipped", payload={"cache_path": str(scored_cache)})

        analysis_input = scored_cache
        if cpk_enabled:
            if modality == "diffusion":
                # Diffusion CP-k logprob ratios are computed during generation.
                analysis_input = scored_cache
                if monitor_cfg["stage_prints"]:
                    print("[DRA workflow] stage_skip=score_cpk reason=diffusion_inline")
                _log_stage_event(
                    wandb_run,
                    "score_cpk",
                    "skipped",
                    payload={"reason": "diffusion_inline", "cache_path": str(scored_cache)},
                )
            else:
                if args.overwrite or not cpk_cache.exists():
                    _run_stage(
                        "score_cpk",
                        lambda: run_score_cpk_cache(
                            cfg,
                            input_path=str(scored_cache),
                            output_path=str(cpk_cache),
                            overwrite=args.overwrite,
                        ),
                    )
                else:
                    if monitor_cfg["stage_prints"]:
                        print(f"[DRA workflow] stage_skip=score_cpk cache={cpk_cache}")
                    _log_stage_event(wandb_run, "score_cpk", "skipped", payload={"cache_path": str(cpk_cache)})
                analysis_input = cpk_cache

        analysis_dir = _run_stage("analyze_dra", lambda: run_analyze_dra_cache(cfg, str(analysis_input)))
        manifest = {
            "workflow": "dra",
            "generation_cache": str(generation_cache),
            "scored_cache": str(scored_cache),
            "cpk_cache": str(cpk_cache) if (cpk_enabled and modality != "diffusion") else None,
            "analysis_dir": str(analysis_dir),
            "config_path": str(config_path),
        }
        write_json(analysis_dir / "workflow_manifest.json", manifest)
        if wandb_run is not None:
            wandb_run.log({"workflow/status": "completed", "workflow/cpk_enabled": bool(cpk_enabled)})
            wandb_run.summary["generation_cache_path"] = str(generation_cache)
            wandb_run.summary["scored_cache_path"] = str(scored_cache)
            wandb_run.summary["cpk_cache_path"] = str(cpk_cache) if (cpk_enabled and modality != "diffusion") else ""
            wandb_run.summary["analysis_dir"] = str(analysis_dir)
            try:
                wandb_run.save(str(analysis_dir / "workflow_manifest.json"), policy="now")
            except Exception:
                pass
        print(f"DRA workflow completed: {analysis_dir}")
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
