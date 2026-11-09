import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from .io_utils import ensure_dir


class RunLogger:
    def __init__(self, run_dir: Path, wandb_cfg: Dict, run_name: str):
        self.run_dir = run_dir
        ensure_dir(run_dir)
        self.wandb_cfg = wandb_cfg or {}
        self.run_name = run_name
        self._wandb = None

    def start(self) -> None:
        if not self.wandb_cfg.get("enabled", False):
            return
        try:
            wandb_dir = self.run_dir / "wandb"
            wandb_cache_dir = self.run_dir / "wandb-cache"
            ensure_dir(wandb_dir)
            ensure_dir(wandb_cache_dir)
            os.environ["WANDB_DIR"] = str(wandb_dir)
            os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
            import wandb  # pylint: disable=import-outside-toplevel

            self._wandb = wandb.init(
                project=self.wandb_cfg.get("project", "copyright-paper"),
                entity=self.wandb_cfg.get("entity"),
                name=self.wandb_cfg.get("run_name") or self.run_name,
                tags=self.wandb_cfg.get("tags") or [],
                dir=str(self.run_dir),
                reinit=True,
            )
        except Exception:
            # Keep local training/logging functional even without wandb auth.
            self._wandb = None

    def log(self, payload: Dict, step: Optional[int] = None) -> None:
        if self._wandb is not None:
            self._wandb.log(payload, step=step)

    def _flatten_payload(self, prefix: str, value: Any, output: Dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, inner_value in value.items():
                next_prefix = f"{prefix}/{key}" if prefix else str(key)
                self._flatten_payload(next_prefix, inner_value, output)
            return

        if isinstance(value, (list, tuple)):
            output[prefix] = json.dumps(value)
            return

        if isinstance(value, (str, int, float, bool)) or value is None:
            output[prefix] = value
            return

        output[prefix] = str(value)

    def log_run_setup(
        self,
        config: Dict[str, Any],
        privacy_stats: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> None:
        payload: Dict[str, Any] = {}
        self._flatten_payload("config", config, payload)
        if privacy_stats:
            self._flatten_payload("privacy_init", privacy_stats, payload)
        self.log(payload, step=step)

    def log_text_generations(self, records: List[Dict], step: int) -> None:
        if not records:
            return
        local_path = self.run_dir / "text_generation_samples.jsonl"
        with local_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        payload = {
            "text_generation/sample_count": len(records),
            "text_generation/avg_generated_chars": sum(len(r.get("generated_text", "")) for r in records) / len(records),
        }
        if self._wandb is not None:
            try:
                import wandb  # pylint: disable=import-outside-toplevel

                step_key = f"step_{step:07d}"
                columns = [
                    "step",
                    "prompt_id",
                    "prompt",
                    "generated_text",
                    "decode_strategy",
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                ]
                payload["text_generation/samples"] = wandb.Table(
                    columns=columns,
                    data=[[record.get(column) for column in columns] for record in records],
                )
                payload[f"text_generation/{step_key}/samples"] = wandb.Table(
                    columns=columns,
                    data=[[record.get(column) for column in columns] for record in records],
                )
            except Exception:
                pass
        self.log(payload, step=step)

    def log_image_generations(self, records: List[Dict], step: int) -> None:
        if not records:
            return
        local_path = self.run_dir / "image_generation_samples.jsonl"
        with local_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        payload = {
            "image_generation/sample_count": len(records),
            "image_generation/avg_num_inference_steps": mean(
                [int(r.get("num_inference_steps", 0)) for r in records]
            ),
            "image_generation/avg_guidance_scale": mean(
                [float(r.get("guidance_scale", 0.0)) for r in records]
            ),
        }
        if self._wandb is not None:
            try:
                import wandb  # pylint: disable=import-outside-toplevel

                step_key = f"step_{step:07d}"
                columns = [
                    "step",
                    "prompt_id",
                    "prompt",
                    "num_inference_steps",
                    "guidance_scale",
                    "seed",
                    "image",
                ]
                payload["image_generation/samples"] = wandb.Table(
                    columns=columns,
                    data=[
                        [
                            record.get("step"),
                            record.get("prompt_id"),
                            record.get("prompt"),
                            record.get("num_inference_steps"),
                            record.get("guidance_scale"),
                            record.get("seed"),
                            wandb.Image(
                                str(record.get("image_path")),
                                caption=f"step={record.get('step')} prompt={record.get('prompt', '')}",
                            ),
                        ]
                        for record in records
                    ],
                )
                payload[f"image_generation/{step_key}/samples"] = wandb.Table(
                    columns=columns,
                    data=[
                        [
                            record.get("step"),
                            record.get("prompt_id"),
                            record.get("prompt"),
                            record.get("num_inference_steps"),
                            record.get("guidance_scale"),
                            record.get("seed"),
                            wandb.Image(
                                str(record.get("image_path")),
                                caption=f"step={record.get('step')} prompt={record.get('prompt', '')}",
                            ),
                        ]
                        for record in records
                    ],
                )
            except Exception:
                pass
        self.log(payload, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

