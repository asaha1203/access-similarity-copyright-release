import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .inference_common import now_iso
from .io_utils import ensure_dir, write_json


CACHE_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    payload = {"config": config, "extra": extra or {}, "schema_version": CACHE_SCHEMA_VERSION}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def infer_model_type(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    if model_cfg.get("type"):
        return str(model_cfg["type"])
    if config.get("cpk", {}).get("enabled"):
        return "cpk"
    target = str(config.get("models", {}).get("target", {}).get("checkpoint_dir", "")).lower()
    if "dp_" in target or "/dp" in target:
        return "dp"
    return "baseline"


def infer_modality(config: Dict[str, Any], default: str = "text") -> str:
    model_cfg = config.get("model", {})
    if model_cfg.get("modality"):
        modality = str(model_cfg["modality"])
    else:
        modality = str(config.get("dataset", {}).get("modality", default))
    if modality == "image":
        return "diffusion"
    return modality


def make_cache(
    *,
    stage: str,
    config: Dict[str, Any],
    data: Dict[str, Any],
    aux: Optional[Dict[str, Any]] = None,
    model_type: Optional[str] = None,
    modality: Optional[str] = None,
    input_cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    extra = {"stage": stage, "input_cache_path": input_cache_path}
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "config": config,
        "config_hash": config_hash(config, extra=extra),
        "model_type": model_type or infer_model_type(config),
        "modality": modality or infer_modality(config),
        "created_at": now_iso(),
        "input_cache_path": input_cache_path,
        "data": data,
        "aux": aux or {},
    }


def resolve_output_path(config: Dict[str, Any], stage_name: str, explicit_output: Optional[str] = None) -> Path:
    if explicit_output:
        return Path(explicit_output)
    cache_cfg = config.get("cache", {})
    stage_outputs = cache_cfg.get("stage_outputs", {})
    if isinstance(stage_outputs, dict) and stage_name in stage_outputs:
        return Path(stage_outputs[stage_name])
    if cache_cfg.get("output_path"):
        return Path(cache_cfg["output_path"])
    experiment = config.get("experiment", {})
    output_dir = experiment.get("output_dir") or config.get("project", {}).get("output_dir")
    if output_dir:
        return Path(output_dir) / f"{stage_name}.pt"
    cache_output_dir = cache_cfg.get("output_dir")
    if cache_output_dir:
        return Path(cache_output_dir) / f"{stage_name}.pt"
    output_root = Path(config.get("project", {}).get("output_root", "outputs/inference"))
    dataset = config.get("dataset", {}).get("name") or config.get("project", {}).get("dataset_name", "unknown_dataset")
    run_name = experiment.get("name") or config.get("project", {}).get("run_name", stage_name)
    return output_root / str(dataset) / str(run_name) / f"{stage_name}.pt"


def save_cache(path: Path, payload: Dict[str, Any], overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {path}. Pass --overwrite to replace it.")
    ensure_dir(path.parent)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    write_manifest(path, payload)
    return path


def load_cache(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Cache payload must be a dict: {path}")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema version in {path}: {payload.get('schema_version')}")
    for key in ("stage", "data", "config_hash", "model_type", "modality"):
        if key not in payload:
            raise ValueError(f"Cache {path} is missing required key: {key}")
    return payload


def write_manifest(path: Path, payload: Dict[str, Any]) -> Path:
    manifest_path = Path(str(path) + ".manifest.json")
    data = payload.get("data", {})
    manifest = {
        "cache_path": str(path),
        "schema_version": payload.get("schema_version"),
        "stage": payload.get("stage"),
        "model_type": payload.get("model_type"),
        "modality": payload.get("modality"),
        "config_hash": payload.get("config_hash"),
        "input_cache_path": payload.get("input_cache_path"),
        "created_at": payload.get("created_at"),
        "data_keys": sorted(data.keys()) if isinstance(data, dict) else [],
    }
    write_json(manifest_path, manifest)
    return manifest_path
