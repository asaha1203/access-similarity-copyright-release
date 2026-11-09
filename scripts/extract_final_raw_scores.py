#!/usr/bin/env python
"""Extract compact raw MIA/DRA statistics from finalized workflow caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch


HEAVY_ROW_FIELDS = {
    "prompt",
    "target_text",
    "generated_text",
    "tokens",
    "minkpp_token_scores",
    "logprob_target_per_token",
}

HEAVY_TOP_LEVEL_FIELDS = {
    "records",
    "prompts",
    "images",
    "labels",
    "target_image_tensors",
}

MANIFEST_SUFFIXES = {
    "mia_dp": "mia_cache.pt.manifest.json",
    "mia_cpk": "mia_cpk_cache.pt.manifest.json",
    "dra_dp": "dra_scored_cache.pt.manifest.json",
    "dra_cpk": "dra_cpk_cache.pt.manifest.json",
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            return fv
        return fv
    return None


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return None
    return None


def _small_tensor_to_json(value: torch.Tensor, max_numel: int = 16) -> Optional[str]:
    if value.numel() <= max_numel:
        return json.dumps(value.detach().cpu().tolist(), ensure_ascii=True)
    return None


def _small_list_to_json(value: Sequence[Any], max_len: int = 16) -> Optional[str]:
    if len(value) > max_len:
        return None
    if all(_safe_scalar(v) is not None for v in value):
        return json.dumps(list(value), ensure_ascii=True)
    return None


def _flatten_dict(prefix: str, payload: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in payload.items():
        full_key = f"{prefix}{key}"
        scalar = _safe_scalar(value)
        if scalar is not None:
            out[full_key] = scalar
            continue
        if isinstance(value, torch.Tensor):
            small = _small_tensor_to_json(value)
            if small is not None:
                out[full_key] = small
            else:
                out[f"{full_key}__numel"] = int(value.numel())
                out[f"{full_key}__shape"] = json.dumps(list(value.shape), ensure_ascii=True)
            continue
        if isinstance(value, (list, tuple)):
            small = _small_list_to_json(value)
            if small is not None:
                out[full_key] = small
            else:
                out[f"{full_key}__len"] = len(value)
            continue
        if isinstance(value, dict):
            _flatten_dict(f"{full_key}__", value, out)


def _indexed_from_tensor_or_list(source: Any, idx: int) -> Any:
    if isinstance(source, torch.Tensor):
        if source.ndim == 0:
            return source.item() if idx == 0 else None
        if idx < len(source):
            value = source[idx]
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return value.item()
            if not isinstance(value, torch.Tensor):
                return value
    elif isinstance(source, (list, tuple)):
        if idx < len(source):
            return source[idx]
    return None


def _select_final_manifests(final_root: Path) -> List[Path]:
    manifests: List[Path] = []
    for manifest in final_root.glob("*/*/*.manifest.json"):
        run_name = manifest.parent.name
        file_name = manifest.name
        if run_name == "mia_cpk" and file_name == MANIFEST_SUFFIXES["mia_cpk"]:
            manifests.append(manifest)
        elif run_name.startswith("mia_dp_") and file_name == MANIFEST_SUFFIXES["mia_dp"]:
            manifests.append(manifest)
        elif run_name == "dra_cpk" and file_name == MANIFEST_SUFFIXES["dra_cpk"]:
            manifests.append(manifest)
        elif run_name.startswith("dra_dp_") and file_name == MANIFEST_SUFFIXES["dra_dp"]:
            manifests.append(manifest)
    return sorted(manifests)


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_cache_path(project_root: Path, manifest: Dict[str, Any]) -> Path:
    cache_path = Path(str(manifest["cache_path"]))
    if cache_path.is_absolute():
        return cache_path
    return project_root / cache_path


def _extract_rows_from_cache(cache_path: Path, manifest_path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    data = payload.get("data", {})
    rows = data.get("records", [])

    dataset = cache_path.parts[-3]
    run_name = cache_path.parts[-2]
    stage = str(payload.get("stage"))
    model_type = str(payload.get("model_type"))
    modality = str(payload.get("modality"))

    mia_scores = data.get("mia", {}).get("scores")
    score_direction = data.get("mia", {}).get("score_direction")
    cpk_block = data.get("cpk", {})
    distances = data.get("distances", {})
    sample_ids = data.get("sample_ids")
    candidate_ids = data.get("candidate_ids")
    target_ids = data.get("target_ids")

    flat_rows: List[Dict[str, Any]] = []
    for idx, row_raw in enumerate(rows):
        row = dict(row_raw)
        compact: Dict[str, Any] = {
            "dataset": dataset,
            "run_name": run_name,
            "stage": stage,
            "model_type": model_type,
            "modality": modality,
            "cache_path": str(cache_path),
            "manifest_path": str(manifest_path),
            "row_index": idx,
        }

        for key, value in row.items():
            if key in HEAVY_ROW_FIELDS:
                continue
            scalar = _safe_scalar(value)
            if scalar is not None:
                compact[key] = scalar
                continue
            if isinstance(value, torch.Tensor):
                small = _small_tensor_to_json(value)
                if small is not None:
                    compact[key] = small
                else:
                    compact[f"{key}__numel"] = int(value.numel())
                    compact[f"{key}__shape"] = json.dumps(list(value.shape), ensure_ascii=True)
                continue
            if isinstance(value, (list, tuple)):
                small = _small_list_to_json(value)
                if small is not None:
                    compact[key] = small
                else:
                    compact[f"{key}__len"] = len(value)
                continue
            if isinstance(value, dict):
                _flatten_dict(f"{key}__", value, compact)

        # Backfill from top-level cache arrays/tensors in case rows do not contain these fields.
        if compact.get("sample_id") is None:
            sid = _indexed_from_tensor_or_list(sample_ids, idx)
            if sid is not None:
                compact["sample_id"] = sid
        if compact.get("candidate_id") is None:
            cid = _indexed_from_tensor_or_list(candidate_ids, idx)
            if cid is not None:
                compact["candidate_id"] = cid
        if compact.get("target_sample_id") is None and isinstance(target_ids, list):
            compact["target_id_count"] = len(target_ids)

        mia_from_top = _indexed_from_tensor_or_list(mia_scores, idx)
        if compact.get("mia_score") is None and mia_from_top is not None:
            compact["mia_score"] = mia_from_top
        if compact.get("mia_score_direction") is None and score_direction is not None:
            compact["mia_score_direction"] = score_direction

        for key in ("logp", "logq1", "logq2", "min_logq", "cpk_stat"):
            value = _indexed_from_tensor_or_list(cpk_block.get(key), idx)
            if value is not None and compact.get(key) is None:
                compact[key] = value

        for d_key, d_values in distances.items():
            d_val = _indexed_from_tensor_or_list(d_values, idx)
            if d_val is not None and compact.get(d_key) is None:
                compact[d_key] = d_val

        flat_rows.append(compact)

    if not rows:
        flat_rows.append(
            {
                "dataset": dataset,
                "run_name": run_name,
                "stage": stage,
                "model_type": model_type,
                "modality": modality,
                "cache_path": str(cache_path),
                "manifest_path": str(manifest_path),
                "row_index": None,
                "empty_cache": True,
            }
        )
    return flat_rows


def _cache_inventory_entry(manifest_path: Path, manifest: Dict[str, Any], cache_path: Path) -> Dict[str, Any]:
    data_keys = [k for k in manifest.get("data_keys", []) if k not in HEAVY_TOP_LEVEL_FIELDS]
    return {
        "manifest_path": str(manifest_path),
        "cache_path": str(cache_path),
        "dataset": cache_path.parts[-3],
        "run_name": cache_path.parts[-2],
        "stage": manifest.get("stage"),
        "model_type": manifest.get("model_type"),
        "modality": manifest.get("modality"),
        "created_at": manifest.get("created_at"),
        "input_cache_path": manifest.get("input_cache_path"),
        "data_keys_compact": data_keys,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact raw MIA/DRA stats from finalized workflow caches.")
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path("outputs/inference/final_workflows"),
        help="Root folder containing finalized workflow outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference/final_workflows/analysis"),
        help="Directory where extracted dataframe/inventory are written.",
    )
    parser.add_argument(
        "--print-cache-paths-only",
        action="store_true",
        help="Only print selected final cache paths and exit.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset folder filter(s), e.g. automathtext writingprompts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    final_root = (project_root / args.final_root).resolve() if not args.final_root.is_absolute() else args.final_root
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    manifests = _select_final_manifests(final_root)
    if args.datasets:
        keep = {str(x) for x in args.datasets}
        manifests = [m for m in manifests if m.parts[-3] in keep]
    if not manifests:
        raise FileNotFoundError(f"No final MIA/DRA manifests found under: {final_root}")

    inventory: List[Dict[str, Any]] = []
    cache_paths: List[Path] = []
    for manifest_path in manifests:
        manifest = _load_manifest(manifest_path)
        cache_path = _resolve_cache_path(project_root, manifest)
        cache_paths.append(cache_path)
        inventory.append(_cache_inventory_entry(manifest_path, manifest, cache_path))

    print("Final cache paths selected for extraction:")
    for path in cache_paths:
        print(path)

    if args.print_cache_paths_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_rows: List[Dict[str, Any]] = []
    for inv in inventory:
        extracted_rows.extend(
            _extract_rows_from_cache(Path(inv["cache_path"]), Path(inv["manifest_path"]))
        )

    df = pd.DataFrame(extracted_rows)
    df.sort_values(
        by=[col for col in ("dataset", "run_name", "row_index") if col in df.columns],
        inplace=True,
        ignore_index=True,
    )

    parquet_path = output_dir / "final_raw_scores_compact.parquet"
    csv_path = output_dir / "final_raw_scores_compact.csv"
    inventory_path = output_dir / "final_cache_inventory.json"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    with inventory_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2, ensure_ascii=True)

    print(f"Wrote dataframe rows={len(df)} cols={len(df.columns)}")
    print(f"Parquet: {parquet_path}")
    print(f"CSV: {csv_path}")
    print(f"Inventory: {inventory_path}")


if __name__ == "__main__":
    main()
