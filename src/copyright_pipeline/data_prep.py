import random
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image as PILImage
from datasets import Dataset, Image, concatenate_datasets, load_dataset, load_from_disk

from .constants import (
    DEFAULT_SPLIT_SEED,
    IMAGE_DATASETS,
    PROCESSED_DATA_ROOT,
    RAW_DATA_ROOT,
    TEXT_DATASETS,
)
from .io_utils import ensure_dir, read_json, stable_indices_hash, write_json

URL_DOWNLOAD_BATCH_SIZE = 256
DEFAULT_DOWNLOAD_WORKERS = 8
HTTP_TIMEOUT = (3, 10)
DEFAULT_MAP_NUM_PROC = 8


@dataclass
class PrepareOptions:
    split_seed: int = DEFAULT_SPLIT_SEED
    overwrite: bool = False
    automath_config: Optional[str] = None
    map_num_proc: int = DEFAULT_MAP_NUM_PROC
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS


class DatasetPreparer:
    def __init__(self, options: PrepareOptions):
        self.options = options
        ensure_dir(RAW_DATA_ROOT)
        ensure_dir(PROCESSED_DATA_ROOT)
        self.cache_dir = RAW_DATA_ROOT / "_hf_cache"
        ensure_dir(self.cache_dir)
        self.hf_home = RAW_DATA_ROOT / "_hf_home"
        ensure_dir(self.hf_home)
        os.environ["HF_HOME"] = str(self.hf_home)
        os.environ["HF_DATASETS_CACHE"] = str(self.hf_home / "datasets")
        os.environ["HF_HUB_CACHE"] = str(self.hf_home / "hub")
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    def prepare_all(self) -> None:
        self.prepare_text_datasets()
        self.prepare_image_datasets()

    def prepare_selected(self, dataset_keys: List[str]) -> None:
        for key in dataset_keys:
            if key in TEXT_DATASETS:
                print(f"Preparing text dataset: {key}", flush=True)
                self._prepare_single_text_dataset(key)
            elif key in IMAGE_DATASETS:
                print(f"Preparing image dataset: {key}", flush=True)
                self._prepare_single_image_dataset(key)
            else:
                raise ValueError(f"Unknown dataset key: {key}")

    def prepare_text_datasets(self) -> None:
        for name in TEXT_DATASETS:
            print(f"Preparing text dataset: {name}", flush=True)
            self._prepare_single_text_dataset(name)

    def prepare_image_datasets(self) -> None:
        for name in IMAGE_DATASETS:
            print(f"Preparing image dataset: {name}", flush=True)
            self._prepare_single_image_dataset(name)

    def _prepare_single_text_dataset(self, dataset_key: str) -> None:
        cfg = TEXT_DATASETS[dataset_key].copy()
        if dataset_key == "automathtext" and self.options.automath_config:
            cfg["hf_name"] = self.options.automath_config
        self._validate_split_config(dataset_key, cfg)

        source = self._materialize_hf_dataset(
            dataset_key=dataset_key,
            hf_path=cfg["hf_path"],
            hf_name=cfg["hf_name"],
            split=cfg["source_split"],
            target_rows=cfg["materialize_n"],
            materialization=cfg["materialization"],
        )
        source = self._to_text_dataset(source, cfg["text_builder"])
        split_payload = self._build_split_payload(dataset_key, cfg, len(source))
        self._save_processed_splits(dataset_key, source, split_payload)

    def _prepare_single_image_dataset(self, dataset_key: str) -> None:
        cfg = IMAGE_DATASETS[dataset_key].copy()
        self._validate_split_config(dataset_key, cfg)
        source = self._materialize_hf_dataset(
            dataset_key=dataset_key,
            hf_path=cfg["hf_path"],
            hf_name=cfg["hf_name"],
            split=cfg["source_split"],
            target_rows=cfg["materialize_n"],
            materialization=cfg["materialization"],
        )
        source = self._to_image_text_dataset(dataset_key, source)
        split_payload = self._build_split_payload(dataset_key, cfg, len(source))
        self._save_processed_splits(dataset_key, source, split_payload)

    def _validate_split_config(self, dataset_key: str, cfg: Dict) -> None:
        expected_rows = cfg["train_full_n"] + cfg["eval_reserve_n"]
        if cfg["materialize_n"] < expected_rows:
            raise ValueError(
                f"{dataset_key}: materialize_n={cfg['materialize_n']} is smaller than "
                f"train_full_n + eval_reserve_n ({expected_rows})."
            )
        if cfg["shard1_n"] + cfg["shard2_n"] != cfg["train_full_n"]:
            raise ValueError(
                f"{dataset_key}: shard1_n + shard2_n must equal train_full_n "
                f"({cfg['shard1_n']} + {cfg['shard2_n']} != {cfg['train_full_n']})."
            )

    def _build_split_payload(self, dataset_key: str, cfg: Dict, materialized_count: int) -> Dict:
        expected_rows = cfg["train_full_n"] + cfg["eval_reserve_n"]
        if materialized_count < expected_rows:
            raise RuntimeError(
                f"{dataset_key}: materialized {materialized_count} rows, but {expected_rows} are required "
                "for train_full plus eval/nonmember reserve."
            )
        indices = self._shuffled_indices(materialized_count)
        train_full = indices[: cfg["train_full_n"]]
        shard1 = train_full[: cfg["shard1_n"]]
        shard2 = train_full[cfg["shard1_n"] : cfg["shard1_n"] + cfg["shard2_n"]]
        eval_reserve = indices[cfg["train_full_n"] : cfg["train_full_n"] + cfg["eval_reserve_n"]]
        splits = {
            "train_full": train_full,
            "shard1": shard1,
            "shard2": shard2,
        }
        if cfg["eval_reserve_n"]:
            splits["mi_reserve_not_in_training"] = eval_reserve
        checks = {
            "materialized_count": materialized_count,
            "train_full_count": len(train_full),
            "shard1_count": len(shard1),
            "shard2_count": len(shard2),
            "full_equals_union_shards": set(train_full) == set(shard1).union(set(shard2)),
            "shards_disjoint": set(shard1).isdisjoint(set(shard2)),
        }
        if cfg["eval_reserve_n"]:
            checks["mi_reserve_count"] = len(eval_reserve)
            checks["mi_reserve_disjoint_from_train_full"] = set(eval_reserve).isdisjoint(set(train_full))
        split_payload = {
            "dataset": dataset_key,
            "split_seed": self.options.split_seed,
            "source_split": cfg["source_split"],
            "source": {
                "hf_path": cfg["hf_path"],
                "hf_name": cfg["hf_name"],
                "materialization": cfg["materialization"],
            },
            "split_sizes": {
                "materialize_n": cfg["materialize_n"],
                "train_full_n": cfg["train_full_n"],
                "shard1_n": cfg["shard1_n"],
                "shard2_n": cfg["shard2_n"],
                "eval_reserve_n": cfg["eval_reserve_n"],
            },
            "splits": splits,
            "checks": checks,
        }
        self._assert_split_checks(dataset_key, split_payload)
        return split_payload

    def _assert_split_checks(self, dataset_key: str, split_payload: Dict) -> None:
        failed = [
            name
            for name, passed in split_payload["checks"].items()
            if not name.endswith("_count") and passed is False
        ]
        if failed:
            raise RuntimeError(f"{dataset_key}: split invariant check(s) failed: {', '.join(failed)}")

    def _save_processed_splits(self, dataset_key: str, dataset: Dataset, split_payload: Dict) -> None:
        ds_root = PROCESSED_DATA_ROOT / dataset_key
        ensure_dir(ds_root)

        for split_name, split_indices in split_payload["splits"].items():
            split_path = ds_root / split_name
            if split_path.exists() and not self.options.overwrite:
                continue
            if split_path.exists():
                shutil.rmtree(split_path)
            split_ds = dataset.select(split_indices)
            split_ds.save_to_disk(str(split_path))

        manifest_path = ds_root / "split_manifest.json"
        fingerprint_path = ds_root / "split_fingerprint.json"
        new_fingerprint = {
            "dataset": dataset_key,
            "split_seed": self.options.split_seed,
            "source_split": split_payload["source_split"],
            "source": split_payload["source"],
            "split_sizes": split_payload["split_sizes"],
            "split_hashes": {
                split_name: stable_indices_hash(indices)
                for split_name, indices in split_payload["splits"].items()
            },
        }

        if fingerprint_path.exists() and not self.options.overwrite:
            old_fingerprint = read_json(fingerprint_path)
            if old_fingerprint != new_fingerprint:
                raise RuntimeError(
                    f"Split fingerprint mismatch for {dataset_key}. "
                    "Use --overwrite to regenerate explicitly."
                )

        write_json(manifest_path, split_payload)
        write_json(fingerprint_path, new_fingerprint)
        self._update_registry(dataset_key, new_fingerprint)

    def _update_registry(self, dataset_key: str, fingerprint: Dict) -> None:
        registry_path = PROCESSED_DATA_ROOT / "dataset_registry.json"
        if registry_path.exists():
            registry = read_json(registry_path)
        else:
            registry = {}
        registry[dataset_key] = fingerprint
        write_json(registry_path, registry)

    def _materialize_hf_dataset(
        self,
        dataset_key: str,
        hf_path: str,
        split: str,
        target_rows: int,
        materialization: str,
        hf_name: Optional[str] = None,
    ) -> Dataset:
        ds_root = RAW_DATA_ROOT / dataset_key
        raw_path = ds_root / "materialized_dataset"
        materialization_meta_path = ds_root / "materialized_dataset_meta.json"
        ensure_dir(ds_root)

        if raw_path.exists() and not self.options.overwrite:
            cached = load_from_disk(str(raw_path))
            cached_has_enough_rows = len(cached) >= target_rows
            same_source = False
            if materialization_meta_path.exists():
                prev_meta = read_json(materialization_meta_path)
                same_source = (
                    prev_meta.get("hf_path") == hf_path
                    and prev_meta.get("hf_name") == hf_name
                    and prev_meta.get("split") == split
                    and prev_meta.get("materialization") == materialization
                )
            if cached_has_enough_rows and same_source:
                return cached
            raise RuntimeError(
                f"Cached materialized dataset for {dataset_key} is incompatible with the requested "
                "source/materialization or is too small. Use --overwrite to regenerate explicitly."
            )
        if raw_path.exists() and self.options.overwrite:
            shutil.rmtree(raw_path)

        print(
            f"{dataset_key}: materializing {target_rows} rows from {hf_path}"
            f"{'/' + hf_name if hf_name else ''} split={split} mode={materialization}",
            flush=True,
        )

        if materialization == "merged_train_test":
            ds = load_dataset(hf_path, name=hf_name, cache_dir=str(self.cache_dir))
            merged = concatenate_datasets([ds["train"], ds["test"]])
            merged = merged.shuffle(seed=self.options.split_seed)
            merged = merged.select(range(min(target_rows, len(merged))))
            merged.save_to_disk(str(raw_path))
            write_json(
                materialization_meta_path,
                {
                    "hf_path": hf_path,
                    "hf_name": hf_name,
                    "split": split,
                    "materialization": materialization,
                    "requested_target_rows": target_rows,
                    "materialized_rows": len(merged),
                },
            )
            return merged

        if materialization == "streaming_url_images":
            ds = self._materialize_url_image_dataset(
                dataset_key=dataset_key,
                hf_path=hf_path,
                hf_name=hf_name,
                split=split,
                target_rows=target_rows,
            )
            ds.save_to_disk(str(raw_path))
            write_json(
                materialization_meta_path,
                {
                    "hf_path": hf_path,
                    "hf_name": hf_name,
                    "split": split,
                    "materialization": materialization,
                    "requested_target_rows": target_rows,
                    "materialized_rows": len(ds),
                },
            )
            return ds

        if materialization == "streaming_rows":
            iterable = load_dataset(
                hf_path,
                name=hf_name,
                split=split,
                cache_dir=str(self.cache_dir),
                streaming=True,
            )
            rows = []
            for idx, row in enumerate(iterable):
                if idx >= target_rows:
                    break
                rows.append(row)
            ds = Dataset.from_list(rows)
            ds.save_to_disk(str(raw_path))
            write_json(
                materialization_meta_path,
                {
                    "hf_path": hf_path,
                    "hf_name": hf_name,
                    "split": split,
                    "materialization": materialization,
                    "requested_target_rows": target_rows,
                    "materialized_rows": len(ds),
                },
            )
            return ds

        if materialization != "hf_rows":
            raise ValueError(f"Unknown materialization mode for {dataset_key}: {materialization}")

        ds = load_dataset(hf_path, name=hf_name, split=split, cache_dir=str(self.cache_dir))
        ds = ds.shuffle(seed=self.options.split_seed)
        ds = ds.select(range(min(target_rows, len(ds))))
        ds.save_to_disk(str(raw_path))
        write_json(
            materialization_meta_path,
            {
                "hf_path": hf_path,
                "hf_name": hf_name,
                "split": split,
                "materialization": materialization,
                "requested_target_rows": target_rows,
                "materialized_rows": len(ds),
            },
        )
        return ds

    def _to_text_dataset(self, ds: Dataset, builder: str) -> Dataset:
        if builder == "text":
            return ds.select_columns(["text"])
        if builder == "title_abstract":
            required_cols = {"title", "abstract"}
            missing_cols = sorted(required_cols.difference(ds.column_names))
            if missing_cols:
                raise ValueError(
                    "title_abstract builder requires columns "
                    f"{sorted(required_cols)}; missing {missing_cols}"
                )
            return ds.map(
                lambda x: {"text": f"{x['title']}[SEP]{x['abstract']}"},
                remove_columns=[c for c in ds.column_names if c not in {"title", "abstract"}],
            ).select_columns(["text"])
        if builder == "prompt_story":
            return ds.map(
                lambda x: {"text": f"{x['prompt']}[SEP]{x['story']}"},
                remove_columns=[c for c in ds.column_names if c not in {"prompt", "story"}],
            ).select_columns(["text"])
        raise ValueError(f"Unknown text builder: {builder}")

    def _to_image_text_dataset(self, dataset_key: str, ds: Dataset) -> Dataset:
        # Normalize caption column name to `text` for downstream training.
        if "caption" in ds.column_names and "text" not in ds.column_names:
            ds = ds.rename_column("caption", "text")
        if "image" not in ds.column_names and "url" in ds.column_names:
            ds = self._download_images_from_urls(ds, dataset_key=dataset_key)
        if "image" in ds.column_names:
            map_kwargs = {"desc": f"{dataset_key}: normalize images"}
            if self.options.map_num_proc and self.options.map_num_proc > 1:
                map_kwargs["num_proc"] = self.options.map_num_proc
            ds = ds.map(self._normalize_image_row, **map_kwargs)
        keep = [c for c in ds.column_names if c in {"image", "text"}]
        return ds.select_columns(keep)

    @staticmethod
    def _normalize_image_row(row: Dict) -> Dict:
        image = row.get("image")
        if hasattr(image, "convert"):
            row["image"] = image.convert("RGB")
        return row

    def _materialize_url_image_dataset(
        self,
        dataset_key: str,
        hf_path: str,
        split: str,
        target_rows: int,
        hf_name: Optional[str] = None,
    ) -> Dataset:
        iterable = load_dataset(
            hf_path,
            name=hf_name,
            split=split,
            cache_dir=str(self.cache_dir),
            streaming=True,
        )
        image_root = RAW_DATA_ROOT / dataset_key / "images"
        ensure_dir(image_root)
        paths = []
        texts = []
        attempted = 0
        skipped = 0
        pending = []
        print(f"{dataset_key}: downloading {target_rows} valid images from {hf_path}/{split} URLs", flush=True)
        for source_idx, row in enumerate(iterable):
            if len(paths) >= target_rows:
                break
            attempted += 1
            url = row.get("url")
            caption = row.get("caption", row.get("text", ""))
            if not url:
                skipped += 1
                continue
            out_path = image_root / f"{source_idx:08d}.jpg"
            pending.append((source_idx, url, caption, out_path))
            if len(pending) >= URL_DOWNLOAD_BATCH_SIZE:
                skipped = self._flush_url_image_batch(dataset_key, pending, paths, texts, skipped, attempted, target_rows)
                pending = []
        if pending and len(paths) < target_rows:
            skipped = self._flush_url_image_batch(dataset_key, pending, paths, texts, skipped, attempted, target_rows)
        if len(paths) < target_rows:
            raise RuntimeError(
                f"{dataset_key}: only materialized {len(paths)} valid image URLs; "
                f"{target_rows} are required after scanning {attempted} URLs."
            )
        rebuilt = Dataset.from_dict({"image": paths, "text": texts})
        return rebuilt.cast_column("image", Image())

    def _flush_url_image_batch(
        self,
        dataset_key: str,
        pending: List[Tuple[int, str, str, Path]],
        paths: List[str],
        texts: List[str],
        skipped: int,
        attempted: int,
        target_rows: int,
    ) -> int:
        results = []
        workers = self.options.download_workers if self.options.download_workers > 0 else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_record = {
                executor.submit(self._download_image, url, out_path): (source_idx, caption, out_path)
                for source_idx, url, caption, out_path in pending
            }
            for future in as_completed(future_to_record):
                source_idx, caption, out_path = future_to_record[future]
                results.append((source_idx, caption, out_path, future.result()))

        for _, caption, out_path, ok in sorted(results, key=lambda item: item[0]):
            if not ok:
                skipped += 1
                continue
            if len(paths) < target_rows:
                paths.append(str(out_path))
                texts.append(caption)

        print(
            f"{dataset_key}: kept {len(paths)}/{target_rows} images "
            f"after scanning {attempted} URLs ({skipped} skipped)",
            flush=True,
        )
        return skipped

    def _download_images_from_urls(self, ds: Dataset, dataset_key: str) -> Dataset:
        image_root = RAW_DATA_ROOT / dataset_key / "images"
        ensure_dir(image_root)
        paths = []
        texts = []
        for idx, row in enumerate(ds):
            url = row.get("url")
            caption = row.get("text", row.get("caption", ""))
            if not url:
                continue
            out_path = image_root / f"{idx:08d}.jpg"
            if not self._download_image(url, out_path):
                continue
            paths.append(str(out_path))
            texts.append(caption)
        rebuilt = Dataset.from_dict({"image": paths, "text": texts})
        return rebuilt.cast_column("image", Image())

    def _download_image(self, url: str, out_path: Path) -> bool:
        if out_path.exists():
            return True
        try:
            resp = requests.get(
                url,
                timeout=HTTP_TIMEOUT,
                headers={"User-Agent": "copyright-pipeline-data-prep/1.0"},
            )
            resp.raise_for_status()
            img = PILImage.open(BytesIO(resp.content))
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            img = img.convert("RGB")
            img.save(out_path, format="JPEG", quality=95)
            return True
        except Exception:
            return False

    def _shuffled_indices(self, n: int) -> List[int]:
        indices = list(range(n))
        random.Random(self.options.split_seed).shuffle(indices)
        return indices

