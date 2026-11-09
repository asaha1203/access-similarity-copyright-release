import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import PROCESSED_DATA_ROOT
from .inference_common import (
    LoadedTextModel,
    bleu_avg_1_to4,
    bleu1,
    build_run_output_dir,
    cpk_shared_path_scores_batch,
    collect_text_prompts,
    compute_mia_metrics_reference_style,
    compute_minkpp_token_scores,
    dump_csv,
    dump_jsonl,
    diffusion_prepare_latent,
    diffusion_logprob_ratios_from_realized_path_batch,
    diffusion_logprob_trajectory_from_xt_batch,
    generate_text_candidates,
    load_diffusion_model,
    load_text_model,
    load_yaml,
    minkpp_token_scores_from_token_ids_batch,
    normalized_levenshtein,
    now_iso,
    prepare_continuation_first_batch,
    read_jsonl,
    realized_token_logprobs,
    realized_token_logprobs_batch,
    realized_token_logprobs_from_token_ids_batch,
    resolve_run_dir,
    safe_quantile,
    tensor_to_image,
    pia_scores_from_latents_batch,
    validate_cpk_diffusion_triplet,
    validate_cpk_text_triplet,
    write_common_artifacts,
    write_json,
)
from .inference_cache import (
    infer_model_type,
    infer_modality,
    load_cache,
    make_cache,
    resolve_output_path,
    save_cache,
)
from .prometheus_v2 import DEFAULT_FLUENCY_RUBRIC, PrometheusV2Scorer


def _tensor_or_empty(values: Sequence[Any], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=dtype)
    return torch.tensor(values, dtype=dtype)


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


DRA_TEXT_DISTANCE_KEYS = ("normalized_levenshtein", "bleu", "bertscore_f1", "rouge_l")
DRA_IMAGE_DISTANCE_KEYS = ("normalized_l2", "lpips", "clip", "ssim", "dreamsim")
DRA_DISTANCE_KEYS = DRA_IMAGE_DISTANCE_KEYS + DRA_TEXT_DISTANCE_KEYS
HIGHER_IS_CLOSER_DRA_DISTANCE_KEYS: set[str] = set()
_CLIP_COMPONENTS_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}
_LPIPS_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_DREAMSIM_COMPONENTS_CACHE: Dict[Tuple[str, str, str, str, bool], Tuple[Any, Any]] = {}
_ROUGE_SCORER_CACHE: Dict[Tuple[bool, str], Any] = {}
_BERT_SCORER_CACHE: Dict[Tuple[str, str, bool, bool, Optional[int]], Any] = {}
_DREAMSIM_DTYPE = torch.float32


def _text_runtime_model_kwargs(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    runtime_cfg = config.get("runtime", {})
    text_dtype = str(runtime_cfg.get("text_dtype", "fp16")).lower()
    torch_dtype = None
    if str(device).startswith("cuda"):
        if text_dtype in {"fp16", "float16"}:
            torch_dtype = torch.float16
        elif text_dtype in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif text_dtype in {"fp32", "float32"}:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported runtime.text_dtype: {text_dtype}")
    default_compile = str(device).startswith("cuda")
    return {
        "torch_dtype": torch_dtype,
        "compile_model": bool(runtime_cfg.get("text_compile_model", default_compile)),
        "compile_mode": str(runtime_cfg.get("text_compile_mode", "reduce-overhead")),
        "merge_lora": bool(runtime_cfg.get("text_merge_lora", True)),
        "attention_implementation": runtime_cfg.get("text_attention_implementation", "auto"),
    }


def _monitor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime_cfg = dict(config.get("runtime", {}))
    monitor_cfg_raw = runtime_cfg.get("monitor", config.get("monitor", {}))
    monitor_cfg = dict(monitor_cfg_raw) if isinstance(monitor_cfg_raw, dict) else {}
    log_interval_batches = max(1, int(monitor_cfg.get("log_interval_batches", 10)))
    return {
        "enabled": bool(monitor_cfg.get("enabled", True)),
        "show_rate": bool(monitor_cfg.get("show_rate", True)),
        "log_interval_batches": log_interval_batches,
        "leave_bars": monitor_cfg.get("leave_bars"),
        "disable_diffusers_inner_bar": bool(monitor_cfg.get("disable_diffusers_inner_bar", False)),
        "stage_prints": bool(monitor_cfg.get("stage_prints", True)),
    }


def _monitor_leave(monitor_cfg: Dict[str, Any], default: bool) -> bool:
    configured = monitor_cfg.get("leave_bars")
    if configured is None:
        return default
    return bool(configured)


def _monitor_print(monitor_cfg: Dict[str, Any], message: str) -> None:
    if bool(monitor_cfg.get("stage_prints", True)):
        print(message)


def _rows_per_second(start_time: float, processed_rows: int) -> float:
    elapsed = max(time.perf_counter() - start_time, 1.0e-9)
    return float(processed_rows / elapsed)


def _require_optional_dependency(import_name: str, package_name: str, metric_name: str) -> Any:
    try:
        module = __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        raise ImportError(
            f"DRA distance metric '{metric_name}' requires optional dependency '{package_name}'. "
            f"Install requirements.txt or remove '{metric_name}' from dra_distance metrics."
        ) from exc
    return module


def _dra_distance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("dra_distance", config.get("distance_metrics", {})))


def _normalize_dra_metric_name(metric: str) -> str:
    key = str(metric).strip().lower().replace("-", "_")
    aliases = {
        "normalized_l2_distance": "normalized_l2",
        "l2_normalized": "normalized_l2",
        "clip_distance": "clip",
        "clipscore": "clip",
        "bert_score_f1": "bertscore_f1",
        "bert_score": "bertscore_f1",
        "rouge": "rouge_l",
        "rouge_l_f1": "rouge_l",
        "rouge_l_distance": "rouge_l",
        "ssim_distance": "ssim",
        "dreamsim_distance": "dreamsim",
    }
    return aliases.get(key, key)


def _dra_distance_metrics(config: Dict[str, Any], modality: str) -> Tuple[str, ...]:
    distance_cfg = _dra_distance_config(config)
    if distance_cfg.get("enabled", True) is False:
        return ()
    modality_key = "text_metrics" if modality == "text" else "image_metrics"
    configured = distance_cfg.get(modality_key, distance_cfg.get("metrics"))
    if configured is None:
        configured = DRA_TEXT_DISTANCE_KEYS if modality == "text" else DRA_IMAGE_DISTANCE_KEYS
    if isinstance(configured, str):
        configured = [configured]
    allowed = set(DRA_TEXT_DISTANCE_KEYS if modality == "text" else DRA_IMAGE_DISTANCE_KEYS)
    metrics: List[str] = []
    for metric in configured:
        normalized = _normalize_dra_metric_name(metric)
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported DRA {modality} distance metric '{metric}'. "
                f"Supported metrics: {sorted(allowed)}"
            )
        if normalized not in metrics:
            metrics.append(normalized)
    return tuple(metrics)


def _image_distance_size(config: Dict[str, Any]) -> int:
    distance_cfg = _dra_distance_config(config)
    return int(distance_cfg.get("image_size", 224))


def _dra_distance_batch_size(config: Dict[str, Any], default: int = 16) -> int:
    distance_cfg = _dra_distance_config(config)
    return max(1, int(distance_cfg.get("batch_size", default)))


def _dra_metric_batch_size(config: Dict[str, Any], metric: str, default: int = 16) -> int:
    distance_cfg = _dra_distance_config(config)
    return max(1, int(distance_cfg.get(f"{metric}_batch_size", distance_cfg.get("batch_size", default))))


def _text_distance_chunk_size(config: Dict[str, Any]) -> int:
    distance_cfg = _dra_distance_config(config)
    configured = distance_cfg.get("text_chunk_size")
    if configured is not None:
        return max(1, int(configured))
    return max(32, _dra_distance_batch_size(config, default=256))


def _text_distance_num_workers(config: Dict[str, Any]) -> int:
    distance_cfg = _dra_distance_config(config)
    return max(1, int(distance_cfg.get("text_num_workers", 1)))


def _map_with_optional_workers(fn, values: Sequence[Any], num_workers: int) -> List[Any]:
    if not values:
        return []
    if num_workers <= 1:
        return [fn(value) for value in values]
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(fn, values))


def _image_distance_chunk_size(config: Dict[str, Any], metrics: Sequence[str]) -> int:
    distance_cfg = _dra_distance_config(config)
    if distance_cfg.get("image_chunk_size") is not None:
        return max(1, int(distance_cfg["image_chunk_size"]))
    base = _dra_distance_batch_size(config)
    return max(_dra_metric_batch_size(config, metric, default=base) for metric in metrics) if metrics else base


def _distance_compute_dtype(config: Dict[str, Any], device: torch.device) -> torch.dtype:
    if not str(device).startswith("cuda"):
        return torch.float32
    distance_cfg = _dra_distance_config(config)
    raw = str(distance_cfg.get("torch_dtype", distance_cfg.get("dtype", "float16"))).lower()
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError("dra_distance.dtype must be 'float16'/'fp16' or 'bfloat16'/'bf16' on CUDA.")


def _dtype_cache_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _unit_distance(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float32).clamp(0.0, 1.0)


def _to_device_batch(batch: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(batch, dict) or hasattr(batch, "items"):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if dtype is not None and value.is_floating_point() else None
                moved[key] = value.to(device=device, dtype=target_dtype)
            else:
                moved[key] = value
        return moved
    if hasattr(batch, "to"):
        try:
            return batch.to(device=device)
        except TypeError:
            return batch.to(device)
    return batch


def _resize_image_batch(images: torch.Tensor, size: int) -> torch.Tensor:
    if images.shape[-2:] == (size, size):
        return images
    return torch.nn.functional.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def _pil_images_from_chw_batch(images: torch.Tensor) -> List[Any]:
    from PIL import Image  # pylint: disable=import-outside-toplevel

    pil_images: List[Any] = []
    for image in images.detach().cpu().clamp(0.0, 1.0):
        arr = (image.to(torch.float32).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        pil_images.append(Image.fromarray(arr))
    return pil_images


def _cached_pil_images_from_chw_batch(
    images: torch.Tensor,
    keys: Sequence[Hashable],
    cache: Dict[Hashable, Any],
) -> List[Any]:
    pil_images: List[Any] = [None] * len(keys)
    missing_indices: List[int] = []
    missing_keys: List[Hashable] = []
    seen_missing: set[Hashable] = set()
    for idx, key in enumerate(keys):
        cached = cache.get(key)
        if cached is not None:
            pil_images[idx] = cached
            continue
        if key in seen_missing:
            continue
        missing_indices.append(idx)
        missing_keys.append(key)
        seen_missing.add(key)
    if missing_indices:
        computed = _pil_images_from_chw_batch(images[missing_indices])
        for key, image in zip(missing_keys, computed):
            cache[key] = image
    for idx, key in enumerate(keys):
        if pil_images[idx] is None:
            pil_images[idx] = cache[key]
    return pil_images


def _as_chw_float_image(image: torch.Tensor) -> torch.Tensor:
    tensor = image.detach().to(torch.float32).cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected a single image tensor, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(tensor.shape)}")
    if tensor.shape[0] not in {1, 3}:
        raise ValueError(f"Expected image tensor with 1 or 3 channels, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor.clamp(0.0, 1.0)


def _prepare_image_distance_batch(
    images: Sequence[torch.Tensor],
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch = torch.stack([_as_chw_float_image(x) for x in images]).to(device=device, dtype=dtype)
    return _resize_image_batch(batch, size).clamp(0.0, 1.0)


def _get_clip_components(model_name: str, device: torch.device, dtype: torch.dtype) -> Tuple[Any, Any]:
    from transformers import CLIPModel, CLIPProcessor  # pylint: disable=import-outside-toplevel

    cache_key = (model_name, str(device), _dtype_cache_name(dtype))
    cached = _CLIP_COMPONENTS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    processor = CLIPProcessor.from_pretrained(model_name)
    model_kwargs: Dict[str, Any] = {}
    if str(device).startswith("cuda"):
        model_kwargs["torch_dtype"] = dtype
    model = CLIPModel.from_pretrained(model_name, **model_kwargs).to(device).eval()
    _CLIP_COMPONENTS_CACHE[cache_key] = (processor, model)
    return processor, model


def _clip_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            if attr == "last_hidden_state" and value.ndim >= 3:
                return value[:, 0]
            return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                if value.ndim >= 3:
                    return value[:, 0]
                return value
    raise TypeError(f"CLIP image feature output did not contain a tensor: {type(output).__name__}")


def _get_lpips_model(backbone: str, device: torch.device, dtype: torch.dtype) -> Any:
    lpips_module = _require_optional_dependency("lpips", "lpips", "lpips")
    cache_key = (backbone, str(device), _dtype_cache_name(dtype))
    cached = _LPIPS_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model = lpips_module.LPIPS(net=backbone).to(device).eval()
    if str(device).startswith("cuda"):
        model = model.to(dtype=dtype)
    _LPIPS_MODEL_CACHE[cache_key] = model
    return model


def _compute_lpips_values(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    distance_cfg = _dra_distance_config(config)
    backbone = str(distance_cfg.get("lpips_backbone", "alex"))
    dtype = _distance_compute_dtype(config, device)
    model = _get_lpips_model(backbone, device, dtype)
    batch_size = _dra_metric_batch_size(config, "lpips")
    values: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            cand = (candidates[start : start + batch_size].to(device=device, dtype=dtype) * 2.0) - 1.0
            tgt = (targets[start : start + batch_size].to(device=device, dtype=dtype) * 2.0) - 1.0
            values.append(_unit_distance(model(cand, tgt).detach().flatten().cpu()))
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)


def _compute_ssim_distance_values(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure  # pylint: disable=import-outside-toplevel
    except ImportError:
        from torchmetrics.image.ssim import (  # type: ignore[no-redef] # pylint: disable=import-outside-toplevel
            StructuralSimilarityIndexMeasure,
        )

    dtype = _distance_compute_dtype(config, device)
    metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    values: List[torch.Tensor] = []
    batch_size = _dra_metric_batch_size(config, "ssim")
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            cand = candidates[start : start + batch_size].to(device=device, dtype=dtype)
            tgt = targets[start : start + batch_size].to(device=device, dtype=dtype)
            scores = metric(cand, tgt).flatten().clamp(0.0, 1.0)
            values.append(_unit_distance((1.0 - scores).detach().cpu()))
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)


def _compute_clip_image_features(
    images: Sequence[Any],
    config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    distance_cfg = _dra_distance_config(config)
    model_name = str(distance_cfg.get("clip_model", "openai/clip-vit-base-patch32"))
    processor, model = _get_clip_components(model_name, device, dtype)
    batch_size = _dra_metric_batch_size(config, "clip", default=8)
    features: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            image_batch = list(images[start : start + batch_size])
            inputs = _to_device_batch(processor(images=image_batch, return_tensors="pt"), device, dtype)
            batch_features = _clip_feature_tensor(model.get_image_features(**inputs))
            batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=-1)
            features.append(batch_features.detach().cpu().to(torch.float32))
    return torch.cat(features) if features else torch.empty(0, dtype=torch.float32)


def _cached_feature_batch(
    cache: Dict[Hashable, torch.Tensor],
    keys: Sequence[Hashable],
    values: Sequence[Any],
    compute_fn: Any,
) -> torch.Tensor:
    missing_positions: List[int] = []
    missing_keys: List[Hashable] = []
    seen_missing: set[Hashable] = set()
    for idx, key in enumerate(keys):
        if key in cache or key in seen_missing:
            continue
        missing_positions.append(idx)
        missing_keys.append(key)
        seen_missing.add(key)
    if missing_keys:
        computed = compute_fn([values[idx] for idx in missing_positions])
        for key, feature in zip(missing_keys, computed):
            cache[key] = feature.detach().cpu().to(torch.float32)
    return torch.stack([cache[key] for key in keys])


def _compute_clip_distance_values(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
    candidate_images: Optional[Sequence[Any]] = None,
    target_images: Optional[Sequence[Any]] = None,
    target_cache_keys: Optional[Sequence[Hashable]] = None,
    target_feature_cache: Optional[Dict[Hashable, torch.Tensor]] = None,
) -> torch.Tensor:
    del targets
    dtype = _distance_compute_dtype(config, device)
    cand_images = candidate_images if candidate_images is not None else _pil_images_from_chw_batch(candidates)
    if target_images is None:
        raise ValueError("CLIP distance requires target PIL images.")
    cand_features = _compute_clip_image_features(cand_images, config, device, dtype)
    if target_cache_keys is not None and target_feature_cache is not None:
        tgt_features = _cached_feature_batch(
            target_feature_cache,
            target_cache_keys,
            target_images,
            lambda image_batch: _compute_clip_image_features(image_batch, config, device, dtype),
        )
    else:
        tgt_features = _compute_clip_image_features(target_images, config, device, dtype)
    similarity = (cand_features * tgt_features).sum(dim=-1).clamp(-1.0, 1.0)
    return _unit_distance(((1.0 - similarity) * 0.5).detach().cpu())


def _get_dreamsim_components(config: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Tuple[Any, Any]:
    dreamsim_module = _require_optional_dependency("dreamsim", "dreamsim", "dreamsim")
    factory = getattr(dreamsim_module, "dreamsim")
    distance_cfg = _dra_distance_config(config)
    model_name = str(distance_cfg.get("dreamsim_model", distance_cfg.get("dreamsim_type", "ensemble")))
    cache_dir = str(distance_cfg.get("dreamsim_cache_dir", ""))
    pretrained = bool(distance_cfg.get("dreamsim_pretrained", True))
    use_patch_model = bool(distance_cfg.get("dreamsim_use_patch_model", False))
    del dtype
    cache_key = (model_name, cache_dir, str(device), _dtype_cache_name(_DREAMSIM_DTYPE), use_patch_model)
    cached = _DREAMSIM_COMPONENTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kwargs: Dict[str, Any] = {"pretrained": pretrained}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if model_name and model_name not in {"default", "ensemble"}:
        kwargs["dreamsim_type"] = model_name
    if use_patch_model:
        kwargs["use_patch_model"] = True
    try:
        model, preprocess = factory(device=str(device), **kwargs)
    except TypeError:
        model, preprocess = factory(**kwargs)
    if hasattr(model, "to"):
        try:
            # DreamSim's ensemble keeps some projection weights in fp32 internally;
            # casting the model/input batch to fp16 can cause dtype mismatches.
            model = model.to(device=device)
        except TypeError:
            model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    _DREAMSIM_COMPONENTS_CACHE[cache_key] = (model, preprocess)
    return model, preprocess


def _dreamsim_distance_from_embeddings(model: Any, cand_embed: torch.Tensor, tgt_embed: torch.Tensor) -> torch.Tensor:
    if getattr(model, "feat_type_list", [None])[0] == "cls_patch":
        cls_a = cand_embed[:, 0]
        patch_a = cand_embed[:, 1:]
        cls_b = tgt_embed[:, 0]
        patch_b = tgt_embed[:, 1:]
        n = patch_a.shape[0]
        side = int(patch_a.shape[1] ** 0.5)
        patch_a = torch.nn.functional.adaptive_avg_pool2d(
            patch_a.reshape(n, side, side, -1).permute(0, 3, 1, 2), (1, 1)
        ).squeeze()
        patch_b = torch.nn.functional.adaptive_avg_pool2d(
            patch_b.reshape(n, side, side, -1).permute(0, 3, 1, 2), (1, 1)
        ).squeeze()
        if patch_a.ndim == 1:
            patch_a = patch_a.unsqueeze(0)
        if patch_b.ndim == 1:
            patch_b = patch_b.unsqueeze(0)
        cand_embed = torch.cat((cls_a, patch_a), dim=-1)
        tgt_embed = torch.cat((cls_b, patch_b), dim=-1)
    return (1.0 - torch.nn.functional.cosine_similarity(cand_embed, tgt_embed, dim=-1)) * 0.5


def _dreamsim_preprocessed_image(preprocess: Any, image: Any) -> torch.Tensor:
    tensor = preprocess(image)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"DreamSim preprocess returned an unsupported image tensor shape: {tuple(tensor.shape)}")
    return tensor


def _dreamsim_embedding_tensor(output: Any, batch_count: int) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    if output.ndim == 1 and batch_count == 1:
        output = output.unsqueeze(0)
    if output.ndim == 0:
        output = output.reshape(1, 1)
    return output


def _compute_dreamsim_embeddings(
    images: Sequence[Any],
    config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    del dtype
    model, preprocess = _get_dreamsim_components(config, device, _DREAMSIM_DTYPE)
    batch_size = _dra_metric_batch_size(config, "dreamsim", default=8)
    embeddings: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            image_batch = list(images[start : start + batch_size])
            batch = torch.stack([_dreamsim_preprocessed_image(preprocess, image) for image in image_batch])
            batch = batch.to(device=device, dtype=_DREAMSIM_DTYPE)
            embedded = _dreamsim_embedding_tensor(model.embed(batch), len(image_batch))
            embeddings.append(embedded.detach().cpu().to(torch.float32))
    return torch.cat(embeddings) if embeddings else torch.empty(0, dtype=torch.float32)


def _compute_dreamsim_direct_distance_values(
    config: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    candidate_images: Sequence[Any],
    target_images: Sequence[Any],
) -> torch.Tensor:
    del dtype
    model, preprocess = _get_dreamsim_components(config, device, _DREAMSIM_DTYPE)
    batch_size = _dra_metric_batch_size(config, "dreamsim", default=8)
    values: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(candidate_images), batch_size):
            cand = torch.stack([_dreamsim_preprocessed_image(preprocess, image) for image in candidate_images[start : start + batch_size]])
            tgt = torch.stack([_dreamsim_preprocessed_image(preprocess, image) for image in target_images[start : start + batch_size]])
            cand = cand.to(device=device, dtype=_DREAMSIM_DTYPE)
            tgt = tgt.to(device=device, dtype=_DREAMSIM_DTYPE)
            distance = model(cand, tgt)
            if not isinstance(distance, torch.Tensor):
                distance = torch.as_tensor(distance, device=device)
            values.append(_unit_distance((distance.detach().flatten().cpu()) * 0.5))
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)


def _compute_dreamsim_distance_values(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
    candidate_images: Sequence[Any],
    target_images: Sequence[Any],
    target_cache_keys: Optional[Sequence[Hashable]] = None,
    target_embedding_cache: Optional[Dict[Hashable, torch.Tensor]] = None,
) -> torch.Tensor:
    del candidates, targets
    dtype = _distance_compute_dtype(config, device)
    model, preprocess = _get_dreamsim_components(config, device, dtype)
    del preprocess
    if not hasattr(model, "embed"):
        return _compute_dreamsim_direct_distance_values(config, device, dtype, candidate_images, target_images)
    cand_embeddings = _compute_dreamsim_embeddings(candidate_images, config, device, dtype)
    if target_cache_keys is not None and target_embedding_cache is not None:
        tgt_embeddings = _cached_feature_batch(
            target_embedding_cache,
            target_cache_keys,
            target_images,
            lambda image_batch: _compute_dreamsim_embeddings(image_batch, config, device, dtype),
        )
    else:
        tgt_embeddings = _compute_dreamsim_embeddings(target_images, config, device, dtype)
    return _unit_distance(_dreamsim_distance_from_embeddings(model, cand_embeddings, tgt_embeddings).detach().cpu())


def _compute_image_distance_values(
    candidates: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
    metrics: Sequence[str],
    target_cache_keys: Optional[Sequence[Hashable]] = None,
    image_cache: Optional[Dict[str, Dict[Hashable, Any]]] = None,
) -> Dict[str, List[float]]:
    if not candidates:
        return {metric: [] for metric in metrics}
    if target_cache_keys is not None and len(target_cache_keys) != len(targets):
        raise ValueError("target_cache_keys must match the number of target tensors.")
    size = _image_distance_size(config)
    dtype = _distance_compute_dtype(config, device)
    cand_batch = _prepare_image_distance_batch(candidates, size, device, dtype)
    target_batch = _prepare_image_distance_batch(targets, size, device, dtype)
    needs_pil = "clip" in metrics or "dreamsim" in metrics
    if image_cache is None:
        image_cache = {}
    cand_images = _pil_images_from_chw_batch(cand_batch) if needs_pil else None
    target_images = None
    if needs_pil:
        if target_cache_keys is not None:
            target_images = _cached_pil_images_from_chw_batch(
                target_batch,
                target_cache_keys,
                image_cache.setdefault("target_pil", {}),
            )
        else:
            target_images = _pil_images_from_chw_batch(target_batch)
    values: Dict[str, List[float]] = {}
    if "normalized_l2" in metrics:
        normalized_l2 = torch.sqrt(((cand_batch - target_batch) ** 2).mean(dim=(1, 2, 3)))
        values["normalized_l2"] = [float(x) for x in _unit_distance(normalized_l2.detach().cpu()).tolist()]
    if "ssim" in metrics:
        ssim_values = _compute_ssim_distance_values(cand_batch, target_batch, config, device)
        values["ssim"] = [float(x) for x in ssim_values.tolist()]
    if "lpips" in metrics:
        lpips_values = _compute_lpips_values(cand_batch, target_batch, config, device)
        values["lpips"] = [float(x) for x in lpips_values.tolist()]
    if "clip" in metrics:
        clip_values = _compute_clip_distance_values(
            cand_batch,
            target_batch,
            config,
            device,
            cand_images,
            target_images,
            target_cache_keys,
            image_cache.setdefault("clip_target_features", {}),
        )
        values["clip"] = [float(x) for x in clip_values.tolist()]
    if "dreamsim" in metrics:
        if cand_images is None or target_images is None:
            cand_images = _pil_images_from_chw_batch(cand_batch)
            if target_cache_keys is not None:
                target_images = _cached_pil_images_from_chw_batch(
                    target_batch,
                    target_cache_keys,
                    image_cache.setdefault("target_pil", {}),
                )
            else:
                target_images = _pil_images_from_chw_batch(target_batch)
        dreamsim_values = _compute_dreamsim_distance_values(
            cand_batch,
            target_batch,
            config,
            device,
            cand_images,
            target_images,
            target_cache_keys,
            image_cache.setdefault("dreamsim_target_embeddings", {}),
        )
        values["dreamsim"] = [float(x) for x in dreamsim_values.tolist()]
    return values


def _get_rouge_l_scorer(config: Dict[str, Any]) -> Any:
    rouge_scorer_module = _require_optional_dependency("rouge_score.rouge_scorer", "rouge-score", "rouge_l")
    distance_cfg = _dra_distance_config(config)
    use_stemmer = bool(distance_cfg.get("rouge_l_use_stemmer", False))
    cache_key = (use_stemmer, "rougeL")
    cached = _ROUGE_SCORER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    _ROUGE_SCORER_CACHE[cache_key] = scorer
    return scorer


def _get_bertscore_scorer(config: Dict[str, Any], device: torch.device) -> Any:
    bert_score_module = _require_optional_dependency("bert_score", "bert-score", "bertscore_f1")
    if not hasattr(bert_score_module, "BERTScorer"):
        return None
    distance_cfg = _dra_distance_config(config)
    model_or_lang = str(distance_cfg.get("bertscore_model") or distance_cfg.get("bertscore_lang", "en"))
    rescale = bool(distance_cfg.get("bertscore_rescale_with_baseline", False))
    half_precision = bool(distance_cfg.get("bertscore_half_precision", False))
    max_length = distance_cfg.get("bertscore_max_length")
    max_length_int = int(max_length) if max_length is not None else None
    cache_key = (model_or_lang, str(device), rescale, half_precision, max_length_int)
    cached = _BERT_SCORER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    kwargs: Dict[str, Any] = {
        "device": str(device),
        "rescale_with_baseline": rescale,
    }
    if distance_cfg.get("bertscore_model"):
        kwargs["model_type"] = str(distance_cfg["bertscore_model"])
    else:
        kwargs["lang"] = str(distance_cfg.get("bertscore_lang", "en"))
    scorer = bert_score_module.BERTScorer(**kwargs)
    if max_length_int is not None:
        tokenizer = getattr(scorer, "_tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
            try:
                tokenizer.model_max_length = int(max_length_int)
            except Exception:
                pass
    if half_precision and str(device).startswith("cuda"):
        model = getattr(scorer, "_model", None)
        if model is not None and hasattr(model, "half"):
            try:
                model.half()
            except Exception:
                pass
    _BERT_SCORER_CACHE[cache_key] = scorer
    return scorer


def _add_text_dra_distances(
    rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    started_at = time.perf_counter()
    timings: Dict[str, float] = {}
    monitor_cfg = _monitor_config(config)
    metrics = _dra_distance_metrics(config, "text")
    if not metrics or not rows:
        timings["total_s"] = max(0.0, time.perf_counter() - started_at)
        return timings
    scalar_metrics = {"normalized_levenshtein", "bleu", "rouge_l"}.intersection(metrics)
    if scalar_metrics:
        rouge_scorer = _get_rouge_l_scorer(config) if "rouge_l" in scalar_metrics else None
        chunk_size = _text_distance_chunk_size(config)
        num_workers = _text_distance_num_workers(config)
        scalar_start = time.perf_counter()
        _monitor_print(
            monitor_cfg,
            (
                f"[DRA text distances/scalar] metrics={','.join(sorted(scalar_metrics))} "
                f"rows={len(rows)} chunk_size={chunk_size} num_workers={num_workers}"
            ),
        )
        iterator = tqdm(
            range(0, len(rows), chunk_size),
            total=max(1, (len(rows) + chunk_size - 1) // chunk_size),
            desc="DRA text distances [scalar]",
            unit="batch",
            leave=_monitor_leave(monitor_cfg, False),
            disable=not bool(monitor_cfg.get("enabled", True)),
        )
        for chunk_idx, start in enumerate(iterator, start=1):
            end = min(start + chunk_size, len(rows))
            chunk_rows = rows[start:end]
            pairs = [(str(row.get("target_text", "")), str(row.get("generated_text", ""))) for row in chunk_rows]
            values_by_metric: Dict[str, List[float]] = {}
            if "normalized_levenshtein" in metrics:
                metric_started = time.perf_counter()
                values_by_metric["normalized_levenshtein"] = [
                    float(value)
                    for value in _map_with_optional_workers(
                        lambda pair: normalized_levenshtein(pair[0], pair[1]),
                        pairs,
                        num_workers,
                    )
                ]
                timings["normalized_levenshtein_s"] = timings.get("normalized_levenshtein_s", 0.0) + (
                    time.perf_counter() - metric_started
                )
            if "bleu" in metrics:
                metric_started = time.perf_counter()
                values_by_metric["bleu"] = [
                    float(max(0.0, min(1.0, 1.0 - value)))
                    for value in _map_with_optional_workers(
                        lambda pair: bleu_avg_1_to4(pair[0], pair[1]),
                        pairs,
                        num_workers,
                    )
                ]
                timings["bleu_s"] = timings.get("bleu_s", 0.0) + (time.perf_counter() - metric_started)
            if rouge_scorer is not None:
                metric_started = time.perf_counter()
                rouge_values = []
                for target_text, candidate_text in pairs:
                    rouge_l = rouge_scorer.score(target_text, candidate_text)["rougeL"].fmeasure
                    rouge_values.append(float(max(0.0, min(1.0, 1.0 - rouge_l))))
                values_by_metric["rouge_l"] = rouge_values
                timings["rouge_l_s"] = timings.get("rouge_l_s", 0.0) + (time.perf_counter() - metric_started)
            for row_idx, row in enumerate(chunk_rows):
                for metric_name, metric_values in values_by_metric.items():
                    row[metric_name] = metric_values[row_idx]
            if chunk_idx % int(monitor_cfg.get("log_interval_batches", 10)) == 0 or end == len(rows):
                iterator.set_postfix(processed=end, total=len(rows), refresh=False)
        timings["scalar_total_s"] = time.perf_counter() - scalar_start
    if "bertscore_f1" in metrics:
        distance_cfg = _dra_distance_config(config)
        bertscore_start = time.perf_counter()
        half_precision = bool(distance_cfg.get("bertscore_half_precision", False))
        deduplicate_pairs = bool(distance_cfg.get("bertscore_deduplicate_pairs", True))
        candidates = [str(row.get("generated_text", "")) for row in rows]
        references = [str(row.get("target_text", "")) for row in rows]
        batch_size = _dra_metric_batch_size(config, "bertscore")
        f1_values: List[Optional[float]] = [None] * len(rows)
        bert_indices: List[int] = []
        bert_candidates: List[str] = []
        bert_references: List[str] = []
        bert_pair_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, (candidate, reference) in enumerate(zip(candidates, references)):
            candidate_empty = not candidate.strip()
            reference_empty = not reference.strip()
            if candidate_empty or reference_empty:
                f1_values[idx] = 1.0 if candidate_empty and reference_empty else 0.0
                continue
            pair = (candidate, reference)
            if deduplicate_pairs:
                bert_pair_to_indices[pair].append(idx)
            else:
                bert_indices.append(idx)
                bert_candidates.append(candidate)
                bert_references.append(reference)
        if deduplicate_pairs and bert_pair_to_indices:
            for pair, indices in bert_pair_to_indices.items():
                bert_candidates.append(pair[0])
                bert_references.append(pair[1])
                bert_indices.append(indices[0])
        if bert_indices:
            _monitor_print(
                monitor_cfg,
                (
                    f"[DRA text distances/bertscore] scoring_pairs={len(bert_indices)} "
                    f"batch_size={batch_size} deduplicated={deduplicate_pairs}"
                ),
            )
            scorer = _get_bertscore_scorer(config, device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if half_precision and str(device).startswith("cuda")
                else nullcontext()
            )
            if scorer is not None:
                with autocast_context:
                    try:
                        _, _, f1_scores = scorer.score(
                            bert_candidates,
                            bert_references,
                            batch_size=batch_size,
                            verbose=False,
                        )
                    except TypeError:
                        _, _, f1_scores = scorer.score(bert_candidates, bert_references)
            else:
                bert_score_module = _require_optional_dependency("bert_score", "bert-score", "bertscore_f1")
                kwargs: Dict[str, Any] = {
                    "batch_size": batch_size,
                    "device": str(device),
                    "verbose": False,
                    "rescale_with_baseline": bool(distance_cfg.get("bertscore_rescale_with_baseline", False)),
                }
                if distance_cfg.get("bertscore_model"):
                    kwargs["model_type"] = str(distance_cfg["bertscore_model"])
                else:
                    kwargs["lang"] = str(distance_cfg.get("bertscore_lang", "en"))
                with autocast_context:
                    _, _, f1_scores = bert_score_module.score(bert_candidates, bert_references, **kwargs)
            _monitor_print(
                monitor_cfg,
                f"[DRA text distances/bertscore] completed scoring_pairs={len(bert_indices)}",
            )
            if deduplicate_pairs and bert_pair_to_indices:
                pair_scores = {
                    pair: float(value)
                    for pair, value in zip(bert_pair_to_indices.keys(), f1_scores.detach().cpu().tolist())
                }
                for pair, indices in bert_pair_to_indices.items():
                    score_value = pair_scores[pair]
                    for idx in indices:
                        f1_values[idx] = score_value
            else:
                for idx, value in zip(bert_indices, f1_scores.detach().cpu().tolist()):
                    f1_values[idx] = float(value)
        for row, value in zip(rows, f1_values):
            if value is None:
                value = 0.0
            row["bertscore_f1"] = float(max(0.0, min(1.0, 1.0 - float(value))))
        timings["bertscore_s"] = time.perf_counter() - bertscore_start
    timings["total_s"] = max(0.0, time.perf_counter() - started_at)
    _monitor_print(
        monitor_cfg,
        (
            "[DRA text distances] timing_s="
            + ",".join(f"{key}:{timings[key]:.2f}" for key in sorted(timings))
        ),
    )
    return timings


def _add_image_dra_distances(
    rows: Sequence[Dict[str, Any]],
    candidate_tensors: Sequence[torch.Tensor],
    target_tensors: Sequence[torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
) -> None:
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    monitor_cfg = _monitor_config(config)
    metrics = _dra_distance_metrics(config, "image")
    if not metrics or not rows:
        return
    compute_rows: List[Dict[str, Any]] = []
    candidates: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    target_cache_keys: List[Hashable] = []
    for row_idx, (row, cand, tgt) in enumerate(zip(rows, candidate_tensors, target_tensors)):
        if row.get("target_available", True) and cand is not None and tgt is not None:
            compute_rows.append(row)
            candidates.append(cand)
            targets.append(tgt)
            if row.get("target_sample_id") is not None:
                target_cache_keys.append(("target_sample_id", row.get("target_sample_id")))
            else:
                target_cache_keys.append(("row", row_idx))
    if not compute_rows:
        return
    chunk_size = _image_distance_chunk_size(config, metrics)
    _monitor_print(
        monitor_cfg,
        (
            f"[DRA image distances] metrics={','.join(metrics)} rows={len(compute_rows)} "
            f"chunk_size={chunk_size}"
        ),
    )
    image_cache: Dict[str, Dict[Hashable, Any]] = {}
    total_chunks = max(1, (len(compute_rows) + chunk_size - 1) // chunk_size)
    iterator = tqdm(
        range(0, len(compute_rows), chunk_size),
        total=total_chunks,
        desc=f"DRA image distances [{','.join(metrics)}]",
        unit="batch",
        leave=_monitor_leave(monitor_cfg, False),
        disable=not bool(monitor_cfg.get("enabled", True)),
    )
    for chunk_idx, start in enumerate(iterator, start=1):
        end = min(start + chunk_size, len(compute_rows))
        chunk_rows = compute_rows[start:end]
        chunk_candidates = candidates[start:end]
        chunk_targets = targets[start:end]
        chunk_target_keys = target_cache_keys[start:end]
        values = _compute_image_distance_values(
            chunk_candidates,
            chunk_targets,
            config,
            device,
            metrics,
            target_cache_keys=chunk_target_keys,
            image_cache=image_cache,
        )
        for metric, metric_values in values.items():
            for row, value in zip(chunk_rows, metric_values):
                row[metric] = value
        if chunk_idx % int(monitor_cfg.get("log_interval_batches", 10)) == 0 or chunk_idx == total_chunks:
            iterator.set_postfix(processed=end, total=len(compute_rows), refresh=False)
    _monitor_print(
        monitor_cfg,
        f"[DRA image distances] completed rows={len(compute_rows)} metrics={len(metrics)}",
    )


def _distance_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: row[key] for key in DRA_DISTANCE_KEYS if key in row and row[key] is not None}


def _target_tensor_for_row(
    row_idx: int,
    candidate_to_target: Sequence[int],
    target_image_tensors: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if target_image_tensors is None:
        return None
    if row_idx < 0:
        return None
    if len(target_image_tensors) == len(candidate_to_target):
        if row_idx >= len(target_image_tensors):
            return None
        return target_image_tensors[row_idx].to(torch.float32)
    if row_idx >= len(candidate_to_target):
        return None
    target_idx = int(candidate_to_target[row_idx])
    if target_idx < 0 or target_idx >= len(target_image_tensors):
        return None
    return target_image_tensors[target_idx].to(torch.float32)


def _rows_to_mia_cache(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    modality: str,
    aux: Optional[Dict[str, Any]] = None,
    image_tensors: Optional[torch.Tensor] = None,
    cpk_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scores = [float(r.get("mia_score", 0.0)) for r in rows]
    labels = [int(r.get("membership_label", 0)) for r in rows]
    sample_ids = [r.get("sample_id", idx) for idx, r in enumerate(rows)]
    direction = str(rows[0].get("mia_score_direction", "higher_is_member")) if rows else "higher_is_member"
    prompts = [str(r.get("prompt", "")) for r in rows]
    data = {
        "sample_ids": sample_ids,
        "labels": _tensor_or_empty(labels, dtype=torch.long),
        "prompts": prompts,
        "records": rows,
        "mia": {
            "attack": "minkplusplus" if modality == "text" else "pia",
            "scores": _tensor_or_empty(scores, dtype=torch.float32),
            "score_direction": direction,
            "aux": {
                "records_have_token_or_pia_artifacts": True,
            },
        },
    }
    if modality == "diffusion":
        data["generations"] = {"image_tensors": image_tensors}
        data["images"] = image_tensors
    if cpk_inputs:
        data["cpk_inputs"] = cpk_inputs
    return make_cache(
        stage="mia",
        config=config,
        data=data,
        aux=aux,
        model_type=infer_model_type(config),
        modality=modality,
    )


def _rows_to_dra_generation_cache(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    modality: str,
    candidate_image_tensors: Optional[torch.Tensor] = None,
    target_image_tensors: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    target_ids = sorted({int(r.get("target_sample_id", 0)) for r in rows})
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_ids)}
    candidate_to_target = [target_to_idx[int(r.get("target_sample_id", 0))] for r in rows]
    data = {
        "target_ids": target_ids,
        "prompts": [str(r.get("prompt", "")) for r in rows],
        "target_paths": [r.get("target_image_path") for r in rows],
        "candidate_ids": [r.get("candidate_id", idx) for idx, r in enumerate(rows)],
        "candidate_to_target": _tensor_or_empty(candidate_to_target, dtype=torch.long),
        "records": rows,
        "generations": {
            "paths": [r.get("image_path") for r in rows],
            "seeds": _tensor_or_empty([int(r.get("seed", 0)) for r in rows], dtype=torch.long),
            "states": None,
            "timesteps": None,
            "tokenized_sequences": None,
            "image_tensors": candidate_image_tensors,
        },
        "target_image_tensors": target_image_tensors,
    }
    return make_cache(
        stage="dra_generation",
        config=config,
        data=data,
        model_type=infer_model_type(config),
        modality=modality,
    )


def _rows_to_dra_scored_cache(
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    modality: str,
    input_cache_path: Optional[str] = None,
    candidate_image_tensors: Optional[torch.Tensor] = None,
    target_image_tensors: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    target_ids = sorted({int(r.get("target_sample_id", 0)) for r in rows})
    target_to_idx = {target_id: idx for idx, target_id in enumerate(target_ids)}
    candidate_to_target = [target_to_idx[int(r.get("target_sample_id", 0))] for r in rows]
    distances: Dict[str, torch.Tensor] = {}
    for key in DRA_DISTANCE_KEYS:
        if any(key in r and r.get(key) is not None for r in rows):
            distances[key] = _tensor_or_empty([float(r.get(key, float("nan"))) for r in rows], dtype=torch.float32)
    data = {
        "target_ids": target_ids,
        "candidate_ids": [r.get("candidate_id", idx) for idx, r in enumerate(rows)],
        "candidate_to_target": _tensor_or_empty(candidate_to_target, dtype=torch.long),
        "records": rows,
        "mia": {
            "attack": "minkplusplus" if modality == "text" else "pia",
            "scores": _tensor_or_empty([float(r.get("mia_score", 0.0)) for r in rows], dtype=torch.float32),
            "score_direction": str(rows[0].get("mia_score_direction", "higher_is_member")) if rows else "higher_is_member",
            "aux": {},
        },
        "distances": distances,
        "cpk_inputs": {
            "states": None,
            "timesteps": None,
            "logp": None,
            "tokenized_sequences": None,
        },
        "generations": {
            "image_tensors": candidate_image_tensors,
        },
        "target_image_tensors": target_image_tensors,
    }
    return make_cache(
        stage="dra_scored",
        config=config,
        data=data,
        model_type=infer_model_type(config),
        modality=modality,
        input_cache_path=input_cache_path,
    )


def _write_cache_if_requested(
    payload: Dict[str, Any],
    config: Dict[str, Any],
    stage_name: str,
    output_path: Optional[str],
    overwrite: bool,
) -> Optional[Path]:
    if output_path is None:
        cache_cfg = config.get("cache", {})
        experiment_cfg = config.get("experiment", {})
        project_cfg = config.get("project", {})
        has_configured_output = bool(
            cache_cfg.get("output_path")
            or cache_cfg.get("output_dir")
            or cache_cfg.get("stage_outputs")
            or experiment_cfg.get("output_dir")
            or project_cfg.get("output_dir")
        )
        if not has_configured_output:
            return None
    path = resolve_output_path(config, stage_name, output_path)
    return save_cache(path, payload, overwrite=overwrite or bool(config.get("cache", {}).get("overwrite", False)))


def _resolve_model_checkpoint_dir(config: Dict[str, Any], role: str) -> Path:
    models_cfg = config.get("models", {})
    if role not in models_cfg:
        raise KeyError(f"models.{role} is required in the config")
    role_cfg = models_cfg.get(role, {})
    if not isinstance(role_cfg, dict):
        raise TypeError(f"models.{role} must be a mapping")
    checkpoint_dir = role_cfg.get("checkpoint_dir")
    if checkpoint_dir:
        return resolve_run_dir(checkpoint_dir)
    root_dir = models_cfg.get("root_dir")
    checkpoint_name = role_cfg.get("checkpoint_name") or role_cfg.get("name")
    if root_dir and checkpoint_name:
        return resolve_run_dir(str(Path(root_dir) / str(checkpoint_name)))
    raise KeyError(
        f"models.{role}.checkpoint_dir is missing. "
        f"Provide checkpoint_dir or set models.root_dir + models.{role}.checkpoint_name."
    )


def _cpk_inline_enabled(config: Dict[str, Any], stage: str) -> bool:
    cpk_cfg = config.get("cpk", {})
    per_stage = cpk_cfg.get("compute_inline_stages")
    if isinstance(per_stage, list):
        return stage in {str(s) for s in per_stage}
    return bool(cpk_cfg.get("compute_inline", False))


def run_infer_cpk_text(config: Dict[str, Any]) -> Path:
    output_root = Path(config["project"]["output_root"])
    dataset_key = config["dataset"]["name"]
    split_name = config["dataset"].get("prompt_split", "train_full")
    prompt_limit = int(config["generation"].get("prompt_limit", 8))
    seed = int(config["project"].get("seed", 1337))
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    run_name = config["project"].get("run_name", "cpk_text")
    run_dir = build_run_output_dir(output_root, dataset_key, run_name)
    write_common_artifacts(
        run_dir,
        config,
        PROCESSED_DATA_ROOT / dataset_key / "split_fingerprint.json",
    )

    draft_dir = _resolve_model_checkpoint_dir(config, "draft_p")
    q1_dir = _resolve_model_checkpoint_dir(config, "shard_q1")
    q2_dir = _resolve_model_checkpoint_dir(config, "shard_q2")
    draft = load_text_model(draft_dir, device)
    q1 = load_text_model(q1_dir, device)
    q2 = load_text_model(q2_dir, device)
    validate_cpk_text_triplet(draft, q1, q2)

    samples = collect_text_prompts(dataset_key, split_name, prompt_limit)
    prompts = [s["prompt"] for s in samples]
    num_candidates_per_prompt = int(config["generation"].get("num_candidates_per_prompt", 1))
    expanded_prompts: List[str] = []
    expanded_samples: List[Dict[str, Any]] = []
    for sample in samples:
        for _ in range(num_candidates_per_prompt):
            expanded_prompts.append(sample["prompt"])
            expanded_samples.append(sample)
    candidates = generate_text_candidates(draft, expanded_prompts, config["generation"], seed=seed)

    rows: List[Dict[str, Any]] = []
    cpk_batch_size = int(config.get("cpk", {}).get("batch_size", config.get("generation", {}).get("batch_size", 8)))
    if cpk_batch_size <= 0:
        cpk_batch_size = 1
    cpk_max_length = int(config.get("cpk", {}).get("max_sequence_tokens", draft.training_config.get("max_seq_len", 2048)))
    for start in range(0, len(candidates), cpk_batch_size):
        end = start + cpk_batch_size
        chunk = candidates[start:end]
        prompts = [str(c["prompt"]) for c in chunk]
        continuations = [str(c["generated_text"]) for c in chunk]
        prepared_batch = prepare_continuation_first_batch(draft.tokenizer, prompts, continuations, cpk_max_length)
        p_batch = realized_token_logprobs_batch(
            draft,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=cpk_max_length,
            prepared_batch=prepared_batch,
        )
        q1_batch = realized_token_logprobs_batch(
            q1,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=cpk_max_length,
            prepared_batch=prepared_batch,
        )
        q2_batch = realized_token_logprobs_batch(
            q2,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=cpk_max_length,
            prepared_batch=prepared_batch,
        )

        for local_idx, (candidate, score_p, score_q1, score_q2) in enumerate(zip(chunk, p_batch, q1_batch, q2_batch)):
            idx = start + local_idx
            sample = expanded_samples[idx]
            prompt = candidate["prompt"]
            continuation = candidate["generated_text"]
            ratio_q1 = score_p["seq_logprob"] - score_q1["seq_logprob"]
            ratio_q2 = score_p["seq_logprob"] - score_q2["seq_logprob"]
            mean_ratio_q1 = score_p["mean_logprob"] - score_q1["mean_logprob"]
            mean_ratio_q2 = score_p["mean_logprob"] - score_q2["mean_logprob"]
            row = {
                "dataset": dataset_key,
                "modality": "text",
                "run_type": "cpk_generation",
                "model_role": "draft_p",
                "model_checkpoint_path": str(draft_dir),
                "split_name": split_name,
                "sample_id": sample["sample_id"],
                "prompt_id": candidate["prompt_id"],
                "prompt": prompt,
                "seed": seed,
                "created_at": now_iso(),
                "candidate_id": f"{sample['sample_id']}_{idx % max(1, num_candidates_per_prompt)}",
                "source_model_role": "draft_p",
                "prompt_token_count": score_p["prompt_token_count"],
                "generated_token_count": score_p["scored_token_count"],
                "tokens": score_p["token_ids"],
                "generated_text": continuation,
                "decode_strategy": candidate["decode_strategy"],
                "temperature": candidate["temperature"],
                "top_p": candidate["top_p"],
                "top_k": candidate["top_k"],
                "max_new_tokens": candidate["max_new_tokens"],
                "eos_reached": candidate["eos_reached"],
                "logprob_p_per_token": score_p["token_logprobs"],
                "logprob_q1_per_token": score_q1["token_logprobs"],
                "logprob_q2_per_token": score_q2["token_logprobs"],
                "seq_logprob_p": score_p["seq_logprob"],
                "seq_logprob_q1": score_q1["seq_logprob"],
                "seq_logprob_q2": score_q2["seq_logprob"],
                "mean_logprob_p": score_p["mean_logprob"],
                "mean_logprob_q1": score_q1["mean_logprob"],
                "mean_logprob_q2": score_q2["mean_logprob"],
                "cpk_logratio_p_over_q1": ratio_q1,
                "cpk_logratio_p_over_q2": ratio_q2,
                "cpk_logratio_max": max(ratio_q1, ratio_q2),
                "cpk_logratio_p_over_q1_mean": mean_ratio_q1,
                "cpk_logratio_p_over_q2_mean": mean_ratio_q2,
                "cpk_logratio_max_mean": max(mean_ratio_q1, mean_ratio_q2),
            }
            rows.append(row)

    dump_jsonl(run_dir / "records.jsonl", rows)
    write_json(run_dir / "summary_metrics.json", {"num_records": len(rows)})
    return run_dir


def run_score_mia_text(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from datasets import load_from_disk  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    output_root = Path(config["project"]["output_root"])
    dataset_key = config["dataset"]["name"]
    split_member = config["dataset"].get("member_split", "train_full")
    split_nonmember = config["dataset"].get("nonmember_split", "mi_reserve_not_in_training")
    limit = int(config["mia"].get("limit", 32))
    batch_size = int(config["mia"].get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("mia.batch_size must be >= 1")
    text_scoring_mode = str(config["mia"].get("text_scoring_mode", "prompt_target"))
    equalize_class_counts = bool(config["mia"].get("equalize_class_counts", False))
    max_sequence_tokens = int(config["mia"].get("max_sequence_tokens", 2048))
    if max_sequence_tokens < 2:
        raise ValueError("mia.max_sequence_tokens must be >= 2")
    chunk_batch_size = int(config["mia"].get("chunk_batch_size", 1))
    if chunk_batch_size <= 0:
        raise ValueError("mia.chunk_batch_size must be >= 1")
    seed = int(config["project"].get("seed", 1337))
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    text_dtype = str(config.get("runtime", {}).get("text_dtype", "fp16")).lower()
    torch_dtype = None
    if str(device).startswith("cuda"):
        if text_dtype in {"fp16", "float16"}:
            torch_dtype = torch.float16
        elif text_dtype in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif text_dtype in {"fp32", "float32"}:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported runtime.text_dtype: {text_dtype}")
    run_name = config["project"].get("run_name", "mia_text")
    run_dir = build_run_output_dir(output_root, dataset_key, run_name)
    write_common_artifacts(
        run_dir,
        config,
        PROCESSED_DATA_ROOT / dataset_key / "split_fingerprint.json",
    )

    k_grid = config["minkpp"]["k_percent_grid"]
    default_k = int(config["minkpp"]["default_k_percent"])
    epsilon = float(config["minkpp"]["epsilon"])

    target_dir = _resolve_model_checkpoint_dir(config, "target")
    target = load_text_model(target_dir, device, torch_dtype=torch_dtype)

    draft = q1 = q2 = None
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False))
    cpk_inline = cpk_enabled and _cpk_inline_enabled(config, stage="mia")
    if cpk_enabled:
        draft = load_text_model(_resolve_model_checkpoint_dir(config, "draft_p"), device, torch_dtype=torch_dtype)
        if cpk_inline:
            q1 = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q1"), device, torch_dtype=torch_dtype)
            q2 = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q2"), device, torch_dtype=torch_dtype)
            validate_cpk_text_triplet(draft, q1, q2)
    score_model = draft if (cpk_enabled and draft is not None) else target

    if text_scoring_mode == "prompt_target":
        members = collect_text_prompts(dataset_key, split_member, limit)
        nonmembers = collect_text_prompts(dataset_key, split_nonmember, limit)
    elif text_scoring_mode == "full_text_chunked":
        def _collect_text_rows(split_name: str) -> List[Dict[str, Any]]:
            ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset_key / split_name))
            n = min(limit, len(ds))
            return [{"sample_id": i, "text": str(ds[i].get("text", ""))} for i in range(n)]

        members = _collect_text_rows(split_member)
        nonmembers = _collect_text_rows(split_nonmember)
    else:
        raise ValueError(f"Unsupported mia.text_scoring_mode: {text_scoring_mode}")

    if equalize_class_counts:
        keep = min(len(members), len(nonmembers))
        members = members[:keep]
        nonmembers = nonmembers[:keep]

    rows: List[Dict[str, Any]] = []
    total_examples = len(members) + len(nonmembers)
    pbar = tqdm(
        total=total_examples,
        desc=f"MIA text scoring [{dataset_key}:{text_scoring_mode}]",
        unit="sample",
    )

    if text_scoring_mode == "prompt_target":
        for label, group in [(1, members), (0, nonmembers)]:
            split_name = split_member if label == 1 else split_nonmember
            for start in range(0, len(group), batch_size):
                chunk = group[start : start + batch_size]
                prompts = [item["prompt"] for item in chunk]
                target_texts = [item["target_text"] for item in chunk]
                score_main_batch = realized_token_logprobs_batch(
                    score_model,
                    prompts,
                    target_texts,
                    score_prompt_tokens=False,
                    max_length=max_sequence_tokens,
                    return_token_logprobs=False,
                    return_token_logits=False,
                    compute_minkpp_cfg={
                        "k_percent_grid": k_grid,
                        "default_k_percent": default_k,
                        "epsilon": epsilon,
                    },
                )
                if cpk_inline and draft is not None and q1 is not None and q2 is not None:
                    prepared_batch = prepare_continuation_first_batch(
                        draft.tokenizer,
                        prompts,
                        target_texts,
                        max_sequence_tokens,
                    )
                    pscore_batch = realized_token_logprobs_batch(
                        draft,
                        prompts,
                        target_texts,
                        score_prompt_tokens=False,
                        max_length=max_sequence_tokens,
                        return_token_logprobs=False,
                        return_token_logits=False,
                        prepared_batch=prepared_batch,
                        seq_logprob_only=True,
                    )
                    q1score_batch = realized_token_logprobs_batch(
                        q1,
                        prompts,
                        target_texts,
                        score_prompt_tokens=False,
                        max_length=max_sequence_tokens,
                        return_token_logprobs=False,
                        return_token_logits=False,
                        prepared_batch=prepared_batch,
                        seq_logprob_only=True,
                    )
                    q2score_batch = realized_token_logprobs_batch(
                        q2,
                        prompts,
                        target_texts,
                        score_prompt_tokens=False,
                        max_length=max_sequence_tokens,
                        return_token_logprobs=False,
                        return_token_logits=False,
                        prepared_batch=prepared_batch,
                        seq_logprob_only=True,
                    )
                else:
                    pscore_batch = q1score_batch = q2score_batch = []

                for idx, item in enumerate(chunk):
                    prompt = item["prompt"]
                    target_text = item["target_text"]
                    score_main = score_main_batch[idx]
                    row = {
                        "dataset": dataset_key,
                        "modality": "text",
                        "run_type": "mia",
                        "sample_id": item["sample_id"],
                        "prompt": prompt,
                        "target_text": target_text,
                        "membership_label": label,
                        "prompt_token_count": score_main["prompt_token_count"],
                        "target_token_count": score_main["scored_token_count"],
                        "max_sequence_tokens": max_sequence_tokens,
                        "tokens": score_main["token_ids"],
                        "minkpp_scores_by_k": score_main["minkpp_scores_by_k"],
                        "minkpp_default_score": score_main["minkpp_default_score"],
                        "minkpp_token_scores": score_main["minkpp_token_scores"],
                        "mia_score": score_main["minkpp_default_score"],
                        "mia_score_direction": "higher_is_member",
                        "text_scoring_mode": text_scoring_mode,
                        "created_at": now_iso(),
                    }
                    if cpk_inline and draft is not None and q1 is not None and q2 is not None:
                        pscore = pscore_batch[idx]
                        q1score = q1score_batch[idx]
                        q2score = q2score_batch[idx]
                        ratio_q1 = pscore["seq_logprob"] - q1score["seq_logprob"]
                        ratio_q2 = pscore["seq_logprob"] - q2score["seq_logprob"]
                        mean_ratio_q1 = pscore["mean_logprob"] - q1score["mean_logprob"]
                        mean_ratio_q2 = pscore["mean_logprob"] - q2score["mean_logprob"]
                        row.update(
                            {
                                "seq_logprob_p": pscore["seq_logprob"],
                                "seq_logprob_q1": q1score["seq_logprob"],
                                "seq_logprob_q2": q2score["seq_logprob"],
                                "cpk_logratio_p_over_q1": ratio_q1,
                                "cpk_logratio_p_over_q2": ratio_q2,
                                "cpk_logratio_max": max(ratio_q1, ratio_q2),
                                "cpk_logratio_max_mean": max(mean_ratio_q1, mean_ratio_q2),
                            }
                        )
                    rows.append(row)
                    pbar.update(1)
                    pbar.set_postfix(scored=len(rows), split=split_name)
    else:
        chunk_token_size = int(config["mia"].get("chunk_token_size", score_model.training_config.get("max_seq_len", 2048)))
        if chunk_token_size < 2:
            chunk_token_size = 2
        chunk_token_size = min(chunk_token_size, max_sequence_tokens)
        for label, group in [(1, members), (0, nonmembers)]:
            split_name = split_member if label == 1 else split_nonmember
            for start in range(0, len(group), batch_size):
                sample_batch = group[start : start + batch_size]
                for item in sample_batch:
                    text = item.get("text", "")
                    token_ids = score_model.tokenizer.encode(
                        text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_sequence_tokens,
                    )
                    token_chunks = [
                        token_ids[i : i + chunk_token_size]
                        for i in range(0, len(token_ids), chunk_token_size)
                        if len(token_ids[i : i + chunk_token_size]) >= 2
                    ]
                    all_token_scores: List[float] = []
                    total_scored_tokens = 0
                    for chunk_start in range(0, len(token_chunks), chunk_batch_size):
                        chunk_batch = token_chunks[chunk_start : chunk_start + chunk_batch_size]
                        score_main_chunks = minkpp_token_scores_from_token_ids_batch(
                            score_model,
                            chunk_batch,
                            epsilon=epsilon,
                        )
                        for chunk_score in score_main_chunks:
                            total_scored_tokens += int(chunk_score["scored_token_count"])
                            all_token_scores.extend(chunk_score["minkpp_token_scores"])
                    n_scores = len(all_token_scores)
                    if total_scored_tokens > 0:
                        sorted_scores = sorted(all_token_scores)
                        mink_by_k = {}
                        for k in k_grid:
                            count = max(1, int(n_scores * (float(k) / 100.0)))
                            mink_by_k[str(k)] = float(sum(sorted_scores[:count]) / count)
                    else:
                        mink_by_k = {str(k): 0.0 for k in k_grid}
                    row = {
                        "dataset": dataset_key,
                        "modality": "text",
                        "run_type": "mia",
                        "sample_id": item["sample_id"],
                        "prompt": "",
                        "target_text": text,
                        "membership_label": label,
                        "prompt_token_count": 0,
                        "target_token_count": total_scored_tokens,
                        "input_token_count": len(token_ids),
                        "max_sequence_tokens": max_sequence_tokens,
                        "tokens": [],
                        "minkpp_scores_by_k": mink_by_k,
                        "minkpp_default_score": float(mink_by_k[str(default_k)]),
                        "minkpp_token_scores": [],
                        "minkpp_chunk_count": len(token_chunks),
                        "minkpp_chunk_batch_size": chunk_batch_size,
                        "mia_score": float(mink_by_k[str(default_k)]),
                        "mia_score_direction": "higher_is_member",
                        "text_scoring_mode": text_scoring_mode,
                        "created_at": now_iso(),
                    }
                    if cpk_inline and draft is not None and q1 is not None and q2 is not None:
                        p_chunks = realized_token_logprobs_from_token_ids_batch(draft, token_chunks)
                        q1_chunks = realized_token_logprobs_from_token_ids_batch(q1, token_chunks)
                        q2_chunks = realized_token_logprobs_from_token_ids_batch(q2, token_chunks)
                        seq_p = float(sum(ch["seq_logprob"] for ch in p_chunks))
                        seq_q1 = float(sum(ch["seq_logprob"] for ch in q1_chunks))
                        seq_q2 = float(sum(ch["seq_logprob"] for ch in q2_chunks))
                        mean_p = seq_p / max(1, total_scored_tokens)
                        mean_q1 = seq_q1 / max(1, total_scored_tokens)
                        mean_q2 = seq_q2 / max(1, total_scored_tokens)
                        row.update(
                            {
                                "seq_logprob_p": seq_p,
                                "seq_logprob_q1": seq_q1,
                                "seq_logprob_q2": seq_q2,
                                "cpk_logratio_p_over_q1": seq_p - seq_q1,
                                "cpk_logratio_p_over_q2": seq_p - seq_q2,
                                "cpk_logratio_max": max(seq_p - seq_q1, seq_p - seq_q2),
                                "cpk_logratio_max_mean": max(mean_p - mean_q1, mean_p - mean_q2),
                            }
                        )
                    rows.append(row)
                    pbar.update(1)
                    pbar.set_postfix(
                        scored=len(rows),
                        split=split_name,
                        chunk_tokens=chunk_token_size,
                        chunk_batch=chunk_batch_size,
                    )
    pbar.close()

    dump_jsonl(run_dir / "records.jsonl", rows)
    write_json(
        run_dir / "summary_metrics.json",
        {
            "num_records": len(rows),
            "cpk_enabled": cpk_enabled,
            "cpk_inline_scoring": cpk_inline,
            "text_scoring_mode": text_scoring_mode,
            "text_dtype": text_dtype,
            "max_sequence_tokens": max_sequence_tokens,
            "chunk_batch_size": chunk_batch_size,
            "equalize_class_counts": equalize_class_counts,
            "num_members_scored": len(members),
            "num_nonmembers_scored": len(nonmembers),
        },
    )
    cache_payload = _rows_to_mia_cache(
        rows,
        config,
        modality="text",
        aux={"legacy_run_dir": str(run_dir), "records_jsonl": str(run_dir / "records.jsonl")},
    )
    cache_path = _write_cache_if_requested(cache_payload, config, "mia_cache", output_path, overwrite)
    if cache_path is not None:
        write_json(run_dir / "cache_manifest.json", {"cache_path": str(cache_path)})
    return run_dir


def run_score_mia_diffusion(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from datasets import load_from_disk  # pylint: disable=import-outside-toplevel
    from torchvision import transforms  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    output_root = Path(config["project"]["output_root"])
    dataset_key = config["dataset"]["name"]
    member_split = config["dataset"].get("member_split", "train_full")
    nonmember_split = config["dataset"].get("nonmember_split", "mi_reserve_not_in_training")
    limit = int(config["mia"].get("limit", 16))
    batch_size = int(config["mia"].get("batch_size", 8))
    num_workers = int(config["mia"].get("num_workers", 4))
    run_name = config["project"].get("run_name", "mia_diffusion")
    run_dir = build_run_output_dir(output_root, dataset_key, run_name)
    write_common_artifacts(run_dir, config, PROCESSED_DATA_ROOT / dataset_key / "split_fingerprint.json")

    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    strict_adapter_load = bool(config.get("runtime", {}).get("strict_adapter_load", True))
    use_amp = bool(config.get("runtime", {}).get("amp", True))
    diffusion_dtype = str(config.get("runtime", {}).get("diffusion_dtype", "fp16")).lower()
    torch_dtype = None
    if str(device).startswith("cuda"):
        if diffusion_dtype in {"fp16", "float16"}:
            torch_dtype = torch.float16
        elif diffusion_dtype in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif diffusion_dtype in {"fp32", "float32"}:
            torch_dtype = torch.float32
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False))
    cpk_inline = cpk_enabled and _cpk_inline_enabled(config, stage="mia")
    if cpk_enabled:
        draft = load_diffusion_model(
            _resolve_model_checkpoint_dir(config, "draft_p"),
            device,
            strict_adapter_load=strict_adapter_load,
            torch_dtype=torch_dtype,
        )
        if cpk_inline:
            q1 = load_diffusion_model(
                _resolve_model_checkpoint_dir(config, "shard_q1"),
                device,
                strict_adapter_load=strict_adapter_load,
                torch_dtype=torch_dtype,
            )
            q2 = load_diffusion_model(
                _resolve_model_checkpoint_dir(config, "shard_q2"),
                device,
                strict_adapter_load=strict_adapter_load,
                torch_dtype=torch_dtype,
            )
            validate_cpk_diffusion_triplet(draft, q1, q2)
        else:
            q1 = q2 = None
        target = draft
    else:
        target = load_diffusion_model(
            _resolve_model_checkpoint_dir(config, "target"),
            device,
            strict_adapter_load=strict_adapter_load,
            torch_dtype=torch_dtype,
        )
        draft = q1 = q2 = None

    resolution = int(target.training_config.get("resolution", 64))
    tf = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    timestep = int(config["pia"].get("default_timestep", 200))
    norm_p = int(config["pia"].get("norm_p", 4))
    guidance_scale = float(config["pia"].get("guidance_scale", 1.0))
    beta_schedule = str(config["pia"].get("beta_schedule", "pipeline"))
    cpk_cfg = config.get("cpk", {})
    trajectory_steps = int(cpk_cfg.get("trajectory_steps", 10))
    store_trajectory_debug = bool(cpk_cfg.get("store_trajectory_debug", False))
    if beta_schedule not in {"pipeline", "reference_linear"}:
        raise ValueError(
            "Unsupported pia.beta_schedule. Use 'pipeline' to match the loaded "
            "model scheduler's training noising distribution, or "
            "'reference_linear' for the legacy hard-coded PIA schedule."
        )
    rows: List[Dict[str, Any]] = []
    cached_image_tensors: List[torch.Tensor] = []
    cached_xt_tensors: List[torch.Tensor] = []
    prompt_embed_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def collate(examples):
        imgs = []
        prompts = []
        ids = []
        for i, ex in examples:
            image = ex.get("image")
            prompt = ex.get("text", "")
            if not hasattr(image, "convert"):
                continue
            rgb = image.convert("RGB")
            imgs.append(tf(rgb))
            prompts.append(prompt)
            ids.append(i)
        if not imgs:
            return None
        return torch.stack(imgs, dim=0), prompts, ids

    for label, split in [(1, member_split), (0, nonmember_split)]:
        ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset_key / split))
        n = min(limit, len(ds))
        indexed = [(i, ds[i]) for i in range(n)]
        loader = torch.utils.data.DataLoader(
            indexed,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            collate_fn=collate,
        )
        total_batches = max(1, (n + batch_size - 1) // batch_size)
        scored = 0
        pbar = tqdm(loader, total=total_batches, desc=f"MIA diffusion scoring [{dataset_key}:{split}]", unit="batch")
        for batch in pbar:
            if batch is None:
                continue
            image_batch, prompts, ids = batch
            if not ids:
                continue
            x0_latents = diffusion_prepare_latent(target, image_batch)
            pia_scores, xt_batch = pia_scores_from_latents_batch(
                target,
                x0_latents,
                prompts,
                timestep=timestep,
                norm_p=norm_p,
                amp_enabled=use_amp,
                guidance_scale=guidance_scale,
                beta_schedule=beta_schedule,
                return_xt=True,
            )
            # Reconstruct [0,1] tensors once for cache artifacts and downstream posthoc scoring.
            image_batch_raw = ((image_batch + 1.0) * 0.5).clamp(0.0, 1.0)
            traj_batch: List[Optional[Dict[str, Any]]] = [None] * len(ids)
            if cpk_inline and draft is not None and q1 is not None and q2 is not None and ids:
                xt_for_cpk = xt_batch.to(device=draft.pipe.device)
                traj_batch = diffusion_logprob_trajectory_from_xt_batch(
                    draft,
                    q1,
                    q2,
                    xt_for_cpk,
                    prompts,
                    start_timestep=timestep,
                    trajectory_steps=trajectory_steps,
                    guidance_scale=guidance_scale,
                    prompt_embed_cache=prompt_embed_cache,
                    store_per_step=store_trajectory_debug,
                )
            for pos, (sample_id, prompt, pia_score) in enumerate(zip(ids, prompts, pia_scores)):
                cached_image_tensors.append(image_batch_raw[pos].detach().cpu().to(torch.float32))
                cached_xt_tensors.append(xt_batch[pos].detach().cpu().to(torch.float32))
                row = {
                    "dataset": dataset_key,
                    "modality": "image",
                    "membership_label": label,
                    "sample_id": int(sample_id),
                    "prompt": prompt,
                    "pia_model_role": "draft_p" if cpk_enabled else "target",
                    "pia_timestep": timestep,
                    "pia_norm_p": norm_p,
                    "pia_guidance_scale": guidance_scale,
                    "pia_beta_schedule": beta_schedule,
                    "pia_score": float(pia_score),
                    "mia_score": float(pia_score),
                    "mia_score_direction": "lower_is_member",
                    "created_at": now_iso(),
                }
                if cpk_inline and traj_batch[pos] is not None:
                    row.update(traj_batch[pos])
                rows.append(row)
                scored += 1
            pbar.set_postfix(scored=scored, seen=n)
    dump_jsonl(run_dir / "records.jsonl", rows)
    write_json(
        run_dir / "summary_metrics.json",
        {"num_records": len(rows), "cpk_enabled": cpk_enabled, "cpk_inline_scoring": cpk_inline},
    )
    cache_payload = _rows_to_mia_cache(
        rows,
        config,
        modality="diffusion",
        aux={"legacy_run_dir": str(run_dir), "records_jsonl": str(run_dir / "records.jsonl")},
        image_tensors=torch.stack(cached_image_tensors) if cached_image_tensors else None,
        cpk_inputs={
            "x_t": torch.stack(cached_xt_tensors) if cached_xt_tensors else None,
            "t_start": int(timestep),
            "trajectory_steps": int(trajectory_steps),
            "guidance_scale": float(guidance_scale),
            "beta_schedule": str(beta_schedule),
        },
    )
    cache_path = _write_cache_if_requested(cache_payload, config, "mia_cache", output_path, overwrite)
    if cache_path is not None:
        write_json(run_dir / "cache_manifest.json", {"cache_path": str(cache_path)})
    return run_dir


def run_analyze_mia(config: Dict[str, Any]) -> Path:
    source_records = Path(config["inputs"]["records_path"])
    rows = read_jsonl(source_records)
    output_root = Path(config["project"]["output_root"])
    run_dir = build_run_output_dir(output_root, config["project"]["dataset_name"], config["project"]["run_name"])
    write_common_artifacts(run_dir, config, None)

    fpr_grid = [float(x) for x in config["mia_analysis"].get("fpr_grid", [0.01, 0.05, 0.1])]
    fpr_points = int(config["mia_analysis"].get("fpr_points", 100))
    fpr_thresholds = np.linspace(0.0, 1.0, max(2, fpr_points))
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False))
    alpha_grid = [float(a) for a in config.get("cpk", {}).get("alpha_grid", [1.0])]

    by_alpha: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    raw_cpk = [float(r.get("cpk_logratio_max", 0.0)) for r in rows]
    finite_raw_cpk = [v for v in raw_cpk if math.isfinite(v)]
    thresholds = (
        {
            alpha: (safe_quantile(finite_raw_cpk, alpha) if finite_raw_cpk else float("nan"))
            for alpha in alpha_grid
        }
        if cpk_enabled
        else {1.0: float("inf")}
    )
    for alpha, thr in thresholds.items():
        if cpk_enabled:
            subset = []
            for row in rows:
                cpk_value = _finite_float_or_none(row.get("cpk_logratio_max", -1e18))
                if cpk_value is not None and cpk_value <= thr:
                    subset.append(row)
        else:
            subset = rows
        labels = [int(r["membership_label"]) for r in subset]
        direction = str(subset[0]["mia_score_direction"]) if subset else "higher_is_member"
        member_scores = [float(r["mia_score"]) for r in subset if int(r["membership_label"]) == 1]
        nonmember_scores = [float(r["mia_score"]) for r in subset if int(r["membership_label"]) == 0]
        metrics = compute_mia_metrics_reference_style(
            torch.tensor(member_scores, dtype=torch.float32).unsqueeze(0),
            torch.tensor(nonmember_scores, dtype=torch.float32).unsqueeze(0),
            fpr_thresholds=fpr_thresholds,
            fpr_eval=fpr_grid,
            score_direction=direction,
        )
        record = {
            "alpha": alpha,
            "threshold": thr,
            "num_samples": len(subset),
            "num_members": int(sum(labels)),
            "num_nonmembers": int(len(labels) - sum(labels)),
            "auroc": float(metrics["auc"]),
        }
        for fpr in fpr_grid:
            record[f"tpr_at_fpr_{fpr:.2f}"] = float(metrics[f"tpr@fpr={fpr:.2f}"])
        by_alpha.append(record)
        for fpr, tpr in zip(fpr_thresholds.tolist(), metrics["tpr_fpr_curve"]):
            roc_rows.append({"alpha": alpha, "fpr": float(fpr), "tpr": float(tpr)})

    ensure = run_dir / "metrics"
    ensure.mkdir(parents=True, exist_ok=True)
    write_json(ensure / "mia_summary.json", {"rows": by_alpha})
    dump_csv(ensure / "mia_by_alpha.csv", by_alpha)
    dump_csv(ensure / "roc_points_by_alpha.csv", roc_rows)
    write_json(ensure / "bootstrap_confidence_intervals.json", {"note": "bootstrap omitted in lightweight implementation"})
    dump_jsonl(run_dir / "records.jsonl", rows)
    write_json(run_dir / "summary_metrics.json", {"num_records": len(rows)})
    return run_dir


def run_score_cpk_cache(
    config: Dict[str, Any],
    input_path: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    cache = load_cache(Path(input_path))
    rows = [dict(r) for r in cache["data"].get("records", [])]
    modality = cache.get("modality", infer_modality(config))
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    model_type = "cpk"
    cached_image_tensors = cache["data"].get("generations", {}).get("image_tensors")
    if cached_image_tensors is None:
        cached_image_tensors = cache["data"].get("candidate_image_tensors")
    if cached_image_tensors is None:
        cached_image_tensors = cache["data"].get("images")

    if modality == "text":
        draft = load_text_model(_resolve_model_checkpoint_dir(config, "draft_p"), device)
        q1 = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q1"), device)
        q2 = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q2"), device)
        validate_cpk_text_triplet(draft, q1, q2)
        cpk_batch_size = int(config.get("cpk", {}).get("batch_size", config.get("mia", {}).get("batch_size", 8)))
        if cpk_batch_size <= 0:
            cpk_batch_size = 1
        cpk_max_length = int(
            config.get("cpk", {}).get("max_sequence_tokens", config.get("mia", {}).get("max_sequence_tokens", 2048))
        )
        total_chunks = max(1, (len(rows) + cpk_batch_size - 1) // cpk_batch_size)
        pbar = tqdm(
            range(0, len(rows), cpk_batch_size),
            total=total_chunks,
            desc="CP-k post-scoring [text]",
            unit="batch",
        )
        for start in pbar:
            chunk = rows[start : start + cpk_batch_size]
            prompts = [str(r.get("prompt", "")) for r in chunk]
            continuations = [str(r.get("target_text", r.get("generated_text", ""))) for r in chunk]
            prepared_batch = prepare_continuation_first_batch(
                draft.tokenizer,
                prompts,
                continuations,
                cpk_max_length,
            )
            pscore_batch = realized_token_logprobs_batch(
                draft,
                prompts,
                continuations,
                score_prompt_tokens=False,
                max_length=cpk_max_length,
                return_token_logprobs=False,
                return_token_logits=False,
                prepared_batch=prepared_batch,
                seq_logprob_only=True,
            )
            q1score_batch = realized_token_logprobs_batch(
                q1,
                prompts,
                continuations,
                score_prompt_tokens=False,
                max_length=cpk_max_length,
                return_token_logprobs=False,
                return_token_logits=False,
                prepared_batch=prepared_batch,
                seq_logprob_only=True,
            )
            q2score_batch = realized_token_logprobs_batch(
                q2,
                prompts,
                continuations,
                score_prompt_tokens=False,
                max_length=cpk_max_length,
                return_token_logprobs=False,
                return_token_logits=False,
                prepared_batch=prepared_batch,
                seq_logprob_only=True,
            )
            for row, pscore, q1score, q2score in zip(chunk, pscore_batch, q1score_batch, q2score_batch):
                logp = float(pscore["seq_logprob"])
                logq1 = float(q1score["seq_logprob"])
                logq2 = float(q2score["seq_logprob"])
                min_logq = min(logq1, logq2)
                row.update(
                    {
                        "seq_logprob_p": logp,
                        "seq_logprob_q1": logq1,
                        "seq_logprob_q2": logq2,
                        "min_logq": min_logq,
                        "cpk_stat": logp - min_logq,
                        "cpk_logratio_p_over_q1": logp - logq1,
                        "cpk_logratio_p_over_q2": logp - logq2,
                        "cpk_logratio_max": logp - min_logq,
                    }
                )
            pbar.set_postfix(scored=min(start + len(chunk), len(rows)), total=len(rows))
    else:
        draft = load_diffusion_model(_resolve_model_checkpoint_dir(config, "draft_p"), device)
        q1 = load_diffusion_model(_resolve_model_checkpoint_dir(config, "shard_q1"), device)
        q2 = load_diffusion_model(_resolve_model_checkpoint_dir(config, "shard_q2"), device)
        validate_cpk_diffusion_triplet(draft, q1, q2)
        cpk_batch_size = int(config.get("cpk", {}).get("batch_size", config.get("mia", {}).get("batch_size", 8)))
        if cpk_batch_size <= 0:
            cpk_batch_size = 1
        cpk_cfg = dict(config.get("cpk", {}))
        trajectory_steps = int(cpk_cfg.get("trajectory_steps", 10))
        store_trajectory_debug = bool(cpk_cfg.get("store_trajectory_debug", False))
        cpk_inputs = dict(cache.get("data", {}).get("cpk_inputs", {}) or {})
        cached_xt_tensors = cpk_inputs.get("x_t")
        start_timestep = int(cpk_inputs.get("t_start", config.get("pia", {}).get("default_timestep", 200)))
        guidance_scale = float(cpk_inputs.get("guidance_scale", config.get("pia", {}).get("guidance_scale", 1.0)))
        is_dra_stage = str(cache.get("stage", "")).startswith("dra")
        prompt_embed_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        total_chunks = max(1, (len(rows) + cpk_batch_size - 1) // cpk_batch_size)
        pbar = tqdm(
            range(0, len(rows), cpk_batch_size),
            total=total_chunks,
            desc="CP-k post-scoring [diffusion]",
            unit="batch",
        )
        for start in pbar:
            end = min(start + cpk_batch_size, len(rows))
            chunk_rows = rows[start:end]
            pending_rows: List[Dict[str, Any]] = []
            pending_prompts: List[str] = []
            pending_xt: List[torch.Tensor] = []

            for idx, row in enumerate(chunk_rows, start=start):
                has_traj = (
                    "trajectory_logscore_p" in row and "trajectory_logscore_q1" in row and "trajectory_logscore_q2" in row
                )
                if has_traj:
                    continue
                if is_dra_stage:
                    raise ValueError(
                        "DRA diffusion CP-k post-scoring is deprecated. "
                        "Expected trajectory logprob fields from generation-time CP-k path."
                    )
                if cached_xt_tensors is not None and idx < len(cached_xt_tensors):
                    xt = cached_xt_tensors[idx].to(torch.float32)
                else:
                    if cached_image_tensors is None:
                        raise ValueError(
                            "Diffusion CP-k scoring requires cached x_t tensors or cached image tensors."
                        )
                    if idx >= len(cached_image_tensors):
                        raise ValueError(
                            f"Diffusion CP-k scoring requires cached tensors; row index {idx} exceeds "
                            f"cached image tensor length {len(cached_image_tensors)}."
                        )
                    image_tensor = cached_image_tensors[idx].to(torch.float32)
                    if image_tensor.ndim != 3:
                        raise ValueError(f"Expected cached image tensor [C,H,W], got shape {tuple(image_tensor.shape)}")
                    resolution = int(draft.training_config.get("resolution", image_tensor.shape[-1]))
                    if image_tensor.shape[-2:] != (resolution, resolution):
                        image_tensor = torch.nn.functional.interpolate(
                            image_tensor.unsqueeze(0),
                            size=(resolution, resolution),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                    x0_latent = diffusion_prepare_latent(draft, (image_tensor.unsqueeze(0) * 2.0) - 1.0)
                    _, xt_rebuilt = pia_scores_from_latents_batch(
                        draft,
                        x0_latent,
                        [str(row.get("prompt", ""))],
                        timestep=start_timestep,
                        norm_p=int(config.get("pia", {}).get("norm_p", 4)),
                        guidance_scale=guidance_scale,
                        beta_schedule=str(config.get("pia", {}).get("beta_schedule", "pipeline")),
                        return_xt=True,
                    )
                    xt = xt_rebuilt[0].detach().to(torch.float32)
                pending_rows.append(row)
                pending_prompts.append(str(row.get("prompt", "")))
                pending_xt.append(xt)

            if pending_rows:
                traj_batch = diffusion_logprob_trajectory_from_xt_batch(
                    draft,
                    q1,
                    q2,
                    torch.stack(pending_xt, dim=0),
                    pending_prompts,
                    start_timestep=start_timestep,
                    trajectory_steps=trajectory_steps,
                    guidance_scale=guidance_scale,
                    prompt_embed_cache=prompt_embed_cache,
                    store_per_step=store_trajectory_debug,
                )
                for row, traj in zip(pending_rows, traj_batch):
                    row.update(traj)

            for row in chunk_rows:
                logp = float(row["trajectory_logscore_p"])
                logq1 = float(row["trajectory_logscore_q1"])
                logq2 = float(row["trajectory_logscore_q2"])
                min_logq = min(logq1, logq2)
                row.update({"min_logq": min_logq, "cpk_stat": logp - min_logq, "cpk_logratio_max": logp - min_logq})

            pbar.set_postfix(scored=end, total=len(rows))

    cpk_payload = {
        "logp": _tensor_or_empty([float(r.get("seq_logprob_p", r.get("trajectory_logscore_p", 0.0))) for r in rows]),
        "logq1": _tensor_or_empty([float(r.get("seq_logprob_q1", r.get("trajectory_logscore_q1", 0.0))) for r in rows]),
        "logq2": _tensor_or_empty([float(r.get("seq_logprob_q2", r.get("trajectory_logscore_q2", 0.0))) for r in rows]),
        "min_logq": _tensor_or_empty([float(r.get("min_logq", 0.0)) for r in rows]),
        "cpk_stat": _tensor_or_empty([float(r.get("cpk_stat", r.get("cpk_logratio_max", 0.0))) for r in rows]),
    }
    data = dict(cache["data"])
    data["records"] = rows
    data["cpk"] = cpk_payload
    output_stage = "mia_cpk_cache" if cache["stage"] == "mia" else "dra_cpk_cache"
    payload = make_cache(
        stage="cpk",
        config=config,
        data=data,
        aux={"source_stage": cache["stage"], "source_config_hash": cache["config_hash"]},
        model_type=model_type,
        modality=modality,
        input_cache_path=str(input_path),
    )
    path = resolve_output_path(config, output_stage, output_path)
    return save_cache(path, payload, overwrite=overwrite or bool(config.get("cache", {}).get("overwrite", False)))


def run_analyze_mia_cache(config: Dict[str, Any], input_path: str) -> Path:
    cache = load_cache(Path(input_path))
    rows = [dict(r) for r in cache["data"].get("records", [])]
    output_root = Path(config.get("project", {}).get("output_root", "outputs/inference"))
    dataset_name = config.get("project", {}).get("dataset_name") or config.get("dataset", {}).get("name", "unknown_dataset")
    run_name = config.get("project", {}).get("run_name", "analyze_mia")
    run_dir = build_run_output_dir(output_root, dataset_name, run_name)
    write_common_artifacts(run_dir, config, None)

    fpr_grid = [float(x) for x in config.get("mia_analysis", {}).get("fpr_grid", [0.01, 0.05, 0.1])]
    fpr_points = int(config.get("mia_analysis", {}).get("fpr_points", 100))
    fpr_thresholds = np.linspace(0.0, 1.0, max(2, fpr_points))
    cpk_enabled = ("cpk" in cache["data"]) or any(
        ("cpk_stat" in r) or ("cpk_logratio_max" in r) for r in rows
    )
    acceptance_rates = config.get("cpk", {}).get("acceptance_rates", config.get("cpk", {}).get("alpha_grid", [1.0]))
    acceptance_rates = [float(a) for a in acceptance_rates]
    cpk_values = [float(r.get("cpk_stat", r.get("cpk_logratio_max", 0.0))) for r in rows]
    finite_cpk_values = [v for v in cpk_values if math.isfinite(v)]
    thresholds = {
        ar: (
            float("inf")
            if (not cpk_enabled or ar >= 1.0)
            else (safe_quantile(finite_cpk_values, ar) if finite_cpk_values else float("nan"))
        )
        for ar in acceptance_rates
    }

    by_alpha: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    for ar, threshold in thresholds.items():
        if not cpk_enabled or ar >= 1.0:
            subset = rows
        else:
            subset = []
            for row in rows:
                cpk_value = _finite_float_or_none(row.get("cpk_stat", row.get("cpk_logratio_max", 0.0)))
                if cpk_value is not None and cpk_value <= threshold:
                    subset.append(row)
        direction = str(cache["data"].get("mia", {}).get("score_direction") or (subset[0].get("mia_score_direction", "higher_is_member") if subset else "higher_is_member"))
        member_scores = [float(r["mia_score"]) for r in subset if int(r.get("membership_label", 0)) == 1]
        nonmember_scores = [float(r["mia_score"]) for r in subset if int(r.get("membership_label", 0)) == 0]
        metrics = compute_mia_metrics_reference_style(
            torch.tensor(member_scores, dtype=torch.float32).unsqueeze(0),
            torch.tensor(nonmember_scores, dtype=torch.float32).unsqueeze(0),
            fpr_thresholds=fpr_thresholds,
            fpr_eval=fpr_grid,
            score_direction=direction,
        )
        record = {
            "acceptance_rate": ar,
            "threshold": threshold,
            "num_samples": len(subset),
            "num_members": len(member_scores),
            "num_nonmembers": len(nonmember_scores),
            "acceptance_fraction": len(subset) / max(1, len(rows)),
            "auroc": float(metrics["auc"]),
            "input_cache_path": str(input_path),
        }
        for fpr in fpr_grid:
            record[f"tpr_at_fpr_{fpr:.2f}"] = float(metrics[f"tpr@fpr={fpr:.2f}"])
        by_alpha.append(record)
        for fpr, tpr in zip(fpr_thresholds.tolist(), metrics["tpr_fpr_curve"]):
            roc_rows.append({"acceptance_rate": ar, "fpr": float(fpr), "tpr": float(tpr)})

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "mia_metrics.json", {"rows": by_alpha, "input_cache_path": str(input_path)})
    dump_csv(metrics_dir / "mia_metrics.csv", by_alpha)
    dump_csv(metrics_dir / "roc_points_by_alpha.csv", roc_rows)
    write_json(run_dir / "summary_metrics.json", {"num_records": len(rows), "input_cache_path": str(input_path)})
    return run_dir


def run_analyze_dra_cache(config: Dict[str, Any], input_path: str) -> Path:
    cache = load_cache(Path(input_path))
    rows = [dict(r) for r in cache["data"].get("records", [])]
    output_root = Path(config.get("project", {}).get("output_root", "outputs/inference"))
    dataset_name = config.get("project", {}).get("dataset_name") or config.get("dataset", {}).get("name", "unknown_dataset")
    run_name = config.get("project", {}).get("run_name", "analyze_dra")
    run_dir = build_run_output_dir(output_root, dataset_name, run_name)
    write_common_artifacts(run_dir, config, None)

    cpk_enabled = ("cpk" in cache["data"]) or any(
        ("cpk_stat" in r) or ("cpk_logratio_max" in r) for r in rows
    )
    acceptance_rates = config.get("cpk", {}).get("acceptance_rates", config.get("cpk", {}).get("alpha_grid", [1.0]))
    acceptance_rates = [float(a) for a in acceptance_rates]
    cpk_values = [float(r.get("cpk_stat", r.get("cpk_logratio_max", 0.0))) for r in rows]
    finite_cpk_values = [v for v in cpk_values if math.isfinite(v)]
    direction = str(cache["data"].get("mia", {}).get("score_direction", "higher_is_member"))
    reverse = direction == "higher_is_member"
    distance_keys = [key for key in DRA_DISTANCE_KEYS if any(key in r for r in rows)]

    summary_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    for ar in acceptance_rates:
        threshold = (
            float("inf")
            if (not cpk_enabled or ar >= 1.0)
            else (safe_quantile(finite_cpk_values, ar) if finite_cpk_values else float("nan"))
        )
        if not cpk_enabled or ar >= 1.0:
            accepted = rows
        else:
            accepted = []
            for row in rows:
                cpk_value = _finite_float_or_none(row.get("cpk_stat", row.get("cpk_logratio_max", 0.0)))
                if cpk_value is not None and cpk_value <= threshold:
                    accepted.append(row)
        by_target: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in accepted:
            by_target[int(row.get("target_sample_id", 0))].append(row)
        selected: List[Dict[str, Any]] = []
        for target_id, group in by_target.items():
            ranked = sorted(group, key=lambda r: float(r.get("mia_score", 0.0)), reverse=reverse)
            if not ranked:
                continue
            top = dict(ranked[0])
            top["target_sample_id"] = target_id
            top["acceptance_rate"] = ar
            top["threshold"] = threshold
            selected.append(top)
            selection_rows.append(top)

        record = {
            "acceptance_rate": ar,
            "threshold": threshold,
            "num_candidates": len(accepted),
            "num_selected": len(selected),
            "num_targets_with_candidate": len(by_target),
            "acceptance_fraction": len(accepted) / max(1, len(rows)),
            "input_cache_path": str(input_path),
        }
        for key in distance_keys:
            values = [float(r[key]) for r in selected if key in r and r[key] is not None]
            if values:
                record[f"mean_top1_mia_selected_{key}"] = float(sum(values) / len(values))
        summary_rows.append(record)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "dra_metrics.json", {"rows": summary_rows, "input_cache_path": str(input_path)})
    dump_csv(metrics_dir / "dra_metrics.csv", summary_rows)
    dump_csv(metrics_dir / "dra_selected_candidates.csv", selection_rows)
    write_json(run_dir / "summary_metrics.json", {"num_records": len(rows), "input_cache_path": str(input_path)})
    return run_dir


def run_score_dra_mia_cache(
    config: Dict[str, Any],
    input_path: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    cache = load_cache(Path(input_path))
    rows = [dict(r) for r in cache["data"].get("records", [])]
    dataset_key = str(config.get("dataset", {}).get("name", "unknown_dataset"))
    modality = cache.get("modality", infer_modality(config))
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    score_batch_size = max(1, int(config.get("mia", {}).get("batch_size", 8)))
    monitor_cfg = _monitor_config(config)
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False)) or cache.get("model_type") == "cpk"
    strict_adapter_load = bool(config.get("runtime", {}).get("strict_adapter_load", True))
    distance_timings: Dict[str, float] = {}
    cached_candidate_tensors = cache["data"].get("generations", {}).get("image_tensors")
    cached_target_tensors = cache["data"].get("target_image_tensors")
    candidate_to_target_raw = cache["data"].get("candidate_to_target")
    if isinstance(candidate_to_target_raw, torch.Tensor):
        candidate_to_target = [int(x) for x in candidate_to_target_raw.detach().cpu().tolist()]
    elif isinstance(candidate_to_target_raw, (list, tuple)):
        candidate_to_target = [int(x) for x in candidate_to_target_raw]
    else:
        candidate_to_target = list(range(len(rows)))

    if modality == "text":
        _monitor_print(
            monitor_cfg,
            f"[DRA MIA scoring/text] dataset={dataset_key} rows={len(rows)} batch_size={score_batch_size}",
        )
        model_key = "draft_p" if cpk_enabled else "target"
        text_model_kwargs = _text_runtime_model_kwargs(config, device)
        mia_model = load_text_model(_resolve_model_checkpoint_dir(config, model_key), device, **text_model_kwargs)
        k_grid = config.get("minkpp", {}).get("k_percent_grid", [5, 10, 20, 30, 40, 50])
        default_k = int(config.get("minkpp", {}).get("default_k_percent", 20))
        epsilon = float(config.get("minkpp", {}).get("epsilon", 1.0e-8))
        max_sequence_tokens = int(config.get("mia", {}).get("max_sequence_tokens", 2048))
        total_chunks = max(1, (len(rows) + score_batch_size - 1) // score_batch_size)
        score_start_time = time.perf_counter()
        pbar = tqdm(
            range(0, len(rows), score_batch_size),
            total=total_chunks,
            desc=f"DRA MIA scoring [text:{dataset_key}]",
            unit="batch",
            disable=not bool(monitor_cfg.get("enabled", True)),
            leave=_monitor_leave(monitor_cfg, True),
        )
        for start in pbar:
            chunk = rows[start : start + score_batch_size]
            prompts = [str(row.get("prompt", "")) for row in chunk]
            generated_texts = [str(row.get("generated_text", "")) for row in chunk]
            scored_batch = realized_token_logprobs_batch(
                mia_model,
                prompts,
                generated_texts,
                score_prompt_tokens=False,
                max_length=max_sequence_tokens,
                return_token_logprobs=False,
                return_token_logits=False,
                compute_minkpp_cfg={
                    "k_percent_grid": k_grid,
                    "default_k_percent": default_k,
                    "epsilon": epsilon,
                },
            )
            for row, scored in zip(chunk, scored_batch):
                row["mia_score"] = scored["minkpp_default_score"]
                row["mia_score_direction"] = "higher_is_member"
            scored = min(start + len(chunk), len(rows))
            postfix: Dict[str, Any] = {"scored": scored, "total": len(rows)}
            if bool(monitor_cfg.get("show_rate", True)) and scored > 0:
                postfix["rows_per_sec"] = round(_rows_per_second(score_start_time, scored), 2)
            pbar.set_postfix(postfix, refresh=False)
        distance_timings = _add_text_dra_distances(rows, config, device)
        _monitor_print(
            monitor_cfg,
            f"[DRA MIA scoring/text] completed scored={len(rows)}",
        )
    else:
        _monitor_print(
            monitor_cfg,
            f"[DRA MIA scoring/image] dataset={dataset_key} rows={len(rows)} batch_size={score_batch_size}",
        )
        model_key = "draft_p" if cpk_enabled else "target"
        mia_model = load_diffusion_model(
            _resolve_model_checkpoint_dir(config, model_key),
            device,
            strict_adapter_load=strict_adapter_load,
        )
        timestep = int(config.get("pia", {}).get("default_timestep", 200))
        norm_p = int(config.get("pia", {}).get("norm_p", 4))
        guidance_scale = float(config.get("pia", {}).get("guidance_scale", 1.0))
        beta_schedule = str(config.get("pia", {}).get("beta_schedule", "pipeline"))
        if beta_schedule not in {"pipeline", "reference_linear"}:
            raise ValueError(
                "Unsupported pia.beta_schedule. Use 'pipeline' to match the loaded "
                "model scheduler's training noising distribution, or "
                "'reference_linear' for the legacy hard-coded PIA schedule."
            )
        resolution = int(mia_model.training_config.get("resolution", 64))
        score_rows: List[Dict[str, Any]] = []
        score_candidates: List[torch.Tensor] = []
        score_prompts: List[str] = []
        distance_rows: List[Dict[str, Any]] = []
        distance_candidates: List[torch.Tensor] = []
        distance_targets: List[torch.Tensor] = []
        if cached_candidate_tensors is None:
            raise ValueError(
                "DRA diffusion MIA scoring requires cached candidate image tensors in the input cache. "
                "Run scripts/generate_dra.py (or the workflow) with tensor-backed caches."
            )
        for idx, row in enumerate(rows):
            if idx >= len(cached_candidate_tensors):
                raise ValueError(
                    f"Candidate tensor cache length mismatch: row index {idx} exceeds cached tensor length "
                    f"{len(cached_candidate_tensors)}."
                )
            cand = cached_candidate_tensors[idx].to(torch.float32)
            if cand.ndim != 3:
                raise ValueError(f"Expected candidate tensor [C,H,W], got {tuple(cand.shape)}")
            if cand.shape[-2:] != (resolution, resolution):
                cand = torch.nn.functional.interpolate(
                    cand.unsqueeze(0), size=(resolution, resolution), mode="bilinear", align_corners=False
                )[0]
            score_rows.append(row)
            score_candidates.append(cand)
            score_prompts.append(str(row.get("prompt", "")))
            tgt = _target_tensor_for_row(idx, candidate_to_target, cached_target_tensors)
            if tgt is not None and bool(row.get("target_available", True)):
                distance_rows.append(row)
                distance_candidates.append(cand)
                distance_targets.append(tgt)
        total_chunks = max(1, (len(score_rows) + score_batch_size - 1) // score_batch_size)
        score_start_time = time.perf_counter()
        pbar = tqdm(
            range(0, len(score_rows), score_batch_size),
            total=total_chunks,
            desc=f"DRA MIA scoring [image:{dataset_key}]",
            unit="batch",
            disable=not bool(monitor_cfg.get("enabled", True)),
            leave=_monitor_leave(monitor_cfg, True),
        )
        for start in pbar:
            end = start + score_batch_size
            chunk_rows = score_rows[start:end]
            if not chunk_rows:
                continue
            cand_batch = torch.stack(score_candidates[start:end], dim=0)
            prompt_batch = score_prompts[start:end]
            x0_latents = diffusion_prepare_latent(mia_model, (cand_batch * 2.0) - 1.0)
            scores = pia_scores_from_latents_batch(
                mia_model,
                x0_latents,
                prompt_batch,
                timestep=timestep,
                norm_p=norm_p,
                guidance_scale=guidance_scale,
                beta_schedule=beta_schedule,
            )
            for row, pia_score in zip(chunk_rows, scores):
                row["mia_score"] = float(pia_score)
                row["mia_score_direction"] = "lower_is_member"
            scored = min(start + len(chunk_rows), len(score_rows))
            postfix = {"scored": scored, "total": len(score_rows)}
            if bool(monitor_cfg.get("show_rate", True)) and scored > 0:
                postfix["rows_per_sec"] = round(_rows_per_second(score_start_time, scored), 2)
            pbar.set_postfix(postfix, refresh=False)
        _add_image_dra_distances(distance_rows, distance_candidates, distance_targets, config, device)
        _monitor_print(
            monitor_cfg,
            (
                f"[DRA MIA scoring/image] completed scored={len(score_rows)} "
                f"distance_rows={len(distance_rows)}"
            ),
        )

    payload = _rows_to_dra_scored_cache(
        rows,
        config,
        modality=modality,
        input_cache_path=str(input_path),
        candidate_image_tensors=cached_candidate_tensors,
        target_image_tensors=cached_target_tensors,
    )
    if modality == "text" and distance_timings:
        payload["data"]["timings"] = {"text_distance": distance_timings}
    path = resolve_output_path(config, "dra_scored_cache", output_path)
    saved = save_cache(path, payload, overwrite=overwrite or bool(config.get("cache", {}).get("overwrite", False)))
    timing_payload = payload["data"].get("timings") if isinstance(payload.get("data"), dict) else None
    if isinstance(timing_payload, dict):
        try:
            write_json(saved.parent / "dra_score_timing.json", timing_payload)
        except Exception:
            pass
    return saved


def run_generate_dra(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from datasets import load_from_disk  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
    from torchvision import transforms  # pylint: disable=import-outside-toplevel

    output_root = Path(config["project"]["output_root"])
    dataset_key = config["dataset"]["name"]
    modality = config["dataset"].get("modality", "text")
    split_name = config["dataset"].get("target_split", "train_full")
    num_targets = int(config["dra_generation"].get("target_limit", 8))
    num_candidates = int(config["dra_generation"].get("num_candidates_per_target", 10))
    seed = int(config["project"].get("seed", 1337))
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    run_name = config["project"].get("run_name", "dra_generation")
    run_dir = build_run_output_dir(output_root, dataset_key, run_name)
    write_common_artifacts(
        run_dir,
        config,
        PROCESSED_DATA_ROOT / dataset_key / "split_fingerprint.json",
    )

    target_dir = _resolve_model_checkpoint_dir(config, "target")
    strict_adapter_load = bool(config.get("runtime", {}).get("strict_adapter_load", True))
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False))
    cpk_inline = cpk_enabled and _cpk_inline_enabled(config, stage="dra_generation")
    monitor_cfg = _monitor_config(config)
    rows: List[Dict[str, Any]] = []
    candidate_image_tensors: List[torch.Tensor] = []
    target_image_tensor_by_id: Dict[int, torch.Tensor] = {}
    prompt_embed_cache: Dict[Tuple[int, str], torch.Tensor] = {}
    random.seed(seed)
    _monitor_print(
        monitor_cfg,
        (
            f"[DRA generation] dataset={dataset_key} modality={modality} split={split_name} "
            f"target_limit={num_targets} candidates_per_target={num_candidates}"
        ),
    )

    if modality == "text":
        text_model_kwargs = _text_runtime_model_kwargs(config, device)
        target = load_text_model(target_dir, device, **text_model_kwargs)
        draft = q1 = q2 = None
        if cpk_enabled:
            draft = load_text_model(_resolve_model_checkpoint_dir(config, "draft_p"), device, **text_model_kwargs)
            if cpk_inline:
                q1 = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q1"), device, **text_model_kwargs)
                q2 = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q2"), device, **text_model_kwargs)
                validate_cpk_text_triplet(draft, q1, q2)

        samples = collect_text_prompts(dataset_key, split_name, num_targets)
        gen_model = draft if cpk_enabled and draft is not None else target
        target_batch_size = max(1, int(config["dra_generation"].get("target_batch_size", 1)))
        score_batch_size = max(1, int(config["generation"].get("score_batch_size", config["generation"].get("batch_size", num_candidates))))
        max_score_length = config["generation"].get("max_sequence_tokens", config.get("mia", {}).get("max_sequence_tokens"))
        max_score_length_int = int(max_score_length) if max_score_length is not None else None
        tokenizer_max_length = getattr(gen_model.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and 1 < tokenizer_max_length < 1_000_000:
            prepared_max_length = tokenizer_max_length
        else:
            prepared_max_length = int(gen_model.training_config.get("max_seq_len", 2048))
        if max_score_length_int is not None:
            prepared_max_length = max_score_length_int
        total_text_batches = max(1, (len(samples) + target_batch_size - 1) // target_batch_size)
        generation_start_time = time.perf_counter()
        pbar = tqdm(
            range(0, len(samples), target_batch_size),
            total=total_text_batches,
            desc=f"DRA text generation [{dataset_key}]",
            unit="target-batch",
            disable=not bool(monitor_cfg.get("enabled", True)),
            leave=_monitor_leave(monitor_cfg, True),
        )
        generated_count = 0
        for start_idx in pbar:
            batch_samples = samples[start_idx : start_idx + target_batch_size]
            batch_prompts: List[str] = []
            batch_meta: List[Tuple[Dict[str, Any], int]] = []
            for sample in batch_samples:
                prompt = sample["prompt"]
                for cand_idx in range(num_candidates):
                    batch_prompts.append(prompt)
                    batch_meta.append((sample, cand_idx))
            if not batch_prompts:
                continue
            cands = generate_text_candidates(
                gen_model,
                batch_prompts,
                config["generation"],
                seed=random.randint(0, 10_000_000),
            )
            prompts_for_scoring = [str(cand["prompt"]) for cand in cands]
            continuations = [str(cand["generated_text"]) for cand in cands]
            score_batches: List[Dict[str, Any]] = []
            pscore_batches: List[Dict[str, Any]] = []
            q1score_batches: List[Dict[str, Any]] = []
            q2score_batches: List[Dict[str, Any]] = []
            for score_start in range(0, len(cands), score_batch_size):
                score_end = min(score_start + score_batch_size, len(cands))
                prompt_chunk = prompts_for_scoring[score_start:score_end]
                continuation_chunk = continuations[score_start:score_end]
                prepared_batch = prepare_continuation_first_batch(
                    gen_model.tokenizer,
                    prompt_chunk,
                    continuation_chunk,
                    prepared_max_length,
                )
                score_chunk = realized_token_logprobs_batch(
                    gen_model,
                    prompt_chunk,
                    continuation_chunk,
                    score_prompt_tokens=False,
                    max_length=max_score_length_int,
                    prepared_batch=prepared_batch,
                )
                score_batches.extend(score_chunk)
                if cpk_inline and draft is not None and q1 is not None and q2 is not None:
                    pscore_batches.extend(score_chunk)
                    q1score_batches.extend(
                        realized_token_logprobs_batch(
                            q1,
                            prompt_chunk,
                            continuation_chunk,
                            score_prompt_tokens=False,
                            max_length=max_score_length_int,
                            return_token_logprobs=False,
                            return_token_logits=False,
                            prepared_batch=prepared_batch,
                            seq_logprob_only=True,
                        )
                    )
                    q2score_batches.extend(
                        realized_token_logprobs_batch(
                            q2,
                            prompt_chunk,
                            continuation_chunk,
                            score_prompt_tokens=False,
                            max_length=max_score_length_int,
                            return_token_logprobs=False,
                            return_token_logits=False,
                            prepared_batch=prepared_batch,
                            seq_logprob_only=True,
                        )
                    )
            for flat_idx, (cand, score) in enumerate(zip(cands, score_batches)):
                sample, cand_idx = batch_meta[flat_idx]
                continuation = cand["generated_text"]
                row = {
                    "dataset": dataset_key,
                    "modality": "text",
                    "target_sample_id": sample["sample_id"],
                    "prompt": str(cand["prompt"]),
                    "target_text": sample["target_text"],
                    "candidate_id": f"{sample['sample_id']}_{cand_idx}",
                    "generated_text": continuation,
                    "seed": seed + int(sample["sample_id"]) * 1000 + cand_idx,
                    "created_at": now_iso(),
                }
                row["seq_logprob_target"] = score["seq_logprob"]
                row["mean_logprob_target"] = score["mean_logprob"]
                row["logprob_target_per_token"] = score["token_logprobs"]
                if cpk_inline and draft is not None and q1 is not None and q2 is not None:
                    pscore = pscore_batches[flat_idx]
                    q1score = q1score_batches[flat_idx]
                    q2score = q2score_batches[flat_idx]
                    ratio_q1 = pscore["seq_logprob"] - q1score["seq_logprob"]
                    ratio_q2 = pscore["seq_logprob"] - q2score["seq_logprob"]
                    row.update(
                        {
                            "seq_logprob_p": pscore["seq_logprob"],
                            "seq_logprob_q1": q1score["seq_logprob"],
                            "seq_logprob_q2": q2score["seq_logprob"],
                            "cpk_logratio_p_over_q1": ratio_q1,
                            "cpk_logratio_p_over_q2": ratio_q2,
                            "cpk_logratio_max": max(ratio_q1, ratio_q2),
                        }
                    )
                rows.append(row)
            generated_count += len(cands)
            targets_done = min(start_idx + len(batch_samples), len(samples))
            postfix: Dict[str, Any] = {
                "targets": f"{targets_done}/{len(samples)}",
                "candidates": generated_count,
            }
            if bool(monitor_cfg.get("show_rate", True)) and generated_count > 0:
                postfix["cand_per_sec"] = round(_rows_per_second(generation_start_time, generated_count), 2)
            pbar.set_postfix(postfix, refresh=False)
        _monitor_print(
            monitor_cfg,
            (
                f"[DRA generation/text] completed targets={len(samples)} "
                f"candidates={generated_count} rows={len(rows)}"
            ),
        )
    elif modality == "image":
        target = load_diffusion_model(target_dir, device, strict_adapter_load=strict_adapter_load)
        draft = q1 = q2 = None
        if cpk_enabled:
            draft = load_diffusion_model(
                _resolve_model_checkpoint_dir(config, "draft_p"),
                device,
                strict_adapter_load=strict_adapter_load,
            )
            q1 = load_diffusion_model(
                _resolve_model_checkpoint_dir(config, "shard_q1"),
                device,
                strict_adapter_load=strict_adapter_load,
            )
            q2 = load_diffusion_model(
                _resolve_model_checkpoint_dir(config, "shard_q2"),
                device,
                strict_adapter_load=strict_adapter_load,
            )
            validate_cpk_diffusion_triplet(draft, q1, q2)
        ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset_key / split_name))
        base_model = draft if cpk_enabled and draft is not None else target
        resolution = int(base_model.training_config.get("resolution", 64))
        tf = transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])
        guidance_scale = float(config["generation"].get("guidance_scale", 7.5))
        num_inference_steps = int(config["generation"].get("num_inference_steps", 50))
        target_batch_size = max(1, int(config["dra_generation"].get("target_batch_size", 1)))
        cpk_cfg = dict(config.get("cpk", {}))
        store_trajectory_debug = bool(cpk_cfg.get("store_trajectory_debug", False))
        save_generated_artifacts = str(config.get("cache", {}).get("save_generated_artifacts", "tensor")).lower()
        max_targets = min(num_targets, len(ds))
        image_cache_dtype = torch.float16
        if bool(monitor_cfg.get("disable_diffusers_inner_bar", False)):
            try:
                base_model.pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
        total_image_batches = max(1, (max_targets + target_batch_size - 1) // target_batch_size)
        generation_start_time = time.perf_counter()
        generated_count = 0
        pbar = tqdm(
            range(0, max_targets, target_batch_size),
            total=total_image_batches,
            desc=f"DRA image generation [{dataset_key}]",
            unit="target-batch",
            disable=not bool(monitor_cfg.get("enabled", True)),
            leave=_monitor_leave(monitor_cfg, True),
        )
        for start_idx in pbar:
            batch_sample_indices = list(range(start_idx, min(start_idx + target_batch_size, max_targets)))
            batch_target_meta: List[Dict[str, Any]] = []
            batch_prompts: List[str] = []
            generators: List[torch.Generator] = []
            for sample_idx in batch_sample_indices:
                sample = ds[sample_idx]
                prompt = str(sample.get("text", ""))
                target_image = sample.get("image")
                if hasattr(target_image, "convert"):
                    target_tensor = tf(target_image.convert("RGB")).to(dtype=image_cache_dtype)
                    target_available = True
                else:
                    target_tensor = torch.zeros((3, resolution, resolution), dtype=image_cache_dtype)
                    target_available = False
                target_image_tensor_by_id[sample_idx] = target_tensor
                if target_available and save_generated_artifacts in {"path", "both"}:
                    target_image_path = run_dir / "targets" / f"target_{sample_idx:04d}.png"
                    target_image_path.parent.mkdir(parents=True, exist_ok=True)
                    target_image.save(target_image_path)
                batch_target_meta.append(
                    {
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "target_available": target_available,
                    }
                )
                for cand_idx in range(num_candidates):
                    batch_prompts.append(prompt)
                    generators.append(
                        torch.Generator(device=str(device)).manual_seed(seed + sample_idx * 1000 + cand_idx)
                    )
            if not batch_prompts:
                continue
            latent_history: List[torch.Tensor] = []
            realized_timesteps: List[int] = []
            with torch.inference_mode():
                if cpk_enabled and draft is not None and q1 is not None and q2 is not None:
                    init_latents = base_model.pipe.prepare_latents(
                        batch_size=len(batch_prompts),
                        num_channels_latents=base_model.pipe.unet.config.in_channels,
                        height=resolution,
                        width=resolution,
                        dtype=next(base_model.pipe.unet.parameters()).dtype,
                        device=base_model.pipe.device,
                        generator=generators,
                    )
                    latent_history.append(init_latents.detach().clone())

                    def _capture_step(_pipe, _step_index, _timestep, callback_kwargs):
                        latents = callback_kwargs.get("latents")
                        if latents is not None:
                            latent_history.append(latents.detach().clone())
                        realized_timesteps.append(int(_timestep))
                        return callback_kwargs

                    out = base_model.pipe(
                        prompt=batch_prompts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generators,
                        latents=init_latents,
                        height=resolution,
                        width=resolution,
                        callback_on_step_end=_capture_step,
                        callback_on_step_end_tensor_inputs=["latents"],
                    )
                else:
                    out = base_model.pipe(
                        prompt=batch_prompts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generators,
                        height=resolution,
                        width=resolution,
                    )
            generated_images = list(out.images)
            candidate_tensors_batch = [tf(image.convert("RGB")).to(dtype=image_cache_dtype) for image in generated_images]
            traj_batch: List[Optional[Dict[str, Any]]] = [None] * len(candidate_tensors_batch)
            if cpk_enabled and draft is not None and q1 is not None and q2 is not None and candidate_tensors_batch:
                if len(latent_history) < 2 or len(realized_timesteps) != len(latent_history) - 1:
                    raise RuntimeError(
                        "Failed to capture realized generation trajectory for CP-k DRA scoring."
                    )
                traj_batch = diffusion_logprob_ratios_from_realized_path_batch(
                    draft,
                    q1,
                    q2,
                    latent_history,
                    batch_prompts,
                    timesteps=realized_timesteps,
                    guidance_scale=guidance_scale,
                    prompt_embed_cache=prompt_embed_cache,
                    store_per_step=store_trajectory_debug,
                )
            for flat_idx, generated_image in enumerate(generated_images):
                target_offset = flat_idx // num_candidates
                cand_idx = flat_idx % num_candidates
                meta = batch_target_meta[target_offset]
                sample_idx = int(meta["sample_idx"])
                gen_seed = seed + sample_idx * 1000 + cand_idx
                candidate_tensor = candidate_tensors_batch[flat_idx]
                candidate_image_tensors.append(candidate_tensor)
                if save_generated_artifacts in {"path", "both"}:
                    image_path = run_dir / "images" / f"candidate_{sample_idx:04d}_{cand_idx:02d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    generated_image.save(image_path)
                row = {
                    "dataset": dataset_key,
                    "modality": "image",
                    "target_sample_id": sample_idx,
                    "prompt": str(meta["prompt"]),
                    "candidate_id": f"{sample_idx}_{cand_idx}",
                    "target_available": bool(meta["target_available"]),
                    "seed": gen_seed,
                    "created_at": now_iso(),
                }
                if cpk_enabled and traj_batch[flat_idx] is not None:
                    row.update(traj_batch[flat_idx])
                rows.append(row)
            generated_count += len(generated_images)
            targets_done = min(start_idx + len(batch_sample_indices), max_targets)
            postfix = {
                "targets": f"{targets_done}/{max_targets}",
                "candidates": generated_count,
            }
            if bool(monitor_cfg.get("show_rate", True)) and generated_count > 0:
                postfix["cand_per_sec"] = round(_rows_per_second(generation_start_time, generated_count), 2)
            pbar.set_postfix(postfix, refresh=False)
        _monitor_print(
            monitor_cfg,
            (
                f"[DRA generation/image] completed targets={max_targets} "
                f"candidates={generated_count} rows={len(rows)}"
            ),
        )
    else:
        raise ValueError(f"Unsupported DRA modality: {modality}")
    dump_jsonl(run_dir / "records.jsonl", rows)
    write_json(
        run_dir / "summary_metrics.json",
        {
            "num_records": len(rows),
            "num_candidates_per_target": num_candidates,
            "cpk_enabled": cpk_enabled,
            "cpk_inline_scoring": cpk_inline,
        },
    )
    cache_payload = _rows_to_dra_generation_cache(
        rows,
        config,
        modality="text" if modality == "text" else "diffusion",
        candidate_image_tensors=torch.stack(candidate_image_tensors) if candidate_image_tensors else None,
        target_image_tensors=(
            torch.stack([target_image_tensor_by_id[target_id] for target_id in sorted(target_image_tensor_by_id)])
            if target_image_tensor_by_id
            else None
        ),
    )
    cache_path = _write_cache_if_requested(cache_payload, config, "dra_generation_cache", output_path, overwrite)
    if cache_path is not None:
        write_json(run_dir / "cache_manifest.json", {"cache_path": str(cache_path)})
    return run_dir


def run_analyze_dra_rank_distance(config: Dict[str, Any]) -> Path:
    inputs_cfg = dict(config.get("inputs", {}))
    cache_path_raw = inputs_cfg.get("scored_cache_path") or inputs_cfg.get("input_cache_path")
    rows: List[Dict[str, Any]]
    cached_candidate_tensors: Optional[torch.Tensor] = None
    cached_target_tensors: Optional[torch.Tensor] = None
    score_direction = "higher_is_member"
    modality = "text"
    input_descriptor = None
    if cache_path_raw:
        cache_path = Path(str(cache_path_raw))
        cache = load_cache(cache_path)
        rows = [dict(r) for r in cache["data"].get("records", [])]
        modality = cache.get("modality", infer_modality(config))
        cached_candidate_tensors = cache["data"].get("generations", {}).get("image_tensors")
        cached_target_tensors = cache["data"].get("target_image_tensors")
        candidate_to_target_raw = cache["data"].get("candidate_to_target")
        if isinstance(candidate_to_target_raw, torch.Tensor):
            candidate_to_target = [int(x) for x in candidate_to_target_raw.detach().cpu().tolist()]
        elif isinstance(candidate_to_target_raw, (list, tuple)):
            candidate_to_target = [int(x) for x in candidate_to_target_raw]
        else:
            candidate_to_target = list(range(len(rows)))
        score_direction = str(cache["data"].get("mia", {}).get("score_direction", "higher_is_member"))
        input_descriptor = str(cache_path)
    else:
        records_path = inputs_cfg.get("records_path")
        if not records_path:
            raise ValueError(
                "DRA rank-distance analysis requires either inputs.scored_cache_path "
                "(preferred) or inputs.records_path."
            )
        rows = read_jsonl(Path(str(records_path)))
        input_descriptor = str(records_path)
        if rows:
            modality = str(rows[0].get("modality", "text"))
        candidate_to_target = list(range(len(rows)))

    output_root = Path(config["project"]["output_root"])
    run_dir = build_run_output_dir(output_root, config["project"]["dataset_name"], config["project"]["run_name"])
    write_common_artifacts(run_dir, config, None)

    analysis_cfg = load_yaml(Path(config["analysis_config"]["path"]))
    rho = float(analysis_cfg["dra_analysis"]["rho"])
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))

    if not rows:
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        dump_csv(metrics_dir / "dra_ranked_candidates.csv", [])
        dump_csv(metrics_dir / "dra_distance_table.csv", [])
        write_json(metrics_dir / "dra_distance_summary.json", {"num_targets": 0, "rho": rho, "selected_candidates": 0})
        dump_jsonl(run_dir / "records.jsonl", [])
        write_json(run_dir / "summary_metrics.json", {"num_records": 0})
        return run_dir

    missing_scores = [idx for idx, row in enumerate(rows) if row.get("mia_score") is None]
    if missing_scores:
        raise ValueError(
            "DRA rank-distance analysis requires precomputed MIA scores in input rows. "
            "Run scripts/score_dra_mia.py (or the workflow) first."
        )

    for idx, row in enumerate(rows):
        row["_row_index"] = idx
        if row.get("mia_score_direction") is None:
            row["mia_score_direction"] = score_direction

    by_target: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[int(row["target_sample_id"])].append(row)

    ranked_rows: List[Dict[str, Any]] = []
    selected_distance_rows: List[Dict[str, Any]] = []
    for target_id, group in by_target.items():
        direction = str(group[0].get("mia_score_direction", score_direction))
        reverse_sort = direction == "higher_is_member"
        group = sorted(group, key=lambda r: float(r["mia_score"]), reverse=reverse_sort)
        keep = max(1, int(math.ceil(len(group) * rho)))
        selected = group[:keep]
        for rank, row in enumerate(group, start=1):
            row["rank_within_target"] = rank
            row["selected_for_distance"] = rank <= keep
            row["rho"] = rho
            ranked_rows.append(row)
        selected_distance_rows.extend(selected)

    distance_rows: List[Dict[str, Any]] = []
    if modality == "text":
        _add_text_dra_distances(selected_distance_rows, config, device)
    else:
        selected_with_images: List[Dict[str, Any]] = []
        candidate_tensors: List[torch.Tensor] = []
        target_tensors: List[torch.Tensor] = []
        for row in selected_distance_rows:
            row_idx = int(row.get("_row_index", -1))
            cand = None
            if cached_candidate_tensors is not None and 0 <= row_idx < len(cached_candidate_tensors):
                cand = cached_candidate_tensors[row_idx].to(torch.float32)
            tgt = _target_tensor_for_row(row_idx, candidate_to_target, cached_target_tensors)

            if cand is None or tgt is None:
                continue
            selected_with_images.append(row)
            candidate_tensors.append(cand)
            target_tensors.append(tgt)
        _add_image_dra_distances(selected_with_images, candidate_tensors, target_tensors, config, device)
        selected_distance_rows = selected_with_images

    for row in selected_distance_rows:
        distance_rows.append(
            {
                "target_sample_id": row.get("target_sample_id"),
                "candidate_id": row["candidate_id"],
                "mia_score": row["mia_score"],
                "mia_score_direction": row.get("mia_score_direction", score_direction),
                "rank_within_target": row["rank_within_target"],
                "cpk_logratio_max": row.get("cpk_logratio_max"),
                **_distance_payload(row),
            }
        )

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(metrics_dir / "dra_ranked_candidates.csv", ranked_rows)
    dump_csv(metrics_dir / "dra_distance_table.csv", distance_rows)
    write_json(
        metrics_dir / "dra_distance_summary.json",
        {
            "num_targets": len(by_target),
            "rho": rho,
            "selected_candidates": len(distance_rows),
            "input": input_descriptor,
        },
    )
    output_rows = [{k: v for k, v in row.items() if k != "_row_index"} for row in ranked_rows]
    dump_jsonl(run_dir / "records.jsonl", output_rows)
    write_json(run_dir / "summary_metrics.json", {"num_records": len(ranked_rows)})
    return run_dir


def run_analyze_dra_success(config: Dict[str, Any]) -> Path:
    table = []
    csv_path = Path(config["inputs"]["distance_table_path"])
    if csv_path.exists():
        import csv as _csv

        with csv_path.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            table = [dict(r) for r in reader]
    output_root = Path(config["project"]["output_root"])
    run_dir = build_run_output_dir(output_root, config["project"]["dataset_name"], config["project"]["run_name"])
    write_common_artifacts(run_dir, config, None)

    thresholds = config["dra_success"].get("thresholds", {"normalized_levenshtein": [0.2], "bleu": [0.9]})
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in table:
        grouped[int(row["target_sample_id"])].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for metric, values in thresholds.items():
        for eta in values:
            successes = 0
            for _, rows in grouped.items():
                ok = False
                for row in rows:
                    v = float(row[metric])
                    ok = ok or (v <= float(eta))
                successes += int(ok)
            total = max(1, len(grouped))
            summary_rows.append(
                {
                    "metric": metric,
                    "threshold": eta,
                    "target_level_success_rate": successes / total,
                }
            )
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(metrics_dir / "dra_success_by_threshold.csv", summary_rows)
    dump_csv(metrics_dir / "dra_success_by_alpha.csv", [])
    write_json(metrics_dir / "dra_success_summary.json", {"rows": summary_rows})
    write_json(metrics_dir / "dra_success_bootstrap_ci.json", {"note": "bootstrap omitted in lightweight implementation"})
    dump_jsonl(run_dir / "records.jsonl", table)
    write_json(run_dir / "summary_metrics.json", {"num_records": len(table)})
    return run_dir


def _utility_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("utility", {}))


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _utility_limit(config: Dict[str, Any], key: str = "limit", default: int = 1000) -> int:
    utility_cfg = _utility_config(config)
    return max(1, int(utility_cfg.get(key, default)))


def _utility_batch_size(config: Dict[str, Any], key: str, default: int) -> int:
    utility_cfg = _utility_config(config)
    base = int(utility_cfg.get("batch_size", default))
    return max(1, int(utility_cfg.get(key, base)))


def _utility_text_max_length(config: Dict[str, Any], fallback: int = 2048) -> int:
    utility_cfg = _utility_config(config)
    if utility_cfg.get("max_sequence_tokens") is not None:
        return max(2, int(utility_cfg.get("max_sequence_tokens")))
    cpk_cfg = dict(config.get("cpk", {}))
    if cpk_cfg.get("max_sequence_tokens") is not None:
        return max(2, int(cpk_cfg.get("max_sequence_tokens")))
    mia_cfg = dict(config.get("mia", {}))
    if mia_cfg.get("max_sequence_tokens") is not None:
        return max(2, int(mia_cfg.get("max_sequence_tokens")))
    return max(2, int(fallback))


def _utility_metric_enabled(config: Dict[str, Any], metric: str) -> bool:
    utility_cfg = _utility_config(config)
    metrics = utility_cfg.get("metrics")
    if metrics is None:
        return True
    if isinstance(metrics, str):
        metrics = [metrics]
    return str(metric).lower() in {str(m).lower() for m in metrics}


def _normalize_score_row_cpk(row: Dict[str, Any]) -> Optional[float]:
    for key in ("cpk_stat", "cpk_logratio_max", "k_stat"):
        value = _finite_float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _quantile_threshold(values: Sequence[float], acceptance_rate: float) -> float:
    if not values or acceptance_rate >= 1.0:
        return float("inf")
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return safe_quantile(finite, acceptance_rate)


def _one_per_target_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_targets: set[int] = set()
    for idx, row in enumerate(rows):
        target_id = int(row.get("target_sample_id", row.get("sample_id", idx)))
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)
        copied = dict(row)
        copied["_row_index"] = idx
        copied["_selector_target_id"] = target_id
        selected.append(copied)
    return selected


def _utility_acceptance_rates(config: Dict[str, Any], cpk_enabled: bool) -> List[float]:
    if not cpk_enabled:
        return [1.0]
    configured = config.get("cpk", {}).get("acceptance_rates")
    if configured is None:
        configured = [0.25, 0.5, 0.75, 1.0]
    if isinstance(configured, (int, float)):
        configured = [configured]
    rates = sorted({max(0.0, min(1.0, float(v))) for v in configured})
    if 1.0 not in rates:
        rates.append(1.0)
    return rates


def _utility_prometheus_config(config: Dict[str, Any]) -> Dict[str, Any]:
    utility_cfg = _utility_config(config)
    raw = utility_cfg.get("prometheusv2", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _load_reference_text_model(
    model_name_or_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> LoadedTextModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

    runtime_kwargs = _text_runtime_model_kwargs(config, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {}
    torch_dtype = runtime_kwargs.get("torch_dtype")
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    requested_attention_impl = runtime_kwargs.get("attention_implementation")
    attempts: List[Optional[str]] = []
    if requested_attention_impl is not None and str(requested_attention_impl).lower() != "auto":
        requested = str(requested_attention_impl)
        attempts.append(requested)
        if requested == "flash_attention_3":
            attempts.append("sdpa")
        attempts.append(None)
    else:
        attempts.append(None)
    model: Optional[Any] = None
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            kwargs = dict(model_kwargs)
            if attempt is not None:
                kwargs["attn_implementation"] = attempt
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
            break
        except Exception as exc:
            last_error = exc
            continue
    if model is None:
        raise RuntimeError(
            f"Failed to load reference text model {model_name_or_path} with attempted attention implementations={attempts}."
        ) from last_error
    model = model.to(device)
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
    compile_enabled = runtime_kwargs.get("compile_model")
    if hasattr(torch, "compile") and bool(compile_enabled):
        try:
            model.forward = torch.compile(model.forward, mode=str(runtime_kwargs.get("compile_mode", "reduce-overhead")), fullgraph=False)  # type: ignore[method-assign]
        except Exception:
            pass
    return LoadedTextModel(
        name=str(model_name_or_path),
        model=model,
        tokenizer=tokenizer,
        run_dir=Path(str(model_name_or_path)),
        training_config={"model_name_or_path": str(model_name_or_path)},
    )


def run_score_utility_cache(
    config: Dict[str, Any],
    input_path: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    cache = load_cache(Path(input_path))
    rows = [dict(r) for r in cache.get("data", {}).get("records", [])]
    utility_cfg = _utility_config(config)
    monitor_cfg = _monitor_config(config)
    device = torch.device(config.get("runtime", {}).get("device", "cpu"))
    dataset_key = str(config.get("dataset", {}).get("name", "unknown_dataset"))
    modality = str(cache.get("modality", infer_modality(config, default="text")))
    split_name = str(
        utility_cfg.get(
            "ppl_split",
            config.get("dataset", {}).get(
                "target_split",
                config.get("dataset", {}).get("eval_split", "train_full"),
            ),
        )
    )
    heldout_limit = _utility_limit(config, key="heldout_limit", default=_utility_limit(config, key="limit", default=1000))
    generation_limit = _utility_limit(config, key="generation_limit", default=_utility_limit(config, key="limit", default=1000))
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False)) or cache.get("model_type") == "cpk"
    selected_generation_rows = _one_per_target_rows(rows)[:generation_limit]
    all_generation_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied["_row_index"] = idx
        copied["_selector_target_id"] = int(row.get("target_sample_id", row.get("sample_id", idx)))
        all_generation_rows.append(copied)
    _monitor_print(
        monitor_cfg,
        (
            f"[Utility scoring] modality={modality} rows={len(rows)} "
            f"selected_one_per_prompt={len(selected_generation_rows)} heldout_limit={heldout_limit}"
        ),
    )

    features: Dict[str, Any] = {
        "source_cache_path": str(input_path),
        "source_cache_fingerprint": _file_fingerprint(Path(input_path)),
        "source_stage": str(cache.get("stage", "")),
        "selector_policy": str(utility_cfg.get("one_per_prompt_policy", "first")),
        "cpk_enabled": cpk_enabled,
        "modality": modality,
        "selected_records": [{"target_sample_id": int(r.get("_selector_target_id", 0)), "row_index": int(r.get("_row_index", -1))} for r in selected_generation_rows],
        "records": [{"target_sample_id": int(r.get("_selector_target_id", 0)), "row_index": int(r.get("_row_index", -1))} for r in all_generation_rows],
        "generation_row_count": len(all_generation_rows),
        "selected_generation_row_count": len(selected_generation_rows),
    }

    if modality != "text":
        raise ValueError(
            "Text utility pipeline currently supports only text generation caches. "
            f"Received modality={modality!r}."
        )

    text_data: Dict[str, Any] = {
        "heldout_split": split_name,
        "heldout_scores": [],
        "reference_ppl_scores": [],
        "prometheus_scores": [],
    }
    heldout_batch_size = _utility_batch_size(config, "heldout_batch_size", default=8)
    reference_batch_size = _utility_batch_size(config, "reference_ppl_batch_size", default=heldout_batch_size)
    text_max_length = _utility_text_max_length(config)
    text_model_kwargs = _text_runtime_model_kwargs(config, device)
    draft_model: Optional[LoadedTextModel] = None
    shard_q1_model: Optional[LoadedTextModel] = None
    shard_q2_model: Optional[LoadedTextModel] = None
    target_model: Optional[LoadedTextModel] = None
    reference_model: Optional[LoadedTextModel] = None

    compute_heldout_ppl = _utility_metric_enabled(config, "heldout_ppl") or _utility_metric_enabled(config, "perplexity")
    compute_reference_ppl = _utility_metric_enabled(config, "reference_ppl")
    compute_prometheus = _utility_metric_enabled(config, "prometheusv2_fluency")
    if compute_heldout_ppl:
        heldout = collect_text_prompts(dataset_key, split_name, heldout_limit)
        if cpk_enabled:
            draft_model = load_text_model(_resolve_model_checkpoint_dir(config, "draft_p"), device, **text_model_kwargs)
            shard_q1_model = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q1"), device, **text_model_kwargs)
            shard_q2_model = load_text_model(_resolve_model_checkpoint_dir(config, "shard_q2"), device, **text_model_kwargs)
            validate_cpk_text_triplet(draft_model, shard_q1_model, shard_q2_model)
            heldout_model = draft_model
        else:
            target_model = load_text_model(_resolve_model_checkpoint_dir(config, "target"), device, **text_model_kwargs)
            heldout_model = target_model
        heldout_iter = tqdm(
            range(0, len(heldout), heldout_batch_size),
            desc=f"Utility heldout PPL [{dataset_key}]",
            unit="batch",
            leave=_monitor_leave(monitor_cfg, False),
            disable=not bool(monitor_cfg.get("enabled", True)),
        )
        heldout_started = time.perf_counter()
        for batch_idx, start in enumerate(heldout_iter, start=1):
            chunk = heldout[start : start + heldout_batch_size]
            prompts = [str(item["prompt"]) for item in chunk]
            target_texts = [str(item["target_text"]) for item in chunk]
            prepared_batch = prepare_continuation_first_batch(
                heldout_model.tokenizer,
                prompts,
                target_texts,
                text_max_length,
            )
            draft_scores = realized_token_logprobs_batch(
                heldout_model,
                prompts,
                target_texts,
                score_prompt_tokens=False,
                max_length=text_max_length,
                return_token_logprobs=False,
                return_token_logits=False,
                prepared_batch=prepared_batch,
            )
            if cpk_enabled and shard_q1_model is not None and shard_q2_model is not None:
                q1_scores = realized_token_logprobs_batch(
                    shard_q1_model,
                    prompts,
                    target_texts,
                    score_prompt_tokens=False,
                    max_length=text_max_length,
                    return_token_logprobs=False,
                    return_token_logits=False,
                    prepared_batch=prepared_batch,
                    seq_logprob_only=True,
                )
                q2_scores = realized_token_logprobs_batch(
                    shard_q2_model,
                    prompts,
                    target_texts,
                    score_prompt_tokens=False,
                    max_length=text_max_length,
                    return_token_logprobs=False,
                    return_token_logits=False,
                    prepared_batch=prepared_batch,
                    seq_logprob_only=True,
                )
            else:
                q1_scores = []
                q2_scores = []
            for local_idx, item in enumerate(chunk):
                score = draft_scores[local_idx]
                seq_logprob_draft = float(score["seq_logprob"])
                token_count = int(max(1, score["scored_token_count"]))
                nll = float(-seq_logprob_draft)
                record: Dict[str, Any] = {
                    "sample_id": int(item["sample_id"]),
                    "prompt": str(item["prompt"]),
                    "target_text": str(item["target_text"]),
                    "token_count": token_count,
                    "seq_logprob_draft": seq_logprob_draft,
                    "nll_draft": nll,
                    "ppl_draft": float(math.exp(nll / max(1, token_count))),
                }
                if cpk_enabled and q1_scores and q2_scores:
                    seq_logprob_q1 = float(q1_scores[local_idx]["seq_logprob"])
                    seq_logprob_q2 = float(q2_scores[local_idx]["seq_logprob"])
                    min_logq = min(seq_logprob_q1, seq_logprob_q2)
                    record.update(
                        {
                            "seq_logprob_p": seq_logprob_draft,
                            "seq_logprob_q1": seq_logprob_q1,
                            "seq_logprob_q2": seq_logprob_q2,
                            "min_logq": min_logq,
                            "k_stat": float(seq_logprob_draft - min_logq),
                        }
                    )
                text_data["heldout_scores"].append(record)
            scored = min(start + len(chunk), len(heldout))
            if batch_idx % int(monitor_cfg.get("log_interval_batches", 10)) == 0 or scored == len(heldout):
                postfix: Dict[str, Any] = {"scored": scored, "total": len(heldout)}
                if bool(monitor_cfg.get("show_rate", True)):
                    postfix["rows_per_sec"] = round(_rows_per_second(heldout_started, scored), 2)
                heldout_iter.set_postfix(postfix, refresh=False)

    reference_all_generated = bool(utility_cfg.get("reference_ppl_all_generated", cpk_enabled))
    reference_rows = all_generation_rows if reference_all_generated else selected_generation_rows
    if compute_reference_ppl and reference_rows:
        reference_model_name = str(utility_cfg.get("reference_model_name_or_path", "mistralai/Mistral-7B-v0.1"))
        reference_model = _load_reference_text_model(reference_model_name, config, device)
        ref_iter = tqdm(
            range(0, len(reference_rows), reference_batch_size),
            desc=f"Utility reference PPL [{dataset_key}]",
            unit="batch",
            leave=_monitor_leave(monitor_cfg, False),
            disable=not bool(monitor_cfg.get("enabled", True)),
        )
        ref_started = time.perf_counter()
        for batch_idx, start in enumerate(ref_iter, start=1):
            chunk = reference_rows[start : start + reference_batch_size]
            prompts = [str(row.get("prompt", "")) for row in chunk]
            generated_texts = [str(row.get("generated_text", "")) for row in chunk]
            prepared_batch = prepare_continuation_first_batch(
                reference_model.tokenizer,
                prompts,
                generated_texts,
                text_max_length,
            )
            score_batch = realized_token_logprobs_batch(
                reference_model,
                prompts,
                generated_texts,
                score_prompt_tokens=False,
                max_length=text_max_length,
                return_token_logprobs=False,
                return_token_logits=False,
                prepared_batch=prepared_batch,
            )
            for local_idx, score in enumerate(score_batch):
                row = chunk[local_idx]
                seq_logprob_ref = float(score["seq_logprob"])
                token_count_ref = int(max(1, score["scored_token_count"]))
                nll_ref = float(-seq_logprob_ref)
                text_data["reference_ppl_scores"].append(
                    {
                        "row_index": int(row.get("_row_index", -1)),
                        "target_sample_id": int(row.get("_selector_target_id", row.get("target_sample_id", 0))),
                        "candidate_id": str(row.get("candidate_id", "")),
                        "prompt": str(row.get("prompt", "")),
                        "generated_text": str(row.get("generated_text", "")),
                        "reference_model_name_or_path": reference_model_name,
                        "reference_seq_logprob": seq_logprob_ref,
                        "reference_token_count": token_count_ref,
                        "reference_nll": nll_ref,
                        "reference_ppl": float(math.exp(nll_ref / max(1, token_count_ref))),
                        "k_stat": _normalize_score_row_cpk(row),
                    }
                )
            scored = min(start + len(chunk), len(reference_rows))
            if batch_idx % int(monitor_cfg.get("log_interval_batches", 10)) == 0 or scored == len(reference_rows):
                postfix_ref: Dict[str, Any] = {"scored": scored, "total": len(reference_rows)}
                if bool(monitor_cfg.get("show_rate", True)):
                    postfix_ref["rows_per_sec"] = round(_rows_per_second(ref_started, scored), 2)
                ref_iter.set_postfix(postfix_ref, refresh=False)

    prometheus_cfg = _utility_prometheus_config(config)
    prometheus_scope = str(prometheus_cfg.get("score_scope", "one_per_prompt")).strip().lower()
    if prometheus_scope in {"all", "all_generated", "all_rows"}:
        prometheus_rows = all_generation_rows
    else:
        prometheus_rows = selected_generation_rows
    if compute_prometheus and prometheus_rows:
        prometheus_model_name = str(prometheus_cfg.get("model_name_or_path", "prometheus-eval/prometheus-7b-v2.0"))
        prometheus_batch_size = int(
            prometheus_cfg.get("batch_size", _utility_batch_size(config, "prometheus_batch_size", default=4))
        )
        prometheus_max_input_tokens = int(prometheus_cfg.get("max_input_tokens", text_max_length))
        # Prometheus often emits rationale before the final "[RESULT] <score>" token;
        # 128 tokens can truncate that tail and cause parse failures.
        prometheus_max_new_tokens = int(prometheus_cfg.get("max_new_tokens", 256))
        prometheus_rubric = str(prometheus_cfg.get("rubric", DEFAULT_FLUENCY_RUBRIC))
        text_runtime = _text_runtime_model_kwargs(config, device)
        compile_model_cfg = prometheus_cfg.get("compile_model", text_runtime.get("compile_model", False))
        compile_mode_cfg = str(prometheus_cfg.get("compile_mode", text_runtime.get("compile_mode", "reduce-overhead")))
        attention_impl_cfg = prometheus_cfg.get(
            "attention_implementation",
            text_runtime.get("attention_implementation"),
        )
        scorer = PrometheusV2Scorer(
            model_name_or_path=prometheus_model_name,
            device=device,
            batch_size=prometheus_batch_size,
            max_input_tokens=prometheus_max_input_tokens,
            max_new_tokens=prometheus_max_new_tokens,
            rubric=prometheus_rubric,
            torch_dtype=text_runtime.get("torch_dtype"),
            compile_model=bool(compile_model_cfg),
            compile_mode=compile_mode_cfg,
            attention_implementation=attention_impl_cfg,
        )
        prom_scores = scorer.score_rows(
            prometheus_rows,
            instruction_key="prompt",
            response_key="generated_text",
            show_progress=bool(monitor_cfg.get("enabled", True)),
            progress_desc=f"Utility Prometheus [{dataset_key}]",
            progress_leave=_monitor_leave(monitor_cfg, False),
        )
        for row, scored in zip(prometheus_rows, prom_scores):
            text_data["prometheus_scores"].append(
                {
                    "row_index": int(row.get("_row_index", -1)),
                    "target_sample_id": int(row.get("_selector_target_id", row.get("target_sample_id", 0))),
                    "candidate_id": str(row.get("candidate_id", "")),
                    "prompt": str(row.get("prompt", "")),
                    "generated_text": str(row.get("generated_text", "")),
                    "prometheusv2_fluency": _finite_float_or_none(scored.get("prometheusv2_fluency")),
                    "parse_ok": bool(scored.get("parse_ok", False)),
                    "parse_error": str(scored.get("parse_error", "")),
                    "k_stat": _normalize_score_row_cpk(row),
                }
            )

    text_data["reference_ppl_scoring_scope"] = "all_generated" if reference_all_generated else "one_per_prompt"
    features["text"] = text_data
    _monitor_print(
        monitor_cfg,
        (
            f"[Utility scoring] completed modality={modality} "
            f"records={len(features.get('records', []))}"
        ),
    )

    payload = make_cache(
        stage="utility_features",
        config=config,
        data=features,
        aux={"source_config_hash": cache.get("config_hash"), "source_stage": cache.get("stage")},
        model_type=infer_model_type(config),
        modality=modality,
        input_cache_path=str(input_path),
    )
    stage_name = "utility_features_cache"
    path = resolve_output_path(config, stage_name, output_path)
    return save_cache(path, payload, overwrite=overwrite or bool(config.get("cache", {}).get("overwrite", False)))


def run_analyze_utility_cache(config: Dict[str, Any], input_path: str) -> Path:
    from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

    cache = load_cache(Path(input_path))
    data = dict(cache.get("data", {}))
    monitor_cfg = _monitor_config(config)
    output_root = Path(config.get("project", {}).get("output_root", "outputs/inference"))
    dataset_name = config.get("project", {}).get("dataset_name") or config.get("dataset", {}).get("name", "unknown_dataset")
    run_name = config.get("project", {}).get("run_name", "analyze_utility")
    run_dir = build_run_output_dir(output_root, dataset_name, run_name)
    # Make utility artifacts easier to identify than raw timestamp folders.
    utility_named_dir = run_dir.with_name(f"utility_{run_dir.name}")
    if not utility_named_dir.exists():
        run_dir.rename(utility_named_dir)
        run_dir = utility_named_dir
    write_common_artifacts(run_dir, config, None)
    cpk_enabled = bool(data.get("cpk_enabled", False))
    acceptance_rates = _utility_acceptance_rates(config, cpk_enabled)
    min_samples = max(1, int(config.get("utility_analysis", {}).get("min_samples", 16)))
    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    _monitor_print(
        monitor_cfg,
        f"[Utility analysis] start acceptance_rates={acceptance_rates} min_samples={min_samples}",
    )

    text_data = dict(data.get("text", {}))
    if not text_data:
        raise ValueError("Utility analysis expects text utility features in cache.data.text.")
    heldout_scores = [dict(r) for r in text_data.get("heldout_scores", [])]
    reference_ppl_scores = [dict(r) for r in text_data.get("reference_ppl_scores", [])]
    prometheus_scores = [dict(r) for r in text_data.get("prometheus_scores", [])]

    heldout_k_values = [float(r["k_stat"]) for r in heldout_scores if _finite_float_or_none(r.get("k_stat")) is not None]
    generation_k_values = [float(r["k_stat"]) for r in reference_ppl_scores if _finite_float_or_none(r.get("k_stat")) is not None]
    if not generation_k_values:
        generation_k_values = [
            float(r["k_stat"])
            for r in prometheus_scores
            if _finite_float_or_none(r.get("k_stat")) is not None
        ]
    text_iter = tqdm(
        acceptance_rates,
        desc="Utility analysis [text]",
        unit="threshold",
        leave=_monitor_leave(monitor_cfg, False),
        disable=not bool(monitor_cfg.get("enabled", True)),
    )
    for ar in text_iter:
        heldout_threshold = _quantile_threshold(heldout_k_values, ar)
        generation_threshold = _quantile_threshold(generation_k_values, ar)
        if not heldout_k_values or ar >= 1.0:
            heldout_subset = heldout_scores
        else:
            heldout_subset = [r for r in heldout_scores if (_finite_float_or_none(r.get("k_stat")) is not None and float(r["k_stat"]) <= heldout_threshold)]
        if not generation_k_values or ar >= 1.0:
            reference_subset = reference_ppl_scores
            prometheus_subset = prometheus_scores
        else:
            reference_subset = [r for r in reference_ppl_scores if (_finite_float_or_none(r.get("k_stat")) is not None and float(r["k_stat"]) <= generation_threshold)]
            prometheus_subset = [
                r
                for r in prometheus_scores
                if (_finite_float_or_none(r.get("k_stat")) is not None and float(r["k_stat"]) <= generation_threshold)
            ]

        heldout_token_total = sum(max(1, int(r.get("token_count", 1))) for r in heldout_subset)
        heldout_nll_total = sum(float(r.get("nll_draft", 0.0)) for r in heldout_subset)
        heldout_ppl = float(math.exp(heldout_nll_total / max(1, heldout_token_total))) if heldout_subset else float("nan")
        reference_ppl_values = [float(r.get("reference_ppl")) for r in reference_subset if _finite_float_or_none(r.get("reference_ppl")) is not None]
        reference_ppl_mean = float(sum(reference_ppl_values) / len(reference_ppl_values)) if reference_ppl_values else float("nan")
        prometheus_values = [
            float(r.get("prometheusv2_fluency"))
            for r in prometheus_subset
            if _finite_float_or_none(r.get("prometheusv2_fluency")) is not None
        ]
        prometheus_mean = float(sum(prometheus_values) / len(prometheus_values)) if prometheus_values else float("nan")
        prometheus_std = float(np.std(prometheus_values)) if prometheus_values else float("nan")
        prometheus_parse_fail = sum(1 for r in prometheus_subset if not bool(r.get("parse_ok", False)))
        summary_rows.append(
            {
                "modality": "text",
                "acceptance_rate": ar,
                "heldout_k_threshold": heldout_threshold,
                "generation_k_threshold": generation_threshold,
                "num_heldout_samples": len(heldout_subset),
                "num_reference_ppl_samples": len(reference_subset),
                "num_prometheus_samples": len(prometheus_subset),
                "heldout_ppl_draft": heldout_ppl,
                "reference_ppl_mean": reference_ppl_mean,
                "prometheusv2_fluency_mean": prometheus_mean,
                "prometheusv2_fluency_std": prometheus_std,
                "num_prometheus_parse_fail": prometheus_parse_fail,
                "small_sample_warning": (
                    len(heldout_subset) < min_samples
                    or len(reference_subset) < min_samples
                    or (bool(prometheus_scores) and len(prometheus_subset) < min_samples)
                ),
                "input_cache_path": str(input_path),
            }
        )
        for row in reference_subset:
            detail_rows.append({"modality": "text", "acceptance_rate": ar, "detail_type": "reference_ppl", **row})
        for row in prometheus_subset:
            detail_rows.append({"modality": "text", "acceptance_rate": ar, "detail_type": "prometheusv2_fluency", **row})
        text_iter.set_postfix(
            heldout=len(heldout_subset),
            reference_ppl=len(reference_subset),
            prometheus=len(prometheus_subset),
            refresh=False,
        )

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        metrics_dir / "utility_metrics.json",
        {
            "rows": summary_rows,
            "input_cache_path": str(input_path),
            "small_sample_guidance": "Metrics computed on small subsets may be high-variance; interpret with caution.",
        },
    )
    dump_csv(metrics_dir / "utility_metrics.csv", summary_rows)
    dump_csv(metrics_dir / "utility_details.csv", detail_rows)
    write_json(run_dir / "summary_metrics.json", {"num_rows": len(summary_rows), "input_cache_path": str(input_path)})
    _monitor_print(
        monitor_cfg,
        f"[Utility analysis] completed rows={len(summary_rows)} detail_rows={len(detail_rows)}",
    )
    return run_dir


def run_evaluate_utility(
    config: Dict[str, Any],
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    monitor_cfg = _monitor_config(config)
    inputs_cfg = dict(config.get("inputs", {}))
    generation_cache_path = inputs_cfg.get("generation_cache_path") or inputs_cfg.get("input_cache_path")
    if not generation_cache_path:
        raise KeyError(
            "inputs.generation_cache_path (or inputs.input_cache_path) is required for utility evaluation. "
            "This workflow is cache-first and does not generate new samples."
        )
    _monitor_print(
        monitor_cfg,
        f"[Utility workflow] stage_start=score_utility input_cache={generation_cache_path}",
    )
    score_started = time.perf_counter()
    utility_features_path = run_score_utility_cache(
        config,
        input_path=str(generation_cache_path),
        output_path=output_path,
        overwrite=overwrite,
    )
    _monitor_print(
        monitor_cfg,
        (
            "[Utility workflow] stage_done=score_utility "
            f"elapsed_s={time.perf_counter() - score_started:.2f} "
            f"utility_features_cache={utility_features_path}"
        ),
    )
    _monitor_print(
        monitor_cfg,
        f"[Utility workflow] stage_start=analyze_utility input_cache={utility_features_path}",
    )
    analyze_started = time.perf_counter()
    analysis_dir = run_analyze_utility_cache(config, str(utility_features_path))
    _monitor_print(
        monitor_cfg,
        (
            "[Utility workflow] stage_done=analyze_utility "
            f"elapsed_s={time.perf_counter() - analyze_started:.2f} "
            f"analysis_dir={analysis_dir}"
        ),
    )
    write_json(
        analysis_dir / "workflow_manifest.json",
        {
            "workflow": "utility",
            "utility_features_cache": str(utility_features_path),
            "analysis_dir": str(analysis_dir),
            "source_generation_cache": str(generation_cache_path),
        },
    )
    return analysis_dir

