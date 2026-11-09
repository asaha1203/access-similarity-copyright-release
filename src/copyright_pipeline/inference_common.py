import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import PROCESSED_DATA_ROOT, RUNS_ROOT
from .io_utils import ensure_dir, read_json, write_json, write_yaml

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein  # type: ignore[import-not-found]
except Exception:
    _RapidfuzzLevenshtein = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_run_dir(path_or_name: str, default_subdir: Optional[str] = None) -> Path:
    candidate = Path(path_or_name)
    if candidate.exists():
        return candidate
    if default_subdir is None:
        fallback = RUNS_ROOT / path_or_name
    else:
        fallback = RUNS_ROOT / default_subdir / path_or_name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve run directory: {path_or_name}")


def infer_device(use_cpu: bool) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_prompt_target(text: str) -> Tuple[str, str]:
    if "[SEP]" not in text:
        raise ValueError(
            "Expected '[SEP]' delimiter between prompt and target text. "
            "Regenerate processed datasets with scripts/prepare_datasets.py --overwrite."
        )
    prompt, target = text.split("[SEP]", maxsplit=1)
    prompt = prompt.strip()
    target = target.strip()
    if not prompt or not target:
        raise ValueError("Prompt/target text around '[SEP]' must both be non-empty.")
    return prompt, target


def _join_prompt_and_continuation(prompt: str, continuation: str) -> str:
    """Join text segments with a stable lexical boundary for scoring."""
    if not prompt:
        return continuation
    if not continuation:
        return prompt
    if prompt[-1].isspace() or continuation[0].isspace():
        return prompt + continuation
    return prompt + " " + continuation


def _prepare_continuation_first_inputs(
    tokenizer,
    prompt: str,
    continuation: str,
    max_length: int,
) -> Dict[str, Any]:
    """
    Build causal-LM inputs that always prioritize continuation tokens.

    When prompt+continuation exceed max_length, we keep continuation tokens first
    and backfill remaining context with a suffix of prompt tokens.
    """
    continuation_text = continuation
    if prompt and continuation and (not prompt[-1].isspace()) and (not continuation[0].isspace()):
        continuation_text = " " + continuation

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation_text, add_special_tokens=False)
    if max_length <= 0:
        max_length = 1

    if prompt_ids:
        max_cont_tokens = max(1, max_length - 1)
    else:
        max_cont_tokens = max_length
    continuation_ids = continuation_ids[:max_cont_tokens]

    remaining = max(0, max_length - len(continuation_ids))
    prompt_keep = min(len(prompt_ids), remaining) if prompt_ids else 0
    prompt_tail = prompt_ids[-prompt_keep:] if prompt_keep else []
    input_ids = prompt_tail + continuation_ids

    prompt_token_count = len(prompt_tail)
    if prompt_token_count > 0:
        score_start = prompt_token_count - 1
        score_count = len(input_ids) - prompt_token_count
    else:
        score_start = 0
        score_count = max(0, len(input_ids) - 1)
    score_end = score_start + score_count

    return {
        "input_ids": input_ids,
        "prompt_token_count": prompt_token_count,
        "score_start": score_start,
        "score_end": score_end,
        "max_length": max_length,
    }


def prepare_continuation_first_batch(
    tokenizer,
    prompts: Sequence[str],
    continuations: Sequence[str],
    max_length: int,
) -> List[Dict[str, Any]]:
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have the same length")
    return [
        _prepare_continuation_first_inputs(tokenizer, prompt, continuation, max_length)
        for prompt, continuation in zip(prompts, continuations)
    ]


def _validate_prepared_batch(
    prepared_batch: Sequence[Dict[str, Any]],
    expected_size: int,
    max_length: int,
) -> None:
    if len(prepared_batch) != expected_size:
        raise ValueError("prepared_batch must have the same length as prompts/continuations")
    required_keys = {"input_ids", "prompt_token_count", "score_start", "score_end"}
    for idx, item in enumerate(prepared_batch):
        if not required_keys.issubset(item.keys()):
            raise ValueError(f"prepared_batch[{idx}] is missing required keys")
        seq_ids = item.get("input_ids")
        if not isinstance(seq_ids, Sequence):
            raise ValueError(f"prepared_batch[{idx}]['input_ids'] must be a sequence of token ids")
        if len(seq_ids) > max_length:
            raise ValueError(
                f"prepared_batch[{idx}] exceeds max_length={max_length}; "
                "use matching max_length/truncation semantics when reusing prepared batches."
            )
        item_max_length = item.get("max_length")
        if item_max_length is not None and int(item_max_length) != int(max_length):
            raise ValueError(
                f"prepared_batch[{idx}] was prepared with max_length={item_max_length}, "
                f"but scorer was called with max_length={max_length}"
            )


def safe_quantile(values: Sequence[float], alpha: float) -> float:
    if not values:
        return 0.0
    if alpha <= 0:
        return min(values)
    if alpha >= 1:
        return max(values)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * alpha
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return (1.0 - frac) * sorted_vals[lo] + frac * sorted_vals[hi]


def simple_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    # Assign average ranks within ties to avoid order-dependent AUROC.
    rank_sum = 0.0
    idx = 0
    n = len(pairs)
    while idx < n:
        j = idx + 1
        score = pairs[idx][0]
        while j < n and pairs[j][0] == score:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0  # 1-indexed inclusive average rank
        pos_in_group = sum(lbl for _, lbl in pairs[idx:j])
        rank_sum += avg_rank * pos_in_group
        idx = j
    u = rank_sum - (pos * (pos + 1) / 2.0)
    return float(u / (pos * neg))


def tpr_at_fpr(scores: Sequence[float], labels: Sequence[int], fpr_target: float) -> float:
    if not scores:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    # Evaluate thresholds at tie groups, not per-sample, for deterministic TPR@FPR.
    tp = 0
    fp = 0
    best = 0.0  # threshold above max score => (fpr=0, tpr=0)
    idx = 0
    n = len(pairs)
    while idx < n:
        j = idx + 1
        score = pairs[idx][0]
        while j < n and pairs[j][0] == score:
            j += 1
        grp_pos = sum(lbl for _, lbl in pairs[idx:j])
        grp_neg = (j - idx) - grp_pos
        tp += grp_pos
        fp += grp_neg
        fpr = fp / max(1, neg)
        tpr = tp / max(1, pos)
        if fpr <= fpr_target:
            best = max(best, tpr)
        idx = j
    return float(best)


def normalized_levenshtein(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a:
        return 1.0 if b else 0.0
    if not b:
        return 1.0
    if _RapidfuzzLevenshtein is not None:
        distance = _RapidfuzzLevenshtein.distance(a, b)
        return float(distance / max(len(a), len(b)))
    # Fail-safe fallback for environments missing rapidfuzz.
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return float(dp[-1] / max(len(a), len(b)))


def bleu1(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not cand_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    hit = sum(min(cand_counts[tok], ref_counts.get(tok, 0)) for tok in cand_counts)
    precision = hit / len(cand_tokens)
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(cand_tokens)))
    return float(bp * precision)


def bleu_avg_1_to4(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not cand_tokens:
        return 0.0
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # pylint: disable=import-outside-toplevel
    except Exception:
        # Keep scoring available even if nltk is missing in a custom environment.
        return bleu1(reference, candidate)
    weights_list = (
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        (0.25, 0.25, 0.25, 0.25),
    )
    smoother = SmoothingFunction().method1
    reference_corpus = [ref_tokens]
    scores = [
        float(
            sentence_bleu(
                reference_corpus,
                cand_tokens,
                weights=weights,
                smoothing_function=smoother,
            )
        )
        for weights in weights_list
    ]
    return float(max(0.0, min(1.0, sum(scores) / len(scores))))


def tensor_to_image(path: Path, tensor: torch.Tensor) -> None:
    x = tensor.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    image = Image.fromarray(arr)
    ensure_dir(path.parent)
    image.save(path)


@dataclass
class LoadedTextModel:
    name: str
    model: Any
    tokenizer: Any
    run_dir: Path
    training_config: Dict[str, Any]


@dataclass
class LoadedDiffusionModel:
    name: str
    pipe: Any
    run_dir: Path
    training_config: Dict[str, Any]


def _is_cuda_device(device: torch.device) -> bool:
    return str(device).startswith("cuda")


def _default_cuda_dtype(device: torch.device, torch_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    if torch_dtype is not None or not _is_cuda_device(device):
        return torch_dtype
    return torch.float16


def _configure_cuda_inference_backends() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]  # pylint: disable=protected-access
    except Exception:
        pass


def _enable_cuda_sdpa_backends() -> None:
    """Best-effort enablement of CUDA SDPA optimized kernels."""
    if not torch.cuda.is_available():
        return
    try:
        # Flash + mem-efficient kernels materially speed up transformer attention.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        # Older torch builds may not expose all toggles.
        pass


def _resolve_attention_implementation(
    attention_implementation: Optional[str],
    use_cuda: bool,
) -> Optional[str]:
    if not use_cuda:
        return None
    requested = "auto" if attention_implementation is None else str(attention_implementation).strip().lower()
    if requested in {"", "none", "default"}:
        return None
    if requested == "auto":
        # Aggressively prefer FA3 on CUDA; load_text_model() handles fallback.
        return "flash_attention_3"
    if requested == "flash_attention_3":
        return "flash_attention_3"
    return requested


def _compile_forward_if_available(module: Any, compile_model: bool, mode: str = "reduce-overhead") -> Any:
    if not compile_model or not hasattr(torch, "compile"):
        return module
    try:
        module.forward = torch.compile(module.forward, mode=mode, fullgraph=False)  # type: ignore[method-assign]
    except Exception:
        pass
    return module


def _channels_last_if_available(module: Any) -> Any:
    try:
        return module.to(memory_format=torch.channels_last)
    except Exception:
        return module


def mark_cuda_graph_step_if_available(device: torch.device) -> None:
    """Mark repeated CUDA graph regions when torch.compile/CUDAGraphs are enabled."""
    if not _is_cuda_device(device):
        return
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        pass


def load_text_model(
    run_dir: Path,
    device: torch.device,
    torch_dtype: Optional[torch.dtype] = None,
    compile_model: Optional[bool] = None,
    compile_mode: str = "reduce-overhead",
    merge_lora: bool = False,
    attention_implementation: Optional[str] = "auto",
) -> LoadedTextModel:
    training_cfg = load_yaml(run_dir / "training_config.yaml")
    base = training_cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(run_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    use_cuda = _is_cuda_device(device)
    torch_dtype = _default_cuda_dtype(device, torch_dtype)
    if use_cuda:
        _configure_cuda_inference_backends()
        _enable_cuda_sdpa_backends()
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    resolved_attention_impl = _resolve_attention_implementation(attention_implementation, use_cuda)
    if resolved_attention_impl is not None:
        model_kwargs["attn_implementation"] = resolved_attention_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
    except Exception:
        # Fallback for architectures/checkpoints that don't support explicit attention implementation.
        fallback_kwargs = dict(model_kwargs)
        if fallback_kwargs.get("attn_implementation") == "flash_attention_3":
            fallback_kwargs["attn_implementation"] = "sdpa"
            try:
                model = AutoModelForCausalLM.from_pretrained(base, **fallback_kwargs)
            except Exception:
                fallback_kwargs.pop("attn_implementation", None)
                model = AutoModelForCausalLM.from_pretrained(base, **fallback_kwargs)
        else:
            fallback_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(base, **fallback_kwargs)
    adapter_cfg = run_dir / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            from peft import PeftModel  # pylint: disable=import-outside-toplevel

            model = PeftModel.from_pretrained(
                model,
                str(run_dir),
                device_map={"": str(device)},
            )
            if merge_lora and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
        except Exception as exc:
            raise RuntimeError(f"Failed to load text adapter from {run_dir}.") from exc
    model.to(device)
    model.eval()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    if hasattr(model, "base_model") and getattr(model.base_model, "generation_config", None) is not None:
        model.base_model.generation_config.max_length = None
    model = _compile_forward_if_available(
        model,
        compile_model=use_cuda if compile_model is None else bool(compile_model),
        mode=compile_mode,
    )
    return LoadedTextModel(
        name=run_dir.name,
        model=model,
        tokenizer=tokenizer,
        run_dir=run_dir,
        training_config=training_cfg,
    )


def load_diffusion_model(
    run_dir: Path,
    device: torch.device,
    strict_adapter_load: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
    compile_model: Optional[bool] = None,
    compile_mode: str = "reduce-overhead",
) -> LoadedDiffusionModel:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel  # pylint: disable=import-outside-toplevel

    training_cfg_path = run_dir / "training_config.yaml"
    if not training_cfg_path.exists() and run_dir.name.startswith("checkpoint-"):
        # Training checkpoints usually store only step artifacts (e.g. unet/, training_state.pt).
        # Reuse the parent run's config when evaluating an intermediate checkpoint directory.
        parent_cfg = run_dir.parent / "training_config.yaml"
        if parent_cfg.exists():
            training_cfg_path = parent_cfg
    training_cfg = load_yaml(training_cfg_path)
    base = training_cfg["model_name_or_path"]
    use_cuda = _is_cuda_device(device)
    torch_dtype = _default_cuda_dtype(device, torch_dtype)
    if use_cuda:
        _configure_cuda_inference_backends()
        _enable_cuda_sdpa_backends()
    pipe = StableDiffusionPipeline.from_pretrained(
        base,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        torch_dtype=torch_dtype,
    )
    finetune_mode = str(training_cfg.get("finetune_mode", "lora"))
    adapter_cfg = run_dir / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            from peft import PeftModel  # pylint: disable=import-outside-toplevel

            pipe.unet = PeftModel.from_pretrained(
                pipe.unet,
                str(run_dir),
                device_map={"": str(device)},
            )
        except Exception as exc:
            if strict_adapter_load:
                raise RuntimeError(
                    f"Failed to load diffusion adapter from {run_dir}. "
                    "Set runtime.strict_adapter_load=false only for local debugging."
                ) from exc
    elif finetune_mode == "full_unet" or (run_dir / "unet").exists():
        unet_dir = run_dir / "unet"
        if not unet_dir.exists():
            raise RuntimeError(
                f"Run {run_dir} is marked as finetune_mode=full_unet but no UNet artifact was found at {unet_dir}."
            )
        pipe.unet = UNet2DConditionModel.from_pretrained(str(unet_dir), torch_dtype=torch_dtype)
    elif finetune_mode not in {"lora", "full_unet"}:
        raise ValueError(f"Unsupported finetune_mode={finetune_mode!r} in training_config.yaml.")
    pipe = pipe.to(device)
    if use_cuda:
        pipe.unet = _channels_last_if_available(pipe.unet)
        pipe.vae = _channels_last_if_available(pipe.vae)
        # Prefer optimized attention kernels when backend support is available.
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        compile_enabled = use_cuda if compile_model is None else bool(compile_model)
        # Match the benchmarking strategy: compile only UNet.
        # Compiling VAE/text_encoder can hurt stability in iterative inference loops.
        pipe.unet = _compile_forward_if_available(pipe.unet, compile_enabled, mode=compile_mode)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return LoadedDiffusionModel(
        name=run_dir.name,
        pipe=pipe,
        run_dir=run_dir,
        training_config=training_cfg,
    )


def validate_cpk_diffusion_triplet(draft: LoadedDiffusionModel, q1: LoadedDiffusionModel, q2: LoadedDiffusionModel) -> None:
    fields = ["dataset_key", "model_name_or_path", "resolution"]
    for field in fields:
        values = {
            draft.training_config.get(field),
            q1.training_config.get(field),
            q2.training_config.get(field),
        }
        if len(values) != 1:
            raise ValueError(f"CP-k diffusion model mismatch in {field}: {values}")


def diffusion_noise_pred_score(model: LoadedDiffusionModel, image_tensor: torch.Tensor, prompt: str, timestep: int) -> float:
    pipe = model.pipe
    device = pipe.device
    vae_dtype = next(pipe.vae.parameters()).dtype
    image_tensor = image_tensor.to(device=device, dtype=vae_dtype)
    with torch.inference_mode():
        latent = pipe.vae.encode(image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latent)
        t = torch.tensor([timestep], dtype=torch.long, device=device)
        noisy = pipe.scheduler.add_noise(latent, noise, t)
        text_inputs = pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        emb = pipe.text_encoder(text_inputs.input_ids)[0]
        pred = pipe.unet(noisy, t, emb).sample
        mse = F.mse_loss(pred.float(), noise.float(), reduction="mean")
    return float(-mse.item())


def diffusion_prepare_latent(model: LoadedDiffusionModel, image_tensor: torch.Tensor) -> torch.Tensor:
    pipe = model.pipe
    device = pipe.device
    vae_dtype = next(pipe.vae.parameters()).dtype
    with torch.inference_mode():
        latent = pipe.vae.encode(image_tensor.to(device=device, dtype=vae_dtype)).latent_dist.sample() * pipe.vae.config.scaling_factor
    return latent


def diffusion_prompt_embeds(model: LoadedDiffusionModel, prompt: str) -> torch.Tensor:
    pipe = model.pipe
    device = pipe.device
    with torch.inference_mode():
        text_inputs = pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        return pipe.text_encoder(text_inputs.input_ids)[0]


def diffusion_prompt_embeds_batch(model: LoadedDiffusionModel, prompts: Sequence[str]) -> torch.Tensor:
    pipe = model.pipe
    device = pipe.device
    with torch.inference_mode():
        text_inputs = pipe.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        return pipe.text_encoder(text_inputs.input_ids)[0]


def diffusion_cfg_prompt_embeds_batch(model: LoadedDiffusionModel, prompts: Sequence[str]) -> torch.Tensor:
    pipe = model.pipe
    device = pipe.device
    with torch.inference_mode():
        cond_inputs = pipe.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        uncond_inputs = pipe.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        uncond_embeds = pipe.text_encoder(uncond_inputs.input_ids)[0]
        cond_embeds = pipe.text_encoder(cond_inputs.input_ids)[0]
        return torch.cat([uncond_embeds, cond_embeds], dim=0)


def diffusion_predict_eps(
    model: LoadedDiffusionModel,
    latent_xt: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep: int,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    pipe = model.pipe
    device = pipe.device
    unet_dtype = next(pipe.unet.parameters()).dtype
    do_classifier_free_guidance = guidance_scale > 1.0 and prompt_embeds.shape[0] == 2 * latent_xt.shape[0]
    latent_model_input = torch.cat([latent_xt] * 2) if do_classifier_free_guidance else latent_xt
    t = torch.full((latent_model_input.shape[0],), int(timestep), dtype=torch.long, device=device)
    with torch.inference_mode():
        pred = pipe.unet(
            latent_model_input.to(device=device, dtype=unet_dtype),
            t,
            prompt_embeds.to(device=device, dtype=unet_dtype),
        ).sample
        # Detach storage from compiled CUDA graph outputs so successive compiled
        # invocations (e.g., eps0 then eps_t) cannot overwrite tensors we still use.
        pred = pred.clone()
    if do_classifier_free_guidance:
        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    return pred


def scheduler_alpha_bar(model: LoadedDiffusionModel, timestep: int) -> float:
    alphas_cumprod = model.pipe.scheduler.alphas_cumprod
    t = min(max(int(timestep), 0), int(alphas_cumprod.shape[0] - 1))
    return float(alphas_cumprod[t].item())


def pia_alpha_bar(model: LoadedDiffusionModel, timestep: int, beta_schedule: str = "pipeline") -> float:
    if beta_schedule == "reference_linear":
        t = min(max(int(timestep), 0), 999)
        betas = torch.linspace(0.00085, 0.012, 1000, dtype=torch.float32)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        return float(alphas_cumprod[t].item())
    if beta_schedule == "pipeline":
        return scheduler_alpha_bar(model, timestep)
    raise ValueError(f"Unsupported PIA beta schedule: {beta_schedule}")


def build_xt_with_pia_schedule(
    model: LoadedDiffusionModel,
    x0_latent: torch.Tensor,
    eps: torch.Tensor,
    timestep: int,
    beta_schedule: str = "pipeline",
) -> torch.Tensor:
    """
    Build the forward-noised latent used by PIA.

    `pipeline` intentionally uses the scheduler's training noising distribution
    (`add_noise` / `alphas_cumprod`) from the loaded checkpoint config. This is
    separate from the scheduler's inference-time sampler class or timestep grid.
    """
    if beta_schedule == "pipeline":
        pipe = model.pipe
        t = torch.full((x0_latent.shape[0],), int(timestep), dtype=torch.long, device=x0_latent.device)
        return pipe.scheduler.add_noise(x0_latent, eps, t)

    alpha_bar_t = pia_alpha_bar(model, timestep, beta_schedule=beta_schedule)
    return build_xt_from_x0_eps(x0_latent, eps, alpha_bar_t)


def build_xt_from_x0_eps(x0_latent: torch.Tensor, eps: torch.Tensor, alpha_bar_t: float) -> torch.Tensor:
    a = math.sqrt(max(alpha_bar_t, 1e-12))
    b = math.sqrt(max(1.0 - alpha_bar_t, 1e-12))
    return a * x0_latent + b * eps


def pia_score_from_latent(
    model: LoadedDiffusionModel,
    x0_latent: torch.Tensor,
    prompt: str,
    timestep: int,
    norm_p: int = 4,
    guidance_scale: float = 1.0,
    beta_schedule: str = "pipeline",
) -> Dict[str, Any]:
    prompt_embeds = diffusion_cfg_prompt_embeds_batch(model, [prompt]) if guidance_scale > 1.0 else diffusion_prompt_embeds(model, prompt)
    eps0 = diffusion_predict_eps(model, x0_latent, prompt_embeds, timestep=0, guidance_scale=guidance_scale)
    xt = build_xt_with_pia_schedule(model, x0_latent, eps0, timestep, beta_schedule=beta_schedule)
    eps_t = diffusion_predict_eps(model, xt, prompt_embeds, timestep=timestep, guidance_scale=guidance_scale)
    delta = (eps0 - eps_t).abs().pow(norm_p)
    score = float(delta.flatten(1).sum(dim=-1)[0].item())
    return {
        "pia_score": score,
        "eps0": eps0,
        "eps_t": eps_t,
        "xt": xt,
    }


def pia_scores_from_latents_batch(
    model: LoadedDiffusionModel,
    x0_latents: torch.Tensor,
    prompts: Sequence[str],
    timestep: int,
    norm_p: int = 4,
    amp_enabled: bool = True,
    guidance_scale: float = 1.0,
    beta_schedule: str = "pipeline",
    return_xt: bool = False,
) -> Any:
    """Compute PIA scores for a batch. Returns list of floats (len == batch)."""
    pipe = model.pipe
    device_type = "cuda" if str(pipe.device).startswith("cuda") else "cpu"
    # If we're on CPU, autocast provides no benefit here.
    autocast_enabled = bool(amp_enabled and device_type == "cuda")

    with torch.inference_mode():
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
            prompt_embeds = (
                diffusion_cfg_prompt_embeds_batch(model, prompts)
                if guidance_scale > 1.0
                else diffusion_prompt_embeds_batch(model, prompts)
            )
            eps0 = diffusion_predict_eps(model, x0_latents, prompt_embeds, timestep=0, guidance_scale=guidance_scale)
            xt = build_xt_with_pia_schedule(model, x0_latents, eps0, timestep, beta_schedule=beta_schedule)
            eps_t = diffusion_predict_eps(model, xt, prompt_embeds, timestep=timestep, guidance_scale=guidance_scale)
            delta = (eps0 - eps_t).abs().pow(norm_p)
            scores = delta.flatten(1).sum(dim=-1)
    out_scores = [float(x) for x in scores.detach().cpu().tolist()]
    if return_xt:
        return out_scores, xt.detach()
    return out_scores


def _ddpm_transition_params_from_eps(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    alpha_bar_t: float,
    alpha_bar_prev: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a_t = max(float(alpha_bar_t), 1.0e-8)
    a_prev = max(float(alpha_bar_prev), 1.0e-8)
    sqrt_a_t = math.sqrt(a_t)
    sqrt_one_minus_a_t = math.sqrt(max(1.0 - a_t, 1.0e-8))
    x0_pred = (x_t - sqrt_one_minus_a_t * eps_pred) / max(sqrt_a_t, 1.0e-8)
    alpha_t = min(max(a_t / max(a_prev, 1.0e-8), 1.0e-8), 1.0)
    beta_t = max(1.0 - alpha_t, 1.0e-8)
    denom = max(1.0 - a_t, 1.0e-8)
    coef1 = math.sqrt(a_prev) * beta_t / denom
    coef2 = math.sqrt(alpha_t) * (1.0 - a_prev) / denom
    mean = coef1 * x0_pred + coef2 * x_t
    var = beta_t * (1.0 - a_prev) / denom
    var_t = torch.full_like(x_t, float(max(var, 1.0e-8)))
    return mean, var_t


def _sample_logprob_diag_gaussian(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    # Use float32 math for numerical stability in log-density computation.
    # In fp16, tiny variance floors can underflow to zero and yield -inf terms.
    x = x.to(torch.float32)
    mean = mean.to(torch.float32)
    var = var.to(torch.float32).clamp_min(1.0e-8)
    log_two_pi = math.log(2.0 * math.pi)
    terms = -0.5 * (((x - mean).pow(2) / var) + torch.log(var) + log_two_pi)
    return terms.flatten(1).sum(dim=1)


def build_descending_timestep_path(start_timestep: int, trajectory_steps: int) -> List[int]:
    start = max(1, int(start_timestep))
    steps = max(1, int(trajectory_steps))
    if steps == 1:
        return [start, 0]
    path = [int(round(x)) for x in np.linspace(start, 0, steps + 1)]
    cleaned: List[int] = []
    for val in path:
        if not cleaned or val != cleaned[-1]:
            cleaned.append(val)
    if cleaned[-1] != 0:
        cleaned.append(0)
    cleaned = sorted(set(cleaned), reverse=True)
    if cleaned[-1] != 0:
        cleaned.append(0)
    return cleaned


def diffusion_logprob_trajectory_from_xt_batch(
    draft_model: LoadedDiffusionModel,
    q1_model: LoadedDiffusionModel,
    q2_model: LoadedDiffusionModel,
    x_t_start: torch.Tensor,
    prompts: Sequence[str],
    start_timestep: int,
    trajectory_steps: int,
    guidance_scale: float = 1.0,
    prompt_embed_cache: Optional[MutableMapping[Tuple[int, str], torch.Tensor]] = None,
    store_per_step: bool = False,
) -> List[Dict[str, Any]]:
    if x_t_start.ndim < 2:
        raise ValueError("x_t_start must be a batch tensor [B,...].")
    batch_size = int(x_t_start.shape[0])
    if batch_size != len(prompts):
        raise ValueError("Batch size mismatch: x_t_start and prompts must have equal length.")
    if batch_size == 0:
        return []

    def _prompt_embeds_with_cache(model: LoadedDiffusionModel) -> torch.Tensor:
        if guidance_scale > 1.0:
            return diffusion_cfg_prompt_embeds_batch(model, prompts)
        if prompt_embed_cache is None:
            return diffusion_prompt_embeds_batch(model, prompts)
        embeds: List[torch.Tensor] = []
        for text in prompts:
            key = (id(model.pipe), text)
            cached = prompt_embed_cache.get(key)
            if cached is None:
                cached = diffusion_prompt_embeds(model, text)
                prompt_embed_cache[key] = cached
            embeds.append(cached)
        return torch.cat(embeds, dim=0)

    prompt_embeds_p = _prompt_embeds_with_cache(draft_model)
    prompt_embeds_q1 = _prompt_embeds_with_cache(q1_model)
    prompt_embeds_q2 = _prompt_embeds_with_cache(q2_model)
    timestep_path = build_descending_timestep_path(start_timestep=start_timestep, trajectory_steps=trajectory_steps)
    x_t = x_t_start.to(device=draft_model.pipe.device)
    traj_p = torch.zeros(batch_size, dtype=torch.float32, device=x_t.device)
    traj_q1 = torch.zeros(batch_size, dtype=torch.float32, device=x_t.device)
    traj_q2 = torch.zeros(batch_size, dtype=torch.float32, device=x_t.device)
    p_steps: List[List[float]] = [[] for _ in range(batch_size)]
    q1_steps: List[List[float]] = [[] for _ in range(batch_size)]
    q2_steps: List[List[float]] = [[] for _ in range(batch_size)]

    for idx in range(len(timestep_path) - 1):
        t_cur = int(timestep_path[idx])
        t_prev = int(timestep_path[idx + 1])
        if t_cur <= t_prev:
            continue
        eps_p = diffusion_predict_eps(draft_model, x_t, prompt_embeds_p, timestep=t_cur, guidance_scale=guidance_scale)
        eps_q1 = diffusion_predict_eps(q1_model, x_t, prompt_embeds_q1, timestep=t_cur, guidance_scale=guidance_scale)
        eps_q2 = diffusion_predict_eps(q2_model, x_t, prompt_embeds_q2, timestep=t_cur, guidance_scale=guidance_scale)
        alpha_bar_t = scheduler_alpha_bar(draft_model, t_cur)
        alpha_bar_prev = scheduler_alpha_bar(draft_model, t_prev)
        mean_p, var_t = _ddpm_transition_params_from_eps(x_t, eps_p, alpha_bar_t=alpha_bar_t, alpha_bar_prev=alpha_bar_prev)
        mean_q1, _ = _ddpm_transition_params_from_eps(x_t, eps_q1, alpha_bar_t=alpha_bar_t, alpha_bar_prev=alpha_bar_prev)
        mean_q2, _ = _ddpm_transition_params_from_eps(x_t, eps_q2, alpha_bar_t=alpha_bar_t, alpha_bar_prev=alpha_bar_prev)
        noise = torch.randn_like(mean_p)
        x_prev = mean_p + torch.sqrt(var_t) * noise
        step_p = _sample_logprob_diag_gaussian(x_prev, mean_p, var_t).to(torch.float32)
        step_q1 = _sample_logprob_diag_gaussian(x_prev, mean_q1, var_t).to(torch.float32)
        step_q2 = _sample_logprob_diag_gaussian(x_prev, mean_q2, var_t).to(torch.float32)
        traj_p = traj_p + step_p
        traj_q1 = traj_q1 + step_q1
        traj_q2 = traj_q2 + step_q2
        if store_per_step:
            vals_p = step_p.detach().cpu().tolist()
            vals_q1 = step_q1.detach().cpu().tolist()
            vals_q2 = step_q2.detach().cpu().tolist()
            for b in range(batch_size):
                p_steps[b].append(float(vals_p[b]))
                q1_steps[b].append(float(vals_q1[b]))
                q2_steps[b].append(float(vals_q2[b]))
        x_t = x_prev

    rows: List[Dict[str, Any]] = []
    traj_p_list = traj_p.detach().cpu().tolist()
    traj_q1_list = traj_q1.detach().cpu().tolist()
    traj_q2_list = traj_q2.detach().cpu().tolist()
    num_realized_steps = max(1, len(timestep_path) - 1)
    for b in range(batch_size):
        logp = float(traj_p_list[b])
        logq1 = float(traj_q1_list[b])
        logq2 = float(traj_q2_list[b])
        row = {
            "trajectory_start_timestep": int(start_timestep),
            "trajectory_timesteps": [int(x) for x in timestep_path],
            "trajectory_logscore_p": logp,
            "trajectory_logscore_q1": logq1,
            "trajectory_logscore_q2": logq2,
            "trajectory_mean_logscore_p": logp / float(num_realized_steps),
            "trajectory_mean_logscore_q1": logq1 / float(num_realized_steps),
            "trajectory_mean_logscore_q2": logq2 / float(num_realized_steps),
            "cpk_logratio_p_over_q1": logp - logq1,
            "cpk_logratio_p_over_q2": logp - logq2,
            "cpk_logratio_max": max(logp - logq1, logp - logq2),
            "cpk_logratio_p_over_q1_mean": (logp - logq1) / float(num_realized_steps),
            "cpk_logratio_p_over_q2_mean": (logp - logq2) / float(num_realized_steps),
            "cpk_logratio_max_mean": max(logp - logq1, logp - logq2) / float(num_realized_steps),
            "score_mode": "trajectory_logprob_ratio",
        }
        if store_per_step:
            row["trajectory_logscore_p_per_step"] = p_steps[b]
            row["trajectory_logscore_q1_per_step"] = q1_steps[b]
            row["trajectory_logscore_q2_per_step"] = q2_steps[b]
        rows.append(row)
    return rows


def diffusion_logprob_ratios_from_realized_path_batch(
    draft_model: LoadedDiffusionModel,
    q1_model: LoadedDiffusionModel,
    q2_model: LoadedDiffusionModel,
    latent_path: Sequence[torch.Tensor],
    prompts: Sequence[str],
    timesteps: Sequence[int],
    guidance_scale: float = 1.0,
    prompt_embed_cache: Optional[MutableMapping[Tuple[int, str], torch.Tensor]] = None,
    store_per_step: bool = False,
) -> List[Dict[str, Any]]:
    if len(latent_path) < 2:
        raise ValueError("latent_path must contain at least two tensors: [x_t, x_{t-1}, ...]")
    if len(timesteps) != len(latent_path) - 1:
        raise ValueError("timesteps must have length len(latent_path) - 1.")
    batch_size = int(latent_path[0].shape[0])
    if batch_size != len(prompts):
        raise ValueError("Batch size mismatch: latent_path batch and prompts must match.")

    def _prompt_embeds_with_cache(model: LoadedDiffusionModel) -> torch.Tensor:
        if guidance_scale > 1.0:
            return diffusion_cfg_prompt_embeds_batch(model, prompts)
        if prompt_embed_cache is None:
            return diffusion_prompt_embeds_batch(model, prompts)
        embeds: List[torch.Tensor] = []
        for text in prompts:
            key = (id(model.pipe), text)
            cached = prompt_embed_cache.get(key)
            if cached is None:
                cached = diffusion_prompt_embeds(model, text)
                prompt_embed_cache[key] = cached
            embeds.append(cached)
        return torch.cat(embeds, dim=0)

    prompt_embeds_p = _prompt_embeds_with_cache(draft_model)
    prompt_embeds_q1 = _prompt_embeds_with_cache(q1_model)
    prompt_embeds_q2 = _prompt_embeds_with_cache(q2_model)
    traj_p = torch.zeros(batch_size, dtype=torch.float32, device=draft_model.pipe.device)
    traj_q1 = torch.zeros(batch_size, dtype=torch.float32, device=draft_model.pipe.device)
    traj_q2 = torch.zeros(batch_size, dtype=torch.float32, device=draft_model.pipe.device)
    p_steps: List[List[float]] = [[] for _ in range(batch_size)]
    q1_steps: List[List[float]] = [[] for _ in range(batch_size)]
    q2_steps: List[List[float]] = [[] for _ in range(batch_size)]

    for step_idx, t_cur in enumerate(timesteps):
        x_t = latent_path[step_idx].to(device=draft_model.pipe.device)
        x_prev = latent_path[step_idx + 1].to(device=draft_model.pipe.device)
        t_cur_int = int(t_cur)
        t_prev = int(timesteps[step_idx + 1]) if (step_idx + 1) < len(timesteps) else 0
        if t_cur_int <= t_prev:
            t_prev = max(0, t_cur_int - 1)
        eps_p = diffusion_predict_eps(draft_model, x_t, prompt_embeds_p, timestep=t_cur_int, guidance_scale=guidance_scale)
        eps_q1 = diffusion_predict_eps(q1_model, x_t, prompt_embeds_q1, timestep=t_cur_int, guidance_scale=guidance_scale)
        eps_q2 = diffusion_predict_eps(q2_model, x_t, prompt_embeds_q2, timestep=t_cur_int, guidance_scale=guidance_scale)
        alpha_bar_t = scheduler_alpha_bar(draft_model, t_cur_int)
        alpha_bar_prev = scheduler_alpha_bar(draft_model, t_prev)
        mean_p, var_t = _ddpm_transition_params_from_eps(x_t, eps_p, alpha_bar_t=alpha_bar_t, alpha_bar_prev=alpha_bar_prev)
        mean_q1, _ = _ddpm_transition_params_from_eps(x_t, eps_q1, alpha_bar_t=alpha_bar_t, alpha_bar_prev=alpha_bar_prev)
        mean_q2, _ = _ddpm_transition_params_from_eps(x_t, eps_q2, alpha_bar_t=alpha_bar_t, alpha_bar_prev=alpha_bar_prev)
        step_p = _sample_logprob_diag_gaussian(x_prev, mean_p, var_t).to(torch.float32)
        step_q1 = _sample_logprob_diag_gaussian(x_prev, mean_q1, var_t).to(torch.float32)
        step_q2 = _sample_logprob_diag_gaussian(x_prev, mean_q2, var_t).to(torch.float32)
        traj_p = traj_p + step_p
        traj_q1 = traj_q1 + step_q1
        traj_q2 = traj_q2 + step_q2
        if store_per_step:
            vals_p = step_p.detach().cpu().tolist()
            vals_q1 = step_q1.detach().cpu().tolist()
            vals_q2 = step_q2.detach().cpu().tolist()
            for b in range(batch_size):
                p_steps[b].append(float(vals_p[b]))
                q1_steps[b].append(float(vals_q1[b]))
                q2_steps[b].append(float(vals_q2[b]))

    rows: List[Dict[str, Any]] = []
    traj_p_list = traj_p.detach().cpu().tolist()
    traj_q1_list = traj_q1.detach().cpu().tolist()
    traj_q2_list = traj_q2.detach().cpu().tolist()
    num_steps = max(1, len(timesteps))
    for b in range(batch_size):
        logp = float(traj_p_list[b])
        logq1 = float(traj_q1_list[b])
        logq2 = float(traj_q2_list[b])
        row = {
            "trajectory_timesteps": [int(x) for x in timesteps],
            "trajectory_logscore_p": logp,
            "trajectory_logscore_q1": logq1,
            "trajectory_logscore_q2": logq2,
            "trajectory_mean_logscore_p": logp / float(num_steps),
            "trajectory_mean_logscore_q1": logq1 / float(num_steps),
            "trajectory_mean_logscore_q2": logq2 / float(num_steps),
            "cpk_logratio_p_over_q1": logp - logq1,
            "cpk_logratio_p_over_q2": logp - logq2,
            "cpk_logratio_max": max(logp - logq1, logp - logq2),
            "cpk_logratio_p_over_q1_mean": (logp - logq1) / float(num_steps),
            "cpk_logratio_p_over_q2_mean": (logp - logq2) / float(num_steps),
            "cpk_logratio_max_mean": max(logp - logq1, logp - logq2) / float(num_steps),
            "score_mode": "trajectory_logprob_ratio",
        }
        if store_per_step:
            row["trajectory_logscore_p_per_step"] = p_steps[b]
            row["trajectory_logscore_q1_per_step"] = q1_steps[b]
            row["trajectory_logscore_q2_per_step"] = q2_steps[b]
        rows.append(row)
    return rows


def cpk_shared_path_scores(
    draft_model: LoadedDiffusionModel,
    q1_model: LoadedDiffusionModel,
    q2_model: LoadedDiffusionModel,
    x0_latent: torch.Tensor,
    prompt: str,
    timestep_grid: Sequence[int],
    prompt_embed_cache: Optional[MutableMapping[Tuple[int, str], torch.Tensor]] = None,
) -> Dict[str, Any]:
    def _prompt_embed_with_cache(model: LoadedDiffusionModel, text: str) -> torch.Tensor:
        if prompt_embed_cache is None:
            return diffusion_prompt_embeds(model, text)
        key = (id(model.pipe), text)
        cached = prompt_embed_cache.get(key)
        if cached is None:
            cached = diffusion_prompt_embeds(model, text)
            prompt_embed_cache[key] = cached
        return cached

    prompt_embeds_p = _prompt_embed_with_cache(draft_model, prompt)
    prompt_embeds_q1 = _prompt_embed_with_cache(q1_model, prompt)
    prompt_embeds_q2 = _prompt_embed_with_cache(q2_model, prompt)

    # Shared trajectory source: draft model proximal initialization.
    eps0_p = diffusion_predict_eps(draft_model, x0_latent, prompt_embeds_p, timestep=0)

    p_steps: List[float] = []
    q1_steps: List[float] = []
    q2_steps: List[float] = []
    for timestep in timestep_grid:
        xt = build_xt_with_pia_schedule(draft_model, x0_latent, eps0_p, timestep, beta_schedule="pipeline")
        eps_p = diffusion_predict_eps(draft_model, xt, prompt_embeds_p, timestep=timestep)
        eps_q1 = diffusion_predict_eps(q1_model, xt, prompt_embeds_q1, timestep=timestep)
        eps_q2 = diffusion_predict_eps(q2_model, xt, prompt_embeds_q2, timestep=timestep)

        p_steps.append(float(-(eps_p - eps0_p).pow(2).mean().item()))
        q1_steps.append(float(-(eps_q1 - eps0_p).pow(2).mean().item()))
        q2_steps.append(float(-(eps_q2 - eps0_p).pow(2).mean().item()))

    traj_p = float(sum(p_steps))
    traj_q1 = float(sum(q1_steps))
    traj_q2 = float(sum(q2_steps))
    return {
        "trajectory_logscore_p_per_step": p_steps,
        "trajectory_logscore_q1_per_step": q1_steps,
        "trajectory_logscore_q2_per_step": q2_steps,
        "trajectory_logscore_p": traj_p,
        "trajectory_logscore_q1": traj_q1,
        "trajectory_logscore_q2": traj_q2,
        "trajectory_mean_logscore_p": traj_p / max(1, len(timestep_grid)),
        "trajectory_mean_logscore_q1": traj_q1 / max(1, len(timestep_grid)),
        "trajectory_mean_logscore_q2": traj_q2 / max(1, len(timestep_grid)),
        "cpk_logratio_p_over_q1": traj_p - traj_q1,
        "cpk_logratio_p_over_q2": traj_p - traj_q2,
        "cpk_logratio_max": max(traj_p - traj_q1, traj_p - traj_q2),
        "cpk_logratio_p_over_q1_mean": (traj_p - traj_q1) / max(1, len(timestep_grid)),
        "cpk_logratio_p_over_q2_mean": (traj_p - traj_q2) / max(1, len(timestep_grid)),
        "cpk_logratio_max_mean": max(traj_p - traj_q1, traj_p - traj_q2) / max(1, len(timestep_grid)),
        "score_mode": "shared_path_neg_noise_mse",
    }


def cpk_shared_path_scores_batch(
    draft_model: LoadedDiffusionModel,
    q1_model: LoadedDiffusionModel,
    q2_model: LoadedDiffusionModel,
    x0_latents: torch.Tensor,
    prompts: Sequence[str],
    timestep_grid: Sequence[int],
    prompt_embed_cache: Optional[MutableMapping[Tuple[int, str], torch.Tensor]] = None,
) -> List[Dict[str, Any]]:
    if x0_latents.ndim < 2:
        raise ValueError("x0_latents must be a batch tensor [B,...].")
    batch_size = int(x0_latents.shape[0])
    if batch_size != len(prompts):
        raise ValueError("Batch size mismatch: x0_latents and prompts must have equal length.")
    if batch_size == 0:
        return []

    def _prompt_embed_with_cache(model: LoadedDiffusionModel, text: str) -> torch.Tensor:
        if prompt_embed_cache is None:
            return diffusion_prompt_embeds(model, text)
        key = (id(model.pipe), text)
        cached = prompt_embed_cache.get(key)
        if cached is None:
            cached = diffusion_prompt_embeds(model, text)
            prompt_embed_cache[key] = cached
        return cached

    prompt_embeds_p = torch.cat([_prompt_embed_with_cache(draft_model, p) for p in prompts], dim=0)
    prompt_embeds_q1 = torch.cat([_prompt_embed_with_cache(q1_model, p) for p in prompts], dim=0)
    prompt_embeds_q2 = torch.cat([_prompt_embed_with_cache(q2_model, p) for p in prompts], dim=0)

    eps0_p = diffusion_predict_eps(draft_model, x0_latents, prompt_embeds_p, timestep=0)

    p_steps: List[List[float]] = [[] for _ in range(batch_size)]
    q1_steps: List[List[float]] = [[] for _ in range(batch_size)]
    q2_steps: List[List[float]] = [[] for _ in range(batch_size)]

    for timestep in timestep_grid:
        xt = build_xt_with_pia_schedule(draft_model, x0_latents, eps0_p, timestep, beta_schedule="pipeline")
        eps_p = diffusion_predict_eps(draft_model, xt, prompt_embeds_p, timestep=timestep)
        eps_q1 = diffusion_predict_eps(q1_model, xt, prompt_embeds_q1, timestep=timestep)
        eps_q2 = diffusion_predict_eps(q2_model, xt, prompt_embeds_q2, timestep=timestep)

        p_vals = (-(eps_p - eps0_p).pow(2).flatten(1).mean(dim=1)).detach().cpu().tolist()
        q1_vals = (-(eps_q1 - eps0_p).pow(2).flatten(1).mean(dim=1)).detach().cpu().tolist()
        q2_vals = (-(eps_q2 - eps0_p).pow(2).flatten(1).mean(dim=1)).detach().cpu().tolist()
        for idx in range(batch_size):
            p_steps[idx].append(float(p_vals[idx]))
            q1_steps[idx].append(float(q1_vals[idx]))
            q2_steps[idx].append(float(q2_vals[idx]))

    rows: List[Dict[str, Any]] = []
    for idx in range(batch_size):
        traj_p = float(sum(p_steps[idx]))
        traj_q1 = float(sum(q1_steps[idx]))
        traj_q2 = float(sum(q2_steps[idx]))
        rows.append(
            {
                "trajectory_logscore_p_per_step": p_steps[idx],
                "trajectory_logscore_q1_per_step": q1_steps[idx],
                "trajectory_logscore_q2_per_step": q2_steps[idx],
                "trajectory_logscore_p": traj_p,
                "trajectory_logscore_q1": traj_q1,
                "trajectory_logscore_q2": traj_q2,
                "trajectory_mean_logscore_p": traj_p / max(1, len(timestep_grid)),
                "trajectory_mean_logscore_q1": traj_q1 / max(1, len(timestep_grid)),
                "trajectory_mean_logscore_q2": traj_q2 / max(1, len(timestep_grid)),
                "cpk_logratio_p_over_q1": traj_p - traj_q1,
                "cpk_logratio_p_over_q2": traj_p - traj_q2,
                "cpk_logratio_max": max(traj_p - traj_q1, traj_p - traj_q2),
                "cpk_logratio_p_over_q1_mean": (traj_p - traj_q1) / max(1, len(timestep_grid)),
                "cpk_logratio_p_over_q2_mean": (traj_p - traj_q2) / max(1, len(timestep_grid)),
                "cpk_logratio_max_mean": max(traj_p - traj_q1, traj_p - traj_q2) / max(1, len(timestep_grid)),
                "score_mode": "shared_path_neg_noise_mse",
            }
        )
    return rows

def validate_cpk_text_triplet(draft: LoadedTextModel, q1: LoadedTextModel, q2: LoadedTextModel) -> None:
    fields = ["dataset_key", "model_name_or_path"]
    for field in fields:
        values = {
            draft.training_config.get(field),
            q1.training_config.get(field),
            q2.training_config.get(field),
        }
        if len(values) != 1:
            raise ValueError(f"CP-k model mismatch in {field}: {values}")


def _effective_text_max_len(loaded: LoadedTextModel) -> int:
    tokenizer_max = getattr(loaded.tokenizer, "model_max_length", 0)
    if isinstance(tokenizer_max, int) and 1 <= tokenizer_max <= 100_000:
        return int(tokenizer_max)
    cfg_max = loaded.training_config.get("max_seq_len")
    if isinstance(cfg_max, int) and cfg_max > 1:
        return int(cfg_max)
    return 2048


def realized_token_logprobs(
    loaded: LoadedTextModel,
    prompt: str,
    continuation: str,
    score_prompt_tokens: bool = False,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    tokenizer = loaded.tokenizer
    max_len = int(max_length) if max_length is not None else _effective_text_max_len(loaded)
    prepared = _prepare_continuation_first_inputs(tokenizer, prompt, continuation, max_len)
    ids = prepared["input_ids"]
    if not ids:
        empty_logits = torch.empty((0, loaded.model.config.vocab_size), dtype=torch.float32)
        return {
            "token_logprobs": [],
            "token_ids": [],
            "token_logits": empty_logits,
            "seq_logprob": 0.0,
            "mean_logprob": 0.0,
            "prompt_token_count": 0,
            "scored_token_count": 0,
        }

    input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.model.device)
    attention_mask = torch.ones_like(input_ids)
    mark_cuda_graph_step_if_available(loaded.model.device)
    with torch.inference_mode():
        logits = loaded.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    realized = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)[0]
    prompt_token_count = int(prepared["prompt_token_count"])
    valid_steps = max(0, len(ids) - 1)
    if score_prompt_tokens:
        start_idx = 0
        end_idx = valid_steps
    else:
        start_idx = min(int(prepared["score_start"]), valid_steps)
        end_idx = min(int(prepared["score_end"]), valid_steps)
    scored = realized[start_idx:end_idx]
    scored_logits = logits[0, start_idx:end_idx, :]
    scored_tokens = targets[0, start_idx:end_idx]
    return {
        "token_logprobs": scored.detach().cpu().tolist(),
        "token_ids": scored_tokens.detach().cpu().tolist(),
        "token_logits": scored_logits.detach().cpu(),
        "seq_logprob": float(scored.sum().item()),
        "mean_logprob": float(scored.mean().item() if scored.numel() else 0.0),
        "prompt_token_count": prompt_token_count,
        "scored_token_count": int(scored.numel()),
    }


def realized_token_logprobs_batch(
    loaded: LoadedTextModel,
    prompts: Sequence[str],
    continuations: Sequence[str],
    score_prompt_tokens: bool = False,
    max_length: Optional[int] = None,
    return_token_logprobs: bool = True,
    return_token_logits: bool = True,
    compute_minkpp_cfg: Optional[Dict[str, Any]] = None,
    prepared_batch: Optional[Sequence[Dict[str, Any]]] = None,
    seq_logprob_only: bool = False,
) -> List[Dict[str, Any]]:
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have the same length")
    if not prompts:
        return []

    tokenizer = loaded.tokenizer
    max_len = int(max_length) if max_length is not None else _effective_text_max_len(loaded)
    if prepared_batch is None:
        prepared = prepare_continuation_first_batch(tokenizer, prompts, continuations, max_len)
    else:
        _validate_prepared_batch(prepared_batch, expected_size=len(prompts), max_length=max_len)
        prepared = [dict(item) for item in prepared_batch]
    if seq_logprob_only:
        if compute_minkpp_cfg is not None:
            raise ValueError("seq_logprob_only is incompatible with compute_minkpp_cfg")
        return_token_logprobs = False
        return_token_logits = False

    seq_lengths = [len(item["input_ids"]) for item in prepared]
    max_batch_len = max(seq_lengths)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    input_ids = torch.full(
        (len(prepared), max_batch_len),
        fill_value=int(pad_token_id),
        dtype=torch.long,
        device=loaded.model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for idx, item in enumerate(prepared):
        ids = item["input_ids"]
        if not ids:
            continue
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=loaded.model.device)
        attention_mask[idx, : len(ids)] = 1

    mark_cuda_graph_step_if_available(loaded.model.device)
    with torch.inference_mode():
        logits = loaded.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    if seq_logprob_only:
        gathered_logprobs = -F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        log_probs = None
        realized = gathered_logprobs
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        realized = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    batch_token_scores: Optional[torch.Tensor] = None
    k_percent_grid: Optional[Sequence[int]] = None
    default_k_percent: Optional[int] = None
    if compute_minkpp_cfg is not None:
        if log_probs is None:
            raise ValueError("Internal error: log_probs unavailable for Min-K++ path")
        epsilon = float(compute_minkpp_cfg.get("epsilon", 1.0e-8))
        k_percent_grid = compute_minkpp_cfg.get("k_percent_grid", [5, 10, 20, 30, 40, 50])
        default_k_percent = int(compute_minkpp_cfg.get("default_k_percent", 20))
        probs = torch.softmax(logits, dim=-1)
        mu = (probs * log_probs).sum(dim=-1)
        second_moment = (probs * (log_probs * log_probs)).sum(dim=-1)
        var = (second_moment - (mu * mu)).clamp_min(0.0)
        batch_token_scores = (realized - mu) / (torch.sqrt(var) + epsilon)

    results: List[Dict[str, Any]] = []
    for idx in range(len(prompts)):
        item = prepared[idx]
        prompt_token_count = int(item["prompt_token_count"])
        valid_steps = max(0, int(seq_lengths[idx]) - 1)
        if score_prompt_tokens:
            start_idx = 0
            end_idx = valid_steps
        else:
            start_idx = min(int(item["score_start"]), valid_steps)
            end_idx = min(int(item["score_end"]), valid_steps)
        scored = realized[idx, start_idx:end_idx]
        scored_logits = None if seq_logprob_only else (logits[idx, start_idx:end_idx, :] if return_token_logits else None)
        scored_tokens = [] if seq_logprob_only else targets[idx, start_idx:end_idx].detach().cpu().tolist()
        result = {
            "token_logprobs": scored.detach().cpu().tolist() if return_token_logprobs else [],
            "token_ids": scored_tokens,
            "token_logits": scored_logits.detach().cpu() if scored_logits is not None else None,
            "seq_logprob": float(scored.sum().item()),
            "mean_logprob": float(scored.mean().item() if scored.numel() else 0.0),
            "prompt_token_count": prompt_token_count,
            "scored_token_count": int(scored.numel()),
        }
        if batch_token_scores is not None and k_percent_grid is not None and default_k_percent is not None:
            scored_token_scores = batch_token_scores[idx, start_idx:end_idx]
            if scored_token_scores.numel() == 0:
                by_k = {str(k): 0.0 for k in k_percent_grid}
                result.update(
                    {
                        "minkpp_token_scores": [],
                        "minkpp_scores_by_k": by_k,
                        "minkpp_default_score": 0.0,
                    }
                )
            else:
                sorted_scores = torch.sort(scored_token_scores, descending=False).values
                n = int(sorted_scores.numel())
                by_k = {}
                for k in k_percent_grid:
                    count = max(1, int(n * (float(k) / 100.0)))
                    by_k[str(k)] = float(sorted_scores[:count].mean().item())
                result.update(
                    {
                        "minkpp_token_scores": scored_token_scores.detach().cpu().tolist(),
                        "minkpp_scores_by_k": by_k,
                        "minkpp_default_score": float(by_k[str(default_k_percent)]),
                    }
                )
        results.append(result)
    return results


def realized_token_logprobs_from_token_ids_batch(
    loaded: LoadedTextModel,
    token_id_sequences: Sequence[Sequence[int]],
) -> List[Dict[str, Any]]:
    if not token_id_sequences:
        return []
    pad_token_id = loaded.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = loaded.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    lengths = [len(seq) for seq in token_id_sequences]
    max_len = max(lengths)
    input_rows = []
    mask_rows = []
    for seq in token_id_sequences:
        ids = list(seq)
        pad_count = max_len - len(ids)
        input_rows.append(ids + [pad_token_id] * pad_count)
        mask_rows.append([1] * len(ids) + [0] * pad_count)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=loaded.model.device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=loaded.model.device)

    mark_cuda_graph_step_if_available(loaded.model.device)
    with torch.inference_mode():
        logits = loaded.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    realized = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)

    results: List[Dict[str, Any]] = []
    for idx, seq_len in enumerate(lengths):
        valid_steps = max(0, int(seq_len) - 1)
        if valid_steps == 0:
            empty_logits = torch.empty((0, logits.shape[-1]), dtype=torch.float32)
            results.append(
                {
                    "token_logprobs": [],
                    "token_ids": [],
                    "token_logits": empty_logits,
                    "seq_logprob": 0.0,
                    "mean_logprob": 0.0,
                    "prompt_token_count": 1,
                    "scored_token_count": 0,
                }
            )
            continue
        scored = realized[idx, :valid_steps]
        scored_logits = logits[idx, :valid_steps, :]
        scored_tokens = targets[idx, :valid_steps]
        results.append(
            {
                "token_logprobs": scored.detach().cpu().tolist(),
                "token_ids": scored_tokens.detach().cpu().tolist(),
                "token_logits": scored_logits.detach().cpu(),
                "seq_logprob": float(scored.sum().item()),
                "mean_logprob": float(scored.mean().item()),
                "prompt_token_count": 1,
                "scored_token_count": int(valid_steps),
            }
        )
    return results


def minkpp_token_scores_from_token_ids_batch(
    loaded: LoadedTextModel,
    token_id_sequences: Sequence[Sequence[int]],
    epsilon: float,
) -> List[Dict[str, Any]]:
    if not token_id_sequences:
        return []
    pad_token_id = loaded.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = loaded.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    lengths = [len(seq) for seq in token_id_sequences]
    max_len = max(lengths)
    input_rows = []
    mask_rows = []
    for seq in token_id_sequences:
        ids = list(seq)
        pad_count = max_len - len(ids)
        input_rows.append(ids + [pad_token_id] * pad_count)
        mask_rows.append([1] * len(ids) + [0] * pad_count)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=loaded.model.device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=loaded.model.device)

    mark_cuda_graph_step_if_available(loaded.model.device)
    with torch.inference_mode():
        logits = loaded.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        realized = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        mu = (probs * log_probs).sum(dim=-1)
        second_moment = (probs * (log_probs * log_probs)).sum(dim=-1)
        var = (second_moment - (mu * mu)).clamp_min(0.0)
        token_scores = (realized - mu) / (torch.sqrt(var) + epsilon)

    results: List[Dict[str, Any]] = []
    for idx, seq_len in enumerate(lengths):
        valid_steps = max(0, int(seq_len) - 1)
        if valid_steps == 0:
            results.append({"minkpp_token_scores": [], "seq_logprob": 0.0, "scored_token_count": 0})
            continue
        scored_logprobs = realized[idx, :valid_steps]
        scores = token_scores[idx, :valid_steps]
        results.append(
            {
                "minkpp_token_scores": scores.detach().cpu().tolist(),
                "seq_logprob": float(scored_logprobs.sum().item()),
                "scored_token_count": int(valid_steps),
            }
        )
    return results


def generate_text_candidates(
    draft: LoadedTextModel,
    prompts: Sequence[str],
    generation_cfg: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    device = draft.model.device
    random.seed(seed)
    torch.manual_seed(seed)
    results: List[Dict[str, Any]] = []
    if not prompts:
        return results
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 64))
    temperature_raw = generation_cfg.get("temperature", 1.0)
    top_p_raw = generation_cfg.get("top_p", 0.95)
    top_k_raw = generation_cfg.get("top_k", 50)
    temperature = float(temperature_raw) if temperature_raw is not None else 1.0
    top_p = float(top_p_raw) if top_p_raw is not None else 1.0
    top_k = int(top_k_raw) if top_k_raw is not None else 0
    do_sample = bool(generation_cfg.get("do_sample", True))
    batch_size = int(generation_cfg.get("batch_size", len(prompts)))
    max_prompt_tokens_raw = generation_cfg.get("max_prompt_tokens")
    max_prompt_tokens = int(max_prompt_tokens_raw) if max_prompt_tokens_raw is not None else None
    pad_to_multiple_raw = generation_cfg.get("pad_to_multiple_of")
    pad_to_multiple_of = int(pad_to_multiple_raw) if pad_to_multiple_raw is not None else None
    prompt_length_buckets = [int(x) for x in generation_cfg.get("prompt_length_buckets", [])]
    cache_implementation = generation_cfg.get("cache_implementation")
    use_cache = generation_cfg.get("use_cache")
    record_timing = bool(generation_cfg.get("record_timing", True))
    if batch_size <= 0:
        raise ValueError("generation.batch_size must be >= 1")
    if max_prompt_tokens is not None and max_prompt_tokens <= 0:
        raise ValueError("generation.max_prompt_tokens must be >= 1")
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise ValueError("generation.pad_to_multiple_of must be >= 1")
    if any(bucket <= 0 for bucket in prompt_length_buckets):
        raise ValueError("generation.prompt_length_buckets values must be >= 1")
    prompt_length_buckets = sorted(set(prompt_length_buckets))

    pad_token_id = draft.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = draft.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start : start + batch_size])
        tokenization_start = time.perf_counter()
        tokenizer_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": max_prompt_tokens is not None,
        }
        if max_prompt_tokens is not None:
            tokenizer_kwargs["max_length"] = max_prompt_tokens
        if pad_to_multiple_of is not None:
            tokenizer_kwargs["pad_to_multiple_of"] = pad_to_multiple_of
        if prompt_length_buckets:
            prompt_lengths = [
                len(
                    draft.tokenizer.encode(
                        prompt,
                        add_special_tokens=False,
                        truncation=max_prompt_tokens is not None,
                        max_length=max_prompt_tokens,
                    )
                )
                for prompt in prompt_batch
            ]
            longest_prompt = max(prompt_lengths) if prompt_lengths else 0
            if max_prompt_tokens is not None:
                longest_prompt = min(longest_prompt, max_prompt_tokens)
            bucket_max = next((bucket for bucket in prompt_length_buckets if bucket >= longest_prompt), prompt_length_buckets[-1])
            if max_prompt_tokens is not None:
                bucket_max = min(bucket_max, max_prompt_tokens)
            if pad_to_multiple_of is not None and bucket_max % pad_to_multiple_of != 0:
                bucket_max = ((bucket_max + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            tokenizer_kwargs["padding"] = "max_length"
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max(1, int(bucket_max))
        enc = draft.tokenizer(prompt_batch, **tokenizer_kwargs).to(device)
        tokenization_seconds = time.perf_counter() - tokenization_start

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": int(pad_token_id),
        }
        if draft.tokenizer.eos_token_id is not None:
            generate_kwargs["eos_token_id"] = draft.tokenizer.eos_token_id
        if do_sample:
            if temperature_raw is not None and temperature != 1.0:
                generate_kwargs["temperature"] = temperature
            if top_p_raw is not None:
                generate_kwargs["top_p"] = top_p
            if top_k_raw is not None:
                generate_kwargs["top_k"] = top_k
        if cache_implementation is not None:
            generate_kwargs["cache_implementation"] = str(cache_implementation)
        if use_cache is not None:
            generate_kwargs["use_cache"] = bool(use_cache)
        mark_cuda_graph_step_if_available(device)
        if record_timing and _is_cuda_device(device):
            torch.cuda.synchronize(device)
        generate_start = time.perf_counter()
        with torch.inference_mode():
            generated = draft.model.generate(
                **enc,
                **generate_kwargs,
            )
        if record_timing and _is_cuda_device(device):
            torch.cuda.synchronize(device)
        generate_seconds = time.perf_counter() - generate_start
        input_width = int(enc["input_ids"].shape[1])
        decode_start = time.perf_counter()
        decoded_rows: List[Tuple[List[int], str, bool]] = []
        total_generated_tokens = 0
        generated_suffix = generated[:, input_width:]
        generated_id_lists = generated_suffix.detach().cpu().tolist()
        if hasattr(draft.tokenizer, "batch_decode"):
            continuations = draft.tokenizer.batch_decode(generated_id_lists, skip_special_tokens=True)
        else:
            continuations = [
                draft.tokenizer.decode(gen_id_list, skip_special_tokens=True)
                for gen_id_list in generated_id_lists
            ]
        eos_token_id = draft.tokenizer.eos_token_id
        for gen_id_list, continuation in zip(generated_id_lists, continuations):
            eos_reached = bool(eos_token_id is not None and eos_token_id in gen_id_list)
            total_generated_tokens += len(gen_id_list)
            decoded_rows.append((gen_id_list, continuation, eos_reached))
        decode_seconds = time.perf_counter() - decode_start
        tokens_per_second = float(total_generated_tokens / max(generate_seconds, 1.0e-9))
        for local_idx, (gen_id_list, continuation, eos_reached) in enumerate(decoded_rows):
            prompt_idx = start + local_idx
            prompt = prompt_batch[local_idx]
            results.append(
                {
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "generated_text": continuation,
                    "generated_ids": gen_id_list,
                    "decode_strategy": "sample" if do_sample else "greedy",
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_new_tokens": max_new_tokens,
                    "eos_reached": eos_reached,
                    "generated_token_count": len(gen_id_list),
                    "generation_batch_size": len(prompt_batch),
                    "generation_input_width": input_width,
                    "generation_tokenization_seconds": tokenization_seconds,
                    "generation_generate_seconds": generate_seconds,
                    "generation_decode_seconds": decode_seconds,
                    "generation_tokens_per_second": tokens_per_second,
                }
            )
    return results


def compute_minkpp(
    token_logits: torch.Tensor,
    realized_logprobs: Sequence[float],
    k_percent_grid: Sequence[int],
    default_k_percent: int,
    epsilon: float,
) -> Dict[str, Any]:
    token_scores = compute_minkpp_token_scores(token_logits, realized_logprobs, epsilon)
    if token_scores.numel() == 0:
        return {
            "minkpp_token_scores": [],
            "minkpp_scores_by_k": {str(k): 0.0 for k in k_percent_grid},
            "minkpp_default_score": 0.0,
        }
    sorted_scores, _ = torch.sort(token_scores, descending=False)
    by_k: Dict[str, float] = {}
    n = sorted_scores.numel()
    for k in k_percent_grid:
        # Use floor-style slicing like the reference implementation, but keep
        # a minimum of 1 token for short sequences.
        count = max(1, int(n * (float(k) / 100.0)))
        by_k[str(k)] = float(sorted_scores[:count].mean().item())
    return {
        "minkpp_token_scores": token_scores.detach().cpu().tolist(),
        "minkpp_scores_by_k": by_k,
        "minkpp_default_score": float(by_k[str(default_k_percent)]),
    }


def compute_minkpp_token_scores(
    token_logits: torch.Tensor,
    realized_logprobs: Sequence[float],
    epsilon: float,
) -> torch.Tensor:
    if token_logits.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    token_logits = token_logits.float()
    realized = torch.tensor(realized_logprobs, dtype=torch.float32, device=token_logits.device)
    log_probs = F.log_softmax(token_logits, dim=-1)
    probs = torch.softmax(token_logits, dim=-1)
    mu = (probs * log_probs).sum(dim=-1)
    # Align with mink-plus-plus reference expression:
    # sigma^2 = E[(log p)^2] - (E[log p])^2
    second_moment = (probs * (log_probs * log_probs)).sum(dim=-1)
    var = (second_moment - (mu * mu)).clamp_min(0.0)
    sigma = torch.sqrt(var.clamp_min(0.0))
    return (realized - mu) / (sigma + epsilon)


def collect_text_prompts(dataset_key: str, split_name: str, limit: int) -> List[Dict[str, Any]]:
    ds = load_from_disk(str(PROCESSED_DATA_ROOT / dataset_key / split_name))
    prompts: List[Dict[str, Any]] = []
    for idx in range(min(limit, len(ds))):
        row = ds[idx]
        try:
            prompt, target = split_prompt_target(row["text"])
        except ValueError as exc:
            raise ValueError(
                f"Malformed text sample for dataset={dataset_key} split={split_name} sample_id={idx}: {exc}"
            ) from exc
        prompts.append({"sample_id": idx, "prompt": prompt, "target_text": target})
    return prompts


def build_run_output_dir(output_root: Path, dataset: str, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / dataset / run_name / stamp
    ensure_dir(run_dir)
    return run_dir


def write_common_artifacts(
    run_dir: Path,
    run_config: Dict[str, Any],
    split_fingerprint_path: Optional[Path],
) -> None:
    write_yaml(run_dir / "run_config.yaml", run_config)
    write_json(
        run_dir / "run_metadata.json",
        {
            "created_at": now_iso(),
            "run_dir": str(run_dir),
        },
    )
    if split_fingerprint_path and split_fingerprint_path.exists():
        payload = read_json(split_fingerprint_path)
    else:
        payload = {}
    write_json(run_dir / "input_split_fingerprint.json", payload)
    dump_jsonl(run_dir / "errors.jsonl", [])


def maybe_orient_scores(scores: Sequence[float], direction: str) -> List[float]:
    if direction == "higher_is_member":
        return list(scores)
    if direction == "lower_is_member":
        return [-float(x) for x in scores]
    raise ValueError(f"Unsupported direction: {direction}")


def _binary_roc_curve(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ROC curve points (FPR, TPR) from scores and binary labels."""
    if preds.numel() == 0 or labels.numel() == 0:
        z = torch.zeros(1, dtype=torch.float32, device=preds.device)
        return z, z

    labels = labels.to(dtype=torch.long)
    pos = int((labels == 1).sum().item())
    neg = int((labels == 0).sum().item())
    if pos == 0 or neg == 0:
        z = torch.zeros(1, dtype=torch.float32, device=preds.device)
        return z, z

    order = torch.argsort(preds, descending=True)
    sorted_preds = preds[order]
    sorted_labels = labels[order]
    tp_cum = torch.cumsum((sorted_labels == 1).to(torch.float32), dim=0)
    fp_cum = torch.cumsum((sorted_labels == 0).to(torch.float32), dim=0)

    # Keep only the last index in each score tie group.
    change = sorted_preds[1:] != sorted_preds[:-1]
    group_ends = torch.where(change)[0]
    group_ends = torch.cat([group_ends, torch.tensor([len(sorted_preds) - 1], device=preds.device)])

    tpr = tp_cum[group_ends] / float(pos)
    fpr = fp_cum[group_ends] / float(neg)
    fpr = torch.cat([torch.zeros(1, dtype=torch.float32, device=preds.device), fpr.to(torch.float32)])
    tpr = torch.cat([torch.zeros(1, dtype=torch.float32, device=preds.device), tpr.to(torch.float32)])
    return fpr, tpr


def _tpr_below_fpr(fpr: torch.Tensor, tpr: torch.Tensor, threshold: float) -> float:
    count = int((fpr <= float(threshold)).sum().item())
    if count <= 0:
        return 0.0
    return float(tpr[count - 1].item())


def compute_mia_metrics_reference_style(
    member_scores: torch.Tensor,
    nonmember_scores: torch.Tensor,
    fpr_thresholds: np.ndarray,
    fpr_eval: Sequence[float],
    score_direction: str,
) -> Dict[str, Any]:
    """
    Match the ROC/AUC and TPR@FPR procedure used in copyright-project.

    member_scores/nonmember_scores are shaped [num_attack_rows, num_samples].
    """
    if member_scores.ndim != 2 or nonmember_scores.ndim != 2:
        raise ValueError("member_scores and nonmember_scores must be rank-2 tensors")
    if member_scores.shape[0] != nonmember_scores.shape[0]:
        raise ValueError("member and nonmember scores must have the same row count")

    num_rows = member_scores.shape[0]
    if num_rows == 0 or member_scores.shape[1] == 0 or nonmember_scores.shape[1] == 0:
        return {
            "tpr_fpr_curve": [0.0 for _ in fpr_thresholds],
            "auc": 0.5,
            **{f"tpr@fpr={float(fpr):.2f}": 0.0 for fpr in fpr_eval},
        }

    all_curves: List[List[float]] = []
    all_auc: List[float] = []
    tpr_at_fpr_rows: Dict[float, List[float]] = {float(fpr): [] for fpr in fpr_eval}

    for row_idx in range(num_rows):
        member_row = member_scores[row_idx].to(torch.float32)
        nonmember_row = nonmember_scores[row_idx].to(torch.float32)

        if score_direction == "lower_is_member":
            # Use rank-preserving orientation without denominator scaling.
            pred_nonmember = -nonmember_row
            pred_member = -member_row
        elif score_direction == "higher_is_member":
            pred_nonmember = nonmember_row
            pred_member = member_row
        else:
            raise ValueError(f"Unsupported direction: {score_direction}")

        preds = torch.cat([pred_nonmember, pred_member], dim=0)
        labels = torch.cat(
            [
                torch.zeros(nonmember_row.shape[0], dtype=torch.long, device=preds.device),
                torch.ones(member_row.shape[0], dtype=torch.long, device=preds.device),
            ],
            dim=0,
        )

        fpr, tpr = _binary_roc_curve(preds, labels)
        for fpr_target in tpr_at_fpr_rows:
            tpr_at_fpr_rows[fpr_target].append(_tpr_below_fpr(fpr, tpr, fpr_target))

        curve = [_tpr_below_fpr(fpr, tpr, float(threshold)) for threshold in fpr_thresholds]
        all_curves.append(curve)
        all_auc.append(float(np.trapezoid(curve, fpr_thresholds)))

    best_idx = int(np.argmax(np.asarray(all_auc, dtype=np.float64)))
    return {
        "tpr_fpr_curve": all_curves[best_idx],
        "auc": float(all_auc[best_idx]),
        **{f"tpr@fpr={fpr:.2f}": float(max(values) if values else 0.0) for fpr, values in tpr_at_fpr_rows.items()},
    }

