import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .constants import PROCESSED_DATA_ROOT, RUNS_ROOT
from .io_utils import ensure_dir, resolve_run_name, write_json, write_yaml
from .logging_utils import RunLogger


@dataclass
class TextRunConfig:
    run_name: str
    dataset_key: str
    split_name: str
    model_name_or_path: str
    output_subdir: str = "text"
    max_seq_len: int = 512
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    learning_rate: float = 3e-4
    mixed_precision: str = "fp16"
    # Deprecated as a training-length control knob; training is epoch-driven.
    # Kept for backward compatibility with existing YAMLs.
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    dp_enabled: bool = False
    target_epsilon: float = 10.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    dp_noise_multiplier: Optional[float] = None
    dp_accountant: str = "rdp"
    dp_secure_mode: bool = False
    dp_grad_sample_mode: str = "ghost"
    dp_poisson_sampling: bool = True
    dp_max_physical_batch_size: Optional[int] = None
    logging_steps: int = 1
    save_steps: int = 1000
    test_generation_steps: int = 500
    test_generation_num_prompts: int = 10
    test_generation_max_prompt_tokens: int = 128
    test_generation_prompts: Optional[List[str]] = None
    test_generation: Optional[Dict[str, Any]] = None
    seed: int = 1337
    use_cpu: bool = False
    wandb: Dict = None


def _resolve_precision(cfg: TextRunConfig):
    device = "cpu" if cfg.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device == "cuda" and cfg.mixed_precision == "fp16" and not cfg.dp_enabled
    use_bf16 = device == "cuda" and cfg.mixed_precision == "bf16" and not cfg.dp_enabled
    model_dtype = None
    if use_fp16:
        model_dtype = torch.float16
    elif use_bf16:
        model_dtype = torch.bfloat16
    return device, use_fp16, use_bf16, model_dtype


def _load_lora_model(cfg: TextRunConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    _, _, _, model_dtype = _resolve_precision(cfg)
    from_pretrained_kwargs = {}
    if model_dtype is not None:
        from_pretrained_kwargs["torch_dtype"] = model_dtype
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **from_pretrained_kwargs)
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model  # pylint: disable=import-outside-toplevel

    preferred_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    available_names = [name for name, _ in model.named_modules()]
    if not any(t in n for n in available_names for t in preferred_targets):
        raise ValueError(
            "Language LoRA target modules q_proj/k_proj/v_proj/o_proj were not found. "
            "Use a Llama2-compatible causal LM or update the training requirements intentionally."
        )

    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=preferred_targets,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def _tokenize(dataset, tokenizer, max_seq_len: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    return dataset.map(tok, batched=True, remove_columns=dataset.column_names)


def _text_generation_cfg(cfg: TextRunConfig) -> Dict[str, Any]:
    generation_cfg = dict(cfg.test_generation or {})
    generation_cfg.setdefault("max_new_tokens", 128)
    generation_cfg.setdefault("temperature", 0.8)
    generation_cfg.setdefault("top_p", 0.95)
    generation_cfg.setdefault("top_k", 50)
    generation_cfg.setdefault("do_sample", True)
    return generation_cfg


def _select_generation_prompts(dataset, cfg: TextRunConfig) -> List[str]:
    if cfg.test_generation_prompts:
        return [prompt for prompt in cfg.test_generation_prompts if prompt]

    prompts: List[str] = []
    if "text" not in dataset.column_names:
        return prompts
    for row in dataset.select(range(min(len(dataset), max(cfg.test_generation_num_prompts * 4, 1)))):
        text = str(row.get("text", "")).strip()
        if text:
            prompts.append(text)
        if len(prompts) >= cfg.test_generation_num_prompts:
            break
    return prompts


def _should_log_generation(step: int, cfg: TextRunConfig) -> bool:
    if cfg.test_generation_steps <= 0 or cfg.test_generation_num_prompts <= 0:
        return False
    return step == 1 or step % cfg.test_generation_steps == 0


def _generate_short_text_samples(
    model,
    tokenizer,
    prompts: Sequence[str],
    cfg: TextRunConfig,
    step: int,
) -> List[Dict[str, Any]]:
    if not prompts:
        return []
    generation_cfg = _text_generation_cfg(cfg)
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 48))
    temperature = float(generation_cfg.get("temperature", 0.8))
    top_p = float(generation_cfg.get("top_p", 0.95))
    top_k = int(generation_cfg.get("top_k", 50))
    do_sample = bool(generation_cfg.get("do_sample", True))

    target_model = model._module if hasattr(model, "_module") else model
    was_training = target_model.training
    target_model.eval()
    device = next(target_model.parameters()).device
    records: List[Dict[str, Any]] = []
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(cfg.seed + step)
        for prompt_idx, prompt in enumerate(prompts[: cfg.test_generation_num_prompts]):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.test_generation_max_prompt_tokens,
            ).to(device)
            with torch.inference_mode():
                generated = target_model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_ids = generated[0, enc["input_ids"].shape[1] :]
            records.append(
                {
                    "step": step,
                    "prompt_id": prompt_idx,
                    "prompt": tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True),
                    "generated_text": tokenizer.decode(gen_ids, skip_special_tokens=True),
                    "decode_strategy": "sample" if do_sample else "greedy",
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "eos_reached": bool(tokenizer.eos_token_id in set(gen_ids.detach().cpu().tolist())),
                }
            )
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            target_model.train()
    return records


def _log_text_generations_if_due(
    model,
    tokenizer,
    prompts: Sequence[str],
    cfg: TextRunConfig,
    logger: RunLogger,
    step: int,
) -> None:
    if not _should_log_generation(step, cfg):
        return
    records = _generate_short_text_samples(model, tokenizer, prompts, cfg, step)
    logger.log_text_generations(records, step=step)


class TextGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts: Sequence[str], cfg: TextRunConfig, logger: RunLogger):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.cfg = cfg
        self.logger = logger

    def on_step_end(self, args, state, control, model=None, **kwargs):  # pylint: disable=unused-argument
        if model is not None:
            _log_text_generations_if_due(model, self.tokenizer, self.prompts, self.cfg, self.logger, state.global_step)
        return control


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    return value


class TextTrainingLogCallback(TrainerCallback):
    def __init__(self, run_dir: Path, logger: RunLogger):
        self.log_path = run_dir / "train_log.jsonl"
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):  # pylint: disable=unused-argument
        if not logs:
            return control
        record = {"step": state.global_step}
        record.update({key: _json_safe(value) for key, value in logs.items()})
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.logger.log(record, step=state.global_step)
        return control


def _save_common_artifacts(run_dir: Path, cfg: TextRunConfig, manifest_path: Path, fingerprint_path: Path) -> None:
    write_yaml(run_dir / "training_config.yaml", asdict(cfg))
    write_json(run_dir / "final_metrics.json", {})
    write_json(run_dir / "split_manifest.json", json.loads(manifest_path.read_text(encoding="utf-8")))
    write_json(run_dir / "split_fingerprint.json", json.loads(fingerprint_path.read_text(encoding="utf-8")))


def _training_args_for_text(cfg: TextRunConfig, run_dir: Path) -> TrainingArguments:
    _, use_fp16, use_bf16, _ = _resolve_precision(cfg)
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to=[],
        remove_unused_columns=False,
        seed=cfg.seed,
        use_cpu=cfg.use_cpu,
        fp16=use_fp16,
        bf16=use_bf16,
    )
    return args


def _effective_physical_batch_size(logical_batch_size: int, configured_physical_batch_size: Optional[int]) -> int:
    if configured_physical_batch_size is None:
        return logical_batch_size
    if configured_physical_batch_size < 1:
        raise ValueError("dp_max_physical_batch_size must be >= 1 when provided.")
    return configured_physical_batch_size


class _TupleTensorDataset(TorchDataset):
    """Adapter so Opacus can infer tensor dtypes/shapes for Poisson empty batches."""

    def __init__(self, base_dataset, feature_keys: Sequence[str]):
        self.base_dataset = base_dataset
        self.feature_keys = list(feature_keys)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        return tuple(torch.as_tensor(sample[key], dtype=torch.long) for key in self.feature_keys)


def _normalize_private_batch(batch: Any, feature_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    """
    Normalize Opacus/loader outputs to a dict batch.

    With Poisson sampling, Opacus can emit empty batches as a list of tensors;
    the training loop expects dict batches keyed like HuggingFace collators.
    """

    if isinstance(batch, Mapping):
        return dict(batch)
    if isinstance(batch, (list, tuple)):
        normalized: Dict[str, torch.Tensor] = {}
        for key, value in zip(feature_keys, batch):
            if isinstance(value, torch.Tensor):
                normalized[key] = value
        return normalized
    return {}


def _is_empty_batch(batch: Dict[str, torch.Tensor]) -> bool:
    if not batch:
        return True
    first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
    if first_tensor is None:
        return True
    return first_tensor.ndim > 0 and first_tensor.shape[0] == 0


def _resolve_sample_rate(optimizer, dataset_size: int) -> Optional[float]:
    sample_rate = getattr(optimizer, "sample_rate", None)
    if sample_rate is not None:
        return float(sample_rate)
    expected_batch_size = getattr(optimizer, "expected_batch_size", None)
    if expected_batch_size is None or dataset_size <= 0:
        return None
    return float(expected_batch_size) / float(dataset_size)


def _run_nonprivate(
    cfg: TextRunConfig,
    run_dir: Path,
    dataset,
    model,
    tokenizer,
    logger: RunLogger,
    max_steps_override: Optional[int] = None,
):
    train_ds = _tokenize(dataset, tokenizer, cfg.max_seq_len)
    generation_prompts = _select_generation_prompts(dataset, cfg)
    args = _training_args_for_text(cfg, run_dir)
    if max_steps_override is not None:
        args.max_steps = max_steps_override
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            TextTrainingLogCallback(run_dir, logger),
            TextGenerationCallback(tokenizer, generation_prompts, cfg, logger),
        ],
    )
    result = trainer.train()
    metrics = result.metrics
    write_json(run_dir / "final_metrics.json", metrics)
    logger.log(metrics)
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))


def _run_private_opacus(
    cfg: TextRunConfig,
    run_dir: Path,
    dataset,
    model,
    tokenizer,
    logger: RunLogger,
    max_steps_override: Optional[int] = None,
):
    if cfg.gradient_accumulation_steps != 1:
        raise ValueError("Opacus backend currently requires gradient_accumulation_steps=1 for correct accounting.")

    tokenized = _tokenize(dataset, tokenizer, cfg.max_seq_len)
    dataset_size = len(tokenized)
    effective_dp_grad_sample_mode = cfg.dp_grad_sample_mode
    effective_physical_batch_size = _effective_physical_batch_size(
        cfg.per_device_train_batch_size,
        cfg.dp_max_physical_batch_size,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    feature_keys = [key for key in ("input_ids", "attention_mask", "token_type_ids") if key in tokenized.column_names]
    tuple_dataset = _TupleTensorDataset(tokenized, feature_keys)

    def tuple_collator(batch_rows):
        feature_rows = [{key: value for key, value in zip(feature_keys, row)} for row in batch_rows]
        return collator(feature_rows)

    loader = DataLoader(
        tuple_dataset,
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=tuple_collator,
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    device = "cpu" if cfg.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    import torch.nn as nn
    from opacus import PrivacyEngine  # pylint: disable=import-outside-toplevel
    from opacus.utils.batch_memory_manager import BatchMemoryManager  # pylint: disable=import-outside-toplevel

    privacy_engine = PrivacyEngine(accountant=cfg.dp_accountant, secure_mode=cfg.dp_secure_mode)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    fallback_reason: Optional[str] = None
    while True:
        try:
            if cfg.dp_noise_multiplier is not None:
                private_output = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    data_loader=loader,
                    noise_multiplier=cfg.dp_noise_multiplier,
                    max_grad_norm=cfg.max_grad_norm,
                    grad_sample_mode=effective_dp_grad_sample_mode,
                    poisson_sampling=cfg.dp_poisson_sampling,
                )
            else:
                private_output = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    data_loader=loader,
                    epochs=cfg.num_train_epochs,
                    target_epsilon=cfg.target_epsilon,
                    target_delta=cfg.delta,
                    max_grad_norm=cfg.max_grad_norm,
                    grad_sample_mode=effective_dp_grad_sample_mode,
                    poisson_sampling=cfg.dp_poisson_sampling,
                )
            break
        except Exception as exc:
            if effective_dp_grad_sample_mode != "ghost":
                raise
            effective_dp_grad_sample_mode = "hooks"
            fallback_reason = str(exc)
    if len(private_output) == 4:
        model, optimizer, criterion, loader = private_output
    else:
        model, optimizer, loader = private_output
        criterion = None

    accountant = getattr(privacy_engine, "accountant", None)
    privacy_stats: Dict[str, Any] = {
        "backend": "opacus",
        "engine_class": privacy_engine.__class__.__name__,
        "accountant": cfg.dp_accountant,
        "accountant_class": accountant.__class__.__name__ if accountant is not None else None,
        "grad_sample_mode": effective_dp_grad_sample_mode,
        "configured_grad_sample_mode": cfg.dp_grad_sample_mode,
        "delta": cfg.delta,
        "target_epsilon": cfg.target_epsilon,
        "noise_multiplier": getattr(optimizer, "noise_multiplier", cfg.dp_noise_multiplier),
        "max_grad_norm": getattr(optimizer, "max_grad_norm", cfg.max_grad_norm),
        "expected_batch_size": getattr(optimizer, "expected_batch_size", None),
        "sample_rate": _resolve_sample_rate(optimizer, dataset_size),
        "loader_batch_size": getattr(loader, "batch_size", None),
        "loader_num_batches": len(loader),
        "dataset_size": dataset_size,
        "poisson_sampling": cfg.dp_poisson_sampling,
        "effective_physical_batch_size": effective_physical_batch_size,
        "opacus_version": version("opacus"),
    }
    if fallback_reason is not None:
        privacy_stats["grad_sample_mode_fallback_reason"] = fallback_reason
    history = getattr(accountant, "history", None)
    if history is not None:
        privacy_stats["accountant_history_len"] = len(history)
        if len(history) > 0:
            privacy_stats["accountant_last_event"] = str(history[-1])
    try:
        privacy_stats["epsilon_at_start"] = float(privacy_engine.get_epsilon(cfg.delta))
    except Exception:
        privacy_stats["epsilon_at_start"] = None

    write_json(
        run_dir / "privacy_report_initial.json",
        {
            "phase": "start",
            "backend": "opacus",
            "accountant": cfg.dp_accountant,
            "grad_sample_mode": effective_dp_grad_sample_mode,
            "configured_grad_sample_mode": cfg.dp_grad_sample_mode,
            "epsilon": privacy_stats["epsilon_at_start"],
            "target_epsilon": cfg.target_epsilon,
            "delta": cfg.delta,
            "max_grad_norm": cfg.max_grad_norm,
            "noise_multiplier": getattr(optimizer, "noise_multiplier", cfg.dp_noise_multiplier),
            "expected_batch_size": getattr(optimizer, "expected_batch_size", None),
            "sample_rate": _resolve_sample_rate(optimizer, dataset_size),
            "loader_num_batches": len(loader),
            "loader_batch_size": getattr(loader, "batch_size", None),
            "dataset_size": dataset_size,
            "poisson_sampling": cfg.dp_poisson_sampling,
            "effective_physical_batch_size": effective_physical_batch_size,
            "opacus_version": version("opacus"),
            "empty_batches_skipped": 0,
        },
    )

    logger.log_run_setup(config=asdict(cfg), privacy_stats=privacy_stats, step=0)

    model.train()
    generation_prompts = _select_generation_prompts(dataset, cfg)
    logs: List[Dict] = []
    step = 0
    empty_batches_skipped = 0
    memory_ctx = BatchMemoryManager(
        data_loader=loader,
        max_physical_batch_size=effective_physical_batch_size,
        optimizer=optimizer,
    )
    with memory_ctx as train_loader:
        progress_total = (
            int(max_steps_override)
            if max_steps_override is not None
            else cfg.num_train_epochs * len(train_loader)
        )
        with tqdm(total=progress_total, desc="Text training", unit="step") as progress:
            for epoch in range(cfg.num_train_epochs):
                progress.set_description(f"Text training (epoch {epoch + 1}/{cfg.num_train_epochs})")
                for batch in train_loader:
                    batch = _normalize_private_batch(batch, feature_keys)
                    if max_steps_override is not None and step >= max_steps_override:
                        break
                    if _is_empty_batch(batch):
                        empty_batches_skipped += 1
                        optimizer.zero_grad(set_to_none=True)
                        if max_steps_override is None:
                            progress.update(1)
                        continue
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if criterion is not None:
                        out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))
                        shift_logits = out.logits[..., :-1, :].contiguous()
                        shift_labels = batch["labels"][..., 1:].contiguous()
                        loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            shape=shift_logits.shape,
                        )
                        loss.backward()
                    else:
                        out = model(**batch)
                        loss = out.loss
                        loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    rec = {"step": step, "loss": float(loss.item() if hasattr(loss, "item") else loss.detach().cpu().item())}
                    logs.append(rec)
                    logger.log(rec, step=step)
                    step += 1
                    _log_text_generations_if_due(model, tokenizer, generation_prompts, cfg, logger, step)
                    if cfg.save_steps > 0 and step % cfg.save_steps == 0:
                        checkpoint_dir = run_dir / f"checkpoint-{step}"
                        to_checkpoint = model._module if hasattr(model, "_module") else model
                        to_checkpoint.save_pretrained(str(checkpoint_dir))
                        tokenizer.save_pretrained(str(checkpoint_dir))
                        torch.save(
                            {
                                "step": step,
                                "epoch": epoch + 1,
                                "optimizer": optimizer.state_dict(),
                                "torch_rng_state": torch.get_rng_state(),
                                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            },
                            checkpoint_dir / "training_state.pt",
                        )
                    progress.update(1)
                    progress.set_postfix(loss=f"{rec['loss']:.4f}", refresh=False)
                if max_steps_override is not None and step >= max_steps_override:
                    break

    epsilon = privacy_engine.get_epsilon(cfg.delta)
    write_json(
        run_dir / "privacy_report.json",
        {
            "phase": "final",
            "backend": "opacus",
            "accountant": cfg.dp_accountant,
            "grad_sample_mode": effective_dp_grad_sample_mode,
            "configured_grad_sample_mode": cfg.dp_grad_sample_mode,
            "epsilon": epsilon,
            "target_epsilon": cfg.target_epsilon,
            "delta": cfg.delta,
            "max_grad_norm": cfg.max_grad_norm,
            "noise_multiplier": getattr(optimizer, "noise_multiplier", cfg.dp_noise_multiplier),
            "expected_batch_size": getattr(optimizer, "expected_batch_size", None),
            "sample_rate": _resolve_sample_rate(optimizer, dataset_size),
            "loader_num_batches": len(loader),
            "loader_batch_size": getattr(loader, "batch_size", None),
            "dataset_size": dataset_size,
            "poisson_sampling": cfg.dp_poisson_sampling,
            "effective_physical_batch_size": effective_physical_batch_size,
            "opacus_version": version("opacus"),
            "empty_batches_skipped": empty_batches_skipped,
        },
    )
    write_json(
        run_dir / "final_metrics.json",
        {"train_steps": step, "last_loss": logs[-1]["loss"] if logs else None, "epsilon": epsilon},
    )
    with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    to_save = model._module if hasattr(model, "_module") else model
    to_save.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))


def _run_private(cfg: TextRunConfig, run_dir: Path, dataset, model, tokenizer, logger: RunLogger):
    _run_private_opacus(cfg, run_dir, dataset, model, tokenizer, logger)


def run_text_training(cfg: TextRunConfig, max_steps_override: Optional[int] = None) -> Path:
    resolved_name = resolve_run_name(cfg.run_name)
    run_dir = RUNS_ROOT / cfg.output_subdir / resolved_name
    ensure_dir(run_dir)
    manifest_path = PROCESSED_DATA_ROOT / cfg.dataset_key / "split_manifest.json"
    fingerprint_path = PROCESSED_DATA_ROOT / cfg.dataset_key / "split_fingerprint.json"
    split_path = PROCESSED_DATA_ROOT / cfg.dataset_key / cfg.split_name
    dataset = load_from_disk(str(split_path))
    model, tokenizer = _load_lora_model(cfg)

    logger = RunLogger(run_dir=run_dir, wandb_cfg=cfg.wandb or {}, run_name=resolved_name)
    logger.start()
    _save_common_artifacts(run_dir, cfg, manifest_path, fingerprint_path)
    if cfg.dp_enabled:
        _run_private_opacus(cfg, run_dir, dataset, model, tokenizer, logger, max_steps_override=max_steps_override)
    else:
        logger.log_run_setup(config=asdict(cfg), step=0)
        _run_nonprivate(cfg, run_dir, dataset, model, tokenizer, logger, max_steps_override=max_steps_override)
    logger.finish()
    return run_dir

