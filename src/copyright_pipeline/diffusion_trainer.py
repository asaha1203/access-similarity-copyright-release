import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from .constants import PROCESSED_DATA_ROOT, RUNS_ROOT
from .io_utils import ensure_dir, resolve_run_name, write_json, write_yaml
from .logging_utils import RunLogger


@dataclass
class DiffusionRunConfig:
    run_name: str
    dataset_key: str
    split_name: str
    model_name_or_path: str
    output_subdir: str = "image"
    resolution: int = 512
    num_train_epochs: int = 1
    train_batch_size: int = 1
    train_num_workers: int = 4
    train_pin_memory: bool = True
    learning_rate: float = 1e-4
    train_text_encoder: bool = False
    text_encoder_learning_rate: float = 1e-5
    mixed_precision: str = "fp16"
    # Deprecated as a training-length control knob; training is epoch-driven.
    # Kept for backward compatibility with existing YAMLs.
    max_steps: Optional[int] = None
    finetune_mode: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
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
    save_steps: int = 1000
    test_generation_steps: int = 500
    test_generation_num_prompts: int = 10
    test_generation_prompts: Optional[List[str]] = None
    test_generation_num_inference_steps: int = 30
    test_generation_guidance_scale: float = 7.5
    test_generation_negative_prompt: Optional[str] = None
    seed: int = 1337
    use_cpu: bool = False
    wandb: Dict = None


def _build_components(cfg: DiffusionRunConfig):
    from diffusers import DDPMScheduler, StableDiffusionPipeline  # pylint: disable=import-outside-toplevel

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_name_or_path,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    if cfg.finetune_mode not in {"lora", "full_unet"}:
        raise ValueError(f"Unsupported finetune_mode={cfg.finetune_mode!r}. Use 'lora' or 'full_unet'.")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(cfg.train_text_encoder)
    if cfg.finetune_mode == "lora":
        from peft import LoraConfig, get_peft_model  # pylint: disable=import-outside-toplevel

        unet.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_cfg)
    else:
        unet.requires_grad_(True)
    pipe.unet = unet
    return pipe, noise_scheduler, tokenizer, text_encoder, vae, unet


def _collate_fn(tokenizer, image_size: int):
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def _inner(batch):
        pixel_values = torch.stack([image_tf(x["image"].convert("RGB")) for x in batch])
        texts = [x.get("text", "") for x in batch]
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": pixel_values, "input_ids": tokens.input_ids}

    return _inner


def _save_common_artifacts(
    run_dir: Path,
    cfg: DiffusionRunConfig,
    manifest_path: Path,
    fingerprint_path: Path,
) -> None:
    write_yaml(run_dir / "training_config.yaml", asdict(cfg))
    write_json(run_dir / "final_metrics.json", {})
    write_json(run_dir / "split_manifest.json", json.loads(manifest_path.read_text(encoding="utf-8")))
    write_json(run_dir / "split_fingerprint.json", json.loads(fingerprint_path.read_text(encoding="utf-8")))


def _save_model_checkpoint(
    unet,
    output_dir: Path,
    finetune_mode: str,
    metadata: Optional[Dict] = None,
) -> None:
    ensure_dir(output_dir)
    to_save = unet._module if hasattr(unet, "_module") else unet
    if finetune_mode == "lora":
        to_save.save_pretrained(str(output_dir))
    elif finetune_mode == "full_unet":
        unet_dir = output_dir / "unet"
        ensure_dir(unet_dir)
        to_save.save_pretrained(str(unet_dir))
    else:
        raise ValueError(f"Unsupported finetune_mode={finetune_mode!r}.")
    if metadata is not None:
        write_json(output_dir / "checkpoint_metadata.json", metadata)


def _save_training_checkpoint(
    unet,
    optimizer,
    scaler,
    output_dir: Path,
    finetune_mode: str,
    metadata: Dict,
) -> None:
    _save_model_checkpoint(unet, output_dir, finetune_mode=finetune_mode, metadata=metadata)
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(training_state, output_dir / "training_state.pt")


def _select_generation_prompts(dataset, cfg: DiffusionRunConfig) -> List[str]:
    if cfg.test_generation_prompts:
        return [prompt for prompt in cfg.test_generation_prompts if prompt]

    prompts: List[str] = []
    for row in dataset.select(range(min(len(dataset), max(cfg.test_generation_num_prompts * 4, 1)))):
        text = str(row.get("text", "")).strip()
        if text:
            prompts.append(text)
        if len(prompts) >= cfg.test_generation_num_prompts:
            break
    return prompts


def _should_log_generation(step: int, cfg: DiffusionRunConfig) -> bool:
    if cfg.test_generation_steps <= 0 or cfg.test_generation_num_prompts <= 0:
        return False
    return step == 1 or step % cfg.test_generation_steps == 0


def _generate_image_samples(
    pipe,
    prompts: Sequence[str],
    cfg: DiffusionRunConfig,
    run_dir: Path,
    step: int,
    use_fp16: bool,
) -> List[Dict]:
    if not prompts:
        return []

    image_dir = run_dir / "generated_images" / f"step_{step:07d}"
    ensure_dir(image_dir)
    to_sample = prompts[: cfg.test_generation_num_prompts]
    target_unet = pipe.unet._module if hasattr(pipe.unet, "_module") else pipe.unet
    was_training = target_unet.training
    target_unet.eval()
    vae_device = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    text_device = next(pipe.text_encoder.parameters()).device
    text_dtype = next(pipe.text_encoder.parameters()).dtype
    records: List[Dict] = []
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        sample_device = next(target_unet.parameters()).device
        pipe.to(sample_device)
        pipe.set_progress_bar_config(disable=True)
        for prompt_idx, prompt in enumerate(to_sample):
            seed = cfg.seed + step * 1000 + prompt_idx
            device_str = str(sample_device)
            generator = torch.Generator(device=device_str).manual_seed(seed)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    output = pipe(
                        prompt=[prompt],
                        negative_prompt=[cfg.test_generation_negative_prompt]
                        if cfg.test_generation_negative_prompt
                        else None,
                        num_inference_steps=cfg.test_generation_num_inference_steps,
                        guidance_scale=cfg.test_generation_guidance_scale,
                        height=cfg.resolution,
                        width=cfg.resolution,
                        generator=generator,
                    )
            generated = output.images[0]
            image_path = image_dir / f"prompt_{prompt_idx:03d}.png"
            generated.save(image_path)
            records.append(
                {
                    "step": step,
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "num_inference_steps": cfg.test_generation_num_inference_steps,
                    "guidance_scale": cfg.test_generation_guidance_scale,
                    "seed": seed,
                    "image_path": str(image_path),
                }
            )
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        # Restore frozen-module placement after sampling; this is required for
        # DP full-UNet runs that intentionally keep VAE/text encoder on CPU.
        pipe.vae.to(device=vae_device, dtype=vae_dtype)
        pipe.text_encoder.to(device=text_device, dtype=text_dtype)
        if was_training:
            target_unet.train()
    return records


def _log_image_generations_if_due(
    pipe,
    prompts: Sequence[str],
    cfg: DiffusionRunConfig,
    run_dir: Path,
    logger: RunLogger,
    step: int,
    use_fp16: bool,
) -> None:
    if not _should_log_generation(step, cfg):
        return
    records = _generate_image_samples(pipe, prompts, cfg, run_dir, step, use_fp16=use_fp16)
    logger.log_image_generations(records, step=step)


def _privacy_init_stats(
    cfg: DiffusionRunConfig,
    privacy_engine,
    optimizer,
    loader,
    effective_dp_grad_sample_mode: str,
    dataset_size: int,
    poisson_sampling: bool,
    effective_physical_batch_size: int,
) -> Dict:
    accountant = getattr(privacy_engine, "accountant", None)
    stats = {
        "backend": "opacus",
        "engine_class": privacy_engine.__class__.__name__,
        "accountant": cfg.dp_accountant,
        "accountant_class": accountant.__class__.__name__ if accountant is not None else None,
        "grad_sample_mode": effective_dp_grad_sample_mode,
        "delta": cfg.delta,
        "target_epsilon": cfg.target_epsilon,
        "noise_multiplier": getattr(optimizer, "noise_multiplier", cfg.dp_noise_multiplier),
        "max_grad_norm": getattr(optimizer, "max_grad_norm", cfg.max_grad_norm),
        "expected_batch_size": getattr(optimizer, "expected_batch_size", None),
        "sample_rate": _resolve_sample_rate(optimizer, dataset_size),
        "loader_batch_size": getattr(loader, "batch_size", None),
        "loader_num_batches": len(loader),
        "dataset_size": dataset_size,
        "poisson_sampling": poisson_sampling,
        "effective_physical_batch_size": effective_physical_batch_size,
        "opacus_version": version("opacus"),
    }
    history = getattr(accountant, "history", None)
    if history is not None:
        stats["accountant_history_len"] = len(history)
        if len(history) > 0:
            stats["accountant_last_event"] = str(history[-1])
    try:
        stats["epsilon_at_start"] = float(privacy_engine.get_epsilon(cfg.delta))
    except Exception:
        stats["epsilon_at_start"] = None
    return stats


def _effective_physical_batch_size(logical_batch_size: int, configured_physical_batch_size: Optional[int]) -> int:
    if configured_physical_batch_size is None:
        return logical_batch_size
    if configured_physical_batch_size < 1:
        raise ValueError("dp_max_physical_batch_size must be >= 1 when provided.")
    return configured_physical_batch_size


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


def _train_loop(
    cfg: DiffusionRunConfig,
    run_dir: Path,
    dataset,
    pipe,
    noise_scheduler,
    text_encoder,
    vae,
    unet,
    tokenizer,
    logger: RunLogger,
    max_steps_override: Optional[int] = None,
):
    if cfg.dp_enabled and cfg.train_text_encoder:
        raise ValueError("Text encoder training is not supported when dp_enabled=true.")

    device = "cpu" if cfg.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device == "cuda" and cfg.mixed_precision == "fp16"
    amp_enabled = use_fp16 and not cfg.dp_enabled
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    frozen_dtype = torch.float32 if cfg.dp_enabled else (torch.float16 if use_fp16 else torch.float32)
    text_encoder_dtype = torch.float32 if cfg.train_text_encoder else frozen_dtype
    text_encoder.to(device=device, dtype=text_encoder_dtype)
    vae.to(device=device, dtype=frozen_dtype)
    unet.to(device)
    if cfg.train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.eval()
    vae.eval()
    unet.train()
    dataset_size = len(dataset)
    effective_physical_batch_size = _effective_physical_batch_size(
        cfg.train_batch_size,
        cfg.dp_max_physical_batch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.train_num_workers,
        pin_memory=cfg.train_pin_memory and device == "cuda",
        collate_fn=_collate_fn(tokenizer, cfg.resolution),
    )
    max_planned_steps = max_steps_override

    unet_trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer_param_groups = [{"params": unet_trainable_params, "lr": cfg.learning_rate}]
    if cfg.train_text_encoder:
        text_encoder_trainable_params = [p for p in text_encoder.parameters() if p.requires_grad]
        optimizer_param_groups.append(
            {"params": text_encoder_trainable_params, "lr": cfg.text_encoder_learning_rate}
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups)
    privacy_engine = None
    dp_criterion = None
    effective_dp_grad_sample_mode = cfg.dp_grad_sample_mode
    if cfg.dp_enabled:
        import torch.nn as nn
        from opacus import PrivacyEngine  # pylint: disable=import-outside-toplevel
        from opacus.utils.batch_memory_manager import BatchMemoryManager  # pylint: disable=import-outside-toplevel

        # Ghost clipping can be incompatible with LoRA PEFT wrappers in some
        # Opacus versions; keep hooks mode there, but allow ghost for full-UNet
        # where hooks mode is often prohibitively memory intensive.
        if effective_dp_grad_sample_mode == "ghost" and cfg.finetune_mode == "lora":
            effective_dp_grad_sample_mode = "hooks"
        privacy_engine = PrivacyEngine(accountant=cfg.dp_accountant, secure_mode=cfg.dp_secure_mode)
        if effective_dp_grad_sample_mode == "ghost":
            class _PerSampleMSELoss(nn.Module):
                reduction = "mean"

                def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                    per_elem = torch.nn.functional.mse_loss(pred, target, reduction="none")
                    return per_elem.reshape(pred.shape[0], -1).mean(dim=1)

            criterion = _PerSampleMSELoss()
        else:
            criterion = nn.MSELoss(reduction="mean")
        if cfg.dp_noise_multiplier is not None:
            private_output = privacy_engine.make_private(
                module=unet,
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
                module=unet,
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
        if len(private_output) == 4:
            unet, optimizer, dp_criterion, loader = private_output
        else:
            unet, optimizer, loader = private_output
        logger.log_run_setup(
            config=asdict(cfg),
            privacy_stats=_privacy_init_stats(
                cfg=cfg,
                privacy_engine=privacy_engine,
                optimizer=optimizer,
                loader=loader,
                effective_dp_grad_sample_mode=effective_dp_grad_sample_mode,
                dataset_size=dataset_size,
                poisson_sampling=cfg.dp_poisson_sampling,
                effective_physical_batch_size=effective_physical_batch_size,
            ),
            step=0,
        )
        epsilon_at_start = None
        try:
            epsilon_at_start = float(privacy_engine.get_epsilon(cfg.delta))
        except Exception:
            epsilon_at_start = None
        write_json(
            run_dir / "privacy_report_initial.json",
            {
                "phase": "start",
                "backend": "opacus",
                "accountant": cfg.dp_accountant,
                "grad_sample_mode": effective_dp_grad_sample_mode,
                "configured_grad_sample_mode": cfg.dp_grad_sample_mode,
                "epsilon": epsilon_at_start,
                "target_epsilon": cfg.target_epsilon,
                "delta": cfg.delta,
                "noise_multiplier": getattr(optimizer, "noise_multiplier", cfg.dp_noise_multiplier),
                "max_grad_norm": cfg.max_grad_norm,
                "expected_batch_size": getattr(optimizer, "expected_batch_size", None),
                "sample_rate": _resolve_sample_rate(optimizer, dataset_size),
                "loader_batch_size": getattr(loader, "batch_size", None),
                "loader_num_batches": len(loader),
                "dataset_size": dataset_size,
                "poisson_sampling": cfg.dp_poisson_sampling,
                "effective_physical_batch_size": effective_physical_batch_size,
                "opacus_version": version("opacus"),
                "empty_batches_skipped": 0,
            },
        )
        memory_ctx = BatchMemoryManager(
            data_loader=loader,
            max_physical_batch_size=effective_physical_batch_size,
            optimizer=optimizer,
        )
    else:
        logger.log_run_setup(config=asdict(cfg), step=0)
        memory_ctx = nullcontext(loader)

    # Keep pipeline modules aligned with training modules/device for interim sampling.
    pipe.unet = unet._module if hasattr(unet, "_module") else unet
    pipe.text_encoder = text_encoder
    pipe.vae = vae
    pipe.to(device)
    pipe.text_encoder.to(device=device, dtype=text_encoder_dtype)
    pipe.vae.to(device=device, dtype=frozen_dtype)

    logs: List[Dict] = []
    generation_prompts = _select_generation_prompts(dataset, cfg)
    step = 0
    empty_batches_skipped = 0
    with memory_ctx as train_loader:
        if max_planned_steps is not None:
            progress_total = max_planned_steps
        else:
            progress_total = cfg.num_train_epochs * len(train_loader)
        with tqdm(total=progress_total, desc="Diffusion training", unit="step") as progress:
            for epoch in range(cfg.num_train_epochs):
                progress.set_description(f"Diffusion training (epoch {epoch + 1}/{cfg.num_train_epochs})")
                for batch in train_loader:
                    if max_planned_steps is not None and step >= max_planned_steps:
                        break
                    if _is_empty_batch(batch):
                        empty_batches_skipped += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    pixel_values = batch["pixel_values"].to(
                        device,
                        dtype=frozen_dtype,
                        non_blocking=True,
                    )
                    input_ids = batch["input_ids"].to(device, non_blocking=True)

                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                    if cfg.train_text_encoder:
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                            encoder_hidden_states = text_encoder(input_ids)[0]
                    else:
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                                encoder_hidden_states = text_encoder(input_ids)[0]

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=latents.device,
                    ).long()
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        if dp_criterion is not None:
                            loss = dp_criterion(model_pred.float(), noise.float())
                            loss_for_log = torch.nn.functional.mse_loss(
                                model_pred.float(),
                                noise.float(),
                                reduction="mean",
                            )
                        else:
                            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                            loss_for_log = loss
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    rec = {"step": step, "loss": float(loss_for_log.detach().cpu().item())}
                    logs.append(rec)
                    logger.log(rec, step=step)
                    step += 1
                    _log_image_generations_if_due(
                        pipe,
                        generation_prompts,
                        cfg,
                        run_dir,
                        logger,
                        step,
                        use_fp16=use_fp16,
                    )

                    if cfg.save_steps > 0 and step % cfg.save_steps == 0:
                        _save_training_checkpoint(
                            unet,
                            optimizer,
                            scaler,
                            run_dir / f"checkpoint-{step}",
                            finetune_mode=cfg.finetune_mode,
                            metadata={
                                "step": step,
                                "loss": rec["loss"],
                                "epoch": epoch + 1,
                                "total_steps": max_planned_steps,
                                "save_steps": cfg.save_steps,
                                "checkpoint_type": f"diffusion_{cfg.finetune_mode}_training_state",
                                "finetune_mode": cfg.finetune_mode,
                            },
                        )

                    progress.update(1)
                    progress.set_postfix(loss=f"{rec['loss']:.4f}")
                if max_planned_steps is not None and step >= max_planned_steps:
                    break

    report = {"train_steps": step, "last_loss": logs[-1]["loss"] if logs else None}
    if cfg.dp_enabled:
        epsilon = privacy_engine.get_epsilon(cfg.delta) if privacy_engine is not None else cfg.target_epsilon
        report["phase"] = "final"
        report["backend"] = "opacus"
        report["accountant"] = cfg.dp_accountant
        report["grad_sample_mode"] = effective_dp_grad_sample_mode
        report["configured_grad_sample_mode"] = cfg.dp_grad_sample_mode
        report["epsilon"] = epsilon
        report["target_epsilon"] = cfg.target_epsilon
        report["delta"] = cfg.delta
        report["noise_multiplier"] = getattr(optimizer, "noise_multiplier", cfg.dp_noise_multiplier)
        report["max_grad_norm"] = cfg.max_grad_norm
        report["expected_batch_size"] = getattr(optimizer, "expected_batch_size", None)
        report["sample_rate"] = _resolve_sample_rate(optimizer, dataset_size)
        report["loader_batch_size"] = getattr(loader, "batch_size", None)
        report["loader_num_batches"] = len(loader)
        report["dataset_size"] = dataset_size
        report["poisson_sampling"] = cfg.dp_poisson_sampling
        report["effective_physical_batch_size"] = effective_physical_batch_size
        report["opacus_version"] = version("opacus")
        report["empty_batches_skipped"] = empty_batches_skipped
        write_json(run_dir / "privacy_report.json", report)

    write_json(run_dir / "final_metrics.json", report)
    with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    _save_training_checkpoint(
        unet,
        optimizer,
        scaler,
        run_dir,
        finetune_mode=cfg.finetune_mode,
        metadata={
            "step": step,
            "loss": logs[-1]["loss"] if logs else None,
            "total_steps": max_planned_steps,
            "save_steps": cfg.save_steps,
            "checkpoint_type": f"diffusion_{cfg.finetune_mode}_final_training_state",
            "finetune_mode": cfg.finetune_mode,
        },
    )


def run_diffusion_training(cfg: DiffusionRunConfig, max_steps_override: Optional[int] = None) -> Path:
    resolved_name = resolve_run_name(cfg.run_name)
    run_dir = RUNS_ROOT / cfg.output_subdir / resolved_name
    ensure_dir(run_dir)
    manifest_path = PROCESSED_DATA_ROOT / cfg.dataset_key / "split_manifest.json"
    fingerprint_path = PROCESSED_DATA_ROOT / cfg.dataset_key / "split_fingerprint.json"
    split_path = PROCESSED_DATA_ROOT / cfg.dataset_key / cfg.split_name
    dataset = load_from_disk(str(split_path))
    pipe, noise_scheduler, tokenizer, text_encoder, vae, unet = _build_components(cfg)

    wandb_cfg = dict(cfg.wandb or {})
    user_tags = list(wandb_cfg.get("tags") or [])
    auto_tags = ["diffusion", f"finetune_mode:{cfg.finetune_mode}"]
    merged_tags: List[str] = []
    for tag in user_tags + auto_tags:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)
    wandb_cfg["tags"] = merged_tags
    cfg.wandb = wandb_cfg
    logger = RunLogger(run_dir=run_dir, wandb_cfg=wandb_cfg, run_name=resolved_name)
    logger.start()
    _save_common_artifacts(run_dir, cfg, manifest_path, fingerprint_path)
    _train_loop(
        cfg,
        run_dir,
        dataset,
        pipe,
        noise_scheduler,
        text_encoder,
        vae,
        unet,
        tokenizer,
        logger,
        max_steps_override=max_steps_override,
    )
    logger.finish()
    return run_dir

