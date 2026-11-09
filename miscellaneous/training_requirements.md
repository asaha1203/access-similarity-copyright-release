## Copyright Paper Training Requirements (Detailed Reference)

This document is the durable reference for preparing data and running LoRA / DP-LoRA training (plus diffusion full-UNet fine-tuning) for the copyright experiments.

## 0) Scope and Non-Goals

### In scope

- Deterministic dataset acquisition and local partitioning.
- LoRA and DP-LoRA adapter training for text and diffusion.
- Full-UNet fine-tuning for diffusion models (non-DP).
- Reproducible run metadata and artifacts for later evaluation.
- Optional Weights & Biases logging.

### Out of scope (for now)

- Inference-time CP-k or CP-fusion.
- Membership inference implementations and MI metrics.
- Reconstruction attacks or utility benchmarks.
- Final model comparison plots/tables.

## 1) Environment and Execution Rules

- All install/run commands must use:
  - `${CP_EXPERIMENTS_PYTHON}`
- Never rely on implicit `python` or `pip` from another environment.
- Prefer explicit HF cache roots under project storage during data prep to reduce cross-project cache collisions.

Recommended bootstrap:

```bash
${CP_EXPERIMENTS_PYTHON} -m pip install -r requirements.txt
```

## 2) Canonical Model + Dataset Grid

| Modality | Base model family | Dataset(s) |
| --- | --- | --- |
| Language | Llama2-7B-compatible Causal LM | `math-ai/AutoMathText`, `euclaise/writingprompts` |
| Diffusion | SD v1.4-compatible text-to-image pipeline | `diffusers/pokemon-gpt4-captions`, `antoniaaa/laion_mi` |

Notes:

- Text pipeline is adapter fine-tuning only (base weights frozen except target LoRA modules).
- Diffusion pipeline supports UNet-attention LoRA and full-UNet fine-tuning (text encoder optional, VAE frozen).

## 3) Data Sources, Semantics, and Split Contracts

## 3.1 Global deterministic controls

```yaml
split_seed: 1337
```

Determinism policy:

- Shuffle with the fixed seed before selecting indices.
- Persist explicit index lists for every split.
- Persist split fingerprints (hash of ordered indices + source metadata).
- Hard-fail on fingerprint mismatch unless user explicitly requests overwrite.

## 3.2 Required split policy

| Dataset | HF source split | Materialized rows | Train full | Shard 1 | Shard 2 | Reserved MI not-in-training set |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `diffusers/pokemon-gpt4-captions` | `train` | 833 | 600 | 300 | 300 | 233 |
| `antoniaaa/laion_mi` | `nonmembers` | 6000 | 1000 | 500 | 500 | 5000 |
| `math-ai/AutoMathText` | configured train source | 6000 | 1000 | 500 | 500 | 5000 |
| `euclaise/writingprompts` | merged `train+test`, deterministic split | 6000 | 1000 | 500 | 500 | 5000 |

Hard constraints:

- Pokemon must be exactly `300/300/233`.
- LAION-MI train and reserved pools both come from `nonmembers` and must be disjoint.
- For first implementation phase, ignore LAION `members` split.

## 3.3 MI counterfactual pool definitions to lock now

- Pokemon MI counterfactual pool: the exact 233 rows not used by Pokemon training full set.
- LAION-MI MI counterfactual pool: the exact 5000 nonmember rows excluded from LAION nonmember training full set.
- Text MI counterfactual pools: the exact 5000 rows excluded from each text dataset's training full set.

These are the pools later MI code must draw from.

## 3.4 Local persistence requirements

Per dataset:

```text
data/raw/<dataset>/materialized_dataset/          # HF data saved locally
data/processed/<dataset>/train_full/
data/processed/<dataset>/shard1/
data/processed/<dataset>/shard2/
data/processed/<dataset>/mi_reserve_not_in_training/   # image datasets now
data/processed/<dataset>/split_manifest.json
data/processed/<dataset>/split_fingerprint.json
```

Global registry:

```text
data/processed/dataset_registry.json
```

## 3.5 LAION-MI practical note

Depending on HF mirror behavior, `laion_mi` may expose URL+caption records rather than already-hosted image tensors. In that case:

- Download images from URL deterministically in index order.
- Skip irretrievable URLs and continue until target cardinality is reached.
- Persist resulting local image-backed dataset and manifest.

## 4) LoRA and DP Requirements

## 4.1 LoRA policy

- Default: LoRA (not qLoRA).
- No 4-bit/8-bit quantization unless explicitly requested later.

Target defaults:

```yaml
language_lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  train_bias: false
  quantization: none
```

```yaml
diffusion_lora:
  r: 16
  alpha: 16
  dropout: 0.0
  target: unet_attention_only
  train_text_encoder: false
  train_vae: false
  train_unet_base_weights: false
  quantization: none
```

Training must use models exposing the canonical `q_proj/k_proj/v_proj/o_proj` target modules unless this requirements file is intentionally revised.

## 4.2 DP policy

```yaml
dp:
  optimizer: dp_adam
  target_epsilons: [10, 20, 50]
  delta: 1.0e-5
  clip_norm: 1.0
  clipping_unit: per_example
  noise_addition: once_per_logical_batch
  trainable_params_only: true
```

Requirements:

- DP acts only on trainable adapter parameters.
- Add noise once per logical batch (not once per microbatch).
- Persist a `privacy_report.json` for all DP runs.

## 5) Training Matrix and Run Semantics

For each dataset:

| Run type | Split | DP |
| --- | --- | --- |
| `baseline_full_lora` | `train_full` | no |
| `shard1_lora` | `shard1` | no |
| `shard2_lora` | `shard2` | no |
| `dp_full_lora_eps10` | `train_full` | yes |
| `dp_full_lora_eps20` | `train_full` | yes |
| `dp_full_lora_eps50` | `train_full` | yes |

Phase-1 exclusion:

- Do not train DP shard models yet.

## 6) W&B Logging Contract

All training entrypoints must support:

```yaml
wandb:
  enabled: false
  project: copyright-paper
  entity: null
  run_name: null
  tags: []
```

Behavior:

- `enabled=false`: local logs only.
- `enabled=true`: W&B + local logs in parallel.
- W&B failures/auth issues should not break local run logging if avoidable.
- Training logs 10 fixed generation prompts by default at the configured generation cadence.
- Generated text/image samples must also be logged under step-specific W&B keys so older samples remain visible after later generations.

## 7) Run Artifacts (Compatibility-Critical)

Every run must write:

```text
adapter_model.safetensors            # LoRA runs
adapter_config.json                  # LoRA runs
unet/config.json                     # full-UNet runs
unet/diffusion_pytorch_model.safetensors  # full-UNet runs
training_config.yaml
split_manifest.json
split_fingerprint.json
train_log.jsonl
final_metrics.json
privacy_report.json                  # DP runs only
```

Rationale:

- Future evaluation should never have to infer which split was used.
- Split + fingerprint files are treated as immutable provenance records.

## 8) Command Reference

Dataset prep:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/prepare_datasets.py --split-seed 1337
```

Text training:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/train_text.py --config <config.yaml>
```

Diffusion training:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/train_diffusion.py --config <config.yaml>
```

## 9) Required Sanity Checks Before Full Matrix

1. LoRA-only trainability is enforced (no accidental base-weight finetune).
2. `train_full == shard1 union shard2` for each dataset.
3. `shard1` and `shard2` are disjoint for each dataset.
4. Pokemon split cardinalities are exactly `600/300/300/233` (full/shard1/shard2/reserve).
5. LAION-MI nonmember split cardinalities are exactly `1000/500/500/5000`.
6. LAION MI reserve is disjoint from LAION training full.
7. DP noise/clipping path runs only on trainable parameters.
8. Privacy report is emitted for every DP run.
9. W&B toggle works in both disabled and enabled modes.
10. Run artifact set is complete and includes split provenance files.

## 10) Future-phase Hand-off Notes

When MI implementation begins:

- Reuse only `mi_reserve_not_in_training` pools recorded here.
- Do not regenerate splits unless intentionally version-bumping fingerprints.
- Any split-policy change must update this requirements file and create new fingerprints.
