# The Access-Similarity Lens: An Operational Copyright Framework for Generative Models

This repository accompanies *The Access-Similarity Lens: An Operational Copyright Framework for Generative Models*. It includes utilities for:

- Deterministic dataset preparation for text and image modalities.
- Training entrypoints for text LoRA / DP-LoRA and diffusion fine-tuning (LoRA or full-UNet, including DP variants).
- Cache-first inference workflows for MIA, DRA, and CP-k post-scoring.
- Utility evaluation (heldout perplexity, reference perplexity, Prometheus-v2 fluency).
- Publication analysis scripts for final figures/tables.

The workflow is organized around three major algorithmic components, along with standard utilities for training and evaluating models. 

- **MIA via Proximal Initialization Attack:** score member vs nonmember examples.
- **DRA via Extraction and Ranking:** generate candidates and evaluate both membership scores and reconstruction distance.
- **CP-k:** compute log-probability ratios statistics across a draft model and shard models (`draft_p`, `shard_q1`, `shard_q2`), then filter/stratify analysis via acceptance rates for understanding and working with the **CP-k** from the Near-Access Freeness framework.

The repository is organized as follows:

- `src/copyright_pipeline/` includes the core training, scoring, cache schema, utility evaluation, and analysis logic.
- `scripts/`: includes CLI entrypoints for data prep, training, workflows, per-stage scoring, and figure/table generation.
- `configs/train/`: training configs (non-private and private, text and diffusion).
- `configs/inference/final_workflows/`: inference configs for each model, dataset, modality. 
- `tests/`: unit tests for cache workflows, distance metrics, token scoring optimizations, and helper utilities.

## Setup

Run from repo root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Prepare Deterministic Datasets

```bash
python scripts/prepare_datasets.py --split-seed 1337
```
<!-- 
Notes:

- Text rows are normalized to `prompt[SEP]target`.
- Splits are deterministic and fingerprinted.
- Use `--overwrite` to regenerate intentionally.
- Use `--datasets` to restrict preparation to selected datasets (`automathtext`, `writingprompts`, `pokemon`, `laion_mi`).

Generated artifacts include:

- `data/raw/<dataset>/materialized_dataset`
- `data/processed/<dataset>/{train_full,shard1,shard2,mi_reserve_not_in_training}`
- `data/processed/<dataset>/split_manifest.json`
- `data/processed/<dataset>/split_fingerprint.json`
- `data/processed/dataset_registry.json` -->

This command generates datasets for `automathtext`, `writingprompts`, `pokemon`, and `laion_mi` deterministically. Note that LAION-MI is downloaded from URLs, so this step may take a while (20+ minutes).

## 2) Train Models

Finetuning models is straightforward. Utilities are provided for LoRA and full training, but the primary setup here uses full diffusion finetuning (i.e., training U-Net) and LoRA finetuning for language models.

### Text (LoRA / DP-LoRA)

```bash
python scripts/train_text.py --config configs/train/text_nonprivate_final/automathtext_baseline_full_lora.yaml
python scripts/train_text.py --config configs/train/text_private_final/automathtext_dp_full_lora_eps10.yaml
```

### Diffusion (LoRA or full-UNet, private or non-private)

```bash
python scripts/train_diffusion.py --config configs/train/diffusion_nonprivate_final/pokemon_baseline_full_finetune.yaml
python scripts/train_diffusion.py --config configs/train/diffusion_private_final/pokemon_dp_full_finetune_eps10.yaml
```

### Dry runs

```bash
python scripts/train_text.py --config <config.yaml> --dry-run
python scripts/train_diffusion.py --config <config.yaml> --dry-run
```

## 3) Run Cache-First Inference Workflows

The following workflows are provided for each dataset and modality. They are designed to run end-to-end cache-first analysis pipelines.

### End-to-end workflows

```bash
python scripts/run_mia_workflow.py --config configs/inference/final_workflows/automathtext/MIA_CPK.yaml
python scripts/run_dra_workflow.py --config configs/inference/final_workflows/automathtext/DRA_CPK.yaml
```

Utility (from an existing DRA generation cache):

```bash
python scripts/run_utility_workflow.py \
  --config configs/inference/final_workflows/automathtext/DRA_CPK.yaml \
  --input outputs/inference/final_workflows/automathtext/dra_cpk/dra_generation_cache.pt
```

By default, workflows skip existing stage caches. Add `--overwrite` to rebuild.

## 4) Stage-by-Stage Commands

Use these for debugging, partial reruns, or custom orchestration.

```bash
# MIA scoring
python scripts/score_mia_text.py --config <config.yaml> --output <mia_cache.pt>
python scripts/score_mia_diffusion.py --config <config.yaml> --output <mia_cache.pt>

# DRA generation + scoring
python scripts/generate_dra.py --config <config.yaml> --output <dra_generation_cache.pt>
python scripts/score_dra_mia.py --config <config.yaml> --input <dra_generation_cache.pt> --output <dra_scored_cache.pt>

# CP-k post-scoring
python scripts/score_cpk.py --config <cpk_config.yaml> --input <mia_or_dra_cache.pt> --output <cpk_cache.pt>

# Analysis
python scripts/analyze_mia.py --config <config.yaml> --input <mia_or_mia_cpk_cache.pt>
python scripts/analyze_dra.py --config <config.yaml> --input <dra_scored_or_dra_cpk_cache.pt>
python scripts/evaluate_utility.py --config <config.yaml> --input <dra_generation_cache.pt>
python scripts/analyze_utility.py --config <config.yaml> --input <utility_features_cache.pt>
```

## 5) Workflow Semantics and Cache Artifacts

Expected stage order:

- **MIA workflow:** `mia_cache.pt` -> optional `mia_cpk_cache.pt` -> analysis artifacts.
- **DRA workflow:** `dra_generation_cache.pt` -> `dra_scored_cache.pt` -> optional `dra_cpk_cache.pt` -> analysis artifacts.
- **Utility workflow:** `utility_features_cache.pt` -> utility analysis outputs.

Cache schema (`.pt`) is standardized:

```python
{
    "schema_version": 1,
    "stage": "mia" | "dra_generation" | "dra_scored" | "cpk" | "utility_features",
    "config": {...},
    "config_hash": "...",
    "model_type": "baseline" | "dp" | "cpk",
    "modality": "text" | "diffusion",
    "created_at": "...",
    "data": {...},
    "aux": {...},
}
```

Each cache also writes a sidecar manifest: `<cache>.manifest.json`.

## 6) Utility Metrics

Current text utility metrics:

- `heldout_ppl` / `perplexity`
- `reference_ppl`
- `prometheusv2_fluency`

`mauve` is not part of the current utility pipeline.
Perplexity uses `utility.ppl_split` when provided; otherwise it defaults to
`dataset.target_split` (typically in-set `train_full`).

Prometheus-v2 scoring is configurable via `utility.prometheusv2` in config (model name, batch size, token limits, rubric, and scoring scope).

## 7) Publication Analysis

First extract compact raw score tables, then render publication plots/tables.

```bash
python scripts/extract_final_raw_scores.py \
  --input-root outputs/inference/final_workflows \
  --output-parquet outputs/inference/final_workflows/analysis/final_raw_scores_compact_all_datasets.parquet

python scripts/mia_publication_analysis.py \
  --input-parquet outputs/inference/final_workflows/analysis/final_raw_scores_compact_all_datasets.parquet \
  --output-dir outputs/inference/final_workflows/analysis/publication/mia

python scripts/dra_publication_analysis.py \
  --input-parquet outputs/inference/final_workflows/analysis/final_raw_scores_compact_all_datasets.parquet \
  --output-dir outputs/inference/final_workflows/analysis/publication/dra
```

Fast iteration mode:

```bash
python scripts/mia_publication_analysis.py --quick
python scripts/dra_publication_analysis.py --quick
```

## 8) Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Practical Notes

- Keep all commands at repository root so relative config/model paths resolve correctly.
- Many final workflow configs are pinned to specific checkpoint directories; update `models.*.checkpoint_dir` when reproducing on a new machine.
- Optional distance metrics rely on additional packages from `requirements.txt` (for example `lpips`, `bert-score`, `dreamsim`, `torchmetrics`, `rouge-score`).

