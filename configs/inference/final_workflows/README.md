# Final workflow configs

This folder contains experiment-ready inference configs keyed by dataset, with
model checkpoints pinned via `models.*.checkpoint_dir` to run folders under:

- `completed_nonprivate_runs/`
- `completed_private_runs/`

## Layout

Each dataset has its own subfolder:

- `automathtext/`
- `writingprompts/`
- `pokemon/`
- `laion_mi/`

Config names use a task prefix:

- `MIA_*` for MIA workflows
- `DRA_*` for DRA workflows
- `CPK_POST_*` for standalone CP-k post-scoring (`scripts/score_cpk.py`)

## Coverage notes

- All four datasets include DP configs for eps `10`, `20`, `50` where private DP runs exist in `completed_private_runs/`.
- All four datasets include CP-k configs (`MIA_CPK`, `DRA_CPK`, `CPK_POST_MIA`, `CPK_POST_DRA`) pinned to non-private triplet runs in `completed_nonprivate_runs/`.

## Running

Run from repo root:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/run_mia_workflow.py --config configs/inference/final_workflows/<dataset>/MIA_*.yaml
${CP_EXPERIMENTS_PYTHON} scripts/run_dra_workflow.py --config configs/inference/final_workflows/<dataset>/DRA_*.yaml
${CP_EXPERIMENTS_PYTHON} scripts/score_cpk.py --config configs/inference/final_workflows/<dataset>/CPK_POST_*.yaml --input <base_cache.pt> --output <cpk_cache.pt>
```

## Monitoring knobs

DRA workflows now support optional runtime monitoring controls through `runtime.monitor`:

- `enabled` (default: `true`): enable/disable tqdm bars.
- `show_rate` (default: `true`): include throughput in tqdm postfix (e.g. rows/s or candidates/s).
- `log_interval_batches` (default: `10`): cadence for tqdm postfix updates in distance loops.
- `leave_bars` (default: stage-specific): force whether progress bars remain in terminal after completion.
- `disable_diffusers_inner_bar` (default: `false`): disable inner Diffusers denoising bars to avoid double bars when outer DRA bars are enabled.
- `stage_prints` (default: `true`): enable stage boundary prints in DRA generation/scoring and workflow orchestration.

Example:

```yaml
runtime:
  device: cuda
  monitor:
    enabled: true
    show_rate: true
    log_interval_batches: 10
    leave_bars: false
    disable_diffusers_inner_bar: true
    stage_prints: true
```

## Text distance performance knobs

Text DRA scoring can become CPU-bound when `normalized_levenshtein` runs over long generations. Use these `dra_distance` options to improve throughput:

- `batch_size`: base batch size used by distance metrics.
- `text_chunk_size`: rows processed per scalar-distance chunk.
- `text_num_workers`: worker threads for scalar metrics (`normalized_levenshtein`, `bleu`).
- `bertscore_batch_size`: explicit BERTScore batch size override.
- `bertscore_half_precision`: run BERTScore in CUDA autocast FP16.
- `bertscore_deduplicate_pairs`: score only unique `(candidate, reference)` pairs then broadcast.
- `bertscore_max_length`: optional tokenizer max length override used by BERTScore.
