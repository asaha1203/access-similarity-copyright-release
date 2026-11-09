Below is a ready-to-paste `requirements.md` version of the design, based on the workflow you provided. 

````markdown
# Requirements: MIA, CP-k, and MIA-Selected DRA Workflows

## 1. Objective

Implement a cache-first inference workflow for membership inference attacks (MIA), CP-k filtering, and data reconstruction attacks (DRA).

The implementation must follow the principle:

```text
expensive model calls → reusable .pt caches → cheap post-processing → final JSON/CSV/statistics
````

The primary goal is to make all expensive inference steps reusable and auditable. Final dataframes, JSON files, CSV files, and summary statistics should only be produced at the aggregation/reporting layer.

---

## 2. Scope

The workflow must support:

1. MIA for DP models and baseline models.
2. MIA for CP-k models, where the draft model `p` is attacked first and CP-k scoring is applied post hoc.
3. DRA by generating many candidates per target prompt, ranking candidates by MIA score, and evaluating reconstruction quality.
4. CP-k filtering for both MIA and DRA.
5. Sweeping multiple CP-k acceptance rates without regenerating samples or rerunning MIA.

The workflow does **not** need to introduce a complicated class hierarchy, folder redesign, or row-oriented intermediate dataframe format.

---

## 3. Global Requirements

### R1. Cache-first design

All expensive stages must write reusable `.pt` cache files.

Intermediate caches should be PyTorch dictionaries containing tensors, lists, paths, seeds, scores, and reusable attack artifacts.

Do not use JSON, CSV, or dataframe outputs for intermediate expensive stages.

JSON/CSV/statistics should only be produced by aggregation commands.

---

### R2. One expensive stage per command

Each command must perform exactly one major stage.

Each command must:

1. Load the config.
2. Load the required prior `.pt` cache, if applicable.
3. Run one expensive or aggregation stage.
4. Write a new `.pt` cache or final report.
5. Refuse to silently overwrite existing outputs unless `--overwrite` is passed.

---

### R3. Stable command interface

The runnable interface should consist of these commands:

```bash
python -m inference.run_mia_cache --config configs/exp.yaml
python -m inference.run_dra_generate_cache --config configs/exp.yaml
python -m inference.run_dra_mia_cache --config configs/exp.yaml --input dra_generation_cache.pt
python -m inference.run_cpk_cache --config configs/exp.yaml --input <cache.pt>
python -m inference.aggregate --config configs/exp.yaml --input <cache.pt>
```

The same command set should work for DP, baseline, and CP-k experiments.

---

### R4. Supported model types

The config must support:

```yaml
model:
  type: dp | baseline | cpk
  modality: diffusion | text
```

For DP or baseline models:

```yaml
model:
  type: dp
  modality: diffusion
  target_model: path/to/model
```

For CP-k models:

```yaml
model:
  type: cpk
  modality: diffusion
  draft_model: path/to/p
  q1_model: path/to/q1
  q2_model: path/to/q2
```

---

### R5. Determinism and auditability

Every cache must contain enough information to reproduce or audit the outputs.

At minimum, caches should store:

```python
{
    "config": ...,          # resolved config or config hash
    "sample_ids": ...,
    "seeds": ...,
    "model_type": ...,
    "modality": ...,
}
```

For generated candidates, cache either:

1. Paths to generated images/text artifacts, plus seeds and metadata; or
2. The generated tensors directly, if configured.

Do not rely on implicit ordering alone. Store explicit IDs and mappings.

---

### R6. No silent overwrite

Every command must check whether its output path already exists.

Default behavior:

```text
if output exists and --overwrite is false:
    raise an error
```

The command may overwrite only when explicitly passed:

```bash
--overwrite
```

---

## 4. Configuration Requirements

A minimal config should support the following structure.

### CP-k diffusion DRA example

```yaml
experiment:
  name: cpk_diffusion_dra
  output_dir: outputs/cpk_diffusion_dra

model:
  type: cpk
  modality: diffusion
  draft_model: path/to/p
  q1_model: path/to/q1
  q2_model: path/to/q2

task:
  type: dra
  attack: pia
  num_generations_per_prompt: 100

cpk:
  enabled: true
  acceptance_rates: [1.0, 0.5, 0.25, 0.1, 0.05]
  calibration_pool: single
  statistic: logp_minus_min_logq
  accept_rule: cpk_stat_below_quantile

cache:
  dtype: fp32
  save_intermediate_states: true
  save_generated_artifacts: path
  overwrite: false

aggregation:
  mia_score_direction: higher_is_member
  num_holdout_splits: 20
  seed: 0
```

### DP model example

```yaml
experiment:
  name: dp_diffusion_mia
  output_dir: outputs/dp_diffusion_mia

model:
  type: dp
  modality: diffusion
  target_model: path/to/model

task:
  type: mia
  attack: pia

cpk:
  enabled: false

cache:
  dtype: fp32
  overwrite: false

aggregation:
  mia_score_direction: higher_is_member
  num_holdout_splits: 20
  seed: 0
```

---

## 5. MIA Workflow

### Stage 1: Compute raw MIA cache

Command:

```bash
python -m inference.run_mia_cache --config configs/exp.yaml
```

Output:

```text
mia_cache.pt
```

#### DP / baseline behavior

For each member and nonmember sample:

1. Run the configured MIA attack directly against the target model.
2. Store raw reusable attack outputs.
3. Do not compute final statistics in this stage.

Required cache fields:

```python
{
    "sample_ids": list,
    "labels": torch.Tensor,        # 1 = member, 0 = nonmember
    "prompts": list,

    "mia": {
        "attack": "pia" | "minkplusplus",
        "scores": torch.Tensor,
        "aux": dict,
    },

    "model_type": "dp" | "baseline",
    "modality": "diffusion" | "text",
}
```

Examples of `mia["aux"]`:

```python
{
    "timesteps": ...,
    "noise_seeds": ...,
    "noised_inputs": ...,
    "token_losses": ...,
    "pia_scores_by_timestep": ...,
    "pia_scores_by_noise_seed": ...,
}
```

The exact scalarization of PIA can be deferred. The cache must store enough raw information to support later aggregation choices such as mean, max, median, or timestep-specific scoring.

---

#### CP-k behavior

For CP-k models, this stage attacks the draft model `p`.

For diffusion:

1. Run PIA on the draft model.
2. Store the PIA score.
3. Store the noised image, latent, timestep, and any state needed to later evaluate `q_1` and `q_2`.
4. If configured, store relevant intermediate states from the 20–30 step inference process.

For text:

1. Score the sequence under the draft model.
2. Store sequence-level and/or token-level log-probabilities.
3. Store the exact tokenized sequence so `q_1` and `q_2` can be evaluated later without regeneration.

Required CP-k cache fields:

```python
{
    "sample_ids": list,
    "labels": torch.Tensor,

    "draft": {
        "mia_scores": torch.Tensor,
        "logp": torch.Tensor | None,
        "aux": dict,
    },

    "cpk_inputs": {
        "modality": "diffusion" | "text",
        "states": object,
        "timesteps": object | None,
        "seeds": object,
        "tokenized_sequences": object | None,
    },

    "model_type": "cpk",
}
```

This cache must be sufficient to run CP-k scoring later without repeating generation or rerunning PIA.

---

### Stage 2: Compute CP-k scores

Command:

```bash
python -m inference.run_cpk_cache --config configs/exp.yaml --input mia_cache.pt
```

This command should only run when:

```python
cache["model_type"] == "cpk"
```

Output:

```text
mia_cpk_cache.pt
```

For every cached sample, evaluate the required likelihoods or transition probabilities under `q_1` and `q_2`.

The CP-k statistic is:

```text
cpk_stat = logp - min(logq1, logq2)
```

The output cache must store the components, not only the final scalar:

```python
{
    "logp": torch.Tensor,
    "logq1": torch.Tensor,
    "logq2": torch.Tensor,

    "min_logq": torch.minimum(logq1, logq2),
    "cpk_stat": logp - torch.minimum(logq1, logq2),
}
```

Even if an optimized implementation thresholds using `min_logq`, the cache must still store:

```text
logp, logq1, logq2, min_logq, cpk_stat
```

This is required for auditability.

Acceptance convention:

```python
accepted = cpk_stat <= threshold_for_acceptance_rate
```

If using an equivalent `min_logq` thresholding implementation, it must be tested against the `cpk_stat` acceptance rule.

The CP-k threshold must be calibrated using a single calibration pool.

---

### Stage 3: Aggregate MIA statistics

Command:

```bash
python -m inference.aggregate --config configs/exp.yaml --input mia_cache.pt
```

or, for CP-k:

```bash
python -m inference.aggregate --config configs/exp.yaml --input mia_cpk_cache.pt
```

Final outputs:

```text
mia_metrics.json
mia_metrics.csv
```

For DP and baseline models:

1. Load `mia_cache.pt`.
2. Use stored MIA scores.
3. Compute MIA metrics.
4. Compute standard deviations over holdout randomization.

For CP-k models:

1. Load `mia_cpk_cache.pt`.
2. For each acceptance rate `AR`, compute the CP-k threshold on the single calibration pool.
3. Apply the acceptance mask.
4. Compute MIA metrics only on accepted samples.
5. Include `AR = 1.0` as the no-rejection baseline.

Aggregation pseudocode:

```python
for ar in acceptance_rates:
    if model_type == "cpk":
        mask = compute_acceptance_mask(
            cpk_stat=cache["cpk"]["cpk_stat"],
            calibration_pool=calibration_pool,
            acceptance_rate=ar,
        )
    else:
        mask = torch.ones_like(labels, dtype=torch.bool)

    metrics = compute_mia_metrics(
        scores=mia_scores[mask],
        labels=labels[mask],
    )
```

Required MIA metrics:

```text
AUROC
AUPRC
TPR at fixed FPR
score gap between members and nonmembers
number accepted
acceptance fraction
standard deviation over holdout randomization
```

---

## 6. DRA Workflow

The DRA workflow must implement MIA-selected reconstruction.

The attacker should:

```text
generate many candidates
score each candidate by MIA
select the highest-MIA candidate
evaluate its distance to the target
```

Distance metrics are evaluation-only. They must not be used for candidate selection.

---

### Stage 1: Generate DRA candidates

Command:

```bash
python -m inference.run_dra_generate_cache --config configs/exp.yaml
```

Output:

```text
dra_generation_cache.pt
```

For each training prompt, generate `n` candidates.

For DP or baseline models:

```text
generate candidates from the target model
```

For CP-k models:

```text
generate candidates from the draft model p
```

For diffusion CP-k, store enough intermediate state to evaluate the CP-k likelihood terms later. Since inference is only 20–30 steps, the implementation may store either the full trajectory or the subset of states required for the CP-k computation.

Required cache fields:

```python
{
    "target_ids": list,
    "prompts": list,
    "target_paths": list,

    "candidate_ids": list,
    "candidate_to_target": torch.Tensor,

    "generations": {
        "paths": list,
        "seeds": torch.Tensor,
        "states": object,
        "timesteps": object | None,
        "tokenized_sequences": object | None,
    },

    "model_type": "dp" | "baseline" | "cpk",
    "modality": "diffusion" | "text",
}
```

---

### Stage 2: Score DRA candidates with MIA

Command:

```bash
python -m inference.run_dra_mia_cache --config configs/exp.yaml --input dra_generation_cache.pt
```

Output:

```text
dra_scored_cache.pt
```

For every generated candidate:

1. Run the configured MIA attack.
2. Store the MIA score and reusable attack artifacts.
3. Compute distances to the original target.
4. Do not use distances for selection.

For CP-k, the MIA score is still computed with respect to the draft model `p`.

Required cache fields:

```python
{
    "target_ids": list,
    "candidate_ids": list,
    "candidate_to_target": torch.Tensor,

    "mia": {
        "attack": "pia" | "minkplusplus",
        "scores": torch.Tensor,
        "aux": dict,
    },

    "distances": {
        "l2": torch.Tensor,
        "lpips": torch.Tensor,
        "clip": torch.Tensor,
    },

    "cpk_inputs": {
        "states": object,
        "timesteps": object | None,
        "logp": torch.Tensor | None,
        "tokenized_sequences": object | None,
    },

    "model_type": "dp" | "baseline" | "cpk",
    "modality": "diffusion" | "text",
}
```

---

### Stage 3: CP-k score DRA candidates

Command:

```bash
python -m inference.run_cpk_cache --config configs/exp.yaml --input dra_scored_cache.pt
```

This command only applies when:

```python
cache["model_type"] == "cpk"
```

Output:

```text
dra_cpk_cache.pt
```

For every generated candidate:

1. Evaluate `logp`, if not already stored.
2. Evaluate `logq1`.
3. Evaluate `logq2`.
4. Store `min_logq`.
5. Store `cpk_stat`.

Required fields:

```python
{
    "cpk": {
        "logp": torch.Tensor,
        "logq1": torch.Tensor,
        "logq2": torch.Tensor,
        "min_logq": torch.Tensor,
        "cpk_stat": torch.Tensor,
    }
}
```

---

### Stage 4: Aggregate DRA

Command:

```bash
python -m inference.aggregate --config configs/exp.yaml --input dra_scored_cache.pt
```

or, for CP-k:

```bash
python -m inference.aggregate --config configs/exp.yaml --input dra_cpk_cache.pt
```

Final outputs:

```text
dra_metrics.json
dra_metrics.csv
```

For each target prompt and acceptance rate:

1. Select all generated candidates for the target.
2. If CP-k is enabled, remove candidates rejected at the chosen acceptance rate.
3. Rank remaining candidates by MIA score only.
4. Select the highest-MIA candidate.
5. Report the selected candidate’s distance to the target.

Pseudocode:

```python
for ar in acceptance_rates:
    accepted = compute_acceptance_mask(...) if model_type == "cpk" else all_true

    for target in targets:
        candidates = candidates_for_target[target]
        candidates = candidates[accepted[candidates]]

        if len(candidates) == 0:
            record_no_accepted_candidate(target, ar)
            continue

        selected = argmax_mia_score(candidates)
        report_distances(selected, target)
```

Primary DRA metrics:

```text
mean top-1 MIA-selected L2
mean top-1 MIA-selected LPIPS
mean top-1 MIA-selected CLIP distance
success rate at L2 threshold τ
success rate at LPIPS threshold τ
success rate at CLIP threshold τ
fraction of targets with at least one accepted candidate
```

Optional metrics:

```text
top-k MIA-selected best distance
top-k MIA-selected success rate
```

For top-k metrics, the attacker must first select the top `k` candidates by MIA score. Distances may only be used afterward for evaluation.

---

## 7. CP-k Requirements

### R7.1. CP-k must be post hoc

For both MIA and DRA, CP-k must be implemented as:

```text
generate / attack under draft model p
score under q1 and q2
apply acceptance masks during aggregation
```

This allows sweeping many acceptance rates without regeneration or rerunning PIA.

---

### R7.2. Store all CP-k components

Every CP-k-scored cache must store:

```python
logp
logq1
logq2
min_logq
cpk_stat
```

Do not store only the accepted mask.

Do not store only the final scalar.

---

### R7.3. Single calibration pool

The CP-k acceptance threshold must be calibrated using one calibration pool.

The implementation should expose the calibration pool through config:

```yaml
cpk:
  calibration_pool: single
```

The same calibration logic should be used when sweeping all acceptance rates.

---

### R7.4. Acceptance rates

The aggregation code must support sweeping:

```yaml
cpk:
  acceptance_rates: [1.0, 0.5, 0.25, 0.1, 0.05]
```

`AR = 1.0` must be included as the no-rejection baseline.

---

## 8. MIA Score Direction

The config must explicitly specify score direction:

```yaml
aggregation:
  mia_score_direction: higher_is_member
```

The implementation must respect this when ranking DRA candidates.

For `higher_is_member`:

```python
selected = candidate_with_largest_mia_score
```

For `lower_is_member`, if supported:

```python
selected = candidate_with_smallest_mia_score
```

---

## 9. Distance Metrics

For diffusion DRA, the implementation should support:

```text
L2
LPIPS
CLIP distance or CLIP-based similarity converted to distance
```

Distances must be computed for evaluation only.

They must not affect candidate generation, CP-k acceptance, or MIA-based ranking.

---

## 10. Output Requirements

### Intermediate outputs

Intermediate outputs must be `.pt` files:

```text
mia_cache.pt
mia_cpk_cache.pt
dra_generation_cache.pt
dra_scored_cache.pt
dra_cpk_cache.pt
```

### Final outputs

Final aggregation may produce:

```text
mia_metrics.json
mia_metrics.csv
dra_metrics.json
dra_metrics.csv
```

Final reports should include:

```text
experiment name
model type
modality
attack type
acceptance rate
number of samples/candidates
number accepted
acceptance fraction
metric values
metric standard deviations, where applicable
config hash or resolved config
input cache path
```

---

## 11. Validation and Acceptance Criteria

A coding agent should consider the implementation correct only if the following tests pass.

### Cache existence and overwrite behavior

1. Running a command writes the expected cache file.
2. Re-running the same command without `--overwrite` fails.
3. Re-running with `--overwrite` succeeds.

---

### MIA cache validation

For a DP or baseline MIA run:

```python
assert "mia" in cache
assert "scores" in cache["mia"]
assert len(cache["mia"]["scores"]) == len(cache["labels"])
assert set(cache["labels"].tolist()).issubset({0, 1})
```

For CP-k MIA:

```python
assert cache["model_type"] == "cpk"
assert "draft" in cache
assert "mia_scores" in cache["draft"]
assert "cpk_inputs" in cache
```

---

### CP-k cache validation

For every CP-k cache:

```python
assert "logp" in cache["cpk"]
assert "logq1" in cache["cpk"]
assert "logq2" in cache["cpk"]
assert "min_logq" in cache["cpk"]
assert "cpk_stat" in cache["cpk"]

assert torch.allclose(
    cache["cpk"]["min_logq"],
    torch.minimum(cache["cpk"]["logq1"], cache["cpk"]["logq2"])
)

assert torch.allclose(
    cache["cpk"]["cpk_stat"],
    cache["cpk"]["logp"] - cache["cpk"]["min_logq"]
)
```

---

### DRA generation validation

For DRA generation:

```python
assert len(candidate_ids) == len(candidate_to_target)
assert candidate_to_target.min() >= 0
assert candidate_to_target.max() < len(target_ids)
```

Each target should have the configured number of generated candidates unless generation failed and the failure is explicitly recorded.

---

### DRA selection validation

For each target and acceptance rate:

1. Candidate selection must use MIA score only.
2. Distances must not be used before selection.
3. If no candidates are accepted, the target should be recorded as having no accepted reconstruction.
4. Top-1 metrics must use the single highest-MIA accepted candidate.

---

### AR = 1 validation

For CP-k aggregation:

```text
AR = 1.0 must be equivalent to no rejection.
```

The accepted mask for `AR = 1.0` should accept all valid candidates/samples.

---

## 12. Open Decisions

The implementation should leave the following configurable or easy to modify:

1. For CP-k diffusion, whether `logp(y | z)` is computed from the full reverse trajectory or from selected transition states.
2. Whether DRA calibration uses the same calibration pool as MIA or a candidate-specific calibration pool.
3. Whether generated image tensors are stored directly in `.pt` or stored as image paths plus latent/intermediate tensors.
4. Whether aggregation emits one combined report over all acceptance rates or one report per acceptance rate.

Default choice for now:

```text
Use one combined report over all acceptance rates.
Store generated artifacts by path.
Store latent/intermediate tensors in .pt.
Use one single calibration pool.
```

```
```
