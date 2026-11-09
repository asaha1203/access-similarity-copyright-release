Below is a requirements-style draft for the inference pipeline. I kept the same execution style and split/model assumptions as the training requirements, while adding CP-k, MIA, DRA, distance, and utility evaluation. The CP-k definition here uses the paper’s three-model setup: a full-data draft model (p), and shard models (q_1,q_2), with filtering based on (\max_i \log(p(y\mid z)/q_i(y\mid z))).  The MIA and metric choices follow the paper’s experimental protocol: PIA for diffusion, MinK%++ for language, and the stated reconstruction/utility metrics. 

---

# Copyright Paper Inference Requirements

This document is the durable reference for implementing inference, membership inference, reconstruction attacks, and utility evaluation for the copyright experiments.

## 0) Scope and Non-Goals

### In scope

* Deterministic inference over trained language and diffusion models.
* CP-k proposal generation using a full-data draft model (p) and shard models (q_1,q_2).
* Membership inference attack generation and scoring for baseline, DP, and CP-k settings.
* Data reconstruction attack generation and analysis.
* Distance metrics for image and text reconstruction.
* Utility metrics for image and text generation.
* Reproducible logging of prompts, seeds, generated outputs, per-token/per-step scores, log-probability traces, and analysis artifacts.

### Out of scope

* Model training.
* Dataset re-splitting.
* Regenerating split manifests or split fingerprints.
* Retraining shard models or DP models during inference.
* Implementing CP-k rejection inside the sampler. The sampler should generate and score proposal samples; thresholding happens later in analysis.

## 1) Environment and Execution Rules

All install and run commands must use:

```bash
${CP_EXPERIMENTS_PYTHON}
```

Never rely on implicit `python` or `pip` from another environment.

Recommended bootstrap:

```bash
${CP_EXPERIMENTS_PYTHON} -m pip install -r requirements-inference.txt
```

The inference code should reuse the trained run artifacts from the training pipeline, including `adapter_model.safetensors`, `adapter_config.json`, `training_config.yaml`, `split_manifest.json`, and `split_fingerprint.json`. The original training requirements already treat these provenance files as compatibility-critical artifacts. 

## 2) Canonical Model and Dataset Grid

The inference pipeline should support the same modality/dataset grid as training.

| Modality  | Model family                              | Dataset(s)                                              |
| --------- | ----------------------------------------- | ------------------------------------------------------- |
| Language  | Llama2-7B-compatible Causal LM            | `math-ai/AutoMathText`, `euclaise/writingprompts`       |
| Diffusion | SD v1.4-compatible text-to-image pipeline | `diffusers/pokemon-gpt4-captions`, `antoniaaa/laion_mi` |

For each dataset, the following trained models may exist:

| Run type             | Split        | Role in inference                       |
| -------------------- | ------------ | --------------------------------------- |
| `baseline_full_lora` | `train_full` | Baseline model and CP-k draft model (p) |
| `shard1_lora`        | `shard1`     | CP-k shard model (q_1)                  |
| `shard2_lora`        | `shard2`     | CP-k shard model (q_2)                  |
| `dp_full_lora_eps10` | `train_full` | DP baseline                             |
| `dp_full_lora_eps20` | `train_full` | DP baseline                             |
| `dp_full_lora_eps50` | `train_full` | DP baseline                             |

The CP-k implementation should use:

[
p = \texttt{baseline_full_lora}, \qquad q_1 = \texttt{shard1_lora}, \qquad q_2 = \texttt{shard2_lora}.
]

The inference code should hard-fail if the three CP-k models do not share the same base model family, tokenizer or scheduler configuration, dataset name, split seed, and compatible split fingerprint metadata.

## 3) CP-k Semantics

CP-k generation should be implemented as a **proposal generation and scoring pipeline**, not as an online rejection sampler.

For a prompt (z), generate a candidate output (y\sim p(\cdot\mid z)) from the draft model. Then score that same candidate under (p,q_1,q_2), and compute

[
r_k(y,z)
========

\max\left{
\log p(y\mid z)-\log q_1(y\mid z),
\log p(y\mid z)-\log q_2(y\mid z)
\right}.
]

The generation pipeline should store all candidates, including candidates that would later be rejected. The analysis pipeline will later choose empirical thresholds (k_\alpha) based on percentiles of the observed (r_k) distribution.

Default acceptance-rate grid:

```yaml
cpk:
  alpha_grid: [0.10, 0.25, 0.50, 0.75, 1.00]
```

For each (\alpha), define (k_\alpha) as the empirical (\alpha)-quantile of `cpk_logratio_max`. The retained set is:

```text
cpk_logratio_max <= k_alpha
```

For language models, store both raw sequence log-ratios and length-normalized log-ratios. For diffusion models, store both raw trajectory log-ratios and per-step-normalized trajectory log-ratios.

## 4) Hardware and Model Loading Requirements

Inference should be optimized for concurrent execution on large GPUs.

Assume all required models can be resident on GPU at the same time, including the CP-k triplet ((p,q_1,q_2)) and any model needed for attack scoring.

Implementation requirements:

1. Default execution should keep all required models loaded concurrently.
2. CP-k proposal scoring should compute scores under ((p,q_1,q_2)) in the same pass over candidates whenever practical.
3. MIA and DRA scoring code paths should avoid model reload churn and should batch work to maximize GPU utilization.

A sequential fallback path is optional for exceptionally constrained environments, but it is not the default or primary target.

## 5) Shared Output and Artifact Contract

Every inference or analysis run must write:

```text
run_config.yaml
run_metadata.json
input_split_fingerprint.json
records.jsonl                  # sample-level records
summary_metrics.json           # aggregate metrics, if applicable
errors.jsonl                   # recoverable per-sample errors
```

Large arrays should be stored separately and referenced by path from `records.jsonl`.

Suggested layout:

```text
outputs/inference/<dataset>/<run_name>/<timestamp>/
  run_config.yaml
  run_metadata.json
  input_split_fingerprint.json
  records.jsonl
  tensors/
  images/
  text/
  metrics/
  errors.jsonl
```

All records must include:

```text
dataset
modality
run_type
model_role
model_checkpoint_path
split_name
split_fingerprint
sample_id
prompt_id
prompt
seed
created_at
```

## 6) CP-k Generation Pipeline

### 6.1 Language CP-k proposal generation

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/infer_cpk_text.py --config <config.yaml>
```

Behavior:

* Load the draft model (p), shard model (q_1), and shard model (q_2).
* Generate completions from (p) only.
* Use the dataset prompt as the conditioning context.
* Store every generated completion.
* Teacher-force the full prompt-plus-completion sequence under (p,q_1,q_2).
* Score only the generated/completion tokens by default. Prompt tokens should be masked out for generation log-probability summaries unless a config option explicitly requests prompt-inclusive scoring.
* Store per-token realized log-probabilities under all three models.
* Store sequence log-probabilities, mean log-probabilities, and CP-k log-ratios.

Required fields per text candidate:

```text
candidate_id
source_model_role = "draft_p"
prompt_token_count
generated_token_count
tokens
generated_text
decode_strategy
temperature
top_p
top_k
max_new_tokens
eos_reached

logprob_p_per_token
logprob_q1_per_token
logprob_q2_per_token

seq_logprob_p
seq_logprob_q1
seq_logprob_q2

mean_logprob_p
mean_logprob_q1
mean_logprob_q2

cpk_logratio_p_over_q1 = seq_logprob_p - seq_logprob_q1
cpk_logratio_p_over_q2 = seq_logprob_p - seq_logprob_q2
cpk_logratio_max = max(cpk_logratio_p_over_q1, cpk_logratio_p_over_q2)

cpk_logratio_p_over_q1_mean
cpk_logratio_p_over_q2_mean
cpk_logratio_max_mean
```

The code should store enough token-level information to recompute MinK%++ without rerunning inference, whenever feasible.

### 6.2 Diffusion CP-k scoring (current workflow path)

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/run_mia_workflow.py --config <config.yaml>
# or
${CP_EXPERIMENTS_PYTHON} scripts/run_dra_workflow.py --config <config.yaml>
# or standalone post-scoring on an existing cache
${CP_EXPERIMENTS_PYTHON} scripts/score_cpk.py --config <config.yaml> --input <mia_or_dra_cache.pt> --output <cpk_cache.pt>
```

Behavior:

* CP-k for diffusion is computed inside the cache-first workflow, not a standalone generation script.
* MIA diffusion computes PIA at `pia.default_timestep` and stores reusable tensors in cache for CP-k post-scoring.
* DRA diffusion computes trajectory-based CP-k during generation-time in the current workflow path.
* Standalone CP-k execution (`score_cpk.py`) reads an existing MIA/DRA cache and appends CP-k statistics.
* Current trajectory controls are `cpk.trajectory_steps` and the cache-provided start timestep (`t_start`, derived from `pia.default_timestep`).

Required fields per image candidate:

```text
candidate_id
source_model_role = "draft_p"
prompt
image_path
latent_path
initial_noise_path
scheduler_name
num_inference_steps
timesteps
guidance_scale
height
width
seed

trajectory_logscore_p_per_step
trajectory_logscore_q1_per_step
trajectory_logscore_q2_per_step

trajectory_logscore_p
trajectory_logscore_q1
trajectory_logscore_q2

trajectory_mean_logscore_p
trajectory_mean_logscore_q1
trajectory_mean_logscore_q2

cpk_logratio_p_over_q1 = trajectory_logscore_p - trajectory_logscore_q1
cpk_logratio_p_over_q2 = trajectory_logscore_p - trajectory_logscore_q2
cpk_logratio_max = max(cpk_logratio_p_over_q1, cpk_logratio_p_over_q2)

cpk_logratio_p_over_q1_mean
cpk_logratio_p_over_q2_mean
cpk_logratio_max_mean
```

The default diffusion trajectory score should be based on the reverse transition likelihood induced by a stochastic DDPM-style scheduler. For a transition (x_t\to x_{t-1}), compute the model-implied Gaussian transition log-probability under each model using the same (x_t), prompt embedding, timestep, classifier-free guidance setting, and scheduler variance. Sum these transition log-probabilities across timesteps.

If the configured scheduler is deterministic and does not define a meaningful transition density, the code should either switch to a stochastic scoring scheduler or explicitly use a pseudo-logscore based on negative noise-prediction error. The score mode must be recorded in `run_config.yaml` and in each record.

## 7) Membership Inference Attack Generation

The MIA generation/scoring pipeline should produce one record per evaluated member or nonmember sample. These records are then consumed by the MIA analysis pipeline.

### 7.1 Baseline and DP MIA: language

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/score_mia_text.py --config <config.yaml>
```

Behavior:

* For each baseline or DP language model, evaluate member and nonmember text samples.
* Members should come from the relevant training split.
* Nonmembers should come from the reserved not-in-training pool or the configured held-out pool.
* Compute MinK%++ scores under the target model.
* Store token-level log-probabilities, calibrated token scores, and aggregate MinK%++ scores.

### 7.2 Baseline and DP MIA: diffusion

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/score_mia_diffusion.py --config <config.yaml>
```

Behavior:

* For each baseline or DP diffusion model, evaluate member and nonmember image-caption samples.
* Compute PIA scores under the target model.
* Store the PIA score, timestep, (p)-norm, prompt, image path, and membership label.
* Smaller PIA scores should be interpreted as stronger membership evidence. The score direction must be stored explicitly as `lower_is_member`.

The paper describes PIA by computing the model’s initial noise estimate, using it to construct a noised sample at timestep (t), querying the model again, and scoring the norm between the two noise estimates; it uses (p=4) and often finds the best timestep near (t\approx 200). 

### 7.3 CP-k MIA: language

For CP-k language MIA, the attack score should be computed with respect to the draft model (p), while the CP-k release statistic should be computed using (p,q_1,q_2).

Behavior:

* Evaluate member and nonmember samples by teacher-forcing them under (p,q_1,q_2).
* Compute MinK%++ using the draft model (p).
* Compute CP-k log-ratio statistics using the same sequence under all three models.
* Store both the MIA score and the CP-k filtering statistic.

Required fields:

```text
membership_label
prompt
target_text
tokens
prompt_token_count
target_token_count

minkpp_scores_by_k
minkpp_default_score
minkpp_token_scores
mia_score = minkpp_default_score
mia_score_direction = "higher_is_member"

seq_logprob_p
seq_logprob_q1
seq_logprob_q2
cpk_logratio_p_over_q1
cpk_logratio_p_over_q2
cpk_logratio_max
cpk_logratio_max_mean
```

The MIA analysis stage will compute AUROC on retained subsets defined by empirical CP-k thresholds.

### 7.4 CP-k MIA: diffusion

For CP-k diffusion MIA, the attack score should be PIA under the draft model (p). Separately, the code should compute a CP-k trajectory statistic under (p,q_1,q_2).

Behavior:

* For each member or nonmember real image (y_0), run PIA using draft model (p).
* Store the PIA score as the primary MIA score.
* Starting from the PIA-induced noised latent or the configured noising point, run the reverse/scoring process and evaluate the trajectory under (p,q_1,q_2).
* Compute the CP-k log-ratio statistic.
* Store the sample even if its CP-k statistic exceeds later thresholds.

Required fields:

```text
membership_label
prompt
image_path
latent_path

pia_model_role = "draft_p"
pia_timestep
pia_norm_p
pia_score
mia_score = pia_score
mia_score_direction = "lower_is_member"

trajectory_logscore_p
trajectory_logscore_q1
trajectory_logscore_q2
cpk_logratio_p_over_q1
cpk_logratio_p_over_q2
cpk_logratio_max
cpk_logratio_max_mean
```

## 8) MinK%++ Implementation Requirements

MinK%++ should be implemented explicitly, not treated as average log-probability. It is a grey-box attack requiring token logits or probabilities from the target language model. 

For a tokenized sequence (x_{1:T}), and for each scored token position (t), compute:

[
\ell_t = \log p_\theta(x_t\mid x_{<t}).
]

Then compute the model’s categorical mean and variance over the full vocabulary:

[
\mu_t = \mathbb{E}*{w\sim p*\theta(\cdot\mid x_{<t})}
[\log p_\theta(w\mid x_{<t})],
]

[
\sigma_t^2 =
\operatorname{Var}*{w\sim p*\theta(\cdot\mid x_{<t})}
[\log p_\theta(w\mid x_{<t})].
]

The calibrated token score is:

[
s_t = \frac{\ell_t-\mu_t}{\sigma_t+\varepsilon}.
]

Implementation requirements:

* Use logits from the model being attacked.
* Mask prompt tokens by default.
* Compute scores only on target/completion tokens unless configured otherwise.
* Use a configurable numerical stability constant `epsilon`.
* Sort calibrated token scores in ascending order.
* For each configured (k), take the lowest (k%) of calibrated token scores and average them.
* Store the full token score vector and the aggregate scores.

Default config:

```yaml
minkpp:
  k_percent_grid: [5, 10, 20, 30, 40, 50]
  default_k_percent: 20
  epsilon: 1.0e-8
  score_prompt_tokens: false
  decision_direction: higher_is_member
```

## 9) PIA Implementation Requirements

PIA should be the default MIA method for diffusion models.

Given a real image (y_0):

1. Encode the image into the model latent space.
2. Query the model at the initial/proximal timestep to obtain its noise estimate (\epsilon_0).
3. Construct a noised latent (y_t) using the deterministic forward map.
4. Query the model again at timestep (t) to obtain (\epsilon_\theta(y_t,t)).
5. Compute

[
R_{t,p}(y_0)=|\epsilon_0-\epsilon_\theta(y_t,t)|_p.
]

Smaller values indicate stronger membership evidence.

Default config:

```yaml
pia:
  norm_p: 4
  timestep_grid: [50, 100, 150, 200, 250, 300]
  default_timestep: 200
  choose_best_timestep_by_validation_auroc: true
  decision_direction: lower_is_member
```

The implementation should support both a fixed timestep and a validation-selected timestep. If selecting by validation AUROC, the selected timestep must be recorded.

## 10) MIA Analysis Pipeline

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/analyze_mia.py --config <config.yaml>
```

### 10.1 Baseline and DP analysis

For baseline and DP models:

* Read MIA score records.
* Use `membership_label` as the binary label.
* Use `mia_score` and `mia_score_direction` to orient the score.
* Compute ROC and AUROC.
* Also compute TPR at fixed FPR values.

Default FPR grid:

```yaml
mia_analysis:
  fpr_grid: [0.01, 0.05, 0.10]
  bootstrap_num_samples: 1000
  bootstrap_seed: 1337
```

### 10.2 CP-k analysis

For CP-k models:

* Read MIA score records containing both MIA scores and `cpk_logratio_max`.
* Compute empirical CP-k thresholds (k_\alpha) from the configured `alpha_grid`.
* For each (\alpha), retain only samples with `cpk_logratio_max <= k_alpha`.
* Compute ROC, AUROC, and TPR at fixed FPR on the retained subset.
* Report the number and fraction of retained member and nonmember examples separately.
* For language, also support optional filtering by maximum target length or maximum scored-token length, because raw log-probability ratios can be length-sensitive.

Required output:

```text
metrics/mia_summary.json
metrics/mia_by_alpha.csv
metrics/roc_points_by_alpha.parquet
metrics/bootstrap_confidence_intervals.json
```

## 11) DRA Generation Pipeline

The DRA generation pipeline should generate candidate reconstructions from prompts associated with training samples. For diffusion, the prompt is the image caption. For language, the prompt is the source/title/prefix associated with the target text.

### 11.1 Baseline and DP DRA generation

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/generate_dra.py --config <config.yaml>
```

Behavior:

* For each target training sample, retrieve its original prompt.
* Generate `num_candidates_per_target` candidates from the target model.
* Store all generated candidates and their seeds.
* For diffusion, save generated images and latent metadata.
* For language, save generated text and token metadata if language DRA is enabled.
* Do not perform ranking or filtering in the generation script.

Default config:

```yaml
dra_generation:
  num_candidates_per_target: 10
  seed_base: 1337
  save_latents: true
  save_token_logprobs: true
```

### 11.2 CP-k DRA generation

For CP-k DRA, generation should use the draft model (p), but every generated candidate must also be scored under (q_1) and (q_2).

Behavior:

* Generate candidates from (p) using prompts from target training samples.
* Store every candidate.
* Track log-probabilities or trajectory scores under (p,q_1,q_2).
* Compute and store `cpk_logratio_max`.
* Do not filter during generation.
* Analysis will later decide which candidates are retained for a given (k_\alpha).

For diffusion, this means generated images should be accompanied by trajectory-level scores. For language, generated sequences should be accompanied by per-token log-probabilities.

## 12) DRA Analysis Pipeline 1: Ranking and Distance Table

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/analyze_dra_rank_and_distance.py --config <config.yaml>
```

This pipeline converts generated candidates into a ranked distance table.

Behavior:

* Read generated DRA candidates.
* Compute MIA scores for each generated candidate using the same attack method used in the MIA pipeline.

  * For language, use MinK%++.
  * For diffusion, use PIA.
  * For CP-k, compute the MIA score with respect to the draft model (p).
* Rank generated candidates by membership-likeness.

  * For MinK%++, larger scores are more member-like.
  * For PIA, smaller scores are more member-like.
* For each target sample, score only the top (\rho) fraction of generated candidates after ranking by MIA score.
* Compute distances between selected candidates and the target sample.
* Store all distances, MIA scores, ranks, and CP-k log-ratio fields.
* For baseline/DP runs, rank using the same target model that produced the reconstructions.
* For CP-k runs, rank using MIA scores from the draft model (p), while still retaining CP-k statistics from ((p,q_1,q_2)).

Default analysis config:

```yaml
# configs/inference/analysis.yaml
dra_analysis:
  rho: 0.10
```

### 12.1 Image distance metrics

Required image reconstruction metrics:

```yaml
image_distance_metrics:
  - normalized_l2
  - clipscore
  - lpips
```

Implementation requirements:

* `normalized_l2` should use a fixed image size and normalization convention.
* `clipscore` should be computed with a fixed CLIP model and preprocessing pipeline.
* `lpips` should use a fixed backbone, such as AlexNet or VGG, recorded in config.
* Directionality must be stored:

  * lower is closer for `normalized_l2`;
  * higher is closer for `clipscore`;
  * lower is closer for `lpips`.

### 12.2 Language distance metrics

Required language reconstruction metrics:

```yaml
language_distance_metrics:
  - bleu
  - normalized_levenshtein
  - bertscore_f1
```

Implementation requirements:

* BLEU should use NLTK sentence BLEU with a fixed whitespace tokenizer and fixed smoothing rule, reported as the average of BLEU-1/2/3/4.
* Normalized Levenshtein distance should be normalized by the maximum of candidate/reference length unless explicitly configured otherwise.
* Levenshtein should use an optimized implementation (e.g. `rapidfuzz`) as the default runtime path.
* BERTScore-F1 should use a fixed model checkpoint and language setting, plus batched GPU execution by default.
* Text distance execution should support chunked scalar processing, worker parallelism, and pair deduplication for BERTScore.
* Directionality must be stored:

  * higher is closer for `bleu`;
  * lower is closer for `normalized_levenshtein`;
  * higher is closer for `bertscore_f1`.

Required output:

```text
metrics/dra_ranked_candidates.parquet
metrics/dra_distance_table.parquet
metrics/dra_distance_summary.json
```

## 13) DRA Analysis Pipeline 2: Reconstruction Success

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/analyze_dra_success.py --config <config.yaml>
```

This pipeline takes the stored distance table and computes probability of reconstruction success.

For a distance metric (d) and threshold (\eta), reconstruction succeeds if the generated candidate is sufficiently close to the target sample. For distance metrics where lower is closer, success means:

```text
distance <= eta
```

For similarity metrics where higher is closer, success means:

```text
similarity >= eta
```

Behavior:

* Read `dra_distance_table.parquet`.
* For each model, dataset, metric, threshold, and top-proportion setting, compute reconstruction success.
* Compute target-level success by checking whether any retained candidate for a target succeeds.
* Average target-level success over target samples.
* Also report candidate-level success as a secondary diagnostic.
* For CP-k, compute reconstruction success separately for each empirical acceptance threshold (k_\alpha).
* Store confidence intervals using bootstrap over target samples.

Required output:

```text
metrics/dra_success_by_threshold.csv
metrics/dra_success_by_alpha.csv
metrics/dra_success_summary.json
metrics/dra_success_bootstrap_ci.json
```

Default threshold grids should be configurable by metric. The code should not hard-code a single reconstruction threshold.

## 14) Utility Evaluation Pipeline

Entrypoint:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/evaluate_utility.py --config <config.yaml>
```

### 14.1 Diffusion utility

Required utility metrics:

```yaml
diffusion_utility_metrics:
  - fid
  - clipscore
```

Behavior:

* Generate a fixed number of images from each model using a fixed prompt set.
* Compute FID against the configured reference image set.
* Compute CLIPScore for image-prompt alignment.
* For CP-k, report utility on:

  * the full proposal pool;
  * each retained subset defined by (k_\alpha).

### 14.2 Language utility

Required utility metrics:

```yaml
language_utility_metrics:
  - perplexity
  - prometheusv2_fluency
```

Behavior:

* Compute perplexity on a fixed held-out evaluation set.
* Generate completions from fixed prompts.
* Score fluency using PrometheusV2.
* For CP-k, report utility on:

  * the full proposal pool;
  * each retained subset defined by (k_\alpha).

## 15) Configuration Schema

Each inference run should be driven by a YAML config.

Example:

```yaml
project:
  name: copyright-paper
  output_root: outputs/inference
  seed: 1337

environment:
  python: ${CP_EXPERIMENTS_PYTHON}
  hf_cache_dir: ${HF_CACHE_DIR}

dataset:
  name: diffusers/pokemon-gpt4-captions
  processed_root: data/processed/diffusers_pokemon-gpt4-captions
  split_fingerprint_path: data/processed/diffusers_pokemon-gpt4-captions/split_fingerprint.json

models:
  draft_p:
    run_type: baseline_full_lora
    checkpoint_dir: runs/pokemon/baseline_full_lora
  shard_q1:
    run_type: shard1_lora
    checkpoint_dir: runs/pokemon/shard1_lora
  shard_q2:
    run_type: shard2_lora
    checkpoint_dir: runs/pokemon/shard2_lora
  target:
    run_type: dp_full_lora_eps10
    checkpoint_dir: runs/pokemon/dp_full_lora_eps10

cpk:
  enabled: true
  alpha_grid: [0.10, 0.25, 0.50, 0.75, 1.00]
  threshold_statistic: cpk_logratio_max
  normalize_for_text_length: true
  normalize_for_diffusion_steps: true

generation:
  num_candidates_per_prompt: 16
  max_new_tokens: 512
  temperature: 1.0
  top_p: 0.95
  guidance_scale: 7.5
  num_inference_steps: 50
  scheduler: ddpm

mia:
  method_text: minkpp
  method_diffusion: pia

minkpp:
  k_percent_grid: [5, 10, 20, 30, 40, 50]
  default_k_percent: 20
  epsilon: 1.0e-8
  score_prompt_tokens: false

pia:
  norm_p: 4
  timestep_grid: [50, 100, 150, 200, 250, 300]
  default_timestep: 200
  choose_best_timestep_by_validation_auroc: true

dra:
  num_candidates_per_target: 10

analysis_config:
  path: configs/inference/analysis.yaml

analysis:
  fpr_grid: [0.01, 0.05, 0.10]
  bootstrap_num_samples: 1000
  bootstrap_seed: 1337

wandb:
  enabled: false
  project: copyright-paper
  entity: null
  tags: []
```

## 16) Command Reference

CP-k text proposal generation:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/infer_cpk_text.py --config configs/inference/cpk_text.yaml
```

CP-k diffusion post-scoring (current):

```bash
${CP_EXPERIMENTS_PYTHON} scripts/score_cpk.py --config <config.yaml> --input <mia_or_dra_cache.pt> --output <cpk_cache.pt>
```

Text MIA scoring:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/score_mia_text.py --config configs/inference/mia_text.yaml
```

Diffusion MIA scoring:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/score_mia_diffusion.py --config configs/inference/mia_diffusion.yaml
```

MIA analysis:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/analyze_mia.py --config configs/inference/analyze_mia.yaml
```

DRA generation:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/generate_dra.py --config configs/inference/dra_generation.yaml
```

DRA ranking and distance analysis:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/analyze_dra_rank_and_distance.py --config configs/inference/dra_rank_distance.yaml
```

DRA reconstruction success analysis:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/analyze_dra_success.py --config configs/inference/dra_success.yaml
```

Utility evaluation:

```bash
${CP_EXPERIMENTS_PYTHON} scripts/evaluate_utility.py --config configs/inference/utility.yaml
```

## 17) Required Sanity Checks

1. All inference scripts must verify that split fingerprints match the training artifacts.
2. CP-k must use `baseline_full_lora` as (p), `shard1_lora` as (q_1), and `shard2_lora` as (q_2).
3. CP-k generation must not reject or discard samples.
4. CP-k analysis must define thresholds empirically from stored `cpk_logratio_max` values.
5. Language scoring must mask prompt tokens by default.
6. Language scoring must store token counts and length-normalized scores.
7. Diffusion scoring must use identical scheduler timesteps and prompt embeddings when scoring a trajectory under (p,q_1,q_2).
8. Diffusion scoring must record whether the score is a transition log-probability or a pseudo-logscore.
9. MIA records must include `membership_label`.
10. MIA records must include `mia_score_direction`.
11. PIA scores must be treated as lower-is-member.
12. MinK%++ scores must be treated as higher-is-member unless a later implementation explicitly validates a different sign convention.
13. DRA analysis must compute target-level reconstruction success, not only candidate-level success.
14. CP-k DRA analysis must report reconstruction success separately for each acceptance level (\alpha).
15. All metrics must record directionality, preprocessing, model checkpoint, and version information.
