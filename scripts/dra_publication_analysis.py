#!/usr/bin/env python
"""Publication-grade DRA figures/tables from finalized raw compact extracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_analysis_common import (
    apply_cpk_acceptance,
    bca_interval,
    ensure_dir,
    file_sha256,
    json_dump,
    pretty_dataset,
)


DATASET_ORDER = ["automathtext", "pokemon", "writingprompts", "laion_mi"]
METHOD_ORDER = [
    "baseline",
    "cpk_alpha_0.25",
    "cpk_alpha_0.50",
    "cpk_alpha_0.75",
    "dp_eps_5",
    "dp_eps_10",
    "dp_eps_20",
    "dp_eps_50",
]
# Plot-specific method subset (request: keep only DP eps 5 and 50).
PLOT_METHOD_ORDER = [
    "baseline",
    "cpk_alpha_0.25",
    "cpk_alpha_0.50",
    "cpk_alpha_0.75",
    "dp_eps_5",
    "dp_eps_50",
]
MODALITY_2X4_METHOD_ORDER = [
    "baseline",
    "cpk_alpha_0.25",
    "cpk_alpha_0.50",
    "cpk_alpha_0.75",
    "dp_eps_5",
    "dp_eps_10",
    "dp_eps_20",
    "dp_eps_50",
]
ALPHA_GRID = [round(float(x), 2) for x in np.linspace(0.10, 1.00, 19)]
METHOD_LABEL = {
    "baseline": "Baseline",
    "cpk_alpha_0.25": r"CP-$k$, $\alpha_k = 0.25$",
    "cpk_alpha_0.50": r"CP-$k$, $\alpha_k = 0.50$",
    "cpk_alpha_0.75": r"CP-$k$, $\alpha_k = 0.75$",
    "dp_eps_5": r"$(5,10^{-5})$-DP",
    "dp_eps_10": r"$(10,10^{-5})$-DP",
    "dp_eps_20": r"$(20,10^{-5})$-DP",
    "dp_eps_50": r"$(50,10^{-5})$-DP",
}

RECONSTRUCTION_2X2_PANEL_LAYOUT = [
    ("automathtext", "bertscore_f1", "MathAbstracts: BERTScore"),
    ("pokemon", "dreamsim", "Pokemon: DreamSim"),
    ("writingprompts", "normalized_levenshtein", "WritingPrompts: Normalized LEV"),
    ("laion_mi", "clip", "LAION-MI: CLIPScore"),
]
ETA_I_QUANTILES = [0.25, 0.50, 0.75]
ETA_I_STYLES = [
    {"color": "#1f77b4", "label": r"25% $\eta$ quantile"},
    {"color": "#2ca02c", "label": r"50% $\eta$ quantile"},
    {"color": "#9467bd", "label": r"75% $\eta$ quantile"},
]

PANEL_LAYOUT = [
    ("automathtext", "bertscore_f1", "MathAbstracts: BERTScore distance"),
    ("automathtext", "normalized_levenshtein", "MathAbstracts: Normalized LEV distance"),
    ("pokemon", "dreamsim", "Pokemon: DreamSim distance"),
    ("pokemon", "clip", "Pokemon: CLIP Similarity"),
    ("writingprompts", "bertscore_f1", "WritingPrompts: BERTScore distance"),
    ("writingprompts", "normalized_levenshtein", "WritingPrompts: Normalized LEV distance"),
    ("laion_mi", "dreamsim", "LAION-MI: DreamSim distance"),
    ("laion_mi", "clip", "LAION-MI: CLIP Similarity"),
]
TEXT_4X2_DATASETS = ["automathtext", "writingprompts"]
TEXT_4X2_METRICS = [
    ("bertscore_f1", "BERTScore"),
    ("rouge_l", "ROUGE-L"),
    ("bleu", "BLEU"),
    ("normalized_levenshtein", "Normalized LEV"),
]
IMAGE_4X2_DATASETS = ["pokemon", "laion_mi"]
IMAGE_4X2_METRICS = [
    ("normalized_l2", r"$\ell_2$ distance"),
    ("clip", "CLIP"),
    ("lpips", "LPIPS"),
    ("dreamsim", "DreamSim"),
]
WRITINGPROMPTS_TEXT_THRESHOLD_WINDOWS = {
    "bertscore_f1": (0.05, 0.30),
    "rouge_l": (0.75, 0.95),
    "bleu": (0.80, 1.00),
    "normalized_levenshtein": (0.65, 1.00),
}

THRESHOLD_TABLE_DEFAULTS = {
    "bertscore_f1": [0.15, 0.20, 0.25],
    "rouge_l": [0.75, 0.85, 0.95],
    "dreamsim": [0.20, 0.30, 0.40],
    "clip": [0.10, 0.20, 0.30],
}

STYLE = {
    "baseline": {"color": "black", "lw": 2.6, "ls": "-"},
    "cpk_alpha_0.25": {"color": "#1f77b4", "lw": 2.2, "ls": "-"},
    "cpk_alpha_0.50": {"color": "#2ca02c", "lw": 2.1, "ls": "-"},
    "cpk_alpha_0.75": {"color": "#9467bd", "lw": 2.1, "ls": "-"},
    "dp_eps_5": {"color": "#d62728", "lw": 2.0, "ls": "-"},
    "dp_eps_10": {"color": "#ff7f0e", "lw": 2.0, "ls": "-"},
    "dp_eps_20": {"color": "#17becf", "lw": 2.0, "ls": "-"},
    "dp_eps_50": {"color": "#8c564b", "lw": 2.0, "ls": "-"},
}

TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 14


def _style_for_method(method_key: str) -> Dict[str, float | str]:
    st = dict(STYLE[method_key])
    if method_key.startswith("cpk_alpha_"):
        alpha = float(method_key.split("_")[-1])
        # Upweight smaller-acceptance curves for visibility.
        st["lw"] = float(st.get("lw", 2.0)) + (1.0 - alpha) * 1.8
        st["zorder"] = 5.0 + (1.0 - alpha) * 3.0
    else:
        st["zorder"] = 3.0
    return st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready DRA figures/tables.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("outputs/inference/final_workflows/analysis/final_raw_scores_compact_all_datasets.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference/final_workflows/analysis/publication/dra"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--curve-points", type=int, default=250)
    parser.add_argument(
        "--panel-xmax-json",
        type=Path,
        default=None,
        help="Optional JSON map for per-panel xmax, e.g. {'automathtext:bleu':0.4}.",
    )
    parser.add_argument("--ci-method", choices=["bca", "percentile"], default="bca")
    parser.add_argument("--quick", action="store_true", help="Fast iteration mode with reduced compute.")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _method_subset(df: pd.DataFrame, method_key: str) -> pd.DataFrame:
    if method_key == "baseline":
        return df[df["run_name"] == "dra_cpk"].copy()
    if method_key.startswith("cpk_alpha_"):
        alpha = float(method_key.split("_")[-1])
        base = df[df["run_name"] == "dra_cpk"].copy()
        return apply_cpk_acceptance(base, alpha=alpha)
    if method_key.startswith("dp_eps_"):
        eps = method_key.replace("dp_eps_", "")
        return df[df["run_name"] == f"dra_dp_eps{eps}"].copy()
    return df.iloc[0:0].copy()


def _top1_by_target(subset: pd.DataFrame, all_target_ids: np.ndarray) -> pd.DataFrame:
    if subset.empty:
        out = pd.DataFrame({"target_sample_id": all_target_ids})
        return out

    direction_values = subset["mia_score_direction"].dropna().astype(str).tolist()
    direction = direction_values[0] if direction_values else "higher_is_member"
    ascending = direction != "higher_is_member"
    ranked = subset.sort_values(["target_sample_id", "mia_score"], ascending=[True, ascending], kind="mergesort")
    top1 = ranked.drop_duplicates(subset=["target_sample_id"], keep="first").copy()
    base_targets = pd.DataFrame({"target_sample_id": all_target_ids})
    merged = base_targets.merge(top1, on="target_sample_id", how="left")
    return merged


def _rate_curve_with_ci(
    distances: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    n = distances.size
    if n <= 0:
        z = np.zeros_like(thresholds, dtype=float)
        return {"point": z, "ci_low": z, "ci_high": z}
    idx = rng.integers(0, n, size=(n_bootstrap, n), endpoint=False)
    sampled = distances[idx]
    point = np.array([(distances <= t).mean() for t in thresholds], dtype=float)
    lo = np.empty_like(point)
    hi = np.empty_like(point)
    for i, thr in enumerate(thresholds):
        boot = (sampled <= thr).mean(axis=1)
        lo[i] = float(np.quantile(boot, 0.025))
        hi[i] = float(np.quantile(boot, 0.975))
    return {"point": point, "ci_low": lo, "ci_high": hi}


def _bootstrap_rates_for_thresholds(
    values: np.ndarray,
    thresholds: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = values.size
    if n <= 0:
        z = np.full(thresholds.shape[0], np.nan, dtype=float)
        return z, z, z
    success = (values[:, None] <= thresholds[None, :]).astype(np.float64)
    point = success.mean(axis=0)
    idx = rng.integers(0, n, size=(n_bootstrap, n), endpoint=False)
    sampled = success[idx]  # [B, N, T]
    boot = sampled.mean(axis=1)  # [B, T]
    if ci_method == "percentile":
        lo = np.quantile(boot, 0.025, axis=0)
        hi = np.quantile(boot, 0.975, axis=0)
        return point, lo, hi

    # BCa: efficient jackknife for Bernoulli rate.
    lo = np.empty(point.shape[0], dtype=float)
    hi = np.empty(point.shape[0], dtype=float)
    for j in range(point.shape[0]):
        col = success[:, j]
        jack = (col.sum() - col) / max(1, n - 1)
        l, h = bca_interval(point[j], boot[:, j], jack, alpha=0.05)
        lo[j] = l
        hi[j] = h
    return point, lo, hi


def _fmt_cell(point: float, low: float, high: float) -> str:
    if not np.isfinite(point):
        return "--"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{point:.3f}"
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def _method_to_experiment_name(dataset: str, method_key: str) -> str:
    if method_key == "baseline":
        return f"dra_cpk_{dataset}"
    if method_key.startswith("dp_eps_"):
        eps = method_key.replace("dp_eps_", "")
        return f"dra_dp_eps{eps}_{dataset}"
    raise ValueError(f"No experiment mapping for method: {method_key}")


def _latest_dra_metrics_path(repo_root: Path, dataset: str, experiment_name: str) -> Path | None:
    root = repo_root / "outputs" / "inference" / dataset / experiment_name
    if not root.exists():
        return None
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for run_dir in reversed(run_dirs):
        cand = run_dir / "metrics" / "dra_metrics.json"
        if cand.exists():
            return cand
    return None


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_bootstrap = min(int(args.n_bootstrap), 120)
        args.curve_points = min(int(args.curve_points), 140)
        args.ci_method = "percentile"
        args.skip_pdf = True
        args.skip_validation = True
    rng = np.random.default_rng(args.seed)

    output_dir = ensure_dir(args.output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    fig_dir = ensure_dir(output_dir / "figures")
    tab_dir = ensure_dir(output_dir / "tables")
    inter_dir = ensure_dir(output_dir / "intermediates")

    df = pd.read_parquet(args.input_parquet)
    dra_df = df[df["run_name"].astype(str).str.startswith("dra_")].copy()

    selected_by_method: Dict[Tuple[str, str], pd.DataFrame] = {}
    dataset_target_ids: Dict[str, np.ndarray] = {}
    for dataset in DATASET_ORDER:
        dataset_df = dra_df[dra_df["dataset"] == dataset].copy()
        # Use run-specific targets as denominator.
        base_targets = dataset_df[dataset_df["run_name"] == "dra_cpk"]["target_sample_id"].dropna().astype(int).unique()
        if base_targets.size == 0:
            base_targets = dataset_df["target_sample_id"].dropna().astype(int).unique()
        base_targets = np.sort(base_targets)
        dataset_target_ids[dataset] = base_targets
        for method_key in METHOD_ORDER:
            subset = _method_subset(dataset_df, method_key=method_key)
            top1 = _top1_by_target(subset, all_target_ids=base_targets)
            top1["dataset"] = dataset
            top1["method_key"] = method_key
            selected_by_method[(dataset, method_key)] = top1

    def _threshold_grid_for_panel(dataset: str, metric: str) -> Tuple[np.ndarray, float]:
        pooled_vals: List[np.ndarray] = []
        for method_key in METHOD_ORDER:
            top1 = selected_by_method.get((dataset, method_key))
            if top1 is None or metric not in top1.columns:
                continue
            vals = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                pooled_vals.append(vals)
        if pooled_vals:
            pooled = np.concatenate(pooled_vals)
            lo = float(np.quantile(pooled, 0.01))
            hi = float(np.quantile(pooled, 0.99))
            if hi <= lo:
                hi = lo + 1e-6
            return np.linspace(0.0, hi, args.curve_points), hi
        return np.linspace(0.0, 1.0, args.curve_points), 1.0

    def _render_modality_reconstruction_4x2(
        datasets: List[str],
        metrics: List[Tuple[str, str]],
        figure_stem: str,
    ) -> Path:
        fig, axes = plt.subplots(2, 4, figsize=(22.0, 9.2), sharex=False, sharey=True)
        fig.subplots_adjust(wspace=0.20, hspace=0.28, top=0.92, bottom=0.11)
        curve_rows_local: List[Dict[str, float | str]] = []
        legend_handle_by_method: Dict[str, any] = {}

        for row_idx, dataset in enumerate(datasets):
            for col_idx, (metric_key, metric_label) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                thresholds, x_max = _threshold_grid_for_panel(dataset=dataset, metric=metric_key)
                if dataset == "writingprompts" and metric_key in WRITINGPROMPTS_TEXT_THRESHOLD_WINDOWS:
                    x0, x1 = WRITINGPROMPTS_TEXT_THRESHOLD_WINDOWS[metric_key]
                    thresholds = np.linspace(min(x0, x1), max(x0, x1), args.curve_points)
                    x_max = max(x0, x1)
                has_curve = False
                for method_key in MODALITY_2X4_METHOD_ORDER:
                    top1 = selected_by_method[(dataset, method_key)]
                    if metric_key not in top1.columns:
                        continue
                    values_all = pd.to_numeric(top1[metric_key], errors="coerce").to_numpy(dtype=float)
                    values = values_all[np.isfinite(values_all)]
                    if values.size <= 0:
                        continue
                    curve = _rate_curve_with_ci(values, thresholds=thresholds, n_bootstrap=args.n_bootstrap, rng=rng)
                    st = _style_for_method(method_key)
                    line = ax.plot(
                        thresholds,
                        100.0 * curve["point"],
                        color=st["color"],
                        lw=st["lw"],
                        ls=st["ls"],
                        label=METHOD_LABEL[method_key],
                        zorder=st.get("zorder", 3.0),
                    )[0]
                    ax.fill_between(
                        thresholds,
                        100.0 * curve["ci_low"],
                        100.0 * curve["ci_high"],
                        color=st["color"],
                        alpha=0.14,
                        linewidth=0,
                    )
                    if method_key not in legend_handle_by_method:
                        legend_handle_by_method[method_key] = line
                    has_curve = True
                    for x, y, lo_ci, hi_ci in zip(thresholds, curve["point"], curve["ci_low"], curve["ci_high"]):
                        curve_rows_local.append(
                            {
                                "figure_key": figure_stem,
                                "dataset": dataset,
                                "dataset_label": pretty_dataset(dataset),
                                "metric": metric_key,
                                "method_key": method_key,
                                "threshold": float(x),
                                "reconstruction_probability": float(y),
                                "ci_low": float(lo_ci),
                                "ci_high": float(hi_ci),
                            }
                        )

                if not has_curve:
                    ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                if dataset == "writingprompts" and metric_key in WRITINGPROMPTS_TEXT_THRESHOLD_WINDOWS:
                    x0, x1 = WRITINGPROMPTS_TEXT_THRESHOLD_WINDOWS[metric_key]
                    ax.set_xlim(x0, x1)
                else:
                    ax.set_xlim(0.0, x_max)
                ax.set_ylim(0.0, 100.0)
                ax.set_title(f"{pretty_dataset(dataset)}: {metric_label}", fontsize=TITLE_FONTSIZE, fontweight="bold")
                ax.set_xlabel(r"Distance threshold $\eta$", fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_ylabel(r"Probability $\mathbf{Pr}(d(C, \hat{C}) \leq \eta)$.", fontsize=AXIS_LABEL_FONTSIZE)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
                ax.grid(alpha=0.25)

        ordered_methods = [m for m in MODALITY_2X4_METHOD_ORDER if m in legend_handle_by_method]
        if ordered_methods:
            handles = [legend_handle_by_method[m] for m in ordered_methods]
            labels = [METHOD_LABEL[m] for m in ordered_methods]
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=8,
                fontsize=11,
                frameon=True,
            )

        png_path = fig_dir / f"{figure_stem}.png"
        fig.savefig(png_path, dpi=350, bbox_inches="tight")
        if not args.skip_pdf:
            fig.savefig(fig_dir / f"{figure_stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(curve_rows_local).to_csv(inter_dir / f"{figure_stem}_with_ci.csv", index=False)
        return png_path

    def _render_modality_alpha_4x2(
        datasets: List[str],
        metrics: List[Tuple[str, str]],
        figure_stem: str,
    ) -> Path:
        fig, axes = plt.subplots(2, 4, figsize=(22.0, 9.0), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.20, hspace=0.24, top=0.92, bottom=0.11)
        alpha_rows_local: List[Dict[str, float | str]] = []
        legend_handles: List[any] = []

        for row_idx, dataset in enumerate(datasets):
            for col_idx, (metric_key, metric_label) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                baseline_top1 = selected_by_method.get((dataset, "baseline"))
                eta_pairs: List[Tuple[float, float]] = []
                if baseline_top1 is not None and metric_key in baseline_top1.columns:
                    baseline_values = pd.to_numeric(baseline_top1[metric_key], errors="coerce").to_numpy(dtype=float)
                    baseline_values = baseline_values[np.isfinite(baseline_values)]
                    if baseline_values.size > 0:
                        for q in ETA_I_QUANTILES:
                            eta_pairs.append((float(q), float(np.quantile(baseline_values, q))))

                if not eta_pairs:
                    ax.set_title(f"{pretty_dataset(dataset)}: {metric_label}", fontsize=TITLE_FONTSIZE, fontweight="bold")
                    ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                    ax.grid(alpha=0.25)
                    continue

                dataset_df = dra_df[dra_df["dataset"] == dataset].copy()
                target_ids = dataset_target_ids.get(dataset, np.array([], dtype=int))
                eta_values = np.array([eta for _, eta in eta_pairs], dtype=float)
                series = {
                    idx: {"xs": [], "ys": [], "ylo": [], "yhi": []}
                    for idx in range(len(eta_pairs))
                }

                for alpha in ALPHA_GRID:
                    subset = _method_subset(dataset_df, method_key=f"cpk_alpha_{alpha:.2f}")
                    top1 = _top1_by_target(subset, all_target_ids=target_ids)
                    if metric_key not in top1.columns:
                        continue
                    values_all = pd.to_numeric(top1[metric_key], errors="coerce").to_numpy(dtype=float)
                    values = values_all[np.isfinite(values_all)]
                    if values.size <= 0:
                        continue
                    point_vec, lo_vec, hi_vec = _bootstrap_rates_for_thresholds(
                        values=values,
                        thresholds=eta_values,
                        n_bootstrap=args.n_bootstrap,
                        rng=rng,
                        ci_method=args.ci_method,
                    )
                    for idx, (eta_quantile, eta_i) in enumerate(eta_pairs):
                        series[idx]["xs"].append(alpha)
                        series[idx]["ys"].append(float(point_vec[idx]))
                        series[idx]["ylo"].append(float(lo_vec[idx]) if np.isfinite(lo_vec[idx]) else float(point_vec[idx]))
                        series[idx]["yhi"].append(float(hi_vec[idx]) if np.isfinite(hi_vec[idx]) else float(point_vec[idx]))
                        alpha_rows_local.append(
                            {
                                "figure_key": figure_stem,
                                "dataset": dataset,
                                "dataset_label": pretty_dataset(dataset),
                                "metric": metric_key,
                                "alpha_k": float(alpha),
                                "eta_i": float(eta_i),
                                "eta_quantile": float(eta_quantile),
                                "reconstruction_probability": float(point_vec[idx]),
                                "ci_low": float(lo_vec[idx]),
                                "ci_high": float(hi_vec[idx]),
                            }
                        )

                for idx, (eta_quantile, _eta_i) in enumerate(eta_pairs):
                    xs = series[idx]["xs"]
                    if not xs:
                        continue
                    style = ETA_I_STYLES[min(idx, len(ETA_I_STYLES) - 1)]
                    x_arr = np.array(xs, dtype=float)
                    y_arr = np.array(series[idx]["ys"], dtype=float)
                    lo_arr = np.array(series[idx]["ylo"], dtype=float)
                    hi_arr = np.array(series[idx]["yhi"], dtype=float)
                    line = ax.plot(
                        x_arr,
                        y_arr * 100.0,
                        color=style["color"],
                        lw=2.0,
                        marker="o",
                        markersize=2.8,
                        label=style["label"],
                    )[0]
                    ax.fill_between(x_arr, lo_arr * 100.0, hi_arr * 100.0, color=style["color"], alpha=0.14, linewidth=0)
                    if row_idx == 0 and col_idx == 0 and len(legend_handles) < len(ETA_I_QUANTILES):
                        legend_handles.append(line)

                ax.set_title(f"{pretty_dataset(dataset)}: {metric_label}", fontsize=TITLE_FONTSIZE, fontweight="bold")
                ax.set_xlabel(r"Acceptance rate $\alpha_k$", fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_ylabel(r"Probability $\mathbf{Pr}(d(C, \hat{C}) \leq \eta)$.", fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_xlim(0.20, 1.00)
                ax.set_ylim(0.0, 100.0)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
                ax.grid(alpha=0.25)

        if legend_handles:
            fig.legend(
                legend_handles,
                [s["label"] for s in ETA_I_STYLES[: len(legend_handles)]],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=3,
                fontsize=11,
                frameon=True,
            )

        png_path = fig_dir / f"{figure_stem}.png"
        fig.savefig(png_path, dpi=350, bbox_inches="tight")
        if not args.skip_pdf:
            fig.savefig(fig_dir / f"{figure_stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(alpha_rows_local).to_csv(inter_dir / f"{figure_stem}_with_ci.csv", index=False)
        return png_path

    # ---- New 4x2 modality figures (text and image) ----
    _render_modality_reconstruction_4x2(
        datasets=TEXT_4X2_DATASETS,
        metrics=TEXT_4X2_METRICS,
        figure_stem="dra_reconstruction_probability_text_4x2",
    )
    _render_modality_alpha_4x2(
        datasets=TEXT_4X2_DATASETS,
        metrics=TEXT_4X2_METRICS,
        figure_stem="dra_reconstruction_vs_alpha_fixed_eta_text_4x2",
    )
    _render_modality_reconstruction_4x2(
        datasets=IMAGE_4X2_DATASETS,
        metrics=IMAGE_4X2_METRICS,
        figure_stem="dra_reconstruction_probability_image_4x2",
    )
    _render_modality_alpha_4x2(
        datasets=IMAGE_4X2_DATASETS,
        metrics=IMAGE_4X2_METRICS,
        figure_stem="dra_reconstruction_vs_alpha_fixed_eta_image_4x2",
    )

    # ---- Figure A: 2x2 reconstruction probability vs eta ----
    fig2x2, axes2x2 = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=False, sharey=True)
    fig2x2.subplots_adjust(wspace=0.22, hspace=0.30)
    recon_curve_rows: List[Dict[str, float]] = []
    for panel_idx, (dataset, metric, panel_title) in enumerate(RECONSTRUCTION_2X2_PANEL_LAYOUT):
        r, c = divmod(panel_idx, 2)
        ax = axes2x2[r, c]

        pooled_vals: List[np.ndarray] = []
        for method_key in METHOD_ORDER:
            top1 = selected_by_method.get((dataset, method_key))
            if top1 is None or metric not in top1.columns:
                continue
            vals = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                pooled_vals.append(vals)
        if pooled_vals:
            pooled = np.concatenate(pooled_vals)
            lo = float(np.quantile(pooled, 0.01))
            hi = float(np.quantile(pooled, 0.99))
            if hi <= lo:
                hi = lo + 1e-6
            thresholds = np.linspace(0.0, hi, args.curve_points)
            x_max = hi
        else:
            thresholds = np.linspace(0.0, 1.0, args.curve_points)
            x_max = 1.0

        for method_key in PLOT_METHOD_ORDER:
            top1 = selected_by_method[(dataset, method_key)]
            if metric not in top1.columns:
                continue
            values_all = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
            values = values_all[np.isfinite(values_all)]
            if values.size <= 0:
                continue
            curve = _rate_curve_with_ci(values, thresholds=thresholds, n_bootstrap=args.n_bootstrap, rng=rng)
            st = _style_for_method(method_key)
            ax.plot(
                thresholds,
                100.0 * curve["point"],
                color=st["color"],
                lw=st["lw"],
                ls=st["ls"],
                label=METHOD_LABEL[method_key],
                zorder=st.get("zorder", 3.0),
            )
            ax.fill_between(thresholds, 100.0 * curve["ci_low"], 100.0 * curve["ci_high"], color=st["color"], alpha=0.16, linewidth=0)
            for x, y, lo_ci, hi_ci in zip(thresholds, curve["point"], curve["ci_low"], curve["ci_high"]):
                recon_curve_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset(dataset),
                        "metric": metric,
                        "method_key": method_key,
                        "threshold": float(x),
                        "reconstruction_probability": float(y),
                        "ci_low": float(lo_ci),
                        "ci_high": float(hi_ci),
                    }
                )

        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 100.0)
        ax.set_title(panel_title, fontsize=17, fontweight="bold")
        ax.set_xlabel(r"Distance threshold $\eta$", fontsize=15)
        ax.set_ylabel(r"Probability $\mathbf{Pr}(d(C, \hat{C}) \leq \eta)$.", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        recon_legend_loc = "lower right" if panel_idx == 3 else "upper left"
        ax.legend(loc=recon_legend_loc, fontsize=12, frameon=True)

    fig2x2.savefig(fig_dir / "dra_reconstruction_probability_2x2.png", dpi=350, bbox_inches="tight")
    if not args.skip_pdf:
        fig2x2.savefig(fig_dir / "dra_reconstruction_probability_2x2.pdf", bbox_inches="tight")
    plt.close(fig2x2)
    pd.DataFrame(recon_curve_rows).to_csv(inter_dir / "dra_reconstruction_probability_2x2_with_ci.csv", index=False)

    # ---- Figure B: alpha_k ablation at fixed eta_i per panel ----
    fig_alpha, axes_alpha = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    fig_alpha.subplots_adjust(wspace=0.20, hspace=0.26)
    alpha_rows: List[Dict[str, float]] = []
    fixed_eta_by_panel: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for dataset, metric, _ in RECONSTRUCTION_2X2_PANEL_LAYOUT:
        baseline_top1 = selected_by_method.get((dataset, "baseline"))
        if baseline_top1 is None or metric not in baseline_top1.columns:
            continue
        baseline_values = pd.to_numeric(baseline_top1[metric], errors="coerce").to_numpy(dtype=float)
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if baseline_values.size <= 0:
            continue
        # Dataset/metric-specific fixed thresholds used across alpha_k.
        eta_pairs: List[Tuple[float, float]] = []
        for q in ETA_I_QUANTILES:
            eta_pairs.append((float(q), float(np.quantile(baseline_values, q))))
        fixed_eta_by_panel[(dataset, metric)] = eta_pairs

    for panel_idx, (dataset, metric, panel_title) in enumerate(RECONSTRUCTION_2X2_PANEL_LAYOUT):
        r, c = divmod(panel_idx, 2)
        ax = axes_alpha[r, c]
        eta_pairs = fixed_eta_by_panel.get((dataset, metric), [])
        if not eta_pairs:
            ax.set_title(panel_title, fontsize=17, fontweight="bold")
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.grid(alpha=0.25)
            continue

        dataset_df = dra_df[dra_df["dataset"] == dataset].copy()
        target_ids = dataset_target_ids.get(dataset, np.array([], dtype=int))
        for style_idx, (eta_quantile, eta_i) in enumerate(eta_pairs):
            xs: List[float] = []
            ys: List[float] = []
            ylo: List[float] = []
            yhi: List[float] = []
            for alpha in ALPHA_GRID:
                subset = _method_subset(dataset_df, method_key=f"cpk_alpha_{alpha:.2f}")
                top1 = _top1_by_target(subset, all_target_ids=target_ids)
                if metric not in top1.columns:
                    continue
                values_all = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
                values = values_all[np.isfinite(values_all)]
                if values.size <= 0:
                    continue
                point_vec, lo_vec, hi_vec = _bootstrap_rates_for_thresholds(
                    values=values,
                    thresholds=np.array([eta_i], dtype=float),
                    n_bootstrap=args.n_bootstrap,
                    rng=rng,
                    ci_method=args.ci_method,
                )
                xs.append(alpha)
                ys.append(float(point_vec[0]))
                ylo.append(float(lo_vec[0]) if np.isfinite(lo_vec[0]) else float(point_vec[0]))
                yhi.append(float(hi_vec[0]) if np.isfinite(hi_vec[0]) else float(point_vec[0]))
                alpha_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset(dataset),
                        "metric": metric,
                        "alpha_k": float(alpha),
                        "eta_i": float(eta_i),
                        "eta_quantile": float(eta_quantile),
                        "reconstruction_probability": float(point_vec[0]),
                        "ci_low": float(lo_vec[0]),
                        "ci_high": float(hi_vec[0]),
                    }
                )
            if xs:
                style = ETA_I_STYLES[min(style_idx, len(ETA_I_STYLES) - 1)]
                x_arr = np.array(xs, dtype=float)
                y_arr = np.array(ys, dtype=float)
                lo_arr = np.array(ylo, dtype=float)
                hi_arr = np.array(yhi, dtype=float)
                ax.plot(
                    x_arr,
                    y_arr * 100.0,
                    color=style["color"],
                    lw=2.1,
                    marker="o",
                    markersize=3.0,
                    label=style["label"],
                )
                ax.fill_between(x_arr, lo_arr * 100.0, hi_arr * 100.0, color=style["color"], alpha=0.14, linewidth=0)

        ax.set_title(panel_title, fontsize=17, fontweight="bold")
        ax.set_xlabel(r"Acceptance rate $\alpha_k$", fontsize=15)
        ax.set_ylabel(r"Probability $\mathbf{Pr}(d(C, \hat{C}) \leq \eta)$.", fontsize=15)
        ax.set_xlim(0.20, 1.00)
        ax.set_ylim(0.0, 100.0)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=12, frameon=True)

    fig_alpha.savefig(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_2x2.png", dpi=350, bbox_inches="tight")
    if not args.skip_pdf:
        fig_alpha.savefig(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_2x2.pdf", bbox_inches="tight")
    plt.close(fig_alpha)
    pd.DataFrame(alpha_rows).to_csv(inter_dir / "dra_reconstruction_vs_alpha_fixed_eta_2x2_with_ci.csv", index=False)

    # ---- 2x4 threshold-curve figure ----
    panel_xmax: Dict[Tuple[str, str], float] = {}
    if args.panel_xmax_json is not None and args.panel_xmax_json.exists():
        raw = json.loads(args.panel_xmax_json.read_text(encoding="utf-8"))
        for key, value in raw.items():
            dataset, metric = str(key).split(":", maxsplit=1)
            panel_xmax[(dataset, metric)] = float(value)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)
    fig.subplots_adjust(wspace=0.18, hspace=0.30, top=0.90, bottom=0.10)
    curve_rows: List[Dict[str, float]] = []
    legend_handle_by_method: Dict[str, any] = {}

    for i, (dataset, metric, panel_title) in enumerate(PANEL_LAYOUT):
        r, c = divmod(i, 4)
        ax = axes[r, c]

        # Default (editable) behavior: infer per-panel threshold range from data.
        pooled_vals = []
        for method_key in METHOD_ORDER:
            top1 = selected_by_method.get((dataset, method_key))
            if top1 is None or metric not in top1.columns:
                continue
            vals = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                pooled_vals.append(vals)
        if pooled_vals:
            pooled = np.concatenate(pooled_vals)
            lo = float(np.quantile(pooled, 0.01))
            hi = float(np.quantile(pooled, 0.99))
            if hi <= lo:
                hi = lo + 1e-6
            x_max = float(panel_xmax.get((dataset, metric), hi))
            thresholds = np.linspace(0.0, x_max, args.curve_points)
        else:
            x_max = float(panel_xmax.get((dataset, metric), 1.0))
            thresholds = np.linspace(0.0, x_max, args.curve_points)

        for method_key in PLOT_METHOD_ORDER:
            top1 = selected_by_method[(dataset, method_key)]
            if metric not in top1.columns:
                continue
            distances_all = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(distances_all)
            distances = distances_all[finite_mask]
            denom_used = int(distances.size)
            denom_total = int(distances_all.size)
            if denom_used <= 0:
                continue
            curve = _rate_curve_with_ci(distances, thresholds=thresholds, n_bootstrap=args.n_bootstrap, rng=rng)
            st = _style_for_method(method_key)
            line = ax.plot(
                thresholds,
                100.0 * curve["point"],
                color=st["color"],
                lw=st["lw"],
                ls=st["ls"],
                label=METHOD_LABEL[method_key],
                zorder=st.get("zorder", 3.0),
            )[0]
            ax.fill_between(thresholds, 100.0 * curve["ci_low"], 100.0 * curve["ci_high"], color=st["color"], alpha=0.14, linewidth=0)
            if method_key not in legend_handle_by_method:
                legend_handle_by_method[method_key] = line
            for x, y, lo_ci, hi_ci in zip(thresholds, curve["point"], curve["ci_low"], curve["ci_high"]):
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset(dataset),
                        "metric": metric,
                        "method_key": method_key,
                        "threshold": float(x),
                        "extraction_rate": float(y),
                        "ci_low": float(lo_ci),
                        "ci_high": float(hi_ci),
                        "n_used": denom_used,
                        "n_total_targets": denom_total,
                        "used_fraction": float(denom_used / max(1, denom_total)),
                    }
                )

        ax.set_title(panel_title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel(r"Distance threshold $\eta$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Fraction reconstructed (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(0.0, x_max)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.25)
    ordered_legend_methods = [m for m in PLOT_METHOD_ORDER if m in legend_handle_by_method]
    if ordered_legend_methods:
        handles = [legend_handle_by_method[m] for m in ordered_legend_methods]
        labels = [METHOD_LABEL[m] for m in ordered_legend_methods]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.040),
            ncol=6,
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
        )

    fig.savefig(fig_dir / "dra_extraction_threshold_curves_2x4.png", dpi=350, bbox_inches="tight")
    if not args.skip_pdf:
        fig.savefig(fig_dir / "dra_extraction_threshold_curves_2x4.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(inter_dir / "dra_threshold_curves_with_ci.csv", index=False)

    # ---- Comprehensive threshold tables ----
    table_rows: List[Dict[str, float]] = []
    for dataset in DATASET_ORDER:
        metrics = (
            ["bertscore_f1", "rouge_l"]
            if dataset in {"automathtext", "writingprompts"}
            else ["dreamsim", "clip"]
        )
        for metric in metrics:
            thresholds = THRESHOLD_TABLE_DEFAULTS[metric]
            for method_key in METHOD_ORDER:
                top1 = selected_by_method[(dataset, method_key)]
                if metric not in top1.columns:
                    continue
                values_all = pd.to_numeric(top1[metric], errors="coerce").to_numpy(dtype=float)
                finite_mask = np.isfinite(values_all)
                values = values_all[finite_mask]
                n_used = int(values.size)
                n_total = int(values_all.size)
                if n_used <= 0:
                    continue
                th = np.asarray(thresholds, dtype=float)
                point_vec, lo_vec, hi_vec = _bootstrap_rates_for_thresholds(
                    values=values,
                    thresholds=th,
                    n_bootstrap=args.n_bootstrap,
                    rng=rng,
                    ci_method=args.ci_method,
                )
                for thr, point, ci_low, ci_high in zip(th, point_vec, lo_vec, hi_vec):
                    table_rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": pretty_dataset(dataset),
                            "metric": metric,
                            "threshold": float(thr),
                            "method_key": method_key,
                            "method_label": METHOD_LABEL[method_key],
                            "n_targets_used": n_used,
                            "n_targets_total": n_total,
                            "used_fraction": float(n_used / max(1, n_total)),
                            "extraction_rate": float(point),
                            "ci_low": float(ci_low),
                            "ci_high": float(ci_high),
                        }
                    )
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(tab_dir / "dra_extraction_rates_long.csv", index=False)

    image_df = table_df[table_df["dataset"].isin(["laion_mi", "pokemon"])].copy()
    text_df = table_df[table_df["dataset"].isin(["automathtext", "writingprompts"])].copy()
    image_df.to_csv(tab_dir / "dra_image_extraction_table.csv", index=False)
    text_df.to_csv(tab_dir / "dra_text_extraction_table.csv", index=False)

    latex_lines: List[str] = []
    for _, row in table_df.sort_values(["metric", "threshold", "dataset", "method_key"]).iterrows():
        latex_lines.append(
            " & ".join(
                [
                    row["dataset_label"],
                    row["metric"],
                    f"{row['threshold']:.4f}",
                    row["method_label"],
                    _fmt_cell(row["extraction_rate"], row["ci_low"], row["ci_high"]),
                ]
            )
            + r" \\"
        )
    (tab_dir / "dra_extraction_table_rows.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    warnings: List[str] = []
    bad = table_df[(table_df["extraction_rate"] < -1e-6) | (table_df["extraction_rate"] > 1.0 + 1e-6)]
    if not bad.empty:
        warnings.append(f"Found {bad.shape[0]} out-of-range extraction rates.")

    validation_df = pd.DataFrame()
    validation_path = inter_dir / "dra_reference_validation.csv"
    if not args.skip_validation:
        validation_rows: List[Dict[str, float | str]] = []
        for dataset in DATASET_ORDER:
            metric_keys = (
                ["bertscore_f1", "rouge_l"]
                if dataset in {"automathtext", "writingprompts"}
                else ["dreamsim", "clip"]
            )
            for method_key in ["baseline", "dp_eps_5", "dp_eps_10", "dp_eps_20", "dp_eps_50"]:
                top1 = selected_by_method.get((dataset, method_key))
                if top1 is None or top1.empty:
                    continue
                exp_name = _method_to_experiment_name(dataset, method_key)
                ref_path = _latest_dra_metrics_path(repo_root, dataset=dataset, experiment_name=exp_name)
                if ref_path is None:
                    continue
                payload = json.loads(ref_path.read_text(encoding="utf-8"))
                ref_rows = payload.get("rows", [])
                if not ref_rows:
                    continue
                ref = None
                for row in ref_rows:
                    if abs(float(row.get("acceptance_rate", 1.0)) - 1.0) < 1e-9:
                        ref = row
                        break
                if ref is None:
                    ref = ref_rows[0]
                for metric in metric_keys:
                    local_values = pd.to_numeric(top1.get(metric), errors="coerce").to_numpy(dtype=float)
                    local_mean = float(np.nanmean(local_values)) if np.isfinite(local_values).any() else float("nan")
                    ref_key = f"mean_top1_mia_selected_{metric}"
                    ref_mean = float(ref.get(ref_key, np.nan))
                    validation_rows.append(
                        {
                            "dataset": dataset,
                            "method_key": method_key,
                            "metric": metric,
                            "reference_metrics_path": str(ref_path),
                            "our_mean_top1": local_mean,
                            "ref_mean_top1": ref_mean,
                            "abs_diff": float(abs(local_mean - ref_mean)) if np.isfinite(local_mean) and np.isfinite(ref_mean) else float("nan"),
                        }
                    )

        validation_df = pd.DataFrame(validation_rows)
        if not validation_df.empty:
            validation_df.to_csv(validation_path, index=False)
            max_abs_diff = float(validation_df["abs_diff"].dropna().max()) if validation_df["abs_diff"].dropna().size else 0.0
            if max_abs_diff > 1e-3:
                warnings.append(f"Reference validation max abs diff={max_abs_diff:.6f}")

    json_dump(
        output_dir / "dra_analysis_manifest.json",
        {
            "input_parquet": str(args.input_parquet),
            "input_parquet_sha256": file_sha256(args.input_parquet),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "ci_method": args.ci_method,
            "quick_mode": bool(args.quick),
            "panel_layout": PANEL_LAYOUT,
            "threshold_table_defaults": THRESHOLD_TABLE_DEFAULTS,
            "methods": METHOD_ORDER,
            "reconstruction_denominator": "targets_with_finite_metric_after_method_filtering",
            "warnings": warnings,
            "outputs": {
                "figures": [
                    str(fig_dir / "dra_reconstruction_probability_text_4x2.png"),
                    str(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_text_4x2.png"),
                    str(fig_dir / "dra_reconstruction_probability_image_4x2.png"),
                    str(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_image_4x2.png"),
                    str(fig_dir / "dra_reconstruction_probability_2x2.png"),
                    str(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_2x2.png"),
                    str(fig_dir / "dra_extraction_threshold_curves_2x4.png"),
                ],
                "tables": [
                    str(tab_dir / "dra_extraction_rates_long.csv"),
                    str(tab_dir / "dra_image_extraction_table.csv"),
                    str(tab_dir / "dra_text_extraction_table.csv"),
                    str(tab_dir / "dra_extraction_table_rows.tex"),
                ],
                "intermediates": [
                    str(inter_dir / "dra_reconstruction_probability_text_4x2_with_ci.csv"),
                    str(inter_dir / "dra_reconstruction_vs_alpha_fixed_eta_text_4x2_with_ci.csv"),
                    str(inter_dir / "dra_reconstruction_probability_image_4x2_with_ci.csv"),
                    str(inter_dir / "dra_reconstruction_vs_alpha_fixed_eta_image_4x2_with_ci.csv"),
                    str(inter_dir / "dra_reconstruction_probability_2x2_with_ci.csv"),
                    str(inter_dir / "dra_reconstruction_vs_alpha_fixed_eta_2x2_with_ci.csv"),
                    str(inter_dir / "dra_threshold_curves_with_ci.csv"),
                ],
            },
        },
    )
    if not args.skip_pdf or not args.skip_validation:
        manifest_path = output_dir / "dra_analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.skip_pdf:
            manifest["outputs"]["figures"].extend(
                [
                    str(fig_dir / "dra_reconstruction_probability_text_4x2.pdf"),
                    str(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_text_4x2.pdf"),
                    str(fig_dir / "dra_reconstruction_probability_image_4x2.pdf"),
                    str(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_image_4x2.pdf"),
                    str(fig_dir / "dra_reconstruction_probability_2x2.pdf"),
                    str(fig_dir / "dra_reconstruction_vs_alpha_fixed_eta_2x2.pdf"),
                    str(fig_dir / "dra_extraction_threshold_curves_2x4.pdf"),
                ]
            )
        if not args.skip_validation:
            manifest["outputs"]["intermediates"].append(str(validation_path))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"DRA publication outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

