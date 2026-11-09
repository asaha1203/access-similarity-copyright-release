#!/usr/bin/env python
"""Publication-grade MIA figures/tables from finalized raw compact extracts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_analysis_common import (
    MIA_PANEL_ORDER,
    apply_cpk_acceptance,
    bca_interval,
    ensure_dir,
    file_sha256,
    json_dump,
    mia_metrics_from_labels_scores,
    orient_scores_for_member_attack,
    pretty_dataset,
    roc_curve_binary,
    stratified_bootstrap_member_indices,
)


FPR_TARGETS = (0.01, 0.05, 0.10)
DATASET_ORDER = ["automathtext", "pokemon", "writingprompts", "laion_mi"]
SEPARATE_THRESHOLD_DATASETS = {"automathtext", "writingprompts"}

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

ROC_METHODS = ["baseline", "dp_eps_5", "dp_eps_50", "cpk_alpha_0.25", "cpk_alpha_0.50", "cpk_alpha_0.75"]
ROC_DP_METHODS = [
    "baseline",
    "dp_eps_5",
    "dp_eps_10",
    "dp_eps_20",
    "dp_eps_50",
    "cpk_alpha_0.25",
    "cpk_alpha_0.50",
    "cpk_alpha_0.75",
]
# Denser alpha sweep makes TPR@FPR confidence bands less blocky.
ALPHA_GRID = [round(float(x), 2) for x in np.linspace(0.10, 1.00, 19)]

STYLE = {
    "baseline": {"color": "black", "lw": 2.8, "ls": "-"},
    "dp_eps_5": {"color": "#d62728", "lw": 2.2, "ls": "-"},
    "cpk_alpha_0.25": {"color": "#1f77b4", "lw": 2.2, "ls": "-"},
    "cpk_alpha_0.50": {"color": "#2ca02c", "lw": 2.1, "ls": "-"},
    "cpk_alpha_0.75": {"color": "#9467bd", "lw": 2.1, "ls": "-"},
    "dp_eps_10": {"color": "#ff7f0e", "lw": 2.0, "ls": "-"},
    "dp_eps_20": {"color": "#17becf", "lw": 2.0, "ls": "-"},
    "dp_eps_50": {"color": "#8c564b", "lw": 2.0, "ls": "-"},
}

TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 15
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
ROC_DP_LEGEND_FONTSIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready MIA figures/tables.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("outputs/inference/final_workflows/analysis/final_raw_scores_compact_all_datasets.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference/final_workflows/analysis/publication/mia"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fpr-max", type=float, default=0.10)
    parser.add_argument("--roc-grid-points", type=int, default=250)
    parser.add_argument("--ci-method", choices=["bca", "percentile"], default="bca")
    parser.add_argument("--quick", action="store_true", help="Fast iteration mode with reduced compute.")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def _method_subset(df: pd.DataFrame, method_key: str) -> pd.DataFrame:
    dataset_values = df["dataset"].dropna().astype(str).unique().tolist() if "dataset" in df.columns else []
    dataset = dataset_values[0] if dataset_values else ""

    def _apply_cpk(df_in: pd.DataFrame, alpha: float) -> pd.DataFrame:
        if dataset in SEPARATE_THRESHOLD_DATASETS and "membership_label" in df_in.columns:
            chunks: List[pd.DataFrame] = []
            for label in (0, 1):
                part = df_in[pd.to_numeric(df_in["membership_label"], errors="coerce") == label].copy()
                if part.empty:
                    continue
                chunks.append(apply_cpk_acceptance(part, alpha=alpha))
            if not chunks:
                return df_in.iloc[0:0].copy()
            return pd.concat(chunks, axis=0, ignore_index=True)
        return apply_cpk_acceptance(df_in, alpha=alpha)

    if method_key == "baseline":
        return df[df["run_name"] == "mia_cpk"].copy()
    if method_key.startswith("cpk_alpha_"):
        alpha = float(method_key.split("_")[-1])
        base = df[df["run_name"] == "mia_cpk"].copy()
        return _apply_cpk(base, alpha=alpha)
    if method_key.startswith("dp_eps_"):
        eps = method_key.replace("dp_eps_", "")
        return df[df["run_name"] == f"mia_dp_eps{eps}"].copy()
    return df.iloc[0:0].copy()


def _labels_scores(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    labels = pd.to_numeric(df["membership_label"], errors="coerce").to_numpy(dtype=float)
    raw_scores = pd.to_numeric(df["mia_score"], errors="coerce").to_numpy(dtype=float)
    direction_values = df["mia_score_direction"].dropna().astype(str).tolist()
    direction = direction_values[0] if direction_values else "higher_is_member"
    valid = np.isfinite(raw_scores) & np.isfinite(labels)
    labels = labels[valid].astype(np.int32)
    scores = orient_scores_for_member_attack(raw_scores[valid], direction=direction)
    return labels, scores


def _bootstrap_metric_bundle(
    dataset: str,
    labels: np.ndarray,
    scores: np.ndarray,
    fpr_targets: Sequence[float],
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_method: str,
) -> Dict[str, Dict[str, float]]:
    point = _mia_metrics_from_labels_scores(
        dataset=dataset,
        labels=labels,
        scores=scores,
        fpr_targets=fpr_targets,
    )
    keys = list(point.keys())
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return {k: {"point": float(point[k]), "ci_low": float("nan"), "ci_high": float("nan")} for k in keys}

    sampled = stratified_bootstrap_member_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
    boot_values = {k: np.empty(n_bootstrap, dtype=float) for k in keys}
    for b in range(n_bootstrap):
        idx = sampled[b]
        metric = _mia_metrics_from_labels_scores(
            dataset=dataset,
            labels=labels[idx],
            scores=scores[idx],
            fpr_targets=fpr_targets,
        )
        for k in keys:
            boot_values[k][b] = metric[k]

    out: Dict[str, Dict[str, float]] = {}
    if ci_method == "percentile":
        for k in keys:
            lo = float(np.quantile(boot_values[k], 0.025))
            hi = float(np.quantile(boot_values[k], 0.975))
            out[k] = {"point": float(point[k]), "ci_low": float(lo), "ci_high": float(hi)}
        return out

    jack_values = {k: np.empty(labels.size, dtype=float) for k in keys}
    for i in range(labels.size):
        mask = np.ones(labels.size, dtype=bool)
        mask[i] = False
        metric = _mia_metrics_from_labels_scores(
            dataset=dataset,
            labels=labels[mask],
            scores=scores[mask],
            fpr_targets=fpr_targets,
        )
        for k in keys:
            jack_values[k][i] = metric[k]

    for k in keys:
        lo, hi = bca_interval(point[k], boot_values[k], jack_values[k], alpha=0.05)
        out[k] = {"point": float(point[k]), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def _separate_threshold_roc(labels: np.ndarray, scores: np.ndarray, n_quantiles: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    member_scores = scores[labels == 1]
    nonmember_scores = scores[labels == 0]
    if member_scores.size == 0 or nonmember_scores.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    q = np.linspace(0.0, 1.0, n_quantiles)
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(member_scores, q),
                np.quantile(nonmember_scores, q),
                np.array([np.inf, -np.inf], dtype=float),
            ]
        )
    )[::-1]

    member_sorted = np.sort(member_scores)
    nonmember_sorted = np.sort(nonmember_scores)
    member_counts = member_sorted.size - np.searchsorted(member_sorted, thresholds, side="left")
    nonmember_counts = nonmember_sorted.size - np.searchsorted(nonmember_sorted, thresholds, side="left")
    tpr = member_counts.astype(float) / float(member_sorted.size)
    fpr = nonmember_counts.astype(float) / float(nonmember_sorted.size)

    order = np.argsort(fpr, kind="mergesort")
    fpr = fpr[order]
    tpr = tpr[order]
    tpr = np.maximum.accumulate(tpr)

    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    return fpr, tpr


def _mia_metrics_from_labels_scores(
    dataset: str,
    labels: np.ndarray,
    scores: np.ndarray,
    fpr_targets: Sequence[float],
) -> Dict[str, float]:
    if dataset in SEPARATE_THRESHOLD_DATASETS:
        fpr, tpr = _separate_threshold_roc(labels=labels, scores=scores)
    else:
        fpr, tpr = roc_curve_binary(labels, scores)
    out: Dict[str, float] = {"auroc": float(np.trapezoid(tpr, fpr))}
    for target in fpr_targets:
        count = int(np.sum(fpr <= float(target)))
        out[f"tpr@fpr={target:.2f}"] = float(tpr[count - 1]) if count > 0 else 0.0
    return out


def _roc_curve_with_ci(
    dataset: str,
    labels: np.ndarray,
    scores: np.ndarray,
    fpr_grid: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    def _interp_local(lb: np.ndarray, sc: np.ndarray) -> np.ndarray:
        if dataset in SEPARATE_THRESHOLD_DATASETS:
            fpr_local, tpr_local = _separate_threshold_roc(labels=lb, scores=sc)
        else:
            fpr_local, tpr_local = roc_curve_binary(lb, sc)
        fpr_unique, idx = np.unique(fpr_local, return_index=True)
        tpr_unique = tpr_local[idx]
        return np.interp(fpr_grid, fpr_unique, tpr_unique)

    point = _interp_local(labels, scores)
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return {"point": point, "ci_low": np.full_like(point, np.nan), "ci_high": np.full_like(point, np.nan)}
    sampled = stratified_bootstrap_member_indices(labels, n_bootstrap=n_bootstrap, rng=rng)
    curves = np.empty((n_bootstrap, fpr_grid.size), dtype=float)
    for b in range(n_bootstrap):
        idx = sampled[b]
        curves[b] = _interp_local(labels[idx], scores[idx])
    lo = np.quantile(curves, 0.025, axis=0)
    hi = np.quantile(curves, 0.975, axis=0)
    return {"point": point, "ci_low": lo, "ci_high": hi}


def _fmt_cell(point: float, low: float, high: float) -> str:
    if not np.isfinite(point):
        return "--"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{point:.3f}"
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def _write_modality_latex(table_df: pd.DataFrame, left_ds: str, right_ds: str, out_path: Path) -> None:
    rows = []
    for key in METHOD_ORDER:
        r_left = table_df[(table_df["dataset"] == left_ds) & (table_df["method_key"] == key)]
        r_right = table_df[(table_df["dataset"] == right_ds) & (table_df["method_key"] == key)]
        if r_left.empty or r_right.empty:
            continue
        l = r_left.iloc[0]
        r = r_right.iloc[0]
        rows.append(
            " & ".join(
                [
                    METHOD_LABEL[key],
                    _fmt_cell(l["auroc"], l["auroc_ci_low"], l["auroc_ci_high"]),
                    _fmt_cell(l["tpr_at_fpr_0.01"], l["tpr1_ci_low"], l["tpr1_ci_high"]),
                    _fmt_cell(l["tpr_at_fpr_0.05"], l["tpr5_ci_low"], l["tpr5_ci_high"]),
                    _fmt_cell(l["tpr_at_fpr_0.10"], l["tpr10_ci_low"], l["tpr10_ci_high"]),
                    _fmt_cell(r["auroc"], r["auroc_ci_low"], r["auroc_ci_high"]),
                    _fmt_cell(r["tpr_at_fpr_0.01"], r["tpr1_ci_low"], r["tpr1_ci_high"]),
                    _fmt_cell(r["tpr_at_fpr_0.05"], r["tpr5_ci_low"], r["tpr5_ci_high"]),
                    _fmt_cell(r["tpr_at_fpr_0.10"], r["tpr10_ci_low"], r["tpr10_ci_high"]),
                ]
            )
            + r" \\"
        )
    text = "\n".join(rows)
    out_path.write_text(text + "\n", encoding="utf-8")


def _method_to_experiment_name(dataset: str, method_key: str) -> str:
    if method_key == "baseline":
        return f"mia_cpk_{dataset}"
    if method_key.startswith("dp_eps_"):
        eps = method_key.replace("dp_eps_", "")
        return f"mia_dp_eps{eps}_{dataset}"
    raise ValueError(f"No experiment mapping for method: {method_key}")


def _latest_run_metrics_path(repo_root: Path, dataset: str, experiment_name: str) -> Path | None:
    root = repo_root / "outputs" / "inference" / dataset / experiment_name
    if not root.exists():
        return None
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for run_dir in reversed(run_dirs):
        cand = run_dir / "metrics" / "mia_metrics.json"
        if cand.exists():
            return cand
    return None


def _build_reference_validation(summary_df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    check_methods = ["baseline", "dp_eps_5", "dp_eps_10", "dp_eps_20", "dp_eps_50"]
    rows: List[Dict[str, float | str]] = []
    for dataset in DATASET_ORDER:
        for method_key in check_methods:
            local = summary_df[(summary_df["dataset"] == dataset) & (summary_df["method_key"] == method_key)]
            if local.empty:
                continue
            exp_name = _method_to_experiment_name(dataset, method_key)
            metrics_path = _latest_run_metrics_path(repo_root, dataset=dataset, experiment_name=exp_name)
            if metrics_path is None:
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            ref_rows = payload.get("rows", [])
            if not ref_rows:
                continue
            # DP runs have one row at acceptance_rate=1, CPK has several; baseline = acceptance_rate=1.
            ref = None
            for row in ref_rows:
                if abs(float(row.get("acceptance_rate", 1.0)) - 1.0) < 1e-9:
                    ref = row
                    break
            if ref is None:
                ref = ref_rows[0]
            loc = local.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "method_key": method_key,
                    "reference_metrics_path": str(metrics_path),
                    "our_auroc": float(loc["auroc"]),
                    "ref_auroc": float(ref.get("auroc", np.nan)),
                    "abs_diff_auroc": float(abs(float(loc["auroc"]) - float(ref.get("auroc", np.nan)))),
                    "our_tpr1": float(loc["tpr_at_fpr_0.01"]),
                    "ref_tpr1": float(ref.get("tpr_at_fpr_0.01", np.nan)),
                    "abs_diff_tpr1": float(abs(float(loc["tpr_at_fpr_0.01"]) - float(ref.get("tpr_at_fpr_0.01", np.nan)))),
                    "our_tpr5": float(loc["tpr_at_fpr_0.05"]),
                    "ref_tpr5": float(ref.get("tpr_at_fpr_0.05", np.nan)),
                    "abs_diff_tpr5": float(abs(float(loc["tpr_at_fpr_0.05"]) - float(ref.get("tpr_at_fpr_0.05", np.nan)))),
                    "our_tpr10": float(loc["tpr_at_fpr_0.10"]),
                    "ref_tpr10": float(ref.get("tpr_at_fpr_0.10", np.nan)),
                    "abs_diff_tpr10": float(abs(float(loc["tpr_at_fpr_0.10"]) - float(ref.get("tpr_at_fpr_0.10", np.nan)))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_bootstrap = min(int(args.n_bootstrap), 120)
        args.roc_grid_points = min(int(args.roc_grid_points), 140)
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
    mia_df = df[df["run_name"].astype(str).str.startswith("mia_")].copy()

    # ---- Table metrics + scalar CIs ----
    summary_rows: List[Dict[str, float]] = []
    for dataset in DATASET_ORDER:
        per_dataset = mia_df[mia_df["dataset"] == dataset].copy()
        for method_key in METHOD_ORDER:
            subset = _method_subset(per_dataset, method_key=method_key)
            if subset.empty:
                continue
            labels, scores = _labels_scores(subset)
            metrics = _bootstrap_metric_bundle(
                dataset=dataset,
                labels=labels,
                scores=scores,
                fpr_targets=FPR_TARGETS,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
                ci_method=args.ci_method,
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": pretty_dataset(dataset),
                    "method_key": method_key,
                    "method_label": METHOD_LABEL[method_key],
                    "num_samples": int(subset.shape[0]),
                    "auroc": metrics["auroc"]["point"],
                    "auroc_ci_low": metrics["auroc"]["ci_low"],
                    "auroc_ci_high": metrics["auroc"]["ci_high"],
                    "tpr_at_fpr_0.01": metrics["tpr@fpr=0.01"]["point"],
                    "tpr1_ci_low": metrics["tpr@fpr=0.01"]["ci_low"],
                    "tpr1_ci_high": metrics["tpr@fpr=0.01"]["ci_high"],
                    "tpr_at_fpr_0.05": metrics["tpr@fpr=0.05"]["point"],
                    "tpr5_ci_low": metrics["tpr@fpr=0.05"]["ci_low"],
                    "tpr5_ci_high": metrics["tpr@fpr=0.05"]["ci_high"],
                    "tpr_at_fpr_0.10": metrics["tpr@fpr=0.10"]["point"],
                    "tpr10_ci_low": metrics["tpr@fpr=0.10"]["ci_low"],
                    "tpr10_ci_high": metrics["tpr@fpr=0.10"]["ci_high"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tab_dir / "mia_summary_long.csv", index=False)

    image_df = summary_df[summary_df["dataset"].isin(["laion_mi", "pokemon"])].copy()
    text_df = summary_df[summary_df["dataset"].isin(["automathtext", "writingprompts"])].copy()
    image_df.to_csv(tab_dir / "mia_image_table.csv", index=False)
    text_df.to_csv(tab_dir / "mia_text_table.csv", index=False)
    _write_modality_latex(image_df, "laion_mi", "pokemon", tab_dir / "mia_image_table_rows.tex")
    _write_modality_latex(text_df, "automathtext", "writingprompts", tab_dir / "mia_text_table_rows.tex")

    # ---- Figure A: 2x2 ROC ----
    roc_grid = np.linspace(0.0, args.fpr_max, args.roc_grid_points)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0.22, hspace=0.30)

    roc_export_rows: List[Dict[str, float]] = []
    for panel_idx, dataset in enumerate(DATASET_ORDER):
        r, c = divmod(panel_idx, 2)
        ax = axes[r, c]
        ddf = mia_df[mia_df["dataset"] == dataset].copy()
        for method_key in ROC_METHODS:
            subset = _method_subset(ddf, method_key=method_key)
            if subset.empty:
                continue
            labels, scores = _labels_scores(subset)
            curve = _roc_curve_with_ci(
                dataset=dataset,
                labels=labels,
                scores=scores,
                fpr_grid=roc_grid,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            st = STYLE[method_key]
            ax.plot(100.0 * roc_grid, 100.0 * curve["point"], color=st["color"], lw=st["lw"], ls=st["ls"], label=METHOD_LABEL[method_key])
            ax.fill_between(
                100.0 * roc_grid,
                100.0 * curve["ci_low"],
                100.0 * curve["ci_high"],
                color=st["color"],
                alpha=0.18,
                linewidth=0,
            )
            for x, y, lo, hi in zip(roc_grid, curve["point"], curve["ci_low"], curve["ci_high"]):
                roc_export_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset(dataset),
                        "method_key": method_key,
                        "fpr": float(x),
                        "tpr": float(y),
                        "tpr_ci_low": float(lo),
                        "tpr_ci_high": float(hi),
                    }
                )

        ax.plot([0.0, 100.0 * args.fpr_max], [0.0, 100.0 * args.fpr_max], color="gray", ls=":", lw=1.8, label="Random attacker")
        ax.set_xlim(0.0, 100.0 * args.fpr_max)
        ax.set_ylim(0.0, 100.0)
        ax.set_title(pretty_dataset(dataset), fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("FPR (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("TPR (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.25)
        roc_legend_loc = "upper left" if panel_idx == 3 else "center right"
        ax.legend(loc=roc_legend_loc, fontsize=LEGEND_FONTSIZE, frameon=True)

    fig.savefig(fig_dir / "mia_roc_2x2.png", dpi=350, bbox_inches="tight")
    if not args.skip_pdf:
        fig.savefig(fig_dir / "mia_roc_2x2.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(roc_export_rows).to_csv(inter_dir / "mia_roc_points_with_ci.csv", index=False)

    # ---- Figure A2: 2x2 ROC (all epsilon-delta DP models) ----
    fig_dp, axes_dp = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=False, sharey=True)
    fig_dp.subplots_adjust(wspace=0.22, hspace=0.30)
    roc_dp_export_rows: List[Dict[str, float]] = []
    for panel_idx, dataset in enumerate(DATASET_ORDER):
        r, c = divmod(panel_idx, 2)
        ax = axes_dp[r, c]
        ddf = mia_df[mia_df["dataset"] == dataset].copy()
        for method_key in ROC_DP_METHODS:
            subset = _method_subset(ddf, method_key=method_key)
            if subset.empty:
                continue
            labels, scores = _labels_scores(subset)
            curve = _roc_curve_with_ci(
                dataset=dataset,
                labels=labels,
                scores=scores,
                fpr_grid=roc_grid,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            st = STYLE[method_key]
            ax.plot(100.0 * roc_grid, 100.0 * curve["point"], color=st["color"], lw=st["lw"], ls=st["ls"], label=METHOD_LABEL[method_key])
            ax.fill_between(
                100.0 * roc_grid,
                100.0 * curve["ci_low"],
                100.0 * curve["ci_high"],
                color=st["color"],
                alpha=0.18,
                linewidth=0,
            )
            for x, y, lo, hi in zip(roc_grid, curve["point"], curve["ci_low"], curve["ci_high"]):
                roc_dp_export_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset(dataset),
                        "method_key": method_key,
                        "fpr": float(x),
                        "tpr": float(y),
                        "tpr_ci_low": float(lo),
                        "tpr_ci_high": float(hi),
                    }
                )

        ax.plot([0.0, 100.0 * args.fpr_max], [0.0, 100.0 * args.fpr_max], color="gray", ls=":", lw=1.8, label="Random attacker")
        ax.set_xlim(0.0, 100.0 * args.fpr_max)
        ax.set_ylim(0.0, 100.0)
        ax.set_title(pretty_dataset(dataset), fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("FPR (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("TPR (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.25)
        roc_dp_legend_loc = "upper left" if panel_idx == 3 else "center right"
        ax.legend(loc=roc_dp_legend_loc, fontsize=ROC_DP_LEGEND_FONTSIZE, frameon=True)

    fig_dp.savefig(fig_dir / "mia_roc_all_eps_delta_2x2.png", dpi=350, bbox_inches="tight")
    if not args.skip_pdf:
        fig_dp.savefig(fig_dir / "mia_roc_all_eps_delta_2x2.pdf", bbox_inches="tight")
    plt.close(fig_dp)
    pd.DataFrame(roc_dp_export_rows).to_csv(inter_dir / "mia_roc_all_eps_delta_points_with_ci.csv", index=False)

    # ---- Figure B: TPR@FPR{1,5,10} as function of alpha_k ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    fig2.subplots_adjust(wspace=0.20, hspace=0.26)
    alpha_metric_targets = [0.01, 0.05, 0.10]
    alpha_metric_styles = {
        0.01: {"color": STYLE["cpk_alpha_0.25"]["color"], "label": r"CP-$k$, 1% FPR"},
        0.05: {"color": STYLE["cpk_alpha_0.50"]["color"], "label": r"CP-$k$, 5% FPR"},
        0.10: {"color": STYLE["cpk_alpha_0.75"]["color"], "label": r"CP-$k$, 10% FPR"},
    }

    alpha_rows: List[Dict[str, float]] = []
    for panel_idx, dataset in enumerate(DATASET_ORDER):
        r, c = divmod(panel_idx, 2)
        ax = axes2[r, c]
        per_dataset_all = mia_df[mia_df["dataset"] == dataset].copy()
        for fpr_target in alpha_metric_targets:
            xs: List[float] = []
            ys: List[float] = []
            ylo: List[float] = []
            yhi: List[float] = []
            for alpha in ALPHA_GRID:
                # Reuse same thresholding rule as the ROC/table path.
                subset = _method_subset(per_dataset_all, method_key=f"cpk_alpha_{alpha:.2f}")
                if subset.empty:
                    continue
                labels, scores = _labels_scores(subset)
                metric = _bootstrap_metric_bundle(
                    dataset=dataset,
                    labels=labels,
                    scores=scores,
                    fpr_targets=[fpr_target],
                    n_bootstrap=args.n_bootstrap,
                    rng=rng,
                    ci_method=args.ci_method,
                )[f"tpr@fpr={fpr_target:.2f}"]
                xs.append(alpha)
                ys.append(metric["point"])
                ylo.append(metric["ci_low"] if np.isfinite(metric["ci_low"]) else metric["point"])
                yhi.append(metric["ci_high"] if np.isfinite(metric["ci_high"]) else metric["point"])
                alpha_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": pretty_dataset(dataset),
                        "alpha_k": alpha,
                        "fpr_target": fpr_target,
                        "tpr": metric["point"],
                        "ci_low": metric["ci_low"],
                        "ci_high": metric["ci_high"],
                    }
                )
            if xs:
                st = alpha_metric_styles[fpr_target]
                x_arr = np.array(xs, dtype=float)
                y_arr = np.array(ys, dtype=float)
                lo_arr = np.array(ylo, dtype=float)
                hi_arr = np.array(yhi, dtype=float)
                ax.plot(
                    xs,
                    y_arr * 100.0,
                    color=st["color"],
                    lw=2.0,
                    label=st["label"],
                )
                ax.fill_between(
                    x_arr,
                    lo_arr * 100.0,
                    hi_arr * 100.0,
                    color=st["color"],
                    alpha=0.18,
                    linewidth=0,
                )
        ax.set_title(pretty_dataset(dataset), fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel(r"Acceptance rate $\alpha_k$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("TPR (%)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(0.10, 1.00)
        ax.margins(x=0.0)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.25)
        alpha_legend_loc = "upper right" if panel_idx == 3 else "lower right"
        ax.legend(loc=alpha_legend_loc, fontsize=LEGEND_FONTSIZE, frameon=True)

    fig2.savefig(fig_dir / "mia_tpratfpr1_5_10_vs_alpha_2x2.png", dpi=350, bbox_inches="tight")
    if not args.skip_pdf:
        fig2.savefig(fig_dir / "mia_tpratfpr1_5_10_vs_alpha_2x2.pdf", bbox_inches="tight")
    plt.close(fig2)
    pd.DataFrame(alpha_rows).to_csv(inter_dir / "mia_tpratfpr1_5_10_vs_alpha_with_ci.csv", index=False)

    # ---- Sanity checks + metadata ----
    warnings: List[str] = []
    for _, row in summary_df.iterrows():
        for k in ("auroc", "tpr_at_fpr_0.01", "tpr_at_fpr_0.05", "tpr_at_fpr_0.10"):
            v = float(row[k])
            if (v < -1e-6) or (v > 1.0 + 1e-6):
                warnings.append(f"Out-of-range metric {k}={v:.5f} for {row['dataset']} / {row['method_key']}")

    validation_df = pd.DataFrame()
    validation_path = inter_dir / "mia_reference_validation.csv"
    if not args.skip_validation:
        validation_df = _build_reference_validation(summary_df=summary_df, repo_root=repo_root)
        if not validation_df.empty:
            validation_df.to_csv(validation_path, index=False)
            for _, row in validation_df.iterrows():
                if float(row["abs_diff_auroc"]) > 1e-3:
                    warnings.append(
                        "Reference mismatch (AUROC) "
                        f"{row['dataset']} {row['method_key']} diff={row['abs_diff_auroc']:.6f}"
                    )

    json_dump(
        output_dir / "mia_analysis_manifest.json",
        {
            "input_parquet": str(args.input_parquet),
            "input_parquet_sha256": file_sha256(args.input_parquet),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "ci_method": args.ci_method,
            "quick_mode": bool(args.quick),
            "fpr_targets": list(FPR_TARGETS),
            "fpr_max": float(args.fpr_max),
            "roc_methods": ROC_METHODS,
            "table_methods": METHOD_ORDER,
            "separate_threshold_datasets": sorted(SEPARATE_THRESHOLD_DATASETS),
            "warnings": warnings,
            "outputs": {
                "figures": [
                    str(fig_dir / "mia_roc_2x2.png"),
                    str(fig_dir / "mia_roc_all_eps_delta_2x2.png"),
                    str(fig_dir / "mia_tpratfpr1_5_10_vs_alpha_2x2.png"),
                ],
                "tables": [
                    str(tab_dir / "mia_summary_long.csv"),
                    str(tab_dir / "mia_image_table.csv"),
                    str(tab_dir / "mia_text_table.csv"),
                    str(tab_dir / "mia_image_table_rows.tex"),
                    str(tab_dir / "mia_text_table_rows.tex"),
                ],
                "intermediates": [
                    str(inter_dir / "mia_roc_points_with_ci.csv"),
                    str(inter_dir / "mia_roc_all_eps_delta_points_with_ci.csv"),
                    str(inter_dir / "mia_tpratfpr1_5_10_vs_alpha_with_ci.csv"),
                ],
            },
        },
    )
    if not args.skip_pdf or not args.skip_validation:
        manifest_path = output_dir / "mia_analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.skip_pdf:
            manifest["outputs"]["figures"].extend(
                [
                    str(fig_dir / "mia_roc_2x2.pdf"),
                    str(fig_dir / "mia_roc_all_eps_delta_2x2.pdf"),
                    str(fig_dir / "mia_tpratfpr1_5_10_vs_alpha_2x2.pdf"),
                ]
            )
        if not args.skip_validation:
            manifest["outputs"]["intermediates"].append(str(validation_path))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"MIA publication outputs written to: {output_dir}")


if __name__ == "__main__":
    main()

