#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


MIA_DATASET_LABELS = {
    "automathtext": "MathAbstracts",
    "writingprompts": "WritingPrompts",
    "laion_mi": "LAION-MI",
    "pokemon": "Pokemon",
}

PANEL_ORDER = ["MathAbstracts", "Pokemon", "WritingPrompts", "LAION-MI"]

STYLE = {
    "baseline_alpha1": {"color": "black", "lw": 2.6, "ls": "-"},
    "cpk_alpha_0.25": {"color": "#1f77b4", "lw": 2.2, "ls": "-"},
    "dp_eps_5": {"color": "#d62728", "lw": 2.1, "ls": "-"},
    "dp_eps_10": {"color": "#ff7f0e", "lw": 2.1, "ls": "-"},
    "dp_eps_20": {"color": "#2ca02c", "lw": 2.1, "ls": "-"},
    "dp_eps_50": {"color": "#9467bd", "lw": 2.1, "ls": "-"},
}


def resolve_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "outputs" / "inference").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root containing outputs/inference.")


def latest_mia_runs(inference_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for manifest_path in inference_root.glob("*/*/*/workflow_manifest.json"):
        dataset, experiment, run_id = manifest_path.relative_to(inference_root).parts[:3]
        if not experiment.startswith("mia_"):
            continue
        rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "run_id": run_id,
                "manifest_path": manifest_path,
            }
        )

    index_df = pd.DataFrame(rows)
    if index_df.empty:
        return index_df

    return (
        index_df.sort_values("run_id")
        .groupby(["dataset", "experiment"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def method_key_and_label(experiment: str, acceptance_rate: float) -> tuple[str, str]:
    if experiment.startswith("mia_cpk_"):
        alpha_k = float(acceptance_rate)
        if abs(alpha_k - 1.0) < 1e-12:
            return "baseline_alpha1", r"Baseline (CP-$k$, $\alpha_k=1$)"
        return f"cpk_alpha_{alpha_k:g}", rf"CP-$k$ ($\alpha_k={alpha_k:g}$)"

    match = re.search(r"mia_dp_eps([0-9]+(?:\.[0-9]+)?)_", experiment)
    if match:
        epsilon = match.group(1)
        return f"dp_eps_{epsilon}", rf"$\epsilon={epsilon}$-DP"

    return experiment, experiment


def load_cache_rows(cache_path: Path) -> tuple[list[dict[str, object]], str]:
    cache = torch.load(cache_path, map_location="cpu")
    rows = [dict(r) for r in cache.get("data", {}).get("records", [])]
    direction = str(cache.get("data", {}).get("mia", {}).get("score_direction", "higher_is_member"))
    return rows, direction


def roc_from_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int32)
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives <= 0 or negatives <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    distinct_idx = np.where(np.diff(scores_sorted) != 0)[0]
    threshold_idx = np.r_[distinct_idx, labels_sorted.size - 1]
    tps = np.cumsum(labels_sorted == 1)[threshold_idx].astype(float)
    fps = np.cumsum(labels_sorted == 0)[threshold_idx].astype(float)
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    return fpr, tpr


def cpk_subset(rows: list[dict[str, object]], acceptance_rate: float) -> list[dict[str, object]]:
    if acceptance_rate >= 1.0:
        return rows
    cpk_values = np.array(
        [float(r.get("cpk_stat", r.get("cpk_logratio_max", np.nan))) for r in rows],
        dtype=float,
    )
    finite = np.isfinite(cpk_values)
    if not finite.any():
        return []
    threshold = float(np.quantile(cpk_values[finite], acceptance_rate))
    subset: list[dict[str, object]] = []
    for row in rows:
        value = float(row.get("cpk_stat", row.get("cpk_logratio_max", np.nan)))
        if np.isfinite(value) and value <= threshold:
            subset.append(row)
    return subset


def rows_to_curve(rows: list[dict[str, object]], score_direction: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(r.get("membership_label", 0)) for r in rows], dtype=np.int32)
    scores = np.array([float(r.get("mia_score", 0.0)) for r in rows], dtype=float)
    if score_direction != "higher_is_member":
        scores = -scores
    return roc_from_scores(labels, scores)


def build_latest_mia_roc_df(
    inference_root: Path,
    repo_root: Path,
    methods_to_keep: list[str],
    fpr_max: float,
) -> pd.DataFrame:
    latest_runs_df = latest_mia_runs(inference_root)
    rows_out: list[dict[str, object]] = []

    for _, run in latest_runs_df.iterrows():
        manifest = json.loads(Path(run["manifest_path"]).read_text())
        experiment = str(run["experiment"])
        dataset = str(run["dataset"])
        dataset_label = MIA_DATASET_LABELS.get(dataset, dataset)

        if experiment.startswith("mia_cpk_"):
            cache_rel = manifest.get("cpk_cache") or manifest.get("mia_cache")
            acceptance_rates = [1.0, 0.25]
        else:
            cache_rel = manifest.get("mia_cache")
            acceptance_rates = [1.0]

        if not cache_rel:
            continue
        cache_path = (repo_root / cache_rel).resolve()
        if not cache_path.exists():
            continue

        cache_rows, score_direction = load_cache_rows(cache_path)
        for alpha_k in acceptance_rates:
            method_key, method_label = method_key_and_label(experiment, alpha_k)
            if method_key not in methods_to_keep:
                continue
            subset = cpk_subset(cache_rows, alpha_k) if experiment.startswith("mia_cpk_") else cache_rows
            if not subset:
                continue
            fpr, tpr = rows_to_curve(subset, score_direction)
            mask = fpr <= fpr_max
            for x, y in zip(fpr[mask], tpr[mask]):
                rows_out.append(
                    {
                        "dataset": dataset,
                        "dataset_label": dataset_label,
                        "experiment": experiment,
                        "method_key": method_key,
                        "method_label": method_label,
                        "alpha_k": alpha_k,
                        "fpr": float(x),
                        "tpr": float(y),
                    }
                )

    return pd.DataFrame(rows_out)


def smooth_dataset_curve(dataset_df: pd.DataFrame, fpr_max: float, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    d = dataset_df.sort_values("fpr").groupby("fpr", as_index=False)["tpr"].max()
    x = np.linspace(0.0, fpr_max, n_grid)
    y = np.interp(x, d["fpr"].to_numpy(), d["tpr"].to_numpy())
    return x, y


def aggregate_roc_over_datasets(roc_df: pd.DataFrame, methods_to_keep: list[str], fpr_max: float, n_grid: int) -> pd.DataFrame:
    fpr_grid = np.linspace(0.0, fpr_max, n_grid)
    rows: list[dict[str, object]] = []

    for method_key in methods_to_keep:
        per_method = roc_df[roc_df["method_key"] == method_key]
        if per_method.empty:
            continue

        method_label = per_method["method_label"].iloc[0]
        curves = []
        for dataset_label in per_method["dataset_label"].drop_duplicates().tolist():
            dataset_df = per_method[per_method["dataset_label"] == dataset_label].sort_values("fpr")
            _, y = smooth_dataset_curve(dataset_df, fpr_max=fpr_max, n_grid=n_grid)
            curves.append(y)

        mean_tpr = np.vstack(curves).mean(axis=0)
        for fpr, tpr in zip(fpr_grid, mean_tpr):
            rows.append({"fpr": fpr, "tpr": tpr, "method_key": method_key, "method_label": method_label})

    return pd.DataFrame(rows)


def plot_aggregate_mia_roc(agg_df: pd.DataFrame, methods_to_plot: list[str], fpr_max: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for key in methods_to_plot:
        method_df = agg_df[agg_df["method_key"] == key]
        if method_df.empty:
            continue
        st = STYLE.get(key, {"color": None, "lw": 2.0, "ls": "-"})
        ax.plot(
            100.0 * method_df["fpr"],
            100.0 * method_df["tpr"],
            label=method_df["method_label"].iloc[0],
            color=st["color"],
            linewidth=st["lw"],
            linestyle=st["ls"],
        )

    ax.plot([0, 100.0 * fpr_max], [0, 100.0 * fpr_max], linestyle=":", color="gray", linewidth=1.6, label="Random attacker")
    ax.set_xlim(0, 100.0 * fpr_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel("FPR (%)")
    ax.set_ylabel("TPR (%)")
    ax.set_title("MIA ROC (latest runs, averaged over datasets)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    return fig


def plot_mia_roc_grid(roc_df: pd.DataFrame, methods_to_plot: list[str], fpr_max: float, n_grid: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0.25, hspace=0.35)

    for i, dataset_label in enumerate(PANEL_ORDER):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        panel_df = roc_df[roc_df["dataset_label"] == dataset_label]

        for key in methods_to_plot:
            method_df = panel_df[panel_df["method_key"] == key]
            if method_df.empty:
                continue
            x, y = smooth_dataset_curve(method_df, fpr_max=fpr_max, n_grid=n_grid)
            st = STYLE.get(key, {"color": None, "lw": 2.0, "ls": "-"})
            ax.plot(
                100.0 * x,
                100.0 * y,
                label=method_df["method_label"].iloc[0],
                color=st["color"],
                linewidth=st["lw"],
                linestyle=st["ls"],
            )

        ax.plot(
            [0, 100.0 * fpr_max],
            [0, 100.0 * fpr_max],
            linestyle=":",
            color="gray",
            linewidth=1.6,
            label="Random attacker",
        )
        ax.set_xlim(0, 100.0 * fpr_max)
        ax.set_ylim(0, 100)
        ax.set_xlabel("FPR (%)", fontsize=12)
        ax.set_ylabel("TPR (%)", fontsize=12)
        ax.set_title(dataset_label, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MIA ROC from latest experiment runs.")
    parser.add_argument("--inference-root", type=Path, default=None, help="Path to outputs/inference.")
    parser.add_argument("--fpr-max", type=float, default=0.10, help="Max FPR in probability units (0.10 = 10%%).")
    parser.add_argument("--grid-points", type=int, default=400, help="Interpolation grid size for smoothing.")
    parser.add_argument(
        "--layout",
        choices=["grid", "aggregate"],
        default="grid",
        help="Figure layout: 'grid' for 2x2 dataset sub-figures (reference-style), or 'aggregate' for one averaged panel.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline_alpha1", "dp_eps_10", "cpk_alpha_0.25"],
        help="Method keys to include in plot.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Output image path (default depends on selected --layout).",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV for plotted ROC points.")
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path.cwd().resolve())
    inference_root = args.inference_root.resolve() if args.inference_root else repo_root / "outputs" / "inference"
    output_figure = (
        args.output_figure.resolve()
        if args.output_figure
        else (repo_root / "outputs" / "inference" / f"mia_roc_latest_{args.layout}.png")
    )
    output_csv = args.output_csv.resolve() if args.output_csv else None

    roc_df = build_latest_mia_roc_df(
        inference_root=inference_root,
        repo_root=repo_root,
        methods_to_keep=args.methods,
        fpr_max=args.fpr_max,
    )
    if roc_df.empty:
        raise RuntimeError(f"No raw MIA cache-based ROC data found under: {inference_root}")

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "aggregate":
        agg_df = aggregate_roc_over_datasets(roc_df, args.methods, fpr_max=args.fpr_max, n_grid=args.grid_points)
        if agg_df.empty:
            raise RuntimeError("No curves available after method filtering. Check --methods keys.")
        fig = plot_aggregate_mia_roc(agg_df, args.methods, fpr_max=args.fpr_max)
    else:
        fig = plot_mia_roc_grid(roc_df, args.methods, fpr_max=args.fpr_max, n_grid=args.grid_points)

    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_figure}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.layout == "aggregate":
            agg_df.to_csv(output_csv, index=False)
            print(f"Saved aggregated table: {output_csv}")
        else:
            roc_df.to_csv(output_csv, index=False)
            print(f"Saved latest per-dataset ROC table: {output_csv}")


if __name__ == "__main__":
    main()
