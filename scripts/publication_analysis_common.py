#!/usr/bin/env python
"""Shared utilities for publication-facing MIA/DRA analyses."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


NORMAL = NormalDist()

DATASET_LABELS = {
    "automathtext": "MathAbstracts",
    "writingprompts": "WritingPrompts",
    "pokemon": "Pokemon",
    "laion_mi": "LAION-MI",
}

MIA_PANEL_ORDER = ["MathAbstracts", "Pokemon", "WritingPrompts", "LAION-MI"]

METHOD_LABELS = {
    "baseline": "Baseline",
    "cpk_alpha_0.25": r"CP-$k$, $\alpha_k=0.25$",
    "cpk_alpha_0.50": r"CP-$k$, $\alpha_k=0.50$",
    "cpk_alpha_0.75": r"CP-$k$, $\alpha_k=0.75$",
    "cpk_alpha_1.00": r"CP-$k$, $\alpha_k=1.00$",
    "dp_eps_5": r"$(5,10^{-5})$-DP",
    "dp_eps_10": r"$(10,10^{-5})$-DP",
    "dp_eps_20": r"$(20,10^{-5})$-DP",
    "dp_eps_50": r"$(50,10^{-5})$-DP",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    kind: str  # "cpk" | "dp"
    run_name: str
    alpha: Optional[float] = None
    eps: Optional[float] = None

    @property
    def label(self) -> str:
        if self.key in METHOD_LABELS:
            return METHOD_LABELS[self.key]
        return self.key


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pretty_dataset(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def parse_dp_eps(run_name: str) -> Optional[float]:
    match = re.search(r"eps([0-9]+(?:\\.[0-9]+)?)", run_name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def cpk_threshold(values: np.ndarray, alpha: float) -> float:
    if alpha >= 1.0:
        return float("inf")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, alpha))


def apply_cpk_acceptance(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if alpha >= 1.0:
        return df.copy()
    if "cpk_stat" in df.columns:
        cpk = pd.to_numeric(df["cpk_stat"], errors="coerce")
    else:
        cpk = pd.to_numeric(df.get("cpk_logratio_max"), errors="coerce")
    thr = cpk_threshold(cpk.to_numpy(dtype=float), alpha=alpha)
    if not math.isfinite(thr):
        return df.iloc[0:0].copy()
    keep = np.isfinite(cpk.to_numpy(dtype=float)) & (cpk.to_numpy(dtype=float) <= thr)
    return df.loc[keep].copy()


def orient_scores_for_member_attack(scores: np.ndarray, direction: str) -> np.ndarray:
    if direction == "higher_is_member":
        return scores
    if direction == "lower_is_member":
        return -scores
    raise ValueError(f"Unsupported score direction: {direction}")


def roc_curve_binary(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int32)
    if labels.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())
    if positives <= 0 or negatives <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    scores_sorted = scores[order]

    distinct = np.where(np.diff(scores_sorted) != 0)[0]
    thresh_idx = np.r_[distinct, labels_sorted.size - 1]
    tp = np.cumsum(labels_sorted == 1)[thresh_idx].astype(float)
    fp = np.cumsum(labels_sorted == 0)[thresh_idx].astype(float)
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    return fpr, tpr


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, fpr_target: float) -> float:
    count = int(np.sum(fpr <= float(fpr_target)))
    if count <= 0:
        return 0.0
    return float(tpr[count - 1])


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr))


def mia_metrics_from_labels_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    fpr_targets: Sequence[float],
) -> Dict[str, float]:
    fpr, tpr = roc_curve_binary(labels, scores)
    out: Dict[str, float] = {"auroc": auc_from_roc(fpr, tpr)}
    for target in fpr_targets:
        out[f"tpr@fpr={target:.2f}"] = tpr_at_fpr(fpr, tpr, target)
    return out


def stratified_bootstrap_member_indices(
    labels: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = labels.astype(np.int32)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    if idx_pos.size == 0 or idx_neg.size == 0:
        raise ValueError("Both member and non-member strata are required for stratified bootstrap.")
    sampled = np.empty((n_bootstrap, labels.size), dtype=np.int64)
    for b in range(n_bootstrap):
        pos_draw = rng.choice(idx_pos, size=idx_pos.size, replace=True)
        neg_draw = rng.choice(idx_neg, size=idx_neg.size, replace=True)
        sampled[b, : idx_pos.size] = pos_draw
        sampled[b, idx_pos.size :] = neg_draw
    return sampled


def bca_interval(
    theta_hat: float,
    bootstrap_values: np.ndarray,
    jackknife_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    boot = bootstrap_values[np.isfinite(bootstrap_values)]
    jack = jackknife_values[np.isfinite(jackknife_values)]
    if boot.size < 10 or jack.size < 3:
        lo = float(np.quantile(boot, alpha / 2.0)) if boot.size else float("nan")
        hi = float(np.quantile(boot, 1.0 - alpha / 2.0)) if boot.size else float("nan")
        return lo, hi

    proportion_less = np.clip(float(np.mean(boot < theta_hat)), 1e-6, 1.0 - 1e-6)
    z0 = NORMAL.inv_cdf(proportion_less)

    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(np.sum((jack_mean - jack) ** 2))
    if den <= 0:
        a = 0.0
    else:
        a = num / (6.0 * (den ** 1.5))

    z_lo = NORMAL.inv_cdf(alpha / 2.0)
    z_hi = NORMAL.inv_cdf(1.0 - alpha / 2.0)

    def _adj(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-8:
            return float("nan")
        return NORMAL.cdf(z0 + (z0 + z) / denom)

    p_lo = _adj(z_lo)
    p_hi = _adj(z_hi)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi):
        p_lo = alpha / 2.0
        p_hi = 1.0 - alpha / 2.0
    p_lo = float(np.clip(p_lo, 0.0, 1.0))
    p_hi = float(np.clip(p_hi, 0.0, 1.0))
    lo = float(np.quantile(boot, min(p_lo, p_hi)))
    hi = float(np.quantile(boot, max(p_lo, p_hi)))
    return lo, hi


def bootstrap_scalar_bca(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    theta_hat = float(statistic_fn(values))
    n = values.shape[0]
    if n <= 1:
        return theta_hat, float("nan"), float("nan")

    boot_vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boot_vals[i] = float(statistic_fn(values[idx]))

    jack_vals = np.empty(n, dtype=float)
    for i in range(n):
        jack = np.delete(values, i)
        jack_vals[i] = float(statistic_fn(jack)) if jack.size else float("nan")

    lo, hi = bca_interval(theta_hat, boot_vals, jack_vals, alpha=alpha)
    return theta_hat, lo, hi


def bootstrap_scalar_percentile(
    values: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    theta_hat = float(statistic_fn(values))
    n = values.shape[0]
    if n <= 1:
        return theta_hat, float("nan"), float("nan")
    boot_vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boot_vals[i] = float(statistic_fn(values[idx]))
    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))
    return theta_hat, lo, hi


def method_hash(inputs: Sequence[Path]) -> str:
    payload = "|".join(str(p.resolve()) for p in inputs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

