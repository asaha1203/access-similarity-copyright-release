#!/usr/bin/env python
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from copyright_pipeline.inference_cache import load_cache
from copyright_pipeline.inference_common import (
    dump_csv,
    maybe_orient_scores,
    safe_quantile,
    simple_auroc,
    tpr_at_fpr,
    write_json,
)


def _parse_quantiles(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        q = float(token)
        if q < 0.0 or q > 1.0:
            raise ValueError(f"Quantiles must be in [0,1], got {q}")
        values.append(q)
    if not values:
        raise ValueError("No quantiles provided.")
    return sorted(set(values))


def _collect_group_scores(rows: List[Dict[str, Any]], score_key: str) -> Dict[str, List[float]]:
    grouped = {"members": [], "nonmembers": []}
    for row in rows:
        label = int(row.get("membership_label", 0))
        raw = row.get(score_key)
        if raw is None and score_key == "cpk_stat":
            raw = row.get("cpk_logratio_max")
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        if label == 1:
            grouped["members"].append(value)
        else:
            grouped["nonmembers"].append(value)
    return grouped


def _row_filter_score(row: Dict[str, Any], score_key: str) -> float:
    raw = row.get(score_key)
    if raw is None and score_key == "cpk_stat":
        raw = row.get("cpk_logratio_max")
    return float(raw) if raw is not None else float("nan")


def _summary_row(group_name: str, values: List[float], quantiles: List[float]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "group": group_name,
        "count": len(values),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "mean": float(sum(values) / len(values)) if values else None,
    }
    for q in quantiles:
        key = f"q_{q:.4f}".rstrip("0").rstrip(".")
        row[key] = float(safe_quantile(values, q)) if values else None
    return row


def _accept_by_threshold(value: float, threshold: float, filter_direction: str) -> bool:
    if filter_direction == "lower":
        return value <= threshold
    if filter_direction == "higher":
        return value >= threshold
    raise ValueError(f"Unsupported filter_direction: {filter_direction}")


def _threshold_for_quantile(values: List[float], quantile: float, filter_direction: str) -> float:
    if filter_direction == "lower":
        return float(safe_quantile(values, quantile))
    if filter_direction == "higher":
        return float(safe_quantile(values, 1.0 - quantile))
    raise ValueError(f"Unsupported filter_direction: {filter_direction}")


def _compute_auroc_rows(
    rows: List[Dict[str, Any]],
    score_key: str,
    quantiles: List[float],
    filter_direction: str,
    mia_direction: str,
    fpr_grid: List[float],
) -> List[Dict[str, Any]]:
    member_filter_scores = [
        _row_filter_score(row, score_key)
        for row in rows
        if int(row.get("membership_label", 0)) == 1 and math.isfinite(_row_filter_score(row, score_key))
    ]
    nonmember_filter_scores = [
        _row_filter_score(row, score_key)
        for row in rows
        if int(row.get("membership_label", 0)) == 0 and math.isfinite(_row_filter_score(row, score_key))
    ]

    total_members = len(member_filter_scores)
    total_nonmembers = len(nonmember_filter_scores)
    results: List[Dict[str, Any]] = []
    for q in quantiles:
        member_threshold = _threshold_for_quantile(member_filter_scores, q, filter_direction) if member_filter_scores else None
        nonmember_threshold = (
            _threshold_for_quantile(nonmember_filter_scores, q, filter_direction) if nonmember_filter_scores else None
        )
        selected_scores: List[float] = []
        selected_labels: List[int] = []
        selected_members = 0
        selected_nonmembers = 0
        for row in rows:
            label = int(row.get("membership_label", 0))
            filt_score = _row_filter_score(row, score_key)
            raw_mia = row.get("mia_score")
            if not math.isfinite(filt_score):
                continue
            if raw_mia is None:
                continue
            mia_value = float(raw_mia)
            if not math.isfinite(mia_value):
                continue
            threshold = member_threshold if label == 1 else nonmember_threshold
            if threshold is None:
                continue
            if not _accept_by_threshold(filt_score, threshold, filter_direction):
                continue
            selected_scores.append(mia_value)
            selected_labels.append(label)
            if label == 1:
                selected_members += 1
            else:
                selected_nonmembers += 1

        oriented = maybe_orient_scores(selected_scores, mia_direction) if selected_scores else []
        auroc = float(simple_auroc(oriented, selected_labels)) if oriented else 0.5
        row = {
            "quantile": q,
            "member_threshold": member_threshold,
            "nonmember_threshold": nonmember_threshold,
            "num_members_selected": selected_members,
            "num_nonmembers_selected": selected_nonmembers,
            "num_selected_total": selected_members + selected_nonmembers,
            "member_acceptance_fraction": (selected_members / total_members) if total_members else 0.0,
            "nonmember_acceptance_fraction": (selected_nonmembers / total_nonmembers) if total_nonmembers else 0.0,
            "auroc": auroc,
        }
        for fpr in fpr_grid:
            row[f"tpr_at_fpr_{fpr:.2f}"] = float(tpr_at_fpr(oriented, selected_labels, fpr)) if oriented else 0.0
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute separate score quantiles for members and nonmembers from a cache."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .pt cache (e.g., mia_cpk_cache.pt).",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="cpk_stat",
        help="Score key to analyze (default: cpk_stat; falls back to cpk_logratio_max).",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated quantiles in [0,1].",
    )
    parser.add_argument(
        "--filter-direction",
        type=str,
        choices=["lower", "higher"],
        default="lower",
        help="Acceptance direction for score_key thresholds.",
    )
    parser.add_argument(
        "--fpr-grid",
        type=str,
        default="0.01,0.05,0.10",
        help="Comma-separated FPR targets used for TPR@FPR stats.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for JSON/CSV results.",
    )
    args = parser.parse_args()

    quantiles = _parse_quantiles(args.quantiles)
    fpr_grid = _parse_quantiles(args.fpr_grid)
    cache_path = Path(args.input)
    cache = load_cache(cache_path)
    rows = [dict(r) for r in cache.get("data", {}).get("records", [])]
    grouped = _collect_group_scores(rows, args.score_key)
    mia_direction = str(cache.get("data", {}).get("mia", {}).get("score_direction", "higher_is_member"))

    summary_rows = [
        _summary_row("members", grouped["members"], quantiles),
        _summary_row("nonmembers", grouped["nonmembers"], quantiles),
    ]
    auroc_rows = _compute_auroc_rows(rows, args.score_key, quantiles, args.filter_direction, mia_direction, fpr_grid)
    payload = {
        "input_cache_path": str(cache_path),
        "score_key": args.score_key,
        "filter_direction": args.filter_direction,
        "mia_score_direction": mia_direction,
        "quantiles": quantiles,
        "fpr_grid": fpr_grid,
        "quantile_rows": summary_rows,
        "auroc_rows": auroc_rows,
    }

    print(json.dumps(payload, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "quantiles_by_membership.json", payload)
        dump_csv(out_dir / "quantiles_by_membership.csv", summary_rows)
        dump_csv(out_dir / "auroc_by_membership_quantile.csv", auroc_rows)
        print(f"Wrote: {out_dir / 'quantiles_by_membership.json'}")
        print(f"Wrote: {out_dir / 'quantiles_by_membership.csv'}")
        print(f"Wrote: {out_dir / 'auroc_by_membership_quantile.csv'}")


if __name__ == "__main__":
    main()
