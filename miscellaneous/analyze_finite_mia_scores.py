#!/usr/bin/env python
"""Run MIA analysis after dropping records with non-finite MIA scores.

This script does not rescore examples. It filters an existing MIA
``records.jsonl`` file to rows whose ``mia_score`` is finite, writes the
filtered records to disk, and runs the standard analyzer on that filtered file.
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_RECORDS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "inference"
    / "automathtext"
    / "minkpp_mia_baseline_full_lora__20260430-153559-74sh"
    / "20260430_193345"
    / "records.jsonl"
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _dump_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_finite_score(row: Dict[str, Any]) -> bool:
    try:
        return math.isfinite(float(row["mia_score"]))
    except (KeyError, TypeError, ValueError):
        return False


def _source_run_name(records_path: Path) -> str:
    if records_path.name == "records.jsonl" and records_path.parent.parent != records_path.parent:
        return records_path.parent.parent.name
    return records_path.stem


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered = yaml.safe_dump(payload, sort_keys=False)
    except TypeError:
        rendered = yaml.safe_dump(payload)
    path.write_text(rendered, encoding="utf-8")


def _safe_quantile(values: List[float], alpha: float) -> float:
    if not values:
        return 0.0
    if alpha <= 0:
        return min(values)
    if alpha >= 1:
        return max(values)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * alpha
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return (1.0 - frac) * sorted_vals[lo] + frac * sorted_vals[hi]


def _maybe_orient_scores(scores: List[float], direction: str) -> List[float]:
    if direction == "higher_is_member":
        return scores
    if direction == "lower_is_member":
        return [-score for score in scores]
    raise ValueError(f"Unsupported mia_score_direction: {direction}")


def _simple_auroc(scores: List[float], labels: List[int]) -> float:
    if not scores:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5

    rank_sum = 0.0
    idx = 0
    n = len(pairs)
    while idx < n:
        j = idx + 1
        score = pairs[idx][0]
        while j < n and pairs[j][0] == score:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        pos_in_group = sum(label for _, label in pairs[idx:j])
        rank_sum += avg_rank * pos_in_group
        idx = j
    u_stat = rank_sum - (pos * (pos + 1) / 2.0)
    return float(u_stat / (pos * neg))


def _tpr_at_fpr(scores: List[float], labels: List[int], fpr_target: float) -> float:
    if not scores:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0

    tp = 0
    fp = 0
    best = 0.0
    idx = 0
    n = len(pairs)
    while idx < n:
        j = idx + 1
        score = pairs[idx][0]
        while j < n and pairs[j][0] == score:
            j += 1
        grp_pos = sum(label for _, label in pairs[idx:j])
        grp_neg = (j - idx) - grp_pos
        tp += grp_pos
        fp += grp_neg
        fpr = fp / max(1, neg)
        tpr = tp / max(1, pos)
        if fpr <= fpr_target:
            best = max(best, tpr)
        idx = j
    return float(best)


def _label_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"members": 0, "nonmembers": 0}
    for row in rows:
        if int(row.get("membership_label", 0)) == 1:
            counts["members"] += 1
        else:
            counts["nonmembers"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an existing MIA records.jsonl after filtering out non-finite mia_score rows."
    )
    parser.add_argument(
        "--records-path",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
        help="Existing scored MIA records.jsonl. Defaults to the latest AutoMathText MinK++ run.",
    )
    parser.add_argument("--dataset", type=str, default="automathtext")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "inference")
    parser.add_argument(
        "--filtered-records-path",
        type=Path,
        default=None,
        help="Where to write filtered records. Defaults to records_finite.jsonl beside --records-path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Analysis run name. Defaults to analyze_finite_<source-score-run-name>.",
    )
    parser.add_argument("--fpr-grid", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--cpk-enabled", action="store_true", help="Apply alpha filtering using cpk_logratio_max.")
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[1.0])
    return parser.parse_args()


def run_finite_analysis(config: Dict[str, Any]) -> Path:
    source_records = Path(config["inputs"]["records_path"])
    rows = _read_jsonl(source_records)
    output_root = Path(config["project"]["output_root"])
    dataset = config["project"]["dataset_name"]
    run_name = config["project"]["run_name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / dataset / run_name / stamp
    metrics_dir = run_dir / "metrics"
    _ensure_dir(metrics_dir)

    _write_yaml(run_dir / "run_config.yaml", config)
    _write_json(run_dir / "run_metadata.json", {"created_at": _now_iso(), "run_dir": str(run_dir)})
    _write_json(run_dir / "input_split_fingerprint.json", {})
    _dump_jsonl(run_dir / "errors.jsonl", [])

    fpr_grid = [float(x) for x in config["mia_analysis"].get("fpr_grid", [0.01, 0.05, 0.1])]
    cpk_enabled = bool(config.get("cpk", {}).get("enabled", False))
    alpha_grid = [float(a) for a in config.get("cpk", {}).get("alpha_grid", [1.0])]
    raw_cpk = [float(row.get("cpk_logratio_max", 0.0)) for row in rows]
    thresholds = {alpha: _safe_quantile(raw_cpk, alpha) for alpha in alpha_grid} if cpk_enabled else {1.0: float("inf")}

    by_alpha: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    for alpha, threshold in thresholds.items():
        subset = [row for row in rows if float(row.get("cpk_logratio_max", -1e18)) <= threshold] if cpk_enabled else rows
        labels = [int(row["membership_label"]) for row in subset]
        direction = subset[0]["mia_score_direction"] if subset else "higher_is_member"
        scores = _maybe_orient_scores([float(row["mia_score"]) for row in subset], direction)
        record = {
            "alpha": alpha,
            "threshold": threshold,
            "num_samples": len(subset),
            "num_members": int(sum(labels)),
            "num_nonmembers": int(len(labels) - sum(labels)),
            "auroc": _simple_auroc(scores, labels),
        }
        for fpr in fpr_grid:
            record[f"tpr_at_fpr_{fpr:.2f}"] = _tpr_at_fpr(scores, labels, fpr)
        by_alpha.append(record)
        for idx, (score, label) in enumerate(zip(scores, labels)):
            roc_rows.append({"alpha": alpha, "rank": idx, "score": score, "label": label})

    _write_json(metrics_dir / "mia_summary.json", {"rows": by_alpha})
    _dump_csv(metrics_dir / "mia_by_alpha.csv", by_alpha)
    _dump_csv(metrics_dir / "roc_points_by_alpha.csv", roc_rows)
    _write_json(metrics_dir / "bootstrap_confidence_intervals.json", {"note": "bootstrap omitted in lightweight implementation"})
    _dump_jsonl(run_dir / "records.jsonl", rows)
    _write_json(run_dir / "summary_metrics.json", {"num_records": len(rows), "source_records_path": str(source_records)})
    return run_dir


def main() -> None:
    args = parse_args()
    records_path = args.records_path.expanduser()
    if not records_path.is_absolute():
        records_path = PROJECT_ROOT / records_path
    if not records_path.exists():
        raise FileNotFoundError(f"Could not find records file: {records_path}")

    filtered_records_path: Path
    if args.filtered_records_path is None:
        filtered_records_path = records_path.with_name(f"{records_path.stem}_finite{records_path.suffix}")
    else:
        filtered_records_path = args.filtered_records_path.expanduser()
        if not filtered_records_path.is_absolute():
            filtered_records_path = PROJECT_ROOT / filtered_records_path

    rows: List[Dict[str, Any]] = _read_jsonl(records_path)
    finite_rows = [row for row in rows if _is_finite_score(row)]
    dropped = len(rows) - len(finite_rows)
    if not finite_rows:
        raise ValueError(f"No finite mia_score rows found in {records_path}")

    _dump_jsonl(filtered_records_path, finite_rows)

    run_name = args.run_name or f"analyze_finite_{_source_run_name(records_path)}"
    analyze_config = {
        "project": {
            "run_name": run_name,
            "dataset_name": args.dataset,
            "output_root": str(args.output_root),
        },
        "inputs": {
            "records_path": str(filtered_records_path),
        },
        "mia_analysis": {
            "fpr_grid": [float(x) for x in args.fpr_grid],
        },
        "cpk": {
            "enabled": bool(args.cpk_enabled),
            "alpha_grid": [float(x) for x in args.alpha_grid],
        },
    }

    config_dir = PROJECT_ROOT / "miscellaneous" / "generated_configs"
    config_path = config_dir / f"{args.dataset}_finite_mia_analyze.yaml"
    _write_yaml(config_path, analyze_config)

    analyze_dir = run_finite_analysis(analyze_config)
    kept_counts = _label_counts(finite_rows)

    print(f"Input records: {records_path}")
    print(f"Filtered records: {filtered_records_path}")
    print(f"Kept finite rows: {len(finite_rows)} / {len(rows)}")
    print(f"Dropped non-finite rows: {dropped}")
    print(f"Kept members: {kept_counts['members']}")
    print(f"Kept nonmembers: {kept_counts['nonmembers']}")
    print(f"Analysis config: {config_path}")
    print(f"Analysis metrics: {analyze_dir / 'metrics' / 'mia_summary.json'}")


if __name__ == "__main__":
    main()
