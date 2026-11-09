#!/usr/bin/env python
"""Print utility summary table from latest valid utility_details.csv."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isnan(out):
            return None
        return out
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none"}:
        return None
    if lowered in {"inf", "infinity"}:
        return float("inf")
    if lowered in {"-inf", "-infinity"}:
        return float("-inf")
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def _extract_utility_timestamp(path: Path) -> str:
    name = path.name
    if name.startswith("utility_"):
        return name.replace("utility_", "", 1)
    return ""


def _latest_valid_utility_dir(run_root: Path, modality: str) -> Path:
    candidates = [p for p in run_root.glob("utility_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No utility_* folders found under: {run_root}")
    candidates.sort(key=lambda p: _extract_utility_timestamp(p), reverse=True)

    for candidate in candidates:
        details_path = candidate / "metrics" / "utility_details.csv"
        if not details_path.exists() or details_path.stat().st_size == 0:
            continue
        if _has_valid_rows(details_path, modality):
            return candidate
    raise FileNotFoundError(
        f"No valid utility_details.csv with modality={modality!r} found under: {run_root}"
    )


def _iter_experiment_dirs(experiments_root: Path, experiment_glob: str) -> List[Path]:
    return sorted([p for p in experiments_root.glob(experiment_glob) if p.is_dir()])


def _has_valid_rows(details_path: Path, modality: str) -> bool:
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("modality", "")).strip().lower() != modality:
                continue
            if _safe_float(row.get("acceptance_rate")) is None:
                continue
            return True
    return False


def _read_image_detail_rows(details_path: Path, modality: str) -> Dict[float, Dict[str, List[float]]]:
    grouped: Dict[float, Dict[str, List[float]]] = {}
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_modality = str(row.get("modality", "")).strip().lower()
            if row_modality != modality:
                continue
            acceptance = _safe_float(row.get("acceptance_rate"))
            if acceptance is None:
                continue
            acceptance_key = float(acceptance)
            bucket = grouped.setdefault(
                acceptance_key,
                {
                    "clip_iqa": [],
                    "clipscore": [],
                },
            )
            clip_iqa = _safe_float(row.get("clip_iqa"))
            clipscore = _safe_float(row.get("clipscore"))
            if clip_iqa is not None and math.isfinite(clip_iqa):
                bucket["clip_iqa"].append(float(clip_iqa))
            if clipscore is not None and math.isfinite(clipscore):
                bucket["clipscore"].append(float(clipscore))
    return grouped


def _read_text_detail_rows(details_path: Path, modality: str) -> Dict[float, Dict[str, List[float]]]:
    grouped: Dict[float, Dict[str, List[float]]] = {}
    with details_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_modality = str(row.get("modality", "")).strip().lower()
            if row_modality != modality:
                continue
            acceptance = _safe_float(row.get("acceptance_rate"))
            if acceptance is None:
                continue
            acceptance_key = float(acceptance)
            bucket = grouped.setdefault(
                acceptance_key,
                {
                    "reference_ppl": [],
                    "heldout_ppl_draft": [],
                    "prometheusv2_fluency": [],
                    "num_prometheus_parse_fail": [],
                },
            )
            reference_ppl = _safe_float(row.get("reference_ppl"))
            if reference_ppl is not None and math.isfinite(reference_ppl):
                bucket["reference_ppl"].append(float(reference_ppl))

            heldout_ppl_draft = _safe_float(row.get("heldout_ppl_draft"))
            if heldout_ppl_draft is not None and math.isfinite(heldout_ppl_draft):
                bucket["heldout_ppl_draft"].append(float(heldout_ppl_draft))

            prometheus = _safe_float(row.get("prometheusv2_fluency"))
            if prometheus is not None and math.isfinite(prometheus):
                bucket["prometheusv2_fluency"].append(float(prometheus))
                parse_ok = str(row.get("parse_ok", "")).strip().lower()
                if parse_ok in {"false", "0", "no"}:
                    bucket["num_prometheus_parse_fail"].append(1.0)
                elif parse_ok in {"true", "1", "yes"}:
                    bucket["num_prometheus_parse_fail"].append(0.0)
    return grouped


def _read_kid_metrics(metrics_path: Path, modality: str) -> Dict[float, Tuple[Optional[float], Optional[int]]]:
    out: Dict[float, Tuple[Optional[float], Optional[int]]] = {}
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return out
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_modality = str(row.get("modality", "")).strip().lower()
            if row_modality != modality:
                continue
            acceptance = _safe_float(row.get("acceptance_rate"))
            if acceptance is None:
                continue
            kid = _safe_float(row.get("kid"))
            num_kid_prompts = _safe_float(row.get("num_kid_prompts"))
            out[float(acceptance)] = (
                kid if kid is not None and math.isfinite(kid) else kid,
                int(num_kid_prompts) if num_kid_prompts is not None else None,
            )
    return out


def _read_text_heldout_metrics(metrics_path: Path, modality: str) -> Dict[float, Tuple[Optional[float], Optional[int]]]:
    out: Dict[float, Tuple[Optional[float], Optional[int]]] = {}
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return out
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_modality = str(row.get("modality", "")).strip().lower()
            if row_modality != modality:
                continue
            acceptance = _safe_float(row.get("acceptance_rate"))
            if acceptance is None:
                continue
            heldout_ppl_draft = _safe_float(row.get("heldout_ppl_draft"))
            num_heldout_samples = _safe_float(row.get("num_heldout_samples"))
            out[float(acceptance)] = (
                heldout_ppl_draft if heldout_ppl_draft is not None and math.isfinite(heldout_ppl_draft) else heldout_ppl_draft,
                int(num_heldout_samples) if num_heldout_samples is not None else None,
            )
    return out


def _mean(values: Iterable[float]) -> Optional[float]:
    values_list = list(values)
    if not values_list:
        return None
    return float(statistics.fmean(values_list))


def _std(values: Iterable[float]) -> Optional[float]:
    values_list = list(values)
    if not values_list:
        return None
    return float(statistics.pstdev(values_list))


def _fmt_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "NA"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return f"{value:.{digits}f}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(value)


def _print_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]], title: str) -> None:
    print(title)
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _build_image_rows(
    experiment: str,
    utility_dir: Path,
    grouped: Dict[float, Dict[str, List[float]]],
    kid_metrics: Dict[float, Tuple[Optional[float], Optional[int]]],
) -> List[List[str]]:
    out: List[List[str]] = []
    for acceptance in sorted(grouped):
        bucket = grouped[acceptance]
        clip_iqa_values = bucket["clip_iqa"]
        clipscore_values = bucket["clipscore"]
        kid, num_kid_prompts = kid_metrics.get(acceptance, (None, None))
        out.append(
            [
                experiment,
                str(utility_dir),
                _fmt_float(acceptance, digits=4),
                str(max(len(clip_iqa_values), len(clipscore_values))),
                _fmt_float(_mean(clip_iqa_values)),
                _fmt_float(_std(clip_iqa_values)),
                _fmt_float(_mean(clipscore_values)),
                _fmt_float(_std(clipscore_values)),
                _fmt_float(kid),
                "NA(group)",
                _fmt_int(num_kid_prompts),
            ]
        )
    return out


def _build_text_rows(
    experiment: str,
    utility_dir: Path,
    grouped: Dict[float, Dict[str, List[float]]],
    heldout_metrics: Dict[float, Tuple[Optional[float], Optional[int]]],
) -> List[List[str]]:
    out: List[List[str]] = []
    for acceptance in sorted(grouped):
        bucket = grouped[acceptance]
        reference_values = bucket["reference_ppl"]
        heldout_draft_values = bucket["heldout_ppl_draft"]
        heldout_ppl_draft, num_heldout_samples = heldout_metrics.get(acceptance, (None, None))
        heldout_rows = num_heldout_samples if num_heldout_samples is not None else len(heldout_draft_values)
        heldout_mean = heldout_ppl_draft if heldout_ppl_draft is not None else _mean(heldout_draft_values)
        heldout_std = _std(heldout_draft_values) if heldout_draft_values else None
        prometheus_values = bucket["prometheusv2_fluency"]
        parse_fail_count = int(sum(bucket["num_prometheus_parse_fail"]))
        out.append(
            [
                experiment,
                str(utility_dir),
                _fmt_float(acceptance, digits=4),
                str(len(reference_values)),
                _fmt_float(_mean(reference_values)),
                _fmt_float(_std(reference_values)),
                str(heldout_rows),
                _fmt_float(heldout_mean),
                _fmt_float(heldout_std),
                str(len(prometheus_values)),
                _fmt_float(_mean(prometheus_values)),
                _fmt_float(_std(prometheus_values)),
                str(parse_fail_count),
            ]
        )
    return out


def _print_image_table(rows: Sequence[Sequence[str]]) -> None:
    headers = [
        "experiment",
        "utility_dir",
        "acceptance_rate",
        "num_rows",
        "clip_iqa_mean",
        "clip_iqa_std",
        "clipscore_mean",
        "clipscore_std",
        "kid",
        "kid_std",
        "num_kid_prompts",
    ]
    _print_markdown_table(headers=headers, rows=rows, title="# utility summary (image)")


def _print_text_table(rows: Sequence[Sequence[str]]) -> None:
    headers = [
        "experiment",
        "utility_dir",
        "acceptance_rate",
        "num_reference_rows",
        "reference_ppl_mean",
        "reference_ppl_std",
        "num_heldout_ppl_draft_rows",
        "heldout_ppl_draft_mean",
        "heldout_ppl_draft_std",
        "num_prometheus_rows",
        "prometheusv2_fluency_mean",
        "prometheusv2_fluency_std",
        "num_prometheus_parse_fail",
    ]
    _print_markdown_table(headers=headers, rows=rows, title="# utility summary (text)")


def _collect_rows_for_experiment(experiment_dir: Path, modality: str) -> List[List[str]]:
    utility_dir = _latest_valid_utility_dir(run_root=experiment_dir, modality=modality)
    details_path = utility_dir / "metrics" / "utility_details.csv"
    experiment = experiment_dir.name
    if modality == "image":
        grouped = _read_image_detail_rows(details_path=details_path, modality=modality)
        if not grouped:
            raise ValueError(f"No image rows found in: {details_path}")
        kid_metrics = _read_kid_metrics(metrics_path=utility_dir / "metrics" / "utility_metrics.csv", modality=modality)
        return _build_image_rows(
            experiment=experiment,
            utility_dir=utility_dir,
            grouped=grouped,
            kid_metrics=kid_metrics,
        )
    grouped_text = _read_text_detail_rows(details_path=details_path, modality=modality)
    if not grouped_text:
        raise ValueError(f"No text rows found in: {details_path}")
    heldout_metrics = _read_text_heldout_metrics(
        metrics_path=utility_dir / "metrics" / "utility_metrics.csv",
        modality=modality,
    )
    return _build_text_rows(
        experiment=experiment,
        utility_dir=utility_dir,
        grouped=grouped_text,
        heldout_metrics=heldout_metrics,
    )


def _print_single_run(run_root: Path, modality: str) -> None:
    rows = _collect_rows_for_experiment(experiment_dir=run_root, modality=modality)
    if modality == "image":
        _print_image_table(rows)
    else:
        _print_text_table(rows)


def _print_aggregate_runs(experiments_root: Path, modality: str, experiment_glob: str) -> None:
    experiments = _iter_experiment_dirs(experiments_root=experiments_root, experiment_glob=experiment_glob)
    if not experiments:
        raise FileNotFoundError(
            f"No experiment directories found under {experiments_root} matching glob={experiment_glob!r}"
        )
    all_rows: List[List[str]] = []
    for experiment_dir in experiments:
        try:
            rows = _collect_rows_for_experiment(experiment_dir=experiment_dir, modality=modality)
        except (FileNotFoundError, ValueError):
            continue
        all_rows.extend(rows)
    if not all_rows:
        raise FileNotFoundError(
            f"No valid utility rows found for modality={modality!r} under experiments root: {experiments_root}"
        )
    if modality == "image":
        _print_image_table(all_rows)
    else:
        _print_text_table(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print utility summary table from latest valid utility outputs. "
            "image: clip_iqa/clipscore mean+std + kid; "
            "text: reference_ppl/prometheusv2_fluency mean+std. "
            "Use --experiments-root to aggregate across experiments."
        )
    )
    root_group = parser.add_mutually_exclusive_group(required=True)
    root_group.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Single experiment directory containing utility_* folders (e.g., outputs/inference/pokemon/dra_cpk_pokemon).",
    )
    root_group.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Directory containing experiment folders (e.g., outputs/inference/pokemon).",
    )
    parser.add_argument(
        "--experiment-glob",
        type=str,
        default="dra_*",
        help="Glob for selecting experiment folders when using --experiments-root.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        help="Modality filter (e.g., image, text).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modality = str(args.modality).strip().lower()
    if modality not in {"image", "text"}:
        raise ValueError(f"Unsupported modality={modality!r}. Expected one of: image, text.")

    if args.run_root is not None:
        run_root = args.run_root if args.run_root.is_absolute() else (Path.cwd() / args.run_root).resolve()
        _print_single_run(run_root=run_root, modality=modality)
        return

    experiments_root = (
        args.experiments_root
        if args.experiments_root.is_absolute()
        else (Path.cwd() / args.experiments_root).resolve()
    )
    _print_aggregate_runs(
        experiments_root=experiments_root,
        modality=modality,
        experiment_glob=args.experiment_glob,
    )


if __name__ == "__main__":
    main()

