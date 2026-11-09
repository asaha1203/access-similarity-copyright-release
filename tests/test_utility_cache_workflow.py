import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline.inference_cache import load_cache, make_cache, save_cache
from copyright_pipeline.inference_tasks import (
    _one_per_target_rows,
    run_analyze_utility_cache,
    run_score_utility_cache,
)


class UtilityCacheWorkflowTest(unittest.TestCase):
    def test_one_per_target_selector_is_stable(self):
        rows = [
            {"target_sample_id": 3, "candidate_id": "3_a"},
            {"target_sample_id": 3, "candidate_id": "3_b"},
            {"target_sample_id": 4, "candidate_id": "4_a"},
            {"target_sample_id": 3, "candidate_id": "3_c"},
            {"target_sample_id": 4, "candidate_id": "4_b"},
        ]
        selected = _one_per_target_rows(rows)
        self.assertEqual([r["candidate_id"] for r in selected], ["3_a", "4_a"])
        self.assertEqual([r["_row_index"] for r in selected], [0, 2])

    def test_analyze_utility_text_quantile_filters_perplexity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "utility_features.pt"
            heldout_rows = [
                {"sample_id": 0, "token_count": 10, "nll_draft": 10.0, "k_stat": 1.0},
                {"sample_id": 1, "token_count": 10, "nll_draft": 40.0, "k_stat": 3.0},
                {"sample_id": 2, "token_count": 10, "nll_draft": 80.0, "k_stat": 5.0},
            ]
            reference_ppl_rows = [
                {"row_index": 0, "target_sample_id": 0, "candidate_id": "0_0", "reference_ppl": 8.0, "k_stat": 1.0},
                {"row_index": 1, "target_sample_id": 1, "candidate_id": "1_0", "reference_ppl": 16.0, "k_stat": 5.0},
            ]
            payload = make_cache(
                stage="utility_features",
                config={"project": {"run_name": "unit"}},
                model_type="cpk",
                modality="text",
                data={
                    "cpk_enabled": True,
                    "text": {
                        "heldout_scores": heldout_rows,
                        "reference_ppl_scores": reference_ppl_rows,
                        "prometheus_scores": [],
                    }
                },
            )
            save_cache(cache_path, payload)
            out = run_analyze_utility_cache(
                {
                    "project": {"output_root": str(Path(tmp) / "outputs"), "dataset_name": "unit", "run_name": "utility_analysis"},
                    "cpk": {"acceptance_rates": [1.0, 0.5]},
                    "utility_analysis": {"min_samples": 1},
                },
                str(cache_path),
            )
            metrics = json.loads((out / "metrics" / "utility_metrics.json").read_text(encoding="utf-8"))
            rows = metrics["rows"]
            self.assertEqual(len(rows), 2)
            all_row = next(r for r in rows if float(r["acceptance_rate"]) == 1.0)
            low_row = next(r for r in rows if float(r["acceptance_rate"]) == 0.5)
            self.assertEqual(all_row["num_heldout_samples"], 3)
            self.assertEqual(low_row["num_heldout_samples"], 2)
            self.assertEqual(all_row["num_reference_ppl_samples"], 2)
            self.assertEqual(low_row["num_reference_ppl_samples"], 1)
            self.assertNotIn("mauve", all_row)
            self.assertNotIn("num_mauve_samples", all_row)
            self.assertLess(float(low_row["heldout_ppl_draft"]), float(all_row["heldout_ppl_draft"]))
            self.assertLess(float(low_row["reference_ppl_mean"]), float(all_row["reference_ppl_mean"]))

    def test_score_utility_cache_does_not_mutate_source_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_cache = Path(tmp) / "dra_generation_cache.pt"
            rows = [
                {"target_sample_id": 0, "candidate_id": "0_0", "prompt": "p", "generated_text": "g0"},
                {"target_sample_id": 1, "candidate_id": "1_0", "prompt": "p", "generated_text": "g1"},
            ]
            payload = make_cache(
                stage="dra_generation",
                config={"project": {"run_name": "unit"}},
                model_type="baseline",
                modality="text",
                data={"records": rows, "generations": {}},
            )
            save_cache(source_cache, payload)
            before = source_cache.read_bytes()
            output_cache = Path(tmp) / "utility_features_cache.pt"
            run_score_utility_cache(
                {
                    "project": {"run_name": "utility_unit"},
                    "dataset": {"name": "unit", "modality": "text"},
                    "utility": {"metrics": []},
                    "cache": {"overwrite": True},
                },
                input_path=str(source_cache),
                output_path=str(output_cache),
                overwrite=True,
            )
            after = source_cache.read_bytes()
            self.assertEqual(before, after)
            loaded = load_cache(output_cache)
            self.assertEqual(loaded["stage"], "utility_features")
            self.assertIn("text", loaded["data"])
            self.assertNotIn("image", loaded["data"])
            self.assertNotIn("mauve_generation_rows", loaded["data"]["text"])
            self.assertNotIn("mauve_reference_rows", loaded["data"]["text"])

    def test_score_utility_cache_records_prometheus_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_cache = Path(tmp) / "dra_generation_cache.pt"
            rows = [
                {"target_sample_id": 0, "candidate_id": "0_0", "prompt": "p0", "generated_text": "g0"},
                {"target_sample_id": 0, "candidate_id": "0_1", "prompt": "p0", "generated_text": "g1"},
                {"target_sample_id": 1, "candidate_id": "1_0", "prompt": "p1", "generated_text": "g2"},
            ]
            payload = make_cache(
                stage="dra_generation",
                config={"project": {"run_name": "unit"}},
                model_type="baseline",
                modality="text",
                data={"records": rows},
            )
            save_cache(source_cache, payload)
            output_cache = Path(tmp) / "utility_features_cache.pt"
            with patch("copyright_pipeline.inference_tasks.PrometheusV2Scorer") as scorer_cls:
                scorer_cls.return_value.score_rows.return_value = [
                    {"prometheusv2_fluency": 4.0, "parse_ok": True, "parse_error": ""},
                    {"prometheusv2_fluency": 2.0, "parse_ok": False, "parse_error": "missing_or_invalid_result_token"},
                ]
                run_score_utility_cache(
                    {
                        "project": {"run_name": "utility_unit"},
                        "dataset": {"name": "unit", "modality": "text"},
                        "runtime": {"device": "cpu"},
                        "utility": {"metrics": ["prometheusv2_fluency"]},
                        "cache": {"overwrite": True},
                    },
                    input_path=str(source_cache),
                    output_path=str(output_cache),
                    overwrite=True,
                )
            loaded = load_cache(output_cache)
            prom_rows = loaded["data"]["text"]["prometheus_scores"]
            self.assertEqual(len(prom_rows), 2)
            self.assertEqual(prom_rows[0]["target_sample_id"], 0)
            self.assertEqual(prom_rows[1]["target_sample_id"], 1)
            self.assertEqual(prom_rows[0]["prometheusv2_fluency"], 4.0)
            self.assertFalse(prom_rows[1]["parse_ok"])


if __name__ == "__main__":
    unittest.main()
