import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline.inference_cache import make_cache, save_cache, load_cache
from copyright_pipeline.inference_tasks import run_analyze_dra_cache, run_analyze_mia_cache


class InferenceCacheWorkflowTest(unittest.TestCase):
    def test_cache_overwrite_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mia_cache.pt"
            payload = make_cache(
                stage="mia",
                config={"project": {"run_name": "unit"}},
                model_type="baseline",
                modality="text",
                data={
                    "records": [],
                    "mia": {"scores": torch.empty(0), "score_direction": "higher_is_member"},
                },
            )
            save_cache(path, payload)
            with self.assertRaises(FileExistsError):
                save_cache(path, payload)
            save_cache(path, payload, overwrite=True)
            loaded = load_cache(path)
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(loaded["stage"], "mia")
            self.assertTrue(Path(str(path) + ".manifest.json").exists())

    def test_mia_cache_ar_one_accepts_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "mia_cpk_cache.pt"
            rows = [
                {"membership_label": 1, "mia_score": 0.9, "mia_score_direction": "higher_is_member", "cpk_stat": 100.0},
                {"membership_label": 1, "mia_score": 0.8, "mia_score_direction": "higher_is_member", "cpk_stat": 50.0},
                {"membership_label": 0, "mia_score": 0.1, "mia_score_direction": "higher_is_member", "cpk_stat": 10.0},
                {"membership_label": 0, "mia_score": 0.2, "mia_score_direction": "higher_is_member", "cpk_stat": 5.0},
            ]
            payload = make_cache(
                stage="cpk",
                config={"project": {"run_name": "unit"}},
                model_type="cpk",
                modality="text",
                data={
                    "records": rows,
                    "mia": {"scores": torch.tensor([0.9, 0.8, 0.1, 0.2]), "score_direction": "higher_is_member"},
                    "cpk": {"cpk_stat": torch.tensor([100.0, 50.0, 10.0, 5.0])},
                },
            )
            save_cache(cache_path, payload)
            out = run_analyze_mia_cache(
                {
                    "project": {"output_root": str(Path(tmp) / "outputs"), "dataset_name": "unit", "run_name": "mia_analysis"},
                    "cpk": {"acceptance_rates": [1.0]},
                    "mia_analysis": {"fpr_grid": [0.1], "fpr_points": 5},
                },
                str(cache_path),
            )
            metrics = json.loads((out / "metrics" / "mia_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["rows"][0]["num_samples"], len(rows))
            self.assertEqual(metrics["rows"][0]["acceptance_fraction"], 1.0)

    def test_dra_analysis_ranks_by_mia_not_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "dra_scored_cache.pt"
            rows = [
                {
                    "target_sample_id": 0,
                    "candidate_id": "bad_distance_high_mia",
                    "mia_score": 10.0,
                    "mia_score_direction": "higher_is_member",
                    "normalized_l2": 9.0,
                },
                {
                    "target_sample_id": 0,
                    "candidate_id": "good_distance_low_mia",
                    "mia_score": 1.0,
                    "mia_score_direction": "higher_is_member",
                    "normalized_l2": 0.0,
                },
            ]
            payload = make_cache(
                stage="dra_scored",
                config={"project": {"run_name": "unit"}},
                model_type="baseline",
                modality="diffusion",
                data={
                    "records": rows,
                    "mia": {"scores": torch.tensor([10.0, 1.0]), "score_direction": "higher_is_member"},
                    "distances": {"normalized_l2": torch.tensor([9.0, 0.0])},
                },
            )
            save_cache(cache_path, payload)
            out = run_analyze_dra_cache(
                {
                    "project": {"output_root": str(Path(tmp) / "outputs"), "dataset_name": "unit", "run_name": "dra_analysis"},
                    "cpk": {"acceptance_rates": [1.0]},
                },
                str(cache_path),
            )
            selected_csv = (out / "metrics" / "dra_selected_candidates.csv").read_text(encoding="utf-8")
            self.assertIn("bad_distance_high_mia", selected_csv)
            self.assertNotIn("good_distance_low_mia", selected_csv)


if __name__ == "__main__":
    unittest.main()
