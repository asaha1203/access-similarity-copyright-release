import sys
import unittest
import math
import random
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline import inference_tasks as tasks  # noqa: E402


def _normalized_levenshtein_reference(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a:
        return 1.0 if b else 0.0
    if not b:
        return 1.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return float(dp[-1] / max(len(a), len(b)))


def _bleu1_reference(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not cand_tokens:
        return 0.0
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    hit = 0
    for token in cand_tokens:
        current = ref_counts.get(token, 0)
        if current > 0:
            hit += 1
            ref_counts[token] = current - 1
    precision = hit / len(cand_tokens)
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(cand_tokens)))
    return float(bp * precision)


def _fake_rouge_modules():
    package = ModuleType("rouge_score")
    scorer_module = ModuleType("rouge_score.rouge_scorer")

    class FakeRougeScorer:
        def __init__(self, keys, use_stemmer=False):
            self.keys = keys
            self.use_stemmer = use_stemmer

        def score(self, target, prediction):
            target_tokens = str(target).split()
            prediction_tokens = str(prediction).split()
            if not target_tokens and not prediction_tokens:
                fmeasure = 1.0
            elif target_tokens == prediction_tokens:
                fmeasure = 1.0
            else:
                common = len(set(target_tokens).intersection(prediction_tokens))
                denom = len(target_tokens) + len(prediction_tokens)
                fmeasure = (2.0 * common / denom) if denom else 0.0
            return {"rougeL": SimpleNamespace(fmeasure=fmeasure)}

    scorer_module.RougeScorer = FakeRougeScorer
    package.rouge_scorer = scorer_module
    return {"rouge_score": package, "rouge_score.rouge_scorer": scorer_module}


class DraDistanceMetricTest(unittest.TestCase):
    def setUp(self):
        tasks._CLIP_COMPONENTS_CACHE.clear()
        tasks._LPIPS_MODEL_CACHE.clear()
        tasks._DREAMSIM_COMPONENTS_CACHE.clear()
        tasks._ROUGE_SCORER_CACHE.clear()
        tasks._BERT_SCORER_CACHE.clear()

    def test_registry_aliases_and_cache_payload_include_new_metrics(self):
        config = {"dra_distance": {"image_metrics": ["ssim_distance"], "text_metrics": ["rouge"]}}
        self.assertEqual(tasks._dra_distance_metrics(config, "image"), ("ssim",))
        self.assertEqual(tasks._dra_distance_metrics(config, "text"), ("rouge_l",))

        rows = [
            {
                "target_sample_id": 0,
                "candidate_id": "0_0",
                "mia_score": 0.5,
                "mia_score_direction": "higher_is_member",
                "ssim": 0.0,
                "dreamsim": 0.25,
                "rouge_l": 0.0,
            }
        ]
        payload = tasks._rows_to_dra_scored_cache(rows, {"project": {"run_name": "unit"}}, modality="text")
        self.assertIn("ssim", payload["data"]["distances"])
        self.assertIn("dreamsim", payload["data"]["distances"])
        self.assertIn("rouge_l", payload["data"]["distances"])
        self.assertEqual(tasks.HIGHER_IS_CLOSER_DRA_DISTANCE_KEYS, set())
        self.assertNotIn("rouge_l", tasks.HIGHER_IS_CLOSER_DRA_DISTANCE_KEYS)

    def test_text_similarity_metrics_are_stored_as_distances(self):
        bert_module = ModuleType("bert_score")

        class FakeBERTScorer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def score(self, candidates, references, batch_size=1, verbose=False):
                del batch_size, verbose
                return None, None, torch.tensor([1.0 if cand == ref else 0.25 for cand, ref in zip(candidates, references)])

        bert_module.BERTScorer = FakeBERTScorer
        rows = [
            {"target_text": "the quick brown fox", "generated_text": "the quick brown fox"},
            {"target_text": "the quick brown fox", "generated_text": "different text"},
        ]
        config = {"dra_distance": {"text_metrics": ["bleu", "bertscore_f1"], "batch_size": 2}}
        with patch.dict(sys.modules, {"bert_score": bert_module}):
            tasks._add_text_dra_distances(rows, config, torch.device("cpu"))
        self.assertEqual(rows[0]["bleu"], 0.0)
        self.assertGreater(rows[1]["bleu"], rows[0]["bleu"])
        self.assertEqual(rows[0]["bertscore_f1"], 0.0)
        self.assertEqual(rows[1]["bertscore_f1"], 0.75)

    def test_bertscore_handles_empty_strings_before_scorer(self):
        bert_module = ModuleType("bert_score")
        seen = {"candidates": None, "references": None}

        class FakeBERTScorer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def score(self, candidates, references, batch_size=1, verbose=False):
                del batch_size, verbose
                if any(not str(text).strip() for text in list(candidates) + list(references)):
                    raise AssertionError("empty strings should not be passed to BERTScorer")
                seen["candidates"] = list(candidates)
                seen["references"] = list(references)
                return None, None, torch.tensor([0.5 for _ in candidates])

        bert_module.BERTScorer = FakeBERTScorer
        rows = [
            {"target_text": "reference text", "generated_text": ""},
            {"target_text": "", "generated_text": ""},
            {"target_text": "reference text", "generated_text": "candidate text"},
        ]
        config = {"dra_distance": {"text_metrics": ["bertscore_f1"], "batch_size": 2}}
        with patch.dict(sys.modules, {"bert_score": bert_module}):
            tasks._add_text_dra_distances(rows, config, torch.device("cpu"))
        self.assertEqual(seen["candidates"], ["candidate text"])
        self.assertEqual(seen["references"], ["reference text"])
        self.assertEqual(rows[0]["bertscore_f1"], 1.0)
        self.assertEqual(rows[1]["bertscore_f1"], 0.0)
        self.assertEqual(rows[2]["bertscore_f1"], 0.5)

    def test_rouge_l_distance_exact_match_is_zero(self):
        rows = [
            {"target_text": "the quick brown fox", "generated_text": "the quick brown fox"},
            {"target_text": "the quick brown fox", "generated_text": "totally different"},
        ]
        config = {"dra_distance": {"text_metrics": ["rouge_l"]}}
        with patch.dict(sys.modules, _fake_rouge_modules()):
            tasks._add_text_dra_distances(rows, config, torch.device("cpu"))
        self.assertEqual(rows[0]["rouge_l"], 0.0)
        self.assertGreater(rows[1]["rouge_l"], rows[0]["rouge_l"])

    def test_normalized_levenshtein_matches_reference_cases(self):
        cases = [
            ("", "", 0.0),
            ("abc", "abc", 0.0),
            ("", "abc", 1.0),
            ("abc", "", 1.0),
            ("kitten", "sitting", 3.0 / 7.0),
        ]
        for a, b, expected in cases:
            self.assertAlmostEqual(tasks.normalized_levenshtein(a, b), expected, places=7)
            self.assertAlmostEqual(tasks.normalized_levenshtein(a, b), _normalized_levenshtein_reference(a, b), places=7)

    def test_bleu1_matches_reference_cases(self):
        cases = [
            ("", "", 0.0),
            ("a b c", "", 0.0),
            ("a b c", "a b c", 1.0),
            ("a b c", "a a a", 1.0 / 3.0),
            ("a b c d", "a b", math.exp(-1.0)),
        ]
        for reference, candidate, expected in cases:
            self.assertAlmostEqual(tasks.bleu1(reference, candidate), expected, places=7)
            self.assertAlmostEqual(tasks.bleu1(reference, candidate), _bleu1_reference(reference, candidate), places=7)

    def test_text_metric_random_parity_with_reference(self):
        rng = random.Random(7)
        vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
        for _ in range(40):
            ref = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 14)))
            cand = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 14)))
            self.assertAlmostEqual(tasks.normalized_levenshtein(ref, cand), _normalized_levenshtein_reference(ref, cand), places=7)
            self.assertAlmostEqual(tasks.bleu1(ref, cand), _bleu1_reference(ref, cand), places=7)

    def test_bleu_distance_is_clamped(self):
        rows = [{"target_text": "reference", "generated_text": "candidate"}]
        config = {"dra_distance": {"text_metrics": ["bleu"]}}
        with patch.object(tasks, "bleu_avg_1_to4", return_value=2.5):
            tasks._add_text_dra_distances(rows, config, torch.device("cpu"))
        self.assertEqual(rows[0]["bleu"], 0.0)

    def test_bleu_avg_1_to4_exact_match_is_one(self):
        self.assertAlmostEqual(tasks.bleu_avg_1_to4("the quick brown fox", "the quick brown fox"), 1.0, places=7)

    def test_bleu_avg_1_to4_is_bounded(self):
        value = tasks.bleu_avg_1_to4("the quick brown fox", "completely different tokens")
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_bertscore_deduplicates_identical_pairs(self):
        bert_module = ModuleType("bert_score")
        seen = {"num_pairs": 0}

        class FakeBERTScorer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def score(self, candidates, references, batch_size=1, verbose=False):
                del batch_size, verbose
                seen["num_pairs"] = len(candidates)
                return None, None, torch.tensor([0.6 for _ in candidates], dtype=torch.float32)

        bert_module.BERTScorer = FakeBERTScorer
        rows = [
            {"target_text": "a", "generated_text": "x"},
            {"target_text": "a", "generated_text": "x"},
            {"target_text": "b", "generated_text": "y"},
        ]
        config = {"dra_distance": {"text_metrics": ["bertscore_f1"], "batch_size": 4, "bertscore_deduplicate_pairs": True}}
        with patch.dict(sys.modules, {"bert_score": bert_module}):
            tasks._add_text_dra_distances(rows, config, torch.device("cpu"))
        self.assertEqual(seen["num_pairs"], 2)
        self.assertEqual(rows[0]["bertscore_f1"], rows[1]["bertscore_f1"])

    def test_ssim_distance_exact_match_is_zero_with_batched_inputs(self):
        torchmetrics_package = ModuleType("torchmetrics")
        image_module = ModuleType("torchmetrics.image")

        class FakeSSIM:
            def __init__(self, data_range=1.0, reduction="none"):
                self.data_range = data_range
                self.reduction = reduction

            def to(self, device):
                self.device = device
                return self

            def __call__(self, preds, target):
                del target
                return torch.ones(preds.shape[0], device=preds.device, dtype=preds.dtype)

        image_module.StructuralSimilarityIndexMeasure = FakeSSIM
        torchmetrics_package.image = image_module
        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"image_metrics": ["ssim"], "image_size": 8, "batch_size": 2}}
        with patch.dict(sys.modules, {"torchmetrics": torchmetrics_package, "torchmetrics.image": image_module}):
            values = tasks._compute_image_distance_values([image, image], [image, image], config, torch.device("cpu"), ("ssim",))
        self.assertEqual(values["ssim"], [0.0, 0.0])

    def test_lpips_model_is_cached(self):
        lpips_module = ModuleType("lpips")
        calls = {"count": 0}

        class FakeLPIPS:
            def __init__(self, net="alex"):
                self.net = net
                calls["count"] += 1

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def __call__(self, cand, tgt):
                return torch.zeros(cand.shape[0], device=cand.device, dtype=cand.dtype)

        lpips_module.LPIPS = FakeLPIPS
        images = torch.rand(2, 3, 8, 8)
        config = {"dra_distance": {"batch_size": 1, "lpips_backbone": "alex"}}
        with patch.dict(sys.modules, {"lpips": lpips_module}):
            tasks._compute_lpips_values(images, images, config, torch.device("cpu"))
            tasks._compute_lpips_values(images, images, config, torch.device("cpu"))
        self.assertEqual(calls["count"], 1)

    def test_dreamsim_components_are_cached_and_batched(self):
        dreamsim_module = ModuleType("dreamsim")
        calls = {"factory": 0, "forward": 0}

        class FakeDreamSimModel:
            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def __call__(self, cand, tgt):
                calls["forward"] += 1
                return (cand - tgt).abs().flatten(1).mean(dim=1)

        def fake_preprocess(image):
            del image
            return torch.zeros(3, 4, 4)

        def fake_dreamsim(**kwargs):
            del kwargs
            calls["factory"] += 1
            return FakeDreamSimModel(), fake_preprocess

        dreamsim_module.dreamsim = fake_dreamsim
        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"image_metrics": ["dreamsim"], "image_size": 8, "batch_size": 2}}
        with patch.dict(sys.modules, {"dreamsim": dreamsim_module}):
            first = tasks._compute_image_distance_values([image, image], [image, image], config, torch.device("cpu"), ("dreamsim",))
            second = tasks._compute_image_distance_values([image], [image], config, torch.device("cpu"), ("dreamsim",))
        self.assertEqual(calls["factory"], 1)
        self.assertEqual(calls["forward"], 2)
        self.assertEqual(first["dreamsim"], [0.0, 0.0])
        self.assertEqual(second["dreamsim"], [0.0])

    def test_clip_target_features_are_cached_for_repeated_targets(self):
        calls = {"images": 0}

        class FakeProcessor:
            def __call__(self, images, return_tensors="pt"):
                del return_tensors
                return {"pixel_values": torch.ones(len(images), 3, 4, 4)}

        class FakeClipModel:
            def get_image_features(self, pixel_values):
                calls["images"] += int(pixel_values.shape[0])
                return torch.ones(pixel_values.shape[0], 4)

        rows = [{"target_sample_id": 7}, {"target_sample_id": 7}, {"target_sample_id": 7}]
        candidates = [torch.rand(3, 8, 8) for _ in rows]
        targets = [torch.rand(3, 8, 8) for _ in rows]
        config = {"dra_distance": {"image_metrics": ["clip"], "image_size": 8, "batch_size": 1}}
        with patch.object(tasks, "_get_clip_components", return_value=(FakeProcessor(), FakeClipModel())):
            tasks._add_image_dra_distances(rows, candidates, targets, config, torch.device("cpu"))
        self.assertEqual(calls["images"], 4)
        self.assertTrue(all(row["clip"] == 0.0 for row in rows))

    def test_clip_distance_is_scaled_to_unit_interval(self):
        calls = {"count": 0}

        class FakeProcessor:
            def __call__(self, images, return_tensors="pt"):
                del return_tensors
                return {"pixel_values": torch.ones(len(images), 3, 4, 4)}

        class FakeClipModel:
            def get_image_features(self, pixel_values):
                del pixel_values
                calls["count"] += 1
                return torch.tensor([[1.0, 0.0]] if calls["count"] == 1 else [[-1.0, 0.0]], dtype=torch.float32)

        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"image_metrics": ["clip"], "image_size": 8, "batch_size": 1}}
        with patch.object(tasks, "_get_clip_components", return_value=(FakeProcessor(), FakeClipModel())):
            values = tasks._compute_image_distance_values([image], [image], config, torch.device("cpu"), ("clip",))
        self.assertEqual(values["clip"], [1.0])

    def test_clip_feature_extraction_accepts_model_output_objects(self):
        class FakeProcessor:
            def __call__(self, images, return_tensors="pt"):
                del return_tensors
                return {"pixel_values": torch.ones(len(images), 3, 4, 4)}

        class FakeClipModel:
            def get_image_features(self, pixel_values):
                return SimpleNamespace(pooler_output=torch.ones(pixel_values.shape[0], 4))

        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"clip_model": "fake", "clip_batch_size": 2}}
        with patch.object(tasks, "_get_clip_components", return_value=(FakeProcessor(), FakeClipModel())):
            features = tasks._compute_clip_image_features([image, image], config, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(features.shape), (2, 4))
        self.assertTrue(torch.allclose(features.norm(p=2, dim=-1), torch.ones(2)))

    def test_dreamsim_target_embeddings_are_cached_for_repeated_targets(self):
        dreamsim_module = ModuleType("dreamsim")
        calls = {"factory": 0, "embedded_images": 0}

        class FakeDreamSimModel:
            feat_type_list = ["cls"]

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def embed(self, batch):
                calls["embedded_images"] += int(batch.shape[0])
                return torch.ones(batch.shape[0], 4, device=batch.device, dtype=batch.dtype)

        def fake_preprocess(image):
            del image
            return torch.ones(3, 4, 4)

        def fake_dreamsim(**kwargs):
            del kwargs
            calls["factory"] += 1
            return FakeDreamSimModel(), fake_preprocess

        dreamsim_module.dreamsim = fake_dreamsim
        rows = [{"target_sample_id": 3}, {"target_sample_id": 3}, {"target_sample_id": 3}]
        candidates = [torch.rand(3, 8, 8) for _ in rows]
        targets = [torch.rand(3, 8, 8) for _ in rows]
        config = {"dra_distance": {"image_metrics": ["dreamsim"], "image_size": 8, "batch_size": 1}}
        with patch.dict(sys.modules, {"dreamsim": dreamsim_module}):
            tasks._add_image_dra_distances(rows, candidates, targets, config, torch.device("cpu"))
        self.assertEqual(calls["factory"], 1)
        self.assertEqual(calls["embedded_images"], 4)
        self.assertTrue(all(row["dreamsim"] == 0.0 for row in rows))

    def test_dreamsim_preprocess_batch_dimension_is_squeezed(self):
        dreamsim_module = ModuleType("dreamsim")

        class FakeDreamSimModel:
            feat_type_list = ["cls"]

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def embed(self, batch):
                if batch.ndim != 4:
                    raise ValueError(f"expected BCHW, got {tuple(batch.shape)}")
                return torch.ones(batch.shape[0], 4, device=batch.device, dtype=batch.dtype)

        def fake_preprocess(image):
            del image
            return torch.ones(1, 3, 4, 4)

        def fake_dreamsim(**kwargs):
            del kwargs
            return FakeDreamSimModel(), fake_preprocess

        dreamsim_module.dreamsim = fake_dreamsim
        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"image_metrics": ["dreamsim"], "image_size": 8, "batch_size": 2}}
        with patch.dict(sys.modules, {"dreamsim": dreamsim_module}):
            features = tasks._compute_dreamsim_embeddings([image, image], config, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(features.shape), (2, 4))

    def test_dreamsim_single_embedding_keeps_batch_dimension(self):
        dreamsim_module = ModuleType("dreamsim")

        class FakeDreamSimModel:
            feat_type_list = ["cls"]

            def to(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def embed(self, batch):
                self.batch_shape = tuple(batch.shape)
                return torch.ones(4, device=batch.device, dtype=batch.dtype)

        def fake_preprocess(image):
            del image
            return torch.ones(1, 3, 4, 4)

        def fake_dreamsim(**kwargs):
            del kwargs
            return FakeDreamSimModel(), fake_preprocess

        dreamsim_module.dreamsim = fake_dreamsim
        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"image_metrics": ["dreamsim"], "image_size": 8, "batch_size": 1}}
        with patch.dict(sys.modules, {"dreamsim": dreamsim_module}):
            features = tasks._compute_dreamsim_embeddings([image], config, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(features.shape), (1, 4))

    def test_dreamsim_embeddings_stay_float32_when_distance_dtype_is_half(self):
        dreamsim_module = ModuleType("dreamsim")
        seen = {"batch_dtype": None, "to_dtype": "unset"}

        class FakeDreamSimModel:
            feat_type_list = ["cls"]

            def to(self, *args, **kwargs):
                seen["to_dtype"] = kwargs.get("dtype")
                return self

            def eval(self):
                return self

            def embed(self, batch):
                seen["batch_dtype"] = batch.dtype
                return torch.ones(batch.shape[0], 4, device=batch.device, dtype=batch.dtype)

        def fake_preprocess(image):
            del image
            return torch.ones(1, 3, 4, 4)

        def fake_dreamsim(**kwargs):
            del kwargs
            return FakeDreamSimModel(), fake_preprocess

        dreamsim_module.dreamsim = fake_dreamsim
        image = torch.rand(3, 8, 8)
        config = {"dra_distance": {"image_metrics": ["dreamsim"], "image_size": 8, "batch_size": 1}}
        with patch.dict(sys.modules, {"dreamsim": dreamsim_module}):
            tasks._compute_dreamsim_embeddings([image], config, torch.device("cpu"), torch.float16)
        self.assertIsNone(seen["to_dtype"])
        self.assertEqual(seen["batch_dtype"], torch.float32)


if __name__ == "__main__":
    unittest.main()
