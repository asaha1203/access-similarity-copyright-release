import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline import inference_tasks as tasks  # noqa: E402


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 64

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [2 + (ord(ch) % 29) for ch in str(text)]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids


class DraTextGenerationBatchingTest(unittest.TestCase):
    def test_text_dra_uses_target_and_score_batches(self):
        samples = [
            {"sample_id": idx, "prompt": f"prompt {idx}", "target_text": f"target {idx}"}
            for idx in range(5)
        ]
        loaded = SimpleNamespace(tokenizer=DummyTokenizer(), training_config={"max_seq_len": 64})
        generation_batch_sizes = []
        score_batch_sizes = []

        def fake_generate(_model, prompts, _generation_cfg, seed):
            del _model, _generation_cfg, seed
            generation_batch_sizes.append(len(prompts))
            return [
                {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "generated_text": f"generated {idx}",
                    "decode_strategy": "sample",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_new_tokens": 4,
                    "eos_reached": False,
                }
                for idx, prompt in enumerate(prompts)
            ]

        def fake_score(_model, prompts, continuations, **kwargs):
            del _model, kwargs
            score_batch_sizes.append(len(prompts))
            return [
                {
                    "seq_logprob": -float(idx + 1),
                    "mean_logprob": -0.5,
                    "token_logprobs": [-0.5],
                    "prompt_token_count": len(prompt),
                    "scored_token_count": len(continuation),
                }
                for idx, (prompt, continuation) in enumerate(zip(prompts, continuations))
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "project": {"output_root": tmpdir, "run_name": "dra_text_test", "seed": 7},
                "dataset": {"name": "dummy_text", "modality": "text", "target_split": "train_full"},
                "models": {"target": {"checkpoint_dir": "."}},
                "cpk": {"enabled": False},
                "dra_generation": {"target_limit": 5, "num_candidates_per_target": 3, "target_batch_size": 2},
                "generation": {"max_new_tokens": 4, "batch_size": 6, "score_batch_size": 4},
            }
            with (
                patch.object(tasks, "load_text_model", return_value=loaded),
                patch.object(tasks, "collect_text_prompts", return_value=samples),
                patch.object(tasks, "generate_text_candidates", side_effect=fake_generate),
                patch.object(tasks, "realized_token_logprobs_batch", side_effect=fake_score),
            ):
                run_dir = tasks.run_generate_dra(config)

            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(generation_batch_sizes, [6, 6, 3])
        self.assertEqual(score_batch_sizes, [4, 2, 4, 2, 3])
        self.assertEqual(len(records), 15)
        self.assertEqual(records[0]["candidate_id"], "0_0")
        self.assertEqual(records[3]["candidate_id"], "1_0")
        self.assertEqual(records[-1]["candidate_id"], "4_2")

    def test_text_dra_passes_generation_runtime_controls(self):
        samples = [{"sample_id": 0, "prompt": "prompt", "target_text": "target"}]
        loaded = SimpleNamespace(tokenizer=DummyTokenizer(), training_config={"max_seq_len": 64})
        load_calls = []
        generation_cfgs = []

        def fake_load_text_model(*args, **kwargs):
            load_calls.append((args, kwargs))
            return loaded

        def fake_generate(_model, prompts, generation_cfg, seed):
            del _model, seed
            generation_cfgs.append(dict(generation_cfg))
            return [
                {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "generated_text": "generated",
                    "decode_strategy": "sample",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_new_tokens": generation_cfg["max_new_tokens"],
                    "eos_reached": False,
                }
                for idx, prompt in enumerate(prompts)
            ]

        def fake_score(_model, prompts, continuations, **kwargs):
            del _model, kwargs
            return [
                {
                    "seq_logprob": -1.0,
                    "mean_logprob": -0.5,
                    "token_logprobs": [-0.5],
                    "prompt_token_count": len(prompt),
                    "scored_token_count": len(continuation),
                }
                for prompt, continuation in zip(prompts, continuations)
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "project": {"output_root": tmpdir, "run_name": "dra_text_controls", "seed": 7},
                "dataset": {"name": "dummy_text", "modality": "text", "target_split": "train_full"},
                "models": {"target": {"checkpoint_dir": "."}},
                "cpk": {"enabled": False},
                "runtime": {"device": "cuda", "text_dtype": "bf16", "text_compile_model": False},
                "dra_generation": {"target_limit": 1, "num_candidates_per_target": 2, "target_batch_size": 1},
                "generation": {
                    "max_new_tokens": 2048,
                    "batch_size": 2,
                    "max_prompt_tokens": 128,
                    "pad_to_multiple_of": 8,
                    "cache_implementation": "dynamic",
                },
            }
            with (
                patch.object(tasks, "load_text_model", side_effect=fake_load_text_model),
                patch.object(tasks, "collect_text_prompts", return_value=samples),
                patch.object(tasks, "generate_text_candidates", side_effect=fake_generate),
                patch.object(tasks, "realized_token_logprobs_batch", side_effect=fake_score),
            ):
                tasks.run_generate_dra(config)

        self.assertEqual(len(load_calls), 1)
        self.assertEqual(load_calls[0][1]["torch_dtype"].__str__(), "torch.bfloat16")
        self.assertFalse(load_calls[0][1]["compile_model"])
        self.assertTrue(load_calls[0][1]["merge_lora"])
        self.assertEqual(generation_cfgs[0]["max_prompt_tokens"], 128)
        self.assertEqual(generation_cfgs[0]["pad_to_multiple_of"], 8)
        self.assertEqual(generation_cfgs[0]["cache_implementation"], "dynamic")


if __name__ == "__main__":
    unittest.main()
