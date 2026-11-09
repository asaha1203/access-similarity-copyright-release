import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline import inference_common  # noqa: E402
from copyright_pipeline.inference_common import (  # noqa: E402
    LoadedTextModel,
    generate_text_candidates,
    load_text_model,
    prepare_continuation_first_batch,
    realized_token_logprobs_batch,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 64
    padding_side = "right"

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int = None,
    ):
        ids = [2 + (ord(ch) % 29) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return ids

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=None,
        pad_to_multiple_of=None,
    ):
        del return_tensors
        encoded = [
            self.encode(text, truncation=truncation, max_length=max_length)
            for text in texts
        ]
        max_len = max(len(ids) for ids in encoded)
        if padding == "max_length":
            max_len = int(max_length)
        if pad_to_multiple_of is not None and max_len % int(pad_to_multiple_of) != 0:
            max_len = ((max_len + int(pad_to_multiple_of) - 1) // int(pad_to_multiple_of)) * int(pad_to_multiple_of)
        rows = []
        masks = []
        for ids in encoded:
            ids = ids[:max_len]
            pad_count = max_len - len(ids)
            if padding and self.padding_side == "left":
                row = [self.pad_token_id] * pad_count + ids
                mask = [0] * pad_count + [1] * len(ids)
            elif padding:
                row = ids + [self.pad_token_id] * pad_count
                mask = [1] * len(ids) + [0] * pad_count
            else:
                row = ids
                mask = [1] * len(ids)
            rows.append(row)
            masks.append(mask)

        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})

        return Batch(
            {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
        if skip_special_tokens:
            values = [value for value in values if value not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(str(value) for value in values)


class DummyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, target_boost: float = 4.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_boost = float(target_boost)
        self.generate_calls = []
        self.register_parameter("_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

    @property
    def device(self):
        return self._anchor.device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        del attention_mask
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -1.0, dtype=torch.float32, device=input_ids.device)
        token_slots = (input_ids % self.vocab_size).unsqueeze(-1)
        logits.scatter_(-1, token_slots, self.target_boost)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, max_new_tokens: int = 1, **kwargs):
        del attention_mask
        self.generate_calls.append({"input_shape": tuple(input_ids.shape), "max_new_tokens": int(max_new_tokens), **kwargs})
        batch = input_ids.shape[0]
        new_tokens = torch.arange(10, 10 + max_new_tokens, device=input_ids.device, dtype=input_ids.dtype)
        new_tokens = new_tokens.unsqueeze(0).repeat(batch, 1)
        return torch.cat([input_ids, new_tokens], dim=1)


def _make_loaded(name: str, target_boost: float = 4.0) -> LoadedTextModel:
    return LoadedTextModel(
        name=name,
        model=DummyCausalLM(target_boost=target_boost),
        tokenizer=DummyTokenizer(),
        run_dir=Path("."),
        training_config={"max_seq_len": 64},
    )


class TextLogprobOptimizationTest(unittest.TestCase):
    def test_generate_text_candidates_slices_after_padded_input_width(self):
        loaded = _make_loaded("base")
        loaded.tokenizer.padding_side = "left"
        candidates = generate_text_candidates(
            loaded,
            ["short", "much longer prompt"],
            {"max_new_tokens": 2, "do_sample": False, "batch_size": 2},
            seed=123,
        )
        self.assertEqual([row["generated_ids"] for row in candidates], [[10, 11], [10, 11]])
        self.assertEqual([row["generated_text"] for row in candidates], ["10 11", "10 11"])
        self.assertEqual(candidates[0]["generation_batch_size"], 2)
        self.assertEqual(candidates[0]["generated_token_count"], 2)

    def test_load_text_model_can_merge_lora_and_clears_generation_max_length(self):
        class FakeTokenizer:
            pad_token = None
            eos_token = "<eos>"

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=False)
                self.generation_config = SimpleNamespace(max_length=4096)
                self.register_parameter("_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class FakePeft:
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.config = base.config
                self.generation_config = SimpleNamespace(max_length=4096)
                self.base_model = SimpleNamespace(generation_config=SimpleNamespace(max_length=4096))
                self.merged = False

            @classmethod
            def from_pretrained(cls, base, *_args, **_kwargs):
                return cls(base)

            def merge_and_unload(self):
                self.base.merged = True
                return self.base

            def to(self, *_args, **_kwargs):
                return self

            def eval(self):
                return self

        with self.subTest("merge_lora"):
            import tempfile
            from unittest.mock import patch

            with tempfile.TemporaryDirectory() as tmpdir:
                run_dir = Path(tmpdir)
                (run_dir / "training_config.yaml").write_text("model_name_or_path: fake\n", encoding="utf-8")
                (run_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
                with (
                    patch.object(inference_common, "AutoTokenizer", FakeTokenizer),
                    patch.object(inference_common, "AutoModelForCausalLM", FakeModel),
                    patch.dict("sys.modules", {"peft": SimpleNamespace(PeftModel=FakePeft)}),
                ):
                    loaded = load_text_model(run_dir, torch.device("cpu"), merge_lora=True, compile_model=False)

            self.assertTrue(getattr(loaded.model, "merged", False))
            self.assertTrue(loaded.model.config.use_cache)
            self.assertIsNone(loaded.model.generation_config.max_length)

    def test_generate_text_candidates_applies_prompt_controls_and_greedy_kwargs(self):
        loaded = _make_loaded("base")
        loaded.tokenizer.padding_side = "left"
        candidates = generate_text_candidates(
            loaded,
            ["short", "much longer prompt"],
            {
                "max_new_tokens": 3,
                "do_sample": False,
                "batch_size": 2,
                "max_prompt_tokens": 5,
                "pad_to_multiple_of": 4,
                "cache_implementation": "dynamic",
                "use_cache": True,
            },
            seed=123,
        )
        call = loaded.model.generate_calls[-1]
        self.assertEqual(call["input_shape"], (2, 8))
        self.assertEqual(call["cache_implementation"], "dynamic")
        self.assertTrue(call["use_cache"])
        self.assertNotIn("temperature", call)
        self.assertNotIn("top_p", call)
        self.assertNotIn("top_k", call)
        self.assertEqual([row["generated_ids"] for row in candidates], [[10, 11, 12], [10, 11, 12]])

    def test_prepared_batch_reuse_parity(self):
        loaded = _make_loaded("base")
        prompts = ["alpha", "delta prompt", "x"]
        continuations = ["beta", " gamma", "yz"]
        max_len = 18
        baseline = realized_token_logprobs_batch(
            loaded,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
        )
        prepared = prepare_continuation_first_batch(loaded.tokenizer, prompts, continuations, max_len)
        reused = realized_token_logprobs_batch(
            loaded,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            prepared_batch=prepared,
        )
        self.assertEqual(len(baseline), len(reused))
        for base_row, reused_row in zip(baseline, reused):
            self.assertAlmostEqual(base_row["seq_logprob"], reused_row["seq_logprob"], places=6)
            self.assertAlmostEqual(base_row["mean_logprob"], reused_row["mean_logprob"], places=6)
            self.assertEqual(base_row["scored_token_count"], reused_row["scored_token_count"])
            self.assertEqual(base_row["prompt_token_count"], reused_row["prompt_token_count"])
            self.assertEqual(base_row["token_ids"], reused_row["token_ids"])
            self.assertEqual(len(base_row["token_logprobs"]), len(reused_row["token_logprobs"]))
            self.assertTrue(torch.allclose(base_row["token_logits"], reused_row["token_logits"], atol=1e-6, rtol=0.0))

    def test_seq_logprob_only_matches_standard_path(self):
        loaded = _make_loaded("base")
        prompts = ["hello", "abc", "longer prompt string"]
        continuations = ["world", " 123", " and continuation"]
        max_len = 20
        prepared = prepare_continuation_first_batch(loaded.tokenizer, prompts, continuations, max_len)
        standard = realized_token_logprobs_batch(
            loaded,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
            prepared_batch=prepared,
        )
        seq_only = realized_token_logprobs_batch(
            loaded,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
            prepared_batch=prepared,
            seq_logprob_only=True,
        )
        for standard_row, seq_row in zip(standard, seq_only):
            self.assertAlmostEqual(standard_row["seq_logprob"], seq_row["seq_logprob"], places=6)
            self.assertAlmostEqual(standard_row["mean_logprob"], seq_row["mean_logprob"], places=6)
            self.assertEqual(standard_row["scored_token_count"], seq_row["scored_token_count"])
            self.assertEqual(standard_row["prompt_token_count"], seq_row["prompt_token_count"])
            self.assertEqual(seq_row["token_logprobs"], [])
            self.assertEqual(seq_row["token_ids"], [])
            self.assertIsNone(seq_row["token_logits"])

    def test_cpk_triplet_seq_only_ratio_parity(self):
        draft = _make_loaded("draft", target_boost=4.2)
        q1 = _make_loaded("q1", target_boost=3.8)
        q2 = _make_loaded("q2", target_boost=3.6)
        prompts = ["prompt a", "prompt b", "prompt c"]
        continuations = ["cont a", "cont b", "cont c"]
        max_len = 24
        prepared = prepare_continuation_first_batch(draft.tokenizer, prompts, continuations, max_len)

        standard_p = realized_token_logprobs_batch(
            draft,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
        )
        standard_q1 = realized_token_logprobs_batch(
            q1,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
        )
        standard_q2 = realized_token_logprobs_batch(
            q2,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
        )
        seq_p = realized_token_logprobs_batch(
            draft,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
            prepared_batch=prepared,
            seq_logprob_only=True,
        )
        seq_q1 = realized_token_logprobs_batch(
            q1,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
            prepared_batch=prepared,
            seq_logprob_only=True,
        )
        seq_q2 = realized_token_logprobs_batch(
            q2,
            prompts,
            continuations,
            score_prompt_tokens=False,
            max_length=max_len,
            return_token_logprobs=False,
            return_token_logits=False,
            prepared_batch=prepared,
            seq_logprob_only=True,
        )
        for idx in range(len(prompts)):
            std_ratio_q1 = standard_p[idx]["seq_logprob"] - standard_q1[idx]["seq_logprob"]
            std_ratio_q2 = standard_p[idx]["seq_logprob"] - standard_q2[idx]["seq_logprob"]
            seq_ratio_q1 = seq_p[idx]["seq_logprob"] - seq_q1[idx]["seq_logprob"]
            seq_ratio_q2 = seq_p[idx]["seq_logprob"] - seq_q2[idx]["seq_logprob"]
            self.assertAlmostEqual(std_ratio_q1, seq_ratio_q1, places=6)
            self.assertAlmostEqual(std_ratio_q2, seq_ratio_q2, places=6)
            self.assertEqual(seq_p[idx]["token_logprobs"], [])
            self.assertEqual(seq_q1[idx]["token_logprobs"], [])
            self.assertEqual(seq_q2[idx]["token_logprobs"], [])


if __name__ == "__main__":
    unittest.main()
