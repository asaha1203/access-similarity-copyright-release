import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ABS_SYSTEM_PROMPT = (
    "You are a strict evaluation assistant. Follow the required output format exactly."
)

ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write concise feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Output: """

DEFAULT_FLUENCY_RUBRIC = """[Is the response fluent, coherent, grammatically correct, and easy to read?]
Score 1: The response is largely incoherent or ungrammatical, with severe fluency issues that make it difficult to understand.
Score 2: The response has frequent fluency and grammar problems, and readability is poor.
Score 3: The response is mostly understandable but contains noticeable fluency or grammar issues.
Score 4: The response is clear and generally fluent, with only minor fluency or grammar issues.
Score 5: The response is highly fluent, coherent, grammatically correct, and consistently easy to read.
Additional instruction: Keep the feedback concise and short.
""".strip()

_RESULT_PATTERN = re.compile(r"\[RESULT\][^\d]*([1-5])(?:\b|$)", flags=re.IGNORECASE)
_SCORE_FALLBACK_PATTERN = re.compile(r"\b(?:score|rating)\s*[:=\-]?\s*([1-5])(?:\b|$)", flags=re.IGNORECASE)
_DIGIT_FALLBACK_PATTERN = re.compile(r"\b([1-5])\b")


def parse_prometheus_result_score(text: str) -> Tuple[Optional[float], bool, str]:
    if not text:
        return None, False, "empty_output"
    marker_idx = text.rfind("[RESULT]")
    search_text = text if marker_idx < 0 else text[marker_idx:]
    match = _RESULT_PATTERN.search(search_text)
    if not match:
        fallback_matches = list(_SCORE_FALLBACK_PATTERN.finditer(text))
        if fallback_matches:
            match = fallback_matches[-1]
        else:
            digit_matches = list(_DIGIT_FALLBACK_PATTERN.finditer(text))
            if not digit_matches:
                return None, False, "missing_or_invalid_result_token"
            match = digit_matches[-1]
    try:
        score = float(int(match.group(1)))
    except Exception:
        return None, False, "invalid_score_parse"
    return score, True, ""


def _build_absolute_prompt(instruction: str, response: str, rubric: str) -> str:
    return ABSOLUTE_PROMPT_WO_REF.format(
        instruction=instruction,
        response=response,
        rubric=rubric,
    )


@dataclass
class PrometheusV2Scorer:
    model_name_or_path: str
    device: torch.device
    batch_size: int = 4
    max_input_tokens: int = 2048
    max_new_tokens: int = 24
    rubric: str = DEFAULT_FLUENCY_RUBRIC
    torch_dtype: Optional[torch.dtype] = None
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    attention_implementation: Optional[str] = None

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=import-outside-toplevel

        self.batch_size = max(1, int(self.batch_size))
        self.max_input_tokens = max(64, int(self.max_input_tokens))
        self.max_new_tokens = max(8, int(self.max_new_tokens))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        model_kwargs: Dict[str, Any] = {}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        if self.attention_implementation and str(self.attention_implementation).lower() != "auto":
            model_kwargs["attn_implementation"] = str(self.attention_implementation)

        model = None
        last_exc: Optional[Exception] = None
        attempts: List[Optional[str]] = []
        if "attn_implementation" in model_kwargs:
            requested = str(model_kwargs["attn_implementation"])
            attempts.extend([requested, "sdpa" if requested == "flash_attention_3" else None, None])
        else:
            attempts.append(None)
        for attn_impl in attempts:
            try:
                kwargs = dict(model_kwargs)
                if attn_impl is None:
                    kwargs.pop("attn_implementation", None)
                else:
                    kwargs["attn_implementation"] = attn_impl
                model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)
                break
            except Exception as exc:
                last_exc = exc
        if model is None:
            raise RuntimeError(
                f"Failed to load PrometheusV2 model {self.model_name_or_path}."
            ) from last_exc

        self.model = model.to(self.device)
        self.model.eval()
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = True
        if self.compile_model and hasattr(torch, "compile"):
            try:
                self.model.forward = torch.compile(
                    self.model.forward,
                    mode=str(self.compile_mode),
                    fullgraph=False,
                )  # type: ignore[method-assign]
            except Exception:
                pass

    def _render_chat_prompt(self, instruction: str, response: str) -> str:
        user_prompt = _build_absolute_prompt(
            instruction=instruction,
            response=response,
            rubric=self.rubric,
        )
        messages = [
            {"role": "system", "content": ABS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def score_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        instruction_key: str = "prompt",
        response_key: str = "generated_text",
        show_progress: bool = True,
        progress_desc: str = "PrometheusV2 scoring",
        progress_leave: bool = False,
    ) -> List[Dict[str, Any]]:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

        prompts = [
            self._render_chat_prompt(
                instruction=str(row.get(instruction_key, "")),
                response=str(row.get(response_key, "")),
            )
            for row in rows
        ]

        results: List[Dict[str, Any]] = []
        iterator = range(0, len(prompts), self.batch_size)
        progress = tqdm(
            iterator,
            desc=progress_desc,
            unit="batch",
            leave=progress_leave,
            disable=not bool(show_progress),
        )
        for start in progress:
            prompt_batch = prompts[start : start + self.batch_size]
            tokenized = self.tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)
            input_len = tokenized["input_ids"].shape[1]

            with torch.inference_mode():
                generated = self.model.generate(
                    **tokenized,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_suffix = generated[:, input_len:]
            decoded = self.tokenizer.batch_decode(generated_suffix, skip_special_tokens=True)
            for text in decoded:
                score, parse_ok, parse_error = parse_prometheus_result_score(text)
                results.append(
                    {
                        "prometheusv2_fluency": score,
                        "parse_ok": bool(parse_ok),
                        "parse_error": str(parse_error),
                        "raw_output_len": len(text),
                    }
                )
            progress.set_postfix(scored=min(start + len(prompt_batch), len(prompts)), total=len(prompts), refresh=False)
        return results
