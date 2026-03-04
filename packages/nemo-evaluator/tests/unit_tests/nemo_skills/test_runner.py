# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for nemo_evaluator.plugins.nemo_skills.runner."""

import asyncio
import json
from pathlib import Path
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.plugins.nemo_skills.runner import (
    EVALUATOR_MAP,
    _SCORER_DISPATCH,
    call_model_batch,
    compute_ns_metrics,
    construct_prompts,
    load_benchmark_data,
    score_bfcl,
    score_math,
    score_multichoice,
    score_mrcr,
    score_ruler,
)

from .conftest import MATH_SAMPLES, MULTICHOICE_SAMPLES


class TestEvaluatorMap:
    """Tests for EVALUATOR_MAP structure (C-021)."""

    def test_is_mapping_proxy(self):
        assert isinstance(EVALUATOR_MAP, MappingProxyType)

    def test_has_17_entries(self):
        assert len(EVALUATOR_MAP) == 17

    def test_all_scorers_in_dispatch(self):
        """Every scorer referenced in EVALUATOR_MAP must exist in _SCORER_DISPATCH."""
        for eval_type, scorer_name in EVALUATOR_MAP.items():
            assert scorer_name in _SCORER_DISPATCH, (
                f"Scorer '{scorer_name}' (for eval_type '{eval_type}') not in _SCORER_DISPATCH"
            )

    def test_math_types_use_score_math(self):
        for eval_type in ["math", "gsm8k", "aime24", "amc23", "aime25"]:
            assert EVALUATOR_MAP[eval_type] == "score_math"

    def test_multichoice_types(self):
        for eval_type in ["mmlu", "mmlu_pro", "gpqa"]:
            assert EVALUATOR_MAP[eval_type] == "score_multichoice"


class TestLoadBenchmarkData:
    """Tests for load_benchmark_data()."""

    def test_loads_jsonl_file(self, tmp_path):
        bench_dir = tmp_path / "gsm8k"
        bench_dir.mkdir()
        test_file = bench_dir / "test.jsonl"
        with open(test_file, "w") as f:
            for sample in MATH_SAMPLES:
                f.write(json.dumps(sample) + "\n")

        data = load_benchmark_data("gsm8k", str(tmp_path), "test")
        assert len(data) == len(MATH_SAMPLES)

    def test_fallback_to_test_split(self, tmp_path):
        """If eval_split file doesn't exist, should fall back to test.jsonl."""
        bench_dir = tmp_path / "gsm8k"
        bench_dir.mkdir()
        test_file = bench_dir / "test.jsonl"
        with open(test_file, "w") as f:
            for sample in MATH_SAMPLES[:2]:
                f.write(json.dumps(sample) + "\n")

        # Request "validation" split which doesn't exist
        data = load_benchmark_data("gsm8k", str(tmp_path), "validation")
        assert len(data) == 2

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_benchmark_data("nonexistent_benchmark", str(tmp_path), "test")

    def test_raises_value_error_for_none_data_dir(self):
        with pytest.raises(ValueError, match="data_dir"):
            load_benchmark_data("gsm8k", None, "test")

    def test_limit_samples_respected(self, tmp_path):
        bench_dir = tmp_path / "gsm8k"
        bench_dir.mkdir()
        test_file = bench_dir / "test.jsonl"
        with open(test_file, "w") as f:
            for sample in MATH_SAMPLES:
                f.write(json.dumps(sample) + "\n")

        data = load_benchmark_data("gsm8k", str(tmp_path), "test", limit_samples=2)
        assert len(data) == 2

    def test_data_parsed_as_dicts(self, tmp_path):
        bench_dir = tmp_path / "gsm8k"
        bench_dir.mkdir()
        test_file = bench_dir / "test.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps(MATH_SAMPLES[0]) + "\n")

        data = load_benchmark_data("gsm8k", str(tmp_path), "test")
        assert isinstance(data[0], dict)
        assert "problem" in data[0]


class TestConstructPrompts:
    """Tests for construct_prompts()."""

    def test_returns_list_of_message_lists(self):
        prompts = construct_prompts(MATH_SAMPLES, None, None)
        assert isinstance(prompts, list)
        assert all(isinstance(p, list) for p in prompts)
        assert len(prompts) == len(MATH_SAMPLES)

    def test_detects_problem_field(self):
        sample = {"problem": "Test question?"}
        prompts = construct_prompts([sample], None, None)
        assert prompts[0][-1]["content"] == "Test question?"
        assert prompts[0][-1]["role"] == "user"

    def test_detects_question_field(self):
        sample = {"question": "What is X?"}
        prompts = construct_prompts([sample], None, None)
        assert prompts[0][-1]["content"] == "What is X?"

    def test_detects_prompt_field(self):
        sample = {"prompt": "Solve this:"}
        prompts = construct_prompts([sample], None, None)
        assert prompts[0][-1]["content"] == "Solve this:"

    def test_falls_back_to_json_dump(self):
        sample = {"data": "value", "other": 42}
        prompts = construct_prompts([sample], None, None)
        content = prompts[0][-1]["content"]
        # Should be JSON serialization of sample
        parsed = json.loads(content)
        assert parsed == sample

    def test_system_prompt_prepended(self):
        sample = {"problem": "Test?"}
        prompts = construct_prompts([sample], None, "You are a math tutor.")
        assert len(prompts[0]) == 2
        assert prompts[0][0]["role"] == "system"
        assert prompts[0][0]["content"] == "You are a math tutor."
        assert prompts[0][1]["role"] == "user"

    def test_no_system_prompt_when_none(self):
        sample = {"problem": "Test?"}
        prompts = construct_prompts([sample], None, None)
        assert len(prompts[0]) == 1
        assert prompts[0][0]["role"] == "user"

    def test_multichoice_uses_question_field(self):
        prompts = construct_prompts(MULTICHOICE_SAMPLES, None, None)
        assert prompts[0][-1]["content"] == MULTICHOICE_SAMPLES[0]["question"]


class TestCallModelBatch:
    """Tests for call_model_batch()."""

    def test_returns_list_of_responses(self):
        client = MagicMock()
        client.chat_completion = AsyncMock(return_value="response text")
        prompts = [[{"role": "user", "content": "Q1"}]]
        responses = asyncio.run(call_model_batch(client, prompts, 0.0, 512))
        assert isinstance(responses, list)
        assert len(responses) == 1
        assert responses[0] == "response text"

    def test_fault_isolation_on_failure(self):
        """Per C-020/INV-009: failed calls return empty string, never raise."""
        client = MagicMock()
        client.chat_completion = AsyncMock(side_effect=Exception("model error"))
        prompts = [[{"role": "user", "content": "Q1"}]]
        responses = asyncio.run(call_model_batch(client, prompts, 0.0, 512))
        assert responses == [""]

    def test_mixed_success_and_failure(self):
        """Some calls succeed, some fail. Batch continues."""
        call_count = 0

        async def mock_chat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API error")
            return "success"

        client = MagicMock()
        client.chat_completion = mock_chat
        prompts = [
            [{"role": "user", "content": "Q1"}],
            [{"role": "user", "content": "Q2"}],
            [{"role": "user", "content": "Q3"}],
        ]
        responses = asyncio.run(call_model_batch(client, prompts, 0.0, 512))
        assert len(responses) == 3
        assert responses[0] == "success"
        assert responses[1] == ""  # failed call
        assert responses[2] == "success"

    def test_same_length_as_prompts(self):
        client = MagicMock()
        client.chat_completion = AsyncMock(return_value="r")
        prompts = [[{"role": "user", "content": f"Q{i}"}] for i in range(5)]
        responses = asyncio.run(call_model_batch(client, prompts, 0.0, 512))
        assert len(responses) == 5


class TestScoreMath:
    """Tests for score_math() scorer (C-022)."""

    def test_extracts_boxed_answer(self):
        data = [{"problem": "Q?", "expected_answer": "42", "generation": "The answer is \\boxed{42}"}]
        result = score_math(data, {})
        assert result[0]["predicted_answer"] == "42"
        assert result[0]["symbolic_correct"] is True

    def test_wrong_boxed_answer(self):
        data = [{"problem": "Q?", "expected_answer": "42", "generation": "\\boxed{43}"}]
        result = score_math(data, {})
        assert result[0]["symbolic_correct"] is False

    def test_no_answer_detected(self):
        data = [{"problem": "Q?", "expected_answer": "42", "generation": "I don't know."}]
        result = score_math(data, {})
        assert result[0]["no_answer"] is True
        assert result[0]["predicted_answer"] == ""

    def test_answer_is_pattern(self):
        data = [{"problem": "Q?", "expected_answer": "21", "generation": "The answer is 21"}]
        result = score_math(data, {})
        # Should extract via "answer is" pattern
        assert "21" in result[0]["predicted_answer"]

    def test_returns_same_list(self):
        """Per INV-003: should mutate and return same list object."""
        data = [{"problem": "Q?", "expected_answer": "1", "generation": "\\boxed{1}"}]
        result = score_math(data, {})
        assert result is data

    def test_multiple_samples(self):
        data = [
            {"problem": "2+2?", "expected_answer": "4", "generation": "\\boxed{4}"},
            {"problem": "3*3?", "expected_answer": "9", "generation": "\\boxed{8}"},
        ]
        result = score_math(data, {})
        assert result[0]["symbolic_correct"] is True
        assert result[1]["symbolic_correct"] is False


class TestScoreMultichoice:
    """Tests for score_multichoice() scorer (C-023)."""

    def test_extracts_boxed_letter(self):
        data = [{"question": "Q?", "expected_answer": "B", "generation": "\\boxed{B}"}]
        result = score_multichoice(data, {})
        assert result[0]["predicted_answer"] == "B"
        assert result[0]["symbolic_correct"] is True

    def test_wrong_letter(self):
        data = [{"question": "Q?", "expected_answer": "B", "generation": "\\boxed{A}"}]
        result = score_multichoice(data, {})
        assert result[0]["symbolic_correct"] is False

    def test_answer_is_pattern(self):
        data = [{"question": "Q?", "expected_answer": "C", "generation": "The answer is (C)"}]
        result = score_multichoice(data, {})
        assert result[0]["predicted_answer"] == "C"

    def test_last_standalone_letter(self):
        data = [{"question": "Q?", "expected_answer": "D", "generation": "I think it is D"}]
        result = score_multichoice(data, {})
        assert result[0]["predicted_answer"] == "D"

    def test_returns_same_list(self):
        data = [{"question": "Q?", "expected_answer": "A", "generation": "\\boxed{A}"}]
        result = score_multichoice(data, {})
        assert result is data


class TestScoreMrcr:
    """Tests for score_mrcr() scorer (C-024)."""

    def test_exact_match_has_ratio_1(self):
        data = [{"generation": "Paris", "expected_answer": "Paris"}]
        result = score_mrcr(data, {})
        assert result[0]["seq_match_ratio"] == pytest.approx(1.0)
        assert result[0]["is_correct"] is True

    def test_empty_generation_low_ratio(self):
        data = [{"generation": "", "expected_answer": "Paris"}]
        result = score_mrcr(data, {})
        assert result[0]["seq_match_ratio"] < 0.8
        assert result[0]["is_correct"] is False

    def test_returns_same_list(self):
        data = [{"generation": "X", "expected_answer": "X"}]
        result = score_mrcr(data, {})
        assert result is data


class TestScoreRuler:
    """Tests for score_ruler() scorer (C-025)."""

    def test_string_containment_true(self):
        data = [{"generation": "The answer is Paris", "expected_answer": "Paris"}]
        result = score_ruler(data, {})
        assert result[0]["is_correct"] is True

    def test_string_containment_false(self):
        data = [{"generation": "The answer is Berlin", "expected_answer": "Paris"}]
        result = score_ruler(data, {})
        assert result[0]["is_correct"] is False

    def test_list_target_partial_match(self):
        data = [{
            "generation": "apple and banana are here",
            "expected_answer": ["apple", "banana", "cherry"]
        }]
        result = score_ruler(data, {})
        # 2 out of 3 found
        assert result[0]["is_correct"] == pytest.approx(2/3, rel=1e-3)

    def test_list_target_full_match(self):
        data = [{
            "generation": "apple banana cherry",
            "expected_answer": ["apple", "banana", "cherry"]
        }]
        result = score_ruler(data, {})
        assert result[0]["is_correct"] == pytest.approx(1.0)

    def test_returns_same_list(self):
        data = [{"generation": "test", "expected_answer": "test"}]
        result = score_ruler(data, {})
        assert result is data


class TestScoreBfcl:
    """Tests for score_bfcl() scorer (C-026)."""

    def test_exact_match(self):
        data = [{"generation": "function_call(a=1)", "expected_answer": "function_call(a=1)"}]
        result = score_bfcl(data, {})
        assert result[0]["is_correct"] is True

    def test_non_match(self):
        data = [{"generation": "function_call(a=2)", "expected_answer": "function_call(a=1)"}]
        result = score_bfcl(data, {})
        assert result[0]["is_correct"] is False

    def test_whitespace_stripped(self):
        data = [{"generation": "  answer  ", "expected_answer": "answer"}]
        result = score_bfcl(data, {})
        assert result[0]["is_correct"] is True

    def test_returns_same_list(self):
        data = [{"generation": "a", "expected_answer": "a"}]
        result = score_bfcl(data, {})
        assert result is data


class TestComputeNsMetrics:
    """Tests for compute_ns_metrics()."""

    def test_returns_dict_with_greedy(self):
        data = [
            {"symbolic_correct": True},
            {"symbolic_correct": False},
        ]
        result = compute_ns_metrics(data, "gsm8k", "math")
        assert "greedy" in result

    def test_accuracy_calculation(self):
        data = [
            {"symbolic_correct": True},
            {"symbolic_correct": True},
            {"symbolic_correct": False},
            {"symbolic_correct": True},
        ]
        result = compute_ns_metrics(data, "gsm8k", "math")
        # 3/4 = 75%
        assert result["greedy"]["symbolic_correct"] == pytest.approx(75.0, rel=1e-3)

    def test_empty_data_returns_zero_entries(self):
        result = compute_ns_metrics([], "gsm8k", "math")
        assert result["greedy"]["num_entries"] == 0

    def test_num_entries_in_metrics(self):
        data = [{"symbolic_correct": True}] * 10
        result = compute_ns_metrics(data, "gsm8k", "math")
        assert result["greedy"]["num_entries"] == 10
