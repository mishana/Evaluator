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

"""Shared fixtures for nemo-skills unit tests."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
    ExecutionMode,
)

# --- Sample data constants ---

MATH_SAMPLES = [
    {
        "problem": "What is 2 + 2?",
        "expected_answer": "4",
        "answer": "4",
    },
    {
        "problem": "What is 3 * 7?",
        "expected_answer": "21",
        "answer": "21",
    },
    {
        "problem": "What is the square root of 144?",
        "expected_answer": "12",
        "answer": "12",
    },
]

MULTICHOICE_SAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "expected_answer": "B",
        "answer": "B",
    },
    {
        "question": "Which element has atomic number 1?",
        "choices": ["Helium", "Oxygen", "Hydrogen", "Carbon"],
        "expected_answer": "C",
        "answer": "C",
    },
]


# --- Fixtures ---


@pytest.fixture
def mock_client():
    """Returns a mock NeMoEvaluatorClient that returns boxed answers for math."""
    client = MagicMock()
    client.chat_completion = AsyncMock(return_value="The answer is \\boxed{42}")
    return client


@pytest.fixture
def benchmark_data_dir(tmp_path):
    """Factory fixture: creates JSONL benchmark data files under tmp_path.

    Usage:
        data_dir = benchmark_data_dir({"gsm8k": MATH_SAMPLES})
    """
    def _factory(benchmarks: Dict[str, List[Dict[str, Any]]]) -> str:
        for benchmark_name, samples in benchmarks.items():
            bench_dir = tmp_path / benchmark_name
            bench_dir.mkdir(parents=True, exist_ok=True)
            test_file = bench_dir / "test.jsonl"
            with open(test_file, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
        return str(tmp_path)

    return _factory


@pytest.fixture
def make_evaluation(tmp_path):
    """Factory fixture: creates Evaluation objects with nemo-skills config."""
    def _factory(
        benchmark_name: str = "gsm8k",
        eval_type: str = "math",
        data_dir: Optional[str] = None,
        eval_split: str = "test",
        num_seeds: int = 1,
        limit_samples: Optional[int] = None,
        execution_mode: ExecutionMode = ExecutionMode.NATIVE,
        output_dir: Optional[str] = None,
        model_url: str = "http://localhost:8000/v1",
        model_id: str = "test-model",
        system_prompt: Optional[str] = None,
    ) -> Evaluation:
        extra = {
            "benchmark_name": benchmark_name,
            "eval_type": eval_type,
            "data_dir": data_dir or str(tmp_path),
            "eval_split": eval_split,
            "num_seeds": num_seeds,
        }
        if system_prompt is not None:
            extra["system_prompt"] = system_prompt

        return Evaluation(
            framework_name="nemo_skills",
            pkg_name=f"nemo_skills.ns_{benchmark_name}",
            execution_mode=execution_mode,
            config=EvaluationConfig(
                output_dir=output_dir or str(tmp_path),
                params=ConfigParams(
                    temperature=0.0,
                    max_new_tokens=512,
                    limit_samples=limit_samples,
                    extra=extra,
                ),
            ),
            target=EvaluationTarget(
                api_endpoint=ApiEndpoint(
                    url=model_url,
                    model_id=model_id,
                    type="chat",
                )
            ),
        )

    return _factory
