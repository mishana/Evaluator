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

"""Tests for ExecutionMode integration with api_dataclasses and evaluate pipeline."""

import pytest

from nemo_evaluator.api.api_dataclasses import (
    ConfigParams,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
    ExecutionMode,
)


class TestExecutionModeEnum:
    """Tests for the ExecutionMode enum."""

    def test_subprocess_value(self):
        assert ExecutionMode.SUBPROCESS == "subprocess"

    def test_native_value(self):
        assert ExecutionMode.NATIVE == "native"

    def test_is_string_enum(self):
        assert isinstance(ExecutionMode.SUBPROCESS, str)
        assert isinstance(ExecutionMode.NATIVE, str)

    def test_from_string_subprocess(self):
        mode = ExecutionMode("subprocess")
        assert mode == ExecutionMode.SUBPROCESS

    def test_from_string_native(self):
        mode = ExecutionMode("native")
        assert mode == ExecutionMode.NATIVE


class TestEvaluationCommandOptional:
    """Tests for optional command field in Evaluation."""

    def test_evaluation_allows_none_command(self):
        evaluation = Evaluation(
            framework_name="test",
            pkg_name="test_pkg",
            execution_mode=ExecutionMode.NATIVE,
            config=EvaluationConfig(
                output_dir="/tmp",
                params=ConfigParams(),
            ),
            target=EvaluationTarget(),
        )
        assert evaluation.command is None

    def test_evaluation_default_execution_mode_is_subprocess(self):
        evaluation = Evaluation(
            command="echo test",
            framework_name="test",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                output_dir="/tmp",
                params=ConfigParams(),
            ),
            target=EvaluationTarget(),
        )
        assert evaluation.execution_mode == ExecutionMode.SUBPROCESS

    def test_evaluation_accepts_native_mode(self):
        evaluation = Evaluation(
            framework_name="test",
            pkg_name="test_pkg",
            execution_mode=ExecutionMode.NATIVE,
            config=EvaluationConfig(
                output_dir="/tmp",
                params=ConfigParams(),
            ),
            target=EvaluationTarget(),
        )
        assert evaluation.execution_mode == ExecutionMode.NATIVE

    def test_evaluation_subprocess_requires_command(self):
        """Subprocess mode without command should raise ValueError."""
        with pytest.raises(ValueError):
            Evaluation(
                framework_name="test",
                pkg_name="test_pkg",
                execution_mode=ExecutionMode.SUBPROCESS,
                command=None,
                config=EvaluationConfig(
                    output_dir="/tmp",
                    params=ConfigParams(),
                ),
                target=EvaluationTarget(),
            )

    def test_render_command_raises_for_none_command(self):
        evaluation = Evaluation(
            framework_name="test",
            pkg_name="test_pkg",
            execution_mode=ExecutionMode.NATIVE,
            config=EvaluationConfig(
                output_dir="/tmp",
                params=ConfigParams(),
            ),
            target=EvaluationTarget(),
        )
        # render_command() should raise ValueError when command is None
        with pytest.raises(ValueError, match="command template"):
            evaluation.render_command()

    def test_render_command_works_with_command(self):
        evaluation = Evaluation(
            command="echo hello",
            framework_name="test",
            pkg_name="test_pkg",
            config=EvaluationConfig(
                output_dir="/tmp",
                params=ConfigParams(),
            ),
            target=EvaluationTarget(),
        )
        rendered = evaluation.render_command()
        assert rendered is not None
        assert "echo hello" in rendered


class TestExecutionModeExported:
    """Tests that ExecutionMode is exported from top-level package."""

    def test_execution_mode_importable_from_nemo_evaluator(self):
        from nemo_evaluator import ExecutionMode as EM
        assert EM.NATIVE == "native"
        assert EM.SUBPROCESS == "subprocess"
