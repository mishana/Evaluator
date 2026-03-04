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

"""Tests for nemo_evaluator.plugins.nemo_skills.metrics_adapter."""

import pytest

from nemo_evaluator.api.api_dataclasses import EvaluationResult, TaskResult
from nemo_evaluator.plugins.nemo_skills.metrics_adapter import translate


class TestTranslateBasic:
    """Basic translate() functionality tests."""

    def test_returns_evaluation_result(self):
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        assert isinstance(result, EvaluationResult)

    def test_task_keyed_by_benchmark_name(self):
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        assert "gsm8k" in result.tasks

    def test_task_result_has_metrics(self):
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        task_result = result.tasks["gsm8k"]
        assert isinstance(task_result, TaskResult)
        assert "greedy" in task_result.metrics

    def test_score_value_preserved(self):
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        score = result.tasks["gsm8k"].metrics["greedy"].scores["symbolic_correct"]
        assert score.value == 75.0

    def test_count_fields_excluded(self):
        """num_entries, num_prompts, num_instructions must not appear as scores."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                    "num_prompts": 100,
                    "num_instructions": 50,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        scores = result.tasks["gsm8k"].metrics["greedy"].scores
        assert "num_entries" not in scores
        assert "num_prompts" not in scores
        assert "num_instructions" not in scores
        assert "symbolic_correct" in scores


class TestTranslateMultipleAggModes:
    """Tests for multiple aggregation modes (greedy, pass@1, etc.)."""

    def test_multiple_agg_modes_all_present(self):
        ns_metrics = {
            "_all_": {
                "greedy": {"symbolic_correct": 70.0, "num_entries": 100},
                "pass@1": {"symbolic_correct": 75.0, "num_entries": 100},
            }
        }
        result = translate(ns_metrics, "aime24")
        metrics = result.tasks["aime24"].metrics
        assert "greedy" in metrics
        assert "pass@1" in metrics

    def test_different_scores_per_mode(self):
        ns_metrics = {
            "_all_": {
                "greedy": {"symbolic_correct": 70.0, "num_entries": 100},
                "pass@1": {"symbolic_correct": 75.0, "num_entries": 100},
            }
        }
        result = translate(ns_metrics, "aime24")
        assert result.tasks["aime24"].metrics["greedy"].scores["symbolic_correct"].value == 70.0
        assert result.tasks["aime24"].metrics["pass@1"].scores["symbolic_correct"].value == 75.0


class TestTranslateSubsets:
    """Tests for named subset -> GroupResult translation."""

    def test_named_subset_creates_group(self):
        ns_metrics = {
            "_all_": {
                "greedy": {"symbolic_correct": 75.0, "num_entries": 100}
            },
            "subset_easy": {
                "greedy": {"symbolic_correct": 90.0, "num_entries": 50}
            },
        }
        result = translate(ns_metrics, "gsm8k")
        assert "gsm8k" in result.groups
        assert "subset_easy" in result.groups["gsm8k"].groups

    def test_subset_score_correct(self):
        ns_metrics = {
            "_all_": {
                "greedy": {"symbolic_correct": 75.0, "num_entries": 100}
            },
            "subset_easy": {
                "greedy": {"symbolic_correct": 90.0, "num_entries": 50}
            },
        }
        result = translate(ns_metrics, "gsm8k")
        subset_metrics = result.groups["gsm8k"].groups["subset_easy"].metrics
        assert "greedy" in subset_metrics
        assert subset_metrics["greedy"].scores["symbolic_correct"].value == 90.0

    def test_config_key_ignored(self):
        """'config' key at top level should not become a group."""
        ns_metrics = {
            "_all_": {
                "greedy": {"symbolic_correct": 75.0, "num_entries": 100}
            },
            "config": {"benchmark_name": "gsm8k"},
        }
        result = translate(ns_metrics, "gsm8k")
        # config key should not appear as a group
        if result.groups:
            assert "config" not in result.groups.get("gsm8k", {}).groups


class TestTranslateErrorCases:
    """Tests for error conditions."""

    def test_missing_all_key_raises_value_error(self):
        ns_metrics = {
            "greedy": {"symbolic_correct": 75.0}
        }
        with pytest.raises(ValueError, match="_all_"):
            translate(ns_metrics, "gsm8k")

    def test_empty_all_raises_value_error(self):
        ns_metrics = {"_all_": {}}
        with pytest.raises(ValueError):
            translate(ns_metrics, "gsm8k")

    def test_none_all_raises_value_error(self):
        ns_metrics = {"_all_": None}
        with pytest.raises((ValueError, TypeError)):
            translate(ns_metrics, "gsm8k")


class TestTranslateStatistics:
    """Tests for ScoreStats scaling (INV-005)."""

    def test_stats_mean_scaled_from_fraction(self):
        """If mean is in [0,1], it should be scaled to percentage."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                    "symbolic_correct_statistics": {
                        "avg": 0.75,
                        "std_err_across_runs": 0.02,
                        "std_dev_across_runs": 0.05,
                    },
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        score = result.tasks["gsm8k"].metrics["greedy"].scores["symbolic_correct"]
        assert score.stats is not None
        assert score.stats.mean == pytest.approx(75.0, rel=1e-3)

    def test_stats_stderr_scaled(self):
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                    "symbolic_correct_statistics": {
                        "avg": 0.75,
                        "std_err_across_runs": 0.02,
                    },
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        score = result.tasks["gsm8k"].metrics["greedy"].scores["symbolic_correct"]
        assert score.stats.stderr == pytest.approx(2.0, rel=1e-3)

    def test_count_in_stats_not_scaled(self):
        """num_entries count should not be scaled."""
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        score = result.tasks["gsm8k"].metrics["greedy"].scores["symbolic_correct"]
        assert score.stats.count == 100

    def test_non_numeric_values_skipped(self):
        ns_metrics = {
            "_all_": {
                "greedy": {
                    "symbolic_correct": 75.0,
                    "model_name": "gpt-4",  # non-numeric, should be skipped
                    "num_entries": 100,
                }
            }
        }
        result = translate(ns_metrics, "gsm8k")
        scores = result.tasks["gsm8k"].metrics["greedy"].scores
        assert "model_name" not in scores
        assert "symbolic_correct" in scores
