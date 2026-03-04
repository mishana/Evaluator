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

"""Tests for native harness registry and plugin discovery."""

import pytest


class TestNativeHarnessRegistry:
    """Tests for core/native_harness.py registry."""

    def setup_method(self):
        """Reset the registry state before each test."""
        import nemo_evaluator.core.native_harness as nh_module
        nh_module._NATIVE_HARNESS_REGISTRY.clear()
        nh_module._discovered = False

    def test_register_and_retrieve_harness(self):
        from nemo_evaluator.core.native_harness import (
            get_native_harness,
            register_native_harness,
        )

        class FakeHarness:
            def execute(self, evaluation, model_call_fn):
                pass

        register_native_harness("fake_prefix", FakeHarness)
        harness = get_native_harness("fake_prefix_benchmark")
        assert isinstance(harness, FakeHarness)

    def test_get_harness_not_found_raises(self):
        from nemo_evaluator.core.native_harness import get_native_harness
        with pytest.raises(ValueError, match="No native harness registered"):
            get_native_harness("nonexistent_benchmark")

    def test_longest_prefix_matched_first(self):
        from nemo_evaluator.core.native_harness import (
            get_native_harness,
            register_native_harness,
        )

        class ShortPrefixHarness:
            def execute(self, evaluation, model_call_fn):
                return "short"

        class LongPrefixHarness:
            def execute(self, evaluation, model_call_fn):
                return "long"

        register_native_harness("nemo_", ShortPrefixHarness)
        register_native_harness("nemo_skills_", LongPrefixHarness)

        harness = get_native_harness("nemo_skills_aime24")
        assert isinstance(harness, LongPrefixHarness)

    def test_register_overrides_existing(self):
        from nemo_evaluator.core.native_harness import (
            get_native_harness,
            register_native_harness,
        )

        class Harness1:
            def execute(self, evaluation, model_call_fn):
                pass

        class Harness2:
            def execute(self, evaluation, model_call_fn):
                pass

        register_native_harness("prefix_", Harness1)
        register_native_harness("prefix_", Harness2)

        harness = get_native_harness("prefix_test")
        assert isinstance(harness, Harness2)


class TestNativeHarnessProtocol:
    """Tests for NativeHarness protocol compliance."""

    def test_skills_harness_implements_protocol(self):
        from unittest.mock import MagicMock, patch
        # Patch client import to avoid dependency on httpx/openai
        with patch.dict("sys.modules", {
            "httpx": MagicMock(),
            "openai": MagicMock(),
            "nemo_evaluator.client.client": MagicMock(),
            "nemo_evaluator.client": MagicMock(),
        }):
            from nemo_evaluator.core.native_harness import NativeHarness
            from nemo_evaluator.plugins.nemo_skills.native_harness import SkillsNativeHarness
            harness = SkillsNativeHarness()
            assert isinstance(harness, NativeHarness)

    def test_skills_harness_has_execute_method(self):
        from unittest.mock import MagicMock, patch
        with patch.dict("sys.modules", {
            "httpx": MagicMock(),
            "openai": MagicMock(),
            "nemo_evaluator.client.client": MagicMock(),
            "nemo_evaluator.client": MagicMock(),
        }):
            from nemo_evaluator.plugins.nemo_skills.native_harness import SkillsNativeHarness
            harness = SkillsNativeHarness()
            assert callable(getattr(harness, "execute", None))


class TestEnsureDiscovered:
    """Tests for _ensure_discovered() lazy loading."""

    def setup_method(self):
        """Reset discovery state."""
        import nemo_evaluator.core.native_harness as nh_module
        nh_module._NATIVE_HARNESS_REGISTRY.clear()
        nh_module._discovered = False

    def test_discovery_runs_once(self):
        from nemo_evaluator.core.native_harness import _ensure_discovered
        import nemo_evaluator.core.native_harness as nh_module

        _ensure_discovered()
        assert nh_module._discovered is True

        # Run again - should be idempotent
        _ensure_discovered()
        assert nh_module._discovered is True

    def test_discovery_registers_nemo_skills_if_available(self):
        from nemo_evaluator.core.native_harness import _ensure_discovered
        import nemo_evaluator.core.native_harness as nh_module

        _ensure_discovered()
        # nemo_skills harness should be registered if import succeeded
        if "nemo_skills" in nh_module._NATIVE_HARNESS_REGISTRY:
            assert callable(nh_module._NATIVE_HARNESS_REGISTRY["nemo_skills"])


class TestCoreEvalsNamespace:
    """Tests for core_evals namespace package."""

    def test_core_evals_importable(self):
        import core_evals  # noqa: F401

    def test_nemo_skills_plugin_importable(self):
        import core_evals.nemo_skills  # noqa: F401

    def test_framework_yml_exists(self):
        import importlib.resources
        try:
            ref = importlib.resources.files("core_evals.nemo_skills").joinpath("framework.yml")
            assert ref.is_file()
        except AttributeError:
            # Python 3.8 fallback
            import importlib.util
            spec = importlib.util.find_spec("core_evals.nemo_skills")
            if spec and spec.origin:
                from pathlib import Path
                framework_yml = Path(spec.origin).parent / "framework.yml"
                assert framework_yml.exists()
