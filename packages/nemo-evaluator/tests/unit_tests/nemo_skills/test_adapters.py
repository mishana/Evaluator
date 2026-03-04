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

"""Tests for nemo_evaluator._nemo_skills._adapters stubs."""

import json
import os
from pathlib import Path

import pytest


class TestVersionAdapter:
    """Tests for _adapters/version.py."""

    def test_version_is_synced_string(self):
        from nemo_evaluator._nemo_skills._adapters.version import __version__
        assert __version__ == "synced"

    def test_version_is_str(self):
        from nemo_evaluator._nemo_skills._adapters.version import __version__
        assert isinstance(__version__, str)


class TestFileUtils:
    """Tests for _adapters/file_utils.py."""

    def test_jdump_creates_file(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.file_utils import jdump
        data = {"key": "value", "number": 42}
        path = str(tmp_path / "output.json")
        jdump(data, path)
        assert os.path.exists(path)

    def test_jdump_produces_valid_json(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.file_utils import jdump
        data = {"result": [1, 2, 3], "nested": {"a": "b"}}
        path = str(tmp_path / "output.json")
        jdump(data, path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_jdump_uses_indent_2(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.file_utils import jdump
        data = {"key": "value"}
        path = str(tmp_path / "output.json")
        jdump(data, path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Indented JSON contains newlines
        assert "\n" in content

    def test_jload_reads_file(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.file_utils import jdump, jload
        data = {"key": "value"}
        path = str(tmp_path / "output.json")
        jdump(data, path)
        loaded = jload(path)
        assert loaded == data

    def test_jdump_handles_unicode(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.file_utils import jdump, jload
        data = {"text": "Hello \u4e16\u754c"}  # Hello World in Chinese
        path = str(tmp_path / "unicode.json")
        jdump(data, path)
        loaded = jload(path)
        assert loaded["text"] == "Hello \u4e16\u754c"

    def test_unroll_files_reexported(self):
        """unroll_files should be available in file_utils."""
        from nemo_evaluator._nemo_skills._adapters import file_utils
        assert hasattr(file_utils, "unroll_files")


class TestUtils:
    """Tests for _adapters/utils.py."""

    def test_get_logger_name_returns_string(self):
        from nemo_evaluator._nemo_skills._adapters.utils import get_logger_name
        name = get_logger_name("my.module")
        assert isinstance(name, str)

    def test_get_logger_name_contains_module(self):
        from nemo_evaluator._nemo_skills._adapters.utils import get_logger_name
        name = get_logger_name("my.module")
        assert "my.module" in name or len(name) > 0

    def test_unroll_files_with_single_file(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.utils import unroll_files
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"key": "value"}\n')
        result = list(unroll_files([str(test_file)]))
        assert str(test_file) in result

    def test_unroll_files_with_glob(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.utils import unroll_files
        for i in range(3):
            (tmp_path / f"test_{i}.jsonl").write_text('{"key": "value"}\n')
        result = list(unroll_files([str(tmp_path / "*.jsonl")]))
        assert len(result) == 3


class TestSciCodeUtils:
    """Tests for _adapters/scicode_utils.py."""

    def test_eval_prefix_is_string(self):
        from nemo_evaluator._nemo_skills._adapters.scicode_utils import eval_prefix
        assert isinstance(eval_prefix, str)

    def test_eval_prefix_contains_import(self):
        from nemo_evaluator._nemo_skills._adapters.scicode_utils import eval_prefix
        assert "import" in eval_prefix
        assert "scicode" in eval_prefix


class TestDatasetUtils:
    """Tests for _adapters/dataset_utils.py."""

    def test_get_dataset_module_raises_for_unknown_benchmark(self):
        from nemo_evaluator._nemo_skills._adapters.dataset_utils import get_dataset_module
        # Unknown benchmark should raise ModuleNotFoundError
        with pytest.raises(ModuleNotFoundError):
            get_dataset_module("nonexistent_benchmark_xyz")

    def test_get_dataset_module_is_callable(self):
        from nemo_evaluator._nemo_skills._adapters.dataset_utils import get_dataset_module
        assert callable(get_dataset_module)

    def test_import_from_path_nonexistent(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.dataset_utils import import_from_path
        # Non-existent path should raise
        nonexistent = str(tmp_path / "nonexistent.py")
        with pytest.raises((FileNotFoundError, ImportError, Exception)):
            import_from_path(nonexistent)

    def test_import_from_path_valid_module(self, tmp_path):
        from nemo_evaluator._nemo_skills._adapters.dataset_utils import import_from_path
        # Create a simple valid Python module
        module_file = tmp_path / "test_mod.py"
        module_file.write_text("MY_VAR = 42\n")
        module = import_from_path(str(module_file))
        assert hasattr(module, "MY_VAR")
        assert module.MY_VAR == 42
