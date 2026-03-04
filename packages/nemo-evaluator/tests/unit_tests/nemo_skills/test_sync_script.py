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

"""Tests for scripts/sync_skills.py functions."""

import os
import sys
from pathlib import Path

import pytest
import yaml

# Add scripts directory to path for import
# parents[0] = nemo_skills/, parents[1] = unit_tests/, parents[2] = tests/, parents[3] = nemo-evaluator/
SCRIPTS_DIR = Path(__file__).parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from sync_skills import (
    _collect_files,
    _is_protected,
    _matches_any_pattern,
    apply_rewrites,
    copy_files,
    load_manifest,
    validate_imports,
)


# --- Sample manifest for testing ---

SAMPLE_MANIFEST = {
    "source_repo": "github.com/NVIDIA-NeMo/Skills",
    "source_ref": "v1.0.0",
    "target_dir": "src/nemo_evaluator/_nemo_skills",
    "pip_dependency_range": ">=1.0.0,<2.0.0",
    "sync_dirs": [
        {
            "source": "nemo_skills/evaluation",
            "target": "evaluation",
            "include": ["*.py"],
            "exclude": ["__pycache__"],
        }
    ],
    "import_rewrite_rules": [
        {"pattern": "from nemo_skills", "replacement": "from nemo_evaluator._nemo_skills"},
    ],
    "import_allowlist": ["nemo_skills.code_execution"],
    "protected_files": ["_adapters/*.py"],
}


class TestLoadManifest:
    """Tests for load_manifest()."""

    def test_loads_valid_manifest(self, tmp_path):
        manifest_file = tmp_path / "SYNC_MANIFEST.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(SAMPLE_MANIFEST, f)
        result = load_manifest(str(manifest_file))
        assert result["source_repo"] == "github.com/NVIDIA-NeMo/Skills"

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(str(tmp_path / "nonexistent.yaml"))

    def test_raises_for_missing_required_keys(self, tmp_path):
        incomplete_manifest = {"source_repo": "test"}
        manifest_file = tmp_path / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(incomplete_manifest, f)
        with pytest.raises(ValueError, match="missing required keys"):
            load_manifest(str(manifest_file))

    def test_raises_for_empty_source_repo(self, tmp_path):
        bad_manifest = dict(SAMPLE_MANIFEST)
        bad_manifest["source_repo"] = ""
        manifest_file = tmp_path / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(bad_manifest, f)
        with pytest.raises(ValueError, match="source_repo"):
            load_manifest(str(manifest_file))

    def test_raises_for_identity_rewrite_rule(self, tmp_path):
        bad_manifest = dict(SAMPLE_MANIFEST)
        bad_manifest["import_rewrite_rules"] = [
            {"pattern": "nemo_skills", "replacement": "nemo_skills"}  # identity
        ]
        manifest_file = tmp_path / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(bad_manifest, f)
        with pytest.raises(ValueError, match="Identity rewrite rule"):
            load_manifest(str(manifest_file))

    def test_raises_for_empty_sync_dirs(self, tmp_path):
        bad_manifest = dict(SAMPLE_MANIFEST)
        bad_manifest["sync_dirs"] = []
        manifest_file = tmp_path / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(bad_manifest, f)
        with pytest.raises(ValueError, match="sync_dirs"):
            load_manifest(str(manifest_file))


class TestMatchesAnyPattern:
    """Tests for _matches_any_pattern()."""

    def test_exact_filename_match(self):
        assert _matches_any_pattern("__init__.py", ["__init__.py"])

    def test_glob_star_match(self):
        assert _matches_any_pattern("foo.py", ["*.py"])

    def test_no_match(self):
        assert not _matches_any_pattern("foo.txt", ["*.py"])

    def test_multiple_patterns_any_match(self):
        assert _matches_any_pattern("test.py", ["*.txt", "*.py"])

    def test_basename_match(self):
        assert _matches_any_pattern("subdir/test.py", ["*.py"])


class TestIsProtected:
    """Tests for _is_protected()."""

    def test_protected_file(self):
        assert _is_protected("_adapters/version.py", ["_adapters/*.py"])

    def test_not_protected(self):
        assert not _is_protected("evaluation/math.py", ["_adapters/*.py"])

    def test_empty_patterns(self):
        assert not _is_protected("any/file.py", [])


class TestApplyRewrites:
    """Tests for apply_rewrites()."""

    def test_rewrites_import_statement(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("from nemo_skills import something\n")
        count = apply_rewrites(str(py_file), [
            {"pattern": "from nemo_skills", "replacement": "from nemo_evaluator._nemo_skills"}
        ])
        assert count == 1
        assert "from nemo_evaluator._nemo_skills" in py_file.read_text()

    def test_returns_replacement_count(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("from nemo_skills import a\nfrom nemo_skills import b\n")
        count = apply_rewrites(str(py_file), [
            {"pattern": "from nemo_skills", "replacement": "from nemo_evaluator._nemo_skills"}
        ])
        assert count == 2

    def test_non_py_file_returns_zero(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("from nemo_skills import something\n")
        count = apply_rewrites(str(txt_file), [
            {"pattern": "from nemo_skills", "replacement": "from nemo_evaluator._nemo_skills"}
        ])
        assert count == 0

    def test_no_match_returns_zero(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("import os\nimport sys\n")
        count = apply_rewrites(str(py_file), [
            {"pattern": "from nemo_skills", "replacement": "from nemo_evaluator._nemo_skills"}
        ])
        assert count == 0

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            apply_rewrites("/nonexistent/path/test.py", [])

    def test_rules_applied_in_order(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("from nemo_skills.evaluation import foo\n")
        apply_rewrites(str(py_file), [
            {"pattern": "from nemo_skills.evaluation", "replacement": "from nemo_evaluator._nemo_skills.evaluation"},
            {"pattern": "from nemo_skills", "replacement": "WRONG"},
        ])
        content = py_file.read_text()
        assert "from nemo_evaluator._nemo_skills.evaluation" in content


class TestValidateImports:
    """Tests for validate_imports()."""

    def test_no_violations_clean_file(self, tmp_path):
        py_file = tmp_path / "clean.py"
        py_file.write_text("from nemo_evaluator._nemo_skills import foo\n")
        violations = validate_imports(str(tmp_path), ["nemo_evaluator"])
        assert violations == []

    def test_detects_bare_nemo_skills_import(self, tmp_path):
        py_file = tmp_path / "bad.py"
        py_file.write_text("from nemo_skills import something\n")
        violations = validate_imports(str(tmp_path), ["nemo_evaluator"])
        assert len(violations) == 1
        assert violations[0]["file_path"] == str(py_file.resolve())

    def test_allowlisted_import_not_violation(self, tmp_path):
        py_file = tmp_path / "allowed.py"
        py_file.write_text("from nemo_skills.code_execution import sandbox\n")
        violations = validate_imports(str(tmp_path), ["nemo_skills.code_execution"])
        assert violations == []

    def test_adapters_directory_excluded(self, tmp_path):
        adapters_dir = tmp_path / "_adapters"
        adapters_dir.mkdir()
        py_file = adapters_dir / "stub.py"
        py_file.write_text("from nemo_skills import something\n")
        violations = validate_imports(str(tmp_path), ["nemo_evaluator"])
        # _adapters/ is excluded from validation
        assert violations == []

    def test_nonexistent_target_returns_empty(self, tmp_path):
        violations = validate_imports(str(tmp_path / "nonexistent"), [])
        assert violations == []

    def test_violation_contains_correct_fields(self, tmp_path):
        py_file = tmp_path / "bad.py"
        py_file.write_text("import nemo_skills\n")
        violations = validate_imports(str(tmp_path), [])
        assert len(violations) == 1
        v = violations[0]
        assert "file_path" in v
        assert "line_number" in v
        assert "import_statement" in v
        assert v["line_number"] == 1


class TestCopyFiles:
    """Tests for copy_files()."""

    def test_copies_matching_files(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "eval").mkdir()
        (src / "eval" / "math.py").write_text("# math module\n")

        target = tmp_path / "target"
        sync_dirs = [
            {"source": "eval", "target": "eval", "include": ["*.py"], "exclude": []}
        ]
        copied = copy_files(str(src), str(target), sync_dirs)
        assert len(copied) == 1
        assert (target / "eval" / "math.py").exists()

    def test_raises_for_missing_source_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            copy_files(str(tmp_path / "nonexistent"), str(tmp_path / "target"), [])

    def test_protected_files_not_overwritten(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "adapters").mkdir()
        (src / "adapters" / "version.py").write_text("# upstream version\n")

        target = tmp_path / "target"
        (target / "adapters").mkdir(parents=True)
        (target / "adapters" / "version.py").write_text("# protected version\n")

        sync_dirs = [
            {"source": "adapters", "target": "adapters", "include": ["*.py"], "exclude": []}
        ]
        copy_files(str(src), str(target), sync_dirs, protected_patterns=["adapters/*.py"])
        # Protected file should NOT be overwritten
        content = (target / "adapters" / "version.py").read_text()
        assert "protected version" in content

    def test_skips_missing_source_dirs(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        target = tmp_path / "target"
        sync_dirs = [
            {"source": "nonexistent_dir", "target": "target", "include": ["*.py"], "exclude": []}
        ]
        # Should not raise, just warn and skip
        copied = copy_files(str(src), str(target), sync_dirs)
        assert copied == []
