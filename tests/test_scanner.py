"""
tests/test_scanner.py
=====================
Unit tests for the ``src.repo_scan`` static analysis pipeline.

Each test suite is pinned to a fixture directory under
``tests/fixtures/`` and asserts on the exact structure of the returned
:class:`RepoManifest`.

Line-number and raw-line assertions are included for at least one
dependency per fixture, making the traceability guarantee concretely
testable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.repo_scan import RepoManifest, scan_repo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Path:
    """Return the absolute path to a named fixture directory."""
    return (FIXTURES_DIR / name).resolve()


# ---------------------------------------------------------------------------
# Fixture: clean_python_repo
# ---------------------------------------------------------------------------

class TestCleanPythonRepo:
    """
    A repo with only ``pyproject.toml`` (PEP 621 format).
    Expectations:
    - 1 ecosystem: python
    - 3 main deps: numpy, requests, click
    - No lock-files, no conflicts
    """

    @pytest.fixture(autouse=True)
    def manifest(self) -> RepoManifest:
        self._manifest = scan_repo(fixture("clean_python_repo"))
        return self._manifest

    def test_python_ecosystem_detected(self):
        assert "python" in self._manifest.ecosystems

    def test_no_other_ecosystems(self):
        assert "node" not in self._manifest.ecosystems
        assert len(self._manifest.ecosystems) == 1

    def test_manifest_files(self):
        eco = self._manifest.ecosystems["python"]
        names = [os.path.basename(f) for f in eco.manifest_files]
        assert "pyproject.toml" in names

    def test_no_lock_files_detected(self):
        eco = self._manifest.ecosystems["python"]
        assert eco.lock_files == []

    def test_numpy_present(self):
        deps = self._manifest.ecosystems["python"].dependencies
        assert "numpy" in deps

    def test_numpy_constraint(self):
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        assert dep.declared_constraint == ">=1.24"

    def test_numpy_sources_populated(self):
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        assert len(dep.sources) >= 1

    def test_numpy_line_number(self):
        """numpy must have an exact line number recorded (traceability)."""
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        src = dep.sources[0]
        assert src.line is not None, "line number must be recovered for numpy"
        # pyproject.toml has numpy on line 11 of the fixture
        assert src.line == 11

    def test_numpy_raw_line(self):
        """raw_line must contain the verbatim declaration text."""
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        src = dep.sources[0]
        assert src.raw_line is not None
        assert "numpy" in src.raw_line.lower()
        assert "1.24" in src.raw_line

    def test_requests_present(self):
        deps = self._manifest.ecosystems["python"].dependencies
        assert "requests" in deps

    def test_click_present(self):
        deps = self._manifest.ecosystems["python"].dependencies
        assert "click" in deps

    def test_no_conflicts(self):
        assert self._manifest.conflicts == []

    def test_explain_numpy(self):
        """explain() must return a string mentioning file and line."""
        result = self._manifest.explain("numpy")
        assert "numpy" in result.lower()
        assert "pyproject.toml" in result
        assert "11" in result  # line number

    def test_explain_unknown_raises(self):
        with pytest.raises(KeyError):
            self._manifest.explain("nonexistent-package-xyz")


# ---------------------------------------------------------------------------
# Fixture: clean_node_repo
# ---------------------------------------------------------------------------

class TestCleanNodeRepo:
    """
    A repo with ``package.json`` + ``package-lock.json`` (lockfileVersion 3).
    Expectations:
    - 1 ecosystem: node
    - express, lodash, jest, eslint all present
    - Lock-file resolves express to 4.18.2
    - No conflicts
    """

    @pytest.fixture(autouse=True)
    def manifest(self) -> RepoManifest:
        self._manifest = scan_repo(fixture("clean_node_repo"))
        return self._manifest

    def test_node_ecosystem_detected(self):
        assert "node" in self._manifest.ecosystems

    def test_no_python_ecosystem(self):
        assert "python" not in self._manifest.ecosystems

    def test_manifest_files_include_package_json(self):
        eco = self._manifest.ecosystems["node"]
        names = [os.path.basename(f) for f in eco.manifest_files]
        assert "package.json" in names

    def test_lock_file_detected(self):
        eco = self._manifest.ecosystems["node"]
        lock_names = [os.path.basename(f) for f in eco.lock_files]
        assert "package-lock.json" in lock_names

    def test_express_present(self):
        deps = self._manifest.ecosystems["node"].dependencies
        assert "express" in deps

    def test_express_declared_constraint(self):
        dep = self._manifest.ecosystems["node"].dependencies["express"]
        assert dep.declared_constraint == "^4.18.2"

    def test_express_resolved_version(self):
        """Lock-file should pin express to an exact version."""
        dep = self._manifest.ecosystems["node"].dependencies["express"]
        assert dep.resolved_version == "4.18.2"

    def test_express_line_number_in_package_json(self):
        """express must have line number from package.json source."""
        dep = self._manifest.ecosystems["node"].dependencies["express"]
        pkg_json_src = next(
            (s for s in dep.sources if "package.json" in s.file and "lock" not in s.file),
            None,
        )
        assert pkg_json_src is not None, "Source from package.json must be present"
        assert pkg_json_src.line is not None, "Line number must be recovered"
        assert pkg_json_src.line == 9  # "express": "^4.18.2" is on line 9 of package.json

    def test_express_raw_line(self):
        dep = self._manifest.ecosystems["node"].dependencies["express"]
        pkg_json_src = next(
            (s for s in dep.sources if "package.json" in s.file and "lock" not in s.file),
            None,
        )
        assert pkg_json_src is not None
        assert pkg_json_src.raw_line is not None
        assert "express" in pkg_json_src.raw_line

    def test_lodash_present(self):
        assert "lodash" in self._manifest.ecosystems["node"].dependencies

    def test_jest_present(self):
        assert "jest" in self._manifest.ecosystems["node"].dependencies

    def test_eslint_present(self):
        assert "eslint" in self._manifest.ecosystems["node"].dependencies

    def test_no_conflicts(self):
        assert self._manifest.conflicts == []

    def test_explain_express(self):
        result = self._manifest.explain("express")
        assert "express" in result.lower()
        assert "package.json" in result


# ---------------------------------------------------------------------------
# Fixture: conflict_repo
# ---------------------------------------------------------------------------

class TestConflictRepo:
    """
    A repo with both ``requirements.txt`` and ``pyproject.toml``.

    Conflicts expected:
    1. Multiple manifest files → "multiple Python manifest files" conflict.
    2. numpy declared with different constraints in each file.
    3. requests declared with different constraints in each file.
    """

    @pytest.fixture(autouse=True)
    def manifest(self) -> RepoManifest:
        self._manifest = scan_repo(fixture("conflict_repo"))
        return self._manifest

    def test_python_ecosystem_detected(self):
        assert "python" in self._manifest.ecosystems

    def test_both_manifests_recorded(self):
        eco = self._manifest.ecosystems["python"]
        names = {os.path.basename(f) for f in eco.manifest_files}
        assert "requirements.txt" in names
        assert "pyproject.toml" in names

    def test_numpy_in_deps(self):
        deps = self._manifest.ecosystems["python"].dependencies
        assert "numpy" in deps

    def test_numpy_has_two_sources(self):
        """numpy is declared in both files — must have at least 2 SourceRefs."""
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        file_names = {os.path.basename(s.file) for s in dep.sources}
        assert "requirements.txt" in file_names
        assert "pyproject.toml" in file_names

    def test_numpy_line_number_requirements_txt(self):
        """Traceability: requirements.txt line number for numpy."""
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        req_src = next(
            (s for s in dep.sources if "requirements.txt" in s.file),
            None,
        )
        assert req_src is not None
        assert req_src.line == 2  # line 2 in requirements.txt (line 1 is comment)
        assert "numpy" in req_src.raw_line.lower()

    def test_numpy_raw_line_requirements_txt(self):
        dep = self._manifest.ecosystems["python"].dependencies["numpy"]
        req_src = next(
            (s for s in dep.sources if "requirements.txt" in s.file), None
        )
        assert req_src is not None
        assert req_src.raw_line is not None
        assert "numpy>=1.20" in req_src.raw_line

    def test_conflicts_not_empty(self):
        assert len(self._manifest.conflicts) > 0

    def test_multiple_manager_conflict_present(self):
        """Multiple manifest files conflict must be reported."""
        combined = " ".join(self._manifest.conflicts)
        assert "multiple" in combined.lower() or "manifest" in combined.lower()

    def test_numpy_constraint_conflict_present(self):
        """numpy has different constraints in two files — must be flagged."""
        combined = " ".join(self._manifest.conflicts)
        assert "numpy" in combined.lower()

    def test_requests_line_number_pyproject(self):
        """Traceability: requests in pyproject.toml line number."""
        dep = self._manifest.ecosystems["python"].dependencies["requests"]
        py_src = next(
            (s for s in dep.sources if "pyproject.toml" in s.file), None
        )
        assert py_src is not None
        assert py_src.line is not None  # line number must be recovered
        assert "requests" in py_src.raw_line.lower()


# ---------------------------------------------------------------------------
# Fixture: empty_repo
# ---------------------------------------------------------------------------

class TestEmptyRepo:
    """
    A repo with no manifest files at all.
    Expectations:
    - No ecosystems detected
    - No conflicts
    - scan_repo must not raise
    """

    @pytest.fixture(autouse=True)
    def manifest(self) -> RepoManifest:
        self._manifest = scan_repo(fixture("empty_repo"))
        return self._manifest

    def test_no_ecosystems(self):
        assert self._manifest.ecosystems == {}

    def test_no_conflicts(self):
        assert self._manifest.conflicts == []

    def test_no_env_configs(self):
        assert self._manifest.environment_configs == []

    def test_explain_raises_for_any_dep(self):
        with pytest.raises(KeyError):
            self._manifest.explain("requests")


# ---------------------------------------------------------------------------
# Fixture: version_pin_conflict_repo
# ---------------------------------------------------------------------------

class TestVersionPinConflictRepo:
    """
    A repo with ``.python-version`` pinning 3.9.18 but ``pyproject.toml``
    declaring ``requires-python = ">=3.11"``.

    Expectations:
    - python ecosystem detected
    - version-pin conflict flagged
    - ``.python-version`` recorded in environment_configs with version_pin="3.9.18"
    """

    @pytest.fixture(autouse=True)
    def manifest(self) -> RepoManifest:
        self._manifest = scan_repo(fixture("version_pin_conflict_repo"))
        return self._manifest

    def test_python_ecosystem_detected(self):
        assert "python" in self._manifest.ecosystems

    def test_python_version_env_config_detected(self):
        types = [c.config_type for c in self._manifest.environment_configs]
        assert "python-version" in types

    def test_python_version_pin_value(self):
        cfg = next(
            c for c in self._manifest.environment_configs
            if c.config_type == "python-version"
        )
        assert cfg.version_pin == "3.9.18"

    def test_version_pin_conflict_flagged(self):
        """The 3.9.18 pin must be flagged against >=3.11."""
        combined = " ".join(self._manifest.conflicts)
        assert "3.9.18" in combined or "python-version" in combined.lower()
        assert "3.11" in combined or "incompatible" in combined.lower()

    def test_conflict_mentions_incompatible(self):
        """Conflict message must use the word 'incompatible'."""
        assert any(
            "incompatible" in c.lower() for c in self._manifest.conflicts
        )

    def test_httpx_present(self):
        deps = self._manifest.ecosystems["python"].dependencies
        assert "httpx" in deps

    def test_httpx_line_number(self):
        """Traceability: httpx must have a line number from pyproject.toml."""
        dep = self._manifest.ecosystems["python"].dependencies["httpx"]
        src = dep.sources[0]
        assert src.line is not None
        assert "httpx" in src.raw_line.lower()

    def test_pydantic_present(self):
        deps = self._manifest.ecosystems["python"].dependencies
        assert "pydantic" in deps


# ---------------------------------------------------------------------------
# Edge-case / cross-cutting tests
# ---------------------------------------------------------------------------

class TestScanRepoErrors:
    """scan_repo on a non-existent path must raise NotADirectoryError."""

    def test_nonexistent_path(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(NotADirectoryError):
            scan_repo(missing)

    def test_file_as_path(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            scan_repo(f)


class TestSourceRefStructure:
    """SourceRef fields are always structured data, never raw strings."""

    def test_sourceref_file_is_str(self):
        from src.repo_scan.models import SourceRef
        ref = SourceRef(file="/some/path.toml", line=10, raw_line="numpy>=1.24")
        assert isinstance(ref.file, str)
        assert isinstance(ref.line, int)
        assert isinstance(ref.raw_line, str)

    def test_sourceref_line_optional(self):
        from src.repo_scan.models import SourceRef
        ref = SourceRef(file="/some/path.toml")
        assert ref.line is None
        assert ref.raw_line is None
