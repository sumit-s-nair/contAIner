"""
tests/test_import_scan.py
=========================
Tests for the AST-based import-inference fallback in
``src.repo_scan.import_scan``.

Covers:
- No-manifest Python repo: stdlib excluded, local modules excluded,
  mapped and unmapped imports, correct line numbers, explain() phrasing.
- Local-module detection: single-file module, local package (dir +
  __init__.py), bare directory without __init__.py (must NOT be excluded).
- Import aggregation: same name in multiple files → one InferredDependency
  with multiple SourceRefs.
- Syntax-error graceful handling: bad.py skipped, good.py still scanned.
- Fallback not triggered when manifests are present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.repo_scan import InferredDependency, RepoManifest, scan_repo
from src.repo_scan.import_scan import is_local_module, scan_imports

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Path:
    return (FIXTURES_DIR / name).resolve()


def _find_inferred(eco_inferred, import_name: str) -> InferredDependency | None:
    """Return the InferredDependency with the given import_name, or None."""
    for dep in eco_inferred:
        if dep.import_name == import_name:
            return dep
    return None


# ---------------------------------------------------------------------------
# Fixture: no_manifest_python_repo
# ---------------------------------------------------------------------------

class TestNoManifestPythonRepo:
    """
    Repo has .py files but no manifest.
    Scanner must run the import-scan fallback.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self._manifest = scan_repo(fixture("no_manifest_python_repo"))
        self._eco = self._manifest.ecosystems.get("python")

    # --- ecosystem detection ------------------------------------------------

    def test_python_ecosystem_detected(self):
        assert "python" in self._manifest.ecosystems

    def test_manifest_files_empty(self):
        """No manifests exist → manifest_files must be empty list."""
        assert self._eco.manifest_files == []

    def test_declared_dependencies_empty(self):
        """Inferred eco must have no declared dependencies."""
        assert self._eco.dependencies == {}

    def test_inferred_dependencies_non_empty(self):
        assert len(self._eco.inferred_dependencies) > 0

    # --- stdlib filtering ---------------------------------------------------

    def test_os_not_inferred(self):
        assert _find_inferred(self._eco.inferred_dependencies, "os") is None

    def test_sys_not_inferred(self):
        assert _find_inferred(self._eco.inferred_dependencies, "sys") is None

    # --- local-module filtering ---------------------------------------------

    def test_utils_not_inferred(self):
        """utils.py is a local single-file module — must be excluded."""
        assert _find_inferred(self._eco.inferred_dependencies, "utils") is None

    def test_myapp_not_inferred(self):
        """myapp/ has __init__.py — local package must be excluded."""
        assert _find_inferred(self._eco.inferred_dependencies, "myapp") is None

    # --- mapped imports -----------------------------------------------------

    def test_cv2_present(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "cv2")
        assert dep is not None

    def test_cv2_mapped_to_opencv(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "cv2")
        assert dep.guessed_package_name == "opencv-python"

    def test_cv2_confidence_mapped(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "cv2")
        assert dep.confidence == "mapped"

    def test_cv2_line_number(self):
        """cv2 is imported on line 16 of app.py."""
        dep = _find_inferred(self._eco.inferred_dependencies, "cv2")
        assert dep is not None
        assert len(dep.sources) >= 1
        src = dep.sources[0]
        assert src.line == 16

    def test_cv2_source_file_is_app_py(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "cv2")
        assert dep is not None
        src_files = [os.path.basename(s.file) for s in dep.sources]
        assert "app.py" in src_files

    def test_yaml_mapped_to_pyyaml(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "yaml")
        assert dep is not None
        assert dep.guessed_package_name == "PyYAML"
        assert dep.confidence == "mapped"

    def test_numpy_mapped(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "numpy")
        assert dep is not None
        assert dep.guessed_package_name == "numpy"
        assert dep.confidence == "mapped"

    # --- unmapped import ----------------------------------------------------

    def test_unusuallib_present(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "unusuallib")
        assert dep is not None

    def test_unusuallib_unmapped_guess(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "unusuallib")
        assert dep.confidence == "unmapped_guess"

    def test_unusuallib_guessed_package_is_import_name(self):
        dep = _find_inferred(self._eco.inferred_dependencies, "unusuallib")
        assert dep.guessed_package_name == "unusuallib"

    def test_unusuallib_line_number(self):
        """unusuallib is imported on line 19 of app.py."""
        dep = _find_inferred(self._eco.inferred_dependencies, "unusuallib")
        assert dep is not None
        assert dep.sources[0].line == 19

    # --- explain() ----------------------------------------------------------

    def test_explain_cv2_contains_inferred(self):
        result = self._manifest.explain("cv2")
        assert "inferred" in result

    def test_explain_cv2_contains_mapped(self):
        result = self._manifest.explain("cv2")
        assert "mapped" in result

    def test_explain_cv2_contains_package_name(self):
        result = self._manifest.explain("cv2")
        assert "opencv-python" in result

    def test_explain_unusuallib_contains_guessed(self):
        result = self._manifest.explain("unusuallib")
        assert "guessed" in result

    def test_explain_unusuallib_contains_inferred(self):
        result = self._manifest.explain("unusuallib")
        assert "inferred" in result

    def test_explain_unknown_raises(self):
        with pytest.raises(KeyError):
            self._manifest.explain("nonexistent-xyz")

    # --- import name deduplication (list uniqueness) -----------------------

    def test_no_duplicate_import_names(self):
        """Every import_name must appear at most once in inferred_dependencies."""
        names = [d.import_name for d in self._eco.inferred_dependencies]
        assert len(names) == len(set(names)), f"Duplicate entries: {names}"


# ---------------------------------------------------------------------------
# Local-module detection unit tests
# ---------------------------------------------------------------------------

class TestLocalPackageDetection:
    """
    Unit tests for ``is_local_module`` using tmp_path — no scanner needed.
    Covers all three cases explicitly.
    """

    def test_single_file_module_detected(self, tmp_path):
        """<root>/utils.py → is_local_module('utils') must be True."""
        (tmp_path / "utils.py").write_text("")
        assert is_local_module("utils", tmp_path) is True

    def test_local_package_detected(self, tmp_path):
        """<root>/myapp/__init__.py → is_local_module('myapp') must be True."""
        pkg = tmp_path / "myapp"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        assert is_local_module("myapp", tmp_path) is True

    def test_bare_directory_not_local(self, tmp_path):
        """
        <root>/emptydir/ (no __init__.py) → is_local_module('emptydir') must
        be False (bare directories without __init__.py are NOT local packages).
        """
        (tmp_path / "emptydir").mkdir()
        assert is_local_module("emptydir", tmp_path) is False

    def test_external_import_not_local(self, tmp_path):
        """'requests' has no file/package in tmp_path → must be False."""
        assert is_local_module("requests", tmp_path) is False


# ---------------------------------------------------------------------------
# Import aggregation correctness
# ---------------------------------------------------------------------------

class TestImportAggregation:
    """
    'requests' is imported in both app.py (line 17) and main.py (line 13).
    The result must be a single InferredDependency with two SourceRefs.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self._deps = scan_imports(fixture("no_manifest_python_repo"))

    def test_requests_appears_exactly_once(self):
        requests_deps = [d for d in self._deps if d.import_name == "requests"]
        assert len(requests_deps) == 1, (
            f"Expected 1 InferredDependency for 'requests', "
            f"got {len(requests_deps)}: {requests_deps}"
        )

    def test_requests_has_two_sources(self):
        dep = next(d for d in self._deps if d.import_name == "requests")
        assert len(dep.sources) == 2, (
            f"Expected 2 SourceRefs for 'requests', got {len(dep.sources)}: "
            f"{dep.sources}"
        )

    def test_requests_sources_cover_both_files(self):
        dep = next(d for d in self._deps if d.import_name == "requests")
        source_basenames = {os.path.basename(s.file) for s in dep.sources}
        assert "app.py" in source_basenames
        assert "main.py" in source_basenames

    def test_requests_app_py_line(self):
        """app.py imports requests on line 17."""
        dep = next(d for d in self._deps if d.import_name == "requests")
        app_src = next(
            (s for s in dep.sources if os.path.basename(s.file) == "app.py"),
            None,
        )
        assert app_src is not None
        assert app_src.line == 17

    def test_requests_main_py_line(self):
        """main.py imports requests on line 13."""
        dep = next(d for d in self._deps if d.import_name == "requests")
        main_src = next(
            (s for s in dep.sources if os.path.basename(s.file) == "main.py"),
            None,
        )
        assert main_src is not None
        assert main_src.line == 13


# ---------------------------------------------------------------------------
# Graceful SyntaxError handling
# ---------------------------------------------------------------------------

class TestSyntaxErrorRepo:
    """
    Repo has bad.py (SyntaxError) alongside good.py.
    scan_repo must not raise; good.py's imports must still be collected.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self._manifest = scan_repo(fixture("syntax_error_python_repo"))
        self._eco = self._manifest.ecosystems.get("python")

    def test_scan_does_not_crash(self):
        """scan_repo on a repo containing bad.py must not raise."""
        assert self._manifest is not None

    def test_python_ecosystem_detected(self):
        assert "python" in self._manifest.ecosystems

    def test_requests_from_good_py_collected(self):
        """good.py's 'import requests' must be in inferred_dependencies."""
        dep = _find_inferred(self._eco.inferred_dependencies, "requests")
        assert dep is not None

    def test_inferred_list_does_not_include_bad_py_content(self):
        """
        bad.py has no parseable imports — inferred_dependencies must only
        reflect what good.py contributed (no phantom entries from bad.py).
        """
        names = {d.import_name for d in self._eco.inferred_dependencies}
        # Only 'requests' is in good.py
        assert names == {"requests"}


# ---------------------------------------------------------------------------
# Fallback NOT triggered when manifests are present
# ---------------------------------------------------------------------------

class TestManifestPresentNoFallback:
    """
    When a manifest exists, the import-scan fallback must not run.
    inferred_dependencies must be an empty list.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self._manifest = scan_repo(fixture("clean_python_repo"))
        self._eco = self._manifest.ecosystems.get("python")

    def test_inferred_dependencies_empty(self):
        """clean_python_repo has pyproject.toml → no fallback scan."""
        assert self._eco.inferred_dependencies == []

    def test_declared_dependencies_still_populated(self):
        """Declared deps must still be present and unaffected."""
        assert "numpy" in self._eco.dependencies
