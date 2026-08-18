"""
src/repo_scan/scanner.py
========================
Orchestrator — the only module that performs file-system I/O.

Public API
----------
    scan_repo(repo_path: str | Path) -> RepoManifest

Walk the repository root, detect ecosystems, delegate to parsers, run
conflict checks, and return a single :class:`RepoManifest`.

Design rules
------------
- Exactly one public function: ``scan_repo``.
- All parsing is delegated to ``parsers.py`` (pure functions).
- All conflict detection is delegated to ``conflicts.py`` (pure functions).
- No LLM calls, no network I/O.
- Deterministic: given the same file tree, always returns the same result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .conflicts import run_all_conflict_checks
from .import_scan import scan_imports
from .models import (
    Dependency,
    EcosystemManifest,
    EnvironmentConfig,
    RepoManifest,
)
from .parsers import (
    parse_package_json,
    parse_package_lock_json,
    parse_pyproject_toml,
    parse_requirements_txt,
    read_version_pin_file,
)


# ---------------------------------------------------------------------------
# Manifest / lock-file filename constants
# ---------------------------------------------------------------------------

_PYTHON_MANIFEST_FILES = {"requirements.txt", "pyproject.toml"}
_PYTHON_LOCK_FILES = {
    "poetry.lock",
    "requirements.lock",
    "uv.lock",
    "Pipfile.lock",
}
_NODE_MANIFEST_FILES = {"package.json"}
_NODE_LOCK_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

_ENV_CONFIG_MAP = {
    "Dockerfile": "docker",
    ".python-version": "python-version",
    ".nvmrc": "nvmrc",
}
_DEVCONTAINER_PATHS = [
    ".devcontainer/devcontainer.json",
    ".devcontainer.json",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_env_configs(root: Path) -> list[EnvironmentConfig]:
    """Detect environment-config files and record presence + version pins."""
    configs: list[EnvironmentConfig] = []

    # Single-file entries
    for filename, config_type in _ENV_CONFIG_MAP.items():
        candidate = root / filename
        if candidate.is_file():
            pin: Optional[str] = None
            if config_type in ("python-version", "nvmrc"):
                pin = read_version_pin_file(str(candidate))
            configs.append(
                EnvironmentConfig(
                    config_type=config_type,
                    path=str(candidate),
                    version_pin=pin,
                )
            )

    # devcontainer (two possible paths)
    for rel in _DEVCONTAINER_PATHS:
        candidate = root / rel
        if candidate.is_file():
            configs.append(
                EnvironmentConfig(
                    config_type="devcontainer",
                    path=str(candidate),
                )
            )
            break  # only record once

    return configs


def _build_python_ecosystem(root: Path) -> tuple[EcosystemManifest | None, Optional[str]]:
    """
    Detect and parse Python manifests under *root*.

    Returns
    -------
    eco:
        Populated :class:`EcosystemManifest` or ``None`` if no Python
        manifests were found.
    requires_python:
        The ``requires-python`` constraint string (from pyproject.toml) if
        found, used by the conflict checker.
    """
    manifest_files = [
        str(root / f)
        for f in _PYTHON_MANIFEST_FILES
        if (root / f).is_file()
    ]
    lock_files = [
        str(root / f)
        for f in _PYTHON_LOCK_FILES
        if (root / f).is_file()
    ]

    if not manifest_files:
        # Fallback: if the repo contains .py files, infer dependencies from
        # import statements (AST-based).  Only triggered when there are truly
        # no manifests; the eco will have empty manifest_files and
        # dependencies so downstream can distinguish it from a declared scan.
        has_py_files = any(
            f.suffix == ".py"
            for f in root.rglob("*.py")
            # Quick existence check — we don't need the full exclude walk here
            if not any(
                part in {".venv", "venv", "__pycache__", ".git",
                         "site-packages", "node_modules"}
                for part in f.relative_to(root).parts
            )
        )
        if not has_py_files:
            return None, None

        eco = EcosystemManifest(
            ecosystem="python",
            manifest_files=[],
            lock_files=[],
        )
        eco.inferred_dependencies = scan_imports(root)
        return eco, None

    eco = EcosystemManifest(
        ecosystem="python",
        manifest_files=manifest_files,
        lock_files=lock_files,
    )
    requires_python: Optional[str] = None

    # --- requirements.txt --------------------------------------------------
    req_path = root / "requirements.txt"
    if req_path.is_file():
        for dep in parse_requirements_txt(str(req_path)):
            eco.add_dependency(dep)

    # --- pyproject.toml ----------------------------------------------------
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        deps, rp = parse_pyproject_toml(str(pyproject_path))
        if rp:
            requires_python = rp
        for dep in deps:
            eco.add_dependency(dep)

    return eco, requires_python


def _build_node_ecosystem(root: Path) -> tuple[EcosystemManifest | None, Optional[str]]:
    """
    Detect and parse Node.js manifests under *root*.

    Returns
    -------
    eco:
        Populated :class:`EcosystemManifest` or ``None``.
    node_engines:
        The ``engines.node`` specifier from ``package.json`` if present.
    """
    manifest_files = [
        str(root / f)
        for f in _NODE_MANIFEST_FILES
        if (root / f).is_file()
    ]
    lock_files = [
        str(root / f)
        for f in _NODE_LOCK_FILES
        if (root / f).is_file()
    ]

    if not manifest_files:
        return None, None

    eco = EcosystemManifest(
        ecosystem="node",
        manifest_files=manifest_files,
        lock_files=lock_files,
    )
    node_engines: Optional[str] = None

    # --- package.json ------------------------------------------------------
    pkg_json_path = root / "package.json"
    if pkg_json_path.is_file():
        deps, engines = parse_package_json(str(pkg_json_path))
        if engines:
            node_engines = engines
        for dep in deps:
            eco.add_dependency(dep)

    # --- package-lock.json (npm v2/v3) ------------------------------------
    lock_path = root / "package-lock.json"
    if lock_path.is_file():
        for dep in parse_package_lock_json(str(lock_path)):
            # Merge: set resolved_version on existing entry if present
            key = dep.name
            if key in eco.dependencies:
                existing = eco.dependencies[key]
                if not existing.resolved_version and dep.resolved_version:
                    existing.resolved_version = dep.resolved_version
                existing.sources.extend(dep.sources)
            else:
                eco.add_dependency(dep)

    return eco, node_engines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_repo(repo_path: str | Path) -> RepoManifest:
    """
    Scan the repository rooted at *repo_path* and return a
    :class:`RepoManifest`.

    Parameters
    ----------
    repo_path:
        Path to the repository root directory.  Must exist.

    Returns
    -------
    RepoManifest:
        Fully populated manifest.  If no manifests are found, an
        empty-but-valid :class:`RepoManifest` is returned (``ecosystems``
        will be empty, ``conflicts`` will be empty).
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"repo_path is not a directory: {repo_path!r}")

    manifest = RepoManifest()

    # --- Environment configs (no parsing, just presence + pin) ------------
    manifest.environment_configs = _scan_env_configs(root)

    # --- Ecosystem parsers -------------------------------------------------
    python_requires: Optional[str] = None
    node_engines: Optional[str] = None

    python_eco, python_requires = _build_python_ecosystem(root)
    if python_eco is not None:
        manifest.ecosystems["python"] = python_eco

    node_eco, node_engines = _build_node_ecosystem(root)
    if node_eco is not None:
        manifest.ecosystems["node"] = node_eco

    # --- Conflict detection ------------------------------------------------
    run_all_conflict_checks(
        manifest,
        python_requires=python_requires,
        node_engines=node_engines,
    )

    return manifest
