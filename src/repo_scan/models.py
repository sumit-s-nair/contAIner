"""
src/repo_scan/models.py
=======================
Pure data classes for the RepoManifest pipeline.

No I/O, no parsing logic — only structure and derived helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Traceability primitive
# ---------------------------------------------------------------------------

@dataclass
class SourceRef:
    """Points to the exact location where a dependency was declared."""

    file: str
    """Absolute or repo-relative path of the file."""

    line: Optional[int] = None
    """1-based line number within *file*, if recoverable."""

    raw_line: Optional[str] = None
    """Verbatim declaration text, e.g. ``"numpy>=1.20"``."""

    def __repr__(self) -> str:          # pragma: no cover
        loc = f":{self.line}" if self.line else ""
        return f"SourceRef({self.file!r}{loc})"


# ---------------------------------------------------------------------------
# Inferred dependency (import-scan fallback)
# ---------------------------------------------------------------------------

@dataclass
class InferredDependency:
    """
    A dependency inferred from ``import`` statements in ``.py`` files,
    used as a fallback when no manifest is present.

    *import_name*
        The top-level name as it appears in source code (e.g. ``"cv2"``).

    *guessed_package_name*
        The PyPI package name: taken from the known-mismatch table when
        available (e.g. ``"opencv-python"``), otherwise the import name
        itself.

    *confidence*
        ``"mapped"`` — import name was found in the static lookup table.
        ``"unmapped_guess"`` — import name used directly as package name.

    *sources*
        Every :class:`SourceRef` (file + line) where this import was seen.
    """

    import_name: str
    guessed_package_name: Optional[str]
    confidence: str  # "mapped" | "unmapped_guess"
    sources: List[SourceRef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

@dataclass
class Dependency:
    """
    A single dependency entry aggregated across all manifests / lock-files
    found for an ecosystem.

    *name*
        Canonical package name (lowercased for normalisation, original
        capitalisation preserved in ``sources[*].raw_line``).

    *declared_constraint*
        The version specifier as written in the manifest (e.g. ``>=1.20``,
        ``^18.0.0``, ``*``).  For lock-files the exact pinned version is
        stored in *resolved_version*.

    *resolved_version*
        Exact pinned version from a lock-file, or ``None`` if no lock-file
        was found.

    *sources*
        Every location (file + line) where this dependency was observed.
        Populated by every parser that touches the package.
    """

    name: str
    declared_constraint: str
    resolved_version: Optional[str] = None
    sources: List[SourceRef] = field(default_factory=list)

    # Keep name normalisation in one place so callers don't have to guess.
    def __post_init__(self) -> None:
        self.name = self.name.lower()


# ---------------------------------------------------------------------------
# Ecosystem manifest
# ---------------------------------------------------------------------------

@dataclass
class EcosystemManifest:
    """
    Aggregated manifest data for a single package-manager ecosystem found
    inside a repository.

    *ecosystem*
        ``"python"`` or ``"node"`` — the identifier used throughout the
        pipeline.  New ecosystems add new members here without changing the
        interface.

    *manifest_files*
        Paths to non-lock manifests detected (e.g. ``pyproject.toml``).

    *lock_files*
        Paths to lock-files detected (e.g. ``package-lock.json``).

    *dependencies*
        ``name -> Dependency`` map.  Keys are normalised (lowercase).
    """

    ecosystem: str
    manifest_files: List[str] = field(default_factory=list)
    lock_files: List[str] = field(default_factory=list)
    dependencies: Dict[str, Dependency] = field(default_factory=dict)
    inferred_dependencies: List["InferredDependency"] = field(default_factory=list)

    # --- convenience -------------------------------------------------------

    def add_dependency(self, dep: Dependency) -> None:
        """Merge *dep* into ``self.dependencies``, accumulating sources."""
        key = dep.name
        if key not in self.dependencies:
            self.dependencies[key] = dep
        else:
            existing = self.dependencies[key]
            existing.sources.extend(dep.sources)
            # Prefer tighter constraint from lock-file if not already set.
            if dep.resolved_version and not existing.resolved_version:
                existing.resolved_version = dep.resolved_version


# ---------------------------------------------------------------------------
# Environment configuration shim
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """
    Records presence + path of a recognised environment-configuration file.

    *config_type*
        One of: ``"docker"``, ``"devcontainer"``, ``"python-version"``,
        ``"nvmrc"``.

    *path*
        Path (repo-relative or absolute) to the config file.

    *version_pin*
        For ``.python-version`` / ``.nvmrc``: the bare version string read
        from the file.  ``None`` for files we don't minimally parse.
    """

    config_type: str
    path: str
    version_pin: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------

@dataclass
class RepoManifest:
    """
    Single structured result returned by ``scan_repo()``.

    *ecosystems*
        ``ecosystem_name -> EcosystemManifest``.

    *environment_configs*
        All recognised environment-config files detected.

    *conflicts*
        Human-readable conflict descriptions generated by
        ``src.repo_scan.conflicts``.
    """

    ecosystems: Dict[str, EcosystemManifest] = field(default_factory=dict)
    environment_configs: List[EnvironmentConfig] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    # --- explainability ----------------------------------------------------

    def explain(self, dependency_name: str) -> str:
        """
        Return a human-readable trace for *dependency_name* across all
        ecosystems.

        Handles both *declared* dependencies (from manifests) and *inferred*
        dependencies (from import scanning when no manifest is found).  The
        distinction is always visible in the returned text.

        Example outputs::

            # declared
            numpy: declared as 'numpy>=1.20' in pyproject.toml:14

            # inferred — known mapping
            cv2: inferred from 'import cv2' in src/main.py:5 \
(no manifest found — mapped to package 'opencv-python')

            # inferred — unmapped guess
            mylib: inferred from 'import mylib' in src/app.py:7 \
(no manifest found — package name guessed from import name)

        If the dependency appears in multiple places, each location is
        listed on its own line.

        Raises ``KeyError`` if the dependency is not found in any ecosystem.
        """
        normalised = dependency_name.lower()
        lines: List[str] = []

        for eco_name, eco in self.ecosystems.items():
            # --- declared dependencies ---------------------------------------
            dep = eco.dependencies.get(normalised)
            if dep is not None:
                for ref in dep.sources:
                    loc = ref.file
                    if ref.line:
                        loc = f"{loc}:{ref.line}"
                    text = ref.raw_line or dep.declared_constraint
                    lines.append(
                        f"{dep.name}: declared as {text!r} in {loc}"
                    )
                if dep.resolved_version:
                    lines.append(
                        f"  └─ resolved/pinned to {dep.resolved_version!r} "
                        f"by lock-file"
                    )

            # --- inferred dependencies (import-scan fallback) ---------------
            for inferred in eco.inferred_dependencies:
                if inferred.import_name.lower() != normalised and (
                    inferred.guessed_package_name is None
                    or inferred.guessed_package_name.lower() != normalised
                ):
                    continue
                for ref in inferred.sources:
                    loc = ref.file
                    if ref.line:
                        loc = f"{loc}:{ref.line}"
                    if inferred.confidence == "mapped":
                        suffix = (
                            f"no manifest found — mapped to package "
                            f"{inferred.guessed_package_name!r}"
                        )
                    else:
                        suffix = (
                            "no manifest found — package name guessed "
                            "from import name"
                        )
                    lines.append(
                        f"{inferred.import_name}: inferred from "
                        f"'import {inferred.import_name}' in {loc} "
                        f"({suffix})"
                    )

        if not lines:
            raise KeyError(
                f"Dependency {dependency_name!r} not found in any "
                f"ecosystem manifest."
            )
        return "\n".join(lines)
