"""
src/repo_scan/conflicts.py
==========================
Pure conflict-detection functions.

All functions take a fully populated :class:`RepoManifest` plus optional
supplementary data, and *append* human-readable conflict descriptions to
``manifest.conflicts``.  Nothing is resolved — just flagged.

Conflict categories detected
-----------------------------
1. **Intra-ecosystem constraint mismatch** — same dependency declared in
   multiple manifest files with different constraints (e.g. ``requirements.txt``
   and ``pyproject.toml`` both declare ``numpy`` but with conflicting specs).

2. **Multi-manager detection** — multiple package-manager manifest types for
   the same ecosystem found simultaneously (e.g. both ``requirements.txt`` and
   ``pyproject.toml``).

3. **Language-version pin vs. manifest constraint** — ``.python-version`` pin
   doesn't satisfy ``requires-python`` in ``pyproject.toml``; ``.nvmrc`` pin
   doesn't satisfy ``engines.node`` in ``package.json``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .models import EcosystemManifest, EnvironmentConfig, RepoManifest

# Lock-file basenames — sources from these files are resolved pins, not
# declared constraints, so they are excluded from cross-file conflict checks.
_LOCK_FILE_NAMES = frozenset({
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "requirements.lock",
    "uv.lock",
    "Pipfile.lock",
})


# ---------------------------------------------------------------------------
# Version compatibility helpers
# ---------------------------------------------------------------------------

def _parse_simple_version(v: str) -> Optional[tuple]:
    """
    Parse a simple ``MAJOR.MINOR.PATCH`` or ``MAJOR.MINOR`` string into a
    comparable tuple of ints.  Returns ``None`` for LTS aliases or complex
    ranges.
    """
    v = v.strip().lstrip("v")
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?$", v)
    if not m:
        return None
    return tuple(int(x) for x in m.groups(default="0"))


def _version_satisfies_constraint(version_str: str, constraint: str) -> Optional[bool]:
    """
    Minimal constraint satisfaction check for simple specifiers.

    Handles: ``>=X.Y``, ``>X.Y``, ``<=X.Y``, ``<X.Y``, ``==X.Y``,
    ``!=X.Y``, and bare ``X.Y`` (exact).

    Returns
    -------
    True  — version satisfies constraint
    False — version violates constraint
    None  — cannot determine (complex constraint, LTS alias, etc.)
    """
    version = _parse_simple_version(version_str)
    if version is None:
        return None

    # A constraint string may have multiple comma-separated specifiers
    specifiers = [s.strip() for s in constraint.split(",") if s.strip()]
    for spec in specifiers:
        m = re.match(r"^([><=!^~]{1,2})\s*(.+)$", spec)
        if not m:
            # Bare version — treat as == for minimum
            continue
        op, val_str = m.group(1), m.group(2).strip()
        val = _parse_simple_version(val_str)
        if val is None:
            return None  # can't compare

        # Pad tuples to equal length
        length = max(len(version), len(val))
        v = version + (0,) * (length - len(version))
        r = val + (0,) * (length - len(val))

        ok = {
            ">=": v >= r,
            ">": v > r,
            "<=": v <= r,
            "<": v < r,
            "==": v == r,
            "!=": v != r,
            "^": v[0] == r[0] and v >= r,   # semver caret (Node)
            "~": v[:2] == r[:2] and v >= r,  # semver tilde
            "~=": v[:-1] == r[:-1] and v >= r,  # Python compatible release
        }.get(op)

        if ok is None:
            return None
        if not ok:
            return False

    return True


# ---------------------------------------------------------------------------
# Conflict detectors
# ---------------------------------------------------------------------------

def detect_dependency_constraint_mismatches(
    eco: EcosystemManifest,
    manifest: RepoManifest,
) -> None:
    """
    Flag cases where the same dependency is declared in multiple *manifest*
    files with *different* version constraints.

    Lock-file sources are intentionally excluded — they contain resolved
    (pinned) versions, not declared constraints, so comparing them against
    manifest entries would produce false positives.
    """
    import os

    for name, dep in eco.dependencies.items():
        # Group sources by file, excluding lock-files
        by_file: Dict[str, List[str]] = {}
        for ref in dep.sources:
            basename = os.path.basename(ref.file)
            if basename in _LOCK_FILE_NAMES:
                continue  # skip resolved-pin sources
            key = ref.file
            if key not in by_file:
                by_file[key] = []
            raw = ref.raw_line or dep.declared_constraint
            by_file[key].append(raw)

        if len(by_file) < 2:
            continue  # only one (or zero) manifest files — no cross-file conflict

        # Collect unique constraints per file
        constraints = {
            f: " ".join(sorted(set(raws))) for f, raws in by_file.items()
        }
        unique_constraints = set(constraints.values())
        if len(unique_constraints) > 1:
            locations = "; ".join(
                f"{f!r}: {c}" for f, c in constraints.items()
            )
            manifest.conflicts.append(
                f"[{eco.ecosystem}] Dependency {name!r} declared with "
                f"conflicting constraints — {locations}"
            )


def detect_multiple_managers(
    eco: EcosystemManifest,
    manifest: RepoManifest,
) -> None:
    """
    Flag when multiple manifest-file types for the same ecosystem co-exist
    (e.g. ``requirements.txt`` AND ``pyproject.toml`` for Python;
    ``package.json`` only has one format so this fires for unusual combos).
    """
    if eco.ecosystem == "python" and len(eco.manifest_files) > 1:
        files = ", ".join(eco.manifest_files)
        manifest.conflicts.append(
            f"[python] Multiple Python manifest files detected — consider "
            f"consolidating: {files}"
        )


def detect_version_pin_conflicts(
    manifest: RepoManifest,
    *,
    python_requires: Optional[str] = None,
    node_engines: Optional[str] = None,
) -> None:
    """
    Compare environment-config version pins against manifest language
    constraints and flag mismatches.

    Parameters
    ----------
    manifest:
        The manifest being built (conflicts are appended to it).
    python_requires:
        The ``requires-python`` string from ``pyproject.toml``, if any.
    node_engines:
        The ``engines.node`` string from ``package.json``, if any.
    """
    for env_cfg in manifest.environment_configs:
        pin = env_cfg.version_pin
        if not pin:
            continue

        if env_cfg.config_type == "python-version" and python_requires:
            result = _version_satisfies_constraint(pin, python_requires)
            if result is False:
                manifest.conflicts.append(
                    f"[python] Version pin {pin!r} in {env_cfg.path!r} is "
                    f"incompatible with requires-python {python_requires!r} "
                    f"from pyproject.toml"
                )
            elif result is None:
                manifest.conflicts.append(
                    f"[python] Cannot determine compatibility between "
                    f".python-version pin {pin!r} and requires-python "
                    f"{python_requires!r} — manual review recommended"
                )

        if env_cfg.config_type == "nvmrc" and node_engines:
            result = _version_satisfies_constraint(pin, node_engines)
            if result is False:
                manifest.conflicts.append(
                    f"[node] Version pin {pin!r} in {env_cfg.path!r} is "
                    f"incompatible with engines.node {node_engines!r} "
                    f"from package.json"
                )
            elif result is None and not pin.startswith("lts"):
                manifest.conflicts.append(
                    f"[node] Cannot determine compatibility between "
                    f".nvmrc pin {pin!r} and engines.node {node_engines!r} "
                    f"— manual review recommended"
                )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_conflict_checks(
    manifest: RepoManifest,
    *,
    python_requires: Optional[str] = None,
    node_engines: Optional[str] = None,
) -> None:
    """
    Run every conflict-detection pass on a fully-populated *manifest*.

    This is the single entry-point called by the scanner after all parsers
    have finished.
    """
    for eco in manifest.ecosystems.values():
        detect_multiple_managers(eco, manifest)
        detect_dependency_constraint_mismatches(eco, manifest)

    detect_version_pin_conflicts(
        manifest,
        python_requires=python_requires,
        node_engines=node_engines,
    )
