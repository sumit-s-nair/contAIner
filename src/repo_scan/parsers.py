"""
src/repo_scan/parsers.py
========================
Pure parsing functions — no I/O beyond reading the file passed as a path.

Each ``parse_*`` function accepts a file path, reads and parses it, and
returns a list of :class:`Dependency` objects with fully populated
``sources`` (including line numbers and raw declaration text).

Design rules
------------
- No side-effects: no logging, no network, no mutation of shared state.
- Every function is independently testable.
- Line numbers are 1-based throughout (matching editor conventions).
- For JSON formats (package.json, package-lock.json) we recover line
  numbers via a secondary text scan after structural parsing.
- For TOML (pyproject.toml) we use ``tomllib`` for structure, then a
  regex text-scan for position recovery — same dual-pass approach.
- For requirements.txt we parse line-by-line so positions are free.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import Dependency, SourceRef


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Normalise package name: lowercase, replace [-_.] with a single dash."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_line(text: str, key: str, start_after: int = 0) -> Optional[int]:
    """
    Return the 1-based line number of the first occurrence of *key* in
    *text* at or after line *start_after* (0-based index).

    Returns ``None`` if not found.
    """
    lines = text.splitlines()
    for idx, line in enumerate(lines[start_after:], start=start_after + 1):
        if key in line:
            return idx
    return None


def _find_json_key_line(
    text: str,
    key: str,
    start_line: int = 1,
) -> Optional[int]:
    """
    Locate ``"<key>"`` inside JSON *text*, returning a 1-based line number.
    *start_line* narrows the search to lines >= start_line.
    """
    pattern = re.compile(r'\"' + re.escape(key) + r'\"')
    lines = text.splitlines()
    for idx, line in enumerate(lines[start_line - 1:], start=start_line):
        if pattern.search(line):
            return idx
    return None


def _find_json_value_line(
    text: str,
    parent_key: str,
    child_key: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Inside JSON *text*, find the line that contains ``"<child_key>":`` inside
    the object for *parent_key*.

    Returns ``(line_number, raw_line_stripped)`` or ``(None, None)``.
    """
    parent_line = _find_json_key_line(text, parent_key)
    if parent_line is None:
        return None, None
    lines = text.splitlines()
    # Scan forward from parent_key for child_key (within ~200 lines)
    for idx, line in enumerate(
        lines[parent_line:parent_line + 200], start=parent_line + 1
    ):
        if f'"{child_key}"' in line:
            return idx, line.strip()
    return None, None


# ---------------------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------------------

# PEP 508 / pip constraint pattern — intentionally permissive
_REQ_LINE = re.compile(
    r"""
    ^
    (?P<name>[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)  # package name
    (?P<extras>\[.*?\])?                                   # optional extras
    \s*
    (?P<constraint>[^;#\n]*)                               # version constraint
    """,
    re.VERBOSE,
)


def parse_requirements_txt(path: str) -> List[Dependency]:
    """
    Parse a ``requirements.txt``-style file.

    Each non-blank, non-comment line that looks like a package spec becomes
    one :class:`Dependency`.  Line numbers are natively available.
    """
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    deps: List[Dependency] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip pip options (e.g. -r other-reqs.txt, --index-url …)
        if stripped.startswith("-"):
            continue
        # Strip inline comment
        spec = stripped.split("#", 1)[0].strip()
        m = _REQ_LINE.match(spec)
        if not m:
            continue
        name = m.group("name")
        constraint = (m.group("constraint") or "").strip()
        deps.append(
            Dependency(
                name=name,
                declared_constraint=constraint or "*",
                sources=[
                    SourceRef(
                        file=str(file_path),
                        line=lineno,
                        raw_line=stripped,
                    )
                ],
            )
        )
    return deps


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

def parse_pyproject_toml(path: str) -> Tuple[List[Dependency], Optional[str]]:
    """
    Parse a ``pyproject.toml`` file.

    Returns
    -------
    deps:
        List of :class:`Dependency` objects found in
        ``[project].dependencies`` and
        ``[tool.poetry.dependencies]``.
    requires_python:
        The ``requires-python`` specifier if present, else ``None``.

    Line-number recovery
    --------------------
    ``tomllib`` parses structure only — no positions.  We therefore do a
    secondary regex scan of the raw text to locate each declared specifier.
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")
    data: Dict[str, Any] = tomllib.loads(raw_text)
    lines = raw_text.splitlines()

    requires_python: Optional[str] = None
    deps: List[Dependency] = []

    def _line_for(spec_text: str, hint: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Search for *hint* (a package name) in raw_text lines.
        Returns (1-based line number, stripped raw line).
        """
        name_pat = re.compile(
            r"""
            ["']?                       # optional quote (Poetry)
            """ + re.escape(hint) + r"""
            ["']?                       # optional closing quote
            \s*                         # optional whitespace
            [=<>!^~\[{"]               # starts constraint, extras, or string
            """,
            re.VERBOSE | re.IGNORECASE,
        )
        for idx, line in enumerate(lines, start=1):
            if name_pat.search(line):
                return idx, line.strip()
        # Fallback: plain substring
        for idx, line in enumerate(lines, start=1):
            if hint.lower() in line.lower():
                return idx, line.strip()
        return None, None

    # --- PEP 621 [project] section -----------------------------------------
    project = data.get("project", {})

    # requires-python
    rp = project.get("requires-python")
    if rp:
        requires_python = rp
        # find in text
        for idx, line in enumerate(lines, start=1):
            if "requires-python" in line:
                break  # line already in scope for callers

    # PEP 621 dependencies list: ["numpy>=1.20", "requests"]
    for spec in project.get("dependencies", []):
        m = _REQ_LINE.match(spec.strip())
        if not m:
            continue
        name = m.group("name")
        constraint = (m.group("constraint") or "").strip() or "*"
        lineno, raw = _line_for(spec, name)
        deps.append(
            Dependency(
                name=name,
                declared_constraint=constraint,
                sources=[
                    SourceRef(
                        file=str(file_path),
                        line=lineno,
                        raw_line=raw or spec.strip(),
                    )
                ],
            )
        )

    # --- Poetry [tool.poetry.dependencies] section -------------------------
    poetry_deps = (
        data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    )
    for pkg_name, constraint_val in poetry_deps.items():
        if pkg_name.lower() == "python":
            # Treat as requires-python equivalent if not already set
            if not requires_python:
                requires_python = (
                    str(constraint_val)
                    if not isinstance(constraint_val, dict)
                    else constraint_val.get("version", "")
                )
            continue
        if isinstance(constraint_val, dict):
            constraint_str = constraint_val.get("version", "*")
        else:
            constraint_str = str(constraint_val)
        lineno, raw = _line_for(constraint_str, pkg_name)
        deps.append(
            Dependency(
                name=pkg_name,
                declared_constraint=constraint_str,
                sources=[
                    SourceRef(
                        file=str(file_path),
                        line=lineno,
                        raw_line=raw or f"{pkg_name} = {constraint_str!r}",
                    )
                ],
            )
        )

    return deps, requires_python


# ---------------------------------------------------------------------------
# package.json
# ---------------------------------------------------------------------------

def parse_package_json(path: str) -> Tuple[List[Dependency], Optional[str]]:
    """
    Parse a ``package.json`` file.

    Returns
    -------
    deps:
        Combined ``dependencies`` + ``devDependencies``.
    engines_node:
        The ``engines.node`` specifier if present, else ``None``.

    Line numbers are recovered via a secondary text scan.
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")
    data: Dict[str, Any] = json.loads(raw_text)

    engines_node: Optional[str] = None
    deps: List[Dependency] = []

    engines = data.get("engines", {})
    if "node" in engines:
        engines_node = engines["node"]

    dep_sections = {
        **data.get("dependencies", {}),
        **data.get("devDependencies", {}),
    }

    for pkg_name, constraint in dep_sections.items():
        lineno, raw = _find_json_value_line(raw_text, pkg_name, pkg_name)
        if lineno is None:
            lineno = _find_json_key_line(raw_text, pkg_name)
            raw = None
        # Rebuild a cleaner raw_line if we have position
        if lineno is not None:
            raw_text_lines = raw_text.splitlines()
            raw = raw_text_lines[lineno - 1].strip()
        deps.append(
            Dependency(
                name=pkg_name,
                declared_constraint=constraint,
                sources=[
                    SourceRef(
                        file=str(file_path),
                        line=lineno,
                        raw_line=raw or f'"{pkg_name}": "{constraint}"',
                    )
                ],
            )
        )

    return deps, engines_node


# ---------------------------------------------------------------------------
# package-lock.json  (npm v2/v3 format)
# ---------------------------------------------------------------------------

def parse_package_lock_json(path: str) -> List[Dependency]:
    """
    Parse an npm ``package-lock.json`` (lockfileVersion 2 or 3).

    Returns one :class:`Dependency` per locked package, with
    ``resolved_version`` set to the pinned version.

    Line numbers are recovered by scanning the raw text for
    ``"<package>"``.
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")
    data: Dict[str, Any] = json.loads(raw_text)

    deps: List[Dependency] = []

    # lockfileVersion 2/3: "packages" is the authoritative map
    packages = data.get("packages", {})
    for pkg_path_key, pkg_data in packages.items():
        if not pkg_path_key:  # "" is the root package entry
            continue
        # pkg_path_key looks like "node_modules/express" or
        # "node_modules/@scope/name"
        pkg_name = pkg_path_key.removeprefix("node_modules/")
        version = pkg_data.get("version", "")
        # Find in raw text: look for the path key "node_modules/<name>"
        lineno = _find_json_key_line(raw_text, pkg_path_key)
        raw: Optional[str] = None
        if lineno:
            raw = raw_text.splitlines()[lineno - 1].strip()

        deps.append(
            Dependency(
                name=pkg_name,
                declared_constraint=pkg_data.get("version", "*"),
                resolved_version=version or None,
                sources=[
                    SourceRef(
                        file=str(file_path),
                        line=lineno,
                        raw_line=raw or f'"{pkg_path_key}": ...',
                    )
                ],
            )
        )

    return deps


# ---------------------------------------------------------------------------
# Environment-config minimal parsers
# ---------------------------------------------------------------------------

def read_version_pin_file(path: str) -> Optional[str]:
    """
    Read a bare-version-pin file (``.python-version`` or ``.nvmrc``).

    Returns the stripped version string, or ``None`` if the file is empty
    or only whitespace.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    # .nvmrc sometimes contains "lts/*" or "node" — return as-is
    return text if text else None
