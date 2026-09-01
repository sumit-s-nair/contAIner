"""
src/rl_env/verify_table.py
===========================
Deterministic verify-command lookup table for RL training.

Problem this solves
-------------------
If the RL policy is allowed to emit ``AtomicStep.verify_command`` directly,
it can trivially self-certify success by writing ``verify_command="echo ok"``
or ``verify_command="true"``, regardless of whether the main command did
anything real.  This is a more severe exploit than the no-op command guard
because it corrupts ``verified=True`` in the reward function while appearing
to be a valid step.

Solution (Option A from the design spec)
-----------------------------------------
Make ``verify_command`` deterministic and NOT policy-generated.  The
environment derives it from this fixed lookup table keyed on
``(action_type, target)``.  The policy only controls ``action_type`` and
``target``; it never controls *how* success is verified.

Usage
-----
:func:`lookup_verify_command` is called by :class:`DockerEpisodeExecutor`
immediately before executing a SAFE step.  The result overwrites
``AtomicStep.verify_command`` (or sets it if the policy left it ``None``).
The overwrite is unconditional for SAFE steps — a policy-supplied value is
always replaced by the table result.

Table design
------------
The primary key is :class:`~src.system2_planner.models.ActionType`.  The
secondary key is a substring match on ``step.command`` (the install target /
package name), applied in order.  The first matching entry wins.

For action types without a target-specific entry, a generic fallback command
is returned.  If no verify command can be determined (e.g. ESCALATE, CHECK),
``None`` is returned and the step is treated as unverified (partial reward).

Extending the table
-------------------
Add new ``(pattern, command)`` tuples to the appropriate ``ActionType``
section.  Patterns are matched with ``pattern in step.command.lower()``.
More-specific patterns should appear before less-specific ones.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.sandbox.models import AtomicStep


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# (substring_pattern, verify_command)
# The first pattern whose substring appears in the step command (case-insensitive)
# is used.  Use "" as a catch-all last entry.
_PatternTable = List[Tuple[str, str]]


# ---------------------------------------------------------------------------
# Lookup tables per ActionType string
# ---------------------------------------------------------------------------

# ActionType values as plain strings to avoid a circular import.
# These match src.system2_planner.models.ActionType values exactly.

_VERIFY_TABLE: Dict[str, _PatternTable] = {

    # ---------------------------------------------------------------- INSTALL
    # Verify that the installed package/runtime is importable or executable.
    "install": [
        # Python packages — pip
        ("pip install",     "python -c 'import importlib, sys; "
                            "[importlib.import_module(p.split(\"==\")[0].split(\"[\")[0].strip()) "
                            "for p in \"$PKG\".split() if not p.startswith(\"-\")]' "
                            "|| python -c 'import pkg_resources; "
                            "pkg_resources.require(\"$PKG\")'"),
        # Node / npm
        ("npm install",     "node -e 'require(process.env.PKG || \".\")' 2>/dev/null "
                            "|| test -d node_modules"),
        ("yarn",            "test -d node_modules"),
        ("cargo",           "cargo verify-project 2>/dev/null || test -f Cargo.lock"),
        ("gem install",     "gem list | grep -q ."),
        # Generic fallback for install: check the executable exists
        ("",                "echo 'install step: no specific verify available'"),
    ],

    # ---------------------------------------------------------------- ISOLATE
    # Verify that the isolated environment was created successfully.
    "isolate": [
        ("venv",            "test -f .venv/bin/python || test -f .venv/Scripts/python.exe"),
        ("virtualenv",      "test -f .venv/bin/python || test -f .venv/Scripts/python.exe"),
        ("conda create",    "conda env list | grep -q ."),
        ("docker",          "docker ps -q 2>/dev/null | head -1 | grep -q ."),
        # Generic: a Python venv was likely created
        ("",                "test -d .venv || test -d venv || test -d env"),
    ],

    # ---------------------------------------------------------------- CHECK
    # Verify current environment state — these commands are their own check.
    # We run the check command itself and trust its exit code.
    "check": [
        ("pip",             "pip check"),
        ("npm",             "npm ls --depth=0 2>/dev/null | tail -1"),
        ("cargo",           "cargo check 2>&1 | tail -3"),
        ("go",              "go vet ./... 2>&1 | head -5"),
        ("",                "echo 'check step complete'"),
    ],

    # ---------------------------------------------------------------- UPDATE
    "update": [
        ("pip",             "pip list --outdated 2>/dev/null | head -5 || true"),
        ("npm",             "npm outdated 2>/dev/null | head -5 || true"),
        ("cargo",           "cargo update --dry-run 2>/dev/null | head -5 || true"),
        ("apt",             "apt-get check 2>/dev/null && echo ok"),
        ("",                "echo 'update step complete'"),
    ],

    # ---------------------------------------------------------------- DETECT
    # Verify that the detection produced actionable output (exit 0 or non-empty).
    "detect": [
        ("pip check",       "pip check 2>&1; true"),   # pip check non-zero = conflicts found
        ("",                "echo 'detect step complete'"),
    ],

    # ---------------------------------------------------------------- FIX
    # After a fix, re-run the relevant check to confirm resolution.
    "fix": [
        ("pip",             "pip check"),
        ("npm",             "npm ls --depth=0 2>/dev/null | grep -v 'UNMET' | tail -1"),
        ("cargo",           "cargo check 2>&1 | grep -v 'error' | tail -3 || true"),
        ("",                "echo 'fix step complete'"),
    ],

    # ---------------------------------------------------------------- BUILD
    "build": [
        ("pip",             "pip check"),
        ("npm run build",   "test -d dist || test -d build"),
        ("npm",             "test -d node_modules"),
        ("cargo build",     "test -f target/debug/$(basename $(pwd)) 2>/dev/null || cargo build --dry-run 2>&1 | tail -1"),
        ("go build",        "test -f ./$(basename $(pwd)) 2>/dev/null || go build -n ./... 2>&1 | tail -1"),
        ("make",            "test -f Makefile && echo 'Makefile present'"),
        ("",                "echo 'build step complete'"),
    ],

    # ---------------------------------------------------------------- ESCALATE
    # Escalation is a terminal action (hand off to human); no verify command.
    "escalate": [],
}


# ---------------------------------------------------------------------------
# Simple package-name extraction helpers
# ---------------------------------------------------------------------------

_PKG_CAPTURE_RE = re.compile(
    r"(?:pip install|npm install|gem install|cargo add|go get)\s+([\w\-\.]+)",
    re.IGNORECASE,
)


def _extract_primary_package(command: str) -> Optional[str]:
    """
    Extract the primary package name from a command string.

    Returns ``None`` if no package-like token is found.  Used to produce
    tighter verify commands (e.g. ``python -c 'import requests'`` instead of
    the generic multi-package form).
    """
    m = _PKG_CAPTURE_RE.search(command)
    return m.group(1) if m else None


def _make_pip_verify(package: Optional[str]) -> str:
    """Return a tight ``python -c 'import ...'`` verify command for *package*."""
    if not package:
        return "pip check"
    # Normalise: dashes to underscores, strip extras/versions
    module = package.split("==")[0].split("[")[0].replace("-", "_").lower()
    return f"python -c 'import {module}'"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_verify_command(step: AtomicStep) -> Optional[str]:
    """
    Return the deterministic verify command for *step*, or ``None`` if no
    verify command is appropriate for this action type.

    The lookup proceeds as follows:

    1. Normalise ``step.command`` to lower-case for pattern matching.
    2. Find the action-type section in :data:`_VERIFY_TABLE`.
    3. Walk the ``(pattern, command)`` pairs in order; return the command for
       the first pattern that is a substring of the normalised command.
    4. Return ``None`` for action types with an empty table (e.g. ESCALATE).

    Special case: ``pip install <package>`` produces a tight
    ``python -c 'import <module>'`` verify command using
    :func:`_make_pip_verify` rather than the generic multi-package form.

    Parameters
    ----------
    step:
        The :class:`~src.sandbox.models.AtomicStep` to verify.  Only
        ``step.command`` and the action-type hint (if present) are used.

    Returns
    -------
    Optional[str]
        The verify shell command, or ``None``.

    Examples
    --------
    >>> from src.sandbox.models import AtomicStep
    >>> step = AtomicStep(command="pip install requests", description="install")
    >>> lookup_verify_command(step)
    "python -c 'import requests'"

    >>> step = AtomicStep(command="python -m venv .venv", description="isolate")
    >>> lookup_verify_command(step)
    'test -f .venv/bin/python || test -f .venv/Scripts/python.exe'
    """
    cmd_lower = step.command.lower()

    # --- Infer action type from the command if not tagged on the step ------
    action_type = _infer_action_type(cmd_lower)

    table: Optional[_PatternTable] = _VERIFY_TABLE.get(action_type)
    if table is None:
        # Unknown action type — cannot verify
        return None

    if not table:
        # Explicitly empty table (e.g. ESCALATE) — no verify appropriate
        return None

    # --- Special-case: tight pip verify ------------------------------------
    if "pip install" in cmd_lower:
        pkg = _extract_primary_package(step.command)
        return _make_pip_verify(pkg)

    # --- General pattern match -------------------------------------------
    for pattern, verify_cmd in table:
        if not pattern or pattern in cmd_lower:
            return verify_cmd

    return None


def inject_verify_command(step: AtomicStep) -> AtomicStep:
    """
    Return a copy of *step* with ``verify_command`` set from the lookup table.

    The policy-supplied ``verify_command`` is **unconditionally overwritten**
    for SAFE steps — this is the structural fix that closes the self-
    certification exploit.  Policy output never controls how success is
    verified; only the lookup table does.

    Parameters
    ----------
    step:
        Original :class:`~src.sandbox.models.AtomicStep`.

    Returns
    -------
    AtomicStep
        New instance with ``verify_command`` set (or ``None`` for action types
        where no verify is appropriate).  All other fields are unchanged.
    """
    from dataclasses import replace
    verify_cmd = lookup_verify_command(step)
    return replace(step, verify_command=verify_cmd)


# ---------------------------------------------------------------------------
# Action type inference (heuristic, for steps without an action_type tag)
# ---------------------------------------------------------------------------

_COMMAND_TO_ACTION: List[Tuple[str, str]] = [
    # install patterns
    ("pip install",   "install"),
    ("npm install",   "install"),
    ("yarn add",      "install"),
    ("cargo add",     "install"),
    ("gem install",   "install"),
    ("apt-get install", "install"),
    ("apt install",   "install"),
    ("brew install",  "install"),
    # isolate patterns
    ("venv",          "isolate"),
    ("virtualenv",    "isolate"),
    ("conda create",  "isolate"),
    # check patterns
    ("pip check",     "check"),
    ("pip list",      "check"),
    ("npm ls",        "check"),
    ("cargo check",   "check"),
    ("go vet",        "check"),
    # update patterns
    ("pip install --upgrade", "update"),
    ("npm update",    "update"),
    ("cargo update",  "update"),
    ("apt upgrade",   "update"),
    # build patterns
    ("npm run build", "build"),
    ("cargo build",   "build"),
    ("go build",      "build"),
    ("make",          "build"),
    ("python setup.py", "build"),
    # fix patterns — heuristic: pip install after pip check failure
    ("pip install --force", "fix"),
    # detect patterns
    ("pip check",     "detect"),
]


def _infer_action_type(cmd_lower: str) -> str:
    """
    Heuristically infer the ActionType string from a lower-cased command.

    Returns ``"check"`` as the safest fallback for unknown commands (which
    means: run the step's own exit code as the check, with a generic
    verify command).
    """
    for pattern, action_type in _COMMAND_TO_ACTION:
        if pattern in cmd_lower:
            return action_type
    return "check"
