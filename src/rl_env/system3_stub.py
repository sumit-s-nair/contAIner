"""
src/rl_env/system3_stub.py
==========================
Deterministic stub to expand PlannedStep into AtomicStep(s).
This is a temporary placeholder for the real trained System 3 model.
TODO: Swap in the real trained System 3 model before this pipeline reflects the intended architecture.
"""

import re
import shlex
from pathlib import PurePosixPath
from typing import List, Optional
from packaging.requirements import Requirement, InvalidRequirement

from src.system2_planner.models import PlannedStep, ActionType
from src.sandbox.models import AtomicStep


# Stage 2G: INSTALL target forms
#   requirements file  "requirements.txt", "requirements/dev.txt"  -> pip install -r <file>
#   project            ".", "pyproject.toml", "setup.py", "setup.cfg", "sub/pyproject.toml"
#                                                                  -> pip install <dir>
#   several specs      "requests>=2 flask" / "requests, flask"     -> pip install <spec> <spec>
#   one spec           "requests>=2.0"                             -> pip install <spec> (verify pip show)
# Manifest / multi installs have no verify_command: the env judges them by the
# container's actual installed packages (ground truth), not by exit code.
_PROJECT_FILES = {"pyproject.toml", "setup.py", "setup.cfg"}
INSTALL_TIMEOUT = 300           # seconds, single-spec install
MANIFEST_INSTALL_TIMEOUT = 600  # seconds, -r / project / multi-spec install


def _safe_rel_path(target: str) -> Optional[PurePosixPath]:
    """Repo-relative path, or None if absolute / escaping / option-like."""
    t = target.replace("\\", "/")
    if not t or t.startswith("-"):
        return None
    p = PurePosixPath(t)
    if p.is_absolute() or ".." in p.parts:
        return None
    return p


def _install_steps(target: str, raw_target: str) -> List[AtomicStep]:
    rel = _safe_rel_path(target)
    if rel is not None and rel.suffix.lower() == ".txt":
        return [AtomicStep(
            command=f"python -m pip install -r {shlex.quote(str(rel))}",
            description=f"Install requirements from {rel}",
            verify_command=None, destructive=False, timeout=MANIFEST_INSTALL_TIMEOUT,
        )]
    if rel is not None and (target in (".", "./") or rel.name in _PROJECT_FILES):
        proj_dir = "." if target in (".", "./") else str(rel.parent)
        return [AtomicStep(
            command=f"python -m pip install {shlex.quote(proj_dir)}",
            description=f"Install project at {proj_dir}",
            verify_command=None, destructive=False, timeout=MANIFEST_INSTALL_TIMEOUT,
        )]
    tokens = [t for t in re.split(r"[\s,]+", target) if t]
    if len(tokens) >= 2:
        try:
            for t in tokens:
                Requirement(t)
        except InvalidRequirement:
            tokens = []
        if tokens:
            return [AtomicStep(
                command="python -m pip install " + " ".join(shlex.quote(t) for t in tokens),
                description=f"Install {', '.join(tokens)}",
                verify_command=None, destructive=False, timeout=MANIFEST_INSTALL_TIMEOUT,
            )]
    quoted_target = shlex.quote(target) if target else shlex.quote(raw_target)
    try:
        dist_name = Requirement(target).name
        verify_cmd = f"python -m pip show {shlex.quote(dist_name)}"
    except Exception:
        verify_cmd = "false"
    return [AtomicStep(
        command=f"python -m pip install {quoted_target}",
        description=f"Install {target}" if target else "Install dependencies",
        verify_command=verify_cmd, destructive=False, timeout=INSTALL_TIMEOUT,
    )]


def expand_planned_step(planned: PlannedStep) -> List[AtomicStep]:
    """
    Expands a high-level PlannedStep into one or more concrete AtomicSteps.
    This simulates System 3 code generation.
    """
    atype = planned.action_type
    # Using getattr for backward compatibility if PlannedStep was initialized without target
    raw_target = getattr(planned, 'target', '') or ''
    target = raw_target.strip() if isinstance(raw_target, str) else ''

    if atype == ActionType.INSTALL:
        return _install_steps(target, raw_target)
    elif atype == ActionType.ISOLATE:
        return [
            AtomicStep(
                command="python -m venv .venv",
                description="Create virtual environment",
                verify_command="test -f .venv/bin/python || test -f .venv/Scripts/python.exe",
                destructive=False,
            )
        ]
    elif atype == ActionType.CHECK:
        pytest_cmd = f"python -m pytest {shlex.quote(target)}" if target else "python -m pytest"
        return [
            AtomicStep(
                command="python -m pip install pytest==8.3.5",
                description="Install pytest",
                verify_command="python -m pip show pytest",
                destructive=False,
            ),
            AtomicStep(
                command=pytest_cmd,
                description=f"Run tests for {target}" if target else "Run tests",
                destructive=False,
                unverified_returncodes=frozenset({5}),
            ),
        ]
    # Default fallback
    return [
        AtomicStep(
            command="__UNMAPPED_ACTION__",
            description=planned.description,
            destructive=False,
        )
    ]
