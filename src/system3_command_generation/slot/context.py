"""Execution context System 3 sees for one planner step."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.system2_planner.models import ActionType

# The sandbox image (docker/sandbox_python) is python:3.11-slim.  Per-repo
# 3.12/3.13 images are an open gap (handoff v13); keep this in one place.
SANDBOX_PYTHON = "3.11"
WORKDIR = "/workspace_rw"

STDERR_MAX_CHARS = 1200
_NOISE = re.compile(r"^\s*\[notice\]|^\s*$|^\s*(Downloading|Collecting|Using cached|Obtaining)\b")
_ERRORISH = re.compile(r"error|failed|cannot|could not|conflict|requires|not found|no such|denied", re.I)


def clean_stderr(text: Optional[str], max_chars: int = STDERR_MAX_CHARS) -> str:
    """Keep the informative part of a pip/pytest stderr tail.

    Drops pip upgrade notices and download chatter, keeps every error-ish line
    plus the last lines, and caps the length (from the end: pip prints the root
    cause last).
    """
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    kept = [ln for ln in lines if not _NOISE.search(ln)]
    errorish = [ln for ln in kept if _ERRORISH.search(ln)]
    tail = kept[-12:]
    seen, out = set(), []
    for ln in errorish + tail:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    # keep original order
    order = {ln: i for i, ln in enumerate(kept)}
    out.sort(key=lambda ln: order.get(ln, 0))
    joined = "\n".join(out)
    if len(joined) > max_chars:
        joined = "..." + joined[-(max_chars - 3):]
    return joined


@dataclass
class Sys3Context:
    os: str = "linux"
    shell: str = "bash"
    python_version: str = SANDBOX_PYTHON
    workdir: str = WORKDIR
    venv_active: bool = False
    manifest_paths: Dict[str, List[str]] = field(default_factory=dict)
    # Last failed attempt at the SAME (action_type, target), if any:
    # {"commands": [...], "returncode": int|None, "stderr": str}
    previous: Optional[Dict[str, Any]] = None
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sys3Context":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def context_from_env(env: Any, action: Any) -> Sys3Context:
    """Build the context for *action* from a PlannerEnv's episode state.

    Uses only state the live pipeline also has: manifest paths, the steps
    already planned, what each expanded to, and how each execution ended.
    """
    from src.sandbox.models import ExecutionStatus  # local: avoid import cycles

    obs = getattr(env, "_obs", None)
    plan = list(getattr(env, "_plan_so_far", []) or [])
    history = list(getattr(env, "_execution_history", []) or [])
    expanded = list(getattr(env, "_expanded_history", []) or [])

    venv = any(
        p.action_type == ActionType.ISOLATE and e.status == ExecutionStatus.SUCCESS
        for p, e in zip(plan, history)
    )

    def _key(p):
        return (p.action_type, (getattr(p, "target", "") or "").strip())

    same = [i for i, p in enumerate(plan) if _key(p) == _key(action)]
    previous = None
    if same:
        i = same[-1]
        e = history[i] if i < len(history) else None
        if e is not None and not (e.status == ExecutionStatus.SUCCESS and e.verified):
            cmds = [s.command for s in expanded[i]] if i < len(expanded) else [e.step.command]
            previous = {
                "commands": cmds,
                "returncode": e.returncode,
                "stderr": clean_stderr(e.stderr or e.stdout),
            }

    return Sys3Context(
        venv_active=venv,
        manifest_paths=dict(getattr(obs, "manifest_paths", {}) or {}) if obs is not None else {},
        previous=previous,
        attempt=len(same) + 1,
    )
