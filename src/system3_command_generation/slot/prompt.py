"""Prompt and target format for System 3 (shared by data, training, inference)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from src.sandbox.models import AtomicStep
from src.system2_planner.models import PlannedStep

from .context import Sys3Context

SYSTEM_PROMPT = (
    "You are System 3 of contAIner. You turn one step from the planner into the "
    "shell commands that carry it out inside a Linux container. Use the "
    "documentation given. If a previous attempt at the same step failed, read "
    "its error and change the commands so they get past it; do not repeat a "
    "command that already failed. Reply with JSON only."
)

OUTPUT_SPEC = (
    '{"steps": [{"command": "<shell command>", "description": "<short>", '
    '"verify_command": "<command that exits 0 on success, or null>", "timeout": <seconds>}]}'
)

RULES = (
    "Rules: 1 to 4 steps; run from the repository root; use `python -m pip`; "
    "no sudo; do not edit, move or delete repository files; no custom package "
    "indexes or URLs; temporary files only under /tmp."
)


def _step_dict(planned: Union[PlannedStep, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(planned, dict):
        at = planned.get("action_type")
        return {
            "action_type": getattr(at, "value", at),
            "target": planned.get("target") or "",
            "description": planned.get("description") or "",
        }
    return {
        "action_type": planned.action_type.value,
        "target": planned.target or "",
        "description": planned.description or "",
    }


def render_user_prompt(planned: Union[PlannedStep, Dict[str, Any]], ctx: Sys3Context, docs_text: str) -> str:
    step = _step_dict(planned)
    manifests = "; ".join(f"{eco}: {', '.join(p) or 'none'}" for eco, p in sorted(ctx.manifest_paths.items())) or "none found"
    lines = [
        "## Planner step",
        json.dumps(step, ensure_ascii=False),
        "",
        "## Environment",
        f"os: {ctx.os}; shell: {ctx.shell}; python: {ctx.python_version}; working directory: {ctx.workdir}",
        f"virtual environment .venv active: {'yes' if ctx.venv_active else 'no'}",
        f"manifests: {manifests}",
        f"attempt at this step: {ctx.attempt}",
    ]
    if ctx.previous:
        lines += [
            "",
            "## Previous attempt (failed)",
            "commands:",
            *[f"  {c}" for c in ctx.previous.get("commands", [])],
            f"exit code: {ctx.previous.get('returncode')}",
            "error output:",
            ctx.previous.get("stderr") or "(empty)",
        ]
    lines += ["", "## Documentation", docs_text or "(none)", "", "## Output", OUTPUT_SPEC, RULES]
    return "\n".join(lines)


def build_messages(planned, ctx: Sys3Context, docs_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(planned, ctx, docs_text)},
    ]


def steps_to_target_json(steps: List[AtomicStep]) -> str:
    """Canonical assistant target for SFT (compact, stable key order)."""
    out = []
    for s in steps:
        out.append({
            "command": s.command,
            "description": s.description or "",
            "verify_command": s.verify_command,
            "timeout": int(s.timeout) if getattr(s, "timeout", None) else 300,
        })
    return json.dumps({"steps": out}, ensure_ascii=False)
