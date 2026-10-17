"""Parse System 3 model output into AtomicSteps, and the command policy.

The policy rejects commands that would change the repository, escalate, or
pull code from outside PyPI.  It is applied to teacher candidates before they
are executed (so they never become training targets) and reported at
evaluation.  The sandbox's own risk classifier still runs on everything.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, List, Optional

from src.sandbox.models import AtomicStep

MAX_STEPS = 4
MAX_COMMAND_CHARS = 600
MIN_TIMEOUT, MAX_TIMEOUT, DEFAULT_TIMEOUT = 30, 900, 300


class Sys3ParseError(ValueError):
    def __init__(self, subtype: str, msg: str):
        super().__init__(f"{subtype}: {msg}")
        self.subtype = subtype


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _first_json(text: str) -> Any:
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                # tolerate trailing commas (seen in System 2 outputs)
                try:
                    obj, _ = dec.raw_decode(re.sub(r",\s*([}\]])", r"\1", text[i:]))
                    return obj
                except json.JSONDecodeError:
                    continue
    raise Sys3ParseError("JSON_DECODE_ERROR", "no JSON object found")


def parse_atomic_steps(text: str) -> List[AtomicStep]:
    obj = _first_json(text)
    steps = obj.get("steps") if isinstance(obj, dict) else obj
    if not isinstance(steps, list) or not steps:
        raise Sys3ParseError("MISSING_STEPS", "expected a non-empty 'steps' list")
    if len(steps) > MAX_STEPS:
        raise Sys3ParseError("TOO_MANY_STEPS", f"{len(steps)} > {MAX_STEPS}")
    out = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise Sys3ParseError("INVALID_STEP", f"step {i} is not an object")
        cmd = s.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            raise Sys3ParseError("MISSING_COMMAND", f"step {i} has no command")
        cmd = cmd.strip()
        if len(cmd) > MAX_COMMAND_CHARS or "\n" in cmd:
            raise Sys3ParseError("INVALID_COMMAND", f"step {i} command too long or multi-line")
        verify = s.get("verify_command")
        if verify is not None and (not isinstance(verify, str) or not verify.strip()):
            verify = None
        try:
            timeout = int(s.get("timeout", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT
        timeout = max(MIN_TIMEOUT, min(MAX_TIMEOUT, timeout))
        out.append(AtomicStep(
            command=cmd,
            description=str(s.get("description") or "")[:200],
            verify_command=verify.strip() if verify else None,
            destructive=False,
            timeout=timeout,
            # System 4 semantics, not the model's job: pytest exit 5 = no tests
            unverified_returncodes=frozenset({5}) if re.search(r"\bpytest\b", cmd) else frozenset(),
        ))
    return out


# --------------------------------------------------------------------------- policy

_BANNED = [
    (re.compile(r"(^|[;&|\s])sudo\b"), "sudo"),
    (re.compile(r"(^|[;&|\s])(rm|rmdir|mv|chmod|chown|truncate|shred)\b"), "file removal/move/permissions"),
    (re.compile(r"\bsed\s+(-[a-zA-Z]*i|--in-place)"), "in-place edit"),
    (re.compile(r"\bgit\s+(checkout|reset|clean|stash|apply|am|rm|mv|restore)\b"), "git tree change"),
    (re.compile(r"\b(curl|wget)\b"), "download"),
    (re.compile(r"\|\s*(ba|z)?sh\b"), "pipe to shell"),
    (re.compile(r"\b(apt|apt-get|yum|dnf|apk|brew|conda)\b"), "system package manager"),
    (re.compile(r"(--index-url|--extra-index-url|--trusted-host|--find-links|(^|\s)-[if]\s)"), "custom index"),
    (re.compile(r"\bgit\+|https?://"), "URL install"),
    # installs code that declares itself incompatible with the sandbox Python;
    # the ground truth (installed + constraints + pip check) cannot see that
    (re.compile(r"--ignore-requires-python"), "ignores Requires-Python"),
    (re.compile(r"(?<!-m )\b(pip|pip3)\s+install\b"), "use python -m pip"),
]
_REDIRECT = re.compile(r"(?:^|[^0-9&])>{1,2}\s*([^\s;&|]+)")
_TEE = re.compile(r"\btee\s+(?:-a\s+)?([^\s;&|]+)")


def policy_violation(command: str) -> Optional[str]:
    """Reason the command is not allowed, or None."""
    for rx, why in _BANNED:
        if rx.search(command):
            return why
    for rx in (_REDIRECT, _TEE):
        for path in rx.findall(command):
            if not (path.startswith("/tmp/") or path == "/dev/null"):
                return f"writes {path}"
    try:
        shlex.split(command)
    except ValueError:
        return "unbalanced quoting"
    return None


def steps_policy_violation(steps: List[AtomicStep]) -> Optional[str]:
    for s in steps:
        for c in (s.command, s.verify_command or ""):
            why = policy_violation(c) if c else None
            if why:
                return f"{why}: {c}"
    return None
