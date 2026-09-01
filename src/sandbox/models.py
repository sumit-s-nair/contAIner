"""
src/sandbox/models.py
=====================
Pure data classes for the sandbox execution harness.

No I/O, no pattern matching — only structure and derived helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Risk classification tier
# ---------------------------------------------------------------------------

class RiskTier(str, Enum):
    """
    Three-tier safety classification for shell commands.

    SAFE
        No destructive patterns detected, command matches a known safe action
        type (install, check, list, read-only query, update without a
        destructive force-flag context).  Auto-executed without user
        interaction.

    REVIEW
        Command matches at least one REVIEW-level pattern (rm, rmdir,
        privilege escalation, permission changes, planner-flagged destructive,
        or a force-flag in a genuinely risky verb context).  Execution halts;
        the user receives a plain-language explanation and must confirm before
        any subprocess is spawned.

    BLOCKED
        Command matches a hard denylist entry (disk formatting, fork bombs,
        root-level rm -rf, network exfil/reverse shell patterns, or writes
        outside the sandbox/repo scope).  Never executed.  No confirmation
        path is offered — the command is logged and refused unconditionally.
    """

    SAFE    = "SAFE"
    REVIEW  = "REVIEW"
    BLOCKED = "BLOCKED"


# ---------------------------------------------------------------------------
# AtomicStep — planner-generated execution unit
# ---------------------------------------------------------------------------

@dataclass
class AtomicStep:
    """
    A single shell command produced by the planner / command generator.

    Parameters
    ----------
    command:
        The literal shell command to run.

    description:
        Human-readable summary of what this step does, shown to the user in
        REVIEW confirmations and execution logs.

    destructive:
        Set by the *planner* when it knows the operation is inherently risky
        (e.g. a removal, overwrite, or force-push step).  The classifier
        will never downgrade a step with ``destructive=True`` to SAFE —
        it will be at least REVIEW regardless of pattern matching.
        Defaults to ``False``; the classifier will set it implicitly when a
        REVIEW or BLOCKED pattern fires and the planner did not set it.

    risk_reason:
        Optional plain-English explanation of *why* this step is marked
        destructive, written by the planner or filled in by the classifier.
        Surfaced verbatim to the user in the confirmation prompt.
    """

    command:        str
    description:    str            = ""
    destructive:    bool           = False
    risk_reason:    Optional[str]  = None
    verify_command: Optional[str]  = None
    """
    Optional shell command to run *after* the main command completes,
    inside the same execution context (container / working directory).

    The verify command must exit 0 for ``ExecutionResult.verified`` to be
    set to ``True``.  If ``None``, the step is treated as unverified (the
    reward function applies a partial credit penalty).

    Design intent: the planner should always emit a verify command so that
    the reward signal distinguishes "command ran" from "command had effect".
    Example: after ``pip install requests``, verify with
    ``python -c "import requests"`` rather than trusting the exit code alone.
    """


# ---------------------------------------------------------------------------
# ClassificationResult — output of CommandRiskClassifier
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    The outcome of running a command through ``CommandRiskClassifier``.

    Parameters
    ----------
    tier:
        The assigned :class:`RiskTier`.

    reason:
        Plain-English explanation surfaced to the user for REVIEW / BLOCKED
        outcomes, and written to the execution log for all outcomes.

    matched_pattern:
        The specific regex or rule name that triggered the classification,
        for debugging and audit logging.  ``None`` for SAFE results (no
        pattern fired).
    """

    tier:            RiskTier
    reason:          str
    matched_pattern: Optional[str] = None


# ---------------------------------------------------------------------------
# ExecutionResult — output of SandboxExecutor
# ---------------------------------------------------------------------------

class ExecutionStatus(str, Enum):
    """Terminal status of a single AtomicStep execution attempt."""

    SUCCESS  = "success"   # Command ran and exited 0
    FAILED   = "failed"    # Command ran and exited non-zero
    ABORTED  = "aborted"   # REVIEW-tier: user denied confirmation
    BLOCKED  = "blocked"   # BLOCKED-tier: refused unconditionally
    ERROR    = "error"     # Unexpected Python-level exception


@dataclass
class ExecutionResult:
    """
    Records the full outcome of attempting to execute one :class:`AtomicStep`.

    Parameters
    ----------
    step:
        The original :class:`AtomicStep` that was processed.

    status:
        Terminal outcome (see :class:`ExecutionStatus`).

    tier:
        The :class:`RiskTier` that was assigned by the classifier.

    stdout:
        Captured standard output (empty if command was not run).

    stderr:
        Captured standard error (empty if command was not run).

    returncode:
        Process exit code, or ``None`` if the command was never spawned.

    classification:
        The full :class:`ClassificationResult` for audit / logging.
    """

    step:           AtomicStep
    status:         ExecutionStatus
    tier:           RiskTier
    stdout:         str                       = ""
    stderr:         str                       = ""
    returncode:     Optional[int]             = None
    classification: Optional[ClassificationResult] = None
    extra:          Dict[str, Any]            = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Verification gate (added for RL reward computation)
    # -----------------------------------------------------------------
    verified: bool = False
    """
    Whether a post-execution *verify* command confirmed the step's side
    effects actually took place.

    This is *distinct* from the exit-code-derived ``status``.  A command
    can return exit-code 0 (``status == SUCCESS``) but leave the system
    unchanged — e.g. a dry-run install flag, or a package manager that
    silently ignores a bad version constraint.

    In the RL reward function:
      * ``status == SUCCESS and verified == True``  → full r_success reward
      * ``status == SUCCESS and verified == False`` → partial reward (between
        r_success and r_failed), because syntactic success ≠ semantic effect
      * ``verified`` is set by SandboxExecutor after running the AtomicStep's
        ``verify_command`` inside the same execution context as the step.

    Defaults to ``False``; callers that do not run a verify command should
    leave it unset — this is conservatively treated as "unverified" by the
    reward function.
    """

    verify_stdout: str = ""
    """Captured stdout of the verify command, if one was run."""

    verify_returncode: Optional[int] = None
    """Return code of the verify command, or ``None`` if not run."""
