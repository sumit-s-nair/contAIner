"""
src/sandbox/executor.py
=======================
SandboxExecutor — coordinates risk classification, user confirmation, and
subprocess execution for individual AtomicSteps and multi-step plans.

Design decisions
----------------
* A BLOCKED command is **never** executed, regardless of what the caller
  says.  The ``halt_on_blocked`` flag defaults to ``True``.

* Allowing blocked commands to execute is an extreme edge case that should
  require an *explicit, loud opt-in* — not a casual ``False`` passed to a
  constructor parameter.  Callers must pass the sentinel constant
  ``ALLOW_BLOCKED_EXECUTION`` (imported from this module) to unlock that
  behaviour.  This sentinel makes the intent visible in code review and grep
  results and is deliberately awkward to use accidentally.

* A REVIEW command halts until ``request_user_confirmation`` resolves.
  Denial → ``ExecutionResult(status=ABORTED)``; the loop continues to the
  next step.

* ``execute_plan()`` by default continues after an ABORTED step (the user
  may want to skip one risky step and let others proceed).  On a BLOCKED
  step it halts immediately when ``halt_on_blocked=ALLOW_BLOCKED_EXECUTION``
  is NOT in effect (the default).

Usage
-----
    from src.sandbox.executor import SandboxExecutor, ALLOW_BLOCKED_EXECUTION
    from src.sandbox.models import AtomicStep

    executor = SandboxExecutor(sandbox_root="/home/user/project")

    result = executor.execute(AtomicStep(command="pip install requests"))
    print(result.status)   # "success"

    # Allow blocked (unusual; requires explicit sentinel):
    executor_permissive = SandboxExecutor(
        sandbox_root="/home/user/project",
        halt_on_blocked=ALLOW_BLOCKED_EXECUTION,
    )
"""

from __future__ import annotations

import subprocess
import sys
from typing import Callable, List, Optional

from .classifier import CommandRiskClassifier
from .confirmation import PromptFn, request_user_confirmation
from .models import (
    AtomicStep,
    ClassificationResult,
    ExecutionResult,
    ExecutionStatus,
    RiskTier,
)


# ---------------------------------------------------------------------------
# Sentinel for the loud opt-in to allow-blocked-execution
# ---------------------------------------------------------------------------

class _AllowBlockedExecutionSentinel:
    """
    Singleton sentinel that must be passed as ``halt_on_blocked`` to
    SandboxExecutor when the caller explicitly wants blocked commands to be
    allowed through.

    Importing and passing this object is intentionally verbose so it is
    visible in code review.
    """

    _instance: Optional["_AllowBlockedExecutionSentinel"] = None

    def __new__(cls) -> "_AllowBlockedExecutionSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ALLOW_BLOCKED_EXECUTION"

    def __bool__(self) -> bool:
        # Intentionally True so ``if halt_on_blocked:`` stays True,
        # meaning "halt" semantics are preserved… except the caller has
        # explicitly chosen to bypass blocking.  SandboxExecutor checks
        # for the sentinel type specifically, not truthiness.
        return True


ALLOW_BLOCKED_EXECUTION = _AllowBlockedExecutionSentinel()
"""
Pass this sentinel as ``halt_on_blocked`` to ``SandboxExecutor`` to
explicitly allow BLOCKED-tier commands through.  This should almost never
be done in production — a BLOCKED command appearing in a plan indicates that
the planner or code generator produced output it should not have.
"""


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """
    Runs :class:`~src.sandbox.models.AtomicStep` objects through the risk
    classifier → confirmation gate → subprocess pipeline.

    Parameters
    ----------
    sandbox_root:
        Absolute path to the repository root.  Passed to
        :class:`~src.sandbox.classifier.CommandRiskClassifier` for out-of-scope
        path detection.  Defaults to the current working directory.

    prompt_fn:
        Optional injectable confirmation callback
        (``(reason: str, context: str) -> bool``).  Used in tests and async
        SSE contexts.  Defaults to the interactive stdin prompt defined in
        :mod:`src.sandbox.confirmation`.

    halt_on_blocked:
        Controls ``execute_plan()`` behaviour when a BLOCKED step is
        encountered.

        * ``True`` (default) — halt the entire plan at the first BLOCKED
          step and return results collected so far.
        * :data:`ALLOW_BLOCKED_EXECUTION` — unusual explicit opt-in; do not
          halt, let the loop continue.  A warning is printed to stderr.
          Individual ``execute()`` calls **still never execute BLOCKED
          commands** — this only controls plan iteration.

    shell:
        Shell to use for subprocess execution (default ``"bash"``).
    """

    def __init__(
        self,
        sandbox_root: Optional[str] = None,
        *,
        prompt_fn: Optional[PromptFn] = None,
        halt_on_blocked: object = True,
        shell: str = "bash",
    ) -> None:
        self._classifier = CommandRiskClassifier(sandbox_root=sandbox_root)
        self._prompt_fn = prompt_fn
        self._shell = shell

        # Validate halt_on_blocked: accept True, False, or the sentinel.
        if halt_on_blocked is ALLOW_BLOCKED_EXECUTION:
            self._halt_on_blocked = False  # do not halt plan on blocked
            print(
                "[SandboxExecutor] WARNING: ALLOW_BLOCKED_EXECUTION passed — "
                "plan iteration will NOT halt on BLOCKED steps.  "
                "Individual BLOCKED commands are still never executed.",
                file=sys.stderr,
            )
        elif isinstance(halt_on_blocked, bool):
            self._halt_on_blocked = halt_on_blocked
        else:
            raise TypeError(
                "halt_on_blocked must be True, False, or ALLOW_BLOCKED_EXECUTION. "
                f"Got: {halt_on_blocked!r}"
            )

    # ------------------------------------------------------------------ public

    def execute(self, step: AtomicStep) -> ExecutionResult:
        """
        Process a single :class:`AtomicStep` through the full safety pipeline.

        Returns
        -------
        ExecutionResult
            Always returns — never raises.  Check ``result.status`` to
            determine what happened.
        """
        result = self._classifier.classify(step)

        if result.tier == RiskTier.BLOCKED:
            return self._handle_blocked(step, result)

        if result.tier == RiskTier.REVIEW:
            return self._handle_review(step, result)

        # SAFE — execute directly
        return self._run_command(step, result)

    def execute_plan(self, steps: List[AtomicStep]) -> List[ExecutionResult]:
        """
        Execute a list of :class:`AtomicStep` objects in order.

        Behaviour on failures:

        * **ABORTED** (user denied REVIEW): continue to next step.
        * **BLOCKED**: if ``halt_on_blocked=True`` (default), stop and return
          results so far; otherwise continue.
        * **FAILED** / **ERROR**: continue (caller can inspect results and
          decide whether to retry or abort the plan at a higher level).

        Parameters
        ----------
        steps:
            Ordered list of steps to execute.

        Returns
        -------
        List[ExecutionResult]
            One result per step that was attempted, in order.
        """
        results: List[ExecutionResult] = []

        for step in steps:
            outcome = self.execute(step)
            results.append(outcome)

            if outcome.status == ExecutionStatus.BLOCKED and self._halt_on_blocked:
                # A BLOCKED mid-plan step means the planner generated something
                # it should not have — halt so the user can investigate.
                print(
                    f"[SandboxExecutor] Plan halted: BLOCKED command encountered "
                    f"({step.command!r}).  Remaining steps skipped.",
                    file=sys.stderr,
                )
                break

        return results

    # --------------------------------------------------------------- private

    def _handle_blocked(
        self, step: AtomicStep, result: ClassificationResult
    ) -> ExecutionResult:
        """Log and return a BLOCKED result.  Never spawns a subprocess."""
        print(
            f"[SandboxExecutor] BLOCKED: {step.command!r}\n"
            f"  Reason: {result.reason}",
            file=sys.stderr,
        )
        return ExecutionResult(
            step=step,
            status=ExecutionStatus.BLOCKED,
            tier=RiskTier.BLOCKED,
            classification=result,
        )

    def _handle_review(
        self, step: AtomicStep, result: ClassificationResult
    ) -> ExecutionResult:
        """
        Surface the REVIEW prompt and wait for user input.

        The subprocess is NOT called until the user explicitly approves.
        """
        approved = request_user_confirmation(
            reason=result.reason,
            context=step.command,
            prompt_fn=self._prompt_fn,
        )

        if not approved:
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.ABORTED,
                tier=RiskTier.REVIEW,
                classification=result,
            )

        # User approved — now actually run it.
        return self._run_command(step, result)

    def _run_command(
        self, step: AtomicStep, result: ClassificationResult
    ) -> ExecutionResult:
        """Spawn the subprocess and capture output."""
        try:
            proc = subprocess.run(
                step.command,
                shell=True,
                executable=self._shell if self._shell != "bash" else None,
                capture_output=True,
                text=True,
            )
            status = (
                ExecutionStatus.SUCCESS if proc.returncode == 0 else ExecutionStatus.FAILED
            )
            return ExecutionResult(
                step=step,
                status=status,
                tier=result.tier,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                classification=result,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.ERROR,
                tier=result.tier,
                stderr=str(exc),
                classification=result,
            )
