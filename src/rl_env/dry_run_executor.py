"""
src/rl_env/dry_run_executor.py
==============================
DryRunExecutor and TrainingReviewPolicy for RL rollouts.

During unattended RL training:

  1. There is no human present to approve REVIEW-tier steps.
  2. Actual subprocess execution may not be safe or reproducible.

This module provides:

  * ``TrainingReviewPolicy`` — a named, documented enum for how the
    training loop handles REVIEW-tier steps.  Default is ``AUTO_DENY``.

  * ``DryRunExecutor`` — a drop-in replacement for ``SandboxExecutor``
    that auto-handles both concerns without real subprocess execution
    (or with it, depending on ``dry_run`` mode).

TrainingReviewPolicy
--------------------
AUTO_DENY (default):
    Any step classified as REVIEW is automatically treated as denied.
    Status = ABORTED, reward = r_review_denied.  No subprocess is ever
    spawned for REVIEW steps during training.  This is the safe default
    because there is no human present during unattended RL rollouts.

AUTO_APPROVE:
    REVIEW steps are automatically approved and executed.  Use only in
    controlled sandbox environments where you have verified that REVIEW
    commands cannot damage the host.  NOT recommended for general use.

BLOCK_AS_HARD_STOP:
    Treat REVIEW-tier steps as if they were BLOCKED.  The episode is
    terminated immediately with the BLOCKED hard penalty.  Use when you
    want to train the policy to never generate REVIEW-tier actions at all.

BLOCKED steps
--------------
Regardless of ``TrainingReviewPolicy``, BLOCKED steps are ALWAYS:
  * Never executed (no subprocess spawned)
  * Returned with status = BLOCKED
  * Hard-penalized (r_blocked — never zero)

This is unconditional and not configurable.  The policy setting
only affects REVIEW-tier steps.

DryRunExecutor
--------------
In ``dry_run=True`` mode (default for smoke tests):
  * SAFE steps return ``SUCCESS + verified=True`` without running a subprocess.
  * REVIEW steps are handled per ``training_review_policy``.
  * BLOCKED steps always return BLOCKED (regardless of dry_run).

In ``dry_run=False`` mode:
  * SAFE steps are actually executed (subprocess).
  * REVIEW steps are handled per ``training_review_policy`` (AUTO_DENY skips).
  * BLOCKED steps are never executed.

Execution state persistence
----------------------------
The design intent is that one executor instance (and one sandbox
directory / container namespace) persists across ALL steps of a single
episode, so later steps can depend on the side effects of earlier steps
(e.g. INSTALL must see the runtime installed by a prior DETECT+FIX step).

In dry_run mode this is trivially true (no real state).
In real execution mode, callers must pass the same ``sandbox_root`` for
all steps in an episode — PlannerEnv handles this automatically.
"""

from __future__ import annotations

import subprocess
import sys
from enum import Enum
from typing import Optional

from src.sandbox.classifier import CommandRiskClassifier
from src.sandbox.models import (
    AtomicStep,
    ClassificationResult,
    ExecutionResult,
    ExecutionStatus,
    RiskTier,
)


# ---------------------------------------------------------------------------
# TrainingReviewPolicy
# ---------------------------------------------------------------------------

class TrainingReviewPolicy(str, Enum):
    """
    Controls how the training executor handles REVIEW-tier steps.

    During RL rollouts there is no human present, so "wait for approval"
    is not a valid option.  This enum names the available policies so the
    behavior is explicit, documented, and configurable — not implicit.

    AUTO_DENY (default):
        Treat all REVIEW-tier steps as denied.  Status = ABORTED.
        Reward = r_review_denied (slightly negative, not catastrophic).

    AUTO_APPROVE:
        Approve and execute all REVIEW-tier steps.  Use only in sandboxed
        environments.  Subject to the same verify flow as SAFE steps.

    BLOCK_AS_HARD_STOP:
        Treat REVIEW-tier steps as BLOCKED.  Hard penalty, episode halt.
    """

    AUTO_DENY        = "auto_deny"
    AUTO_APPROVE     = "auto_approve"
    BLOCK_AS_HARD_STOP = "block_as_hard_stop"


# ---------------------------------------------------------------------------
# DryRunExecutor
# ---------------------------------------------------------------------------

class DryRunExecutor:
    """
    Drop-in executor for RL training rollouts.

    Parameters
    ----------
    sandbox_root:
        Path to the episode's working directory.  Passed to the classifier
        for out-of-scope path detection.  All steps in one episode should
        use the same sandbox_root (state persistence).

    review_policy:
        How to handle REVIEW-tier steps.  Default: AUTO_DENY.

    dry_run:
        If True (default), SAFE steps return synthetic SUCCESS without
        running a subprocess.  BLOCKED and REVIEW are handled regardless.

    shell:
        Shell for real subprocess execution (used only when dry_run=False).
    """

    def __init__(
        self,
        sandbox_root:   Optional[str]        = None,
        *,
        review_policy:  TrainingReviewPolicy = TrainingReviewPolicy.AUTO_DENY,
        dry_run:        bool                 = True,
        shell:          str                  = "bash",
    ) -> None:
        self._classifier   = CommandRiskClassifier(sandbox_root=sandbox_root)
        self._review_policy = review_policy
        self._dry_run      = dry_run
        self._shell        = shell
        self._sandbox_root = sandbox_root

    # ------------------------------------------------------------------ public

    def execute(self, step: AtomicStep) -> ExecutionResult:
        """
        Process one step through classifier → policy gate → (optional) subprocess.

        BLOCKED steps never reach subprocess execution regardless of any setting.
        REVIEW steps are handled per ``review_policy``.
        SAFE steps are executed (or faked in dry_run mode).
        """
        classification = self._classifier.classify(step)

        # --- BLOCKED: unconditional hard stop -----------------------------------
        if classification.tier == RiskTier.BLOCKED:
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.BLOCKED,
                tier=RiskTier.BLOCKED,
                classification=classification,
                verified=False,
            )

        # --- REVIEW: policy-determined ----------------------------------------
        if classification.tier == RiskTier.REVIEW:
            return self._handle_review(step, classification)

        # --- SAFE: dry-run or real execution ----------------------------------
        return self._handle_safe(step, classification)

    # --------------------------------------------------------------- private

    def _handle_review(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        """Handle a REVIEW-tier step according to the configured policy."""
        if self._review_policy == TrainingReviewPolicy.AUTO_DENY:
            # Automatically deny — no subprocess, no randomness, deterministic
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.ABORTED,
                tier=RiskTier.REVIEW,
                classification=classification,
                verified=False,
                stderr=(
                    f"[DryRunExecutor] REVIEW step auto-denied by policy "
                    f"'{self._review_policy}': {step.command!r}"
                ),
            )

        elif self._review_policy == TrainingReviewPolicy.BLOCK_AS_HARD_STOP:
            # Escalate to BLOCKED treatment
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.BLOCKED,
                tier=RiskTier.REVIEW,   # keep original tier for audit
                classification=classification,
                verified=False,
                stderr=(
                    f"[DryRunExecutor] REVIEW step escalated to BLOCKED by policy "
                    f"'{self._review_policy}': {step.command!r}"
                ),
            )

        else:  # AUTO_APPROVE
            # Execute as if SAFE
            print(
                f"[DryRunExecutor] WARNING: REVIEW step auto-approved by policy "
                f"'{self._review_policy}': {step.command!r}",
                file=sys.stderr,
            )
            return self._handle_safe(step, classification)

    def _handle_safe(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        """Execute a SAFE (or auto-approved REVIEW) step."""
        if self._dry_run:
            return self._fake_success(step, classification)
        return self._run_subprocess(step, classification)

    def _fake_success(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        """Return synthetic SUCCESS+verified without running a subprocess."""
        return ExecutionResult(
            step=step,
            status=ExecutionStatus.SUCCESS,
            tier=classification.tier,
            classification=classification,
            returncode=0,
            stdout="[DryRunExecutor] Synthetic SUCCESS (dry_run=True)",
            verified=True,   # dry-run treats all SAFE steps as verified
            verify_returncode=0,
            verify_stdout="[DryRunExecutor] Synthetic VERIFY OK (dry_run=True)",
        )

    def _run_subprocess(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        """Run the step for real and then run its verify_command if present."""
        try:
            proc = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self._sandbox_root,
            )
            main_status = (
                ExecutionStatus.SUCCESS if proc.returncode == 0
                else ExecutionStatus.FAILED
            )

            # --- verify command -----------------------------------------------
            verified         = False
            verify_stdout    = ""
            verify_returncode = None

            if proc.returncode == 0 and step.verify_command:
                try:
                    vproc = subprocess.run(
                        step.verify_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=self._sandbox_root,
                    )
                    verify_returncode = vproc.returncode
                    verify_stdout     = vproc.stdout
                    verified          = (vproc.returncode == 0)
                except Exception as exc:
                    verify_stdout = f"[verify error] {exc}"

            return ExecutionResult(
                step=step,
                status=main_status,
                tier=classification.tier,
                classification=classification,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                verified=verified,
                verify_returncode=verify_returncode,
                verify_stdout=verify_stdout,
            )

        except Exception as exc:
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.ERROR,
                tier=classification.tier,
                classification=classification,
                stderr=str(exc),
                verified=False,
            )
