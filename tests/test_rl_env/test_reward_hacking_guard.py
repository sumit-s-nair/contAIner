"""
tests/test_rl_env/test_reward_hacking_guard.py
===============================================
Reward-hacking guard tests.

These tests are HARD ASSERTIONS, not soft checks.  They guard against
two real exploit patterns a learned RL policy is likely to discover:

Case 1 — Trivial no-op (exit 0, no action)
    Command: "true"  (Unix no-op, always exits 0)
    Exploit: policy spams "true" to get SUCCESS status reward.
    Guard: validator sees no matching ActionType → structural penalty dominates.
    Assert: total_reward <= 0.0

Case 2 — Realistic verify-fail exploit (MORE IMPORTANT)
    Command matches a valid ActionType pattern AND exits 0, but the
    verify check fails — the command had no real effect.
    Example: "pip install requests --dry-run"
      - Matches INSTALL ActionType pattern (structural reward neutral/positive)
      - Exits 0 (status == SUCCESS)
      - verify_command fails ("import requests" fails because nothing was installed)
      - ExecutionResult.verified == False
    Exploit: policy learns to use dry-run flags or stub commands that pass
    the exit-code gate but don't change system state.
    Guard: r_success requires BOTH exit-0 AND verified==True.
           Unverified success → r_success_unverified < r_success.
    Assert: total_reward < r_success * w_execution
            i.e. reward is strictly less than what full-verified success would give.

Both assertions are HARD (pytest assert with explicit failure messages),
not warnings or log checks.

Design note on Case 2
----------------------
This is the most important test.  The trivial no-op (Case 1) is an obvious
exploit that a policy is unlikely to find because the validator will penalize
it heavily on the first few rollouts.  The realistic verify-fail exploit
(Case 2) is what a policy trained with a naive exit-code-only reward would
converge to: find commands that look structurally valid AND exit 0, but do
nothing useful.  The ``verified`` flag in ``ExecutionResult`` is the only
thing that blocks this.
"""

from __future__ import annotations

import pytest

from src.repo_scan.models import (
    Dependency, EcosystemManifest, RepoManifest, SourceRef
)
from src.sandbox.models import (
    AtomicStep, ClassificationResult, ExecutionResult, ExecutionStatus, RiskTier
)
from src.system2_planner.models import TemplateInstance
from src.system2_planner.templates import TEMPLATES
from src.rl_env.reward import RewardConfig, compute_reward


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_conflict_manifest() -> RepoManifest:
    """A manifest with dependency conflicts (requires ISOLATE before INSTALL)."""
    manifest = RepoManifest()
    eco = EcosystemManifest(
        ecosystem="python",
        manifest_files=["requirements.txt"],
    )
    eco.add_dependency(Dependency(
        name="requests",
        declared_constraint=">=2.0",
        sources=[SourceRef(file="requirements.txt", line=1,
                           raw_line="requests>=2.0")],
    ))
    eco.add_dependency(Dependency(
        name="flask",
        declared_constraint=">=2.0",
        sources=[SourceRef(file="requirements.txt", line=2,
                           raw_line="flask>=2.0")],
    ))
    manifest.ecosystems["python"] = eco
    manifest.conflicts = [
        "Version conflict: requests>=2.0 conflicts with flask 1.x pinning"
    ]
    return manifest


def _make_setup_project_instance(manifest: RepoManifest) -> TemplateInstance:
    return TemplateInstance(
        template=TEMPLATES["setup_project"],
        manifest=manifest,
    )


def _safe_exec_result(step: AtomicStep, *, verified: bool) -> ExecutionResult:
    """Build a synthetic ExecutionResult for a SAFE, exit-0 step."""
    return ExecutionResult(
        step=step,
        status=ExecutionStatus.SUCCESS,
        tier=RiskTier.SAFE,
        classification=ClassificationResult(
            tier=RiskTier.SAFE,
            reason="Matches known safe action type.",
        ),
        returncode=0,
        stdout="",
        verified=verified,
        verify_returncode=0 if verified else 1,
        verify_stdout="ok" if verified else "FAILED",
    )


# ---------------------------------------------------------------------------
# Case 1: Trivial no-op
# ---------------------------------------------------------------------------

class TestTrivialNoOp:
    """
    'true' command — always exits 0, does nothing.

    Expected behaviour:
    - Validator: 'true' maps to no recognizable ActionType → treated as CHECK
      (neutral/default).  For setup_project with conflicts, CHECK is not in
      the expected sequence (ISOLATE → INSTALL), so the plan is invalid.
    - Execution: SUCCESS but verified=False (dry-run fake success uses True,
      so we construct the result manually as unverified).
    - Total reward must be <= 0.0.
    """

    def test_noop_receives_nonpositive_reward(self) -> None:
        manifest = _make_conflict_manifest()
        instance = _make_setup_project_instance(manifest)
        cfg      = RewardConfig()

        step = AtomicStep(
            command="true",
            description="no-op",
            verify_command=None,   # no verify command → unverified
        )

        # Construct unverified SUCCESS (exit 0 but no real effect)
        exec_result = _safe_exec_result(step, verified=False)

        total, breakdown = compute_reward(
            step=step,
            plan_so_far=[step],
            instance=instance,
            execution_result=exec_result,
            episode_complete=False,
            config=cfg,
        )

        assert total <= 0.0, (
            f"No-op step 'true' received positive reward {total:.4f}.\n"
            f"Breakdown: {breakdown}\n"
            "This indicates the reward function is not guarding against "
            "trivial exit-0 steps that do nothing.  The validator penalty "
            "must dominate when the step has no matching ActionType."
        )

    def test_noop_validated_as_invalid(self) -> None:
        """Confirm the validator explicitly rejects a plan containing only 'true'."""
        from src.system2_planner.validator import validate_plan
        from src.system2_planner.models import PlannedStep, ActionType

        manifest = _make_conflict_manifest()
        instance = _make_setup_project_instance(manifest)

        # 'true' maps to CHECK (default fallback in _atomic_to_planned)
        plan = [PlannedStep(action_type=ActionType.CHECK, description="no-op")]
        result = validate_plan(instance, plan)

        assert not result.is_valid, (
            "Validator accepted a plan containing only CHECK for a setup_project "
            "episode with conflicts.  Expected: ISOLATE → INSTALL."
        )
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Case 2: Realistic verify-fail exploit
# ---------------------------------------------------------------------------

class TestVerifyFailExploit:
    """
    Command matches a valid ActionType AND exits 0, but verify fails.

    Simulates: 'pip install requests --dry-run'
      - Exits 0 (real pip --dry-run exits 0)
      - Matches INSTALL pattern → validator sees a valid INSTALL step
        (for a no-conflict setup_project, INSTALL is the only required action)
      - verify_command: 'python -c "import requests"' → fails (nothing installed)
      - verified = False

    The reward must be strictly less than the full success reward
    (r_success * w_execution) because verified == False.

    This is the primary anti-exploit guard.
    """

    def test_verify_fail_does_not_earn_full_success_reward(self) -> None:
        # No-conflict manifest: setup_project only requires INSTALL
        manifest = RepoManifest()
        eco = EcosystemManifest(
            ecosystem="python",
            manifest_files=["requirements.txt"],
        )
        eco.add_dependency(Dependency(
            name="requests",
            declared_constraint=">=2.0",
            sources=[SourceRef(file="requirements.txt", line=1,
                               raw_line="requests>=2.0")],
        ))
        manifest.ecosystems["python"] = eco
        # No conflicts → INSTALL only

        instance = TemplateInstance(
            template=TEMPLATES["setup_project"],
            manifest=manifest,
        )
        cfg = RewardConfig()

        # Step: pip install requests --dry-run
        step = AtomicStep(
            command="pip install requests --dry-run",
            description="install requests package",
            verify_command='python -c "import requests"',
        )

        # Execution: exit-0 but verify failed (verified=False)
        exec_result = _safe_exec_result(step, verified=False)

        total, breakdown = compute_reward(
            step=step,
            plan_so_far=[step],
            instance=instance,
            execution_result=exec_result,
            episode_complete=False,
            config=cfg,
        )

        # What FULL verified success would have given
        full_success_execution_reward = cfg.w_execution * cfg.r_success
        # Validator reward for a valid INSTALL-only plan
        valid_validator_reward = cfg.w_validator * cfg.r_valid

        full_success_total = (
            full_success_execution_reward
            + valid_validator_reward
            - cfg.w_step_penalty   # step penalty always deducted
        )

        assert total < full_success_total, (
            f"Verify-fail step earned reward {total:.4f}, which equals or exceeds "
            f"the full verified-success reward {full_success_total:.4f}.\n"
            f"Breakdown: {breakdown}\n"
            "This means the reward function grants full credit for a step that "
            "exited 0 but failed its verify check.  This is the primary exploit "
            "vector: a policy can find dry-run commands that return 0 without "
            "changing system state.  Fix: r_success must require verified==True."
        )

    def test_verify_fail_reward_strictly_less_than_r_success_execution(self) -> None:
        """
        The execution component alone for unverified must be < r_success * w_execution.
        """
        cfg = RewardConfig()
        full_exec = cfg.w_execution * cfg.r_success
        unverified_exec = cfg.w_execution * cfg.r_success_unverified

        assert unverified_exec < full_exec, (
            f"r_success_unverified ({cfg.r_success_unverified}) * w_execution "
            f"({cfg.w_execution}) = {unverified_exec:.4f} is NOT less than "
            f"r_success ({cfg.r_success}) * w_execution = {full_exec:.4f}.\n"
            "Check RewardConfig: r_success_unverified must be strictly less than r_success."
        )

    def test_verified_false_with_realistic_step_gets_partial_credit(self) -> None:
        """
        Confirmed: unverified success yields partial credit, not zero and not full.
        The reward must be in (r_failed * w_execution, r_success * w_execution).
        """
        manifest = RepoManifest()
        eco = EcosystemManifest(ecosystem="python", manifest_files=["requirements.txt"])
        eco.add_dependency(Dependency(
            name="numpy",
            declared_constraint=">=1.0",
            sources=[SourceRef(file="requirements.txt", line=1, raw_line="numpy>=1.0")],
        ))
        manifest.ecosystems["python"] = eco

        instance = TemplateInstance(template=TEMPLATES["setup_project"], manifest=manifest)
        cfg = RewardConfig()

        step = AtomicStep(
            command="pip install numpy",
            description="install numpy package",
            verify_command='python -c "import numpy"',
        )
        exec_result = _safe_exec_result(step, verified=False)

        total, breakdown = compute_reward(
            step=step,
            plan_so_far=[step],
            instance=instance,
            execution_result=exec_result,
            episode_complete=False,
            config=cfg,
        )

        lower_bound = cfg.w_execution * cfg.r_failed
        upper_bound = cfg.w_execution * cfg.r_success

        # We check the execution component only (to isolate the signal)
        exec_component = breakdown["execution"]
        assert lower_bound < exec_component < upper_bound, (
            f"Unverified success execution reward {exec_component:.4f} is not in "
            f"({lower_bound:.4f}, {upper_bound:.4f}).\n"
            f"Breakdown: {breakdown}\n"
            "r_success_unverified must be strictly between r_failed and r_success."
        )
