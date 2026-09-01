"""
tests/test_rl_env/test_review_policy.py
========================================
Tests confirming REVIEW-tier step behavior during training rollouts.

Key invariants tested
---------------------
1. AUTO_DENY (default): A REVIEW-tier step NEVER reaches subprocess execution.
   Status is always ABORTED.  Reward is r_review_denied (slightly negative).
   This is deterministic — no randomness or flakiness.

2. BLOCKED: A BLOCKED-tier step NEVER reaches subprocess execution regardless
   of the review policy.  Hard penalty (r_blocked) always applies.

3. BLOCK_AS_HARD_STOP: REVIEW steps treated as BLOCKED — hard penalty.

4. Policy is named and configurable — the behavior is NOT implicit.

All tests use ``DryRunExecutor`` (or a mock classifier) to avoid real
subprocess execution.  The determinism check runs 10 iterations to confirm
there is no stochastic approval.
"""

from __future__ import annotations

import pytest

from src.sandbox.models import (
    AtomicStep, ExecutionStatus, RiskTier
)
from src.rl_env.dry_run_executor import DryRunExecutor, TrainingReviewPolicy
from src.rl_env.reward import RewardConfig, compute_reward
from src.repo_scan.models import (
    Dependency, EcosystemManifest, RepoManifest, SourceRef
)
from src.system2_planner.models import TemplateInstance
from src.system2_planner.templates import TEMPLATES


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _simple_manifest() -> RepoManifest:
    manifest = RepoManifest()
    eco = EcosystemManifest(ecosystem="python", manifest_files=["requirements.txt"])
    eco.add_dependency(Dependency(
        name="requests",
        declared_constraint=">=2.0",
        sources=[SourceRef(file="requirements.txt", line=1, raw_line="requests>=2.0")],
    ))
    manifest.ecosystems["python"] = eco
    return manifest


def _instance() -> TemplateInstance:
    return TemplateInstance(
        template=TEMPLATES["setup_project"],
        manifest=_simple_manifest(),
    )


# ---------------------------------------------------------------------------
# A REVIEW-tier step: 'rm' triggers REVIEW (file deletion)
# ---------------------------------------------------------------------------

REVIEW_STEP = AtomicStep(
    command="rm somefile.txt",
    description="remove a file",
)

# A BLOCKED step: 'rm -rf /' triggers BLOCKED
BLOCKED_STEP = AtomicStep(
    command="rm -rf /",
    description="delete everything",
)


# ---------------------------------------------------------------------------
# Test: AUTO_DENY policy
# ---------------------------------------------------------------------------

class TestAutoDenyPolicy:
    """Under AUTO_DENY, REVIEW steps are always ABORTED, never executed."""

    def test_review_step_returns_aborted(self) -> None:
        executor = DryRunExecutor(
            review_policy=TrainingReviewPolicy.AUTO_DENY,
            dry_run=True,
        )
        result = executor.execute(REVIEW_STEP)
        assert result.status == ExecutionStatus.ABORTED, (
            f"Expected ABORTED, got {result.status}.  "
            "AUTO_DENY policy must never execute REVIEW steps."
        )
        assert result.tier == RiskTier.REVIEW

    def test_review_step_never_runs_subprocess(self) -> None:
        """
        Verify no subprocess was spawned — confirmed by dry_run=False and
        checking that the command would be REVIEW-classified but still ABORTED.
        """
        executor = DryRunExecutor(
            review_policy=TrainingReviewPolicy.AUTO_DENY,
            dry_run=False,  # real mode, but AUTO_DENY should still prevent execution
        )
        result = executor.execute(REVIEW_STEP)
        # Even in non-dry-run mode, AUTO_DENY must abort before subprocess
        assert result.status == ExecutionStatus.ABORTED, (
            f"Expected ABORTED (no subprocess), got {result.status}.\n"
            "AUTO_DENY must skip subprocess execution even in dry_run=False mode."
        )
        assert result.returncode is None, (
            f"returncode should be None (no subprocess was run), got {result.returncode}"
        )

    def test_review_step_is_deterministic_no_randomness(self) -> None:
        """
        Run 10 iterations: REVIEW step must ALWAYS return ABORTED.
        No randomness, no stochastic approval.
        """
        executor = DryRunExecutor(
            review_policy=TrainingReviewPolicy.AUTO_DENY,
            dry_run=True,
        )
        statuses = []
        for _ in range(10):
            result = executor.execute(REVIEW_STEP)
            statuses.append(result.status)

        assert all(s == ExecutionStatus.ABORTED for s in statuses), (
            f"REVIEW step returned non-ABORTED status in some iterations: {statuses}\n"
            "AUTO_DENY must be fully deterministic — no stochastic approval."
        )

    def test_review_step_reward_is_r_review_denied(self) -> None:
        """The reward for an AUTO_DENY REVIEW step is w_execution * r_review_denied."""
        executor = DryRunExecutor(
            review_policy=TrainingReviewPolicy.AUTO_DENY,
            dry_run=True,
        )
        result = executor.execute(REVIEW_STEP)

        cfg = RewardConfig()
        inst = _instance()
        total, breakdown = compute_reward(
            step=REVIEW_STEP,
            plan_so_far=[REVIEW_STEP],
            instance=inst,
            execution_result=result,
            episode_complete=False,
            config=cfg,
        )

        expected_exec = cfg.w_execution * cfg.r_review_denied
        assert breakdown["execution"] == pytest.approx(expected_exec, rel=1e-6), (
            f"Expected execution reward {expected_exec:.4f}, got {breakdown['execution']:.4f}.\n"
            "Review-denied reward must equal w_execution * r_review_denied."
        )

        # Must be slightly negative, not catastrophic, not positive
        assert breakdown["execution"] < 0.0, (
            "Review-denied execution reward must be negative."
        )
        assert breakdown["execution"] > cfg.w_execution * cfg.r_blocked, (
            "Review-denied reward must be greater (less negative) than BLOCKED reward."
        )


# ---------------------------------------------------------------------------
# Test: BLOCKED steps — hard penalty regardless of review_policy
# ---------------------------------------------------------------------------

class TestBlockedStepsUnconditional:
    """BLOCKED steps are never executed regardless of review policy."""

    @pytest.mark.parametrize("policy", list(TrainingReviewPolicy))
    def test_blocked_step_always_blocked(self, policy: TrainingReviewPolicy) -> None:
        executor = DryRunExecutor(review_policy=policy, dry_run=True)
        result = executor.execute(BLOCKED_STEP)

        assert result.status == ExecutionStatus.BLOCKED, (
            f"BLOCKED step returned {result.status} under policy {policy}.\n"
            "BLOCKED steps must ALWAYS return BLOCKED, regardless of review_policy."
        )
        assert result.returncode is None, (
            "BLOCKED step should never have a returncode (no subprocess spawned)."
        )
        assert result.verified is False, (
            "BLOCKED step must never be verified."
        )

    @pytest.mark.parametrize("policy", list(TrainingReviewPolicy))
    def test_blocked_step_reward_is_hard_negative(self, policy: TrainingReviewPolicy) -> None:
        executor = DryRunExecutor(review_policy=policy, dry_run=True)
        result = executor.execute(BLOCKED_STEP)

        cfg = RewardConfig()
        inst = _instance()
        total, breakdown = compute_reward(
            step=BLOCKED_STEP,
            plan_so_far=[BLOCKED_STEP],
            instance=inst,
            execution_result=result,
            episode_complete=False,
            config=cfg,
        )

        expected_exec = cfg.w_execution * cfg.r_blocked
        assert breakdown["execution"] == pytest.approx(expected_exec, rel=1e-6), (
            f"BLOCKED execution reward {breakdown['execution']:.4f} != "
            f"expected {expected_exec:.4f}."
        )
        assert breakdown["execution"] < 0.0, (
            "BLOCKED reward must be negative."
        )
        # r_blocked must never be zero
        assert cfg.r_blocked < 0.0, (
            "RewardConfig.r_blocked must be strictly negative — never zero."
        )


# ---------------------------------------------------------------------------
# Test: BLOCK_AS_HARD_STOP policy
# ---------------------------------------------------------------------------

class TestBlockAsHardStopPolicy:
    """Under BLOCK_AS_HARD_STOP, REVIEW steps are treated as BLOCKED."""

    def test_review_step_treated_as_blocked(self) -> None:
        executor = DryRunExecutor(
            review_policy=TrainingReviewPolicy.BLOCK_AS_HARD_STOP,
            dry_run=True,
        )
        result = executor.execute(REVIEW_STEP)

        # Status should be BLOCKED (escalated from REVIEW)
        assert result.status == ExecutionStatus.BLOCKED, (
            f"Expected BLOCKED, got {result.status}.\n"
            "BLOCK_AS_HARD_STOP must escalate REVIEW to BLOCKED."
        )
        assert result.returncode is None

    def test_review_reward_equals_blocked_reward_under_block_policy(self) -> None:
        executor = DryRunExecutor(
            review_policy=TrainingReviewPolicy.BLOCK_AS_HARD_STOP,
            dry_run=True,
        )
        result = executor.execute(REVIEW_STEP)

        cfg = RewardConfig()
        inst = _instance()
        _, breakdown = compute_reward(
            step=REVIEW_STEP,
            plan_so_far=[REVIEW_STEP],
            instance=inst,
            execution_result=result,
            episode_complete=False,
            config=cfg,
        )

        expected_exec = cfg.w_execution * cfg.r_blocked
        assert breakdown["execution"] == pytest.approx(expected_exec, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: Policy is a named enum, not implicit
# ---------------------------------------------------------------------------

class TestPolicyIsExplicitAndNamed:
    """Confirm TrainingReviewPolicy is a documented enum with named values."""

    def test_auto_deny_is_default(self) -> None:
        executor = DryRunExecutor()
        assert executor._review_policy == TrainingReviewPolicy.AUTO_DENY, (
            "Default review_policy must be AUTO_DENY."
        )

    def test_all_policies_are_named(self) -> None:
        names = {p.value for p in TrainingReviewPolicy}
        assert "auto_deny" in names
        assert "auto_approve" in names
        assert "block_as_hard_stop" in names

    def test_policy_enum_has_string_value(self) -> None:
        assert TrainingReviewPolicy.AUTO_DENY == "auto_deny"
        assert TrainingReviewPolicy.AUTO_APPROVE == "auto_approve"
        assert TrainingReviewPolicy.BLOCK_AS_HARD_STOP == "block_as_hard_stop"
