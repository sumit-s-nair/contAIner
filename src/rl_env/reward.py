"""
src/rl_env/reward.py
====================
Reward function for System 2 planner GRPO-style RL training.

Formula
-------
    total = (w_validator  * step_validator_reward)
          + (w_execution  * step_execution_reward)
          + (w_completion * episode_completion_bonus)
          - (w_step_penalty)

All weights are in ``RewardConfig`` — nothing is hard-coded.

Execution reward tiers
-----------------------
BLOCKED   → r_blocked          (hard negative; NEVER zero — key anti-hacking signal)
ABORTED   → r_review_denied    (REVIEW-tier step denied by auto_deny policy)
SUCCESS + verified   → r_success        (full credit; requires BOTH exit-0 AND verified)
SUCCESS + !verified  → r_success_unverified  (partial credit; syntactic ≠ semantic)
FAILED    → r_failed
ERROR     → r_error

Validator reward
-----------------
Valid plan → r_valid bonus.
Invalid plan → r_invalid_base + r_invalid_per_error * error_count.

Episode completion bonus
------------------------
Applied once when ALL steps in the episode have been verified successfully.

Step penalty
-------------
Subtracted every step unconditionally to discourage plan padding.

Key design decision — verified flag
-------------------------------------
``r_success`` is only granted when BOTH ``status == SUCCESS`` AND
``verified == True``.  An exit-code-0 step that fails its verify command
gets ``r_success_unverified`` (between r_failed and r_success), which is
strictly less than r_success.  This is the primary guard against the most
realistic reward-hacking exploit: a command that returns 0 but has no effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.sandbox.models import AtomicStep, ExecutionResult, ExecutionStatus, RiskTier
from src.system2_planner.models import PlannedStep, TemplateInstance, ActionType
from src.system2_planner.validator import validate_plan


# ---------------------------------------------------------------------------
# RewardConfig — all weights are configurable for ablation
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """
    All reward weights and per-outcome values for the RL training loop.

    Every numeric value here is a knob — no constants are baked into
    ``compute_reward()``.  Change these to run ablations without touching
    the reward function logic.

    Weights (scale each term's contribution to the total)
    -------------------------------------------------------
    w_validator:   float = 1.0
        Weight on the structural validator signal.
    w_execution:   float = 2.0
        Weight on the sandbox execution signal.
    w_completion:  float = 5.0
        Weight on the episode completion bonus.
    w_step_penalty: float = 0.1
        Per-step penalty subtracted unconditionally (discourages padding).

    Per-outcome execution rewards (applied before w_execution scaling)
    -------------------------------------------------------------------
    r_blocked:             float = -3.0
        Hard negative for BLOCKED steps.  Must never be zero — this is the
        primary signal that the policy generated a dangerous action.
    r_review_denied:       float = -0.5
        For REVIEW-tier steps during training (auto_deny policy).
        Neutral-to-slightly-negative; the policy should learn to avoid REVIEW
        commands, not crash the episode.
    r_success:             float = 1.0
        Full success: status == SUCCESS AND verified == True.
    r_success_unverified:  float = 0.2
        Partial credit: status == SUCCESS but verified == False.
        Strictly less than r_success, strictly greater than r_failed.
    r_failed:              float = -1.0
        Command ran and exited non-zero.
    r_error:               float = -0.5
        Python-level exception during execution.

    Per-outcome validator rewards (applied before w_validator scaling)
    ------------------------------------------------------------------
    r_valid:             float = 1.0
        Bonus for a structurally valid plan up to this step.
    r_invalid_base:      float = -0.5
        Base penalty for an invalid plan.
    r_invalid_per_error: float = -0.25
        Additional penalty per validation error (multiplied by error count).
        Encourages the policy to fix ALL ordering violations, not just one.
    """

    # --- weights (scale each reward term) ------------------------------------
    w_validator:    float = 1.0
    w_execution:    float = 2.0
    w_completion:   float = 5.0
    w_step_penalty: float = 0.1

    # --- execution tier rewards ----------------------------------------------
    r_blocked:             float = -3.0   # hard negative; NEVER zero
    r_review_denied:       float = -0.5   # auto_deny during training
    r_success:             float = 1.0    # exit-0 AND verified
    r_success_unverified:  float = 0.2    # exit-0 but NOT verified
    r_failed:              float = -1.0   # non-zero exit
    r_error:               float = -0.5   # Python exception

    # --- validator rewards ---------------------------------------------------
    r_valid:             float =  1.0
    r_invalid_base:      float = -0.5
    r_invalid_per_error: float = -0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validator_reward(
    plan_so_far: List[PlannedStep],
    instance: TemplateInstance,
    cfg: RewardConfig,
) -> Tuple[float, dict]:
    """Run the structural validator and return scaled reward + breakdown."""
    result = validate_plan(instance, plan_so_far)

    if result.is_valid:
        raw = cfg.r_valid
        breakdown = {"validator_raw": raw, "valid": True, "errors": []}
    else:
        error_count = len(result.errors)
        raw = cfg.r_invalid_base + cfg.r_invalid_per_error * error_count
        breakdown = {"validator_raw": raw, "valid": False, "errors": result.errors}

    return cfg.w_validator * raw, breakdown


def _execution_reward(
    exec_result: ExecutionResult,
    cfg: RewardConfig,
) -> Tuple[float, dict]:
    """Map the execution result to a scaled reward + breakdown."""
    status = exec_result.status
    tier   = exec_result.tier

    # BLOCKED is unconditional — never 0, never negotiable
    if tier == RiskTier.BLOCKED or status == ExecutionStatus.BLOCKED:
        raw = cfg.r_blocked
        label = "blocked"
    elif status == ExecutionStatus.ABORTED:
        # REVIEW-tier auto-denied during training
        raw = cfg.r_review_denied
        label = "review_denied"
    elif status == ExecutionStatus.SUCCESS:
        if exec_result.verified:
            raw = cfg.r_success
            label = "success_verified"
        else:
            raw = cfg.r_success_unverified
            label = "success_unverified"
    elif status == ExecutionStatus.FAILED:
        raw = cfg.r_failed
        label = "failed"
    else:  # ERROR
        raw = cfg.r_error
        label = "error"

    return cfg.w_execution * raw, {
        "execution_raw":   raw,
        "execution_label": label,
        "verified":        exec_result.verified,
        "returncode":      exec_result.returncode,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    step:              PlannedStep,
    plan_so_far:       List[PlannedStep],
    instance:          TemplateInstance,
    execution_result:  ExecutionResult,
    episode_complete:  bool,
    config:            Optional[RewardConfig] = None,
) -> Tuple[float, dict]:
    """
    Compute the full reward signal for one step in a planner episode.

    Parameters
    ----------
    step:
        The ``PlannedStep`` just taken.
    plan_so_far:
        All ``PlannedStep`` objects taken in this episode *including* the
        current step.
    instance:
        The ``TemplateInstance`` for the current episode (template + manifest).
    execution_result:
        The ``ExecutionResult`` returned by the executor for this step.
    episode_complete:
        True if the episode has reached successful termination (all steps
        verified).  False if we're mid-episode.
    config:
        ``RewardConfig`` with all weights.  Defaults to ``RewardConfig()``.

    Returns
    -------
    (total_reward, breakdown_dict)
        ``total_reward`` is a float.
        ``breakdown_dict`` contains per-term values for logging/debugging.

    Notes
    -----
    * BLOCKED steps receive ``r_blocked`` (hard negative) regardless of any
      other signal.  They cannot earn validator bonus.  The validator reward
      is still computed (so the policy sees it was structurally wrong too),
      but the execution term dominates.
    * ``episode_completion_bonus`` is only awarded when ``episode_complete``
      is True — not on every success.
    * ``step_count_penalty`` is always subtracted.
    """
    cfg = config or RewardConfig()

    # --- Validator reward ---------------------------------------------------
    v_reward, v_breakdown = _validator_reward(plan_so_far, instance, cfg)

    # --- Execution reward ---------------------------------------------------
    e_reward, e_breakdown = _execution_reward(execution_result, cfg)

    # --- Episode completion bonus -------------------------------------------
    completion_bonus = cfg.w_completion if episode_complete else 0.0

    # --- Step count penalty (always subtracted) -----------------------------
    step_penalty = cfg.w_step_penalty

    # --- Total ---------------------------------------------------------------
    total = v_reward + e_reward + completion_bonus - step_penalty

    breakdown = {
        "total":              total,
        "validator":          v_reward,
        "execution":          e_reward,
        "completion_bonus":   completion_bonus,
        "step_penalty":       step_penalty,
        **v_breakdown,
        **e_breakdown,
    }

    return total, breakdown
