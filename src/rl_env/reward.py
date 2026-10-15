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

Validator reward (Stage 2F / D5)
--------------------------------
Per step: only ordering violations *introduced by this step* are penalised
(r_invalid_base + r_invalid_per_error * n_new).  A violation is keyed by its
(prerequisite, target) pair, so it is charged once, not on every later step.
No per-step bonus for a valid partial plan (that bonus made any action taken
after the first INSTALL worth +1, which turned denied CHECK / UNMAPPED spam
into the cheapest action).
At episode end (terminated or truncated): missing mandatory actions are
penalised once (r_invalid_base + r_invalid_per_error * n_missing); a fully
valid final plan earns r_valid once.

Progress gating (Stage 2F)
--------------------------
A positive execution reward is paid only when the step makes new progress
toward the episode goal:
  * the action type is one the template expects, and
  * INSTALL: the canonical target is a required package not already credited
    (or, when no packages are required, fewer distinct installs credited than
    the template expects);
  * other types: fewer successes of that type credited than expected.
A success with no progress gets 0 execution reward (label suffix
``_no_progress``), so repeating or padding never pays.  Negative outcomes are
never gated.

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

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Set, Tuple

from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

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
        One-time bonus at episode end when the final plan is fully valid.
    r_invalid_base:      float = -0.5
        Base penalty, applied per step that introduces new ordering
        violations and once at episode end if mandatory actions are missing.
    r_invalid_per_error: float = -0.25
        Additional penalty per new ordering violation / per missing action.
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
    r_unmapped:            float = -0.5   # unmapped action

    # --- validator rewards ---------------------------------------------------
    r_valid:             float =  1.0
    r_invalid_base:      float = -0.5
    r_invalid_per_error: float = -0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _alias_map() -> Dict[str, str]:
    from src.repo_scan.import_scan import IMPORT_TO_PACKAGE
    m = {k.lower(): v.lower() for k, v in IMPORT_TO_PACKAGE.items()}
    m["node"] = "nodejs"  # explicitly handle node/nodejs alias
    return m


def canonicalize_package(pkg: str) -> str:
    """Canonical package key: PEP 508 name (specifiers/extras/markers stripped),
    lower-cased, import-name aliases resolved, then PEP 503 normalised.

    ``requests>=2.0`` -> ``requests``; ``PyYAML`` / ``yaml`` -> ``pyyaml``;
    ``ruamel.yaml`` -> ``ruamel-yaml``.  Non-PEP 508 names (e.g. npm
    ``@scope/pkg``) are kept as-is apart from lower-casing/normalisation.
    """
    name = (pkg or "").strip()
    try:
        name = Requirement(name).name
    except InvalidRequirement:
        pass
    n = name.lower()
    n = _alias_map().get(n, n)
    return re.sub(r"[-_.]+", "-", n)


# Sandbox interpreter (contai-sandbox-python:3.11 = python:3.11-slim + git,
# linux/amd64) for PEP 508 markers.
_SANDBOX_MARKER_ENV = {
    "python_version": "3.11", "python_full_version": "3.11.0",
    "sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
    "platform_machine": "x86_64", "implementation_name": "cpython",
    "platform_python_implementation": "CPython", "extra": "",
}
_NEVER_REQUIRED = {"python"}  # interpreter constraint, not a package


def _marker_text(raw_line: str) -> Optional[str]:
    if ";" not in raw_line:
        return None
    t = raw_line.split(";", 1)[1].split("#", 1)[0].strip().rstrip(",").strip()
    if t and t[-1] in "\"'" and t.count(t[-1]) % 2 == 1:  # closing quote of a TOML string
        t = t[:-1].strip()
    return t or None


def _applies_in_sandbox(dep) -> bool:
    """False only if every source line carries a marker that is False for the
    sandbox (e.g. ``pywin32; sys_platform == 'win32'``).  Unparseable -> True."""
    raws = [getattr(sr, "raw_line", None) for sr in getattr(dep, "sources", []) or []]
    raws = [r for r in raws if r]
    if not raws:
        return True
    for raw in raws:
        mt = _marker_text(raw)
        if mt is None:
            return True
        try:
            if Marker(mt).evaluate(_SANDBOX_MARKER_ENV):
                return True
        except (InvalidMarker, Exception):
            return True
    return False


def required_packages(instance: TemplateInstance) -> Set[str]:
    """Canonical set of packages the episode must install (declared + inferred),
    excluding deps whose PEP 508 marker excludes the sandbox and 'python'."""
    req: Set[str] = set()
    for eco in instance.manifest.ecosystems.values():
        for declared_dep, dep in eco.dependencies.items():
            if _applies_in_sandbox(dep):
                req.add(canonicalize_package(declared_dep))
        for d in eco.inferred_dependencies:
            pkg = d.guessed_package_name or d.import_name
            if pkg:
                req.add(canonicalize_package(pkg))
    return req - _NEVER_REQUIRED


# ---------------------------------------------------------------------------
# Stage 2G-fix-3: version-aware ground truth
# ---------------------------------------------------------------------------
# A required package counts as installed only if its installed version
# satisfies a constraint the repo declared for it.  Before this, success was
# name-only: new92/instatools has mutually unsatisfiable pins, yet the oracle
# "won" by installing every package unpinned.  A policy could learn the same
# shortcut (drop the pins, install names one by one).
#
# Semantics:
#   * declared in several manifests -> satisfying ANY of them counts (that is
#     what a conflict repo asks the agent to choose between);
#   * no constraint ("*", "", URL / VCS refs, anything unparseable) -> name only;
#   * a constraint exists but the installed version is unknown/invalid -> NOT
#     satisfied (strict: unknown never earns credit).

# Version of the "winnable" criterion recorded in pilot_winnability.json.
# Bump whenever ground truth / completion semantics change: verdicts from an
# older criterion are then stale everywhere (oracle --resume/--only-lost AND
# the pilot corpus filter).
#   1: name-only (verdicts without a "criterion" field)
#   2: version-aware satisfied_required + pip check gate
WINNABILITY_CRITERION_VERSION = 2

_POETRY_BARE = re.compile(r"^\d+(\.\d+)*(\.\*)?$")


def _bump(nums: List[int], idx: int) -> str:
    """Upper bound: increment component idx, zero the rest (1.2.3, idx 0 -> 2.0.0)."""
    return ".".join(map(str, nums[:idx] + [nums[idx] + 1] + [0] * (len(nums) - idx - 1)))


def _poetry_to_pep440(c: str) -> str:
    """Poetry ^ / ~ / bare-version syntax -> PEP 440 (each comma part)."""
    out = []
    for part in [x.strip() for x in c.split(",") if x.strip()]:
        if part.startswith("^"):
            v = part[1:].strip()
            nums = [int(x) for x in v.split(".")]
            # first non-zero component is bumped (^0.2.3 -> <0.3.0; ^0.0.3 -> <0.0.4)
            idx = next((i for i, n in enumerate(nums) if n != 0), len(nums) - 1)
            out += [f">={v}", f"<{_bump(nums, idx)}"]
        elif part.startswith("~") and not part.startswith("~="):
            v = part[1:].strip()
            nums = [int(x) for x in v.split(".")]
            idx = 0 if len(nums) == 1 else 1
            out += [f">={v}", f"<{_bump(nums, idx)}"]
        elif _POETRY_BARE.match(part):
            out.append(f"=={part}")
        else:
            out.append(part)
    return ",".join(out)


def to_specifier(constraint: Optional[str]) -> Optional[SpecifierSet]:
    """Declared constraint -> SpecifierSet; None means 'any version'."""
    if constraint is None:
        return None
    c = constraint.strip()
    c = c.split(" --", 1)[0].strip()          # pip options after the spec (--hash ...)
    if c.startswith("[") and "]" in c:        # extras left on the constraint
        c = c.split("]", 1)[1].strip()
    if c in ("", "*") or c.startswith("@") or "://" in c:
        return None
    try:
        return SpecifierSet(_poetry_to_pep440(c), prereleases=True)
    except (InvalidSpecifier, ValueError):
        return None


def required_specifiers(instance: TemplateInstance) -> Dict[str, List[Optional[SpecifierSet]]]:
    """Canonical required package -> list of acceptable specifiers (one per
    declaration; None = any version).  Keys == required_packages(instance)."""
    out: Dict[str, List[Optional[SpecifierSet]]] = {}
    for eco in instance.manifest.ecosystems.values():
        for declared_dep, dep in eco.dependencies.items():
            if not _applies_in_sandbox(dep):
                continue
            k = canonicalize_package(declared_dep)
            cons = getattr(dep, "constraints", None) or [dep.declared_constraint]
            out.setdefault(k, []).extend(to_specifier(c) for c in cons)
        for d in eco.inferred_dependencies:
            pkg = d.guessed_package_name or d.import_name
            if pkg:
                out.setdefault(canonicalize_package(pkg), []).append(None)
    for k in _NEVER_REQUIRED:
        out.pop(k, None)
    return out


def version_satisfies(version: Optional[str], specs: List[Optional[SpecifierSet]]) -> bool:
    if not specs or any(sp is None for sp in specs):
        return True
    if version is None:
        return False
    try:
        v = Version(version)
    except InvalidVersion:
        return False
    return any(sp.contains(v, prereleases=True) for sp in specs)


def satisfied_required(instance: TemplateInstance, installed: Optional[Mapping[str, Optional[str]]]) -> Set[str]:
    """Required packages that are installed at a version satisfying one of the
    repo's declared constraints.  ``installed``: canonical name -> version
    (a plain set is accepted and treated as 'version unknown')."""
    if installed is None:
        return set()
    if not isinstance(installed, Mapping):
        installed = {n: None for n in installed}
    return {
        pkg for pkg, specs in required_specifiers(instance).items()
        if pkg in installed and version_satisfies(installed[pkg], specs)
    }


def pinned_target(pkg: str, specs: List[Optional[SpecifierSet]]) -> str:
    """INSTALL target that respects the declared constraint (first non-empty)."""
    if any(sp is None for sp in specs):
        return pkg
    sp = next((sp for sp in specs if sp is not None and str(sp)), None)
    return f"{pkg}{sp}" if sp is not None else pkg


def is_progress(
    step: PlannedStep,
    plan_so_far: List[PlannedStep],
    history: Optional[List[ExecutionResult]],
    instance: TemplateInstance,
    exec_result: ExecutionResult,
) -> bool:
    """True if this (successful) step makes new progress toward completion.

    ``history`` holds the ExecutionResults of ``plan_so_far[:-1]`` (previous
    steps).  ``None`` means no previous step has been credited.
    """
    expected = Counter(a.action_type for a in instance.get_expected_actions())
    if step.action_type not in expected:
        return False
    verified_now = bool(exec_result.verified)
    prior = list(zip(plan_so_far[:-1], history or []))

    def _counts(e: ExecutionResult) -> bool:
        # A prior success "uses up" credit if it is at least as good as now.
        return e.status == ExecutionStatus.SUCCESS and (e.verified or not verified_now)

    if step.action_type == ActionType.INSTALL:
        req = required_packages(instance)
        if req:
            if not step.target:
                return False
            tgt = canonicalize_package(step.target)
            if tgt not in req:
                return False
            return not any(
                p.action_type == ActionType.INSTALL and p.target
                and canonicalize_package(p.target) == tgt and _counts(e)
                for p, e in prior
            )

    n_prior = sum(1 for p, e in prior if p.action_type == step.action_type and _counts(e))
    return n_prior < expected[step.action_type]


def _ordering_pairs(instance: TemplateInstance, plan: List[PlannedStep]) -> Set[Tuple[ActionType, ActionType]]:
    if not plan:
        return set()
    return set(validate_plan(instance, plan).ordering_violations)


def _validator_reward(
    plan_so_far: List[PlannedStep],
    instance: TemplateInstance,
    cfg: RewardConfig,
    episode_done: bool,
) -> Tuple[float, dict]:
    """Per-step new-ordering-violation penalty + one-time end-of-episode term."""
    new_pairs = _ordering_pairs(instance, plan_so_far) - _ordering_pairs(instance, plan_so_far[:-1])
    raw = 0.0
    if new_pairs:
        raw += cfg.r_invalid_base + cfg.r_invalid_per_error * len(new_pairs)

    result = validate_plan(instance, plan_so_far)
    missing_at_end: List[str] = []
    end_bonus = 0.0
    if episode_done:
        if result.missing:
            missing_at_end = [a.value for a in result.missing]
            raw += cfg.r_invalid_base + cfg.r_invalid_per_error * len(result.missing)
        elif result.is_valid:
            end_bonus = cfg.r_valid
            raw += end_bonus

    breakdown = {
        "validator_raw": raw,
        "valid": result.is_valid,
        "errors": result.errors,
        "new_ordering_violations": sorted(f"{a.value}->{b.value}" for a, b in new_pairs),
        "missing_at_end": missing_at_end,
        "valid_plan_bonus": end_bonus,
    }
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
    elif status == ExecutionStatus.UNMAPPED:
        raw = cfg.r_unmapped
        label = "unmapped"
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
    history:           Optional[List[ExecutionResult]] = None,
    episode_done:      bool = False,
    install_progress:  Optional[float] = None,
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
    history:
        ExecutionResults of the previous steps (aligned with
        ``plan_so_far[:-1]``), used for progress gating.  ``None`` = none.
    episode_done:
        True on the final step of the episode (terminated or truncated);
        triggers the one-time presence check / valid-plan bonus.  Implied by
        ``episode_complete``.
    install_progress:
        Stage 2G ground truth for INSTALL steps: fraction of the required
        packages newly installed by this step (high-water mark, measured in
        the container).  When given, it replaces target matching: progress
        is ``install_progress > 0`` and a positive execution reward is scaled
        by it, so the progress reward summed over an episode is the same
        whether packages go in one by one or via ``-r requirements.txt``
        (fewer steps then win through the step penalty).  ``None`` = no
        ground truth (dry run) -> target-based ``is_progress``.

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
    episode_done = episode_done or episode_complete
    v_reward, v_breakdown = _validator_reward(plan_so_far, instance, cfg, episode_done)

    # --- Execution reward ---------------------------------------------------
    e_reward, e_breakdown = _execution_reward(execution_result, cfg)
    if step.action_type == ActionType.INSTALL and install_progress is not None:
        progress = install_progress > 0
        if e_reward > 0 and progress:
            e_reward *= install_progress
            e_breakdown["execution_raw"] *= install_progress
    else:
        progress = is_progress(step, plan_so_far, history, instance, execution_result)
    if e_reward > 0 and not progress:
        e_reward = 0.0
        e_breakdown["execution_raw"] = 0.0
        e_breakdown["execution_label"] += "_no_progress"
    e_breakdown["progress"] = progress
    e_breakdown["install_progress"] = install_progress

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
