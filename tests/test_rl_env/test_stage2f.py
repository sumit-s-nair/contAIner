"""
tests/test_rl_env/test_stage2f.py
=================================
Stage 2F (D5): validator reward restructure + progress gating + PEP 508/503
canonicalisation.

Background (B10): the validator ran on the partial plan every step and paid
+1.0 per step once any INSTALL existed, so a denied CHECK / UNMAPPED step
netted -0.1 while a failed INSTALL netted -1.1 -> spam was optimal.
"""

from __future__ import annotations

import pytest

from src.repo_scan.models import Dependency, EcosystemManifest, RepoManifest, SourceRef
from src.rl_env.env import EnvConfig, PlannerEnv
from src.rl_env.reward import (
    RewardConfig,
    canonicalize_package,
    compute_reward,
    required_packages,
)
from src.sandbox.models import AtomicStep, ExecutionResult, ExecutionStatus, RiskTier
from src.system2_planner.models import ActionType, PlannedStep, TemplateInstance
from src.system2_planner.templates import TEMPLATES

CFG = RewardConfig()


def _manifest(deps, conflicts=False) -> RepoManifest:
    m = RepoManifest()
    eco = EcosystemManifest(ecosystem="python", manifest_files=["requirements.txt"])
    for i, (name, spec) in enumerate(deps, 1):
        eco.add_dependency(Dependency(
            name=name, declared_constraint=spec,
            sources=[SourceRef(file="requirements.txt", line=i, raw_line=f"{name}{spec}")],
        ))
    m.ecosystems["python"] = eco
    if conflicts:
        m.conflicts = ["synthetic conflict"]
    return m


def _inst(deps=(("requests", ">=2.0"),), conflicts=False) -> TemplateInstance:
    return TemplateInstance(template=TEMPLATES["setup_project"], manifest=_manifest(deps, conflicts))


def _res(status=ExecutionStatus.SUCCESS, verified=True, tier=RiskTier.SAFE, rc=0) -> ExecutionResult:
    return ExecutionResult(step=AtomicStep(command="x"), status=status, tier=tier,
                           verified=verified, returncode=rc)


def _ps(t, target=None) -> PlannedStep:
    return PlannedStep(action_type=t, target=target, description=f"{t.value} {target or ''}")


def _reward(plan, history, res, inst, done=False, complete=False):
    return compute_reward(step=plan[-1], plan_so_far=plan, instance=inst, execution_result=res,
                          episode_complete=complete, config=CFG, history=history, episode_done=done)


# --- canonicalisation ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("requests>=2.0", "requests"),
    ("requests", "requests"),
    ("Requests[socks]>=2.0; python_version>'3.8'", "requests"),
    ("PyYAML", "pyyaml"),
    ("yaml", "pyyaml"),            # import-name alias
    ("ruamel.yaml==0.18", "ruamel-yaml"),
    ("Foo_Bar", "foo-bar"),
    ("@types/node", "@types/node"),  # non-PEP 508 kept
])
def test_canonicalize_package(raw, expected):
    assert canonicalize_package(raw) == expected


def test_versioned_install_satisfies_completion():
    """INSTALL 'requests>=2.0' (verified) completes a repo requiring requests."""
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._instance = _inst()

    class _Exec:
        def execute(self, step):
            return ExecutionResult(step=step, status=ExecutionStatus.SUCCESS, tier=RiskTier.SAFE,
                                   verified=True, returncode=0)
    env._executor = _Exec()
    _, reward, terminated, _, info = env.step(_ps(ActionType.INSTALL, "requests>=2.0"))
    assert info["episode_complete"] is True and terminated is True
    assert reward > CFG.w_completion


# --- validator restructure -------------------------------------------------

def test_no_per_step_validator_bonus_after_install():
    """The B10 exploit: after an INSTALL, a later step must not earn +r_valid."""
    inst = _inst()
    plan = [_ps(ActionType.INSTALL, "requests"), _ps(ActionType.CHECK, ".")]
    total, bd = _reward(plan, [_res()], _res(verified=False, rc=5), inst)
    assert bd["validator"] == 0.0
    assert bd["execution"] == 0.0          # CHECK not expected by setup_project -> no progress
    assert total == pytest.approx(-CFG.w_step_penalty)


def test_ordering_violation_charged_once():
    inst = _inst(conflicts=True)
    p1 = [_ps(ActionType.INSTALL, "requests")]
    _, bd1 = _reward(p1, [], _res(), inst)
    assert bd1["new_ordering_violations"] == ["isolate->install"]
    assert bd1["validator"] == pytest.approx(CFG.r_invalid_base + CFG.r_invalid_per_error)
    p2 = p1 + [_ps(ActionType.ISOLATE, ".venv")]   # message changes, pair is the same
    _, bd2 = _reward(p2, [_res()], _res(), inst)
    assert bd2["validator"] == 0.0
    p3 = p2 + [_ps(ActionType.INSTALL, "flask")]
    _, bd3 = _reward(p3, [_res(), _res()], _res(), inst)
    assert bd3["validator"] == 0.0


def test_presence_checked_only_at_episode_end():
    inst = _inst(conflicts=True)
    plan = [_ps(ActionType.CHECK, ".")]
    _, mid = _reward(plan, [], _res(verified=False), inst, done=False)
    assert mid["validator"] == 0.0
    _, end = _reward(plan, [], _res(verified=False), inst, done=True)
    assert set(end["missing_at_end"]) == {"isolate", "install"}
    assert end["validator"] == pytest.approx(CFG.r_invalid_base + 2 * CFG.r_invalid_per_error)


def test_valid_plan_bonus_once_at_end():
    inst = _inst()
    plan = [_ps(ActionType.INSTALL, "requests")]
    _, bd = _reward(plan, [], _res(), inst, complete=True)
    assert bd["valid_plan_bonus"] == CFG.r_valid
    _, bd_mid = _reward(plan, [], _res(), inst)
    assert bd_mid["valid_plan_bonus"] == 0.0


# --- progress gating -------------------------------------------------------

def test_duplicate_install_gets_no_execution_credit():
    inst = _inst(deps=(("requests", ""), ("flask", "")))
    plan = [_ps(ActionType.INSTALL, "requests"), _ps(ActionType.INSTALL, "requests>=2.0")]
    _, bd = _reward(plan, [_res()], _res(), inst)
    assert bd["progress"] is False and bd["execution"] == 0.0
    assert bd["execution_label"] == "success_verified_no_progress"


def test_unrelated_install_gets_no_execution_credit():
    inst = _inst()
    _, bd = _reward([_ps(ActionType.INSTALL, "numpy")], [], _res(), inst)
    assert bd["execution"] == 0.0


def test_verified_after_unverified_still_credited():
    inst = _inst()
    plan = [_ps(ActionType.INSTALL, "requests"), _ps(ActionType.INSTALL, "requests")]
    _, bd = _reward(plan, [_res(verified=False)], _res(verified=True), inst)
    assert bd["progress"] is True
    assert bd["execution"] == pytest.approx(CFG.w_execution * CFG.r_success)


def test_negative_outcomes_never_gated():
    inst = _inst()
    plan = [_ps(ActionType.INSTALL, "requests"), _ps(ActionType.INSTALL, "requests")]
    _, bd = _reward(plan, [_res()], _res(status=ExecutionStatus.FAILED, verified=False, rc=1), inst)
    assert bd["execution"] == pytest.approx(CFG.w_execution * CFG.r_failed)


def test_isolate_credit_limited_to_expected_count():
    inst = _inst(conflicts=True)
    p = [_ps(ActionType.ISOLATE, ".venv")]
    _, first = _reward(p, [], _res(), inst)
    _, second = _reward(p + [_ps(ActionType.ISOLATE, ".venv2")], [_res()], _res(), inst)
    assert first["execution"] > 0 and second["execution"] == 0.0


# --- reward ordering -------------------------------------------------------

def test_per_step_reward_ordering():
    """success(progress) > success(no progress) > denied > failed > blocked."""
    inst = _inst(deps=(("requests", ""), ("flask", "")))
    def tot(res, target="requests", hist_target=None):
        plan = ([_ps(ActionType.INSTALL, hist_target)] if hist_target else []) + [_ps(ActionType.INSTALL, target)]
        hist = [_res()] if hist_target else []
        return _reward(plan, hist, res, inst)[0]
    success = tot(_res())
    no_prog = tot(_res(), hist_target="requests")
    denied = tot(_res(status=ExecutionStatus.ABORTED, verified=False, tier=RiskTier.REVIEW, rc=None))
    failed = tot(_res(status=ExecutionStatus.FAILED, verified=False, rc=1))
    blocked = tot(_res(status=ExecutionStatus.BLOCKED, verified=False, tier=RiskTier.BLOCKED, rc=None))
    assert success > no_prog > denied > failed > blocked
    assert no_prog < 0  # pure padding is never free


def test_spam_episode_worse_than_oracle():
    """A max-length episode of CHECK spam must be negative and worse than the oracle plan."""
    inst = _inst(deps=(("requests", ""), ("flask", "")), conflicts=True)
    n = 10
    spam_plan, spam_hist, spam_total = [], [], 0.0
    for i in range(n):
        spam_plan.append(_ps(ActionType.CHECK, "."))
        r = _res(verified=False, rc=5)
        spam_total += _reward(spam_plan, spam_hist, r, inst, done=(i == n - 1))[0]
        spam_hist.append(r)
    oracle = [_ps(ActionType.ISOLATE, ".venv"), _ps(ActionType.INSTALL, "requests"), _ps(ActionType.INSTALL, "flask")]
    o_total, o_hist = 0.0, []
    for i in range(len(oracle)):
        last = i == len(oracle) - 1
        o_total += _reward(oracle[: i + 1], o_hist, _res(), inst, done=last, complete=last)[0]
        o_hist.append(_res())
    assert spam_total < 0
    assert o_total > 0 and o_total > spam_total


def test_required_packages_canonical():
    inst = _inst(deps=(("PyYAML", ">=5"), ("ruamel.yaml", "")))
    assert required_packages(inst) == {"pyyaml", "ruamel-yaml"}
