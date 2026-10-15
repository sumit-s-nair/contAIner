"""
tests/test_rl_env/test_stage2g.py
=================================
Stage 2G (winnability): manifest-level INSTALL targets, ground-truth
progress/completion from the container's installed packages, PEP 508 marker
filtering of required packages, and the observation changes the planner needs
(manifest files, per-step outcomes, restricted action list).
"""

from __future__ import annotations

import pytest

from src.repo_scan.models import Dependency, EcosystemManifest, RepoManifest, SourceRef
from src.rl_env.env import EnvConfig, PlannerEnv, _rel_manifest_paths
from src.rl_env.observation import ObservationSerializer
from src.rl_env.reward import RewardConfig, canonicalize_package, compute_reward, required_packages, satisfied_required
from src.rl_env.system3_stub import MANIFEST_INSTALL_TIMEOUT, expand_planned_step
from src.sandbox.models import AtomicStep, ExecutionResult, ExecutionStatus, RiskTier
from src.system2_planner.models import ActionType, PlannedStep, TemplateInstance
from src.system2_planner.templates import TEMPLATES

CFG = RewardConfig()


def _ps(t, target="", desc=None):
    return PlannedStep(action_type=t, target=target, description=desc or f"{t.value} {target}")


def _manifest(lines, conflicts=False):
    m = RepoManifest()
    eco = EcosystemManifest(ecosystem="python", manifest_files=["/repo/requirements.txt"])
    for i, (name, raw) in enumerate(lines, 1):
        eco.add_dependency(Dependency(name=name, declared_constraint="*",
                                      sources=[SourceRef(file="requirements.txt", line=i, raw_line=raw)]))
    m.ecosystems["python"] = eco
    if conflicts:
        m.conflicts = ["synthetic"]
    return m


def _inst(lines, conflicts=False):
    return TemplateInstance(template=TEMPLATES["setup_project"], manifest=_manifest(lines, conflicts))


# --- System 3 stub -----------------------------------------------------------

@pytest.mark.parametrize("target,cmd", [
    ("requirements.txt", "python -m pip install -r requirements.txt"),
    ("requirements/dev.txt", "python -m pip install -r requirements/dev.txt"),
    (".", "python -m pip install ."),
    ("pyproject.toml", "python -m pip install ."),
    ("sub/setup.py", "python -m pip install sub"),
    ("requests>=2 flask", "python -m pip install 'requests>=2' flask"),
    ("requests, flask", "python -m pip install requests flask"),
])
def test_manifest_and_multi_install_forms(target, cmd):
    (step,) = expand_planned_step(_ps(ActionType.INSTALL, target))
    assert step.command == cmd
    assert step.verify_command is None            # judged by ground truth
    assert step.timeout == MANIFEST_INSTALL_TIMEOUT


def test_single_spec_install_unchanged():
    (step,) = expand_planned_step(_ps(ActionType.INSTALL, "requests>=2.0"))
    assert step.command == "python -m pip install 'requests>=2.0'"
    assert step.verify_command == "python -m pip show requests"


@pytest.mark.parametrize("target", ["../x.txt", "/abs/requirements.txt", "-r requirements.txt"])
def test_unsafe_paths_are_not_manifest_installs(target):
    (step,) = expand_planned_step(_ps(ActionType.INSTALL, target))
    assert " -r " not in step.command.replace("'-r ", "")
    assert step.verify_command == "false"


# --- required packages ---------------------------------------------------------

def test_markers_exclude_non_linux_deps_and_python():
    inst = _inst([
        ("requests", "requests>=2"),
        ("pywin32", 'pywin32>=300; sys_platform == "win32"'),
        ("uvloop", "uvloop; sys_platform != 'win32'"),
        ("tomli", 'tomli; python_version < "3.11"'),
        ("python", "python"),
    ])
    assert required_packages(inst) == {"requests", "uvloop"}


def test_toml_style_marker_line():
    inst = _inst([("pywin32", '"pywin32; sys_platform == \'win32\'",')])
    assert required_packages(inst) == set()


# --- reward scaling ------------------------------------------------------------

def _res(status=ExecutionStatus.SUCCESS, verified=True):
    return ExecutionResult(step=AtomicStep(command="x"), status=status, tier=RiskTier.SAFE,
                           verified=verified, returncode=0)


def test_install_progress_scales_execution_reward():
    inst = _inst([("a", "a"), ("b", "b"), ("c", "c"), ("d", "d")])
    plan = [_ps(ActionType.INSTALL, "requirements.txt")]
    _, half = compute_reward(step=plan[-1], plan_so_far=plan, instance=inst, execution_result=_res(),
                             episode_complete=False, install_progress=0.5)
    assert half["execution"] == pytest.approx(CFG.w_execution * CFG.r_success * 0.5)
    _, zero = compute_reward(step=plan[-1], plan_so_far=plan, instance=inst, execution_result=_res(),
                             episode_complete=False, install_progress=0.0)
    # Stage 2H: zero-progress success now costs r_no_progress (was 0.0)
    assert zero["execution"] == pytest.approx(CFG.w_execution * CFG.r_no_progress) and zero["progress"] is False


# --- env with ground truth -----------------------------------------------------

class _GTExecutor:
    """Fake container: tracks installed packages per interpreter (base vs .venv)."""

    def __init__(self, manifest_pkgs, fail_targets=()):
        self.manifest_pkgs = manifest_pkgs
        self.fail_targets = set(fail_targets)
        self.base = {"pip", "setuptools"}
        self.venv = None

    def _env(self):
        return self.venv if self.venv is not None else self.base

    def installed_packages(self):
        return {canonicalize_package(p) for p in self._env()}

    def execute(self, step):
        cmd = step.command
        ok = True
        if cmd.startswith("python -m venv"):
            self.venv = {"pip"}
        elif cmd.startswith("python -m pip install -r "):
            self._env().update(self.manifest_pkgs)
        elif cmd.startswith("python -m pip install "):
            pkgs = [t.strip("'") for t in cmd[len("python -m pip install "):].split()]
            if any(p in self.fail_targets for p in pkgs):
                ok = False
            else:
                self._env().update(canonicalize_package(p) for p in pkgs)
        return ExecutionResult(step=step, status=ExecutionStatus.SUCCESS if ok else ExecutionStatus.FAILED,
                               tier=RiskTier.SAFE, verified=ok, returncode=0 if ok else 1)


def _gt_env(lines, conflicts=False, **kw):
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._instance = _inst(lines, conflicts)
    env._executor = _GTExecutor([n for n, _ in lines], **kw)
    env._installed_now = env._query_installed()
    env._credited_required = satisfied_required(env._instance, env._installed_now)  # mirrors reset()
    return env


LINES = [("requests", "requests"), ("flask", "flask"), ("numpy", "numpy")]


def test_requirements_install_completes_in_one_step():
    env = _gt_env(LINES)
    _, reward, term, _, info = env.step(_ps(ActionType.INSTALL, "requirements.txt"))
    assert info["install_progress"] == pytest.approx(1.0)
    assert info["episode_complete"] is True and term is True
    assert info["required_installed"] == info["required_total"] == 3
    assert reward == pytest.approx(CFG.w_execution * CFG.r_success + CFG.w_completion + CFG.r_valid - CFG.w_step_penalty)


def test_progress_sum_is_split_invariant():
    env = _gt_env(LINES)
    rewards = []
    for t in ("requests", "flask", "numpy"):
        _, r, term, _, info = env.step(_ps(ActionType.INSTALL, t))
        rewards.append(info["reward_breakdown"]["execution"])
    assert term is True
    assert sum(rewards) == pytest.approx(CFG.w_execution * CFG.r_success)


def test_reinstall_after_isolate_not_recredited():
    env = _gt_env(LINES, conflicts=False)
    env.step(_ps(ActionType.INSTALL, "requests flask"))       # 2/3 credited
    env.step(_ps(ActionType.ISOLATE, ".venv"))                # fresh venv: state drops
    _, _, _, _, info = env.step(_ps(ActionType.INSTALL, "requirements.txt"))
    assert info["install_progress"] == pytest.approx(1 / 3)   # only numpy is new credit
    assert info["episode_complete"] is True                   # completion = current state


def test_failed_install_gets_no_credit():
    env = _gt_env(LINES, fail_targets={"numpy"})
    _, _, _, _, info = env.step(_ps(ActionType.INSTALL, "numpy"))
    assert info["install_progress"] == 0.0
    assert info["reward_breakdown"]["execution_label"] == "failed"


def test_base_image_packages_pre_credited():
    env = _gt_env([("setuptools", "setuptools"), ("requests", "requests")])
    _, _, term, _, info = env.step(_ps(ActionType.INSTALL, "requests"))
    assert info["install_progress"] == pytest.approx(0.5)     # setuptools was already there
    assert term is True


def test_completion_requires_current_state_not_targets():
    env = _gt_env(LINES)
    _, _, term, _, info = env.step(_ps(ActionType.INSTALL, "requests flask"))
    assert term is False and info["episode_complete"] is False


# --- observation ---------------------------------------------------------------

def test_rel_manifest_paths(tmp_path):
    (tmp_path / "requirements").mkdir()
    m = RepoManifest()
    m.ecosystems["python"] = EcosystemManifest(
        ecosystem="python",
        manifest_files=[str(tmp_path / "requirements.txt"), str(tmp_path / "requirements" / "dev.txt")])
    assert _rel_manifest_paths(m, tmp_path) == {"python": ["requirements.txt", "requirements/dev.txt"]}


def test_prompt_shows_files_outcomes_and_restricted_actions():
    env = _gt_env(LINES)
    env._obs.manifest_paths = {"python": ["requirements.txt"]}
    env.step(_ps(ActionType.INSTALL, "requests", desc="Install requests"))
    msgs = ObservationSerializer().to_chat_prompt(env._obs)
    sys_msg, user = msgs[0]["content"], msgs[1]["content"]
    assert "Must be one of: ISOLATE, INSTALL, CHECK." in sys_msg
    assert "files: requirements.txt" in user
    assert "Install requests [success, verified]" in user
