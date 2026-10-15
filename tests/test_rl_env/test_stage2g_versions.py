"""Stage 2G-fix-3: version-aware ground truth.

Before: a required package counted as installed by NAME.  new92/instatools has
mutually unsatisfiable pins (ResolutionImpossible), yet the oracle "won" by
installing every package unpinned -- a shortcut a policy could learn too.
Now a required package counts only at a version satisfying one of the repo's
declared constraints (any declaring manifest).  These tests prove the
rejections, not just the acceptances.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from src.repo_scan.models import Dependency, EcosystemManifest, RepoManifest, SourceRef  # noqa: E402
from src.repo_scan.scanner import scan_repo  # noqa: E402
from src.rl_env import docker_executor as de  # noqa: E402
from src.rl_env.env import EnvConfig, PlannerEnv  # noqa: E402
from src.rl_env.reward import (  # noqa: E402
    canonicalize_package, pinned_target, required_specifiers, satisfied_required, to_specifier, version_satisfies)
from src.sandbox.models import ExecutionResult, ExecutionStatus, RiskTier  # noqa: E402
from src.system2_planner.models import ActionType, PlannedStep, TemplateInstance  # noqa: E402
from src.system2_planner.templates import TEMPLATES  # noqa: E402

import stage2g_winnability as w  # noqa: E402


def _inst(deps, conflicts=False):
    """deps: list of (name, constraint) -- one declaration each (repeat a name for two manifests)."""
    m = RepoManifest()
    eco = EcosystemManifest(ecosystem="python", manifest_files=["/repo/requirements.txt"])
    for i, (name, c) in enumerate(deps, 1):
        eco.add_dependency(Dependency(name=name, declared_constraint=c,
                                      sources=[SourceRef(file="requirements.txt", line=i, raw_line=f"{name}{c}")]))
    m.ecosystems["python"] = eco
    if conflicts:
        m.conflicts = ["synthetic"]
    return TemplateInstance(template=TEMPLATES["setup_project"], manifest=m)


def _ps(t, target=""):
    return PlannedStep(action_type=t, target=target, description=f"{t.value} {target}")


# ------------------------------------------------------------- constraint parsing

@pytest.mark.parametrize("c,spec", [
    ("==2.33.0", "==2.33.0"), (">=1,<2", "<2,>=1"), ("~=1.4", "~=1.4"),
    ("^1.2.3", "<2.0.0,>=1.2.3"), ("^0.2.3", "<0.3.0,>=0.2.3"), ("^0.0.3", "<0.0.4,>=0.0.3"),
    ("~1.2.3", "<1.3.0,>=1.2.3"), ("1.2.3", "==1.2.3"), ("1.2.*", "==1.2.*"),
    (">=2.0 --hash=sha256:ab", ">=2.0"),
])
def test_to_specifier(c, spec):
    assert str(to_specifier(c)) == spec


@pytest.mark.parametrize("c", ["*", "", "@ git+https://github.com/a/b", "foo bar", None])
def test_to_specifier_any_version(c):
    assert to_specifier(c) is None


def test_version_satisfies_rejections():
    assert version_satisfies("2.31.0", [to_specifier("==2.33.0")]) is False   # wrong version
    assert version_satisfies(None, [to_specifier(">=1")]) is False             # unknown version, pinned
    assert version_satisfies("not-a-version", [to_specifier(">=1")]) is False
    assert version_satisfies(None, [None]) is True                             # unpinned: name only
    assert version_satisfies("3.0rc1", [to_specifier(">=2")]) is True          # prereleases allowed


# ------------------------------------------------------------- satisfied_required

def test_pinned_requirement_needs_matching_version():
    inst = _inst([("requests", "==2.33.0"), ("rich", "*")])
    assert satisfied_required(inst, {"requests": "2.31.0", "rich": "13.0"}) == {"rich"}
    assert satisfied_required(inst, {"requests": "2.33.0", "rich": "13.0"}) == {"requests", "rich"}


def test_plain_set_means_unknown_version():
    inst = _inst([("requests", "==2.33.0"), ("rich", "*")])
    assert satisfied_required(inst, {"requests", "rich"}) == {"rich"}
    assert satisfied_required(inst, None) == set()


def test_conflicting_manifests_either_constraint_counts():
    inst = _inst([("requests", "==2.31.0"), ("requests", ">=2.32")], conflicts=True)
    assert [str(s) for s in required_specifiers(inst)["requests"]] == ["==2.31.0", ">=2.32"]
    assert satisfied_required(inst, {"requests": "2.31.0"}) == {"requests"}
    assert satisfied_required(inst, {"requests": "2.33.0"}) == {"requests"}
    assert satisfied_required(inst, {"requests": "2.30.0"}) == set()          # satisfies neither


def test_scan_keeps_constraint_from_every_manifest(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests==2.31.0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["requests>=2.32"]\n', encoding="utf-8")
    m = scan_repo(str(tmp_path))
    dep = m.ecosystems["python"].dependencies["requests"]
    assert dep.declared_constraint == "==2.31.0"                 # unchanged (first declaration)
    assert dep.constraints == ["==2.31.0", ">=2.32"]


def test_pinned_target():
    assert pinned_target("requests", [to_specifier("==2.33.0")]) == "requests==2.33.0"
    assert pinned_target("requests", [to_specifier("==2.33.0"), None]) == "requests"
    assert pinned_target("rich", [None]) == "rich"


# ------------------------------------------------------------- env, fake container with versions

class _VerExecutor:
    """Fake container with versions.  `pip install -r` installs `manifest_versions`
    (what pip would resolve); `pip install name[spec]` installs `latest` unless a
    spec is given, in which case the spec's == version (or latest if a range)."""

    def __init__(self, manifest_versions, latest, base=None):
        self.manifest_versions, self.latest = manifest_versions, latest
        self.env = dict(base or {"pip": "24.0"})

    def installed_versions(self):
        return dict(self.env)

    def execute(self, step):
        cmd = step.command
        if cmd.startswith("python -m pip install -r "):
            self.env.update(self.manifest_versions)
        elif cmd.startswith("python -m pip install "):
            for tok in cmd[len("python -m pip install "):].split():
                tok = tok.strip("'")
                name = canonicalize_package(tok)
                ver = tok.split("==", 1)[1] if "==" in tok else self.latest[name]
                self.env[name] = ver
        return ExecutionResult(step=step, status=ExecutionStatus.SUCCESS, tier=RiskTier.SAFE,
                               verified=True, returncode=0)


def _env(inst, ex):
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._instance = inst
    env._executor = ex
    env._installed_now = env._query_installed()
    env._credited_required = satisfied_required(env._instance, env._installed_now)
    return env


def test_unpinned_installs_do_not_complete_a_pinned_repo():
    # the instatools shortcut: pins ignored, latest versions installed by name
    inst = _inst([("requests", "==2.33.0"), ("rich", ">=13,<14")])
    env = _env(inst, _VerExecutor({}, latest={"requests": "2.34.1", "rich": "14.2.0"}))
    for t in ("requests", "rich"):
        _, _, term, _, info = env.step(_ps(ActionType.INSTALL, t))
        assert info["install_progress"] == 0.0
        assert info["reward_breakdown"]["execution"] <= 0
    assert term is False and info["episode_complete"] is False
    assert info["required_installed"] == 0


def test_pinned_installs_complete():
    inst = _inst([("requests", "==2.33.0"), ("rich", "*")])
    env = _env(inst, _VerExecutor({}, latest={"requests": "2.34.1", "rich": "14.2.0"}))
    _, _, _, _, a = env.step(_ps(ActionType.INSTALL, "requests==2.33.0"))
    _, _, term, _, b = env.step(_ps(ActionType.INSTALL, "rich"))
    assert a["install_progress"] == pytest.approx(0.5) and b["install_progress"] == pytest.approx(0.5)
    assert term is True and b["episode_complete"] is True


def test_manifest_install_at_declared_versions_completes_in_one_step():
    inst = _inst([("requests", "==2.33.0"), ("rich", ">=13")])
    env = _env(inst, _VerExecutor({"requests": "2.33.0", "rich": "13.9.4"}, latest={}))
    _, _, term, _, info = env.step(_ps(ActionType.INSTALL, "requirements.txt"))
    assert info["install_progress"] == pytest.approx(1.0) and term is True


def test_base_image_package_at_wrong_version_not_pre_credited():
    inst = _inst([("setuptools", "<60"), ("rich", "*")])
    env = _env(inst, _VerExecutor({}, latest={"rich": "14.0", "setuptools": "80.0"},
                                  base={"pip": "24.0", "setuptools": "79.0"}))
    assert env._credited_required == set()                      # 79.0 does not satisfy <60
    _, _, term, _, info = env.step(_ps(ActionType.INSTALL, "rich"))
    assert term is False and info["required_installed"] == 1


def test_query_installed_accepts_set_only_executor():
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env._executor = SimpleNamespace(installed_packages=lambda: {"rich"})
    assert env._query_installed() == {"rich": None}


# ------------------------------------------------------------- docker executor + oracle tail

def test_docker_installed_versions_parses_pip_list():
    ex = de.DockerEpisodeExecutor.__new__(de.DockerEpisodeExecutor)
    ex._container_id = "c"
    out = json.dumps([{"name": "Requests", "version": "2.33.0"}, {"name": "ruamel.yaml", "version": "0.18"}])
    ex._run_docker_exec = lambda cmd, timeout=60: SimpleNamespace(returncode=0, stdout=out, stderr="")
    assert ex.installed_versions() == {"requests": "2.33.0", "ruamel-yaml": "0.18"}
    assert ex.installed_packages() == {"requests", "ruamel-yaml"}


def test_tail_reinstalls_wrong_version_with_declared_pin():
    inst = _inst([("requests", "==2.33.0"), ("rich", "*"), ("tabulate", "*")])
    env = SimpleNamespace(_instance=inst, _installed_now={"requests": "2.31.0", "rich": "13.0"})
    assert [s.target for s in w._tail_steps(env)] == ["requests==2.33.0", "tabulate"]


# ------------------------------------------------------------- pip check gate

def test_all_pins_met_but_inconsistent_set_is_not_complete():
    # instatools-style: each pin installed separately, pip warns but installs;
    # the set is broken, so the episode must not complete.
    inst = _inst([("requests", "==2.33.0"), ("rich", "*")])
    ex = _VerExecutor({}, latest={"requests": "2.34.1", "rich": "14.2.0"})
    ex.dependency_check = lambda: (False, "instagrapi 2.1 has requirement requests==2.32.3, but you have requests 2.33.0.")
    env = _env(inst, ex)
    env.step(_ps(ActionType.INSTALL, "requests==2.33.0"))
    _, _, term, _, info = env.step(_ps(ActionType.INSTALL, "rich"))
    assert info["required_installed"] == 2
    assert info["episode_complete"] is False and term is False
    assert info["dependency_check_ok"] is False and "requests==2.32.3" in info["dependency_check_output"]


def test_consistent_set_completes_and_reports_check():
    inst = _inst([("rich", "*")])
    ex = _VerExecutor({}, latest={"rich": "14.2.0"})
    ex.dependency_check = lambda: (True, "No broken requirements found.")
    env = _env(inst, ex)
    _, _, term, _, info = env.step(_ps(ActionType.INSTALL, "rich"))
    assert term is True and info["dependency_check_ok"] is True


def test_docker_dependency_check():
    ex = de.DockerEpisodeExecutor.__new__(de.DockerEpisodeExecutor)
    ex._container_id = "c"
    ex._run_docker_exec = lambda cmd, timeout=60: SimpleNamespace(returncode=1, stdout="a 1 has requirement b<2, but you have b 3.\n", stderr="")
    ok, out = ex.dependency_check()
    assert ok is False and "has requirement" in out
    ex._container_id = None
    assert ex.dependency_check() is None
