"""Stage 2G-fix-2: root causes from the second winnability run (2026-10-15, 2/6 won).

* VectifyAI/PageIndex (13/15) and soxoj/maigret (26/29): the Poetry parser
  counted ``optional = true`` deps (extras) as required; ``pip install .``
  never installs them, so the episode could not complete.
* dask/dask: ``pip install .`` failed in setuptools-scm ("unable to detect
  version") because python:3.11-slim has no git -> local sandbox image with git.
* new92/instatools: requirements.txt pins are mutually unsatisfiable
  (ResolutionImpossible) -> recorded by the oracle run, excluded by the pilot
  builder as oracle_unwinnable.
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

import build_pilot_corpus as b  # noqa: E402
import stage2g_winnability as w  # noqa: E402
from src.repo_scan.parsers import _poetry_is_optional, parse_pyproject_toml  # noqa: E402
from src.repo_scan.scanner import scan_repo  # noqa: E402
from src.rl_env import docker_executor as de  # noqa: E402
from src.rl_env.reward import required_packages  # noqa: E402
from src.system2_planner.models import ActionType, TemplateInstance  # noqa: E402
from src.system2_planner.templates import TEMPLATES  # noqa: E402

# Trimmed from VectifyAI/PageIndex@b058e036 pyproject.toml
PAGEINDEX = '''[tool.poetry]
name = "pageindex"
version = "0.2.10"

[tool.poetry.dependencies]
python = ">=3.10"
requests = ">=2.28.0"
openai = ">=1.70.0"
pyyaml = ">=6.0"
claude-agent-sdk = { version = ">=0.1.53", optional = true }
anthropic = { version = ">=0.122.0", optional = true }

[tool.poetry.extras]
claude = ["claude-agent-sdk"]
anthropic = ["anthropic"]

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0"
'''


# --------------------------------------------------------------- Poetry optional

@pytest.mark.parametrize("val,expected", [
    ({"version": ">=1", "optional": True}, True),
    ({"version": ">=1", "optional": False}, False),
    ({"version": ">=1"}, False),
    ({"extras": ["async"], "version": "^3.1.1"}, False),
    (">=1", False),
    ([{"version": "<2", "python": "<3.8", "optional": True},
      {"version": ">=2", "python": ">=3.8", "optional": True}], True),
    ([{"version": "<2", "python": "<3.8", "optional": True},
      {"version": ">=2", "python": ">=3.8"}], False),
    ([], False),
])
def test_poetry_is_optional(val, expected):
    assert _poetry_is_optional(val) is expected


def test_parse_pyproject_skips_poetry_optional(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text(PAGEINDEX, encoding="utf-8")
    deps, rp = parse_pyproject_toml(str(p))
    names = {d.name.lower() for d in deps}
    assert names == {"requests", "openai", "pyyaml"}
    assert "anthropic" not in names and "claude-agent-sdk" not in names
    assert rp == ">=3.10"


def test_required_packages_excludes_poetry_extras(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PAGEINDEX, encoding="utf-8")
    m = scan_repo(str(tmp_path))
    req = required_packages(TemplateInstance(template=TEMPLATES["setup_project"], manifest=m))
    assert {"requests", "openai", "pyyaml"} <= req
    assert not req & {"anthropic", "claude-agent-sdk", "pytest"}


def test_non_optional_dict_dep_still_required(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\npython = "^3.10"\n'
        'flask = {extras = ["async"], version = "^3.1.1"}\n'
        'xhtml2pdf = {version = "^0.2.11", optional = true}\n', encoding="utf-8")
    deps, _ = parse_pyproject_toml(str(tmp_path / "pyproject.toml"))
    assert [d.name.lower() for d in deps] == ["flask"]


# --------------------------------------------------------------- sandbox image with git

def test_python_ecosystem_uses_local_git_image():
    assert de.select_image(["python"]) == de.PYTHON_SANDBOX_IMAGE == "contai-sandbox-python:3.11"
    assert de.select_image(None) == de.PYTHON_SANDBOX_IMAGE


def test_dockerfile_installs_git_and_safe_directory():
    df = (ROOT / "docker" / "sandbox_python" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in df
    assert "install -y --no-install-recommends git" in df
    assert "safe.directory '*'" in df


def test_dockerfile_installs_c_toolchain_but_no_repo_specific_dev_libs():
    # 4/6 oracle losses in the --all run were "No such file or directory: 'gcc'/'g++'"
    # or "pkg-config is required" -> generic toolchain in the image.
    df = (ROOT / "docker" / "sandbox_python" / "Dockerfile").read_text(encoding="utf-8")
    install = next(ln for ln in df.splitlines() if "apt-get install" in ln)
    assert "build-essential" in install and "pkg-config" in install
    assert "-dev" not in install          # no per-repo libraries (libcairo2-dev, libav*-dev ...)
    assert "python-3.11-git-toolchain-v2" in df


def _executor(policy="missing"):
    ex = de.DockerEpisodeExecutor.__new__(de.DockerEpisodeExecutor)
    ex._config = SimpleNamespace(pull_policy=policy)
    return ex


def test_local_image_missing_raises_with_build_hint_and_never_pulls():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, b"", b"No such image")

    with patch.object(de.subprocess, "run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="build_sandbox_image.py"):
            _executor("always")._pull_image_if_needed(de.PYTHON_SANDBOX_IMAGE)
    assert calls == [["docker", "image", "inspect", de.PYTHON_SANDBOX_IMAGE]]
    assert not any("pull" in c for c in calls)


def test_local_image_present_no_pull():
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, b"[]", b""))
    with patch.object(de.subprocess, "run", run):
        _executor("always")._pull_image_if_needed(de.PYTHON_SANDBOX_IMAGE)
    assert run.call_count == 1 and run.call_args[0][0][:3] == ["docker", "image", "inspect"]


def test_registry_image_still_pulled_when_missing():
    results = iter([subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)])
    run = MagicMock(side_effect=lambda *a, **k: next(results))
    with patch.object(de.subprocess, "run", run):
        _executor("missing")._pull_image_if_needed("node:20-slim")
    assert run.call_args_list[1][0][0] == ["docker", "pull", "node:20-slim"]


# --------------------------------------------------------------- oracle verdicts

def test_unwinnable_applies_only_on_matching_sha_and_not_won():
    C = b.WINNABILITY_CRITERION_VERSION
    v = {"o/a": {"sha": "abc", "won": False, "why": {"stderr": "ERROR: ResolutionImpossible"}, "criterion": C},
         "o/b": {"sha": "abc", "won": True, "criterion": C},
         "o/c": {"sha": "old", "won": False, "why": "x", "criterion": C}}
    assert b._unwinnable({"repo": "o/a", "sha": "abc"}, v).startswith("oracle_unwinnable:")
    assert "ResolutionImpossible" in b._unwinnable({"repo": "o/a", "sha": "abc"}, v)
    assert b._unwinnable({"repo": "o/b", "sha": "abc"}, v) is None
    assert b._unwinnable({"repo": "o/c", "sha": "new"}, v) is None   # stale verdict ignored
    assert b._unwinnable({"repo": "o/d", "sha": "abc"}, v) is None   # never run


def test_verdict_won_and_lost():
    won = w._verdict({"sha": "s"}, {"complete": True, "total": 7.9, "ended": "terminated", "seconds": 4, "steps": []})
    assert won["won"] is True and "why" not in won
    lost = w._verdict({"sha": "s"}, {
        "complete": False, "total": -2.1, "ended": "plan_exhausted", "seconds": 5,
        "missing_required": ["instagrapi"],
        "steps": [{"action": "install requirements.txt", "status": "failed",
                   "stderr_tail": "ERROR: Cannot install -r requirements.txt\nERROR: ResolutionImpossible: x\n\n[notice] pip"}]})
    assert lost["won"] is False
    assert lost["why"]["failed_steps"] == ["install requirements.txt"]
    assert lost["why"]["stderr"].startswith("ERROR: Cannot install")
    assert lost["why"]["missing_required"] == ["instagrapi"]
    neg = w._verdict({"sha": "s"}, {"complete": True, "total": -0.5, "ended": "terminated", "seconds": 1, "steps": []})
    assert neg["won"] is False   # complete but non-positive total is not a win


def test_tail_steps_install_each_missing_required_package(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\nrich\ntabulate\n", encoding="utf-8")
    inst = TemplateInstance(template=TEMPLATES["setup_project"], manifest=scan_repo(str(tmp_path)))
    env = SimpleNamespace(_instance=inst, _installed_now={"requests", "pip"})
    steps = w._tail_steps(env)
    assert [(s.action_type, s.target) for s in steps] == [
        (ActionType.INSTALL, "rich"), (ActionType.INSTALL, "tabulate")]
    assert w._tail_steps(SimpleNamespace(_instance=inst, _installed_now=None)) == []


def test_select_only_lost_reruns_lost_and_unknown_keeps_won():
    entries = [{"repo": r, "sha": "s"} for r in ("o/won", "o/lost", "o/new", "o/stale")]
    C = w.CRITERION_VERSION
    v = {"o/won": {"sha": "s", "won": True, "criterion": C}, "o/lost": {"sha": "s", "won": False, "criterion": C},
         "o/stale": {"sha": "old", "won": True, "criterion": C}}
    pick = lambda **kw: [e["repo"] for e in w._select(entries, v, **kw)]
    assert pick(only_lost=True) == ["o/lost", "o/new", "o/stale"]
    assert pick(resume=True) == ["o/new", "o/stale"]
    assert pick() == ["o/won", "o/lost", "o/new", "o/stale"]


def test_verdict_records_image():
    v = w._verdict({"sha": "s"}, {"complete": True, "total": 7.9, "ended": "terminated",
                                  "seconds": 1, "steps": [], "image": "python-3.11-git-toolchain-v2"})
    assert v["image"] == "python-3.11-git-toolchain-v2"


def test_select_treats_old_criterion_verdict_as_stale():
    entries = [{"repo": "o/a", "sha": "s"}]
    old = {"o/a": {"sha": "s", "won": True}}            # pre-v2 verdict, no criterion
    assert [e["repo"] for e in w._select(entries, old, resume=True)] == ["o/a"]
    assert [e["repo"] for e in w._select(entries, old, only_lost=True)] == ["o/a"]


def test_move_to_val_takes_first_kept_in_preference_order():
    splits = {"train": [{"repo": r} for r in ("a", "b", "c", "d")], "val": [{"repo": "v"}]}
    moved = b._move_to_val(splits, preference=["x", "c", "a", "d"], n=2)
    assert moved == ["c", "a"]                                   # "x" not kept -> skipped
    assert [e["repo"] for e in splits["train"]] == ["b", "d"]
    assert [e["repo"] for e in splits["val"]] == ["v", "c", "a"]


def test_unwinnable_ignores_verdict_from_older_criterion():
    # the v1 (name-only, no-toolchain) losers must not stay excluded forever
    v = {"o/a": {"sha": "abc", "won": False, "why": "gcc missing"}}                    # v1: no criterion
    assert b._unwinnable({"repo": "o/a", "sha": "abc"}, v) is None
    v["o/a"]["criterion"] = b.WINNABILITY_CRITERION_VERSION
    assert b._unwinnable({"repo": "o/a", "sha": "abc"}, v).startswith("oracle_unwinnable:")
    assert w.CRITERION_VERSION == b.WINNABILITY_CRITERION_VERSION                       # one source of truth
