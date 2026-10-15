"""
tests/test_rl_env/test_stage2e_fix.py
=====================================
Stage 2E-fix: Tests for the three fixes from Stage 2E.

Cases:
1. python -m pip install 'requests>=2.0' -> SAFE;
2. echo x > out.txt -> still REVIEW;
3. cmd 2>/dev/null -> unchanged from before;
4. cmd 'unbalanced -> falls back to the regex;
5. CHECK where pytest exits 5 -> env info has status success and verified False;
6. CHECK where pytest exits 0 -> verified True;
7. _matches_safe_override refuses commands containing backtick character.
"""

from __future__ import annotations

import pytest

from src.rl_env.docker_executor import DockerEpisodeExecutor
from src.rl_env.env import EnvConfig, PlannerEnv
from src.rl_env.system3_stub import expand_planned_step
from src.sandbox.classifier import CommandRiskClassifier
from src.sandbox.models import (
    AtomicStep,
    ExecutionResult,
    ExecutionStatus,
    RiskTier,
)
from src.system2_planner.models import ActionType, PlannedStep


def test_requests_with_version_specifier_is_safe():
    """Case 1: python -m pip install 'requests>=2.0' -> SAFE."""
    clf = CommandRiskClassifier()
    step = AtomicStep(command="python -m pip install 'requests>=2.0'")
    res = clf.classify(step)
    assert res.tier == RiskTier.SAFE
    assert res.reason == "Matches known safe action type."


def test_output_redirect_is_still_review():
    """Case 2: echo x > out.txt -> still REVIEW."""
    clf = CommandRiskClassifier()
    step = AtomicStep(command="echo x > out.txt")
    res = clf.classify(step)
    assert res.tier == RiskTier.REVIEW
    assert res.matched_pattern == r"(?<!>)>\s*(?!/dev/null)[^\s>]"
    assert res.reason == "Output redirect that may overwrite an existing file without a backup."


def test_cmd_redirect_dev_null_unchanged_from_before():
    """Case 3: cmd 2>/dev/null -> unchanged from before."""
    clf = CommandRiskClassifier()
    step = AtomicStep(command="cmd 2>/dev/null")
    res = clf.classify(step)
    # The out-of-scope check in Pass 1 flags /dev/null
    assert res.tier == RiskTier.BLOCKED
    assert res.matched_pattern == "<out-of-scope-write>"


def test_unbalanced_quotes_falls_back_to_regex():
    """Case 4: cmd 'unbalanced -> falls back to the regex."""
    clf = CommandRiskClassifier()

    # Without redirect: falls back to regex which returns None, then hits fail-safe REVIEW
    step_no_redir = AtomicStep(command="cmd 'unbalanced")
    res_no_redir = clf.classify(step_no_redir)
    assert res_no_redir.tier == RiskTier.REVIEW
    assert res_no_redir.matched_pattern == "<default-fail-safe>"

    # With redirect: falls back to regex which matches > out.txt, triggering output redirect REVIEW
    step_with_redir = AtomicStep(command="cmd 'unbalanced > out.txt")
    res_with_redir = clf.classify(step_with_redir)
    assert res_with_redir.tier == RiskTier.REVIEW
    assert res_with_redir.matched_pattern == r"(?<!>)>\s*(?!/dev/null)[^\s>]"


class FakeCheckExecutor:
    """Fake executor returning predetermined ExecutionResults for CHECK steps."""

    def __init__(self, pytest_rc: int):
        self.pytest_rc = pytest_rc

    def execute(self, step: AtomicStep) -> ExecutionResult:
        if "pip install" in step.command:
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.SUCCESS,
                tier=RiskTier.SAFE,
                verified=True,
                returncode=0,
            )
        else:  # pytest step
            unverified_rcs = getattr(step, "unverified_returncodes", frozenset()) or frozenset()
            status = (
                ExecutionStatus.SUCCESS
                if (self.pytest_rc == 0 or self.pytest_rc in unverified_rcs)
                else ExecutionStatus.FAILED
            )
            return ExecutionResult(
                step=step,
                status=status,
                tier=RiskTier.SAFE,
                verified=False,
                returncode=self.pytest_rc,
            )


def test_env_aggregation_check_exit_code_5():
    """Case 5: CHECK where pytest exits 5 -> env info has status success and verified False."""
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._executor = FakeCheckExecutor(pytest_rc=5)

    step = PlannedStep(action_type=ActionType.CHECK, target=".", description="check tests")
    obs, reward, done, truncated, info = env.step(step)

    assert info.get("exec_status") == "success"
    assert info.get("verified") is False
    assert info.get("returncode") == 5


def test_env_aggregation_check_exit_code_0():
    """Case 6: CHECK where pytest exits 0 -> verified True."""
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._executor = FakeCheckExecutor(pytest_rc=0)

    step = PlannedStep(action_type=ActionType.CHECK, target=".", description="check tests")
    obs, reward, done, truncated, info = env.step(step)

    assert info.get("exec_status") == "success"
    assert info.get("verified") is True
    assert info.get("returncode") == 0


def test_matches_safe_override_refuses_backtick():
    """Case 7: _matches_safe_override refuses command containing backtick (`, U+0060)."""
    executor = DockerEpisodeExecutor()
    assert executor._matches_safe_override("pytest `id`") is False
    assert executor._matches_safe_override("python -m pytest `whoami`") is False
    assert executor._matches_safe_override("pytest `echo tests`") is False
    # Backslash is also refused
    assert executor._matches_safe_override("pytest \\tests") is False


# ---------------------------------------------------------------------------
# Stage 2E-fix2: quote-masking redirect matcher + tier_override propagation
# ---------------------------------------------------------------------------

import re as _re

from src.sandbox.classifier import _OutputRedirectPattern

_OLD_REDIRECT_RE = _re.compile(r"(?<!>)>\s*(?!/dev/null)[^\s>]", _re.IGNORECASE | _re.DOTALL)
_REDIRECT_REASON = "Output redirect that may overwrite an existing file without a backup."


@pytest.mark.parametrize("cmd", [
    "echo a#b>out.txt",
    "cat setup.py#>x",
    "echo x &>out.txt",
    "echo x >& out.txt",
    "echo x>out.txt",
    "echo x > out.txt",
    "cmd 2>&1",
    "cmd >> out.txt",
    "cmd > /dev/null",
    "cmd 2>/dev/null",
    "echo x \\> out.txt",
    "ls -la",
])
def test_unquoted_commands_identical_to_original_regex(cmd):
    """Without quotes the matcher must behave exactly like the original regex."""
    assert bool(_OutputRedirectPattern().search(cmd)) == bool(_OLD_REDIRECT_RE.search(cmd))


@pytest.mark.parametrize("cmd", [
    "echo a#b>out.txt",
    "cat setup.py#>x",
    "echo x &>out.txt",
    "echo x >& out.txt",
])
def test_former_shlex_bypasses_are_review(cmd):
    """Regression: these were SAFE under the shlex-based matcher."""
    res = CommandRiskClassifier().classify(AtomicStep(command=cmd))
    assert res.tier == RiskTier.REVIEW
    assert res.reason == _REDIRECT_REASON


@pytest.mark.parametrize("cmd,expect_redirect", [
    ("python -m pip install 'requests>=2.0'", False),
    ('python -m pip install "requests>=2.0,<3"', False),
    ("echo 'a>b'", False),
    ('echo "a\\">b"', False),
    ('echo "a" > out.txt', True),
    ("echo 'a' >out.txt", True),
    ("echo \\' > out.txt \\'", True),   # escaped quotes must not open a masked region
    ("cmd 'unbalanced > out.txt", True),  # unbalanced -> raw regex
])
def test_quoted_redirect_handling(cmd, expect_redirect):
    assert bool(_OutputRedirectPattern().search(cmd)) is expect_redirect


class _OverrideCheckExecutor:
    """Fake executor: pip step plain SAFE, pytest step SAFE via REVIEW->SAFE override."""

    def execute(self, step: AtomicStep) -> ExecutionResult:
        if "pytest" in step.command and "pip install" not in step.command:
            return ExecutionResult(
                step=step, status=ExecutionStatus.SUCCESS, tier=RiskTier.SAFE,
                verified=False, returncode=0, tier_override="REVIEW->SAFE",
                extra={"tier_override": "REVIEW->SAFE"},
            )
        return ExecutionResult(
            step=step, status=ExecutionStatus.SUCCESS, tier=RiskTier.SAFE,
            verified=True, returncode=0,
        )


def test_check_aggregation_propagates_tier_override():
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._executor = _OverrideCheckExecutor()
    step = PlannedStep(action_type=ActionType.CHECK, target=".", description="check tests")
    _, _, _, _, info = env.step(step)
    assert info.get("tier_override") == "REVIEW->SAFE"


def test_check_aggregation_no_override_when_none():
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env._executor = FakeCheckExecutor(pytest_rc=0)
    step = PlannedStep(action_type=ActionType.CHECK, target=".", description="check tests")
    _, _, _, _, info = env.step(step)
    assert "tier_override" not in info
