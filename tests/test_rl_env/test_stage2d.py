"""Stage 2D: Unit tests for venv on PATH, pytest SAFE override (D3), and proxy defaults.

Asserts:
1. The PATH env starts with /workspace_rw/.venv/bin: and contains the inspected default.
2. A REVIEW pytest command runs as SAFE with tier_override="REVIEW->SAFE" recorded.
3. A REVIEW command that isn't pytest stays REVIEW.
4. A BLOCKED command that starts with pytest (e.g. pytest; rm -rf /) stays BLOCKED.
5. Both train_system2_grpo.py and evaluate_system2.py default to proxy mode.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.rl_env.docker_executor import (
    DockerEpisodeExecutor,
    DockerExecutorConfig,
    NetworkConfig,
    _IMAGE_DEFAULT_PATH_CACHE,
    get_image_default_path,
)
from src.sandbox.models import (
    AtomicStep,
    ExecutionResult,
    ExecutionStatus,
    RiskTier,
)
from scripts.train_system2_grpo import get_parser as get_train_parser
from scripts.evaluate_system2 import get_parser as get_eval_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subprocess_dispatcher(custom_default_path: str = "/custom/usr/local/bin:/usr/bin"):
    """Dispatcher for subprocess.run in DockerEpisodeExecutor."""
    def fake_run(cmd, *args, **kwargs):
        cmd_list = list(cmd)

        if cmd_list[:2] == ["docker", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Docker version 27.0.0\n", "")

        if cmd_list[:3] == ["docker", "image", "inspect"]:
            # Format inspect for Env
            if "--format" in cmd_list:
                env_json = json.dumps([f"PATH={custom_default_path}", "LANG=C.UTF-8"])
                return subprocess.CompletedProcess(cmd, 0, f"{env_json}\n", "")
            return subprocess.CompletedProcess(cmd, 0, "[]\n", "")

        if cmd_list[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")

        if cmd_list[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")

        if cmd_list[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, "mock_cid_stage2d\n", "")

        if cmd_list[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if cmd_list[:2] == ["docker", "stop"] or cmd_list[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")

        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_path_env_starts_with_venv_and_contains_inspected_default(tmp_path):
    """Assert: the PATH env starts with /workspace_rw/.venv/bin: and contains the inspected default."""
    _IMAGE_DEFAULT_PATH_CACHE.clear()

    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    custom_default = "/custom/opt/bin:/custom/usr/bin"
    dispatcher = _make_subprocess_dispatcher(custom_default_path=custom_default)

    executor = DockerEpisodeExecutor()
    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=dispatcher) as mock_run:
        executor.start_episode(str(repo_dir), repo_name="test/stage2d_repo", ecosystems=["python"])

        # Check docker run calls
        run_calls = [
            c for c in mock_run.call_args_list
            if len(c[0][0]) > 2 and c[0][0][0] == "docker" and c[0][0][1] == "run"
        ]
        assert len(run_calls) == 1
        cmd = run_calls[0][0][0]

        # Find PATH in cmd
        path_entries = [
            cmd[i + 1] for i, arg in enumerate(cmd)
            if arg == "--env" and i + 1 < len(cmd) and cmd[i + 1].startswith("PATH=")
        ]
        assert len(path_entries) == 1, f"Expected 1 PATH env entry, found: {path_entries}"
        path_val = path_entries[0].split("=", 1)[1]

        assert path_val.startswith("/workspace_rw/.venv/bin:"), (
            f"PATH must start with /workspace_rw/.venv/bin:, got: {path_val}"
        )
        assert custom_default in path_val, (
            f"PATH must contain inspected default {custom_default}, got: {path_val}"
        )

        executor.end_episode()


def test_review_pytest_runs_as_safe_with_tier_override():
    """Assert: a REVIEW pytest command runs as SAFE with tier_override recorded."""
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    # Mock _run_docker_exec so it doesn't contact docker daemon
    from src.rl_env.docker_executor import _ExecResult
    executor._run_docker_exec = MagicMock(return_value=_ExecResult(returncode=0, stdout="1 passed", stderr=""))

    step = AtomicStep(command="pytest tests/")
    # Confirm classifier normally considers this REVIEW
    classification = executor._classifier.classify(step)
    assert classification.tier == RiskTier.REVIEW

    # Execute step with real executor
    result = executor.execute(step)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.tier == RiskTier.SAFE
    assert result.tier_override == "REVIEW->SAFE"
    assert result.extra.get("tier_override") == "REVIEW->SAFE"


def test_review_command_not_pytest_stays_review():
    """Assert: a REVIEW command that isn't pytest stays REVIEW."""
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    step = AtomicStep(command="curl https://example.com")
    classification = executor._classifier.classify(step)
    assert classification.tier == RiskTier.REVIEW

    result = executor.execute(step)

    assert result.status == ExecutionStatus.ABORTED
    assert result.tier == RiskTier.REVIEW
    assert result.tier_override is None


def test_blocked_command_starting_with_pytest_stays_blocked():
    """Assert: a BLOCKED command that starts with pytest (e.g. pytest; rm -rf /) stays BLOCKED."""
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    step = AtomicStep(command="pytest; rm -rf /")
    classification = executor._classifier.classify(step)
    assert classification.tier == RiskTier.BLOCKED, (
        f"Classifier did not return BLOCKED for 'pytest; rm -rf /': got {classification.tier}"
    )

    result = executor.execute(step)

    assert result.status == ExecutionStatus.BLOCKED
    assert result.tier == RiskTier.BLOCKED
    assert result.tier_override is None


def test_both_scripts_default_to_proxy():
    """Assert: both train_system2_grpo.py and evaluate_system2.py default to proxy mode."""
    train_parser = get_train_parser()
    train_args_default = train_parser.parse_args([])
    assert train_args_default.network == "proxy"

    train_args_none = train_parser.parse_args(["--network", "none"])
    assert train_args_none.network == "none"

    eval_parser = get_eval_parser()
    eval_args_default = eval_parser.parse_args([])
    assert eval_args_default.network == "proxy"

    eval_args_none = eval_parser.parse_args(["--network", "none"])
    assert eval_args_none.network == "none"
