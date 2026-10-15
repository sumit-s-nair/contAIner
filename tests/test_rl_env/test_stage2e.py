"""
tests/test_rl_env/test_stage2e.py
=================================
Stage 2E: Harden the System 3 stub and the SAFE override (B11, B12, D6).

Cases:
1. requests>=2.0 is quoted, and its verify is pip show requests;
2. PyYAML verifies as pip show PyYAML;
3. a target like . ; rm -rf ~ is quoted, and the override refuses it;
4. CHECK expands to 2 steps;
5. pytest exit code 5 -> SUCCESS with verified False;
6. exit code 1 -> FAILED;
7. pytest x && curl y is not overridden.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.rl_env.docker_executor import (
    DockerEpisodeExecutor,
    DockerExecutorConfig,
)
from src.rl_env.system3_stub import expand_planned_step
from src.sandbox.models import (
    AtomicStep,
    ExecutionStatus,
    RiskTier,
)
from src.system2_planner.models import ActionType, PlannedStep


def test_requests_with_version_is_quoted_and_verifies_as_pip_show():
    """Case 1: requests>=2.0 is quoted, and its verify is pip show requests."""
    planned = PlannedStep(
        action_type=ActionType.INSTALL,
        target="requests>=2.0",
        description="Install requests with version requirement",
    )
    steps = expand_planned_step(planned)
    assert len(steps) == 1
    step = steps[0]

    # Target is quoted with shlex
    assert step.command == "python -m pip install 'requests>=2.0'"
    # Verify command is pip show <dist_name>
    assert step.verify_command == "python -m pip show requests"


def test_pyyaml_verifies_as_pip_show_pyyaml():
    """Case 2: PyYAML verifies as pip show PyYAML."""
    planned = PlannedStep(
        action_type=ActionType.INSTALL,
        target="PyYAML",
        description="Install PyYAML",
    )
    steps = expand_planned_step(planned)
    assert len(steps) == 1
    step = steps[0]

    assert step.command == "python -m pip install PyYAML"
    assert step.verify_command == "python -m pip show PyYAML"

    # Call real executor with mocked subprocess
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid_2e_2"

    with patch("src.rl_env.docker_executor.subprocess.Popen") as mock_popen, \
         patch("src.rl_env.docker_executor.subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("Name: PyYAML\nVersion: 6.0", "")
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        mock_run.return_value = subprocess.CompletedProcess([], 0, "true\n", "")

        res = executor.execute(step)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.verified is True
        all_executed_cmds = [call.args[0] for call in mock_popen.call_args_list]
        assert any("python -m pip show PyYAML" in " ".join(c) for c in all_executed_cmds)


def test_target_with_shell_injection_is_quoted_and_override_refuses():
    """Case 3: a target like . ; rm -rf ~ is quoted, and the override refuses it."""
    malicious_target = ". ; rm -rf ~"
    planned = PlannedStep(
        action_type=ActionType.INSTALL,
        target=malicious_target,
        description="Install with command injection",
    )
    steps = expand_planned_step(planned)
    assert len(steps) == 1
    step = steps[0]

    # Command has quoted target
    assert step.command == "python -m pip install '. ; rm -rf ~'"
    # Does not parse as requirement -> verify is false
    assert step.verify_command == "false"

    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid_2e_3"

    # Safe override refuses it due to ';'
    assert executor._matches_safe_override(step.command) is False

    # Also test that a pytest command containing this target is refused by the override
    pytest_injected = "pytest '. ; rm -rf ~'"
    assert executor._matches_safe_override(pytest_injected) is False

    # Executing the step should not run via SAFE override
    res = executor.execute(step)
    assert res.tier in (RiskTier.REVIEW, RiskTier.BLOCKED)
    assert res.status in (ExecutionStatus.ABORTED, ExecutionStatus.BLOCKED)
    assert res.tier_override is None


def test_check_expands_to_two_steps():
    """Case 4: CHECK expands to 2 steps."""
    # With target
    planned_with_target = PlannedStep(
        action_type=ActionType.CHECK,
        target="tests/unit",
        description="Run test suite",
    )
    steps = expand_planned_step(planned_with_target)
    assert len(steps) == 2

    # Step 1: install pytest
    assert steps[0].command == "python -m pip install pytest==8.3.5"
    assert steps[0].verify_command == "python -m pip show pytest"

    # Step 2: run pytest on target with unverified_returncodes={5}
    assert steps[1].command == "python -m pytest tests/unit"
    assert steps[1].unverified_returncodes == frozenset({5})

    # Without target
    planned_no_target = PlannedStep(
        action_type=ActionType.CHECK,
        target="",
        description="Run test suite default",
    )
    steps_no_target = expand_planned_step(planned_no_target)
    assert len(steps_no_target) == 2
    assert steps_no_target[0].command == "python -m pip install pytest==8.3.5"
    assert steps_no_target[1].command == "python -m pytest"
    assert steps_no_target[1].unverified_returncodes == frozenset({5})


def test_pytest_exit_code_5_yields_success_unverified():
    """Case 5: pytest exit code 5 -> SUCCESS with verified False."""
    step = AtomicStep(
        command="python -m pytest",
        unverified_returncodes=frozenset({5}),
    )

    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid_2e_5"

    with patch("src.rl_env.docker_executor.subprocess.Popen") as mock_popen, \
         patch("src.rl_env.docker_executor.subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("=== no tests ran in 0.01s ===", "")
        mock_proc.poll.return_value = 5
        mock_proc.returncode = 5
        mock_popen.return_value = mock_proc
        # Container inspect confirms running
        mock_run.return_value = subprocess.CompletedProcess([], 0, "true\n", "")

        res = executor.execute(step)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.verified is False
        assert res.returncode == 5


def test_exit_code_1_yields_failed():
    """Case 6: exit code 1 -> FAILED."""
    step = AtomicStep(
        command="python -m pytest",
        unverified_returncodes=frozenset({5}),
    )

    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid_2e_6"

    with patch("src.rl_env.docker_executor.subprocess.Popen") as mock_popen, \
         patch("src.rl_env.docker_executor.subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("FAILED tests/test_foo.py::test_bar", "AssertionError")
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc
        # Container inspect confirms running
        mock_run.return_value = subprocess.CompletedProcess([], 0, "true\n", "")

        res = executor.execute(step)
        assert res.status == ExecutionStatus.FAILED
        assert res.verified is False
        assert res.returncode == 1


def test_pytest_with_chained_command_is_not_overridden():
    """Case 7: pytest x && curl y is not overridden."""
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid_2e_7"

    command = "pytest x && curl y"
    assert executor._matches_safe_override(command) is False

    step = AtomicStep(command=command)
    classification = executor._classifier.classify(step)
    assert classification.tier == RiskTier.REVIEW

    res = executor.execute(step)
    assert res.tier == RiskTier.REVIEW
    assert res.status == ExecutionStatus.ABORTED
    assert res.tier_override is None
