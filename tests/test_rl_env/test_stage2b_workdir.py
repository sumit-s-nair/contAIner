#!/usr/bin/env python3
"""
tests/test_rl_env/test_stage2b_workdir.py
=========================================
Tests for Stage 2B: writable repository working directory (/workspace_rw).
Asserts:
(a) the cp runs after mkdir on a fresh start;
(b) the cp is skipped when a snapshot image is used;
(c) a cp failure triggers cleanup and raises;
(d) step and verify exec calls both include -w /workspace_rw.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.rl_env.docker_executor import DockerEpisodeExecutor
from src.sandbox.models import AtomicStep


def test_cp_runs_after_mkdir_on_fresh_start():
    """Assert (a): the cp runs after mkdir on a fresh start."""
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    call_history = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        call_history.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="cid_test_fresh\n", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_subprocess_run):
        executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    mkdir_indices = [
        i for i, cmd in enumerate(call_history)
        if cmd[:2] == ["docker", "exec"] and "mkdir" in cmd and "/workspace_rw" in cmd
    ]
    cp_indices = [
        i for i, cmd in enumerate(call_history)
        if cmd[:2] == ["docker", "exec"] and "cp" in cmd and "/workspace/." in cmd and "/workspace_rw/" in cmd
    ]

    assert len(mkdir_indices) == 1, f"Expected 1 mkdir call, found {len(mkdir_indices)}"
    assert len(cp_indices) == 1, f"Expected 1 cp call, found {len(cp_indices)}"
    assert cp_indices[0] > mkdir_indices[0], "cp must run after mkdir on a fresh start"


def test_cp_is_skipped_when_snapshot_image_used():
    """Assert (b): the cp is skipped when a snapshot image is used."""
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    call_history = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        call_history.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="cid_test_snap\n", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_subprocess_run):
        executor.start_episode(
            repo_path="/fake/repo",
            repo_name="owner/repo",
            snapshot_image="grpo-snapshot-abc12345",
        )

    cp_calls = [
        cmd for cmd in call_history
        if cmd[:2] == ["docker", "exec"] and "cp" in cmd
    ]
    assert len(cp_calls) == 0, f"Expected cp to be skipped with snapshot_image, but got: {cp_calls}"


def test_cp_failure_triggers_cleanup_and_raises():
    """Assert (c): a cp failure triggers cleanup and raises."""
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    call_history = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        call_history.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="cid_fail\n", stderr="")
        if cmd[:2] == ["docker", "exec"] and "cp" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="cp: write error")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(RuntimeError, match="Failed to copy /workspace to /workspace_rw"):
            executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    teardown_calls = [
        cmd for cmd in call_history
        if cmd[:2] in (["docker", "stop"], ["docker", "rm"])
    ]
    assert len(teardown_calls) >= 1, "Expected cleanup (docker rm/stop) on cp failure"
    assert executor._container_id is None, "Container ID must be None after end_episode"


def test_step_and_verify_exec_calls_include_w_workspace_rw():
    """Assert (d): step and verify exec calls both include -w /workspace_rw."""
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="cid_exec\n", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_subprocess_run):
        executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    step = AtomicStep(
        command="pip install requests",
        description="install requests",
        verify_command="python -c 'import requests'",
    )

    popen_calls = []

    def fake_popen(cmd, *args, **kwargs):
        popen_calls.append(list(cmd))
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("ok\n", "")
        mock_proc.returncode = 0
        return mock_proc

    with patch("src.rl_env.docker_executor.subprocess.Popen", side_effect=fake_popen):
        res = executor.execute(step)

    assert len(popen_calls) == 2, f"Expected 2 exec calls (step + verify), got {len(popen_calls)}"

    step_cmd = popen_calls[0]
    verify_cmd = popen_calls[1]

    # Step call checks
    assert "-w" in step_cmd, f"-w flag missing from step exec: {step_cmd}"
    w_idx_step = step_cmd.index("-w")
    assert step_cmd[w_idx_step + 1] == "/workspace_rw", f"Expected /workspace_rw after -w in step: {step_cmd}"

    # Verify call checks
    assert "-w" in verify_cmd, f"-w flag missing from verify exec: {verify_cmd}"
    w_idx_ver = verify_cmd.index("-w")
    assert verify_cmd[w_idx_ver + 1] == "/workspace_rw", f"Expected /workspace_rw after -w in verify: {verify_cmd}"
