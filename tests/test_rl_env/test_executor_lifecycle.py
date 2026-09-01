"""
tests/test_rl_env/test_executor_lifecycle.py
=============================================
Lifecycle contract tests for DockerEpisodeExecutor.

All tests are mock-based (no real Docker daemon required).
They verify that the correct ``docker`` CLI commands are issued in the correct
sequence — not that containers actually run.

Tests marked ``pytest.mark.integration`` require a live Docker daemon and are
skipped in CI by default.  Run them manually before the first training run::

    pytest tests/test_rl_env/test_executor_lifecycle.py -v -m integration

Contract invariants asserted here
-----------------------------------
1. ``start_episode()`` issues one ``docker run`` with the correct flags
   (ro bind-mount, tmpfs, memory/cpu/pids limits, network mode).
2. ``execute()`` issues ``docker exec`` for SAFE steps — never for BLOCKED.
3. The verify command runs through ``docker exec`` in the same container.
4. Both the main command and verify command use the container's environment
   (PYTHONUSERBASE=/workspace_rw) — the executor injects this at run time,
   not at exec time, so packages installed by a step are visible to verify.
5. ``end_episode()`` issues ``docker stop`` then ``docker rm -f`` (in that order).
6. The same container ID appears in all ``docker exec`` calls within one episode
   (container is NOT re-created between steps).
7. A second ``start_episode()`` call destroys the first container before
   creating a new one.  The new container gets a different ID.
8. A BLOCKED-classified step never reaches ``docker exec``.
9. On step timeout:
     a. ``proc.kill()`` is called — SIGKILL to the host-side docker exec process.
     b. ``proc.wait()`` is called — zombie is reaped.
     c. ``ExecutionResult.status == FAILED`` with ``[timeout]`` in stderr.
10. When Docker is unavailable, ``start_episode()`` raises ``DockerNotAvailableError``.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

from src.rl_env.docker_executor import (
    DockerEpisodeExecutor,
    DockerExecutorConfig,
    DockerNotAvailableError,
    NetworkNotConfiguredError,
    NetworkConfig,
    select_image,
)
from src.sandbox.models import AtomicStep, ExecutionStatus, RiskTier


# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------

# Container IDs used in tests.
_CONTAINER_ID_1 = "sha256abcdef1234567890abcdef1234567890ab"
_CONTAINER_ID_2 = "sha256fedcba0987654321fedcba0987654321fe"

_FAKE_REPO_PATH = "/fake/clone/owner_repo_abc1234a"
_FAKE_REPO_NAME = "owner/repo"


def _make_run_mock(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a MagicMock that looks like a subprocess.CompletedProcess."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout     = stdout
    m.stderr     = stderr
    return m


def _docker_run_dispatcher(container_id: str = _CONTAINER_ID_1):
    """
    Side-effect function for ``subprocess.run`` that:
      - Returns a version-check success for ``docker version``.
      - Returns a cache-hit for ``docker image inspect``.
      - Returns success for ``docker network inspect`` (with correct label).
      - Returns *container_id* for ``docker run``.
      - Returns success for ``docker stop`` and ``docker rm``.
    """
    def dispatch(cmd: List[str], **kwargs: Any) -> MagicMock:
        if "version" in cmd:
            return _make_run_mock(stdout="27.0.0")
        if "image" in cmd and "inspect" in cmd:
            return _make_run_mock(stdout="[]")          # image cached
        if "network" in cmd and "inspect" in cmd:
            import json
            labels = {"com.contai.rl.network.verified": "allowlist-v1"}
            return _make_run_mock(stdout=json.dumps(labels))
        if "run" in cmd:
            return _make_run_mock(stdout=container_id)  # container started
        if "stop" in cmd or "rm" in cmd:
            return _make_run_mock()                     # teardown success
        return _make_run_mock()                         # default: success
    return dispatch


def _make_executor_with_started_episode(
    container_id: str = _CONTAINER_ID_1,
    config: DockerExecutorConfig | None = None,
) -> tuple[DockerEpisodeExecutor, MagicMock]:
    """
    Create a DockerEpisodeExecutor and call start_episode(), returning
    (executor, subprocess_run_mock).  All docker CLI calls are mocked.
    """
    run_mock = MagicMock(side_effect=_docker_run_dispatcher(container_id))

    with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
        executor = DockerEpisodeExecutor(config=config)
        executor.start_episode(
            repo_path=_FAKE_REPO_PATH,
            repo_name=_FAKE_REPO_NAME,
            ecosystems=["python"],
        )

    return executor, run_mock


# ---------------------------------------------------------------------------
# 1. start_episode() — docker run flags
# ---------------------------------------------------------------------------

class TestStartEpisode:

    def test_docker_run_called_exactly_once(self) -> None:
        """start_episode() issues exactly one 'docker run' call."""
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())
        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            executor = DockerEpisodeExecutor()
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_calls = [c for c in run_mock.call_args_list if "run" in c.args[0]]
        assert len(run_calls) == 1, (
            f"Expected exactly 1 'docker run' call, got {len(run_calls)}"
        )

    def test_docker_run_includes_read_only_repo_mount(self) -> None:
        """Repo is bind-mounted read-only at /workspace."""
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())
        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            executor = DockerEpisodeExecutor()
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_cmd = next(
            c.args[0] for c in run_mock.call_args_list if "run" in c.args[0]
        )
        assert f"{_FAKE_REPO_PATH}:/workspace:ro" in run_cmd, (
            "docker run must bind-mount the repo read-only at /workspace"
        )

    def test_docker_run_includes_workspace_rw_tmpfs(self) -> None:
        """A writable tmpfs is created at /workspace_rw for episode side-effects."""
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())
        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            executor = DockerEpisodeExecutor()
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_cmd = next(
            c.args[0] for c in run_mock.call_args_list if "run" in c.args[0]
        )
        assert "--tmpfs" in run_cmd, "docker run must include --tmpfs flag"
        tmpfs_idx = run_cmd.index("--tmpfs")
        tmpfs_val = run_cmd[tmpfs_idx + 1]
        assert tmpfs_val.startswith("/workspace_rw"), (
            f"tmpfs must be at /workspace_rw, got: {tmpfs_val!r}"
        )
        assert "exec" in tmpfs_val, (
            "tmpfs must be mounted exec (pip wheels require executable pages)"
        )

    def test_docker_run_injects_workspace_rw_env_vars(self) -> None:
        """
        /workspace_rw-aware env vars are injected so package managers install
        to the writable tmpfs and verify commands can import installed packages.
        """
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())
        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            executor = DockerEpisodeExecutor()
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_cmd = next(
            c.args[0] for c in run_mock.call_args_list if "run" in c.args[0]
        )
        cmd_str = " ".join(run_cmd)
        assert "PYTHONUSERBASE=/workspace_rw" in cmd_str, (
            "PYTHONUSERBASE must be set so pip --user installs land in /workspace_rw"
        )
        assert "NPM_CONFIG_PREFIX=/workspace_rw" in cmd_str, (
            "NPM_CONFIG_PREFIX must be set for npm installs"
        )

    def test_docker_run_applies_resource_limits(self) -> None:
        """Memory, CPU quota, and PID limits are passed to docker run."""
        cfg = DockerExecutorConfig(memory_limit="1g", cpu_quota=25_000, pids_limit=32)
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())
        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            executor = DockerEpisodeExecutor(config=cfg)
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_cmd = next(
            c.args[0] for c in run_mock.call_args_list if "run" in c.args[0]
        )
        cmd_str = " ".join(run_cmd)
        assert "--memory" in run_cmd and "1g" in run_cmd, "memory limit not set"
        assert "--cpu-quota" in run_cmd and "25000" in run_cmd, "cpu-quota not set"
        assert "--pids-limit" in run_cmd and "32" in run_cmd, "pids-limit not set"

    def test_docker_run_uses_none_network_by_default(self) -> None:
        """Default network mode is 'none' (no outbound network)."""
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())
        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            executor = DockerEpisodeExecutor()
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_cmd = next(
            c.args[0] for c in run_mock.call_args_list if "run" in c.args[0]
        )
        assert "--network" in run_cmd, "docker run must include --network flag"
        net_idx = run_cmd.index("--network")
        assert run_cmd[net_idx + 1] == "none", (
            f"Default network must be 'none', got {run_cmd[net_idx + 1]!r}"
        )

    def test_docker_run_uses_allowlist_network_mode_with_add_hosts(self) -> None:
        """
        Allowlist mode resolves registry hostnames to IPs and injects them
        as --add-host entries.
        """
        net_cfg = NetworkConfig(
            mode="allowlist",
            allowed_registry_hosts=["pypi.org"],
            docker_network_name="rl_allowlist",
        )
        cfg = DockerExecutorConfig(network=net_cfg)
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())

        with patch("src.rl_env.docker_executor.subprocess.run", run_mock), \
             patch("src.rl_env.docker_executor.socket.gethostbyname",
                   return_value="151.101.0.63"):
            executor = DockerEpisodeExecutor(config=cfg)
            executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

        run_cmd = next(
            c.args[0] for c in run_mock.call_args_list if "run" in c.args[0]
        )
        assert "--network" in run_cmd
        net_idx = run_cmd.index("--network")
        assert run_cmd[net_idx + 1] == "rl_allowlist", (
            "Named allowlist network should be used when docker_network_name is set"
        )
        assert "--add-host" in run_cmd, (
            "--add-host must be injected for resolved registry hosts"
        )
        add_idx = run_cmd.index("--add-host")
        assert "pypi.org:151.101.0.63" == run_cmd[add_idx + 1], (
            "Resolved IP must match the mocked return value"
        )

    def test_container_id_is_captured_after_start(self) -> None:
        """executor.container_id is set to the container ID after start_episode."""
        executor, _ = _make_executor_with_started_episode(_CONTAINER_ID_1)
        assert executor.container_id == _CONTAINER_ID_1, (
            f"Expected container_id={_CONTAINER_ID_1!r}, "
            f"got {executor.container_id!r}"
        )

    def test_per_ecosystem_image_python(self) -> None:
        """Python ecosystem → python:3.11-slim image."""
        assert select_image(["python"]) == "python:3.11-slim"

    def test_per_ecosystem_image_javascript(self) -> None:
        """JavaScript ecosystem → node:20-slim image."""
        assert select_image(["javascript"]) == "node:20-slim"

    def test_per_ecosystem_image_mixed_python_node(self) -> None:
        """Python + Node/JS → combined python-nodejs image."""
        img = select_image(["python", "node"])
        assert "python" in img and "node" in img, (
            f"Mixed Python+Node should use a combined image, got: {img!r}"
        )

    def test_per_ecosystem_image_default_for_empty(self) -> None:
        """None or empty ecosystems → python:3.11-slim fallback."""
        assert select_image(None) == "python:3.11-slim"
        assert select_image([]) == "python:3.11-slim"

    def test_multi_ecosystem_fallback_logs_warning(self) -> None:
        """Multi-ecosystem without combined image falls back to priority and logs explicitly."""
        import sys
        from io import StringIO
        
        # Rust has higher priority than Java/Ruby in _ECO_PRIORITY
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            img = select_image(["rust", "ruby"])
            assert img == "rust:1.75-slim"
            
            log_output = sys.stderr.getvalue()
            assert "multi-ecosystem repo" in log_output
            assert "has no combined image" in log_output
            assert "ruby" in log_output
        finally:
            sys.stderr = old_stderr



class TestNetworkConfigFailClosed:
    
    def test_allowlist_mode_without_network_name_raises_error(self) -> None:
        """Allowlist mode with docker_network_name=None raises NetworkNotConfiguredError."""
        net_cfg = NetworkConfig(mode="allowlist", docker_network_name=None)
        cfg = DockerExecutorConfig(network=net_cfg)
        
        with patch("src.rl_env.docker_executor.subprocess.run", side_effect=_docker_run_dispatcher()):
            executor = DockerEpisodeExecutor(config=cfg)
            with pytest.raises(NetworkNotConfiguredError, match="requires docker_network_name"):
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)
            
    def test_allowlist_network_not_found_raises_error(self) -> None:
        """If the named network doesn't exist in Docker, raises NetworkNotConfiguredError."""
        net_cfg = NetworkConfig(mode="allowlist", docker_network_name="missing_net")
        cfg = DockerExecutorConfig(network=net_cfg)
        
        def dispatch(cmd: List[str], **kwargs: Any) -> MagicMock:
            if "version" in cmd:
                return _make_run_mock(stdout="27.0.0")
            if "network" in cmd and "inspect" in cmd:
                return _make_run_mock(returncode=1, stderr="Error: No such network")
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run", side_effect=dispatch):
            executor = DockerEpisodeExecutor(config=cfg)
            with pytest.raises(NetworkNotConfiguredError, match="does not exist"):
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

    def test_allowlist_network_missing_label_raises_error(self) -> None:
        """If the network exists but lacks the setup-script label, raises NetworkNotConfiguredError."""
        net_cfg = NetworkConfig(mode="allowlist", docker_network_name="unverified_net")
        cfg = DockerExecutorConfig(network=net_cfg)
        
        def dispatch(cmd: List[str], **kwargs: Any) -> MagicMock:
            if "version" in cmd:
                return _make_run_mock(stdout="27.0.0")
            if "network" in cmd and "inspect" in cmd:
                import json
                return _make_run_mock(stdout=json.dumps({"some_other_label": "val"}))
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run", side_effect=dispatch):
            executor = DockerEpisodeExecutor(config=cfg)
            with pytest.raises(NetworkNotConfiguredError, match="missing the required setup-script label"):
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

# ---------------------------------------------------------------------------
# 2. execute() — docker exec routing
# ---------------------------------------------------------------------------

class TestExecute:

    def test_safe_step_uses_docker_exec(self) -> None:
        """SAFE steps are run via docker exec into the episode container."""
        executor, _ = _make_executor_with_started_episode()
        step = AtomicStep(command="pip install requests", description="install requests")

        popen_mock = MagicMock()
        popen_mock.communicate.return_value = ("Successfully installed\n", "")
        popen_mock.returncode = 0

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   return_value=popen_mock) as popen_cls:
            executor.execute(step)

        # It may be called twice if verify_table.py injects a verify command.
        # We check the first call which is the main command.
        assert popen_cls.call_count >= 1
        exec_cmd = popen_cls.call_args_list[0].args[0]
        assert exec_cmd[0] == "docker", "Must use docker CLI"
        assert exec_cmd[1] == "exec",   "Must use docker exec (not docker run)"
        assert _CONTAINER_ID_1 in exec_cmd, (
            "Container ID must appear in docker exec command"
        )

    def test_verify_command_uses_docker_exec_in_same_container(self) -> None:
        """
        The verify command runs via docker exec in the same container as the
        main command.  PYTHONUSERBASE is set at container start, not re-injected
        here — packages installed in the main command are visible automatically.
        """
        executor, _ = _make_executor_with_started_episode()
        step = AtomicStep(
            command="pip install requests",
            description="install requests",
            verify_command="python -c 'import requests'",
        )

        exec_calls: List[List[str]] = []

        def fake_popen(cmd, **kwargs):
            exec_calls.append(list(cmd))
            m = MagicMock()
            m.communicate.return_value = ("ok", "")
            m.returncode = 0
            return m

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   side_effect=fake_popen):
            result = executor.execute(step)

        # Two docker exec calls: main command + verify command
        assert len(exec_calls) == 2, (
            f"Expected 2 docker exec calls (main + verify), got {len(exec_calls)}"
        )
        # Both must target the same container
        for cmd in exec_calls:
            assert _CONTAINER_ID_1 in cmd, (
                "Both main and verify exec calls must reference the same container"
            )
        # The second call must include the verify command string
        verify_call_str = " ".join(exec_calls[1])
        assert "import requests" in verify_call_str, (
            "Second docker exec call must run the verify command"
        )
        assert result.verified is True, "verified must be True when verify exits 0"

    def test_verified_false_when_verify_exits_nonzero(self) -> None:
        """Verify command exiting non-zero → verified=False."""
        executor, _ = _make_executor_with_started_episode()
        step = AtomicStep(
            command="pip install requests",
            description="install requests",
            verify_command="python -c 'import requests'",
        )

        call_count = [0]

        def fake_popen(cmd, **kwargs):
            call_count[0] += 1
            m = MagicMock()
            if call_count[0] == 1:
                # Main command: success
                m.communicate.return_value = ("ok", "")
                m.returncode = 0
            else:
                # Verify command: failure
                m.communicate.return_value = ("", "ModuleNotFoundError")
                m.returncode = 1
            return m

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   side_effect=fake_popen):
            result = executor.execute(step)

        assert result.verified is False
        assert result.status == ExecutionStatus.SUCCESS   # main still succeeded
        assert result.verify_returncode == 1

    def test_blocked_step_never_reaches_docker_exec(self) -> None:
        """
        A BLOCKED-classified step never triggers docker exec.
        The container is never contacted.
        """
        executor, _ = _make_executor_with_started_episode()
        # rm -rf / is in the hard BLOCKED denylist
        step = AtomicStep(
            command="rm -rf /",
            description="wipe filesystem",
        )

        with patch("src.rl_env.docker_executor.subprocess.Popen") as popen_mock:
            result = executor.execute(step)

        popen_mock.assert_not_called()
        assert result.status == ExecutionStatus.BLOCKED
        assert result.tier == RiskTier.BLOCKED
        assert result.verified is False

    def test_review_step_auto_denied_no_docker_exec(self) -> None:
        """REVIEW-tier steps are auto-denied; docker exec is never called."""
        executor, _ = _make_executor_with_started_episode()
        # rm with -rf is REVIEW-tier
        step = AtomicStep(
            command="rm -rf ./dist",
            description="remove dist directory",
        )

        with patch("src.rl_env.docker_executor.subprocess.Popen") as popen_mock:
            result = executor.execute(step)

        popen_mock.assert_not_called()
        assert result.status == ExecutionStatus.ABORTED
        assert result.verified is False

    def test_execute_raises_without_start_episode(self) -> None:
        """execute() raises RuntimeError if called before start_episode()."""
        executor = DockerEpisodeExecutor()
        step = AtomicStep(command="pip install requests", description="install")
        with pytest.raises(RuntimeError, match="start_episode"):
            executor.execute(step)


# ---------------------------------------------------------------------------
# 3. end_episode() — container teardown
# ---------------------------------------------------------------------------

class TestEndEpisode:

    def test_end_episode_calls_docker_stop_then_rm(self) -> None:
        """end_episode() calls docker stop then docker rm -f in that order."""
        executor, _ = _make_executor_with_started_episode()
        teardown_calls: List[tuple] = []

        def record_teardown(cmd, **kwargs):
            if "stop" in cmd or "rm" in cmd:
                teardown_calls.append(tuple(cmd))
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run",
                   side_effect=record_teardown):
            executor.end_episode()

        assert len(teardown_calls) == 2, (
            f"Expected docker stop + docker rm -f, got: {teardown_calls}"
        )
        stop_cmd, rm_cmd = teardown_calls
        assert "stop" in stop_cmd,  "First teardown call must be docker stop"
        assert "rm" in rm_cmd,      "Second teardown call must be docker rm"
        assert "-f" in rm_cmd,      "docker rm must use -f flag"

    def test_end_episode_references_correct_container_id(self) -> None:
        """docker stop and docker rm -f reference the episode's container ID."""
        executor, _ = _make_executor_with_started_episode(_CONTAINER_ID_1)
        teardown_cmds: List[List[str]] = []

        def record(cmd, **kwargs):
            if "stop" in cmd or "rm" in cmd:
                teardown_cmds.append(list(cmd))
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run", side_effect=record):
            executor.end_episode()

        for cmd in teardown_cmds:
            assert _CONTAINER_ID_1 in cmd, (
                f"docker stop/rm must reference container {_CONTAINER_ID_1!r}"
            )

    def test_end_episode_clears_container_id(self) -> None:
        """After end_episode(), container_id is None."""
        executor, _ = _make_executor_with_started_episode()
        with patch("src.rl_env.docker_executor.subprocess.run",
                   return_value=_make_run_mock()):
            executor.end_episode()
        assert executor.container_id is None

    def test_end_episode_is_idempotent(self) -> None:
        """Calling end_episode() twice does not raise."""
        executor, _ = _make_executor_with_started_episode()
        with patch("src.rl_env.docker_executor.subprocess.run",
                   return_value=_make_run_mock()):
            executor.end_episode()
            executor.end_episode()   # second call: no-op, must not raise


# ---------------------------------------------------------------------------
# 4. State persistence — same container across steps
# ---------------------------------------------------------------------------

class TestContainerPersistenceWithinEpisode:

    def test_same_container_id_used_for_all_steps(self) -> None:
        """
        All docker exec calls within one episode reference the same container ID.
        The container is NOT recreated between steps.
        """
        executor, _ = _make_executor_with_started_episode(_CONTAINER_ID_1)
        exec_container_ids: List[str] = []

        def fake_popen(cmd, **kwargs):
            if "exec" in cmd:
                # The container ID is the 3rd element: docker exec <cid> ...
                exec_idx = cmd.index("exec")
                exec_container_ids.append(cmd[exec_idx + 1])
            m = MagicMock()
            m.communicate.return_value = ("ok", "")
            m.returncode = 0
            return m

        steps = [
            AtomicStep(command="pip install requests",  description="install requests"),
            AtomicStep(command="pip install flask",     description="install flask"),
            AtomicStep(command="pip list",              description="list packages"),
        ]

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   side_effect=fake_popen):
            for step in steps:
                executor.execute(step)

        assert len(exec_container_ids) == 6, (
            "Expected 6 docker exec calls (main + verify per step)"
        )
        assert all(cid == _CONTAINER_ID_1 for cid in exec_container_ids), (
            f"All steps must exec into the same container {_CONTAINER_ID_1!r}.\n"
            f"Got IDs: {exec_container_ids}"
        )


# ---------------------------------------------------------------------------
# 5. Episode isolation — fresh container between episodes
# ---------------------------------------------------------------------------

class TestContainerLifecycleBetweenEpisodes:

    def test_fresh_container_per_episode(self) -> None:
        """
        A second start_episode() call:
          1. Destroys the first container (docker stop + docker rm -f).
          2. Creates a new container with a different ID.
        The new container ID is distinct from the first.
        """
        call_log: List[tuple] = []
        run_counter = [0]
        container_ids = [_CONTAINER_ID_1, _CONTAINER_ID_2]

        def dispatch(cmd, **kwargs):
            if "version" in cmd or "inspect" in cmd:
                return _make_run_mock(stdout="ok")
            if "run" in cmd:
                idx = run_counter[0]
                cid = container_ids[idx] if idx < len(container_ids) else "extra"
                run_counter[0] += 1
                call_log.append(("run", cid))
                return _make_run_mock(stdout=cid)
            if "stop" in cmd:
                # The container ID is the last argument to docker stop
                call_log.append(("stop", cmd[-1]))
                return _make_run_mock()
            if "rm" in cmd:
                # The container ID is the last argument to docker rm -f
                call_log.append(("rm", cmd[-1]))
                return _make_run_mock()
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run", side_effect=dispatch):
            executor = DockerEpisodeExecutor()

            executor.start_episode("/path/ep1", "owner/ep1", ["python"])
            assert executor.container_id == _CONTAINER_ID_1

            executor.start_episode("/path/ep2", "owner/ep2", ["python"])
            assert executor.container_id == _CONTAINER_ID_2

        # ep1 container must be stopped before ep2 starts
        stop_events = [(op, cid) for op, cid in call_log if op == "stop"]
        run_events  = [(op, cid) for op, cid in call_log if op == "run"]

        assert len(run_events) == 2, "Expected exactly 2 docker run calls"

        ep1_run_pos  = next(i for i, (op, cid) in enumerate(call_log)
                            if op == "run" and cid == _CONTAINER_ID_1)
        ep1_stop_pos = next(i for i, (op, cid) in enumerate(call_log)
                            if op == "stop" and cid == _CONTAINER_ID_1)
        ep2_run_pos  = next(i for i, (op, cid) in enumerate(call_log)
                            if op == "run" and cid == _CONTAINER_ID_2)

        assert ep1_stop_pos < ep2_run_pos, (
            "Episode 1 container must be stopped BEFORE episode 2 container starts.\n"
            f"call_log: {call_log}"
        )
        assert _CONTAINER_ID_1 != _CONTAINER_ID_2, (
            "Each episode must get a distinct container ID"
        )

    def test_context_manager_calls_end_episode(self) -> None:
        """
        Using DockerEpisodeExecutor as a context manager calls end_episode()
        on __exit__, even if an exception is raised.
        """
        run_mock = MagicMock(side_effect=_docker_run_dispatcher())

        with patch("src.rl_env.docker_executor.subprocess.run", run_mock):
            with DockerEpisodeExecutor() as executor:
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)
                assert executor.container_id is not None

        # After __exit__: container should be gone
        assert executor.container_id is None


# ---------------------------------------------------------------------------
# 6. Timeout — real process kill confirmation
# ---------------------------------------------------------------------------

class TestStepTimeout:

    def test_timeout_kills_exec_process_and_reaps_zombie(self) -> None:
        """
        On TimeoutExpired:
          - proc.kill() is called (SIGKILL to the docker exec host process).
          - proc.wait() is called (zombie reap).
        These are the real OS-level cleanup calls — their invocation is the
        contract, not that a specific process is killed in a real container.
        """
        executor, _ = _make_executor_with_started_episode()

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd=["docker", "exec"], timeout=5
        )
        mock_proc.returncode = None

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   return_value=mock_proc), \
             patch("src.rl_env.docker_executor.subprocess.run",
                   return_value=_make_run_mock()):
            step = AtomicStep(
                command="pip install heavy-package",
                description="install large package",
            )
            result = executor.execute(step)

        # --- Real process-kill assertions ---
        mock_proc.kill.assert_called_once(), (
            "proc.kill() must be called on timeout to SIGKILL the docker exec process"
        )
        mock_proc.wait.assert_called_once(), (
            "proc.wait() must be called after kill() to reap the zombie process"
        )

    def test_timeout_returns_failed_result_with_timeout_message(self) -> None:
        """
        After a timeout, ExecutionResult has:
          - status == FAILED (non-zero returncode)
          - '[timeout]' in stderr
          - verified == False
        """
        executor, _ = _make_executor_with_started_episode()

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd=["docker", "exec"], timeout=5
        )
        mock_proc.returncode = None

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   return_value=mock_proc), \
             patch("src.rl_env.docker_executor.subprocess.run",
                   return_value=_make_run_mock()):
            step = AtomicStep(
                command="pip install heavy-package",
                description="install large package",
            )
            result = executor.execute(step)

        assert result.status == ExecutionStatus.FAILED, (
            f"Timed-out step must return FAILED, got {result.status!r}"
        )
        assert "[timeout]" in result.stderr, (
            f"Timed-out step must include '[timeout]' in stderr, got: {result.stderr!r}"
        )
        assert result.verified is False, "Timed-out step must not be verified"
        assert result.returncode != 0, "returncode must be non-zero on timeout"

    def test_timeout_in_step_does_not_kill_container(self) -> None:
        """
        A timed-out step kills the exec process but leaves the container
        running.  Subsequent steps can still execute.
        """
        executor, _ = _make_executor_with_started_episode()
        call_count = [0]

        def fake_popen(cmd, **kwargs):
            call_count[0] += 1
            m = MagicMock()
            if call_count[0] == 1:
                # First step: timeout
                m.communicate.side_effect = subprocess.TimeoutExpired(
                    cmd=cmd, timeout=5
                )
                m.returncode = None
            else:
                # Second step: success
                m.communicate.return_value = ("ok", "")
                m.returncode = 0
            return m

        with patch("src.rl_env.docker_executor.subprocess.Popen",
                   side_effect=fake_popen), \
             patch("src.rl_env.docker_executor.subprocess.run",
                   return_value=_make_run_mock()):
            step1 = AtomicStep(command="pip install heavy", description="slow install")
            step2 = AtomicStep(command="pip list",          description="list packages")

            result1 = executor.execute(step1)
            result2 = executor.execute(step2)

        assert result1.status == ExecutionStatus.FAILED
        assert result2.status == ExecutionStatus.SUCCESS, (
            "Container should still be running after a timed-out step; "
            "the second step must succeed."
        )


# ---------------------------------------------------------------------------
# 7. Docker availability check
# ---------------------------------------------------------------------------

class TestDockerAvailability:

    def test_docker_not_on_path_raises_DockerNotAvailableError(self) -> None:
        """FileNotFoundError from docker binary → DockerNotAvailableError."""
        executor = DockerEpisodeExecutor()

        with patch("src.rl_env.docker_executor.subprocess.run",
                   side_effect=FileNotFoundError("docker: not found")):
            with pytest.raises(DockerNotAvailableError, match="docker binary"):
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

    def test_docker_daemon_down_raises_DockerNotAvailableError(self) -> None:
        """docker version returning non-zero → DockerNotAvailableError."""
        executor = DockerEpisodeExecutor()

        def daemon_down(cmd, **kwargs):
            if "version" in cmd:
                return _make_run_mock(returncode=1, stderr="Cannot connect to daemon")
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run",
                   side_effect=daemon_down):
            with pytest.raises(DockerNotAvailableError, match="daemon"):
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)

    def test_docker_version_timeout_raises_DockerNotAvailableError(self) -> None:
        """docker version timing out → DockerNotAvailableError."""
        executor = DockerEpisodeExecutor()

        def version_timeout(cmd, **kwargs):
            if "version" in cmd:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)
            return _make_run_mock()

        with patch("src.rl_env.docker_executor.subprocess.run",
                   side_effect=version_timeout):
            with pytest.raises(DockerNotAvailableError, match="timed out"):
                executor.start_episode(_FAKE_REPO_PATH, _FAKE_REPO_NAME)


# ---------------------------------------------------------------------------
# 8. PlannerEnv integration (smoke, no Docker daemon)
# ---------------------------------------------------------------------------

class TestPlannerEnvDockerRouting:
    """
    Verify that PlannerEnv routes dry_run=True to DryRunExecutor and
    dry_run=False to DockerEpisodeExecutor, without requiring a real Docker
    daemon (executor is mocked at the PlannerEnv level).
    """

    def test_dry_run_env_uses_dry_run_executor(self) -> None:
        """dry_run=True → _executor is a DryRunExecutor, not DockerEpisodeExecutor."""
        from src.rl_env.env import PlannerEnv, EnvConfig
        from src.rl_env.dry_run_executor import DryRunExecutor

        cfg = EnvConfig(corpus_manifest_path=None, dry_run=True, seed=0)
        env = PlannerEnv(cfg)
        env.reset()
        assert isinstance(env._executor, DryRunExecutor), (
            "dry_run=True must use DryRunExecutor, not DockerEpisodeExecutor"
        )

    def test_real_mode_env_creates_docker_executor(self) -> None:
        """
        dry_run=False + mocked DockerEpisodeExecutor → _docker_executor is set.
        The DockerEpisodeExecutor's start_episode() is called with the repo's
        detected ecosystems.
        """
        from src.rl_env.env import PlannerEnv, EnvConfig
        from src.repo_scan.models import RepoManifest

        cfg = EnvConfig(corpus_manifest_path=None, dry_run=False, seed=0)
        env = PlannerEnv(cfg)

        # Mock the heavy dependencies so no real Docker or git is needed
        fake_manifest = RepoManifest()
        fake_manifest.ecosystems["python"] = __import__(
            "src.repo_scan.models", fromlist=["EcosystemManifest"]
        ).EcosystemManifest(ecosystem="python", manifest_files=["requirements.txt"])

        mock_docker_exec = MagicMock(spec=DockerEpisodeExecutor)
        mock_docker_exec.container_id = "fake-cid"

        with patch.object(env, "_prepare_repo",
                          return_value=(fake_manifest, "/fake/local/path")), \
             patch("src.rl_env.env.DockerEpisodeExecutor",
                   return_value=mock_docker_exec):
            env.reset(repo_entry={
                "repo": "owner/repo",
                "category": "manifest_present",
            })

        mock_docker_exec.start_episode.assert_called_once()
        call_kwargs = mock_docker_exec.start_episode.call_args
        assert call_kwargs.kwargs.get("repo_path") == "/fake/local/path" or \
               (call_kwargs.args and call_kwargs.args[0] == "/fake/local/path"), (
            "start_episode must receive the local clone path"
        )


# ---------------------------------------------------------------------------
# 9. Real Docker integration (requires daemon)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """
    Real end-to-end execution against a live Docker daemon.
    Run manually: pytest -v -m integration tests/test_rl_env/test_executor_lifecycle.py
    """

    def test_install_and_verify_round_trip(self, tmp_path) -> None:
        """
        Proves the complete stack works:
        1. Episode container starts with tmpfs /workspace_rw
        2. PYTHONUSERBASE=/workspace_rw is active
        3. A real 'pip install requests' succeeds
        4. The lookup-table verify command 'python -c "import requests"'
           succeeds in the same container.
        """
        # We don't need a real repo, just an empty directory to mount
        repo_dir = tmp_path / "dummy_repo"
        repo_dir.mkdir()
        
        executor = DockerEpisodeExecutor()
        
        # Create a network with the correct label so start_episode doesn't fail
        # (or just use "none" mode since pip can't reach the internet then? Wait,
        # we need internet for 'pip install requests'. So we must use bridge for the test).
        executor._config.network = NetworkConfig(mode="bridge")
        
        try:
            executor.start_episode(
                repo_path=str(repo_dir),
                repo_name="test_integration",
                ecosystems=["python"],
            )
            
            # Step 1: Install a tiny package that has no dependencies (e.g. 'certifi' or 'six' or 'requests')
            # Let's use 'six' since it's very small.
            step = AtomicStep(
                command="pip install --user six",
                description="install six",
                # Don't provide verify_command; verify_table.py will inject it!
            )
            
            result = executor.execute(step)
            
            assert result.status == ExecutionStatus.SUCCESS, (
                f"pip install failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            
            # The verify_command injected by verify_table.py should be: python -c 'import six'
            # And it should have executed and returned verified=True.
            assert result.verified is True, (
                f"verify command failed.\nSTDOUT:\n{result.verify_stdout}"
            )
            
        finally:
            executor.end_episode()

