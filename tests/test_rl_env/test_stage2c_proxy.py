"""Stage 2C: Tests for sandbox egress allowlisting proxy (unit and integration).

Covers:
- Proxy mode network flag (--network contai_sandbox) and proxy env vars.
- Fail-fast on 3 bad states:
  1. Docker network 'contai_sandbox' is missing.
  2. Docker network 'contai_sandbox' is not internal (Internal != true).
  3. Proxy container 'contai_egress_proxy' is not running.
- Mode 'none' is unchanged (default mode).
- Integration test (marked @pytest.mark.integration):
  (a) pip download --no-deps six -d /tmp/x succeeds;
  (b) urllib GET of https://example.com through the proxy fails (403 / tunnel refused);
  (c) with proxy env vars unset, direct connection to pypi.org:443 fails;
  (d) logs/egress_proxy/access.log contains allowed CONNECT to pypi/pythonhosted AND denied CONNECT to example.com.
"""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.rl_env.docker_executor import (
    DockerEpisodeExecutor,
    DockerExecutorConfig,
    NetworkConfig,
)


# ---------------------------------------------------------------------------
# Unit tests (subprocess mocked)
# ---------------------------------------------------------------------------

def _make_mock_subprocess_run(
    *,
    network_exists: bool = True,
    network_internal: bool = True,
    proxy_running: bool = True,
):
    """Helper to mock subprocess.run for DockerEpisodeExecutor."""
    def fake_run(cmd, *args, **kwargs):
        cmd_list = list(cmd)

        if cmd_list[:2] == ["docker", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, "Docker version 27.0.0\n", "")

        if cmd_list[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "[]\n", "")

        if cmd_list[:3] == ["docker", "network", "inspect"] and "contai_sandbox" in cmd_list:
            if not network_exists:
                return subprocess.CompletedProcess(cmd, 1, "", "Error: No such network: contai_sandbox\n")
            return subprocess.CompletedProcess(
                cmd,
                0,
                "true\n" if network_internal else "false\n",
                "",
            )

        if cmd_list[:2] == ["docker", "inspect"] and "contai_egress_proxy" in cmd_list:
            if not proxy_running:
                return subprocess.CompletedProcess(cmd, 1, "", "Error: No such container: contai_egress_proxy\n")
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")

        if cmd_list[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(cmd, 0, "mock_container_id_123\n", "")

        if cmd_list[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if cmd_list[:2] == ["docker", "stop"] or cmd_list[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")

        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def test_proxy_mode_emits_network_and_env_flags(tmp_path):
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    config = DockerExecutorConfig(network=NetworkConfig(mode="proxy"))
    executor = DockerEpisodeExecutor(config=config)

    fake_run = _make_mock_subprocess_run()
    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_run) as mock_run:
        executor.start_episode(str(repo_dir), repo_name="test/proxy_repo")

        run_calls = [
            c for c in mock_run.call_args_list
            if len(c[0][0]) > 2 and c[0][0][0] == "docker" and c[0][0][1] == "run"
        ]
        assert len(run_calls) == 1, "Expected exactly one docker run call"
        run_cmd = run_calls[0][0][0]

        # Check network flag
        assert "--network" in run_cmd
        net_idx = run_cmd.index("--network")
        assert run_cmd[net_idx + 1] == "contai_sandbox"

        # Check proxy env vars
        expected_envs = [
            "HTTP_PROXY=http://contai_egress_proxy:3128",
            "HTTPS_PROXY=http://contai_egress_proxy:3128",
            "http_proxy=http://contai_egress_proxy:3128",
            "https_proxy=http://contai_egress_proxy:3128",
            "PIP_PROXY=http://contai_egress_proxy:3128",
            "npm_config_https_proxy=http://contai_egress_proxy:3128",
        ]
        for env_var in expected_envs:
            assert "--env" in run_cmd
            assert env_var in run_cmd, f"Missing expected env var: {env_var}"

        executor.end_episode()


def test_proxy_mode_fails_fast_on_missing_network(tmp_path):
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    config = DockerExecutorConfig(network=NetworkConfig(mode="proxy"))
    executor = DockerEpisodeExecutor(config=config)

    fake_run = _make_mock_subprocess_run(network_exists=False)
    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match=r"contai_sandbox.*not found"):
            executor.start_episode(str(repo_dir), repo_name="test/proxy_repo")


def test_proxy_mode_fails_fast_on_non_internal_network(tmp_path):
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    config = DockerExecutorConfig(network=NetworkConfig(mode="proxy"))
    executor = DockerEpisodeExecutor(config=config)

    fake_run = _make_mock_subprocess_run(network_internal=False)
    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match=r"contai_sandbox.*not internal"):
            executor.start_episode(str(repo_dir), repo_name="test/proxy_repo")


def test_proxy_mode_fails_fast_on_proxy_container_not_running(tmp_path):
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    config = DockerExecutorConfig(network=NetworkConfig(mode="proxy"))
    executor = DockerEpisodeExecutor(config=config)

    fake_run = _make_mock_subprocess_run(proxy_running=False)
    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match=r"contai_egress_proxy.*not (found|running)"):
            executor.start_episode(str(repo_dir), repo_name="test/proxy_repo")


def test_none_mode_unchanged(tmp_path):
    repo_dir = tmp_path / "dummy_repo"
    repo_dir.mkdir()

    # Default NetworkConfig is mode="none"
    config = DockerExecutorConfig(network=NetworkConfig())
    assert config.network.mode == "none"
    executor = DockerEpisodeExecutor(config=config)

    fake_run = _make_mock_subprocess_run(network_exists=False, proxy_running=False)
    with patch("src.rl_env.docker_executor.subprocess.run", side_effect=fake_run) as mock_run:
        executor.start_episode(str(repo_dir), repo_name="test/none_repo")

        run_calls = [
            c for c in mock_run.call_args_list
            if len(c[0][0]) > 2 and c[0][0][0] == "docker" and c[0][0][1] == "run"
        ]
        assert len(run_calls) == 1
        run_cmd = run_calls[0][0][0]

        assert "--network" in run_cmd
        net_idx = run_cmd.index("--network")
        assert run_cmd[net_idx + 1] == "none"

        # Verify no proxy env vars were passed
        cmd_str = " ".join(run_cmd)
        assert "HTTP_PROXY" not in cmd_str
        assert "contai_egress_proxy" not in cmd_str

        executor.end_episode()


# ---------------------------------------------------------------------------
# Integration test (real Docker sandbox container)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_stage2c_proxy_integration(tmp_path):
    """Real container test verifying proxy egress restrictions.

    Checks:
    (a) pip download --no-deps six -d /tmp/x succeeds;
    (b) python -c urllib GET of https://example.com through the proxy fails (403 / tunnel refused);
    (c) with the proxy env vars unset, a direct connection to pypi.org:443 fails;
    (d) logs/egress_proxy/access.log contains an allowed CONNECT to pypi.org or files.pythonhosted.org
        AND a denied one to example.com.
    """
    res = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip("Docker daemon not running")

    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("# Proxy integration test\n")

    executor = DockerEpisodeExecutor(config=DockerExecutorConfig(network=NetworkConfig(mode="proxy")))

    log_path = Path("logs/egress_proxy/access.log")
    assert log_path.exists(), f"Access log file {log_path} must exist before running test"

    try:
        executor.start_episode(str(repo_dir), repo_name="test/proxy_integration")

        # (a) pip download --no-deps six -d /tmp/x succeeds
        pip_res = executor._run_docker_exec("pip download --no-deps six -d /tmp/x", timeout=60)
        assert pip_res.returncode == 0, (
            f"(a) pip download failed (rc={pip_res.returncode}):\n"
            f"stdout: {pip_res.stdout}\nstderr: {pip_res.stderr}"
        )

        # (b) urllib GET of https://example.com through the proxy fails (403 / tunnel refused)
        get_res = executor._run_docker_exec(
            "python -c \"import urllib.request; urllib.request.urlopen('https://example.com', timeout=10)\"",
            timeout=15,
        )
        assert get_res.returncode != 0, (
            f"(b) GET https://example.com succeeded unexpectedly:\n"
            f"stdout: {get_res.stdout}"
        )
        err_lower = (get_res.stderr + get_res.stdout).lower()
        assert any(term in err_lower for term in ["403", "tunnel", "forbidden", "refused"]), (
            f"(b) Expected 403 / tunnel error, got stderr: {get_res.stderr}"
        )

        # (c) with proxy env vars unset, direct connection to pypi.org:443 fails
        direct_cmd = (
            "env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u PIP_PROXY -u npm_config_https_proxy "
            "python -c \"import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)\""
        )
        direct_res = executor._run_docker_exec(direct_cmd, timeout=15)
        assert direct_res.returncode != 0, (
            f"(c) Direct connection to pypi.org:443 succeeded unexpectedly without proxy:\n"
            f"stdout: {direct_res.stdout}"
        )

        # (d) logs/egress_proxy/access.log contains allowed CONNECT to pypi/pythonhosted AND denied to example.com
        time.sleep(1)  # allow squid to flush log entry
        access_log_text = log_path.read_text(encoding="utf-8", errors="replace")

        has_allowed_pypi = (
            ("CONNECT pypi.org:443" in access_log_text or "CONNECT files.pythonhosted.org:443" in access_log_text)
            and ("200" in access_log_text or "TCP_TUNNEL" in access_log_text)
        )
        assert has_allowed_pypi, f"(d) Missing allowed CONNECT to pypi/pythonhosted in access.log:\n{access_log_text}"

        has_denied_example = (
            "CONNECT example.com:443" in access_log_text
            and ("403" in access_log_text or "TCP_DENIED" in access_log_text)
        )
        assert has_denied_example, f"(d) Missing denied CONNECT to example.com in access.log:\n{access_log_text}"

    finally:
        executor.end_episode()
