import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.rl_env.docker_executor import DockerEpisodeExecutor, DockerExecutorConfig
from src.rl_env.env import PlannerEnv, EnvConfig, EnvForkError
from src.rl_env.parser import (
    parse_planned_step,
    MissingFieldError,
    InvalidFieldError,
    ParseError,
)
from src.sandbox.models import RiskTier, ExecutionStatus, AtomicStep
from src.system2_planner.models import PlannedStep, ActionType
from scripts.train_system2_grpo import (
    evaluate_completion,
    CompletionResult,
    compute_group_advantages,
    masked_token_loss,
    select_best_action,
    FailFastTracker,
    write_emergency_abort,
)
from scripts.evaluate_system2 import evaluate_model


# ============================================================================
# A. start_episode atomicity tests (mock subprocess.run)
# ============================================================================

def test_start_episode_run_timeout_calls_rm_and_raises():
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run) as mock_run:
        with pytest.raises(RuntimeError, match="docker run timed out after 60s"):
            executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    stored_name = executor._container_name
    assert stored_name is not None
    assert stored_name.startswith("rl_ep_owner_repo_")
    rm_calls = [c for c in mock_run.call_args_list if c[0][0][:3] == ["docker", "rm", "-f"]]
    assert len(rm_calls) >= 1
    assert rm_calls[0][0][0][3] == stored_name


def test_start_episode_run_failure_calls_rm_and_raises():
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="docker run failed")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run) as mock_run:
        with pytest.raises(RuntimeError, match="docker run failed for image"):
            executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    stored_name = executor._container_name
    assert stored_name is not None
    rm_calls = [c for c in mock_run.call_args_list if c[0][0][:3] == ["docker", "rm", "-f"]]
    assert len(rm_calls) >= 1
    assert rm_calls[0][0][0][3] == stored_name


def test_start_episode_mkdir_failure_calls_end_episode_and_raises():
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="cid12345\n", stderr="")
        if len(cmd) > 2 and cmd[0] == "docker" and cmd[1] == "exec":
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="mkdir permission denied")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run) as mock_run:
        with pytest.raises(RuntimeError, match="Failed to create /workspace_rw"):
            executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    stop_rm_calls = [c for c in mock_run.call_args_list if c[0][0][:2] in (["docker", "stop"], ["docker", "rm"])]
    assert len(stop_rm_calls) >= 1


def test_start_episode_happy_path():
    executor = DockerEpisodeExecutor()
    executor._check_docker_available = MagicMock()
    executor._pull_image_if_needed = MagicMock()

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="cid_success\n", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_subprocess_run):
        executor.start_episode(repo_path="/fake/repo", repo_name="owner/repo", ecosystems=["python"])

    assert executor._container_id == "cid_success"
    assert executor._container_name.startswith("rl_ep_owner_repo_")


def test_env_make_executor_clears_old_docker_executor():
    env = PlannerEnv(EnvConfig(dry_run=True))
    old_mock = MagicMock()
    env._docker_executor = old_mock

    with patch.object(PlannerEnv, "_prepare_repo", return_value=(MagicMock(), "/fake/path")):
        with patch("src.rl_env.env.DockerEpisodeExecutor") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            env._cfg.dry_run = False
            created = env._make_executor({"repo": "test/repo"}, MagicMock(), "/fake/path")
            old_mock.end_episode.assert_called_once()
            assert env._docker_executor is created
            assert env._docker_executor is not old_mock


# ============================================================================
# B. Infrastructure detection in _run_docker_exec via inspect
# ============================================================================

def test_docker_exec_rc1_inspect_true_returns_failed_result():
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "No such container: simulated compiler error\n")
    mock_proc.returncode = 1

    insp_completed = subprocess.CompletedProcess(
        args=["docker", "inspect", "-f", "{{.State.Running}}", "test_cid"],
        returncode=0,
        stdout="true\n",
        stderr=""
    )

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("subprocess.run", return_value=insp_completed) as mock_run:
            step = AtomicStep(command="pip install requests", description="Install")
            res = executor.execute(step)

    assert res.status == ExecutionStatus.FAILED
    assert res.returncode == 1
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0][:4] == ["docker", "inspect", "-f", "{{.State.Running}}"]


def test_docker_exec_rc1_inspect_false_raises():
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "Process killed\n")
    mock_proc.returncode = 1

    insp_completed = subprocess.CompletedProcess(
        args=["docker", "inspect", "-f", "{{.State.Running}}", "test_cid"],
        returncode=0,
        stdout="false\n",
        stderr=""
    )

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("subprocess.run", return_value=insp_completed):
            step = AtomicStep(command="pip install requests", description="Install")
            with pytest.raises(RuntimeError, match="is not running"):
                executor.execute(step)


def test_docker_exec_rc1_inspect_rc_nonzero_raises():
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "Failed\n")
    mock_proc.returncode = 1

    insp_completed = subprocess.CompletedProcess(
        args=["docker", "inspect", "-f", "{{.State.Running}}", "test_cid"],
        returncode=1,
        stdout="",
        stderr="Error: No such object: test_cid"
    )

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("subprocess.run", return_value=insp_completed):
            step = AtomicStep(command="pip install requests", description="Install")
            with pytest.raises(RuntimeError, match="is not running"):
                executor.execute(step)


def test_docker_exec_rc0_no_inspect_call():
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("success\n", "")
    mock_proc.returncode = 0

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("subprocess.run") as mock_run:
            step = AtomicStep(command="pip install requests", description="Install")
            res = executor.execute(step)

    assert res.status == ExecutionStatus.SUCCESS
    assert res.returncode == 0
    mock_run.assert_not_called()


def test_docker_exec_rc_126_and_127_treated_as_normal_failed():
    executor = DockerEpisodeExecutor()
    executor._container_id = "test_cid"

    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "/bin/sh: command not found\n")
    mock_proc.returncode = 127

    insp_completed = subprocess.CompletedProcess(
        args=["docker", "inspect", "-f", "{{.State.Running}}", "test_cid"],
        returncode=0,
        stdout="true\n",
        stderr=""
    )

    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("subprocess.run", return_value=insp_completed):
            step = AtomicStep(command="pip install requests", description="Test 127")
            res = executor.execute(step)

    assert res.status == ExecutionStatus.FAILED
    assert res.returncode == 127
    assert res.verified is False


# ============================================================================
# C. Parser / evaluate_completion
# ============================================================================

@pytest.mark.parametrize("bad_atype", [123, ["CHECK"], {"act": "CHECK"}, True])
def test_parser_non_string_action_type_raises_invalid_field_error(bad_atype):
    comp = json.dumps({"action_type": bad_atype, "target": "requests", "description": "test"})
    with pytest.raises(InvalidFieldError, match="'action_type' field must be a string"):
        parse_planned_step(comp)


@pytest.mark.parametrize("empty_atype", ["", "   ", "\t\n"])
def test_parser_empty_whitespace_action_type_raises_missing_field_error(empty_atype):
    comp = json.dumps({"action_type": empty_atype, "target": "requests", "description": "test"})
    with pytest.raises(MissingFieldError, match="Empty or whitespace 'action_type' field"):
        parse_planned_step(comp)


def test_evaluate_completion_unexpected_parser_exception_maps_to_env_error():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()
    comp = json.dumps({"action_type": "CHECK", "target": "python", "description": "test"})

    with patch("scripts.train_system2_grpo.parse_planned_step", side_effect=TypeError("Unexpected parser internal type error")):
        res: CompletionResult = evaluate_completion(env, comp, snapshot_tag=None)

    assert res.status == "ENV_ERROR"
    assert res.reward is None
    assert "Parser harness error" in res.info["execution_error"]
    assert "TypeError" in res.info["execution_error"]


# ============================================================================
# D. fork() on completed environment
# ============================================================================

def test_fork_on_done_env_raises_env_fork_error():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()
    env._done = True
    with pytest.raises(EnvForkError, match="Cannot fork an environment that has finished"):
        env.fork()


# ============================================================================
# E. Advantages, masking, and best-of-G selection functions
# ============================================================================

def test_compute_group_advantages_rewards_masking():
    # Rewards: [1.0, None (ENV_ERROR), -1.0, None (ENV_ERROR)]
    results = [
        CompletionResult(action=None, info={}, reward=1.0, status="OK"),
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
        CompletionResult(action=None, info={}, reward=-1.0, status="PARSE_ERROR"),
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
    ]

    res = compute_group_advantages(results, G=4)
    assert res is not None
    advantages, valid_mask = res

    assert advantages.shape == (4,)
    assert valid_mask.shape == (4, 1)

    # Valid mask is 1.0 for valid rows (0 and 2), 0.0 for ENV_ERROR rows (1 and 3)
    assert valid_mask[0, 0].item() == 1.0
    assert valid_mask[1, 0].item() == 0.0
    assert valid_mask[2, 0].item() == 1.0
    assert valid_mask[3, 0].item() == 0.0

    # Advantages computed only over the 2 valid values: 1.0 (positive) and -1.0 (negative)
    assert advantages[0].item() > 0.0
    assert advantages[2].item() < 0.0
    assert advantages[1].item() == 0.0
    assert advantages[3].item() == 0.0


def test_compute_group_advantages_flat_std_gives_zero_advantages():
    results = [
        CompletionResult(action=None, info={}, reward=1.0, status="OK"),
        CompletionResult(action=None, info={}, reward=1.0, status="OK"),
    ]
    res = compute_group_advantages(results, G=2)
    assert res is not None
    advantages, valid_mask = res
    assert torch.all(advantages == 0.0)
    assert torch.all(valid_mask == 1.0)


def test_compute_group_advantages_fewer_than_2_valid_returns_none():
    results = [
        CompletionResult(action=None, info={}, reward=1.0, status="OK"),
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
    ]
    assert compute_group_advantages(results, G=3) is None


def test_compute_group_advantages_raises_on_none_reward_for_non_env_error():
    results = [
        CompletionResult(action=None, info={}, reward=1.0, status="OK"),
        CompletionResult(action=None, info={}, reward=None, status="PARSE_ERROR"),
    ]
    with pytest.raises(ValueError) as exc_info:
        compute_group_advantages(results, G=2)
    assert "1" in str(exc_info.value)
    assert "PARSE_ERROR" in str(exc_info.value)
    assert "None" in str(exc_info.value)


def test_masked_token_loss():
    G, seq_len = 4, 5
    per_token_pg_loss = torch.ones(G, seq_len)
    per_token_kl = torch.zeros(G, seq_len)
    pad_mask = torch.ones(G, seq_len)
    valid_mask = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    loss, normalizer = masked_token_loss(per_token_pg_loss, per_token_kl, pad_mask, valid_mask, beta=0.1)

    # Normalizer sum must equal tokens from valid rows only (2 rows * 5 = 10)
    assert normalizer.item() == 10.0
    assert torch.isclose(loss, torch.tensor(1.0))


def test_select_best_action_zero_ok_returns_none():
    results = [
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
        CompletionResult(action=PlannedStep(action_type=ActionType.INSTALL, description="p"), info={}, reward=-1.0, status="PARSE_ERROR"),
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
    ]
    assert select_best_action(results) is None


def test_select_best_action_picks_highest_reward_among_ok():
    act1 = PlannedStep(action_type=ActionType.CHECK, description="low reward")
    act2 = PlannedStep(action_type=ActionType.INSTALL, description="high reward")
    results = [
        CompletionResult(action=None, info={}, reward=None, status="ENV_ERROR"),
        CompletionResult(action=act1, info={}, reward=0.2, status="OK"),
        CompletionResult(action=act2, info={}, reward=1.5, status="OK"),
        CompletionResult(action=None, info={}, reward=5.0, status="PARSE_ERROR"),
    ]
    best = select_best_action(results)
    assert best is not None
    best_idx, best_reward, best_action = best
    assert best_idx == 2
    assert best_reward == 1.5
    assert best_action == act2


# ============================================================================
# F. FailFastTracker and emergency checkpoint saving
# ============================================================================

def test_fail_fast_tracker_aborts_on_3rd_consecutive():
    tracker = FailFastTracker(max_steps=10, G=4)
    tracker.record("ENV_ERROR", "tb1")
    assert not tracker.should_abort
    tracker.record("ENV_ERROR", "tb2")
    assert not tracker.should_abort
    tracker.record("ENV_ERROR", "tb3")
    assert tracker.should_abort
    assert "3 consecutive completions are ENV_ERROR" in tracker.reason
    assert tracker.consecutive_env_errors == 3


def test_fail_fast_tracker_aborts_on_exceeding_10_percent():
    # max_steps=10, G=4 -> threshold is 4.0
    tracker = FailFastTracker(max_steps=10, G=4)

    # 4 ENV_ERRORs interleaved with OK (so consecutive < 3)
    for _ in range(4):
        tracker.record("ENV_ERROR", "tb")
        tracker.record("OK")

    assert not tracker.should_abort
    assert tracker.episode_env_errors == 4

    # 5th ENV_ERROR exceeds threshold of 4.0
    tracker.record("ENV_ERROR", "tb5")
    assert tracker.should_abort
    assert "10% threshold" in tracker.reason


def test_write_emergency_abort_records_checkpoint_save_failure(tmp_path):
    mock_model = MagicMock()
    mock_model.save_pretrained.side_effect = RuntimeError("Disk quota exceeded during emergency save")

    tracker = FailFastTracker(max_steps=10, G=4)
    tracker.record("ENV_ERROR", "Docker daemon unreachable")
    tracker.record("ENV_ERROR", "Docker daemon unreachable")
    tracker.record("ENV_ERROR", "Docker daemon unreachable")
    assert tracker.should_abort

    emergency_dir = write_emergency_abort(tmp_path, mock_model, None, None, tracker, "Docker daemon unreachable")
    assert emergency_dir == tmp_path / "emergency_ckpt"

    reason_file = emergency_dir / "reason.txt"
    assert reason_file.exists()
    content = reason_file.read_text(encoding="utf-8")

    assert "3 consecutive completions are ENV_ERROR" in content
    assert "Docker daemon unreachable" in content
    assert "CHECKPOINT SAVE FAILED" in content
    assert "Disk quota exceeded during emergency save" in content


# ============================================================================
# I. evaluate_system2.py env_error separation and denominator exclusion
# ============================================================================

class MockBatchEncoding(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(item)

    def to(self, device):
        return self


def test_evaluate_system2_env_reset_error_separation():
    test_repos = [
        {"repo": "repo/success1", "category": "python"},
        {"repo": "repo/fail_reset", "category": "python"},
        {"repo": "repo/success2", "category": "node"},
    ]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"
    mock_tokenizer.return_value = MockBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_tokenizer.batch_decode.return_value = ['{"action_type": "CHECK", "description": "check"}']
    mock_tokenizer.decode.return_value = '{"action_type": "CHECK", "description": "check"}'

    def fake_reset(repo_entry=None, **kwargs):
        if repo_entry["repo"] == "repo/fail_reset":
            raise RuntimeError("Docker daemon connection failed during reset")
        obs_mock = MagicMock()
        obs_mock.canonical_intent.intent_text = "test intent"
        return obs_mock, {}

    def fake_step(action):
        obs_mock = MagicMock()
        return obs_mock, 1.0, True, False, {"episode_complete": True, "reward_breakdown": {"execution": 1.0}}

    mock_env = MagicMock()
    mock_env.reset.side_effect = fake_reset
    mock_env.step.side_effect = fake_step
    mock_env._serializer.to_chat_prompt.return_value = []

    with patch("scripts.evaluate_system2.PlannerEnv", return_value=mock_env):
        with patch("scripts.evaluate_system2.extract_step_dict", return_value={"action_type": "CHECK", "description": "check"}):
            metrics = evaluate_model(
                model_name="TestModel",
                model=mock_model,
                tokenizer=mock_tokenizer,
                test_repos=test_repos,
                demo_repos=[],
                out_dir=Path("outputs/test_eval_metrics")
            )

    # Reset failure was counted under env_errors
    assert metrics["env_errors"] == 1
    # Reset failure was excluded from execution_failed
    assert metrics["failures"]["execution_failed"] == 0
    # Reset failure was excluded from total_episodes (2 succeeded resets out of 3 repos)
    assert metrics["total_episodes"] == 2
    # Reset failure was excluded from category total
    assert metrics["by_category"]["python"]["total"] == 1
    assert metrics["by_category"]["node"]["total"] == 1
    assert metrics["full_completion"] == 2


def test_evaluate_system2_step_error_preserves_global_metrics():
    # Step 1 OK, step 2 raises -> global reward/steps/valid_structure unchanged, env_errors == 1
    test_repos = [
        {"repo": "repo/step_fail", "category": "python"},
    ]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"
    mock_tokenizer.return_value = MockBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_tokenizer.batch_decode.return_value = ['{"action_type": "CHECK", "description": "check"}']
    mock_tokenizer.decode.return_value = '{"action_type": "CHECK", "description": "check"}'

    step_count = 0
    def fake_step(action):
        nonlocal step_count
        step_count += 1
        if step_count == 1:
            obs_mock = MagicMock()
            return obs_mock, 1.0, False, False, {"episode_complete": False, "reward_breakdown": {"execution": 1.0}}
        raise RuntimeError("Container crashed during step 2")

    mock_env = MagicMock()
    obs_mock = MagicMock()
    obs_mock.canonical_intent.intent_text = "test intent"
    mock_env.reset.return_value = (obs_mock, {})
    mock_env.step.side_effect = fake_step
    mock_env._serializer.to_chat_prompt.return_value = []

    with patch("scripts.evaluate_system2.PlannerEnv", return_value=mock_env):
        with patch("scripts.evaluate_system2.extract_step_dict", return_value={"action_type": "CHECK", "description": "check"}):
            metrics = evaluate_model(
                model_name="TestModel",
                model=mock_model,
                tokenizer=mock_tokenizer,
                test_repos=test_repos,
                demo_repos=[],
                out_dir=Path("outputs/test_eval_metrics")
            )

    # Step error: env_errors must be 1, and no local metrics committed to global
    assert metrics["env_errors"] == 1
    assert metrics["total_episodes"] == 0
    assert metrics["valid_structure"] == 0
    assert metrics["valid_json"] == 0
    assert metrics["execution_success"] == 0
    assert metrics["total_steps"] == 0
    assert metrics["full_completion"] == 0
    assert metrics["failures"]["execution_failed"] == 0
