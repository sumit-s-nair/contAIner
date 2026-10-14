import json
from unittest.mock import MagicMock, patch
import pytest

from scripts.train_system2_grpo import evaluate_completion, CompletionResult
from src.rl_env.env import PlannerEnv, EnvConfig, EnvForkError
from src.rl_env.parser import (
    parse_planned_step,
    MissingFieldError,
    InvalidFieldError,
    UnrecognizedActionTypeError,
)
from src.sandbox.models import RiskTier, ExecutionResult, ExecutionStatus, AtomicStep
from src.system2_planner.models import PlannedStep, ActionType


def test_fork_unreset_raises_env_fork_error():
    env = PlannerEnv(EnvConfig(dry_run=True))
    with pytest.raises(EnvForkError, match="Cannot fork an environment that has not been reset"):
        env.fork()


def test_fork_state_independence():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()

    step_1 = PlannedStep(
        action_type=ActionType.CHECK,
        target="python",
        description="Check python",
    )
    env.step(step_1)

    parent_plan_len = len(env._plan_so_far)
    parent_hist_len = len(env._execution_history)
    parent_obs_len = len(env._obs.episode_steps)
    parent_step_idx = env._obs.step_index

    # Fork the environment
    child = env.fork()

    # Step the child only
    step_2 = PlannedStep(
        action_type=ActionType.INSTALL,
        target="requests",
        description="Install requests",
    )
    child.step(step_2)

    # Child has progressed
    assert len(child._plan_so_far) == parent_plan_len + 1
    assert len(child._execution_history) == parent_hist_len + 1
    assert len(child._obs.episode_steps) == parent_obs_len + 1
    assert child._obs.step_index == parent_step_idx + 1

    # Parent state is strictly unchanged
    assert len(env._plan_so_far) == parent_plan_len
    assert len(env._execution_history) == parent_hist_len
    assert len(env._obs.episode_steps) == parent_obs_len
    assert env._obs.step_index == parent_step_idx


def test_dry_run_fork_replay_reproduces_history():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset(repo_entry={"repo": "test/repo", "category": "known_conflict"})

    step_1 = PlannedStep(
        action_type=ActionType.ISOLATE,
        target="venv",
        description="Isolate venv",
    )
    step_2 = PlannedStep(
        action_type=ActionType.CHECK,
        target="python",
        description="Check python",
    )
    env.step(step_1)
    env.step(step_2)

    parent_statuses = [e.status for e in env._execution_history]

    # Fork with snapshot_image=None triggers fresh executor + replay
    child = env.fork(snapshot_image=None)

    child_statuses = [e.status for e in child._execution_history]
    assert child_statuses == parent_statuses
    assert len(child._plan_so_far) == len(env._plan_so_far)
    assert child._obs.step_index == env._obs.step_index


def test_evaluate_completion_forced_attribute_error():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()

    comp = json.dumps({
        "action_type": "CHECK",
        "target": "python",
        "description": "Check python environment",
        "rationale": "Verify interpreter",
    })

    with patch.object(
        PlannerEnv,
        "step",
        side_effect=AttributeError("'NoneType' object has no attribute 'start_episode'")
    ):
        res: CompletionResult = evaluate_completion(env, comp, snapshot_tag=None)

    assert res.status == "ENV_ERROR"
    assert res.reward is None
    assert "execution_error" in res.info
    assert "AttributeError" in res.info["execution_error"]
    assert "start_episode" in res.info["execution_error"]
    assert "Traceback" in res.info["execution_error"]


def test_evaluate_completion_malformed_json():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()

    malformed_comp = '{"action_type": "BUILD", "target": unquoted_value}'
    res: CompletionResult = evaluate_completion(env, malformed_comp, snapshot_tag=None)

    assert res.status == "PARSE_ERROR"
    assert res.reward == -1.0
    assert res.info["subtype"] == "JSON_DECODE_ERROR"
    assert res.info.get("exec_status") == "JSON_DECODE_ERROR"
    assert res.info.get("exec_status") != "UNRECOGNIZED_ACTION_TYPE"


def test_parser_missing_fields_and_casing():
    # Missing action_type
    with pytest.raises(MissingFieldError):
        parse_planned_step('{"target": "requests", "description": "Install"}')

    # Null action_type
    with pytest.raises(MissingFieldError):
        parse_planned_step('{"action_type": null, "target": "requests"}')

    # Missing target
    with pytest.raises(MissingFieldError):
        parse_planned_step('{"action_type": "BUILD", "description": "Build repo"}')

    # Null target
    with pytest.raises(MissingFieldError):
        parse_planned_step('{"action_type": "BUILD", "target": null}')

    # Lowercase action_type parses
    step_lower = parse_planned_step('{"action_type": "install", "target": "requests"}')
    assert step_lower.action_type == ActionType.INSTALL
    assert step_lower.target == "requests"
    assert step_lower.description == ""
    assert step_lower.rationale == ""

    # Uppercase action_type parses
    step_upper = parse_planned_step('{"action_type": "BUILD", "target": "my_target"}')
    assert step_upper.action_type == ActionType.BUILD
    assert step_upper.target == "my_target"

    # Unknown action_type raises UnrecognizedActionTypeError
    with pytest.raises(UnrecognizedActionTypeError):
        parse_planned_step('{"action_type": "NON_EXISTENT_ACTION", "target": "my_target"}')


def test_step_info_contains_stderr_tail_in_dry_run():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()

    # Step 1: Check denied step has stderr_tail
    step_check = PlannedStep(
        action_type=ActionType.CHECK,
        target="python",
        description="Check python",
    )
    _, _, _, _, info_denied = env.step(step_check)

    assert "stdout_tail" in info_denied
    assert "stderr_tail" in info_denied
    assert "returncode" in info_denied
    assert "REVIEW step auto-denied" in info_denied["stderr_tail"]

    # Step 2: SAFE verified step has verify_stdout_tail and verify_returncode where present
    step_install = PlannedStep(
        action_type=ActionType.INSTALL,
        target="requests",
        description="Install requests",
    )
    _, _, _, _, info_safe = env.step(step_install)

    assert "stdout_tail" in info_safe
    assert "stderr_tail" in info_safe
    assert "returncode" in info_safe
    assert "verify_stdout_tail" in info_safe
    assert "verify_returncode" in info_safe

    assert isinstance(info_safe["stdout_tail"], str)
    assert isinstance(info_safe["stderr_tail"], str)
    assert info_safe["returncode"] == 0
    assert info_safe["verify_returncode"] == 0
    assert "Synthetic VERIFY OK" in info_safe["verify_stdout_tail"]


def test_fork_replay_mismatch_calls_end_episode_and_sets_cause():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset(repo_entry={"repo": "test/repo", "category": "known_conflict"})
    step_isolate = PlannedStep(
        action_type=ActionType.ISOLATE,
        target="venv",
        description="Isolate venv",
    )
    env.step(step_isolate)

    mock_executor = MagicMock()
    mock_executor.execute.return_value = ExecutionResult(
        step=AtomicStep(command="python -m venv .venv", description="Isolate"),
        status=ExecutionStatus.FAILED,
        tier=RiskTier.SAFE,
        verified=False,
        returncode=1,
    )

    with patch.object(PlannerEnv, "_make_executor", return_value=mock_executor):
        with pytest.raises(EnvForkError) as exc_info:
            env.fork()

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "Replayed execution history does not match parent history" in str(exc_info.value.__cause__)
    mock_executor.end_episode.assert_called_once()


def test_fork_with_none_repo_entry_raises_env_fork_error():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()
    env._repo_entry = None
    with pytest.raises(EnvForkError, match="Cannot fork an environment with _repo_entry=None"):
        env.fork()


def test_fork_replay_mismatch_on_verified_alone():
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset(repo_entry={"repo": "test/repo", "category": "known_conflict"})
    step_isolate = PlannedStep(
        action_type=ActionType.ISOLATE,
        target="venv",
        description="Isolate venv",
    )
    env.step(step_isolate)

    # In parent, status=SUCCESS and verified=True
    parent_res = env._execution_history[0]
    assert parent_res.status == ExecutionStatus.SUCCESS
    assert parent_res.verified is True

    mock_executor = MagicMock()
    # status is SUCCESS, tier is SAFE, but verified is False
    mock_executor.execute.return_value = ExecutionResult(
        step=AtomicStep(command="python -m venv .venv", description="Isolate"),
        status=ExecutionStatus.SUCCESS,
        tier=RiskTier.SAFE,
        verified=False,
        returncode=0,
    )

    with patch.object(PlannerEnv, "_make_executor", return_value=mock_executor):
        with pytest.raises(EnvForkError) as exc_info:
            env.fork()

    cause = exc_info.value.__cause__
    assert cause is not None
    assert "at step 0" in str(cause)
    assert "parent=(status=ExecutionStatus.SUCCESS, verified=True, tier=RiskTier.SAFE)" in str(cause)
    assert "replayed=(status=ExecutionStatus.SUCCESS, verified=False, tier=RiskTier.SAFE)" in str(cause)


@pytest.mark.parametrize("bad_target", [
    ["requests", "urllib3"],
    {"pkg": "requests"},
    123,
    True,
])
def test_parser_non_string_target_gives_invalid_field(bad_target):
    env = PlannerEnv(EnvConfig(dry_run=True))
    env.reset()
    comp = json.dumps({
        "action_type": "INSTALL",
        "target": bad_target,
        "description": "Install package",
    })
    res: CompletionResult = evaluate_completion(env, comp, snapshot_tag=None)
    assert res.status == "PARSE_ERROR"
    assert res.reward == -1.0
    assert res.info["subtype"] == "INVALID_FIELD"
    assert res.info["exec_status"] == "INVALID_FIELD"
    assert "target" in res.info["execution_error"]


def test_parser_empty_string_target_allowed():
    comp = json.dumps({
        "action_type": "CHECK",
        "target": "",
        "description": "Check with empty target",
    })
    step = parse_planned_step(comp)
    assert step.target == ""


def test_container_names_unique_across_100_rapid_calls():
    from src.rl_env.docker_executor import DockerEpisodeExecutor
    executor = DockerEpisodeExecutor()
    executor._repo_name = "test/repo"
    names = [executor._make_container_name() for _ in range(100)]
    assert len(names) == 100
    assert len(set(names)) == 100
    for name in names:
        assert name.startswith("rl_ep_test_repo_")
        assert len(name.split("_")[-1]) == 8

