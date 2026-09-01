"""
tests/test_rl_env/test_smoke.py
================================
Smoke test: 5 dry-run episodes with a fixed dummy policy.

No real model, no real subprocess execution, no network I/O.
Uses PlannerEnv with its synthetic fixture corpus (no corpus manifest).

What is tested
--------------
1. Observation serializes to valid JSON without error.
2. step() returns the correct 5-tuple (obs, reward, terminated, truncated, info).
3. Episode terminates correctly (terminated=True when all steps verified).
4. Reward breakdown is logged per step.
5. info dict contains expected keys.
6. Episode resets cleanly (reset() clears episode history).

Dummy policy
-------------
A ``FixedDummyPolicy`` that cycles through the template's expected action
types in order (ISOLATE → INSTALL for conflict episodes, INSTALL only for
clean episodes, etc.).  Each action is an ``AtomicStep`` with a description
that matches the ActionType keyword so the validator sees a valid plan.

DryRunExecutor behavior in smoke tests
----------------------------------------
- SAFE steps → synthetic SUCCESS + verified=True (no subprocess).
- BLOCKED steps → BLOCKED + hard penalty (even in dry_run mode).
- REVIEW steps → ABORTED (AUTO_DENY policy).
"""

from __future__ import annotations

import json
from typing import Iterator, List

import pytest

from src.sandbox.models import AtomicStep
from src.system2_planner.models import ActionType, TemplateInstance
from src.system2_planner.templates import TEMPLATES
from src.rl_env.env import PlannerEnv, EnvConfig
from src.rl_env.dry_run_executor import TrainingReviewPolicy
from src.rl_env.observation import ObservationSerializer
from src.rl_env.reward import RewardConfig


# ---------------------------------------------------------------------------
# Dummy policy
# ---------------------------------------------------------------------------

# Map ActionType → a step description that matches the validator's keywords
_ACTION_DESCRIPTIONS = {
    ActionType.ISOLATE:  "isolate the environment (create venv)",
    ActionType.INSTALL:  "install all dependencies",
    ActionType.CHECK:    "check environment requirements",
    ActionType.UPDATE:   "update the component",
    ActionType.DETECT:   "detect the missing or broken component",
    ActionType.FIX:      "fix the detected issue",
    ActionType.ESCALATE: "escalate the issue to the user",
    ActionType.BUILD:    "build the project",
}

_ACTION_COMMANDS = {
    ActionType.ISOLATE:  "python -m venv .venv",
    ActionType.INSTALL:  "pip install -r requirements.txt",
    ActionType.CHECK:    "pip list",
    ActionType.UPDATE:   "npm update",
    ActionType.DETECT:   "pip check",
    ActionType.FIX:      "pip install --upgrade pip",
    ActionType.ESCALATE: "echo 'manual intervention required'",
    ActionType.BUILD:    "cargo build",
}

_ACTION_VERIFY_COMMANDS = {
    ActionType.ISOLATE:  "test -f .venv/bin/python",
    ActionType.INSTALL:  "pip list",
    ActionType.CHECK:    "pip list",
    ActionType.UPDATE:   "npm list",
    ActionType.DETECT:   "pip check",
    ActionType.FIX:      "pip --version",
    ActionType.ESCALATE: "true",
    ActionType.BUILD:    "true",
}


class FixedDummyPolicy:
    """
    Dummy policy that produces the template's expected actions in order.

    This is NOT a model — it's a deterministic oracle used only for
    smoke testing that the environment loop works end-to-end.
    """

    def __init__(self, instance: TemplateInstance) -> None:
        self._actions = instance.get_expected_actions()
        self._cursor  = 0

    def next_action(self) -> AtomicStep:
        if self._cursor >= len(self._actions):
            # If the policy has exhausted its planned actions, emit a benign INSTALL
            return AtomicStep(
                command="pip list",
                description="check installed packages",
                verify_command="pip list",
            )
        action_node = self._actions[self._cursor]
        self._cursor += 1
        atype = action_node.action_type
        return AtomicStep(
            command=_ACTION_COMMANDS.get(atype, "echo done"),
            description=_ACTION_DESCRIPTIONS.get(atype, "unknown action"),
            verify_command=_ACTION_VERIFY_COMMANDS.get(atype, "true"),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> PlannerEnv:
    """A PlannerEnv in dry_run mode with default reward config."""
    cfg = EnvConfig(
        corpus_manifest_path=None,    # use synthetic fixtures
        dry_run=True,
        review_policy=TrainingReviewPolicy.AUTO_DENY,
        max_steps=8,
        seed=42,
    )
    return PlannerEnv(config=cfg)


# ---------------------------------------------------------------------------
# Episode runner helper
# ---------------------------------------------------------------------------

def run_episode(env: PlannerEnv, repo_entry: dict) -> dict:
    """
    Run one full episode with the FixedDummyPolicy.

    Returns a summary dict with per-step info for assertions.
    """
    obs, info = env.reset(repo_entry=repo_entry)
    assert obs is not None, "reset() returned None observation"

    # Verify observation serializes to valid JSON
    serializer = ObservationSerializer()
    ctx = serializer.to_context_string(obs)
    parsed = json.loads(ctx)
    assert "intent" in parsed
    assert "ecosystem_summary" in parsed
    assert "episode_history" in parsed

    # Build policy from the instance
    instance = env._instance
    assert instance is not None
    policy = FixedDummyPolicy(instance)

    step_summaries = []
    step_idx = 0

    while True:
        action = policy.next_action()
        result = env.step(action)

        # Validate 5-tuple structure
        assert len(result) == 5, f"step() must return 5-tuple, got {len(result)}"
        new_obs, reward, terminated, truncated, step_info = result

        assert isinstance(reward, (int, float)), f"reward must be numeric, got {type(reward)}"
        assert isinstance(terminated, bool), f"terminated must be bool"
        assert isinstance(truncated, bool), f"truncated must be bool"
        assert isinstance(step_info, dict), f"info must be dict"

        # Check required info keys
        required_keys = {
            "reward_breakdown", "exec_status", "exec_tier",
            "verified", "blocked", "episode_complete", "step_index"
        }
        missing = required_keys - set(step_info.keys())
        assert not missing, f"Missing info keys: {missing}"

        # reward_breakdown must have sub-terms
        breakdown = step_info["reward_breakdown"]
        for term in ("total", "validator", "execution", "step_penalty"):
            assert term in breakdown, f"Missing breakdown term: {term}"

        step_summaries.append({
            "step_idx":    step_idx,
            "action_desc": action.description,
            "reward":      reward,
            "breakdown":   breakdown,
            "terminated":  terminated,
            "truncated":   truncated,
            "exec_status": step_info["exec_status"],
            "verified":    step_info["verified"],
        })

        # Verify observation updated correctly
        assert new_obs.step_index == step_idx + 1, (
            f"step_index should be {step_idx + 1}, got {new_obs.step_index}"
        )
        assert len(new_obs.episode_steps) == step_idx + 1, (
            f"episode_steps should have {step_idx + 1} entries"
        )

        step_idx += 1
        if terminated or truncated:
            break

    return {
        "info":     info,
        "steps":    step_summaries,
        "terminated": terminated,
        "truncated":  truncated,
    }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

EPISODE_FIXTURES = [
    {"repo": "fixture/manifest_present",  "category": "manifest_present"},
    {"repo": "fixture/manifest_less",     "category": "manifest_less"},
    {"repo": "fixture/known_conflict",    "category": "known_conflict"},
    {"repo": "fixture/manifest_present",  "category": "manifest_present"},
    {"repo": "fixture/known_conflict",    "category": "known_conflict"},
]


class TestSmokeEpisodes:
    """5 dry-run episodes covering all three corpus categories."""

    @pytest.mark.parametrize("fixture", EPISODE_FIXTURES[:3], ids=[
        "manifest_present", "manifest_less", "known_conflict"
    ])
    def test_episode_terminates_correctly(self, env: PlannerEnv, fixture: dict) -> None:
        summary = run_episode(env, fixture)
        steps = summary["steps"]

        assert len(steps) > 0, "Episode produced no steps"
        last = steps[-1]

        assert last["terminated"] or last["truncated"], (
            "Episode did not terminate (terminated=False, truncated=False after final step)"
        )

    @pytest.mark.parametrize("fixture", EPISODE_FIXTURES, ids=[
        f"ep{i}" for i in range(len(EPISODE_FIXTURES))
    ])
    def test_step_returns_correct_5tuple(self, env: PlannerEnv, fixture: dict) -> None:
        """Step must return (obs, float, bool, bool, dict)."""
        obs, info = env.reset(repo_entry=fixture)
        instance = env._instance
        policy = FixedDummyPolicy(instance)
        action = policy.next_action()

        result = env.step(action)
        assert len(result) == 5
        new_obs, reward, terminated, truncated, step_info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)

    def test_observation_json_schema(self, env: PlannerEnv) -> None:
        """Observation must serialize to valid JSON with required top-level keys."""
        obs, _ = env.reset(repo_entry=EPISODE_FIXTURES[0])
        serializer = ObservationSerializer()
        ctx = serializer.to_context_string(obs)

        parsed = json.loads(ctx)
        assert "intent" in parsed
        assert "ecosystem_summary" in parsed
        assert "dependency_details" in parsed
        assert "conflicts" in parsed
        assert "episode_history" in parsed

        intent = parsed["intent"]
        assert "text" in intent
        assert "template" in intent
        assert "repo_id" in intent

        history = parsed["episode_history"]
        assert "step_index" in history
        assert "max_steps" in history
        assert "steps_taken" in history

    def test_observation_updates_after_each_step(self, env: PlannerEnv) -> None:
        """episode_steps in the observation grows by 1 per step."""
        obs, _ = env.reset(repo_entry=EPISODE_FIXTURES[0])
        instance = env._instance
        policy = FixedDummyPolicy(instance)

        for expected_len in range(1, 3):
            action = policy.next_action()
            new_obs, _, terminated, truncated, _ = env.step(action)
            assert len(new_obs.episode_steps) == expected_len, (
                f"Expected {expected_len} steps in history, got {len(new_obs.episode_steps)}"
            )
            if terminated or truncated:
                break

    def test_reset_clears_episode_state(self, env: PlannerEnv) -> None:
        """After reset(), step_index and episode_steps are zeroed."""
        obs, _ = env.reset(repo_entry=EPISODE_FIXTURES[0])
        instance = env._instance
        policy = FixedDummyPolicy(instance)
        action = policy.next_action()
        env.step(action)

        # Reset with a different fixture
        new_obs, _ = env.reset(repo_entry=EPISODE_FIXTURES[1])
        assert new_obs.step_index == 0, (
            f"After reset(), step_index should be 0, got {new_obs.step_index}"
        )
        assert len(new_obs.episode_steps) == 0, (
            f"After reset(), episode_steps should be empty, got {new_obs.episode_steps}"
        )

    def test_dry_run_safe_steps_are_verified(self, env: PlannerEnv) -> None:
        """In dry_run mode, SAFE steps must return verified=True."""
        obs, _ = env.reset(repo_entry=EPISODE_FIXTURES[0])
        instance = env._instance
        policy = FixedDummyPolicy(instance)

        # The dummy policy produces SAFE steps (install, check, etc.)
        action = policy.next_action()
        _, _, _, _, step_info = env.step(action)

        if step_info["exec_tier"] == "SAFE":
            assert step_info["verified"] is True, (
                "Dry-run SAFE steps must return verified=True (synthetic success)."
            )

    def test_reward_breakdown_logged_each_step(self, env: PlannerEnv) -> None:
        """reward_breakdown must be present in info at every step."""
        obs, _ = env.reset(repo_entry=EPISODE_FIXTURES[0])
        instance = env._instance
        policy = FixedDummyPolicy(instance)

        for _ in range(min(3, env._cfg.max_steps)):
            action = policy.next_action()
            _, _, terminated, truncated, step_info = env.step(action)
            breakdown = step_info.get("reward_breakdown", {})
            assert "total" in breakdown, "reward_breakdown must contain 'total'"
            assert "validator" in breakdown
            assert "execution" in breakdown
            assert "step_penalty" in breakdown
            if terminated or truncated:
                break

    def test_test_split_blocked(self, env: PlannerEnv) -> None:
        """PlannerEnv must raise ValueError if 'test' split is requested."""
        import pytest
        cfg = EnvConfig(
            corpus_manifest_path="datasets/repo_corpus/corpus_manifest_v1.json",
            split="test",
            dry_run=True,
        )
        test_env = PlannerEnv(config=cfg)
        # Test split is only blocked when loading the corpus
        # (the manifest file may not exist, so we mock the corpus load path)
        test_env._corpus = None  # force reload
        # Simulate the split being "test" with a dummy manifest
        import json, tempfile, os
        dummy_manifest = {
            "splits": {
                "train": [{"repo": "a/b", "category": "manifest_present"}],
                "val": [],
                "test": [{"repo": "c/d", "category": "manifest_present"}],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dummy_manifest, f)
            tmp_path = f.name
        try:
            cfg2 = EnvConfig(corpus_manifest_path=tmp_path, split="test", dry_run=True)
            env2 = PlannerEnv(config=cfg2)
            with pytest.raises(ValueError, match="test"):
                env2._load_corpus()
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Full episode summary report (printed, not asserted — human readability)
# ---------------------------------------------------------------------------

class TestFullEpisodeReport:
    """
    Run all 5 episodes and print a summary table.

    This test always passes (no assertions beyond 5-tuple validity) —
    its purpose is to produce readable output that confirms the environment
    works end-to-end and provides the reward breakdowns for review.
    """

    def test_five_episode_dry_run_report(self, env: PlannerEnv, capsys) -> None:
        print("\n" + "=" * 70)
        print("  System 2 Planner RL Environment — Dry-Run Smoke Test Report")
        print("=" * 70)

        for i, fixture in enumerate(EPISODE_FIXTURES):
            summary = run_episode(env, fixture)
            steps   = summary["steps"]
            cat     = fixture["category"]
            print(f"\nEpisode {i+1} [{cat}] — {fixture['repo']}")
            print(f"  Template:   {summary['info']['template']}")
            print(f"  Steps:      {len(steps)}")
            print(f"  Terminated: {summary['terminated']}")
            print(f"  Truncated:  {summary['truncated']}")
            for s in steps:
                bd = s["breakdown"]
                print(
                    f"    Step {s['step_idx']+1}: [{s['exec_status']:>8}] "
                    f"verified={str(s['verified']):<5}  "
                    f"reward={s['reward']:+.3f}  "
                    f"(val={bd['validator']:+.2f} "
                    f"exec={bd['execution']:+.2f} "
                    f"pen={-bd['step_penalty']:.2f})"
                )
            total_reward = sum(s["reward"] for s in steps)
            print(f"  Total episode reward: {total_reward:+.4f}")

        print("\n" + "=" * 70)
        captured = capsys.readouterr()
        print(captured.out, end="")
