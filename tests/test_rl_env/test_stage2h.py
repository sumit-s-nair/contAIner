"""
tests/test_rl_env/test_stage2h.py
=================================
Stage 2H: fixes for the ISOLATE collapse seen in pilot run_3997be.

Pilot facts (50 episodes, 1,600 completions, Qwen2.5-0.5B): best reward -0.1,
91% of actions ISOLATE .venv (77% -> 96% over training), 26 INSTALLs, all with
invalid targets (.venv, .gitignore, .git/config).  Cause: an idle success with
no progress cost -0.1 while a failed INSTALL cost -2.1.
"""

from __future__ import annotations

import pytest

from src.repo_scan.models import Dependency, EcosystemManifest, RepoManifest, SourceRef
from src.rl_env.env import EnvConfig, PlannerEnv
from src.rl_env.observation import CanonicalIntent, Observation, ObservationSerializer
from src.rl_env.reward import RewardConfig, compute_reward
from src.sandbox.models import AtomicStep, ExecutionResult, ExecutionStatus, RiskTier
from src.system2_planner.models import ActionType, PlannedStep, TemplateInstance
from src.system2_planner.templates import TEMPLATES

CFG = RewardConfig()


def _inst(conflicts=False) -> TemplateInstance:
    m = RepoManifest()
    eco = EcosystemManifest(ecosystem="python", manifest_files=["requirements.txt"])
    eco.add_dependency(Dependency(
        name="requests", declared_constraint=">=2.0",
        sources=[SourceRef(file="requirements.txt", line=1, raw_line="requests>=2.0")],
    ))
    m.ecosystems["python"] = eco
    if conflicts:
        m.conflicts = ["synthetic conflict"]
    return TemplateInstance(template=TEMPLATES["setup_project"], manifest=m)


def _res(status=ExecutionStatus.SUCCESS, verified=True, rc=0) -> ExecutionResult:
    return ExecutionResult(step=AtomicStep(command="x"), status=status, tier=RiskTier.SAFE,
                           verified=verified, returncode=rc)


def _ps(t, target=None) -> PlannedStep:
    return PlannedStep(action_type=t, target=target, description=f"{t.value} {target or ''}")


def _r(plan, history, res, inst, **kw):
    return compute_reward(step=plan[-1], plan_so_far=plan, instance=inst, execution_result=res,
                          config=CFG, history=history, episode_complete=kw.get("complete", False),
                          episode_done=kw.get("done", False), install_progress=kw.get("ip"))


# --- reward ------------------------------------------------------------------

def test_no_progress_is_strictly_negative():
    """Rejects the collapse: an idle success must cost more than the step penalty."""
    assert CFG.r_no_progress < 0
    total, bd = _r([_ps(ActionType.ISOLATE, ".venv")], [], _res(), _inst(conflicts=False))
    assert bd["execution_label"] == "success_verified_no_progress"
    assert bd["execution"] == pytest.approx(CFG.w_execution * CFG.r_no_progress)
    assert total < -CFG.w_step_penalty


def test_repeated_isolate_with_conflicts_is_penalised():
    inst = _inst(conflicts=True)
    p = [_ps(ActionType.ISOLATE, ".venv")]
    first, _ = _r(p, [], _res(), inst)
    second, _ = _r(p + [_ps(ActionType.ISOLATE, ".venv")], [_res()], _res(), inst)
    assert first > 0 > second


def test_failed_vs_idle_gap_narrowed():
    """Pilot gap was 2.0 (-2.1 vs -0.1); now it is w_exec * (r_no_progress - r_failed) = 1.0."""
    inst = _inst()
    idle, _ = _r([_ps(ActionType.ISOLATE, ".venv")], [], _res(), inst)
    failed, _ = _r([_ps(ActionType.INSTALL, ".venv")], [],
                   _res(status=ExecutionStatus.FAILED, verified=False, rc=1), inst, ip=0.0)
    assert failed < idle                              # failing is still worse than idling
    assert idle - failed == pytest.approx(CFG.w_execution * (CFG.r_no_progress - CFG.r_failed))
    assert idle - failed <= 1.0 + 1e-9


def test_failed_stays_below_review_denied():
    """Stage 2F ordering denied > failed must survive the Stage 2H retune."""
    assert CFG.r_failed < CFG.r_review_denied


def test_progress_install_beats_idle_and_failure():
    inst = _inst()
    good, _ = _r([_ps(ActionType.INSTALL, "requirements.txt")], [], _res(), inst, ip=1.0)
    idle, _ = _r([_ps(ActionType.ISOLATE, ".venv")], [], _res(), inst)
    failed, _ = _r([_ps(ActionType.INSTALL, ".gitignore")], [],
                   _res(status=ExecutionStatus.FAILED, verified=False, rc=1), inst, ip=0.0)
    assert good > idle > failed
    assert good > 0


def test_successful_install_with_zero_progress_is_not_free():
    """An exit-0 INSTALL that adds no required package (e.g. already credited) is padding."""
    _, bd = _r([_ps(ActionType.INSTALL, "requirements.txt")], [], _res(), _inst(), ip=0.0)
    assert bd["execution"] == pytest.approx(CFG.w_execution * CFG.r_no_progress)


def test_blocked_still_dominates():
    total, _ = _r([_ps(ActionType.INSTALL, "x")], [], ExecutionResult(
        step=AtomicStep(command="x"), status=ExecutionStatus.BLOCKED, tier=RiskTier.BLOCKED,
        verified=False, returncode=1), _inst())
    idle, _ = _r([_ps(ActionType.ISOLATE, ".venv")], [], _res(), _inst())
    assert total < idle


# --- observation / prompt ----------------------------------------------------

def _obs(conflicts=False, steps=None) -> Observation:
    m = RepoManifest(
        ecosystems={"python": EcosystemManifest(ecosystem="python", manifest_files=["requirements.txt"],
                                               dependencies={"requests": None})},
        conflicts=["c"] if conflicts else [],
    )
    return Observation(
        canonical_intent=CanonicalIntent(intent_text="Set up", template_name="setup_project", repo_id="o/r"),
        repo_manifest=m, step_index=len(steps or []), max_steps=8, episode_steps=steps or [],
    )


def test_system_prompt_has_rules_and_install_example():
    sys_c = ObservationSerializer().to_chat_prompt(_obs())[0]["content"]
    assert "ISOLATE only when the prompt lists Conflicts" in sys_c
    assert '"target": "requirements.txt"' in sys_c
    assert "Never install '.venv'" in sys_c


def test_dependency_list_survives_longer_system_prompt():
    user = ObservationSerializer().to_chat_prompt(_obs())[1]["content"]
    assert "Dependencies:" in user and "requests" in user


def test_history_shows_action_and_target():
    env = PlannerEnv(config=EnvConfig(dry_run=True, corpus_manifest_path=None))
    env.reset()
    env.step(PlannedStep(action_type=ActionType.ISOLATE, target=".venv", description="Make venv"))
    line = env._obs.episode_steps[-1]
    assert line.startswith("ISOLATE .venv: Make venv [")


# --- trainer -----------------------------------------------------------------

def test_trainer_defaults_to_1_5b_and_accepts_overrides():
    pytest.importorskip("torch")
    from scripts.train_system2_grpo import get_parser
    a = get_parser().parse_args([])
    assert a.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert a.max_episodes == 1000
    b = get_parser().parse_args(["--model_id", "Qwen/Qwen2.5-0.5B-Instruct", "--max_episodes", "50"])
    assert b.model_id == "Qwen/Qwen2.5-0.5B-Instruct" and b.max_episodes == 50


# --- trainer logging (run beab44: comp_rate logged 0.00, 47/50 actually completed) ---

def test_breakdown_logging_reads_reward_keys():
    pytest.importorskip("torch")
    from scripts.train_system2_grpo import accumulate_breakdown
    inst = _inst()
    _, bd = _r([_ps(ActionType.INSTALL, "requirements.txt")], [], _res(), inst, ip=1.0, complete=True)
    acc = {"validator": 0.0, "execution": 0.0, "completion": 0.0, "penalty": 0.0}
    accumulate_breakdown(acc, bd)
    assert acc["completion"] == pytest.approx(CFG.w_completion)      # was always 0
    assert acc["penalty"] == pytest.approx(CFG.w_step_penalty)       # was always 0
    assert acc["validator"] + acc["execution"] + acc["completion"] - acc["penalty"] == pytest.approx(bd["total"])


def test_episode_completed_reads_env_key():
    pytest.importorskip("torch")
    from scripts.train_system2_grpo import episode_completed
    assert episode_completed({"episode_complete": True}) == 1
    assert episode_completed({"episode_complete": False}) == 0
    assert episode_completed({"verified_complete": True}) == 0       # the old, never-set key


# --- snapshot leak (run beab44: images 1 GB -> 21.9 GB after 2 snapshot failures) ---

def _bare_executor():
    from src.rl_env.docker_executor import DockerEpisodeExecutor
    ex = object.__new__(DockerEpisodeExecutor)
    ex._container_id = "cid"
    return ex


@pytest.mark.parametrize("mode", ["timeout", "nonzero"])
def test_failed_commit_removes_the_image(monkeypatch, mode):
    import subprocess
    from src.rl_env import docker_executor as de
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "commit"]:
            if mode == "timeout":
                raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(de.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        _bare_executor().snapshot(tag="grpo-snapshot-x")
    assert ["docker", "rmi", "-f", "grpo-snapshot-x"] in calls


def test_commit_timeout_raised_from_60():
    from src.rl_env.docker_executor import SNAPSHOT_COMMIT_TIMEOUT
    assert SNAPSHOT_COMMIT_TIMEOUT >= 300


# --- KL spikes (run b45f5d ep 92/228/265): EOS padding was inside the loss mask ---

def test_completion_mask_stops_after_first_eos():
    torch = pytest.importorskip("torch")
    from scripts.train_system2_grpo import completion_mask
    IM_END, ENDOFTEXT = 151645, 151643
    gen = torch.tensor([
        [11, 12, IM_END, IM_END, IM_END],   # finished early, padded with eos_token_id
        [11, 12, 13, 14, 15],               # hit max_new_tokens, no EOS
        [11, ENDOFTEXT, IM_END, IM_END, 9], # ended on the other EOS id
    ])
    m = completion_mask(gen, [ENDOFTEXT, IM_END])
    assert m.tolist() == [
        [1, 1, 1, 0, 0],   # the EOS itself is trained; the padding is not
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0],
    ]


def test_old_pad_mask_leaked_eos_padding():
    """Documents the bug: pad_token_id != eos_token_id for Qwen2.5-Instruct."""
    torch = pytest.importorskip("torch")
    IM_END, ENDOFTEXT = 151645, 151643
    gen = torch.tensor([[11, 12, IM_END, IM_END, IM_END]])
    old = (gen != ENDOFTEXT).float()
    assert old.sum().item() == 5     # two forced padding tokens were in the loss


def test_eos_token_ids_merges_tokenizer_and_generation_config():
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from scripts.train_system2_grpo import eos_token_ids
    tok = SimpleNamespace(eos_token_id=151645, pad_token_id=151643)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
    assert eos_token_ids(tok, model) == [151643, 151645]
    assert eos_token_ids(SimpleNamespace(eos_token_id=2, pad_token_id=None)) == [2]


def test_trainer_scores_tokens_at_sampling_temperature():
    """Log-probs must use the temperature generate() sampled at (was T=1 vs sampled T=0.8)."""
    pytest.importorskip("torch")
    import inspect
    import scripts.train_system2_grpo as t
    src = inspect.getsource(t.main)
    assert src.count('logits[:, input_length-1:-1, :] / SAMPLING_CONFIG["temperature"]') == 2  # policy + ref


# --- evaluate_system2 rewrite: mirrors training (prompt, parser, step budget, recovery) ---

class _Enc(dict):
    def __getattr__(self, k):
        return self[k]

    def to(self, device):
        return self


def _eval_env(step_infos, conflicts=False):
    """Mock env whose step() returns the given (reward, done, info) in order."""
    from unittest.mock import MagicMock
    env = MagicMock()
    obs = MagicMock()
    obs.repo_manifest.conflicts = ["c"] if conflicts else []
    env.reset.return_value = (obs, {})
    env._serializer.to_chat_prompt.return_value = []
    seq = iter(step_infos)

    def step(action):
        r, done, info = next(seq)
        return obs, r, done, False, info
    env.step.side_effect = step
    return env


def _eval_model(n_outputs):
    torch = pytest.importorskip("torch")
    from unittest.mock import MagicMock
    model, tok = MagicMock(), MagicMock()
    tok.apply_chat_template.return_value = "p"
    tok.return_value = _Enc({"input_ids": torch.tensor([[1, 2]])})
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model, tok


def test_eval_continues_after_failed_step_and_counts_unnecessary_isolate():
    pytest.importorskip("torch")
    from unittest.mock import patch
    from scripts.evaluate_system2 import run_episode, summarize
    model, tok = _eval_model(3)
    env = _eval_env([
        (-0.6, False, {"episode_complete": False}),          # ISOLATE on a no-conflict repo
        (-1.6, False, {"episode_complete": False}),          # failed INSTALL: old eval stopped here
        (7.9, True, {"episode_complete": True}),             # recovery
    ])
    steps = iter([PlannedStep(action_type=ActionType.ISOLATE, description="d", target=".venv"),
                  PlannedStep(action_type=ActionType.INSTALL, description="d", target="requirements.txt"),
                  PlannedStep(action_type=ActionType.INSTALL, description="d", target="requirements.txt")])
    with patch("scripts.evaluate_system2.parse_planned_step", side_effect=lambda _: next(steps)):
        rec = run_episode(env, model, tok, {"repo": "o/r", "category": "python"})
    assert rec["completed"] and rec["steps"] == 3 and not rec["env_error"]
    m = summarize([rec])
    assert m["completion_rate"] == 1.0
    assert m["unnecessary_isolate_rate"] == 1.0
    assert m["first_step_isolate_rate"] == 1.0
    assert m["mean_steps"] == 3


def test_eval_isolate_with_conflicts_is_not_unnecessary():
    pytest.importorskip("torch")
    from unittest.mock import patch
    from scripts.evaluate_system2 import run_episode, summarize
    model, tok = _eval_model(1)
    env = _eval_env([(1.0, True, {"episode_complete": True})], conflicts=True)
    with patch("scripts.evaluate_system2.parse_planned_step",
               return_value=PlannedStep(action_type=ActionType.ISOLATE, description="d", target=".venv")):
        rec = run_episode(env, model, tok, {"repo": "o/r"})
    m = summarize([rec])
    assert m["no_conflict_episodes"] == 0 and m["unnecessary_isolate_rate"] == 0.0


def test_eval_parse_error_ends_episode_as_failure():
    pytest.importorskip("torch")
    from unittest.mock import patch
    from src.rl_env.parser import ParseError
    from scripts.evaluate_system2 import run_episode, summarize
    model, tok = _eval_model(1)
    env = _eval_env([])
    with patch("scripts.evaluate_system2.parse_planned_step", side_effect=ParseError("bad")):
        rec = run_episode(env, model, tok, {"repo": "o/r"})
    assert rec["parse_error"] and not rec["completed"] and rec["steps"] == 1
    assert summarize([rec])["parse_error_rate"] == 1.0
    env.step.assert_not_called()


def test_eval_defaults_match_training():
    pytest.importorskip("torch")
    from scripts.evaluate_system2 import get_parser, MAX_NEW_TOKENS
    from scripts.train_system2_grpo import MODEL_ID, CORPUS_MANIFEST, SAMPLING_CONFIG
    a = get_parser().parse_args([])
    assert a.model_id == MODEL_ID
    assert a.manifest == CORPUS_MANIFEST
    assert MAX_NEW_TOKENS == SAMPLING_CONFIG["max_new_tokens"]
    assert a.splits == ["val", "test"]


# --- verify_table: pip options were captured as the package (4090 integration failure) ---

@pytest.mark.parametrize("cmd,pkg,verify", [
    ("pip install --user six", "six", "python -c 'import six'"),          # was import __user
    ("pip install -U --user flask", "flask", "python -c 'import flask'"),
    ("pip install -i https://x/simple numpy", "numpy", "python -c 'import numpy'"),
    ("pip install --index-url=https://x six", "six", "python -c 'import six'"),
    ("pip install \"requests>=2.0\"", "requests", "python -c 'import requests'"),
    ("python -m pip install pytest==8.3.5", "pytest", "python -c 'import pytest'"),
    ("python -m pip install -r requirements.txt", None, "pip check"),     # was import _r
    ("pip install -e .", None, "pip check"),
    ("python -m pip install .", None, "pip check"),
])
def test_pip_verify_skips_options(cmd, pkg, verify):
    from src.rl_env.verify_table import _extract_primary_package, lookup_verify_command
    from src.sandbox.models import AtomicStep
    assert _extract_primary_package(cmd) == pkg
    assert lookup_verify_command(AtomicStep(command=cmd, description="d")) == verify
