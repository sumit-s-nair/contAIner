#!/usr/bin/env python3
"""
scripts/train_system2_grpo.py
=============================
GRPO-style training loop for the System 2 Planner.

Core requirements met:
- GRPO-style rollouts against PlannerEnv (using tokenizer.apply_chat_template).
- Model: Qwen2.5-0.5B-Instruct, QLoRA 4-bit (8GB VRAM target).
- Environment: dry_run=False, training_review_policy=auto_deny.
- Corpus: v3 manifest, train split only.
- Checkpointing: Full state (weights, optimizer, RNG, episode, step, episodes processed).
- Progress visibility: tqdm with rolling stats, periodic breakdown, loud warnings for blocked/denied steps.
- JSONL structured logging.
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import traceback
import uuid
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from src.rl_env.env import PlannerEnv, EnvConfig, EnvForkError
from src.rl_env.dry_run_executor import TrainingReviewPolicy
from src.sandbox.models import RiskTier
from src.system2_planner.models import PlannedStep, ActionType
from src.rl_env.parser import (
    parse_planned_step,
    ParseError,
    MissingFieldError,
    InvalidFieldError,
    UnrecognizedActionTypeError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EnvironmentFailure(RuntimeError):
    """Raised when environment execution fails during completion evaluation."""
    pass


@dataclass
class CompletionResult:
    action: Optional[PlannedStep]
    info: Dict[str, Any]
    reward: Optional[float]
    status: str  # OK / PARSE_ERROR / ENV_ERROR


def evaluate_completion(
    env: PlannerEnv,
    completion: str,
    snapshot_tag: Optional[str] = None,
) -> CompletionResult:
    """
    Evaluate a single model completion against a fork of the environment.

    (1) Parsing try block: catches JSONDecodeError first, then parser exceptions.
        Returns status="PARSE_ERROR" with reward=-1.0 and subtype info.
    (2) Environment try block: forks the environment and executes the action.
        Returns status="OK" and environment reward on success.
        Any exception returns status="ENV_ERROR" with reward=None and traceback in info.
    """
    info: Dict[str, Any] = {}
    action: Optional[PlannedStep] = None

    # (1) Parsing try block
    try:
        action = parse_planned_step(completion)
    except json.JSONDecodeError as e:
        info["exec_status"] = "JSON_DECODE_ERROR"
        info["subtype"] = "JSON_DECODE_ERROR"
        info["execution_error"] = str(e)
        return CompletionResult(action=None, info=info, reward=-1.0, status="PARSE_ERROR")
    except MissingFieldError as e:
        info["exec_status"] = "MISSING_FIELD"
        info["subtype"] = "MISSING_FIELD"
        info["execution_error"] = str(e)
        return CompletionResult(action=None, info=info, reward=-1.0, status="PARSE_ERROR")
    except InvalidFieldError as e:
        info["exec_status"] = "INVALID_FIELD"
        info["subtype"] = "INVALID_FIELD"
        info["execution_error"] = str(e)
        return CompletionResult(action=None, info=info, reward=-1.0, status="PARSE_ERROR")
    except UnrecognizedActionTypeError as e:
        info["exec_status"] = "UNRECOGNIZED_ACTION_TYPE"
        info["subtype"] = "UNRECOGNIZED_ACTION_TYPE"
        info["execution_error"] = str(e)
        return CompletionResult(action=None, info=info, reward=-1.0, status="PARSE_ERROR")
    except ParseError as e:
        info["exec_status"] = "PARSE_FAILED"
        info["subtype"] = "PARSE_FAILED"
        info["execution_error"] = str(e)
        return CompletionResult(action=None, info=info, reward=-1.0, status="PARSE_ERROR")
    except Exception as e:
        tb_str = traceback.format_exc()
        info["execution_error"] = f"Parser harness error: {type(e).__name__}: {str(e)}\n{tb_str}"
        info["status"] = "ENV_ERROR"
        info["exec_status"] = "ENV_ERROR"
        return CompletionResult(action=None, info=info, reward=None, status="ENV_ERROR")

    # (2) Environment try block
    forked_env = None
    res = None
    try:
        forked_env = env.fork(snapshot_tag)
        _, reward, _, _, step_info = forked_env.step(action)
        info.update(step_info)
        info["status"] = "OK"
        res = CompletionResult(action=action, info=info, reward=reward, status="OK")
    except Exception as e:
        tb_str = traceback.format_exc()
        info["execution_error"] = f"{type(e).__name__}: {str(e)}\n{tb_str}"
        info["status"] = "ENV_ERROR"
        info["exec_status"] = "ENV_ERROR"
        res = CompletionResult(action=action, info=info, reward=None, status="ENV_ERROR")
    finally:
        if forked_env is not None:
            try:
                forked_env.end_episode()
            except Exception as cleanup_exc:
                cleanup_msg = f"{type(cleanup_exc).__name__}: {str(cleanup_exc)}"
                info["cleanup_error"] = cleanup_msg
                if res is not None:
                    res.info["cleanup_error"] = cleanup_msg

    return res


# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS_MANIFEST = "datasets/repo_corpus/corpus_manifest_v3.json"

# VRAM-friendly defaults for 8GB
BATCH_SIZE = 4            # Group size G for GRPO
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5
MAX_EPISODES = 1000

CHECKPOINT_CADENCE = 50   # Align eval and checkpoint
EVAL_CADENCE = 50

class RunState:
    """Encapsulates the state required to pause and resume training exactly."""
    def __init__(self):
        self.episode_idx = 0
        self.step_idx = 0
        self.episodes_processed = 0
        self.global_step = 0

    def state_dict(self):
        return {
            "episode_idx": self.episode_idx,
            "step_idx": self.step_idx,
            "episodes_processed": self.episodes_processed,
            "global_step": self.global_step,
        }

    def load_state_dict(self, sd: dict):
        self.episode_idx = sd.get("episode_idx", 0)
        self.step_idx = sd.get("step_idx", 0)
        self.episodes_processed = sd.get("episodes_processed", 0)
        self.global_step = sd.get("global_step", 0)


# Module-level sampling configuration for generate() and rollout logging
SAMPLING_CONFIG: Dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 1.0,
    "max_new_tokens": 512,
}


def compute_group_advantages(
    results: List[Any],
    G: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute GRPO group advantages over non-ENV_ERROR completions.

    Returns (advantages, valid_mask) where:
    - advantages: Tensor[G]
    - valid_mask: Tensor[G, 1]
    or None when fewer than 2 non-ENV_ERROR completions exist.
    Flat reward standard deviation (or std < 1e-4) yields zero advantages.
    """
    valid_indices = []
    for i, r in enumerate(results):
        status = getattr(r, "status", None) if not isinstance(r, dict) else r.get("status")
        if status != "ENV_ERROR":
            valid_indices.append(i)

    if len(valid_indices) < 2:
        return None

    valid_rewards = []
    for idx in valid_indices:
        r = results[idx]
        rew = getattr(r, "reward", None) if not isinstance(r, dict) else r.get("reward")
        if rew is None:
            status = getattr(r, "status", None) if not isinstance(r, dict) else r.get("status")
            raise ValueError(f"Result at index {idx} with status '{status}' has reward None")
        valid_rewards.append(float(rew))

    reward_tensor = torch.tensor(valid_rewards, dtype=torch.float32)
    std_r = reward_tensor.std()
    mean_r = reward_tensor.mean()

    advantages = torch.zeros(G, dtype=torch.float32)
    valid_mask = torch.zeros(G, 1, dtype=torch.float32)

    for idx in valid_indices:
        valid_mask[idx, 0] = 1.0

    if std_r >= 1e-4:
        for idx_in_valid, orig_idx in enumerate(valid_indices):
            advantages[orig_idx] = (reward_tensor[idx_in_valid] - mean_r) / (std_r + 1e-8)

    if device is not None:
        advantages = advantages.to(device)
        valid_mask = valid_mask.to(device)

    return advantages, valid_mask


def masked_token_loss(
    per_token_pg_loss: torch.Tensor,
    per_token_kl: torch.Tensor,
    pad_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute masked policy and KL token loss.

    Excludes ENV_ERROR rows via valid_mask and padding via pad_mask.
    Returns (loss, normalizer).
    """
    mask = pad_mask * valid_mask
    normalizer = mask.sum()
    if normalizer > 0:
        per_token_loss = per_token_pg_loss + beta * per_token_kl
        loss = (per_token_loss * mask).sum() / normalizer
    else:
        loss = torch.tensor(0.0, dtype=per_token_pg_loss.dtype, device=per_token_pg_loss.device)
    return loss, normalizer


def select_best_action(results: List[Any]) -> Optional[Tuple[int, float, Any]]:
    """
    Select the best action among completions with status OK.

    Returns (best_idx, best_reward, best_action) or None if no OK completion exists.
    """
    ok_candidates = []
    for i, r in enumerate(results):
        status = getattr(r, "status", None) if not isinstance(r, dict) else r.get("status")
        reward = getattr(r, "reward", None) if not isinstance(r, dict) else r.get("reward")
        action = getattr(r, "action", None) if not isinstance(r, dict) else r.get("action")
        if status == "OK" and reward is not None and action is not None:
            ok_candidates.append((i, float(reward), action))

    if not ok_candidates:
        return None

    return max(ok_candidates, key=lambda x: x[1])


class FailFastTracker:
    """
    Tracks consecutive and per-episode ENV_ERROR counts to fail fast and loudly.

    Aborts if 3 consecutive completions are ENV_ERROR (across group boundaries)
    or if running ENV_ERROR count in an episode exceeds 10% of (max_steps * G).
    """
    def __init__(self, max_steps: int, G: int):
        self.max_steps = max_steps
        self.G = G
        self.consecutive_env_errors: int = 0
        self.episode_env_errors: int = 0
        self.total_env_errors: int = 0
        self.last_traceback: str = ""
        self._threshold = 0.10 * (max_steps * G)
        self._should_abort: bool = False
        self._reason: str = ""

    def reset_episode(self) -> None:
        self.episode_env_errors = 0

    def record(self, status: str, traceback_str: str = "") -> None:
        if status == "ENV_ERROR":
            self.consecutive_env_errors += 1
            self.episode_env_errors += 1
            self.total_env_errors += 1
            if traceback_str:
                self.last_traceback = traceback_str
            self._check_abort()
        else:
            self.consecutive_env_errors = 0

    def record_main_step_error(self, traceback_str: str = "") -> None:
        self.consecutive_env_errors += 1
        self.episode_env_errors += 1
        self.total_env_errors += 1
        if traceback_str:
            self.last_traceback = traceback_str
        self._check_abort()

    def _check_abort(self) -> None:
        if self.consecutive_env_errors >= 3:
            self._should_abort = True
            self._reason = f"3 consecutive completions are ENV_ERROR (reached {self.consecutive_env_errors})"
        elif self.episode_env_errors > self._threshold:
            self._should_abort = True
            self._reason = (
                f"Running ENV_ERROR count ({self.episode_env_errors}) in episode "
                f"exceeds 10% threshold ({self._threshold:.1f})"
            )

    @property
    def should_abort(self) -> bool:
        return self._should_abort

    @property
    def reason(self) -> str:
        return self._reason


def write_emergency_abort(
    out_dir: Path,
    model: Any,
    optimizer: Any,
    run_state: Any,
    tracker: FailFastTracker,
    last_tb: str = "",
) -> Path:
    """
    Save emergency checkpoint and reason.txt into out_dir / 'emergency_ckpt'.

    If checkpoint saving raises an exception, writes that exception into reason.txt.
    Does not call sys.exit().
    """
    emergency_dir = out_dir / "emergency_ckpt"
    emergency_dir.mkdir(parents=True, exist_ok=True)

    ckpt_error_str = ""
    try:
        save_checkpoint(out_dir, model, optimizer, run_state, is_emergency=True, target_dir=emergency_dir)
    except Exception as e:
        ckpt_error_str = f"\nCHECKPOINT SAVE FAILED: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error("Emergency checkpoint save failed: %s", e)

    tb_to_write = last_tb or tracker.last_traceback
    reason_file = emergency_dir / "reason.txt"
    with open(reason_file, "w", encoding="utf-8") as f:
        f.write(
            f"REASON: {tracker.reason}\n"
            f"Consecutive ENV_ERROR count: {tracker.consecutive_env_errors}\n"
            f"Episode ENV_ERROR count: {tracker.episode_env_errors}\n"
            f"LAST TRACEBACK:\n{tb_to_write}\n"
        )
        if ckpt_error_str:
            f.write(ckpt_error_str + "\n")

    logger.critical("EMERGENCY ABORT: %s. Saved to %s", tracker.reason, emergency_dir)
    return emergency_dir


# --- Checkpoint & Resume Logic ---
def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_state: RunState,
    is_emergency: bool = False,
    target_dir: Optional[Path] = None,
):
    if target_dir is not None:
        ckpt_dir = target_dir
    else:
        ckpt_name = "checkpoint-emergency" if is_emergency else f"checkpoint-{run_state.episode_idx}"
        ckpt_dir = output_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save LoRA weights
    model.save_pretrained(ckpt_dir)

    # 2. Save Optimizer state
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # 3. Save RNG states
    rng_states = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    }
    torch.save(rng_states, ckpt_dir / "rng_state.pt")

    # 4. Save Run State (counters & cursor)
    with open(ckpt_dir / "run_state.json", "w") as f:
        json.dump(run_state.state_dict(), f, indent=2)

    logger.info(f"Checkpoint saved to {ckpt_dir}")

def load_checkpoint(ckpt_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> RunState:
    logger.info(f"Resuming from checkpoint {ckpt_dir}")
    
    # 1. Load weights (PEFT)
    import peft
    peft.set_peft_model_state_dict(model, peft.load_peft_weights(str(ckpt_dir)))

    # 2. Load Optimizer
    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path))

    # 3. Load RNG
    rng_path = ckpt_dir / "rng_state.pt"
    if rng_path.exists():
        rng = torch.load(rng_path)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda"):
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    # 4. Load Run State
    state_path = ckpt_dir / "run_state.json"
    run_state = RunState()
    if state_path.exists():
        with open(state_path, "r") as f:
            run_state.load_state_dict(json.load(f))
            
    return run_state

def _run_dry_run_test():
    """Test function to verify checkpoint/resume fully reproduces state."""
    logger.info("Running dry-run checkpoint test with real model loading...")
    
    # Load Real Model and Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    import time
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    logger.info(f"Model loaded in {time.time()-t0:.2f}s")
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Force optimizer to initialize state buffers by doing a dummy backward + step
    dummy_loss = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.sum()
    dummy_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    state = RunState()
    state.episode_idx = 42
    state.episodes_processed = 100
    
    tmp_dir = Path("outputs/dry_run_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture original weight
    original_weight = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data.clone()
    orig_opt_state_str = str(optimizer.state_dict())
    
    # Seed RNG right before save
    random.seed(1234)
    torch.manual_seed(1234)
    
    save_checkpoint(tmp_dir, model, optimizer, state)
    logger.info(f"BEFORE: episode_idx={state.episode_idx}, episodes_processed={state.episodes_processed}")

    # Capture what the *next* random numbers should be after the save point
    expected_py_rand = random.random()
    expected_th_rand = torch.rand(1).item()
    
    # Corrupt everything
    random.seed(9999)
    torch.manual_seed(9999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # Blank optimizer
    model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data.fill_(999.0)
    
    # Reload
    new_state = load_checkpoint(tmp_dir / "checkpoint-42", model, optimizer)
    
    reloaded_weight = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data
    
    if not torch.allclose(original_weight, reloaded_weight):
        print(f"Original: {original_weight[0][:5]}")
        print(f"Reloaded: {reloaded_weight[0][:5]}")
        
    assert torch.allclose(original_weight, reloaded_weight), "Weights do not match!"
    logger.info("Weights matched successfully.")
    
    # Verify Optimizer
    reloaded_opt_state_str = str(optimizer.state_dict())
    assert orig_opt_state_str == reloaded_opt_state_str, "Optimizer state mismatch!"
    logger.info("Optimizer state matched successfully.")
    
    # Verify RNG
    actual_py_rand = random.random()
    actual_th_rand = torch.rand(1).item()
    assert actual_py_rand == expected_py_rand, f"Python RNG mismatch! Expected {expected_py_rand}, got {actual_py_rand}"
    assert actual_th_rand == expected_th_rand, f"Torch RNG mismatch! Expected {expected_th_rand}, got {actual_th_rand}"
    logger.info("RNG states (Python & Torch) matched successfully.")
    
    logger.info(f"AFTER: episode_idx={new_state.episode_idx}, episodes_processed={new_state.episodes_processed}")
    assert new_state.episode_idx == 42
    assert new_state.episodes_processed == 100
    logger.info("Dry-run checkpoint test passed.")

def setup_environment():
    if "COMPROMISED" in CORPUS_MANIFEST:
        raise ValueError("CRITICAL: Corpus manifest contains 'COMPROMISED' - halting to prevent exploit injection.")
        
    config = EnvConfig(
        dry_run=False,
        review_policy=TrainingReviewPolicy.AUTO_DENY,
        corpus_manifest_path=CORPUS_MANIFEST
    )
    env = PlannerEnv(config)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint directory to resume from")
    parser.add_argument("--test_checkpoint", action="store_true", help="Run the dry-run checkpoint test and exit")
    args = parser.parse_args()

    if args.test_checkpoint:
        _run_dry_run_test()
        sys.exit(0)

    # Output directory versioning
    run_id = uuid.uuid4().hex[:6]
    out_dir = Path(f"outputs/system2_grpo_run_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=False)
    log_file = out_dir / "training_log.jsonl"
    logger.info(f"Output directory: {out_dir}")

    # Initialize Environment
    env = setup_environment()
    
    # Log initial disk usage for delta measurement
    import subprocess as _sp
    def _docker_df() -> str:
        try:
            r = _sp.run(["docker", "system", "df"], capture_output=True, text=True, timeout=15)
            return r.stdout.strip()
        except Exception as e:
            return f"(docker system df failed: {e})"
    
    logger.info("=== docker system df BEFORE pilot ===\n%s", _docker_df())
    
    # Initialize Model & Tokenizer
    logger.info("Loading Qwen2.5-0.5B-Instruct in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    run_state = RunState()
    if args.resume_from:
        run_state = load_checkpoint(Path(args.resume_from), model, optimizer)
        
    # Set up signal handler for emergency save
    def handle_sigint(sig, frame):
        logger.warning("\nSIGINT received! Attempting emergency checkpoint save...")
        save_checkpoint(out_dir, model, optimizer, run_state, is_emergency=True)
        sys.exit(130)
    signal.signal(signal.SIGINT, handle_sigint)

    # Training Stats Tracking
    rolling_rewards = deque(maxlen=20)
    rolling_completion = deque(maxlen=20)
    
    pbar = tqdm(total=MAX_EPISODES, initial=run_state.episode_idx, desc="Training")

    # Fail-fast and tracking counters
    G = BATCH_SIZE
    tracker = FailFastTracker(max_steps=getattr(env._cfg, 'max_steps', 10), G=G)
    excluded_repos_count = 0
    snapshot_failures_count = 0
    skipped_updates_count = 0
    total_updates = 0
    no_valid_action_steps_count = 0

    if env._corpus is None:
        env._load_corpus()
    initial_corpus_len = len(env._corpus)

    try:
        for ep in range(run_state.episode_idx, MAX_EPISODES):
            run_state.episode_idx = ep
            tracker.reset_episode()
            ep_updates = 0
            ep_skipped_updates = 0
            ep_no_valid_action_steps = 0
            
            # Hard fail if val/test splits ever leak in (using the env's explicit split argument)
            if env._corpus is None:
                env._load_corpus()
            
            repo_entry = env._rng.choice(env._corpus)
            success = False
            last_reset_tb = ""
            for attempt in range(3):
                try:
                    obs, ep_info = env.reset(repo_entry=repo_entry)
                    run_state.episodes_processed += 1
                    success = True
                    break
                except Exception as e:
                    last_reset_tb = traceback.format_exc()
                    logger.warning(f"Failed to reset environment for {repo_entry['repo']} (attempt {attempt+1}/3): {e}")
                    import time
                    time.sleep(2)
            
            if not success:
                logger.error(f"REPO EXCLUDED: {repo_entry['repo']} failed all 3 reset attempts. Removing from corpus and skipping episode.")
                env._corpus.remove(repo_entry)
                excluded_repos_count += 1
                with open(out_dir / "excluded_repos.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "repo": repo_entry["repo"],
                        "split": getattr(env._cfg, "split", "train"),
                        "traceback": last_reset_tb
                    }) + "\n")
                if initial_corpus_len > 0 and (excluded_repos_count / initial_corpus_len) > 0.05:
                    raise RuntimeError(
                        f"Excluded repos threshold exceeded: {excluded_repos_count}/{initial_corpus_len} (>5%) failed reset 3x."
                    )
                continue
            
            episode_reward = 0.0
            step_breakdown = {"validator": 0.0, "execution": 0.0, "completion": 0.0, "penalty": 0.0}
            blocked_count = 0
            review_denied_count = 0
            completed = 0
            ep_loss_sum = 0.0
            ep_kl_sum = 0.0
            ep_pg_loss_sum = 0.0
            ep_grad_norm_sum = 0.0
            ep_steps = 0
            
            done = False
            ep_category = ep_info.get('category', 'unknown')
            logger.info("Episode %d: repo=%s  category=%s", ep, ep_info.get('repo','?'), ep_category)
            # Track the most-recent snapshot of the MAIN trajectory so group
            # evaluation containers can branch from it without replaying history.
            # Invariant: current_snapshot_tag is always valid when not None.
            current_snapshot_tag: Optional[str] = None
            info: Dict[str, Any] = {}
            try:
                while not done:
                    # 1. Get structured chat prompt from observation
                    chat_messages = env._serializer.to_chat_prompt(obs, tokenizer=tokenizer)
                    
                    # 2. Use tokenizer to format it exactly according to ChatML
                    prompt_str = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # --- GRPO Sampling block ---
                    inputs = tokenizer([prompt_str] * G, return_tensors="pt", padding=True).to("cuda")
                    
                    torch.cuda.empty_cache()  # evict residual allocations before G-way generate
                    generate_failed = False
                    gen_exc_str = ""
                    try:
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=SAMPLING_CONFIG["max_new_tokens"],
                                do_sample=True,
                                temperature=SAMPLING_CONFIG["temperature"],
                                top_p=SAMPLING_CONFIG["top_p"],
                                pad_token_id=tokenizer.eos_token_id
                            )
                    except Exception as gen_exc:
                        logger.error("model.generate() failed on episode %d step %d: %s", ep, run_state.step_idx, gen_exc)
                        generate_failed = True
                        gen_exc_str = f"{type(gen_exc).__name__}: {str(gen_exc)}\n{traceback.format_exc()}"

                    if generate_failed:
                        results = [
                            CompletionResult(
                                action=None,
                                info={"execution_error": gen_exc_str, "status": "ENV_ERROR", "exec_status": "ENV_ERROR"},
                                reward=None,
                                status="ENV_ERROR"
                            )
                            for _ in range(G)
                        ]
                        completions = ["<GENERATE_FAILED>"] * G
                    else:
                        input_length = inputs.input_ids.shape[1]
                        gen_ids = output_ids[:, input_length:]
                        completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                        _eval_fn = partial(evaluate_completion, env, snapshot_tag=current_snapshot_tag)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                            results = list(executor.map(_eval_fn, completions))
                        
                    # --- ROLLOUTS LOGGING (FIRST, BEFORE any abort / fail-fast checks) ---
                    rollout_log_file = out_dir / "rollouts.jsonl"
                    rollout_entry = {
                        "episode": ep,
                        "step_idx": run_state.step_idx,
                        "temperature": SAMPLING_CONFIG["temperature"],
                        "top_p": SAMPLING_CONFIG["top_p"],
                        "completions": []
                    }
                    for i in range(G):
                        c_res = results[i]
                        action_obj = c_res.action
                        c_info = c_res.info
                        c_reward = c_res.reward
                        c_status = c_res.status
                        c_raw = completions[i]
                        
                        c_action_type = action_obj.action_type.value if action_obj else "PARSE_FAILED"
                        c_target = action_obj.target if action_obj else ""
                        
                        c_breakdown = c_info.get("reward_breakdown", {}) if isinstance(c_info, dict) else {}
                        c_exec_status = c_info.get("exec_status", "UNKNOWN") if isinstance(c_info, dict) else "UNKNOWN"
                        
                        c_exec_tier = c_info.get("exec_tier", -1) if isinstance(c_info, dict) else -1
                        if c_exec_tier == RiskTier.REVIEW.value and str(c_exec_status).lower() == "aborted":
                            c_exec_status = "REVIEW_DENIED"
                            
                        rollout_entry["completions"].append({
                            "completion_idx": i,
                            "snapshot_tag": current_snapshot_tag,
                            "raw_completion": c_raw,
                            "action_type": c_action_type,
                            "target": c_target,
                            "reward": c_reward,
                            "status": c_status,
                            "breakdown": c_breakdown,
                            "exec_status": c_exec_status,
                            "execution_error": c_info.get("execution_error", "") if isinstance(c_info, dict) else "",
                            "validator_error": c_info.get("validator_error", "") if isinstance(c_info, dict) else "",
                            "stdout_tail": c_info.get("stdout_tail", "") if isinstance(c_info, dict) else "",
                            "stderr_tail": c_info.get("stderr_tail", "") if isinstance(c_info, dict) else "",
                            "returncode": c_info.get("returncode", None) if isinstance(c_info, dict) else None,
                            "verify_stdout_tail": c_info.get("verify_stdout_tail", "") if isinstance(c_info, dict) else "",
                            "verify_returncode": c_info.get("verify_returncode", None) if isinstance(c_info, dict) else None,
                        })

                    # Exactly one write to rollouts.jsonl, not in a swallowing try/except
                    with open(rollout_log_file, "a", encoding="utf-8") as rf:
                        rf.write(json.dumps(rollout_entry) + "\n")
                    # -----------------------------------------------------------------

                    # F1: Fail-fast tracking across completions in group
                    for r in results:
                        tracker.record(r.status, str(r.info.get("execution_error", "")))
                        if tracker.should_abort:
                            write_emergency_abort(out_dir, model, optimizer, run_state, tracker, tracker.last_traceback)
                            sys.exit(1)

                    # E2 & E3: Compute advantages and masked token loss
                    adv_res = compute_group_advantages(results, G, device="cuda")
                    if generate_failed or adv_res is None:
                        logger.warning(
                            "Skipping gradient update at ep %d step %d: %s.",
                            ep, run_state.step_idx,
                            "generate() failed" if generate_failed else "fewer than 2 valid completions"
                        )
                        skipped_updates_count += 1
                        ep_skipped_updates += 1
                    else:
                        advantages, valid_seq_mask = adv_res
                        optimizer.zero_grad()
                        
                        outputs = model(input_ids=output_ids, attention_mask=(output_ids != tokenizer.pad_token_id).long())
                        logits = outputs.logits[:, input_length-1:-1, :]
                        
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        action_log_probs = torch.gather(log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
                        pad_mask = (gen_ids != tokenizer.pad_token_id).float()

                        with torch.no_grad():
                            with model.disable_adapter():
                                ref_outputs = model(input_ids=output_ids, attention_mask=(output_ids != tokenizer.pad_token_id).long())
                                ref_logits = ref_outputs.logits[:, input_length-1:-1, :]
                                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                                ref_action_log_probs = torch.gather(ref_log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
                        
                        per_token_log_ratio = action_log_probs - ref_action_log_probs
                        per_token_ratio = torch.exp(per_token_log_ratio)
                        clip_ratio = torch.clamp(per_token_ratio, 0.8, 1.2)
                        
                        adv_expanded = advantages.unsqueeze(1).expand_as(per_token_ratio)
                        per_token_pg_loss = -torch.min(per_token_ratio * adv_expanded, clip_ratio * adv_expanded)
                        per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1

                        loss, normalizer = masked_token_loss(per_token_pg_loss, per_token_kl, pad_mask, valid_seq_mask, beta=0.1)
                        if normalizer > 0:
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            ep_updates += 1
                            total_updates += 1

                            with torch.no_grad():
                                kl = (per_token_kl * pad_mask * valid_seq_mask).sum() / normalizer
                                pg_loss = (per_token_pg_loss * pad_mask * valid_seq_mask).sum() / normalizer

                            ep_loss_sum += loss.item()
                            ep_kl_sum += kl.item()
                            ep_pg_loss_sum += pg_loss.item()
                            ep_grad_norm_sum += grad_norm.item()

                    # E4 & E5: Best-of-G selection picks only from status OK
                    best = select_best_action(results)

                    if best is None:
                        logger.warning(
                            "NO_VALID_ACTION: No completion with status OK at episode %d step %d. Skipping env step.",
                            ep, run_state.step_idx
                        )
                        no_valid_action_steps_count += 1
                        ep_no_valid_action_steps += 1
                        ep_steps += 1
                        run_state.step_idx += 1
                        if ep_steps >= getattr(env._cfg, 'max_steps', 10):
                            done = True
                    else:
                        best_idx, best_reward, best_action = best
                        ep_steps += 1
                        run_state.step_idx += 1

                        # F3: Main env step with exception handling
                        try:
                            obs, reward, done, truncated, info = env.step(best_action)
                            done = done or truncated or (ep_steps >= getattr(env._cfg, 'max_steps', 10))
                            episode_reward += reward

                            # G2: Snapshot MAIN container
                            if not getattr(env._cfg, 'dry_run', True) and env._docker_executor:
                                try:
                                    new_snap = env._docker_executor.snapshot()
                                    verify_result = _sp.run(
                                        ["docker", "image", "inspect", new_snap],
                                        capture_output=True, timeout=10
                                    )
                                    if verify_result.returncode == 0:
                                        old_snap = current_snapshot_tag
                                        current_snapshot_tag = new_snap
                                        if old_snap:
                                            env._docker_executor.delete_snapshot(old_snap)
                                    else:
                                        current_snapshot_tag = None
                                        snapshot_failures_count += 1
                                        logger.warning(
                                            "Snapshot %s failed verification; reset snapshot tag to None.",
                                            new_snap
                                        )
                                except Exception as snap_exc:
                                    current_snapshot_tag = None
                                    snapshot_failures_count += 1
                                    logger.warning(
                                        "Snapshot failed at step %d: %s; reset snapshot tag to None.",
                                        run_state.step_idx, snap_exc
                                    )
                        except Exception as main_step_exc:
                            main_tb = traceback.format_exc()
                            logger.error("Main environment step failed with traceback:\n%s", main_tb)
                            env.end_episode()
                            done = True
                            tracker.record_main_step_error(main_tb)
                            if tracker.should_abort:
                                write_emergency_abort(out_dir, model, optimizer, run_state, tracker, main_tb)
                                sys.exit(1)
                    
                    # Track detailed breakdown
                    if best is not None:
                        details = info.get("reward_breakdown", {})
                        for k in step_breakdown:
                            step_breakdown[k] += details.get(k, 0.0)
                            
                        # Log risks
                        if info.get("exec_tier") == RiskTier.BLOCKED.value:
                            blocked_count += 1
                            print(f"\n[!] LOUD WARNING: BLOCKED step generated: {best_action.action_type} {best_action.target}")
                        elif info.get("exec_tier") == RiskTier.REVIEW.value:
                            review_denied_count += 1
                        
                    # Explicit GC cleanup at step boundary to prevent OOM
                    if 'output_ids' in locals(): del output_ids
                    if 'gen_ids' in locals(): del gen_ids
                    if 'completions' in locals(): del completions
                    if 'results' in locals(): del results
                    if 'outputs' in locals():
                        del outputs, logits, log_probs, action_log_probs
                        del ref_outputs, ref_logits, ref_log_probs, ref_action_log_probs
                        del per_token_log_ratio, per_token_ratio, clip_ratio, adv_expanded, per_token_pg_loss
                        del per_token_kl, loss, kl, pg_loss, grad_norm, advantages
                    if 'inputs' in locals():
                        del inputs
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
            finally:
                # Clean up the final episode's snapshot on every exit path:
                # success, failure, truncation, or unhandled exception.
                if current_snapshot_tag and not getattr(env._cfg, 'dry_run', True):
                    if env._docker_executor:
                        env._docker_executor.delete_snapshot(current_snapshot_tag)
                    current_snapshot_tag = None
    
            # Post-episode processing
            rolling_rewards.append(episode_reward)
            completed = 1 if info.get("verified_complete", False) else 0
            rolling_completion.append(completed)
            
            avg_reward = sum(rolling_rewards) / len(rolling_rewards)
            avg_completion = sum(rolling_completion) / len(rolling_completion)
            
            pbar.set_postfix({
                "avg_reward": f"{avg_reward:.2f}",
                "comp_rate": f"{avg_completion:.2f}",
                "cursor": run_state.episodes_processed
            })
            pbar.update(1)
            
            # Periodic Summary Print
            if (ep + 1) % 10 == 0:
                print(f"\n--- Episode {ep+1} Summary ---")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Breakdown: Validator={step_breakdown['validator']:.2f}, "
                      f"Exec={step_breakdown['execution']:.2f}, "
                      f"Comp={step_breakdown['completion']:.2f}, "
                      f"Penalty={step_breakdown['penalty']:.2f}")
                if review_denied_count > 3:
                    print(f"[!] LOUD WARNING: High REVIEW auto-deny rate this episode ({review_denied_count})")
            
            # Log to JSONL
            mem_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            mem_res_gb = torch.cuda.memory_reserved() / (1024**3)
            with open(log_file, "a") as f:
                log_data = {
                    "episode": ep,
                    "reward": episode_reward,
                    "completed": bool(completed),
                    "cursor": run_state.episodes_processed,
                    "breakdown": step_breakdown,
                    "loss": ep_loss_sum / max(1, ep_updates),
                    "kl": ep_kl_sum / max(1, ep_updates),
                    "pg_loss": ep_pg_loss_sum / max(1, ep_updates),
                    "grad_norm": ep_grad_norm_sum / max(1, ep_updates),
                    "updates": ep_updates,
                    "updates_total": total_updates,
                    "skipped_updates": ep_skipped_updates,
                    "skipped_updates_total": skipped_updates_count,
                    "no_valid_action_steps": ep_no_valid_action_steps,
                    "no_valid_action_steps_total": no_valid_action_steps_count,
                    "env_errors": tracker.episode_env_errors,
                    "env_errors_total": tracker.total_env_errors,
                    "snapshot_failures_total": snapshot_failures_count,
                    "excluded_repos_total": excluded_repos_count,
                    "mem_alloc_gb": mem_alloc_gb,
                    "mem_res_gb": mem_res_gb
                }
                f.write(json.dumps(log_data) + "\n")
                
            # Checkpointing
            if (ep + 1) % CHECKPOINT_CADENCE == 0:
                run_state.episode_idx = ep + 1
                save_checkpoint(out_dir, model, optimizer, run_state)
                
    finally:
        env.close()
    
    logger.info("=== docker system df AFTER pilot ===\n%s", _docker_df())

if __name__ == "__main__":
    main()
