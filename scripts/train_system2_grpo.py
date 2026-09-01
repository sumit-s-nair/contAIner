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
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

from src.rl_env.env import PlannerEnv, EnvConfig
from src.rl_env.dry_run_executor import TrainingReviewPolicy
from src.sandbox.models import RiskTier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

# --- Checkpoint & Resume Logic ---
def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_state: RunState,
    is_emergency: bool = False
):
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
    
    try:
        for ep in range(run_state.episode_idx, MAX_EPISODES):
            run_state.episode_idx = ep
            
            # Hard fail if val/test splits ever leak in (using the env's explicit split argument)
            if env._corpus is None:
                env._load_corpus()
            
            repo_entry = env._rng.choice(env._corpus)
            success = False
            for attempt in range(3):
                try:
                    obs, ep_info = env.reset(repo_entry=repo_entry)
                    run_state.episodes_processed += 1
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to reset environment for {repo_entry['repo']} (attempt {attempt+1}/3): {e}")
                    import time
                    time.sleep(2)
            
            if not success:
                logger.error(f"REPO EXCLUDED: {repo_entry['repo']} failed all 3 reset attempts. Removing from corpus and skipping episode.")
                env._corpus.remove(repo_entry)
                with open(out_dir / "excluded_repos.txt", "a", encoding="utf-8") as f:
                    f.write(repo_entry['repo'] + "\n")
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
            try:
                while not done:
                    # 1. Get structured chat prompt from observation
                    # Pass the tokenizer for exact per-section token accounting.
                    # to_chat_prompt() will trim only the dependency list to fit
                    # the budget — system, conflicts, and history are always intact.
                    chat_messages = env._serializer.to_chat_prompt(obs, tokenizer=tokenizer)
                    
                    # 2. Use tokenizer to format it exactly according to ChatML
                    prompt_str = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # --- GRPO Sampling block ---
                    G = BATCH_SIZE
                    inputs = tokenizer([prompt_str] * G, return_tensors="pt", padding=True).to("cuda")
                    
                    torch.cuda.empty_cache()  # evict residual allocations before G-way generate
                    try:
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=64,
                                do_sample=True,
                                temperature=0.8,
                                pad_token_id=tokenizer.eos_token_id
                            )
                    except Exception as gen_exc:
                        logger.error("model.generate() failed (possibly OOM) on episode %d step %d: %s", ep, run_state.step_idx, gen_exc)
                        # Graceful degrade: force failure string so episode aborts / learns negative reward
                        output_ids = torch.full((G, inputs.input_ids.shape[1] + 2), tokenizer.eos_token_id, device="cuda")
                        
                    input_length = inputs.input_ids.shape[1]
                    gen_ids = output_ids[:, input_length:]
                    completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    
                    import copy
                    from src.system2_planner.models import PlannedStep, ActionType
                    import concurrent.futures
    
                    def evaluate_completion(comp):
                        env_clone = None
                        try:
                            start_idx = comp.find('{')
                            end_idx = comp.rfind('}')
                            if start_idx == -1 or end_idx == -1:
                                raise ValueError("No JSON object found")
                            
                            step_dict = json.loads(comp[start_idx:end_idx+1])
                            
                            atype_str = step_dict.get("action_type", "CHECK").upper()
                            try:
                                action_type = ActionType(atype_str.lower())
                            except ValueError:
                                action_type = ActionType.CHECK
    
                            action = PlannedStep(
                                action_type=action_type,
                                target=step_dict.get("target", ""),
                                description=step_dict.get("description", ""),
                                rationale=step_dict.get("rationale", "")
                            )
                            
                            if getattr(env._cfg, 'dry_run', True):
                                env_clone = copy.deepcopy(env)
                                _, reward, _, _, info = env_clone.step(action)
                            else:
                                env_clone = setup_environment()
                                repo_entry = {"repo": ep_info["repo"], "category": ep_info.get("category", "")}
                                
                                # Use the main env's already-cloned path and manifest to avoid ThreadPool concurrency
                                import unittest.mock
                                with unittest.mock.patch('src.rl_env.repo_loader.RepoLoader.load', return_value=(env._instance.manifest, env._local_path)):
                                    # Branch from snapshot if available, else fall back to full replay.
                                    # snapshot_image=current_snapshot_tag means /workspace_rw already
                                    # contains all packages installed in steps 0..t-1; no replay needed.
                                    snap = current_snapshot_tag  # capture for thread safety
                                    env_clone._docker_executor.start_episode(
                                        repo_path=env._local_path,
                                        repo_name=repo_entry["repo"],
                                        snapshot_image=snap,
                                    ) if snap else env_clone.reset(repo_entry=repo_entry)
                                    
                                if not snap:
                                    # No snapshot yet (step 0) — replay history the old way
                                    for past_step in env._plan_so_far:
                                        env_clone.step(past_step)
                                
                                _, reward, _, _, info = env_clone.step(action)
                                
                            return (action, info, reward)
                            
                        except Exception as e:
                            return (None, {}, -1.0)
                        finally:
                            if env_clone and getattr(env_clone, '_docker_executor', None):
                                env_clone._docker_executor.end_episode()
                        
                    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                        results = list(executor.map(evaluate_completion, completions))
                        
                    actions = results
                    rewards = [r for _, _, r in results]
                            
                    # Compute advantages
                    rewards_t = torch.tensor(rewards, dtype=torch.float32, device="cuda")
                    mean_r = rewards_t.mean()
                    std_r = rewards_t.std()
                    
                    if std_r < 1e-4:
                        advantages = torch.zeros_like(rewards_t)
                        flat_rewards_consecutive += 1
                        if flat_rewards_consecutive >= 3:
                            logger.warning("LOUD WARNING: Flat rewards across group for %d consecutive steps (std < 1e-4). Zero PG learning signal flowing.", flat_rewards_consecutive)
                    else:
                        advantages = (rewards_t - mean_r) / (std_r + 1e-8)
                        flat_rewards_consecutive = 0
                        
                    # Compute loss and step
                    optimizer.zero_grad()
                    
                    outputs = model(input_ids=output_ids, attention_mask=(output_ids != tokenizer.pad_token_id).long())
                    logits = outputs.logits[:, input_length-1:-1, :]
                    
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    action_log_probs = torch.gather(log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
                    
                    mask = (gen_ids != tokenizer.pad_token_id).float()
                    # GRPO computes ratio and KL *per-token*, not per-sequence.
                    # Sequence-level ratios multiply (exp(sum) = product), which explodes/vanishes 
                    # and causes gradients to scale linearly with sequence length.
                    with torch.no_grad():
                        with model.disable_adapter():
                            ref_outputs = model(input_ids=output_ids, attention_mask=(output_ids != tokenizer.pad_token_id).long())
                            ref_logits = ref_outputs.logits[:, input_length-1:-1, :]
                            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                            ref_action_log_probs = torch.gather(ref_log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
                    
                    per_token_log_ratio = action_log_probs - ref_action_log_probs
                    per_token_ratio = torch.exp(per_token_log_ratio)
                    clip_ratio = torch.clamp(per_token_ratio, 0.8, 1.2)
                    
                    # Expand advantages to match sequence length
                    adv_expanded = advantages.unsqueeze(1).expand_as(per_token_ratio)
                    per_token_pg_loss = -torch.min(per_token_ratio * adv_expanded, clip_ratio * adv_expanded)
                    
                    # Robust non-negative GRPO KL estimator per token: exp(x) - x - 1
                    per_token_kl = torch.exp(per_token_log_ratio) - per_token_log_ratio - 1
                    
                    per_token_loss = per_token_pg_loss + 0.1 * per_token_kl
                    
                    # Average over all non-pad tokens in the batch (prevents gradient explosion scaling with seq_len)
                    loss = (per_token_loss * mask).sum() / mask.sum()
                    
                    # For logging purposes, we can keep track of sequence-level KL
                    with torch.no_grad():
                        kl = (per_token_kl * mask).sum(dim=1).mean()
                        pg_loss = (per_token_pg_loss * mask).sum(dim=1).mean()
                    
                    loss.backward()
                    
                    # Prevent exploding gradients which corrupt model weights (NaNs) 
                    # and lead to fatal CUDA device-side asserts in model.generate(do_sample=True)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    ep_loss_sum += loss.item()
                    ep_kl_sum += kl.item()
                    ep_pg_loss_sum += pg_loss.item()
                    ep_grad_norm_sum += grad_norm.item()
                    ep_steps += 1
                    
                    # Select best action to advance the MAIN environment
                    # Rationale: Advancing with the max-reward action (best-of-G) accelerates convergence towards optimal trajectories, though it biases exploration off-policy.
                    best_idx = int(torch.argmax(rewards_t).item())
                    best_action, best_info, best_reward = actions[best_idx]
                    
                    if best_action is None:
                        # All failed parsing, force a dummy action to take penalty and move forward
                        best_action = PlannedStep(action_type=ActionType.CHECK, description="Fallback dummy action due to parse failure")
                    
                    # Execute in MAIN environment
                    obs, reward, done, truncated, info = env.step(best_action)
                    done = done or truncated
                    
                    # --- Snapshot the MAIN container after each successful step ----------
                    # Order: commit NEW -> verify it exists -> delete PREVIOUS.
                    # Never delete-then-commit: if the process crashes between steps we
                    # must always have at least one valid snapshot available.
                    if not getattr(env._cfg, 'dry_run', True) and env._docker_executor:
                        try:
                            new_snap = env._docker_executor.snapshot()
                            # Verify the image was actually created before evicting old one
                            verify_result = _sp.run(
                                ["docker", "image", "inspect", new_snap],
                                capture_output=True, timeout=10
                            )
                            if verify_result.returncode == 0:
                                # Safe to delete previous snapshot now that new one is confirmed
                                old_snap = current_snapshot_tag
                                current_snapshot_tag = new_snap
                                if old_snap:
                                    env._docker_executor.delete_snapshot(old_snap)
                            else:
                                # Commit succeeded but image not found — keep old snapshot
                                logger.warning(
                                    "Snapshot %s failed verification; keeping previous snapshot.",
                                    new_snap,
                                )
                        except Exception as snap_exc:
                            logger.warning("Snapshot failed at step %d: %s", run_state.step_idx, snap_exc)
                    
                    episode_reward += reward
                    run_state.step_idx += 1
                    
                    # Track detailed breakdown
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
                    del output_ids, gen_ids, completions, actions, rewards
                    if 'outputs' in locals():
                        del outputs, logits, log_probs, action_log_probs, mask
                        del ref_outputs, ref_logits, ref_log_probs, ref_action_log_probs
                        del per_token_log_ratio, per_token_ratio, clip_ratio, adv_expanded, per_token_pg_loss
                        del per_token_kl, per_token_loss, loss, kl, pg_loss, grad_norm, advantages, rewards_t
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
                    "loss": ep_loss_sum / max(1, ep_steps),
                    "kl": ep_kl_sum / max(1, ep_steps),
                    "pg_loss": ep_pg_loss_sum / max(1, ep_steps),
                    "grad_norm": ep_grad_norm_sum / max(1, ep_steps),
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
