import os
import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from pydantic import ValidationError # keep if needed, otherwise ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.rl_env.env import PlannerEnv, EnvConfig, TrainingReviewPolicy
from src.system2_planner.models import PlannedStep, ActionType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_base_model():
    logger.info("Loading Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    return model, tokenizer

def load_trained_model():
    logger.info("Loading Trained Model (LoRA Adapter)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "outputs/system2_grpo_run_669565/checkpoint-1000")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    return model, tokenizer

def extract_step_dict(completion_text: str):
    start_idx = completion_text.find('{')
    end_idx = completion_text.rfind('}')
    if start_idx == -1 or end_idx == -1:
        return None
    try:
        return json.loads(completion_text[start_idx:end_idx+1])
    except json.JSONDecodeError:
        return None

def evaluate_model(model_name: str, model, tokenizer, test_repos, demo_repos, out_dir: Path):
    logger.info(f"Evaluating {model_name}...")
    env_cfg = EnvConfig(dry_run=False, review_policy=TrainingReviewPolicy.AUTO_DENY, corpus_manifest_path="datasets/repo_corpus/corpus_manifest_v3.json")
    env = PlannerEnv(config=env_cfg)
    
    metrics = {
        "total_episodes": 0,
        "valid_json": 0,
        "valid_structure": 0,
        "execution_success": 0, # >0 execution reward
        "full_completion": 0, # episode completed
        "total_steps": 0,
        "env_errors": 0,
        "failures": {
            "invalid_json": 0,
            "invalid_structure": 0,
            "execution_failed": 0,
            "max_steps_reached": 0
        },
        "by_category": {}
    }
    
    demo_dir = out_dir / "demo_traces"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    for repo in tqdm(test_repos):
        repo_name = repo["repo"]
        cat = repo["category"]
        
        is_demo = repo_name in demo_repos
        trace_log = []
        if is_demo:
            trace_log.append(f"# Demo Trace: {repo_name} ({cat})\n**Model**: {model_name}\n\n")
            
        try:
            obs, _ = env.reset(repo_entry=repo)
        except Exception as e:
            logger.error(f"Failed to reset env for {repo_name}: {e}")
            metrics["env_errors"] += 1
            if is_demo:
                trace_log.append(f"**Result:** FAILED - Env Reset Error: {e}\n")
                safe_name = repo_name.replace("/", "_")
                with open(demo_dir / f"{model_name}_{safe_name}.md", "w") as f:
                    f.write("\n".join(trace_log))
            continue
        
        # Local metrics accumulated for this episode
        ep_valid_json = 0
        ep_valid_structure = 0
        ep_execution_success = 0
        ep_steps_taken = 0
        ep_failures = {
            "invalid_json": 0,
            "invalid_structure": 0,
            "execution_failed": 0,
            "max_steps_reached": 0
        }
        ep_reward = 0.0
        env_error_occurred = False
        failure_logged = False
        terminated = False
        info = {}
        
        for step_idx in range(10): # max 10 steps
            prompt_dicts = env._serializer.to_chat_prompt(obs)
            prompt_str = tokenizer.apply_chat_template(prompt_dicts, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer([prompt_str], return_tensors="pt").to("cuda")
            
            if is_demo:
                import dataclasses
                trace_log.append(f"## Step {step_idx+1}\n")
                trace_log.append(f"**Observation:**\n```json\n{json.dumps(dataclasses.asdict(obs), indent=2)}\n```\n")
                
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False, # greedy decoding for eval
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            if is_demo:
                trace_log.append(f"**Raw LLM Output:**\n```json\n{completion}\n```\n")
            
            step_dict = extract_step_dict(completion)
            ep_steps_taken += 1
            
            if step_dict is None:
                if not failure_logged:
                    ep_failures["invalid_json"] += 1
                    failure_logged = True
                if is_demo: trace_log.append("**Result:** FAILED - Invalid JSON\n")
                break
                
            ep_valid_json += 1
            
            # Strict structural validation
            if "action_type" not in step_dict or "description" not in step_dict:
                if not failure_logged:
                    ep_failures["invalid_structure"] += 1
                    failure_logged = True
                if is_demo: trace_log.append("**Result:** FAILED - Structural Validation Error (Missing required fields)\n")
                break
            
            try:
                action_type = ActionType(step_dict["action_type"].lower())
            except ValueError:
                if not failure_logged:
                    ep_failures["invalid_structure"] += 1
                    failure_logged = True
                if is_demo: trace_log.append(f"**Result:** FAILED - Structural Validation Error (Invalid action_type: {step_dict['action_type']})\n")
                break
                
            action = PlannedStep(
                action_type=action_type,
                target=step_dict.get("target", ""),
                description=step_dict["description"],
                rationale=step_dict.get("rationale", "")
            )
            ep_valid_structure += 1
                
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                logger.error(f"Env step failed for {repo_name}: {e}")
                metrics["env_errors"] += 1
                env_error_occurred = True
                if is_demo: trace_log.append(f"**Result:** FAILED - Env Step Error: {e}\n")
                break

            ep_reward += reward
            
            # Check for blocked or execution failed directly from info
            is_blocked = info.get("blocked", False)
            
            if is_blocked:
                if not failure_logged:
                    ep_failures["execution_failed"] += 1
                    failure_logged = True
                if is_demo: trace_log.append(f"**Result:** FAILED - Action Blocked (AUTO_DENY). Reward: {reward}\n")
                break
                
            step_exec_reward = info.get("reward_breakdown", {}).get("execution", 0.0)
            if step_exec_reward > 0 or action.action_type.value == "check":
                ep_execution_success += 1
            else:
                # If execution failed for a non-blocked action
                if not failure_logged:
                    ep_failures["execution_failed"] += 1
                    failure_logged = True
                if is_demo: trace_log.append(f"**Result:** FAILED - Execution/Verification Failed. Reward: {reward}\n")
                break
                
            if is_demo:
                trace_log.append(f"**Step Reward:** {reward} (Breakdown: {info.get('reward_breakdown', {})})\n")
                
            if terminated or truncated:
                break
                
        if env_error_occurred:
            # An episode with an env error contributes ONLY to env_errors
            if is_demo:
                safe_name = repo_name.replace("/", "_")
                with open(demo_dir / f"{model_name}_{safe_name}.md", "w") as f:
                    f.write("\n".join(trace_log))
            continue

        # Commit local episode metrics into global metrics
        if cat not in metrics["by_category"]:
            metrics["by_category"][cat] = {"total": 0, "completed": 0, "reward": 0.0}
            
        metrics["total_episodes"] += 1
        metrics["by_category"][cat]["total"] += 1
        metrics["valid_json"] += ep_valid_json
        metrics["valid_structure"] += ep_valid_structure
        metrics["execution_success"] += ep_execution_success
        metrics["total_steps"] += ep_steps_taken
        metrics["by_category"][cat]["reward"] += ep_reward
        for k in ep_failures:
            metrics["failures"][k] += ep_failures[k]
        
        if info.get("episode_complete", False) and not failure_logged:
            metrics["full_completion"] += 1
            metrics["by_category"][cat]["completed"] += 1
            if is_demo: trace_log.append("## Status: SUCCESS\nEpisode completed successfully.")
        else:
            if not failure_logged:
                if info.get("blocked", False):
                    metrics["failures"]["execution_failed"] += 1
                else:
                    metrics["failures"]["max_steps_reached"] += 1
            if is_demo: trace_log.append("## Status: FAILED\nEpisode terminated without completion.")
            
        if is_demo:
            safe_name = repo_name.replace("/", "_")
            with open(demo_dir / f"{model_name}_{safe_name}.md", "w") as f:
                f.write("\n".join(trace_log))
                
    logger.info(
        f"Evaluation finished for {model_name}: "
        f"total_episodes={metrics['total_episodes']}, "
        f"completed={metrics['full_completion']}, "
        f"execution_failed={metrics['failures']['execution_failed']}, "
        f"env_errors={metrics['env_errors']}"
    )
    return metrics

def main():
    logger.info("Starting System 2 Evaluation...")
    out_dir = Path("results/system2_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open("datasets/repo_corpus/corpus_manifest_v3.json", "r") as f:
        manifest = json.load(f)
        
    test_repos = manifest["splits"]["test"]
    
    # Filter to only repos that actually exist in the test split
    test_repo_names = [r["repo"] for r in test_repos]
    demo_repos = test_repo_names[:5] # just pick the first 5
        
    logger.info(f"Demo repos selected: {demo_repos}")
    
    all_metrics = {}
    
    # Base Model
    model, tokenizer = load_base_model()
    base_metrics = evaluate_model("Base_Untrained", model, tokenizer, test_repos, demo_repos, out_dir)
    all_metrics["Base_Untrained"] = base_metrics
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    # Trained Model
    model, tokenizer = load_trained_model()
    trained_metrics = evaluate_model("Trained_GRPO", model, tokenizer, test_repos, demo_repos, out_dir)
    all_metrics["Trained_GRPO"] = trained_metrics
    
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    logger.info("Evaluation complete! Metrics saved.")

if __name__ == "__main__":
    main()
