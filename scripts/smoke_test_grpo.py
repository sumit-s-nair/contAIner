import os
import torch
import copy
import json
import random
import logging
from pathlib import Path
from src.rl_env.env import PlannerEnv, Observation, EnvConfig, TrainingReviewPolicy
from src.sandbox.models import AtomicStep
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing smoke test for GRPO loop...")
    G = 4
    EPISODES = 2
    
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    env_cfg = EnvConfig(dry_run=True, review_policy=TrainingReviewPolicy.AUTO_DENY)
    env = PlannerEnv(config=env_cfg)
    
    # Capture a weight before
    original_weight = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data.clone()
    
    for ep in range(EPISODES):
        logger.info(f"--- Episode {ep+1}/{EPISODES} ---")
        obs, _ = env.reset()
        prompt_dicts = env._serializer.to_chat_prompt(obs)
        prompt_str = tokenizer.apply_chat_template(prompt_dicts, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([prompt_str] * G, return_tensors="pt", padding=True).to("cuda")
        
        logger.info(f"Generating {G} completions...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs.input_ids.shape[1]
        gen_ids = output_ids[:, input_length:]
        completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        rewards = []
        for i, comp in enumerate(completions):
            # 3. Parse action and interact with env
            try:
                # Basic JSON extraction (assume model outputs a single JSON object)
                start_idx = comp.find('{')
                end_idx = comp.rfind('}')
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON object found")
                
                step_dict = json.loads(comp[start_idx:end_idx+1])
                
                from src.system2_planner.models import PlannedStep, ActionType
                
                # Use upper case to map string to enum, default to CHECK if missing or invalid
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
                
                # We need to branch the environment to evaluate each completion independently.
                # In a real rollout, we'd use a batched environment or deepcopy.
                import copy
                env_clone = copy.deepcopy(env)
                _, reward, _, _, _ = env_clone.step(action)
                rewards.append(reward)
                
                logger.info(f"Completion {i} parsed! Action: {action.action_type} {action.target}... Reward: {reward}")
            except Exception as e:
                rewards.append(-1.0)
                logger.info(f"Completion {i} parse FAILED. Error: {e}. Raw: {comp[:40].replace(chr(10), ' ')}... Reward: -1.0")
                
        # 4. Compute advantages
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        
        if std_r < 1e-4:
            advantages = torch.zeros_like(rewards_t)
        else:
            advantages = (rewards_t - mean_r) / std_r
        
        logger.info(f"Rewards: {rewards_t.tolist()}")
        logger.info(f"Advantages: {advantages.tolist()}")
        
        # 5. Compute loss and step
        optimizer.zero_grad()
        
        outputs = model(input_ids=output_ids, attention_mask=(output_ids != tokenizer.pad_token_id).long())
        logits = outputs.logits[:, input_length-1:-1, :]
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
        
        mask = (gen_ids != tokenizer.pad_token_id).float()
        sequence_log_probs = (action_log_probs * mask).sum(dim=1)
        
        with torch.no_grad():
            with model.disable_adapter():
                ref_outputs = model(input_ids=output_ids, attention_mask=(output_ids != tokenizer.pad_token_id).long())
                ref_logits = ref_outputs.logits[:, input_length-1:-1, :]
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                ref_action_log_probs = torch.gather(ref_log_probs, 2, gen_ids.unsqueeze(-1)).squeeze(-1)
                ref_sequence_log_probs = (ref_action_log_probs * mask).sum(dim=1)
        
        ratio = torch.exp(sequence_log_probs - ref_sequence_log_probs)
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        pg_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        kl = (ref_sequence_log_probs - sequence_log_probs).mean()
        loss = pg_loss + 0.1 * kl
        
        logger.info(f"Loss: {loss.item():.4f} (PG: {pg_loss.item():.4f}, KL: {kl.item():.4f})")
        
        loss.backward()
        optimizer.step()
        
        logger.info(f"Step {ep+1} completed.\n")
        
    new_weight = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data
    changed = not torch.allclose(original_weight, new_weight)
    logger.info(f"Weights changed after GRPO steps? {changed}")
    if changed:
        diff = torch.abs(original_weight - new_weight).mean().item()
        logger.info(f"Mean absolute weight difference: {diff:.6f}")
    
    assert changed, "Weights did not change, backward pass failed!"
    logger.info("End-to-End GRPO Smoke Test Passed.")

if __name__ == "__main__":
    main()
