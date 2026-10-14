import json
import matplotlib.pyplot as plt
from pathlib import Path
import collections
import numpy as np

def moving_average(data, window_size=50):
    if not data:
        return []
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'valid')

def main():
    log_file = Path("outputs/system2_grpo_run_669565/training_log.jsonl")
    manifest_file = Path("datasets/repo_corpus/corpus_manifest_v3.json")
    
    # Load manifest to resolve categories
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    train_repos = manifest["splits"]["train"]
    num_train = len(train_repos)
    
    episodes = []
    rewards = []
    pg_losses = []
    kls = []
    grad_norms = []
    
    # Dictionaries to hold rewards by category
    category_rewards = collections.defaultdict(list)
    category_episodes = collections.defaultdict(list)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ep = data["episode"]
            reward = data["reward"]
            
            # ep is 1-indexed in the log
            # The code in train uses ep_idx = ep - 1
            ep_idx = ep - 1
            category = train_repos[ep_idx % num_train]["category"]
            
            episodes.append(ep)
            rewards.append(reward)
            pg_losses.append(data.get("pg_loss", 0.0))
            kls.append(data.get("kl", 0.0))
            grad_norms.append(data.get("grad_norm", 0.0))
            
            category_rewards[category].append(reward)
            category_episodes[category].append(ep)
            
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("System 2 GRPO Training Curves (1000 Episodes)", fontsize=16)
    
    # 1. Total Reward
    axs[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Reward')
    if len(rewards) >= 50:
        axs[0, 0].plot(episodes[49:], moving_average(rewards, 50), color='darkblue', linewidth=2, label='50-ep MA')
    axs[0, 0].set_title("Total Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()
    
    # 2. PG Loss
    axs[0, 1].plot(episodes, pg_losses, alpha=0.5, color='red')
    if len(pg_losses) >= 50:
        axs[0, 1].plot(episodes[49:], moving_average(pg_losses, 50), color='darkred', linewidth=2)
    axs[0, 1].set_title("Policy Gradient Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("PG Loss")
    
    # 3. KL Divergence
    axs[1, 0].plot(episodes, kls, alpha=0.5, color='green')
    if len(kls) >= 50:
        axs[1, 0].plot(episodes[49:], moving_average(kls, 50), color='darkgreen', linewidth=2)
    axs[1, 0].set_title("KL Divergence")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("KL")
    
    # 4. Grad Norm
    axs[1, 1].plot(episodes, grad_norms, alpha=0.5, color='orange')
    if len(grad_norms) >= 50:
        axs[1, 1].plot(episodes[49:], moving_average(grad_norms, 50), color='darkorange', linewidth=2)
    axs[1, 1].set_title("Gradient Norm")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Grad Norm")
    
    # 5. Per-Category Reward MA
    for cat, rews in category_rewards.items():
        eps = category_episodes[cat]
        window = min(20, len(rews))
        if window > 0:
            ma_rews = moving_average(rews, window)
            axs[2, 0].plot(eps[window-1:], ma_rews, label=f"{cat} ({window}-ep MA)")
    
    axs[2, 0].set_title("Per-Category Reward")
    axs[2, 0].set_xlabel("Episode")
    axs[2, 0].set_ylabel("Reward")
    axs[2, 0].legend()
    
    # 6. Empty
    axs[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_dir = Path("results/system2_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "training_curves.png"
    plt.savefig(out_file)
    print(f"Saved training curves to {out_file}")
    
    # Simple text summary
    summary_md = f"""
# Training Summary (Episodes: {len(episodes)})

**Final 50 Episodes Average:**
- Reward: {np.mean(rewards[-50:]):.3f}
- PG Loss: {np.mean(pg_losses[-50:]):.3f}
- KL Divergence: {np.mean(kls[-50:]):.3f}
- Grad Norm: {np.mean(grad_norms[-50:]):.3f}

**Category Final Averages (Last 20 encounters):**
"""
    for cat, rews in category_rewards.items():
        summary_md += f"- {cat}: {np.mean(rews[-20:]):.3f}\n"
        
    with open(out_dir / "training_summary.md", "w") as f:
        f.write(summary_md)
    print("Saved training_summary.md")

if __name__ == "__main__":
    main()
