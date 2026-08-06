import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    with open("artifacts/results/variance_summary.json", "r") as f:
        summary = json.load(f)
        
    metrics = ["intent_accuracy", "f1"]
    labels = ["Intent Accuracy", "NER F1 (Overall)"]
    
    means = [summary[m]["mean"] for m in metrics]
    stds = [summary[m]["std"] for m in metrics]
    
    per_entity = summary.get("per_entity", {})
    entity_names = list(per_entity.keys())
    ent_means = [per_entity[e]["f1-score"]["mean"] for e in entity_names]
    ent_stds = [per_entity[e]["f1-score"]["std"] for e in entity_names]
    
    all_labels = labels + [f"NER: {e}" for e in entity_names]
    all_means = means + ent_means
    all_stds = stds + ent_stds
    
    x_pos = np.arange(len(all_labels))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bars = ax.bar(x_pos, all_means, yerr=all_stds, align='center', alpha=0.8, ecolor='black', capsize=10, 
                  color=['#1f77b4', '#ff7f0e'] + ['#2ca02c']*len(entity_names))
    
    ax.set_ylabel('Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_title('Model Performance & Variance Across 5 Seeds')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Set tight limits so variance is visible
    min_val = min(all_means) - max(all_stds) - 0.002
    ax.set_ylim([max(0.0, min_val), 1.002])
    
    # Add values
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval - (yval - min_val)*0.1, f'{yval:.4f}', 
                ha='center', va='top', color='white', fontweight='bold', rotation=90)
    
    plt.tight_layout()
    os.makedirs("artifacts/results", exist_ok=True)
    plt.savefig("artifacts/results/variance_plot.png", dpi=300)
    print("Graph saved to artifacts/results/variance_plot.png")

if __name__ == "__main__":
    main()
