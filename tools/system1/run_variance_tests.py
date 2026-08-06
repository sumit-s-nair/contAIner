import os
import sys
import json
import subprocess
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    seeds = [42, 100, 2026, 888, 1234]
    models_dir = "outputs/intent_classifier_variance"

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "system1_intent_understanding")))
    from scripts.generate_ner_stats import evaluate_ner, test_data

    all_results = {}

    for i, seed in enumerate(seeds):
        run_dir = os.path.join(models_dir, f"run_seed{seed}")
        print(f"--- Training seed {seed} ({i+1}/{len(seeds)}) ---")
        
        model_path = os.path.join(run_dir, "final_model")
        
        if not os.path.exists(model_path):
            cmd = [
                sys.executable,
                "src/system1_intent_understanding/train_intent_classifier.py",
                "--seed", str(seed),
                "--output", run_dir,
                "--epochs", "8",
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"Training failed for seed {seed}")
                continue
        
        model_path = os.path.join(run_dir, "final_model")
        
        print(f"--- Evaluating seed {seed} ---")
        m_test, _ = evaluate_ner(model_path, test_data)
        
        out_file = f"artifacts/results/variance_run_seed{seed}.json"
        os.makedirs("artifacts/results", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(m_test, f, indent=2, cls=NpEncoder)
        
        all_results[seed] = m_test

    summary = {}
    metrics_to_track = ["intent_accuracy", "precision", "recall", "f1"]
    entity_types = ["software", "version", "project", "file"]

    metric_values = {m: [] for m in metrics_to_track}
    entity_values = {e: {"precision": [], "recall": [], "f1-score": []} for e in entity_types}

    for seed, res in all_results.items():
        for m in metrics_to_track:
            metric_values[m].append(res[m])
        
        report = res.get("report", {})
        for e in entity_types:
            if e in report:
                entity_values[e]["precision"].append(report[e]["precision"])
                entity_values[e]["recall"].append(report[e]["recall"])
                entity_values[e]["f1-score"].append(report[e]["f1-score"])
            else:
                entity_values[e]["precision"].append(0.0)
                entity_values[e]["recall"].append(0.0)
                entity_values[e]["f1-score"].append(0.0)

    for m in metrics_to_track:
        summary[m] = {
            "mean": float(np.mean(metric_values[m])),
            "std": float(np.std(metric_values[m]))
        }

    summary["per_entity"] = {}
    for e in entity_types:
        summary["per_entity"][e] = {
            "precision": {
                "mean": float(np.mean(entity_values[e]["precision"])),
                "std": float(np.std(entity_values[e]["precision"]))
            },
            "recall": {
                "mean": float(np.mean(entity_values[e]["recall"])),
                "std": float(np.std(entity_values[e]["recall"]))
            },
            "f1-score": {
                "mean": float(np.mean(entity_values[e]["f1-score"])),
                "std": float(np.std(entity_values[e]["f1-score"]))
            }
        }

    with open("artifacts/results/variance_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)

    print("Variance testing complete.")

if __name__ == "__main__":
    main()
