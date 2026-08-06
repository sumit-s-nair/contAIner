import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoConfig

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

print("Starting NER stats generation...")

# We need the model class to load it
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src" / "system1_intent_understanding"))
from src.system1_intent_understanding.verify_intent_classifier import JointIntentNER

def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

import os

def load_data_fallback(path, fallback_path):
    if os.path.exists(path):
        return load_data(path)
    return load_data(fallback_path)

train_data = load_data_fallback("datasets/intent-dataset/data/train_corrected.jsonl", "datasets/intent-dataset/data/train.jsonl")
val_data = load_data_fallback("datasets/intent-dataset/data/validation_corrected.jsonl", "datasets/intent-dataset/data/validation.jsonl")
test_data = load_data_fallback("datasets/intent-dataset/data/test_corrected.jsonl", "datasets/intent-dataset/data/test.jsonl")

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=100, ci=95):
    np.random.seed(42)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        yt = [y_true[i] for i in indices]
        yp = [y_pred[i] for i in indices]
        try:
            scores.append(metric_fn(yt, yp))
        except:
            pass
    if not scores:
        return 0, 0, 0
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    low = np.percentile(scores, (100 - ci) / 2)
    high = np.percentile(scores, 100 - (100 - ci) / 2)
    return float(std), float(low), float(high)

def evaluate_ner(model_dir, dataset):
    model_dir = Path(model_dir)
    print(f"Loading {model_dir}")
    
    with open(model_dir / "intent_label_map.json") as f:
        intent_label_map = json.load(f)
    num_intent_labels = len(intent_label_map)
    
    ner_map_path = model_dir / "ner_label_map.json"
    if ner_map_path.exists():
        with open(ner_map_path) as f:
            id2label = {int(k): v for k, v in json.load(f).get("id2label", {}).items()}
    else:
        # Fallback
        labels = ["O"] + [f"B-{e}" for e in ["runtime", "package", "version", "virtual_env", "package_manager", "project"]] + \
                 [f"I-{e}" for e in ["runtime", "package", "version", "virtual_env", "package_manager", "project"]]
        id2label = {i: l for i, l in enumerate(labels)}
        
    num_ner_labels = len(id2label)
    
    cfg_path = model_dir / "training_config.json"
    max_length = 128
    if cfg_path.exists():
        with open(cfg_path) as f:
            max_length = int(json.load(f).get("max_length", 128))
            
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    config = AutoConfig.from_pretrained(str(model_dir))
    model = JointIntentNER.from_pretrained(
        str(model_dir), config=config, 
        num_intent_labels=num_intent_labels, num_ner_labels=num_ner_labels
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_true_seqs = []
    all_pred_seqs = []
    all_true_intents = []
    all_pred_intents = []
    
    print(f"  Evaluating {len(dataset)} items...")
    
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        texts = [row["instruction"] for row in batch]
        spans_list = [row.get("entity_spans", {}) for row in batch]
        
        inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_offsets_mapping=True, return_tensors="pt")
        offsets_batch = inputs.pop("offset_mapping").tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            ner_ids_batch = torch.argmax(outputs.ner_logits, dim=-1).tolist()
            intent_ids_batch = torch.argmax(outputs.intent_logits, dim=-1).tolist()
            
        for b_idx in range(len(batch)):
            offsets = offsets_batch[b_idx]
            ner_ids = ner_ids_batch[b_idx]
            spans = spans_list[b_idx]
            
            true_intent_label = batch[b_idx].get("intent_type")
            true_intent_id = intent_label_map.get(true_intent_label, -1)
            pred_intent_id = intent_ids_batch[b_idx]
            
            all_true_intents.append(true_intent_id)
            all_pred_intents.append(pred_intent_id)
            
            pred_seq = []
            true_seq = []
            
            for idx, (start, end) in enumerate(offsets):
                if start == 0 and end == 0:
                    continue
                    
                # determine true tag
                true_tag = "O"
                for etype, span_info in spans.items():
                    if not span_info: continue
                    s_start, s_end = span_info.get("start"), span_info.get("end")
                    if s_start is None or s_end is None: continue
                    
                    if start >= s_start and end <= s_end:
                        if start == s_start:
                            true_tag = f"B-{etype}"
                        else:
                            true_tag = f"I-{etype}"
                        break
                        
                pred_tag = id2label.get(ner_ids[idx], "O")
                
                true_seq.append(true_tag)
                pred_seq.append(pred_tag)
                
            all_true_seqs.append(true_seq)
            all_pred_seqs.append(pred_seq)
        
    metrics = {
        "n": len(dataset),
        "intent_accuracy": float(accuracy_score(all_true_intents, all_pred_intents)),
        "precision": float(precision_score(all_true_seqs, all_pred_seqs, average="weighted", zero_division=0)),
        "recall": float(recall_score(all_true_seqs, all_pred_seqs, average="weighted", zero_division=0)),
        "f1": float(f1_score(all_true_seqs, all_pred_seqs, average="weighted", zero_division=0)),
        "report": classification_report(all_true_seqs, all_pred_seqs, output_dict=True, zero_division=0)
    }
    
    def my_f1(yt, yp):
        return f1_score(yt, yp, average="weighted", zero_division=0)
        
    std, low, high = bootstrap_ci(all_true_seqs, all_pred_seqs, my_f1, n_bootstraps=50) # use 50 to be fast
    
    metrics["f1_std_bootstrap"] = std
    metrics["f1_ci95_low"] = low
    metrics["f1_ci95_high"] = high
    
    return metrics, device

if __name__ == "__main__":
    stats = {}

    models_to_run = [
        ("intent_classifier_v2", "outputs/intent_classifier/final_model_v2")
    ]

    for name, path in models_to_run:
        if not os.path.exists(path):
            print(f"Skipping {name}, path not found: {path}")
            continue
            
        m_train, dev = evaluate_ner(path, train_data)
        m_val, _ = evaluate_ner(path, val_data)
        m_test, _ = evaluate_ner(path, test_data)
        
        stats[name] = {
            "model_dir": path,
            "device": str(dev),
            "training_config_sizes": {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data)
            },
            "split_metrics": {
                "train": m_train,
                "validation": m_val,
                "test": m_test
            },
            "split_variation": {
                "train_minus_test_f1": float(m_train["f1"] - m_test["f1"]),
                "train_minus_validation_f1": float(m_train["f1"] - m_val["f1"]),
                "validation_minus_test_f1": float(m_val["f1"] - m_test["f1"])
            }
        }

    out_path = "artifacts/results/ner_research_stats.json"
    os.makedirs("artifacts/results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, cls=NpEncoder)

    print(f"Stats written to {out_path}")
