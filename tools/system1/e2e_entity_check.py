import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from electron_app.bridge.system1 import System1Predictor

def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    path = "datasets/intent-dataset/data/test_corrected.jsonl"
    if not os.path.exists(path):
        path = "datasets/intent-dataset/data/test.jsonl"
        print(f"Fallback to {path}")
    
    test_data = load_data(path)
    
    model_dir = "outputs/intent_classifier/final_model_v2"
    if not os.path.exists(model_dir):
        print("using variance seed 42 as fallback model")
        model_dir = "outputs/intent_classifier_variance/run_seed42/final_model"
        
    predictor = System1Predictor(model_dir)
    
    exact_matches = defaultdict(int)
    total_expected = defaultdict(int)
    mismatches = []
    
    for row in test_data:
        instruction = row.get("instruction", "")
        expected_spans = row.get("entity_spans", {})
        
        res = predictor.predict(instruction)
        actual_entities = res.get("entities", {})
        
        for etype, expected_info in expected_spans.items():
            if not expected_info: continue
            expected_text = expected_info.get("text")
            if not expected_text: continue
            
            total_expected[etype] += 1
            actual_text = actual_entities.get(etype)
            
            if actual_text == expected_text:
                exact_matches[etype] += 1
            else:
                mismatches.append({
                    "instruction": instruction,
                    "entity_type": etype,
                    "expected": expected_text,
                    "actual": actual_text
                })
                
    print("End-to-End Exact Match Rate per Entity Type:")
    for etype, total in total_expected.items():
        match = exact_matches[etype]
        rate = match / total if total > 0 else 0
        print(f"  {etype}: {match}/{total} ({rate:.2%})")
        
    out_file = "artifacts/results/e2e_mismatches.json"
    os.makedirs("artifacts/results", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(mismatches, f, indent=2)
    print(f"Mismatches written to {out_file}")

if __name__ == "__main__":
    main()
