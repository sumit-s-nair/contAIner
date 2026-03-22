"""
Verification Script for System 1A: Intent Classifier

Loads the fine-tuned model and lets you interactively test predictions,
or run batch verification against a dataset file.

Usage:
    # Interactive mode (type instructions, see predictions)
    python src/system1_intent_understanding/verify_intent_classifier.py \
        --model_dir outputs/system1_intent_classifier/final_model

    # Batch mode (verify against a JSONL file, shows mismatches)
    python src/system1_intent_understanding/verify_intent_classifier.py \
        --model_dir outputs/system1_intent_classifier/final_model \
        --verify_file datasets/intent-dataset/data/software_dataset_combined.jsonl \
        --sample_size 50
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_dir: str):
    """Load the fine-tuned model, tokenizer, and label map."""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    # Load label map
    label_map_path = model_dir / "label_map.json"
    if not label_map_path.exists():
        print(f"Error: label_map.json not found in {model_dir}")
        sys.exit(1)

    with open(label_map_path) as f:
        label_map = json.load(f)

    id_to_label = {int(v): k for k, v in label_map.items()}

    # Load model and tokenizer
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")
    print(f"Labels: {list(label_map.keys())}")
    print()

    return model, tokenizer, label_map, id_to_label, device


# =============================================================================
# Prediction
# =============================================================================


def predict(
    text: str,
    model,
    tokenizer,
    id_to_label: Dict[int, str],
    device: torch.device,
    max_length: int = 128,
) -> Dict:
    """
    Predict intent for a single instruction.

    Returns dict with predicted label, confidence, and all class probabilities.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    pred_id = torch.argmax(probs).item()
    pred_label = id_to_label[pred_id]
    confidence = probs[pred_id].item()

    all_probs = {id_to_label[i]: probs[i].item() for i in range(len(id_to_label))}

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": all_probs,
    }


def format_prediction(text: str, result: Dict, expected: Optional[str] = None) -> str:
    """Format a prediction result for display."""
    lines = []
    lines.append(f'  Input:      "{text}"')
    lines.append(
        f"  Predicted:  {result['prediction']}  (confidence: {result['confidence']:.4f})"
    )

    if expected is not None:
        match = "✓" if result["prediction"] == expected else "✗"
        lines.append(f"  Expected:   {expected}  [{match}]")

    # Show all probabilities sorted by value
    sorted_probs = sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    prob_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted_probs)
    lines.append(f"  Probs:      {prob_str}")

    return "\n".join(lines)


# =============================================================================
# Interactive Mode
# =============================================================================


def interactive_mode(model, tokenizer, id_to_label, device):
    """Run interactive prediction loop."""
    print("=" * 60)
    print("INTERACTIVE INTENT VERIFICATION")
    print("=" * 60)
    print("Type an instruction and press Enter to see the prediction.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Provide some example prompts
    examples = [
        "install python on my machine",
        "upgrade git on my windows machine",
        "please get chrome installed",
        "update nodejs to version 18",
        "i need to install docker",
        "can you update pip for me",
        "set up ruby on linux",
        "downgrade flask to 2.0",
    ]
    print("Example instructions to try:")
    for ex in examples:
        print(f"  → {ex}")
    print()

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        result = predict(text, model, tokenizer, id_to_label, device)
        print(format_prediction(text, result))
        print()


# =============================================================================
# Batch Verification Mode
# =============================================================================


def batch_verify(
    model,
    tokenizer,
    id_to_label,
    device,
    verify_file: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
    show_correct: bool = False,
):
    """
    Run batch verification against a JSONL file.

    Loads samples, predicts, compares to ground truth, and reports mismatches.
    """
    print("=" * 60)
    print("BATCH VERIFICATION")
    print("=" * 60)

    # Load data
    data = []
    with open(verify_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} rows from {verify_file}")

    # Sample if requested
    if sample_size and sample_size < len(data):
        random.seed(seed)
        data = random.sample(data, sample_size)
        print(f"Sampled {sample_size} rows for verification")

    print()

    # Run predictions
    correct = 0
    incorrect = 0
    mismatches: List[Dict] = []

    for i, row in enumerate(data):
        text = row["instruction"]
        expected = row["intent_type"]
        result = predict(text, model, tokenizer, id_to_label, device)
        predicted = result["prediction"]

        if predicted == expected:
            correct += 1
            if show_correct:
                print(f"[{i+1}/{len(data)}] ✓ CORRECT")
                print(format_prediction(text, result, expected))
                print()
        else:
            incorrect += 1
            mismatches.append(
                {
                    "index": i + 1,
                    "instruction": text,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                }
            )

    # Report
    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0

    print("-" * 60)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.2%})")
    print(f"  Correct:   {correct}")
    print(f"  Incorrect: {incorrect}")
    print("-" * 60)

    if mismatches:
        print(f"\nMISCLASSIFIED SAMPLES ({len(mismatches)}):\n")
        for m in mismatches:
            print(f"  [{m['index']}] \"{m['instruction']}\"")
            match_marker = "✗"
            print(f"       Expected:  {m['expected']}")
            print(
                f"       Predicted: {m['predicted']}  (conf: {m['confidence']:.4f})  [{match_marker}]"
            )
            sorted_probs = sorted(
                m["probabilities"].items(), key=lambda x: x[1], reverse=True
            )
            prob_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted_probs)
            print(f"       Probs:     {prob_str}")
            print()
    else:
        print("\n✓ All samples classified correctly!")

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "mismatches": mismatches,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify intent classifier predictions interactively or in batch mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/system1_intent_classifier/final_model",
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--verify_file",
        type=str,
        default=None,
        help="JSONL file to verify against (enables batch mode). "
        "If not provided, runs in interactive mode.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to verify in batch mode (default: all)",
    )
    parser.add_argument(
        "--show_correct",
        action="store_true",
        help="Also print correctly classified samples in batch mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model, tokenizer, label_map, id_to_label, device = load_model(args.model_dir)

    if args.verify_file:
        batch_verify(
            model,
            tokenizer,
            id_to_label,
            device,
            verify_file=args.verify_file,
            sample_size=args.sample_size,
            seed=args.seed,
            show_correct=args.show_correct,
        )
    else:
        interactive_mode(model, tokenizer, id_to_label, device)
