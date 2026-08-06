"""Verify a trained System 1 intent model in interactive or batch mode.

This utility loads an exported model directory and prints intent/entity
predictions for free-text instructions, or compares predictions against a
JSONL file with expected `intent_type` labels.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from ner_utils import extract_entities_from_offsets
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

NER_LABEL_FALLBACK = [
    "O",
    "B-runtime",
    "B-package",
    "B-version",
    "B-virtual_env",
    "B-package_manager",
    "B-project",
    "I-runtime",
    "I-package",
    "I-version",
    "I-virtual_env",
    "I-package_manager",
    "I-project",
]


@dataclass
class JointOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    intent_logits: torch.FloatTensor | None = None
    ner_logits: torch.FloatTensor | None = None


class JointIntentNER(PreTrainedModel):
    """Joint intent classification + token-level NER model used in training."""

    def __init__(
        self,
        config,
        num_intent_labels: int,
        num_ner_labels: int,
        ner_loss_weight: float = 0.5,
    ):
        super().__init__(config)
        self.roberta = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        self.intent_dropout = nn.Dropout(p=0.1)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_intent_labels),
        )

        self.ner_dropout = nn.Dropout(p=0.1)
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

        self.num_intent_labels = num_intent_labels
        self.num_ner_labels = num_ner_labels
        self.ner_loss_weight = ner_loss_weight

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        intent_labels=None,
        ner_labels=None,
        **kwargs,
    ) -> JointOutput:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        intent_logits = self.intent_head(self.intent_dropout(cls_output))
        ner_logits = self.ner_head(self.ner_dropout(sequence_output))

        loss = None
        if intent_labels is not None and ner_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            ner_loss = ner_loss_fn(
                ner_logits.view(-1, self.num_ner_labels),
                ner_labels.view(-1),
            )
            loss = intent_loss + self.ner_loss_weight * ner_loss

        return JointOutput(
            loss=loss,
            intent_logits=intent_logits,
            ner_logits=ner_logits,
        )


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_dir: str):
    """Load the fine-tuned model, tokenizer, and label map."""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    # Load intent label map (new model format)
    label_map_path = model_dir / "intent_label_map.json"
    if not label_map_path.exists():
        print(f"Error: intent_label_map.json not found in {model_dir}")
        sys.exit(1)

    with open(label_map_path) as f:
        intent_label_map = json.load(f)

    id_to_label = {int(v): k for k, v in intent_label_map.items()}

    ner_map_path = model_dir / "ner_label_map.json"
    if ner_map_path.exists():
        with open(ner_map_path) as f:
            ner_map = json.load(f)
        id_to_ner = {int(k): v for k, v in ner_map.get("id2label", {}).items()}
    else:
        id_to_ner = {i: label for i, label in enumerate(NER_LABEL_FALLBACK)}

    max_length = 128
    training_cfg_path = model_dir / "training_config.json"
    if training_cfg_path.exists():
        with open(training_cfg_path) as f:
            training_cfg = json.load(f)
        max_length = int(training_cfg.get("max_length", 128))

    # Load model and tokenizer
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    config = AutoConfig.from_pretrained(str(model_dir))
    model = JointIntentNER.from_pretrained(
        str(model_dir),
        config=config,
        num_intent_labels=len(intent_label_map),
        num_ner_labels=len(id_to_ner),
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")
    print(f"Intent labels: {list(intent_label_map.keys())}")
    print()

    return (
        model,
        tokenizer,
        intent_label_map,
        id_to_label,
        id_to_ner,
        device,
        max_length,
    )


# =============================================================================
# Prediction
# =============================================================================


def predict(
    text: str,
    model,
    tokenizer,
    id_to_label: dict[int, str],
    id_to_ner: dict[int, str],
    device: torch.device,
    max_length: int = 128,
) -> dict:
    """
    Predict intent for a single instruction.

    Returns dict with predicted label, confidence, and all class probabilities.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = inputs["offset_mapping"].squeeze(0).tolist()
    inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        intent_logits = outputs.intent_logits
        ner_logits = outputs.ner_logits
        probs = torch.softmax(intent_logits, dim=-1).squeeze()

    pred_id = torch.argmax(probs).item()
    pred_label = id_to_label[pred_id]
    confidence = probs[pred_id].item()

    all_probs = {id_to_label[i]: probs[i].item() for i in range(len(id_to_label))}

    ner_ids = torch.argmax(ner_logits, dim=-1).squeeze(0).tolist()

    entities = extract_entities_from_offsets(text, ner_ids, offsets, id_to_ner)

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": all_probs,
        "entities": entities,
    }


def format_prediction(text: str, result: dict, expected: str | None = None) -> str:
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

    if result.get("entities"):
        entity_str = ", ".join(
            f"{k}={v}" for k, v in sorted(result["entities"].items())
        )
        lines.append(f"  Entities:   {entity_str}")

    return "\n".join(lines)


# =============================================================================
# Interactive Mode
# =============================================================================


def interactive_mode(model, tokenizer, id_to_label, id_to_ner, device, max_length):
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

        result = predict(
            text,
            model,
            tokenizer,
            id_to_label,
            id_to_ner,
            device,
            max_length=max_length,
        )
        print(format_prediction(text, result))
        print()


# =============================================================================
# Batch Verification Mode
# =============================================================================


def batch_verify(
    model,
    tokenizer,
    id_to_label,
    id_to_ner,
    device,
    verify_file: str,
    max_length: int,
    sample_size: int | None = None,
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
    mismatches: list[dict] = []

    for i, row in enumerate(data):
        text = row["instruction"]
        expected = row["intent_type"]
        result = predict(
            text,
            model,
            tokenizer,
            id_to_label,
            id_to_ner,
            device,
            max_length=max_length,
        )
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
                    "entities": result.get("entities", {}),
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

            if m.get("entities"):
                entity_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(m["entities"].items())
                )
                print(f"       Entities:  {entity_str}")
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
        default="outputs/intent_classifier/final_model_v2",
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

    model, tokenizer, label_map, id_to_label, id_to_ner, device, max_length = (
        load_model(args.model_dir)
    )

    if args.verify_file:
        batch_verify(
            model,
            tokenizer,
            id_to_label,
            id_to_ner,
            device,
            verify_file=args.verify_file,
            max_length=max_length,
            sample_size=args.sample_size,
            seed=args.seed,
            show_correct=args.show_correct,
        )
    else:
        interactive_mode(model, tokenizer, id_to_label, id_to_ner, device, max_length)
