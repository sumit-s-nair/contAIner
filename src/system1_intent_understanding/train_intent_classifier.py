"""
Training Script for System 1A: Intent Classification

Fine-tunes DistilBERT to classify user instructions into intent types
(e.g., install_package, update_package, install_runtime, update_runtime).

Usage:
    python src/system1_intent_understanding/train_intent_classifier.py \
        --data_path datasets/intent-dataset/data/software_dataset_combined.jsonl \
        --output_dir outputs/system1_intent_classifier \
        --num_epochs 5

The script will:
    1. Download distilbert-base-uncased from HuggingFace
    2. Load and split the intent dataset (80/10/10 stratified)
    3. Fine-tune with HuggingFace Trainer
    4. Evaluate on the test split
    5. Save model, tokenizer, and label mapping
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for training."""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("intent_classifier")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "training.log"), mode="w"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(console_fmt)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
    return data


def build_label_map(data: List[Dict]) -> Dict[str, int]:
    """Build a mapping from intent_type strings to integer labels."""
    intent_types = sorted(set(row["intent_type"] for row in data))
    return {intent: idx for idx, intent in enumerate(intent_types)}


def split_dataset(
    data: List[Dict],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split dataset into train/validation/test with stratification.

    Args:
        data: Full dataset rows
        test_size: Fraction for test split
        val_size: Fraction for validation split
        seed: Random seed

    Returns:
        Dict with 'train', 'validation', 'test' keys
    """
    labels = [row["intent_type"] for row in data]

    # First split: separate out test
    train_val_data, test_data, train_val_labels, _ = train_test_split(
        data, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Second split: separate train and validation
    # Adjust val_size relative to the remaining data
    relative_val_size = val_size / (1.0 - test_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }


def preprocess_dataset(
    data: List[Dict],
    tokenizer,
    label_map: Dict[str, int],
    max_length: int = 128,
) -> Dataset:
    """
    Tokenize instructions and convert labels to integers.

    Args:
        data: List of dataset rows
        tokenizer: HuggingFace tokenizer
        label_map: Mapping from intent_type to label ID
        max_length: Max token length for instructions

    Returns:
        HuggingFace Dataset ready for Trainer
    """
    instructions = [row["instruction"] for row in data]
    labels = [label_map[row["intent_type"]] for row in data]

    encodings = tokenizer(
        instructions,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )
    dataset.set_format("torch")
    return dataset


# =============================================================================
# Metrics
# =============================================================================


def compute_metrics(eval_pred):
    """Compute classification metrics for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision,
        "recall_weighted": recall,
    }


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(
    trainer: Trainer,
    test_dataset: Dataset,
    label_map: Dict[str, int],
    logger: logging.Logger,
) -> Dict:
    """
    Run full evaluation on the test set and print detailed metrics.

    Args:
        trainer: Trained HuggingFace Trainer
        test_dataset: Test split Dataset
        label_map: Label mapping
        logger: Logger instance

    Returns:
        Dict of evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    # Get predictions
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=-1)

    # Reverse label map
    id_to_label = {v: k for k, v in label_map.items()}
    target_names = [id_to_label[i] for i in range(len(label_map))]

    # Classification report
    report = classification_report(
        true_labels, preds, target_names=target_names, digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Overall metrics
    metrics = predictions_output.metrics
    logger.info(f"\nOverall Metrics:")
    for key, value in sorted(metrics.items()):
        logger.info(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    return metrics


# =============================================================================
# Main Training Pipeline
# =============================================================================


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 3,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    logging_steps: int = 50,
    save_total_limit: int = 2,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Full training pipeline for intent classification.

    Args:
        data_path: Path to the JSONL dataset file
        output_dir: Directory to save model and logs
        model_name: HuggingFace model identifier
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Peak learning rate
        max_length: Max tokenization length for instructions
        warmup_ratio: Warmup fraction of total steps
        weight_decay: AdamW weight decay
        gradient_accumulation_steps: Gradient accumulation steps
        fp16: Use mixed precision
        seed: Random seed
        early_stopping_patience: Early stopping patience (epochs)
        eval_strategy: When to evaluate ("epoch" or "steps")
        save_strategy: When to save checkpoints
        logging_steps: Log every N steps
        save_total_limit: Max checkpoints to keep
        resume_from_checkpoint: Path to resume from
    """
    # Setup
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("System 1A: Intent Classification Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Data:  {data_path}")
    logger.info(f"Output: {output_dir}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    logger.info("\n--- Loading Dataset ---")
    raw_data = load_jsonl(data_path)
    logger.info(f"Loaded {len(raw_data)} rows from {data_path}")

    # Build label map
    label_map = build_label_map(raw_data)
    num_labels = len(label_map)
    logger.info(f"Intent types ({num_labels}): {list(label_map.keys())}")

    # -------------------------------------------------------------------------
    # 2. Split data
    # -------------------------------------------------------------------------
    logger.info("\n--- Splitting Dataset ---")
    splits = split_dataset(raw_data, test_size=0.1, val_size=0.1, seed=seed)
    for split_name, split_data in splits.items():
        label_counts = {}
        for row in split_data:
            label_counts[row["intent_type"]] = (
                label_counts.get(row["intent_type"], 0) + 1
            )
        logger.info(f"  {split_name}: {len(split_data)} rows | {label_counts}")

    # -------------------------------------------------------------------------
    # 3. Load tokenizer and model
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Loading Model: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={str(v): k for k, v in label_map.items()},
        label2id=label_map,
    )
    logger.info(f"Model loaded: {model.config.model_type} with {num_labels} labels")
    logger.info(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} total, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
    )

    # -------------------------------------------------------------------------
    # 4. Preprocess datasets
    # -------------------------------------------------------------------------
    logger.info("\n--- Tokenizing ---")
    train_dataset = preprocess_dataset(
        splits["train"], tokenizer, label_map, max_length
    )
    val_dataset = preprocess_dataset(
        splits["validation"], tokenizer, label_map, max_length
    )
    test_dataset = preprocess_dataset(splits["test"], tokenizer, label_map, max_length)
    logger.info(
        f"Tokenized: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    # -------------------------------------------------------------------------
    # 5. Configure training
    # -------------------------------------------------------------------------
    # Check CUDA availability
    use_fp16 = fp16 and torch.cuda.is_available()
    device_info = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else "CPU"
    )
    logger.info(f"\nDevice: {device_info}")
    logger.info(f"FP16: {use_fp16}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=use_fp16,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard by default
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_num_workers=0,  # Windows compatibility
    )

    # -------------------------------------------------------------------------
    # 6. Train
    # -------------------------------------------------------------------------
    logger.info("\n--- Starting Training ---")
    logger.info(
        f"Epochs: {num_epochs} | Batch: {batch_size} | "
        f"Grad Accum: {gradient_accumulation_steps} | "
        f"Effective Batch: {batch_size * gradient_accumulation_steps} | "
        f"LR: {learning_rate}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ],
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Log training results
    logger.info("\n--- Training Complete ---")
    logger.info(f"Training Loss: {train_result.training_loss:.4f}")
    logger.info(f"Training Time: {train_result.metrics.get('train_runtime', 0):.1f}s")

    # -------------------------------------------------------------------------
    # 7. Evaluate
    # -------------------------------------------------------------------------
    evaluate_model(trainer, test_dataset, label_map, logger)

    # -------------------------------------------------------------------------
    # 8. Save model, tokenizer, and label map
    # -------------------------------------------------------------------------
    logger.info("\n--- Saving Model ---")
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save label map
    label_map_path = os.path.join(final_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Save training config
    config_path = os.path.join(final_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "fp16": use_fp16,
                "seed": seed,
                "num_labels": num_labels,
                "label_map": label_map,
                "dataset": data_path,
                "dataset_size": len(raw_data),
                "train_size": len(splits["train"]),
                "val_size": len(splits["validation"]),
                "test_size": len(splits["test"]),
            },
            f,
            indent=2,
        )

    logger.info(f"Model saved to: {final_dir}")
    logger.info(f"Label map saved to: {label_map_path}")
    logger.info(f"Config saved to: {config_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return final_dir


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for intent classification (System 1A)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/intent-dataset/data/software_dataset_combined.jsonl",
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/system1_intent_classifier",
        help="Directory to save model and logs",
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model identifier to fine-tune",
    )

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Per-device batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Peak learning rate"
    )
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision (auto-disabled on CPU)",
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable mixed precision"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training control
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every N steps"
    )
    parser.add_argument(
        "--save_total_limit", type=int, default=2, help="Max checkpoints to keep"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint path",
    )

    args = parser.parse_args()

    # Handle --no_fp16 flag
    if args.no_fp16:
        args.fp16 = False

    return args


if __name__ == "__main__":
    args = parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,  # Match eval strategy
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
