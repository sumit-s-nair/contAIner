"""Train the System 1 joint intent-classification and NER model.

This script fine-tunes a shared DistilRoBERTa encoder with two heads:
- intent classification head over the instruction
- token-level NER head using BIO tags

The training objective combines both losses so entity context can improve
intent prediction quality and vice versa.

Typical usage:
        python train_intent_classifier.py
        python train_intent_classifier.py --train datasets/intent-dataset/data/train.jsonl \
                                          --val datasets/intent-dataset/data/validation.jsonl \
                                          --test datasets/intent-dataset/data/test.jsonl \
                                          --output outputs/intent_classifier
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    PretrainedConfig,
)
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL  = "distilroberta-base"   # small, fast, strong on short instructions
MAX_LENGTH  = 128                    # instructions are short — 64 would also work

# NER label scheme: BIO tagging for 6 entity types + O
ENTITY_TYPES = ["software", "version", "project", "file"]
NER_LABELS   = ["O"] + [f"B-{e}" for e in ENTITY_TYPES] + [f"I-{e}" for e in ENTITY_TYPES]
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: l for l, i in NER_LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("intent_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(output_dir, "training.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {i}: {e}")
    return rows


def build_intent_label_map(rows: list[dict]) -> dict[str, int]:
    intents = sorted(set(r["intent_type"] for r in rows))
    return {intent: i for i, intent in enumerate(intents)}


# ---------------------------------------------------------------------------
# NER label alignment
# Entity spans give us character offsets → we need token-level BIO labels
# ---------------------------------------------------------------------------

def align_ner_labels(
    instruction: str,
    entity_spans: dict,
    encoding,           # tokenizer output for this instruction
    tokenizer,
) -> list[int]:
    """
    Convert character-level entity_spans to token-level BIO labels.
    Tokens that are special tokens or padding get label -100 (ignored in loss).
    """
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    n_tokens = len(tokens)
    labels = [-100] * n_tokens  # default: ignored

    # Build a char→token index map using offset_mapping
    # We need the tokenizer called with return_offsets_mapping=True
    offsets = encoding.get("offset_mapping", [])

    # Mark non-special tokens as O first
    for tok_idx, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue  # special token [CLS], [SEP], <pad>
        labels[tok_idx] = NER_LABEL2ID["O"]

    # Assign BIO labels from entity_spans
    for entity_key, span in entity_spans.items():
        if not isinstance(span, dict):
            continue
        char_start = span.get("start")
        char_end   = span.get("end")
        if char_start is None or char_end is None:
            continue

        first = True
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue  # special token
            # Token overlaps with entity span
            if tok_start >= char_start and tok_end <= char_end:
                tag = f"B-{entity_key}" if first else f"I-{entity_key}"
                if tag in NER_LABEL2ID:
                    labels[tok_idx] = NER_LABEL2ID[tag]
                    first = False
            elif tok_start >= char_end:
                break

    return labels


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_split(
    rows: list[dict],
    tokenizer,
    intent_label_map: dict[str, int],
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """Tokenize and build a HuggingFace Dataset for one split."""
    input_ids_list      = []
    attention_mask_list = []
    intent_labels_list  = []
    ner_labels_list     = []

    for row in rows:
        instruction   = row.get("instruction", "")
        intent_type   = row.get("intent_type", "")
        entity_spans  = row.get("entity_spans", {})

        # Tokenize with offset mapping so we can align NER labels
        encoding = tokenizer(
            instruction,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors=None,
        )

        ner_labels = align_ner_labels(
            instruction, entity_spans, encoding, tokenizer
        )

        # Pad NER labels to max_length
        while len(ner_labels) < max_length:
            ner_labels.append(-100)
        ner_labels = ner_labels[:max_length]

        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])
        intent_labels_list.append(intent_label_map.get(intent_type, 0))
        ner_labels_list.append(ner_labels)

    dataset = Dataset.from_dict({
        "input_ids":       input_ids_list,
        "attention_mask":  attention_mask_list,
        "intent_labels":   intent_labels_list,
        "ner_labels":      ner_labels_list,
    })
    dataset.set_format("torch")
    return dataset


# ---------------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------------

@dataclass
class JointOutput(ModelOutput):
    loss:              Optional[torch.FloatTensor] = None
    intent_logits:     Optional[torch.FloatTensor] = None
    ner_logits:        Optional[torch.FloatTensor] = None
    hidden_states:     Optional[tuple]             = None
    attentions:        Optional[tuple]             = None


class JointIntentNER(PreTrainedModel):
    """
    distilroberta-base with two heads:
      1. Classification head on [CLS] → intent_type
      2. Token classification head on all tokens → BIO NER tags

    Loss = intent_loss + ner_loss_weight * ner_loss
    The NER weight is set lower (0.5) because intent accuracy is the
    primary objective and NER data is noisier (many inferred entities
    have no spans).
    """

    def __init__(self, config, num_intent_labels: int, num_ner_labels: int,
                 ner_loss_weight: float = 0.5):
        super().__init__(config)
        self.roberta         = AutoModel.from_config(config)
        hidden_size          = config.hidden_size

        # Intent classification head
        self.intent_dropout  = nn.Dropout(p=0.1)
        self.intent_head     = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_intent_labels),
        )

        # NER token classification head
        self.ner_dropout     = nn.Dropout(p=0.1)
        self.ner_head        = nn.Linear(hidden_size, num_ner_labels)

        self.num_intent_labels = num_intent_labels
        self.num_ner_labels    = num_ner_labels
        self.ner_loss_weight   = ner_loss_weight

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
        sequence_output = outputs.last_hidden_state   # (B, T, H)
        cls_output      = sequence_output[:, 0, :]    # (B, H)  — [CLS] token

        # Intent logits
        intent_logits = self.intent_head(self.intent_dropout(cls_output))

        # NER logits
        ner_logits = self.ner_head(self.ner_dropout(sequence_output))

        loss = None
        if intent_labels is not None and ner_labels is not None:
            intent_loss_fn = nn.CrossEntropyLoss()
            ner_loss_fn    = nn.CrossEntropyLoss(ignore_index=-100)

            intent_loss = intent_loss_fn(intent_logits, intent_labels)
            ner_loss    = ner_loss_fn(
                ner_logits.view(-1, self.num_ner_labels),
                ner_labels.view(-1),
            )
            loss = intent_loss + self.ner_loss_weight * ner_loss

        return JointOutput(
            loss=loss,
            intent_logits=intent_logits,
            ner_logits=ner_logits,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(intent_id2label: dict[int, str]):
    """Returns a compute_metrics function for the Trainer."""

    def compute_metrics(eval_pred):
        # eval_pred.predictions is a tuple: (intent_logits, ner_logits)
        (intent_logits, ner_logits), labels = eval_pred
        # labels is a dict-like: we packed intent_labels and ner_labels together
        # Trainer passes them as a tuple when there are multiple label columns
        intent_labels, ner_labels = labels

        # --- Intent metrics ---
        intent_preds = np.argmax(intent_logits, axis=-1)
        intent_acc   = accuracy_score(intent_labels, intent_preds)
        intent_f1    = f1_score(intent_labels, intent_preds,
                                average="weighted", zero_division=0)

        # --- NER metrics (seqeval — ignores -100) ---
        ner_preds_flat = np.argmax(ner_logits, axis=-1)

        true_seqs = []
        pred_seqs = []
        for pred_row, label_row in zip(ner_preds_flat, ner_labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                true_seq.append(NER_ID2LABEL.get(int(l), "O"))
                pred_seq.append(NER_ID2LABEL.get(int(p), "O"))
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)

        try:
            ner_report  = seq_classification_report(true_seqs, pred_seqs,
                                                     output_dict=True, zero_division=0)
            ner_f1      = ner_report.get("weighted avg", {}).get("f1-score", 0.0)
            ner_precision = ner_report.get("weighted avg", {}).get("precision", 0.0)
            ner_recall    = ner_report.get("weighted avg", {}).get("recall", 0.0)
        except Exception:
            ner_f1 = ner_precision = ner_recall = 0.0

        return {
            "intent_accuracy":    round(intent_acc,   4),
            "intent_f1_weighted": round(intent_f1,    4),
            "ner_f1_weighted":    round(ner_f1,       4),
            "ner_precision":      round(ner_precision, 4),
            "ner_recall":         round(ner_recall,    4),
            # Combined metric used for best model selection
            "combined_f1": round((intent_f1 + ner_f1) / 2, 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def evaluate_model(trainer, test_dataset, intent_label_map, logger):
    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)

    pred_output = trainer.predict(test_dataset)
    intent_logits, ner_logits = pred_output.predictions
    intent_labels, ner_labels = pred_output.label_ids

    intent_preds = np.argmax(intent_logits, axis=-1)
    id2intent    = {v: k for k, v in intent_label_map.items()}

    logger.info("\n--- Intent Classification ---")
    logger.info(f"Accuracy: {accuracy_score(intent_labels, intent_preds):.4f}")
    logger.info("\n" + classification_report(
        intent_labels, intent_preds,
        target_names=[id2intent[i] for i in sorted(id2intent)],
        zero_division=0,
    ))

    # NER per-entity breakdown
    logger.info("\n--- NER Entity Extraction ---")
    ner_preds_flat = np.argmax(ner_logits, axis=-1)
    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(ner_preds_flat, ner_labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(NER_ID2LABEL.get(int(l), "O"))
            pred_seq.append(NER_ID2LABEL.get(int(p), "O"))
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    try:
        logger.info("\n" + seq_classification_report(true_seqs, pred_seqs,
                                                      zero_division=0))
    except Exception as e:
        logger.warning(f"Could not generate NER report: {e}")

    return pred_output.metrics


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    train_path:    str,
    val_path:      str,
    test_path:     str,
    output_dir:    str,
    model_name:    str  = BASE_MODEL,
    num_epochs:    int  = 8,
    batch_size:    int  = 32,
    learning_rate: float = 3e-5,
    max_length:    int  = MAX_LENGTH,
    warmup_ratio:  float = 0.1,
    weight_decay:  float = 0.01,
    grad_accum:    int  = 1,
    ner_loss_weight: float = 0.5,
    fp16:          bool = True,
    seed:          int  = 42,
    early_stopping_patience: int = 3,
    resume_from_checkpoint: Optional[str] = None,
):
    logger = setup_logging(output_dir)
    logger.info(f"Base model:  {model_name}")
    logger.info(f"Output dir:  {output_dir}")
    logger.info(f"Train:       {train_path}")
    logger.info(f"Val:         {val_path}")
    logger.info(f"Test:        {test_path}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    logger.info("\n--- Loading data ---")
    train_rows = load_jsonl(train_path)
    val_rows   = load_jsonl(val_path)
    test_rows  = load_jsonl(test_path)
    logger.info(f"Rows — train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Build label maps from training data
    intent_label_map = build_intent_label_map(train_rows)
    num_intent_labels = len(intent_label_map)
    num_ner_labels    = len(NER_LABELS)
    logger.info(f"Intent classes: {num_intent_labels} → {list(intent_label_map.keys())}")
    logger.info(f"NER labels:     {num_ner_labels} → {NER_LABELS}")

    # -------------------------------------------------------------------------
    # 2. Tokenizer
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Loading tokenizer: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -------------------------------------------------------------------------
    # 3. Tokenize datasets
    # -------------------------------------------------------------------------
    logger.info("--- Tokenizing ---")
    train_dataset = prepare_split(train_rows, tokenizer, intent_label_map, max_length)
    val_dataset   = prepare_split(val_rows,   tokenizer, intent_label_map, max_length)
    test_dataset  = prepare_split(test_rows,  tokenizer, intent_label_map, max_length)
    logger.info(f"Tokenized: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # -------------------------------------------------------------------------
    # 4. Model
    # -------------------------------------------------------------------------
    logger.info(f"\n--- Loading model: {model_name} ---")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    model = JointIntentNER(
        config,
        num_intent_labels=num_intent_labels,
        num_ner_labels=num_ner_labels,
        ner_loss_weight=ner_loss_weight,
    )
    # Load pretrained encoder weights — only the roberta part
    pretrained = AutoModel.from_pretrained(model_name)
    model.roberta.load_state_dict(pretrained.state_dict(), strict=False)
    del pretrained

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # -------------------------------------------------------------------------
    # 5. Device
    # -------------------------------------------------------------------------
    use_fp16 = fp16 and torch.cuda.is_available()
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Device: {device_name} ({vram:.1f} GB VRAM)")
    else:
        logger.info("Device: CPU (training will be slow)")
    logger.info(f"FP16: {use_fp16}")

    # -------------------------------------------------------------------------
    # 6. Training arguments
    # -------------------------------------------------------------------------
    # Effective batch = batch_size * grad_accum
    # For RTX 5060 8GB: batch_size=32, grad_accum=1 → effective=32
    effective_batch = batch_size * grad_accum
    logger.info(f"\nBatch size: {batch_size} × grad_accum {grad_accum} = effective {effective_batch}")
    logger.info(f"Learning rate: {learning_rate} | Epochs: {num_epochs} | NER loss weight: {ner_loss_weight}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=grad_accum,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="combined_f1",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_num_workers=0,   # Windows compatibility
        label_names=["intent_labels", "ner_labels"],
    )

    # -------------------------------------------------------------------------
    # 7. Train
    # -------------------------------------------------------------------------
    logger.info("\n--- Starting training ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics({v: k for k, v in intent_label_map.items()}),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training complete.")

    # -------------------------------------------------------------------------
    # 8. Evaluate on test set
    # -------------------------------------------------------------------------
    evaluate_model(trainer, test_dataset, intent_label_map, logger)

    # -------------------------------------------------------------------------
    # 9. Save
    # -------------------------------------------------------------------------
    logger.info("\n--- Saving model ---")
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save label maps so inference code can reconstruct predictions
    with open(os.path.join(final_dir, "intent_label_map.json"), "w") as f:
        json.dump(intent_label_map, f, indent=2)

    with open(os.path.join(final_dir, "ner_label_map.json"), "w") as f:
        json.dump({"label2id": NER_LABEL2ID, "id2label": NER_ID2LABEL}, f, indent=2)

    with open(os.path.join(final_dir, "entity_types.json"), "w") as f:
        json.dump(ENTITY_TYPES, f, indent=2)

    with open(os.path.join(final_dir, "training_config.json"), "w") as f:
        json.dump({
            "base_model":          model_name,
            "num_intent_labels":   num_intent_labels,
            "num_ner_labels":      num_ner_labels,
            "ner_loss_weight":     ner_loss_weight,
            "max_length":          max_length,
            "num_epochs":          num_epochs,
            "batch_size":          batch_size,
            "learning_rate":       learning_rate,
            "warmup_ratio":        warmup_ratio,
            "weight_decay":        weight_decay,
            "gradient_accumulation_steps": grad_accum,
            "fp16":                use_fp16,
            "seed":                seed,
            "train_size":          len(train_rows),
            "val_size":            len(val_rows),
            "test_size":           len(test_rows),
            "intent_label_map":    intent_label_map,
            "entity_types":        ENTITY_TYPES,
            "ner_labels":          NER_LABELS,
        }, f, indent=2)

    logger.info(f"Saved to: {final_dir}")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    return final_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train joint intent classifier + NER model for contAIner Stage 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",  default="datasets/intent-dataset/data/train.jsonl")
    parser.add_argument("--val",    default="datasets/intent-dataset/data/validation.jsonl")
    parser.add_argument("--test",   default="datasets/intent-dataset/data/test.jsonl")
    parser.add_argument("--output", default="outputs/intent_classifier")

    parser.add_argument("--model",          default=BASE_MODEL,
                        help="HuggingFace model identifier")
    parser.add_argument("--epochs",         type=int,   default=8)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=3e-5)
    parser.add_argument("--max_length",     type=int,   default=MAX_LENGTH)
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--grad_accum",     type=int,   default=1)
    parser.add_argument("--ner_loss_weight",type=float, default=0.5,
                        help="Weight for NER loss relative to intent loss")
    parser.add_argument("--no_fp16",        action="store_true")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--patience",       type=int,   default=3,
                        help="Early stopping patience")
    parser.add_argument("--resume",         default=None,
                        help="Resume from checkpoint path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Install check
    missing = []
    for pkg in ["transformers", "datasets", "torch", "sklearn", "seqeval"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    train(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        ner_loss_weight=args.ner_loss_weight,
        fp16=not args.no_fp16,
        seed=args.seed,
        early_stopping_patience=args.patience,
        resume_from_checkpoint=args.resume,
    )