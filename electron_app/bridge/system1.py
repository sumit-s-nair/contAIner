"""
System 1 inference wrapper.

Loads the JointIntentNER model once and exposes a single `predict(text)` call
that returns { intent, confidence, entities }.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import sys
import os
# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "system1_intent_understanding")))
from ner_utils import extract_entities_from_offsets

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional


# ── Minimal model replication (no dependency on train script) ─────────────────

@dataclass
class JointOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    intent_logits: Optional[torch.FloatTensor] = None
    ner_logits: Optional[torch.FloatTensor] = None


class JointIntentNER(PreTrainedModel):
    def __init__(self, config, num_intent_labels: int, num_ner_labels: int,
                 ner_loss_weight: float = 0.5):
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

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> JointOutput:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        cls = seq[:, 0, :]
        return JointOutput(
            intent_logits=self.intent_head(self.intent_dropout(cls)),
            ner_logits=self.ner_head(self.ner_dropout(seq)),
        )


# ── Predictor ─────────────────────────────────────────────────────────────────

class System1Predictor:
    """Load once, call many times."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        with open(model_dir / "intent_label_map.json") as f:
            intent_label_map: Dict[str, int] = json.load(f)
        self.id_to_label = {v: k for k, v in intent_label_map.items()}

        ner_map_path = model_dir / "ner_label_map.json"
        if ner_map_path.exists():
            with open(ner_map_path) as f:
                ner_map = json.load(f)
            self.id_to_ner = {int(k): v for k, v in ner_map.get("id2label", {}).items()}
        else:
            labels = ["O"] + [f"B-{e}" for e in
                               ["software", "version", "project", "file", "runtime", "package", "virtual_env", "package_manager"]] + \
                              [f"I-{e}" for e in
                               ["software", "version", "project", "file", "runtime", "package", "virtual_env", "package_manager"]]
            self.id_to_ner = {i: l for i, l in enumerate(labels)}

        cfg_path = model_dir / "training_config.json"
        self.max_length = 128
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.max_length = int(json.load(f).get("max_length", 128))

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        config = AutoConfig.from_pretrained(str(model_dir))
        self.model = JointIntentNER.from_pretrained(
            str(model_dir), config=config,
            num_intent_labels=len(intent_label_map),
            num_ner_labels=len(self.id_to_ner),
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> Dict:
        """Returns { intent, confidence, entities, probabilities }."""
        inputs = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = inputs.pop("offset_mapping").squeeze(0).tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out.intent_logits, dim=-1).squeeze()
            ner_ids = torch.argmax(out.ner_logits, dim=-1).squeeze(0).tolist()

        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())
        pred_label = self.id_to_label[pred_id]

        all_probs = {self.id_to_label[i]: float(probs[i].item())
                     for i in range(len(self.id_to_label))}

        # NER entity extraction from offsets
        entities = extract_entities_from_offsets(text, ner_ids, offsets, self.id_to_ner)

        return {
            "intent": pred_label,
            "confidence": confidence,
            "entities": entities,
            "probabilities": all_probs,
        }
