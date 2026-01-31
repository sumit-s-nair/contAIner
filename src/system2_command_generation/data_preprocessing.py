"""
Data Preprocessing Module for System 2: Command Generation

Simplified preprocessing that works directly with the command-dataset schema:
{instruction, intent_type, entities, os, shell, command, source}

The model learns: (intent_type, entities, os, shell) → CommandPlan
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import jsonschema

from .config import (
    COMMAND_PLAN_SCHEMA,
    OS_SHELL_COMPATIBILITY,
    TrainingConfig,
)


# =============================================================================
# Dataset Schema (matches command-dataset)
# =============================================================================

DATASET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["instruction", "intent_type", "entities", "os", "shell", "command", "source"],
    "properties": {
        "instruction": {"type": "string"},
        "intent_type": {"type": "string"},
        "entities": {
            "type": "object",
            "properties": {
                "runtime": {"type": ["string", "null"]},
                "package": {"type": ["string", "null"]},
                "version": {"type": ["string", "null"]},
            },
        },
        "os": {"type": "string", "enum": ["linux", "windows", "macos"]},
        "shell": {"type": "string", "enum": ["bash", "powershell", "cmd", "zsh"]},
        "command": {"type": "string"},
        "source": {"type": "string"},
    },
}


# =============================================================================
# Prompt Templates
# =============================================================================

INPUT_TEMPLATE = """Generate a CommandPlan for the following intent:

Intent Type: {intent_type}
Entities: {entities}
Target OS: {os}
Target Shell: {shell}

Output a valid JSON CommandPlan:"""

# Compact prompt format for better token efficiency
INPUT_TEMPLATE_COMPACT = """<intent>{intent_type}</intent>
<entities>{entities}</entities>
<os>{os}</os>
<shell>{shell}</shell>
<command>"""


def format_input(row: Dict[str, Any], compact: bool = True) -> str:
    """Format a dataset row into model input text."""
    entities_str = json.dumps(row["entities"], separators=(",", ":"))
    
    if compact:
        return INPUT_TEMPLATE_COMPACT.format(
            intent_type=row["intent_type"],
            entities=entities_str,
            os=row["os"],
            shell=row["shell"],
        )
    
    return INPUT_TEMPLATE.format(
        intent_type=row["intent_type"],
        entities=entities_str,
        os=row["os"],
        shell=row["shell"],
    )


def format_output(row: Dict[str, Any]) -> str:
    """Format a dataset row into model output (CommandPlan JSON)."""
    # Determine step type from intent
    intent_to_step_type = {
        "install_runtime": "install",
        "install_package": "install",
        "update_runtime": "update",
        "update_package": "update",
        "remove_runtime": "remove",
        "remove_package": "remove",
        "list_packages": "verify",
        "create_environment": "configure",
        "activate_environment": "configure",
        "deactivate_environment": "configure",
        "configure_setting": "configure",
        "run_script": "verify",
        "check_version": "verify",
    }
    
    step_type = intent_to_step_type.get(row["intent_type"], "configure")
    
    # Check if elevation is required
    command = row["command"]
    requires_elevation = any(kw in command.lower() for kw in [
        "sudo", "admin", "runas", "gsudo", "doas",
        "apt install", "apt-get install", "yum install", 
        "dnf install", "pacman -s", "winget install",
    ])
    
    command_plan = {
        "intent_type": row["intent_type"],
        "entities": row["entities"],
        "os": row["os"],
        "shell": row["shell"],
        "steps": [
            {
                "step_number": 1,
                "type": step_type,
                "command": command,
                "description": row.get("instruction", f"{step_type.capitalize()} operation"),
            }
        ],
        "confidence": 0.95,
        "requires_elevation": requires_elevation,
    }
    
    return json.dumps(command_plan, separators=(",", ":"))


# =============================================================================
# Validation
# =============================================================================

def validate_dataset_row(row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a dataset row against the schema."""
    try:
        jsonschema.validate(row, DATASET_SCHEMA)
        
        # Check OS/shell compatibility
        os_type = row.get("os")
        shell_type = row.get("shell")
        
        if os_type and shell_type:
            compatible = OS_SHELL_COMPATIBILITY.get(os_type, [])
            if shell_type not in compatible:
                return False, f"Shell '{shell_type}' incompatible with OS '{os_type}'"
        
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)


def validate_command_plan(plan: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a generated CommandPlan."""
    try:
        jsonschema.validate(plan, COMMAND_PLAN_SCHEMA)
        
        # Check OS/shell compatibility
        os_type = plan.get("os")
        shell_type = plan.get("shell")
        
        if os_type and shell_type:
            compatible = OS_SHELL_COMPATIBILITY.get(os_type, [])
            if shell_type not in compatible:
                return False, f"Shell '{shell_type}' incompatible with OS '{os_type}'"
        
        # Check step sequence
        steps = plan.get("steps", [])
        for i, step in enumerate(steps, start=1):
            if step.get("step_number") != i:
                return False, f"Step sequence error at position {i}"
        
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)


# =============================================================================
# PyTorch Dataset
# =============================================================================

class CommandGenerationDataset(Dataset):
    """
    PyTorch Dataset for command generation training.
    
    Directly processes command-dataset rows without intermediate transformations.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        max_output_length: int = 512,
        compact_prompts: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of dataset rows
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            compact_prompts: Use compact prompt format for efficiency
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.compact_prompts = compact_prompts
        
        # Process and filter data
        self.data = []
        for row in data:
            is_valid, error = validate_dataset_row(row)
            if is_valid:
                self.data.append(row)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]
        
        # Format input and output
        input_text = format_input(row, compact=self.compact_prompts)
        output_text = format_output(row)
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize output
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        labels = output_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }
    
    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Get raw dataset row for evaluation."""
        row = self.data[idx]
        return {
            "row": row,
            "input_text": format_input(row, compact=self.compact_prompts),
            "output_text": format_output(row),
        }


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class CommandDataCollator:
    """Data collator with padding."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


# =============================================================================
# Data Processor
# =============================================================================

class CommandDataProcessor:
    """
    Main data processor for command generation training.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
    
    def load_huggingface_dataset(
        self,
        dataset_name: str = "sumit-s-nair/command-dataset",
    ) -> DatasetDict:
        """Load dataset from HuggingFace Hub."""
        print(f"📦 Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        print(f"  ✓ Loaded splits: {list(dataset.keys())}")
        return dataset
    
    def load_local_dataset(self, data_dir: str) -> Dict[str, List[Dict]]:
        """Load dataset from local JSONL files."""
        print(f"📦 Loading from: {data_dir}")
        
        splits = {}
        for split in ["train", "validation", "test"]:
            filepath = os.path.join(data_dir, f"{split}.jsonl")
            if os.path.exists(filepath):
                data = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                splits[split] = data
                print(f"  ✓ {split}: {len(data)} samples")
        
        return splits
    
    def create_datasets(
        self,
        raw_data: Union[DatasetDict, Dict[str, List]],
    ) -> Dict[str, CommandGenerationDataset]:
        """Create PyTorch datasets from raw data."""
        datasets = {}
        
        for split in ["train", "validation", "test"]:
            if split not in raw_data:
                continue
            
            split_data = raw_data[split]
            if hasattr(split_data, "to_list"):
                split_data = split_data.to_list()
            elif hasattr(split_data, "__iter__") and not isinstance(split_data, list):
                split_data = list(split_data)
            
            datasets[split] = CommandGenerationDataset(
                data=split_data,
                tokenizer=self.tokenizer,
                max_input_length=self.config.max_input_length,
                max_output_length=self.config.max_output_length,
            )
            print(f"  ✓ {split}: {len(datasets[split])} valid samples")
        
        return datasets
    
    def get_data_collator(self) -> CommandDataCollator:
        return CommandDataCollator(tokenizer=self.tokenizer)


# =============================================================================
# Inference Utilities
# =============================================================================

def parse_model_output(output_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse model output text into CommandPlan dict."""
    try:
        output_text = output_text.strip()
        
        # Find JSON in output
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        
        if start == -1 or end == 0:
            return None, "No JSON found"
        
        plan = json.loads(output_text[start:end])
        
        is_valid, error = validate_command_plan(plan)
        if not is_valid:
            return None, error
        
        return plan, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def prepare_inference_input(
    intent_type: str,
    entities: Dict[str, Any],
    os_type: str,
    shell: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare input for model inference.
    
    This is called at inference time with System 1's CanonicalIntent output.
    
    Args:
        intent_type: From CanonicalIntent.intent_type
        entities: From CanonicalIntent.entities
        os_type: From CanonicalIntent.os_hint (resolved)
        shell: From CanonicalIntent.shell_type
        tokenizer: Model tokenizer
        max_length: Max sequence length
        device: Target device
        
    Returns:
        Tokenized input ready for model.generate()
    """
    row = {
        "intent_type": intent_type,
        "entities": entities,
        "os": os_type,
        "shell": shell,
    }
    
    input_text = format_input(row, compact=True)
    
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


def inference_from_canonical_intent(
    canonical_intent: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare input from a full CanonicalIntent object (System 1 output).
    
    This bridges System 1 output → System 2 input at runtime.
    """
    return prepare_inference_input(
        intent_type=canonical_intent["intent_type"],
        entities=canonical_intent["entities"],
        os_type=canonical_intent.get("os_hint") or "linux",  # Default to linux
        shell=canonical_intent["shell_type"],
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )
