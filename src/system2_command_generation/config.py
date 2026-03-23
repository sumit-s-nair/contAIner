"""Configuration and schema definitions for System 2 training.

Includes model presets, JSON schemas for CanonicalIntent/CommandPlan, and the
`TrainingConfig` dataclass used across the training pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


# =============================================================================
# Schema Definitions (matching the output contract)
# =============================================================================

INTENT_TYPES = [
    "install_runtime",
    "install_package",
    "update_runtime",
    "update_package",
    "remove_runtime",
    "remove_package",
    "list_packages",
    "create_environment",
    "activate_environment",
    "deactivate_environment",
    "configure_setting",
    "run_script",
    "check_version",
]

OS_TYPES = ["windows", "linux", "macos"]

SHELL_TYPES = ["bash", "powershell", "cmd", "zsh"]

STEP_TYPES = ["install", "update", "remove", "configure", "verify"]

SCOPE_TYPES = ["system", "user", "project"]

# OS-Shell compatibility mapping
OS_SHELL_COMPATIBILITY = {
    "windows": ["powershell", "cmd"],
    "linux": ["bash", "zsh"],
    "macos": ["bash", "zsh"],
}

# Default shells per OS
DEFAULT_SHELL = {
    "windows": "powershell",
    "linux": "bash",
    "macos": "zsh",
}


# =============================================================================
# JSON Schema for Validation
# =============================================================================

CANONICAL_INTENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CanonicalIntent",
    "type": "object",
    "required": [
        "intent_type",
        "entities",
        "scope",
        "os_hint",
        "shell_type",
        "confidence",
        "missing_fields",
        "needs_clarification",
        "clarification_question",
    ],
    "properties": {
        "intent_type": {"type": "string", "enum": INTENT_TYPES},
        "entities": {
            "type": "object",
            "properties": {
                "runtime": {"type": ["string", "null"]},
                "package": {"type": ["string", "null"]},
                "version": {"type": ["string", "null"]},
            },
        },
        "scope": {"type": "string", "enum": SCOPE_TYPES},
        "os_hint": {"type": ["string", "null"], "enum": OS_TYPES + [None]},
        "shell_type": {"type": "string", "enum": SHELL_TYPES},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": ["string", "null"]},
    },
}

COMMAND_PLAN_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CommandPlan",
    "type": "object",
    "required": [
        "intent_type",
        "entities",
        "os",
        "shell",
        "steps",
        "confidence",
        "requires_elevation",
    ],
    "properties": {
        "intent_type": {"type": "string", "enum": INTENT_TYPES},
        "entities": {
            "type": "object",
            "properties": {
                "runtime": {"type": ["string", "null"]},
                "package": {"type": ["string", "null"]},
                "version": {"type": ["string", "null"]},
            },
        },
        "os": {"type": "string", "enum": OS_TYPES},
        "shell": {"type": "string", "enum": SHELL_TYPES},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step_number", "type", "command", "description"],
                "properties": {
                    "step_number": {"type": "integer", "minimum": 1},
                    "type": {"type": "string", "enum": STEP_TYPES},
                    "command": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                },
            },
            "minItems": 1,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "requires_elevation": {"type": "boolean"},
    },
}


# =============================================================================
# Model Configurations
# =============================================================================

class ModelType(Enum):
    """Supported model types for command generation."""
    CODET5_PLUS = "codet5plus"
    FLAN_T5 = "flan_t5"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_id: str
    tokenizer_id: str
    max_input_length: int = 512
    max_output_length: int = 1024
    is_encoder_decoder: bool = True


# Pre-defined model configurations
MODEL_CONFIGS = {
    ModelType.CODET5_PLUS: ModelConfig(
        name="CodeT5+ 770M",
        model_id="Salesforce/codet5p-770m",
        tokenizer_id="Salesforce/codet5p-770m",
        max_input_length=512,
        max_output_length=1024,
    ),
    ModelType.FLAN_T5: ModelConfig(
        name="FLAN-T5 Base",
        model_id="google/flan-t5-base",
        tokenizer_id="google/flan-t5-base",
        max_input_length=512,
        max_output_length=1024,
    ),
}


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration for command generation models.
    
    Attributes:
        model_type: Which model to train (CodeT5+ or FLAN-T5)
        output_dir: Directory to save trained models and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps for scheduler
        weight_decay: Weight decay for AdamW optimizer
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        fp16: Whether to use mixed precision training
        seed: Random seed for reproducibility
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        early_stopping_patience: Patience for early stopping
    """
    
    # Model settings
    model_type: ModelType = ModelType.CODET5_PLUS
    
    # Paths
    output_dir: str = "./outputs/system2_command_generation"
    cache_dir: str = "./cache"
    logging_dir: str = "./logs"
    
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Mixed precision and optimization
    fp16: bool = True
    bf16: bool = False  # Use bf16 if available (better for newer GPUs)
    optim: str = "adamw_torch"
    
    # Reproducibility
    seed: int = 42
    
    # Evaluation and checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_normalized_match"
    greater_is_better: bool = True
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Data processing
    max_input_length: int = 512
    max_output_length: int = 1024
    num_workers: int = 4
    
    # Metrics thresholds (from requirements)
    target_exact_match: float = 0.70
    target_normalized_match: float = 0.85
    target_intent_preservation: float = 1.00
    target_entity_preservation: float = 0.95
    target_syntax_validity: float = 0.95
    target_os_shell_compatibility: float = 1.00

    def get_model_config(self) -> ModelConfig:
        """Get the model configuration for the selected model type."""
        return MODEL_CONFIGS[self.model_type]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "logging_dir": self.logging_dir,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "optim": self.optim,
            "seed": self.seed,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "num_workers": self.num_workers,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        if "model_type" in config_dict:
            config_dict["model_type"] = ModelType(config_dict["model_type"])
        return cls(**config_dict)

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

# =============================================================================
# Note: Prompt templates are defined in data_preprocessing.py 
# so the config is focused only on hyperparameters and schemas
# =============================================================================
