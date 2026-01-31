"""
System 2: Command Generation Module

This module trains and evaluates models that convert structured intent data
into OS-aware CommandPlan objects.

Training Input (from command-dataset):
    {instruction, intent_type, entities, os, shell, command, source}

Training Output (model learns to produce):
    CommandPlan JSON

At Inference Time:
    System 1 output (CanonicalIntent) → System 2 → CommandPlan

The module does NOT execute commands - it only generates command plans.
"""

from .config import TrainingConfig, INTENT_TYPES, OS_TYPES, SHELL_TYPES, STEP_TYPES
from .data_preprocessing import CommandDataProcessor
from .models import CommandGenerationModel, ModelType
from .metrics import CommandMetrics
from .train import CommandGenerationTrainer

__all__ = [
    "TrainingConfig",
    "CommandDataProcessor",
    "CommandGenerationModel",
    "ModelType",
    "CommandMetrics",
    "CommandGenerationTrainer",
    "INTENT_TYPES",
    "OS_TYPES",
    "SHELL_TYPES",
    "STEP_TYPES",
]
