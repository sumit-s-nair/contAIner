"""System 2 package exports for command-generation training and evaluation.

This package contains training-time components for producing CommandPlan output
from structured intent metadata. Command execution is intentionally out of scope.
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
