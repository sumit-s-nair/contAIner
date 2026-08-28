"""System 2 package exports for command-generation training and evaluation.

This package contains training-time components for producing CommandPlan output
from structured intent metadata. Command execution is intentionally out of scope.
"""

from typing import TYPE_CHECKING

from .config import TrainingConfig, INTENT_TYPES, OS_TYPES, SHELL_TYPES, STEP_TYPES
from .data_preprocessing import CommandDataProcessor
from .models import CommandGenerationModel, ModelType
from .metrics import CommandMetrics

if TYPE_CHECKING:
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


def __getattr__(name: str):
    if name == "CommandGenerationTrainer":
        from .train import CommandGenerationTrainer

        return CommandGenerationTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
