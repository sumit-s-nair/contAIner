"""System 3 (command generation) package.

Two generations live here:
  * slot/ (Stage 3A): PlannedStep + context + docs -> AtomicSteps, the contract
    the pipeline uses (dataset builder, SFT, evaluation, PlannerEnv expander).
  * config / data_preprocessing / models / metrics / train: the earlier
    System-1-intent -> CommandPlan trainer, kept as a baseline.

Exports are lazy: importing ``src.system3_command_generation.slot`` must not
import the old trainer, which needs Hugging Face ``datasets`` (on the 4090
venv it is absent and the repo's own datasets/ folder shadows it).
"""

from importlib import import_module
from typing import TYPE_CHECKING

_LAZY = {
    "TrainingConfig": ".config",
    "INTENT_TYPES": ".config",
    "OS_TYPES": ".config",
    "SHELL_TYPES": ".config",
    "STEP_TYPES": ".config",
    "CommandDataProcessor": ".data_preprocessing",
    "CommandGenerationModel": ".models",
    "ModelType": ".models",
    "CommandMetrics": ".metrics",
    "CommandGenerationTrainer": ".train",
}

if TYPE_CHECKING:
    from .config import TrainingConfig, INTENT_TYPES, OS_TYPES, SHELL_TYPES, STEP_TYPES
    from .data_preprocessing import CommandDataProcessor
    from .models import CommandGenerationModel, ModelType
    from .metrics import CommandMetrics
    from .train import CommandGenerationTrainer

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
