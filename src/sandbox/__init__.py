"""
src/sandbox/__init__.py
=======================
Public API for the contAIner sandbox execution harness.

Exports
-------
- Models  : AtomicStep, RiskTier, ClassificationResult, ExecutionResult
- Classify : CommandRiskClassifier
- Gate    : request_user_confirmation
- Execute : SandboxExecutor, ALLOW_BLOCKED_EXECUTION
"""

from .models import (
    AtomicStep,
    RiskTier,
    ClassificationResult,
    ExecutionResult,
)
from .classifier import CommandRiskClassifier
from .confirmation import request_user_confirmation
from .executor import SandboxExecutor, ALLOW_BLOCKED_EXECUTION

__all__ = [
    # models
    "AtomicStep",
    "RiskTier",
    "ClassificationResult",
    "ExecutionResult",
    # classifier
    "CommandRiskClassifier",
    # confirmation gate
    "request_user_confirmation",
    # executor
    "SandboxExecutor",
    "ALLOW_BLOCKED_EXECUTION",
]
