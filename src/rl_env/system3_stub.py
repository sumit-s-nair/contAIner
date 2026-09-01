"""
src/rl_env/system3_stub.py
==========================
Deterministic stub to expand PlannedStep into AtomicStep(s).
This is a temporary placeholder for the real trained System 3 model.
TODO: Swap in the real trained System 3 model before this pipeline reflects the intended architecture.
"""

from typing import List
from src.system2_planner.models import PlannedStep, ActionType
from src.sandbox.models import AtomicStep

def expand_planned_step(planned: PlannedStep) -> List[AtomicStep]:
    """
    Expands a high-level PlannedStep into one or more concrete AtomicSteps.
    This simulates System 3 code generation.
    """
    atype = planned.action_type
    # Using getattr for backward compatibility if PlannedStep was initialized without target
    target = getattr(planned, 'target', '') or ''
    
    if atype == ActionType.INSTALL:
        return [
            AtomicStep(
                command=f"pip install {target}" if target else "pip install -r requirements.txt",
                description=f"Install {target}",
                verify_command=f"python -c 'import {target}'" if target else None,
                destructive=False
            )
        ]
    elif atype == ActionType.ISOLATE:
        return [
            AtomicStep(
                command="python -m venv .venv",
                description="Create virtual environment",
                verify_command="test -d .venv",
                destructive=False
            )
        ]
    elif atype == ActionType.CHECK:
        return [
            AtomicStep(
                command=f"pytest {target}" if target else "pytest",
                description=f"Run tests for {target}",
                destructive=False
            )
        ]
    # Default fallback
    return [
        AtomicStep(
            command=f"echo '{atype.value} {target}'",
            description=planned.description,
            destructive=False
        )
    ]
