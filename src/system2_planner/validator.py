"""
src/system2_planner/validator.py
================================
Structural validator for System 2 planner outputs.
"""

from typing import List, Set, Tuple
from dataclasses import dataclass
from .models import PlannedStep, TemplateInstance, ActionType

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]

def validate_plan(instance: TemplateInstance, plan: List[PlannedStep]) -> ValidationResult:
    """
    Validates a System 2 plan against the constraints encoded in the TemplateInstance.
    Returns a ValidationResult with rich training signals as errors if invalid.
    """
    errors = []
    
    # Extract planned action types in order
    actual_types = [step.action_type for step in plan]
    
    # We extract constraints based on the specific template and manifest conditions.
    # This directly enforces the rules:
    # - isolation before install-under-conflict
    # - detect before fix-else-escalate
    # - check before install-else-update
    
    template_name = instance.template.name
    
    if template_name == "setup_project":
        if instance.resolve_condition("has_conflicts"):
            # Constraint: Must isolate, must install, isolate must be before install
            _require_presence(actual_types, ActionType.ISOLATE, errors, "Dependency conflicts detected; must ISOLATE environment.")
            _require_presence(actual_types, ActionType.INSTALL, errors, "Must INSTALL dependencies.")
            _require_ordering(actual_types, ActionType.ISOLATE, ActionType.INSTALL, errors)
        else:
            # Constraint: Must install
            _require_presence(actual_types, ActionType.INSTALL, errors, "Must INSTALL dependencies.")
            
    elif template_name == "fix_environment":
        # Constraint: Detect before fix or escalate
        _require_presence(actual_types, ActionType.DETECT, errors, "Must DETECT environment issue.")
        _require_ordering(actual_types, ActionType.DETECT, ActionType.FIX, errors)
        _require_ordering(actual_types, ActionType.DETECT, ActionType.ESCALATE, errors)
        
        # Constraint: Reinstall-in-isolation (ISOLATE or INSTALL) cannot precede DETECT
        _require_ordering(actual_types, ActionType.DETECT, ActionType.ISOLATE, errors)
        _require_ordering(actual_types, ActionType.DETECT, ActionType.INSTALL, errors)
        
    elif template_name == "setup_environment":
        # Constraint: Check before install or update
        _require_presence(actual_types, ActionType.CHECK, errors, "Must CHECK environment requirements.")
        _require_ordering(actual_types, ActionType.CHECK, ActionType.INSTALL, errors)
        _require_ordering(actual_types, ActionType.CHECK, ActionType.UPDATE, errors)
        
    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors)

def _require_presence(actual: List[ActionType], target: ActionType, errors: List[str], msg: str):
    if target not in actual:
        errors.append(f"Missing mandatory action '{target.value}': {msg}")

def _require_ordering(actual: List[ActionType], prerequisite: ActionType, target: ActionType, errors: List[str]):
    if prerequisite in actual and target in actual:
        idx_pre = actual.index(prerequisite)
        # Handle multiple occurrences by checking if any prerequisite is before the first target
        idx_tgt = actual.index(target)
        if idx_pre > idx_tgt:
            errors.append(f"Ordering violation: '{prerequisite.value}' must occur before '{target.value}'.")
    elif target in actual and prerequisite not in actual:
        errors.append(f"Ordering violation: '{target.value}' requires '{prerequisite.value}' to occur first.")
