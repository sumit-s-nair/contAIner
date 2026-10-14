import json
import re
from src.system2_planner.models import PlannedStep, ActionType

class ParseError(ValueError):
    """Base exception for PlannedStep parsing errors."""
    pass


class MissingFieldError(ParseError):
    """Raised when a required field (action_type or target) is missing or null."""
    pass


class UnrecognizedActionTypeError(ParseError):
    """Raised when action_type is not a recognized ActionType enum value."""
    pass


class InvalidFieldError(ParseError):
    """Raised when a field has an invalid type (e.g. non-string target)."""
    pass


def parse_planned_step(comp: str) -> PlannedStep:
    start_idx = comp.find('{')
    end_idx = comp.rfind('}')
    if start_idx == -1 or end_idx == -1:
        raise json.JSONDecodeError("No JSON object found", comp, 0)
    
    json_str = comp[start_idx:end_idx+1]
    
    # Strip trailing commas that violate strict JSON
    json_str = re.sub(r',\s*}', '}', json_str)
    
    # Also strip trailing commas before array closing brackets if the model hallucinates those
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    step_dict = json.loads(json_str)
    
    # Validate action_type
    if "action_type" not in step_dict or step_dict["action_type"] is None:
        raise MissingFieldError("Missing or null 'action_type' field")
    
    raw_atype = step_dict["action_type"]
    if not isinstance(raw_atype, str):
        raise InvalidFieldError(f"'action_type' field must be a string, got {type(raw_atype).__name__}")
    if not raw_atype.strip():
        raise MissingFieldError("Empty or whitespace 'action_type' field")
        
    try:
        action_type = ActionType(raw_atype.strip().lower())
    except ValueError:
        raise UnrecognizedActionTypeError(f"Unrecognized action_type: {raw_atype}")

    # Validate target
    if "target" not in step_dict or step_dict["target"] is None:
        raise MissingFieldError("Missing or null 'target' field")
    if not isinstance(step_dict["target"], str):
        raise InvalidFieldError(f"'target' field must be a string, got {type(step_dict['target']).__name__}")
    target = step_dict["target"]

    # description and rationale may default to "" (they don't affect execution)
    description = step_dict.get("description")
    if description is None:
        description = ""
    else:
        description = str(description)

    rationale = step_dict.get("rationale")
    if rationale is None:
        rationale = ""
    else:
        rationale = str(rationale)

    return PlannedStep(
        action_type=action_type,
        target=target,
        description=description,
        rationale=rationale,
    )
