import json
import re
from src.system2_planner.models import PlannedStep, ActionType

def parse_planned_step(comp: str) -> PlannedStep:
    start_idx = comp.find('{')
    end_idx = comp.rfind('}')
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON object found")
    
    json_str = comp[start_idx:end_idx+1]
    
    # Strip trailing commas that violate strict JSON
    json_str = re.sub(r',\s*}', '}', json_str)
    
    # Also strip trailing commas before array closing brackets if the model hallucinates those
    json_str = re.sub(r',\s*\]', ']', json_str)
    
    step_dict = json.loads(json_str)
    
    atype_str = step_dict.get("action_type", "CHECK")
    # Sometimes action_type is missing or null
    if atype_str is None:
        atype_str = "CHECK"
    else:
        atype_str = str(atype_str).upper()
        
    try:
        action_type = ActionType(atype_str.lower())
    except ValueError:
        raise ValueError(f"Unrecognized action_type: {atype_str}")

    return PlannedStep(
        action_type=action_type,
        target=step_dict.get("target", ""),
        description=step_dict.get("description", ""),
        rationale=step_dict.get("rationale", "")
    )
