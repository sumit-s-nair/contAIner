import pytest
from src.rl_env.parser import parse_planned_step
from src.system2_planner.models import ActionType

def test_parse_valid_planned_step():
    # A hand-written valid PlannedStep JSON string
    valid_json = '''{
        "action_type": "INSTALL",
        "target": "requests",
        "description": "Install the requests library",
        "rationale": "Required for web scraping"
    }'''
    
    step = parse_planned_step(valid_json)
    assert step.action_type == ActionType.INSTALL
    assert step.target == "requests"
    assert step.description == "Install the requests library"
    assert step.rationale == "Required for web scraping"

def test_parse_with_trailing_comma():
    # Model often hallucinates a trailing comma
    json_with_comma = '''{
        "action_type": "INSTALL",
        "target": "python",
        "description": "Install necessary Python packages",
        "rationale": "This will ensure all required libraries are installed.",
    }'''
    
    step = parse_planned_step(json_with_comma)
    assert step.action_type == ActionType.INSTALL

import pytest
import json

def test_parse_with_trailing_comma_and_syntax_error():
    # Model has trailing comma BUT also unescaped quotes which makes it genuinely malformed
    json_with_comma_and_error = '''{
        "action_type": "INSTALL",
        "target": "python",
        "description": "Install necessary "Python" packages",
        "rationale": "This will ensure all required libraries are installed.",
    }'''
    
    with pytest.raises(json.JSONDecodeError):
        parse_planned_step(json_with_comma_and_error)

def test_parse_unrecognized_action_type():
    # Model produces an action type that does not exist in ActionType
    json_unknown_action = '''{
        "action_type": "MAGIC_ACTION",
        "target": "python",
        "description": "Do magic",
        "rationale": "Because."
    }'''
    
    with pytest.raises(ValueError, match="Unrecognized action_type: MAGIC_ACTION"):
        parse_planned_step(json_unknown_action)
