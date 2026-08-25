"""
src/system2_planner/models.py
=============================
DSL for System 2 Canonical Workflow Templates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
import abc

from src.repo_scan.models import RepoManifest

class ActionType(str, Enum):
    ISOLATE = "isolate"
    INSTALL = "install"
    CHECK = "check"
    UPDATE = "update"
    DETECT = "detect"
    FIX = "fix"
    ESCALATE = "escalate"
    BUILD = "build"

@dataclass
class PlannedStep:
    """A step output by the System 2 Planner."""
    action_type: ActionType
    description: str

class WorkflowNode(abc.ABC): pass

@dataclass
class Action(WorkflowNode):
    action_type: ActionType
    description: str

@dataclass
class Sequence(WorkflowNode):
    steps: List[WorkflowNode]

@dataclass
class Branch(WorkflowNode):
    condition_name: str
    true_branch: WorkflowNode
    false_branch: Optional[WorkflowNode] = None

@dataclass
class Fallback(WorkflowNode):
    primary: WorkflowNode
    fallback: WorkflowNode

@dataclass
class WorkflowTemplate:
    name: str
    description: str
    root: WorkflowNode

@dataclass
class TemplateInstance:
    template: WorkflowTemplate
    manifest: RepoManifest
    
    def resolve_condition(self, condition_name: str) -> bool:
        if condition_name == "has_conflicts":
            return len(self.manifest.conflicts) > 0
        if condition_name == "has_multiple_ecosystems":
            return len(self.manifest.ecosystems) > 1
        return False
        
    def get_expected_actions(self) -> List[Action]:
        return self._flatten(self.template.root)
        
    def _flatten(self, node: WorkflowNode) -> List[Action]:
        if isinstance(node, Action): return [node]
        elif isinstance(node, Sequence):
            actions = []
            for step in node.steps: actions.extend(self._flatten(step))
            return actions
        elif isinstance(node, Branch):
            if self.resolve_condition(node.condition_name):
                return self._flatten(node.true_branch)
            elif node.false_branch:
                return self._flatten(node.false_branch)
            return []
        elif isinstance(node, Fallback):
            actions = self._flatten(node.primary)
            actions.extend(self._flatten(node.fallback))
            return actions
        return []
