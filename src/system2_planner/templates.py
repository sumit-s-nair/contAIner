"""
src/system2_planner/templates.py
================================
Canonical Workflow Templates definitions.
"""

from .models import (
    WorkflowTemplate,
    Action,
    ActionType,
    Sequence,
    Branch,
    Fallback,
)

# 1. setup_project
# isolation before install-under-conflict
SETUP_PROJECT_TEMPLATE = WorkflowTemplate(
    name="setup_project",
    description="Sets up a repository project, using isolation if dependency conflicts are detected.",
    root=Branch(
        condition_name="has_conflicts",
        true_branch=Sequence([
            Action(ActionType.ISOLATE, "Isolate the environment (e.g., virtual environment)."),
            Action(ActionType.INSTALL, "Install dependencies in the isolated environment.")
        ]),
        false_branch=Action(ActionType.INSTALL, "Install dependencies directly.")
    )
)

# 2. fix_environment
# detect before fix-else-escalate
FIX_ENVIRONMENT_TEMPLATE = WorkflowTemplate(
    name="fix_environment",
    description="Diagnose and attempt to fix an environment issue, escalating to the user if the fix fails.",
    root=Sequence([
        Action(ActionType.DETECT, "Detect the missing or broken component."),
        Fallback(
            primary=Action(ActionType.FIX, "Attempt to automatically fix the issue (e.g., install missing system tool)."),
            fallback=Action(ActionType.ESCALATE, "Escalate the issue to the user for manual intervention.")
        )
    ])
)

# 3. setup_environment
# check before install-else-update
SETUP_ENVIRONMENT_TEMPLATE = WorkflowTemplate(
    name="setup_environment",
    description="Ensure an environment requirement is met, installing or updating it as necessary.",
    root=Sequence([
        Action(ActionType.CHECK, "Check if the required environment component is present and up to date."),
        Fallback(
            primary=Action(ActionType.INSTALL, "Install the component if it is missing."),
            fallback=Action(ActionType.UPDATE, "Update the component if it is outdated or install failed.")
        )
    ])
)

TEMPLATES = {
    "setup_project": SETUP_PROJECT_TEMPLATE,
    "fix_environment": FIX_ENVIRONMENT_TEMPLATE,
    "setup_environment": SETUP_ENVIRONMENT_TEMPLATE,
}
