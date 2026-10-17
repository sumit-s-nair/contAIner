"""System 3 "slot" contract (Stage 3A).

System 3 turns one System 2 ``PlannedStep`` into the ``AtomicStep`` list that
System 4 (the Docker sandbox) executes.  This package is the contract shared by
the dataset builder, the SFT trainer, the evaluator and the live pipeline:

    PlannedStep + Sys3Context (+ MCP doc chunk)  ->  prompt  ->  model
    model text  ->  parse_atomic_steps()  ->  List[AtomicStep]

It replaces the old System-1-intent -> CommandPlan contract in
``src/system3_command_generation/{data_preprocessing,train}.py``, which never
matched the slot the stub (``src/rl_env/system3_stub.py``) fills.
"""

from .context import Sys3Context, context_from_env, clean_stderr
from .docs import DocProvider, render_docs
from .prompt import SYSTEM_PROMPT, build_messages, render_user_prompt, steps_to_target_json
from .parse import (
    Sys3ParseError,
    parse_atomic_steps,
    policy_violation,
    steps_policy_violation,
    MAX_STEPS,
)

__all__ = [
    "Sys3Context",
    "context_from_env",
    "clean_stderr",
    "DocProvider",
    "render_docs",
    "SYSTEM_PROMPT",
    "build_messages",
    "render_user_prompt",
    "steps_to_target_json",
    "Sys3ParseError",
    "parse_atomic_steps",
    "policy_violation",
    "steps_policy_violation",
    "MAX_STEPS",
]
