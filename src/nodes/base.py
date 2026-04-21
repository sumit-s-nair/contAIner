"""Base node interface for the contAIner pipeline.

Every pipeline node subclasses ``BaseNode`` and implements
``async run(state) -> PipelineState``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Mutable state object passed through the pipeline.

    Fields are populated progressively as each node runs.
    """

    # ── From System 1 (intent understanding) ───────────────────────────
    raw_instruction: str = ""
    intent_type: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    scope: str = "user"
    os_hint: str | None = None
    shell_type: str = "bash"
    confidence: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

    # ── MCP documentation layer ────────────────────────────────────────
    doc_chunks: list[dict[str, Any]] = field(default_factory=list)

    # ── From System 2 (command generation) ─────────────────────────────
    command_plan: dict[str, Any] | None = None

    # ── Pipeline metadata ──────────────────────────────────────────────
    errors: list[str] = field(default_factory=list)
    stage: str = "init"


class BaseNode(ABC):
    """Abstract base for all pipeline nodes.

    Each node receives the current ``PipelineState``, performs its work
    (possibly calling external services), and returns the updated state.
    """

    name: str = "BaseNode"

    @abstractmethod
    async def run(self, state: PipelineState) -> PipelineState:
        """Execute this node's logic and return the updated state."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
