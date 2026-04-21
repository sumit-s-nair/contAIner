"""Data models for the MCP Documentation Server.

Defines DocChunk (the output payload), DocRequest (the input payload),
and serialization helpers used across adapters, router, and server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DocRequest:
    """Incoming request for documentation lookup.

    Fields mirror the JSON body of POST /fetch_docs.
    """
    tool: str
    operation: str
    package: str = ""
    runtime: str = ""
    os: str = "linux"
    version: str = ""

    # ── Serialization ──────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocRequest":
        return cls(
            tool=data.get("tool", ""),
            operation=data.get("operation", ""),
            package=data.get("package", ""),
            runtime=data.get("runtime", ""),
            os=data.get("os", "linux"),
            version=data.get("version", ""),
        )

    def cache_key(self) -> tuple[str, str, str, str]:
        """Return the 4-tuple used as cache key."""
        return (self.tool.lower(), self.operation.lower(),
                self.package.lower(), self.os.lower())


@dataclass
class DocChunk:
    """Documentation chunk returned by an adapter.

    Every adapter must populate as many fields as possible.  Fields that
    could not be resolved are left at their defaults so the response is
    always structurally complete.
    """
    tool: str = ""
    operation: str = ""
    package_metadata: dict[str, Any] = field(default_factory=dict)
    command_syntax: str = ""
    key_flags: list[dict[str, str]] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    tool_version: str | None = None
    tokens_estimate: int = 0
    os_specific_notes: str = ""
    error: str | None = None

    # ── Serialization ──────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocChunk":
        return cls(
            tool=data.get("tool", ""),
            operation=data.get("operation", ""),
            package_metadata=data.get("package_metadata", {}),
            command_syntax=data.get("command_syntax", ""),
            key_flags=data.get("key_flags", []),
            examples=data.get("examples", []),
            source_urls=data.get("source_urls", []),
            tool_version=data.get("tool_version"),
            tokens_estimate=data.get("tokens_estimate", 0),
            os_specific_notes=data.get("os_specific_notes", ""),
            error=data.get("error"),
        )

    def estimate_tokens(self) -> int:
        """Rough token count (≈ 4 chars per token) for context-window planning."""
        text = json.dumps(self.to_dict(), default=str)
        self.tokens_estimate = max(1, len(text) // 4)
        return self.tokens_estimate
