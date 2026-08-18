"""Data models for the MCP Documentation Server.

Defines DocChunk (the output payload), DocRequest (the input payload),
CompressedDocChunk / CompressionReport (compression pipeline outputs),
and serialization helpers used across adapters, router, and server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


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


# ══════════════════════════════════════════════════════════════════════
# Compression pipeline output types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CompressionReport:
    """Metadata produced by the reassembly stage after doc-compression.

    ``code_token_count`` must always equal the original code token count
    (enforced via hard assertion in ``reassemble.py``).
    """
    original_token_count: int = 0
    compressed_token_count: int = 0
    code_token_count: int = 0          # tokens in all CODE segments (unchanged)
    prose_reduction_ratio: float = 0.0  # 1 - compressed_prose / original_prose
    method: Literal["extractive", "abstractive"] = "extractive"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompressedDocChunk(DocChunk):
    """DocChunk extended with a compression report.

    All DocChunk fields are inherited unchanged; ``compression_report``
    is added by the reassembly stage.
    """
    compression_report: CompressionReport | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["compression_report"] = (
            self.compression_report.to_dict()
            if self.compression_report is not None
            else None
        )
        return d

    @classmethod
    def from_doc_chunk(
        cls,
        chunk: DocChunk,
        report: CompressionReport | None = None,
    ) -> "CompressedDocChunk":
        """Wrap an existing ``DocChunk`` as a ``CompressedDocChunk``."""
        return cls(
            tool=chunk.tool,
            operation=chunk.operation,
            command_syntax=chunk.command_syntax,
            key_flags=chunk.key_flags,
            examples=chunk.examples,
            source_urls=chunk.source_urls,
            tool_version=chunk.tool_version,
            tokens_estimate=chunk.tokens_estimate,
            os_specific_notes=chunk.os_specific_notes,
            error=chunk.error,
            compression_report=report,
        )
