"""MCP Documentation Server package.

Provides a standalone HTTP server that fetches tool documentation
(pip, npm, apt, brew, conda, docker, cargo, go, maven) and returns
structured DocChunk payloads for the contAIner command planner.

The compression sub-pipeline (segmentation → extractive/abstractive →
reassembly) reduces DocChunk narrative text before System 2 prompt
insertion while preserving CODE segments byte-for-byte.

Lazy imports
------------
``DocRouter`` (and the adapter chain which requires ``bs4`` / ``aiohttp``)
is only imported when first accessed.  This allows the compression modules
and data models to be imported in test environments that do not have the
full adapter dependencies installed.
"""

from .models import DocChunk, DocRequest, CompressedDocChunk, CompressionReport
from .cache import TTLCache
from .compress import compress_chunk


def __getattr__(name: str):  # type: ignore[override]
    """Lazy-load heavy symbols on first access."""
    if name == "DocRouter":
        from .router import DocRouter as _DocRouter
        return _DocRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DocChunk",
    "DocRequest",
    "DocRouter",
    "TTLCache",
    "CompressedDocChunk",
    "CompressionReport",
    "compress_chunk",
]
