"""MCP Documentation Server package.

Provides a standalone HTTP server that fetches tool documentation
(pip, npm, apt, brew, conda, docker, cargo, go, maven) and returns
structured DocChunk payloads for the contAIner command planner.
"""

from .models import DocChunk, DocRequest
from .router import DocRouter
from .cache import TTLCache

__all__ = ["DocChunk", "DocRequest", "DocRouter", "TTLCache"]
