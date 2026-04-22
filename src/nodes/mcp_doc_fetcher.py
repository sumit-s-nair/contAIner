"""MCPDocFetcher — orchestrator node that calls the MCP documentation server.

Replaces ``StubMCPDocFetcher``.  Sends structured intent data to
``http://localhost:11435/fetch_docs`` and appends the returned
``DocChunk`` to ``state.doc_chunks``.

Falls back to a minimal stub template when the server is unreachable.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .base import BaseNode, PipelineState

log = logging.getLogger("mcp_doc_fetcher")

# Default MCP server URL
_DEFAULT_MCP_URL = "http://localhost:11435/fetch_docs"
_TIMEOUT = 10  # seconds


# ═══════════════════════════════════════════════════════════════════════
# Stub fallback templates (used when the live server is unreachable)
# ═══════════════════════════════════════════════════════════════════════

_STUB_TEMPLATES: dict[str, dict[str, str]] = {
    "pip":    {"install": "pip install {package}", "uninstall": "pip uninstall -y {package}"},
    "npm":    {"install": "npm install {package}", "uninstall": "npm uninstall {package}"},
    "apt":    {"install": "sudo apt-get install -y {package}", "remove": "sudo apt-get remove -y {package}"},
    "brew":   {"install": "brew install {package}", "uninstall": "brew uninstall {package}"},
    "conda":  {"install": "conda install {package}", "remove": "conda remove {package}"},
    "docker": {"run": "docker run {package}", "pull": "docker pull {package}"},
    "cargo":  {"install": "cargo install {package}", "add": "cargo add {package}"},
    "go":     {"get": "go get {package}", "install": "go install {package}"},
    "maven":  {"add": "<dependency>...</dependency>"},
}

_SYSTEM_TOOL_BY_OS: dict[str, str] = {
    "linux": "apt",
    "macos": "brew",
    "windows": "conda",
}


def _intent_to_tool(intent_type: str) -> str:
    """Heuristically map an intent_type to a tool name.

    Examples::

        "install_package" → inferred from entities.runtime
        "install_runtime" → depends on OS
    """
    # Will be refined once we have entity data; return empty and let the
    # caller fill in from entities.
    return ""


def _build_request(state: PipelineState) -> dict[str, Any]:
    """Convert ``PipelineState`` fields into a /fetch_docs request body."""
    entities = state.entities or {}
    os_hint = state.os_hint or "linux"
    runtime = entities.get("runtime", "")
    package = entities.get("package", "")

    # Resolve tool from entities
    tool = entities.get("tool", "")
    if not tool:
        tool = _resolve_tool(state.intent_type, runtime, os_hint)

    # Resolve operation from intent_type
    operation = _intent_to_operation(state.intent_type)

    if state.intent_type.endswith("_runtime") and not package:
        package = runtime

    return {
        "tool": tool,
        "operation": operation,
        "package": package,
        "runtime": runtime,
        "os": os_hint,
        "version": entities.get("version", ""),
    }


def _resolve_tool(intent_type: str, runtime: str, os_hint: str) -> str:
    """Resolve tool from intent/runtime/os."""
    if intent_type.endswith("_runtime"):
        return _SYSTEM_TOOL_BY_OS.get((os_hint or "linux").lower(), "")
    return _runtime_to_tool(runtime)


def _runtime_to_tool(runtime: str) -> str:
    """Map a runtime name to its package manager."""
    mapping = {
        "python": "pip",
        "node": "npm",
        "nodejs": "npm",
        "rust": "cargo",
        "go": "go",
        "golang": "go",
        "java": "maven",
        "ruby": "gem",
    }
    return mapping.get(runtime.lower(), runtime.lower())


def _intent_to_operation(intent_type: str) -> str:
    """Map intent_type to a tool operation keyword."""
    mapping = {
        "install_package": "install",
        "install_runtime": "install",
        "update_package": "update",
        "update_runtime": "update",
        "remove_package": "uninstall",
        "remove_runtime": "uninstall",
        "list_packages": "list",
        "check_version": "show",
    }
    return mapping.get(intent_type, "install")


def _stub_fallback(request: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal DocChunk-like dict from the stub templates."""
    tool = request.get("tool", "")
    operation = request.get("operation", "install")
    package = request.get("package", "{package}")

    tool_templates = _STUB_TEMPLATES.get(tool, {})
    syntax = tool_templates.get(operation, f"{tool} {operation} {package}")
    syntax = syntax.replace("{package}", package or "{package}")

    return {
        "tool": tool,
        "operation": operation,
        "package_metadata": {},
        "command_syntax": syntax,
        "key_flags": [],
        "examples": [syntax],
        "source_urls": [],
        "tool_version": None,
        "tokens_estimate": len(syntax) // 4,
        "os_specific_notes": "",
        "error": "MCP server unreachable — using stub fallback",
    }


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator Node
# ═══════════════════════════════════════════════════════════════════════

class MCPDocFetcher(BaseNode):
    """Fetches documentation from the MCP server and adds it to
    ``state.doc_chunks``.

    If the server is unreachable or returns an error, a stub template
    is used instead so the pipeline never blocks on external failures.
    """

    name = "MCPDocFetcher"

    def __init__(self, mcp_url: str = _DEFAULT_MCP_URL, timeout: int = _TIMEOUT):
        self.mcp_url = mcp_url
        self.timeout = timeout

    async def run(self, state: PipelineState) -> PipelineState:
        state.stage = "mcp_doc_fetch"

        request_body = _build_request(state)

        # ── Validate we have enough to make the call ───────────────────
        if not request_body.get("tool"):
            log.warning("Cannot determine tool from state — skipping MCP fetch")
            state.errors.append("MCPDocFetcher: unable to determine tool from entities")
            return state

        # ── Call the MCP server ────────────────────────────────────────
        try:
            doc_chunk = self._call_server(request_body)
        except Exception as exc:
            log.warning("MCP server call failed (%s) — using stub fallback", exc)
            doc_chunk = _stub_fallback(request_body)

        state.doc_chunks.append(doc_chunk)
        log.info("DocChunk added for %s/%s (tokens≈%s)",
                 doc_chunk.get("tool"), doc_chunk.get("operation"),
                 doc_chunk.get("tokens_estimate"))

        return state

    def _call_server(self, body: dict[str, Any]) -> dict[str, Any]:
        """Synchronous HTTP POST to the MCP server using stdlib only.

        Uses ``urllib`` so the node has zero extra dependencies beyond
        the standard library.
        """
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.mcp_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw)
        except urllib.error.URLError as exc:
            raise ConnectionError(f"MCP server unreachable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"MCP server returned invalid JSON: {exc}") from exc
