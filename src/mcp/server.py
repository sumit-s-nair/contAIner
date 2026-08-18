"""MCP Documentation Server — pure-stdlib HTTP server.

Exposes:
    POST /fetch_docs      — main documentation lookup
    GET  /health          — liveness + adapter list
    GET  /supported_tools — full adapter map

Usage::

    python src/mcp/server.py --port 11435
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import os
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

# ── Ensure the project root is on sys.path so ``src.mcp`` is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.mcp.router import DocRouter
from src.mcp.models import DocRequest, DocChunk
from src.mcp.cache import TTLCache, REGISTRY_TTL, DOCS_TTL
from src.mcp.compress import compress_chunk

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mcp-server")

# ── Globals ────────────────────────────────────────────────────────────
_router = DocRouter()
_cache = TTLCache(max_size=500)


# ── Request normalization / unsupported-tool fallbacks ────────────────
_TOOL_ALIASES: dict[str, str] = {
    "apt-get": "apt",
    "aptget": "apt",
    "homebrew": "brew",
    "dockercompose": "docker-compose",
    "golang": "go",
    "mvn": "maven",
    "node": "npm",
    "nodejs": "npm",
    "node.js": "npm",
    "javascript": "npm",
    "js": "npm",
    "coffeescript": "npm",
    "coffee": "npm",
}

_RUNTIME_TO_SYSTEM_PACKAGE: dict[str, str] = {
    "c": "build-essential",
    "gcc": "gcc",
    "erlang": "erlang",
    "gnuplot": "gnuplot",
    "octave": "octave",
    "perl": "perl",
}

_OPERATION_ALIASES: dict[str, dict[str, str]] = {
    "apt": {
        "uninstall": "remove",
        "delete": "remove",
    },
    "brew": {
        "remove": "uninstall",
        "delete": "uninstall",
    },
    "npm": {
        "add": "install",
        "remove": "uninstall",
        "delete": "uninstall",
    },
}

_UNSUPPORTED_FALLBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "cpan": {
        "install": "cpan -i {package}",
        "uninstall": "cpan -U {package}",
        "list": "cpan -l",
        "show": "cpan -D {package}",
    },
}


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool / operation aliases and infer package-manager defaults."""
    normalized = dict(body)

    tool = str(body.get("tool", "") or "").strip().lower()
    runtime = str(body.get("runtime", "") or "").strip().lower()
    operation = str(body.get("operation", "") or "").strip().lower()
    os_hint = str(body.get("os", "linux") or "linux").strip().lower()
    package = str(body.get("package", "") or "").strip()

    # Some callers send runtime labels instead of canonical tool names.
    if not tool and runtime:
        tool = runtime

    compact_tool = _compact_token(tool)
    mapped_tool = _TOOL_ALIASES.get(tool) or _TOOL_ALIASES.get(compact_tool) or tool

    # For runtime binaries on Linux/macOS, route install/remove to system PM.
    if mapped_tool == tool and compact_tool in _RUNTIME_TO_SYSTEM_PACKAGE:
        system_tool = "apt" if os_hint == "linux" else "brew" if os_hint == "macos" else ""
        if system_tool:
            mapped_tool = system_tool
            if not package:
                package = _RUNTIME_TO_SYSTEM_PACKAGE[compact_tool]

    # Perl with a concrete module name is usually a CPAN workflow.
    if compact_tool == "perl":
        perl_pkg = package.lower()
        if package and perl_pkg not in {"perl", "perl5"}:
            mapped_tool = "cpan"
        elif not package and mapped_tool in {"apt", "brew"}:
            package = "perl"

    operation = _OPERATION_ALIASES.get(mapped_tool, {}).get(operation, operation)

    normalized["tool"] = mapped_tool
    normalized["operation"] = operation
    normalized["os"] = os_hint
    normalized["package"] = package

    return normalized


def _build_unsupported_syntax(request: DocRequest) -> str:
    tool = request.tool.lower()
    operation = request.operation.lower()
    package = request.package

    syntax = _UNSUPPORTED_FALLBACK_TEMPLATES.get(tool, {}).get(operation, "")
    if syntax:
        return syntax.replace("{package}", package or "{package}")

    pieces = [request.tool, request.operation, package]
    return " ".join(p for p in pieces if p).strip()


def _build_unsupported_chunk(request: DocRequest, reason: str) -> DocChunk:
    syntax = _build_unsupported_syntax(request)
    chunk = DocChunk(
        tool=request.tool,
        operation=request.operation,
        command_syntax=syntax,
        key_flags=[],
        examples=[syntax] if syntax else [],
        source_urls=[],
        tool_version=None,
        os_specific_notes=(
            "No dedicated adapter is available for this tool yet; "
            "returned a generic fallback command syntax."
        ),
        error=reason,
    )
    chunk.estimate_tokens()
    return chunk


# ═══════════════════════════════════════════════════════════════════════
# Async handler logic
# ═══════════════════════════════════════════════════════════════════════

async def _handle_fetch_docs(body: dict[str, Any]) -> tuple[int, dict]:
    """Process a /fetch_docs request.  Returns (status_code, response_dict)."""

    body = _normalize_request_body(body)

    # ── Validate required fields ───────────────────────────────────────
    tool = body.get("tool", "").strip()
    operation = body.get("operation", "").strip()
    if not tool or not operation:
        return 400, {
            "error": "Both 'tool' and 'operation' are required fields.",
            "supported_tools": _router.supported_tools(),
        }

    # Parse ?compress=0 opt-out flag (passed via body for POST convenience)
    enable_compression: bool = str(body.get("compress", "1")).strip() != "0"

    request = DocRequest.from_dict(body)

    # ── Cache lookup ───────────────────────────────────────────────────
    cache_key = request.cache_key()
    cached: DocChunk | None = _cache.get(cache_key)
    if cached is not None:
        log.info("cache HIT  %s", cache_key)
        return 200, cached.to_dict()

    log.info("cache MISS %s — fetching", cache_key)

    # ── Route to adapter ───────────────────────────────────────────────
    try:
        chunk: DocChunk = await _router.route(request)
    except ValueError as exc:
        chunk = _build_unsupported_chunk(request, str(exc))
        _cache.set(cache_key, chunk, ttl=REGISTRY_TTL)
        return 200, chunk.to_dict()
    except Exception as exc:
        log.exception("Unhandled adapter error")
        return 500, {"error": f"Internal error: {exc}"}

    # ── Doc-compression stage (after retrieval, before cache + return) ──
    if enable_compression:
        try:
            chunk = compress_chunk(chunk, method="extractive", cache=_cache)
            log.info(
                "compressed %s → %d tok (was %d tok, prose_ratio=%.2f)",
                cache_key,
                chunk.tokens_estimate,
                getattr(chunk, "compression_report", None) and
                    chunk.compression_report.original_token_count or chunk.tokens_estimate,
                getattr(chunk, "compression_report", None) and
                    chunk.compression_report.prose_reduction_ratio or 0.0,
            )
        except Exception as exc:
            log.warning("Compression failed (%s) — using raw chunk", exc)

    # ── Cache the result ───────────────────────────────────────────────
    # Use the longer TTL if we got docs content, shorter if only registry
    ttl = DOCS_TTL if chunk.command_syntax else REGISTRY_TTL
    _cache.set(cache_key, chunk, ttl=ttl)

    return 200, chunk.to_dict()


def _handle_health() -> tuple[int, dict]:
    """Process GET /health."""
    return 200, {
        "status": "ok",
        "adapters": _router.supported_tools(),
        "cache_size": _cache.size,
    }


def _handle_supported_tools() -> tuple[int, dict]:
    """Process GET /supported_tools."""
    return 200, {
        "tools": _router.adapter_info(),
        "operations": _router.supported_operations(),
    }


# ═══════════════════════════════════════════════════════════════════════
# HTTP Request Handler
# ═══════════════════════════════════════════════════════════════════════

class MCPRequestHandler(BaseHTTPRequestHandler):
    """Thin HTTP handler that delegates to async functions via the event loop."""

    # Suppress default stderr logging from BaseHTTPRequestHandler
    def log_message(self, fmt: str, *args: Any) -> None:
        log.info(fmt, *args)

    # ── Helpers ────────────────────────────────────────────────────────

    def _send_json(self, status: int, data: dict) -> None:
        payload = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ── GET routes ─────────────────────────────────────────────────────

    def do_GET(self) -> None:
        path = self.path.rstrip("/")

        if path == "/health":
            status, data = _handle_health()
            self._send_json(status, data)
        elif path == "/supported_tools":
            status, data = _handle_supported_tools()
            self._send_json(status, data)
        else:
            self._send_json(404, {"error": f"Not found: {self.path}"})

    # ── POST routes ────────────────────────────────────────────────────

    def do_POST(self) -> None:
        path = self.path.rstrip("/")

        if path == "/fetch_docs":
            raw = self._read_body()
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON body"})
                return

            # Run the async handler on the current event loop (or create one)
            loop = _get_or_create_loop()
            status, data = loop.run_until_complete(_handle_fetch_docs(body))
            self._send_json(status, data)
        else:
            self._send_json(404, {"error": f"Not found: {self.path}"})

    # ── OPTIONS (CORS preflight) ───────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ═══════════════════════════════════════════════════════════════════════
# Event-loop helpers
# ═══════════════════════════════════════════════════════════════════════

_loop: asyncio.AbstractEventLoop | None = None


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


# ═══════════════════════════════════════════════════════════════════════
# Server entry point
# ═══════════════════════════════════════════════════════════════════════

def run_server(host: str = "0.0.0.0", port: int = 11435) -> None:
    """Start the MCP documentation server."""
    server = HTTPServer((host, port), MCPRequestHandler)
    log.info("🚀  MCP Documentation Server starting on http://%s:%d", host, port)
    log.info("   Adapters: %s", ", ".join(_router.supported_tools()))
    log.info("   Endpoints:")
    log.info("     POST /fetch_docs")
    log.info("     GET  /health")
    log.info("     GET  /supported_tools")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("⏹  Shutting down…")
    finally:
        server.server_close()
        if _loop and not _loop.is_closed():
            _loop.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Documentation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=11435, help="Bind port")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
