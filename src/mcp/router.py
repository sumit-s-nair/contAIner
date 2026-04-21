"""DocRouter — maps (tool, operation, os) → adapter + fetch strategy.

Handles tool normalisation, OS-specific notes, and cross-platform
suggestions (e.g. apt on Windows → suggest WSL).
"""

from __future__ import annotations

from typing import Any

from .adapters import (
    BaseAdapter,
    PipAdapter,
    NpmAdapter,
    AptAdapter,
    BrewAdapter,
    CondaAdapter,
    DockerAdapter,
    CargoAdapter,
    GoAdapter,
    MavenAdapter,
)
from .models import DocRequest


# ═══════════════════════════════════════════════════════════════════════
# Adapter singleton pool — one instance per adapter class
# ═══════════════════════════════════════════════════════════════════════

_ADAPTER_INSTANCES: dict[str, BaseAdapter] = {}


def _get_adapter(cls: type[BaseAdapter]) -> BaseAdapter:
    """Return (or create) a singleton adapter instance."""
    key = cls.__name__
    if key not in _ADAPTER_INSTANCES:
        _ADAPTER_INSTANCES[key] = cls()
    return _ADAPTER_INSTANCES[key]


# ═══════════════════════════════════════════════════════════════════════
# Tool → Adapter mapping
# ═══════════════════════════════════════════════════════════════════════

_TOOL_ADAPTER_MAP: dict[str, type[BaseAdapter]] = {
    "pip":            PipAdapter,
    "npm":            NpmAdapter,
    "yarn":           NpmAdapter,
    "pnpm":           NpmAdapter,
    "apt":            AptAdapter,
    "apt-get":        AptAdapter,
    "brew":           BrewAdapter,
    "homebrew":       BrewAdapter,
    "conda":          CondaAdapter,
    "docker":         DockerAdapter,
    "docker-compose": DockerAdapter,
    "cargo":          CargoAdapter,
    "go":             GoAdapter,
    "golang":         GoAdapter,
    "maven":          MavenAdapter,
    "mvn":            MavenAdapter,
}


class DocRouter:
    """Routes a ``DocRequest`` to the correct adapter.

    Usage::

        router = DocRouter()
        chunk = await router.route(request)
    """

    def __init__(self) -> None:
        self._map = _TOOL_ADAPTER_MAP

    # ── Main routing entry point ───────────────────────────────────────

    async def route(self, request: DocRequest) -> Any:
        """Resolve *request* to an adapter and return its ``DocChunk``.

        Raises ``ValueError`` if the tool is unrecognised.
        """
        adapter = self._resolve(request)
        chunk = await adapter.fetch(request)

        # Append OS cross-platform note if applicable
        os_note = self._cross_platform_note(request)
        if os_note and not chunk.os_specific_notes:
            chunk.os_specific_notes = os_note

        return chunk

    # ── Adapter resolution ─────────────────────────────────────────────

    def _resolve(self, request: DocRequest) -> BaseAdapter:
        tool_key = request.tool.lower().strip()
        adapter_cls = self._map.get(tool_key)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported tool: '{request.tool}'. "
                f"Supported tools: {', '.join(sorted(self.supported_tools()))}"
            )
        return _get_adapter(adapter_cls)

    # ── Cross-platform advisory notes ──────────────────────────────────

    @staticmethod
    def _cross_platform_note(request: DocRequest) -> str:
        tool = request.tool.lower()
        os_ = request.os.lower()

        if tool == "apt" and os_ == "windows":
            return ("apt is not available on Windows natively. "
                    "Use WSL (Windows Subsystem for Linux) to run apt commands, "
                    "or consider winget / chocolatey.")
        if tool == "apt" and os_ == "macos":
            return ("apt is not available on macOS. "
                    "Use `brew` (Homebrew) instead.")
        if tool == "brew" and os_ == "windows":
            return ("Homebrew is not supported on Windows. "
                    "Consider winget or chocolatey.")
        if tool == "brew" and os_ == "linux":
            return ("Using Homebrew on Linux (Linuxbrew). "
                    "Ensure /home/linuxbrew/.linuxbrew/bin is on PATH.")
        return ""

    # ── Introspection ──────────────────────────────────────────────────

    def supported_tools(self) -> list[str]:
        """Return deduplicated list of canonical tool names."""
        seen: set[str] = set()
        result: list[str] = []
        for key, cls in self._map.items():
            canonical = cls.tool_name or key
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return sorted(result)

    def supported_operations(self) -> dict[str, list[str]]:
        """Return ``{tool: [operations]}`` map for all adapters."""
        ops: dict[str, list[str]] = {}
        for key, cls in self._map.items():
            canonical = cls.tool_name or key
            if canonical not in ops:
                adapter = _get_adapter(cls)
                ops[canonical] = adapter.supported_operations
        return ops

    def adapter_info(self) -> list[dict[str, Any]]:
        """Return a list of adapter metadata dicts (for /health, /supported_tools)."""
        seen: set[str] = set()
        info: list[dict[str, Any]] = []
        for key, cls in self._map.items():
            canonical = cls.tool_name or key
            if canonical not in seen:
                seen.add(canonical)
                adapter = _get_adapter(cls)
                info.append({
                    "tool": canonical,
                    "adapter": cls.__name__,
                    "operations": adapter.supported_operations,
                })
        return info
