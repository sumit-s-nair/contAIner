"""Pip / PyPI adapter.

Registry : https://pypi.org/pypi/{package}/json
Docs     : https://pip.pypa.io/en/stable/cli/pip_{operation}/
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class PipAdapter(BaseAdapter):
    tool_name = "pip"
    supported_operations = ["install", "uninstall", "list", "show", "freeze", "upgrade"]

    _REGISTRY_URL = "https://pypi.org/pypi/{package}/json"
    _DOCS_URL = "https://pip.pypa.io/en/stable/cli/pip_{operation}/"

    async def fetch(self, request: DocRequest) -> DocChunk:
        registry_url = self._REGISTRY_URL.format(package=request.package)
        docs_url = self._DOCS_URL.format(operation=request.operation)

        registry_data, docs_html = await asyncio.gather(
            self._fetch_json(registry_url),
            self._fetch_html(docs_url),
            return_exceptions=True,
        )

        # ── Handle exceptions from gather ──────────────────────────────
        if isinstance(registry_data, BaseException):
            registry_data = {}
        if isinstance(docs_html, BaseException):
            docs_html = ""

        errors: list[str] = []

        # ── Parse registry ─────────────────────────────────────────────
        metadata: dict[str, Any] = {}
        if registry_data:
            info = registry_data.get("info", {})
            metadata = {
                "name": info.get("name", request.package),
                "version": info.get("version", ""),
                "summary": info.get("summary", ""),
                "home_page": info.get("home_page", ""),
                "license": info.get("license", ""),
                "requires_python": info.get("requires_python", ""),
                "requires_dist": (info.get("requires_dist") or [])[:15],
            }
        else:
            errors.append("PyPI registry fetch failed")

        # ── Parse docs ─────────────────────────────────────────────────
        command_syntax = ""
        key_flags: list[dict[str, str]] = []
        examples: list[str] = []

        if docs_html:
            soup = self._parse_soup(docs_html)
            command_syntax = self._extract_synopsis(soup)
            key_flags = self._extract_options_table(soup)
            examples = self._extract_examples(soup)
        else:
            errors.append("pip docs fetch failed")

        # ── Fallback if docs failed ────────────────────────────────────
        if not command_syntax:
            version_spec = self._version_spec(request)
            fallback = FALLBACK_TEMPLATES["pip"].get(request.operation, "")
            command_syntax = (fallback
                              .replace("{package}", request.package or "{package}")
                              .replace("{version_spec}", version_spec))

        if not examples and command_syntax:
            examples = [command_syntax]

        # ── Build source URLs ──────────────────────────────────────────
        source_urls = []
        if registry_data:
            source_urls.append(registry_url)
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="pip",
            operation=request.operation,
            package_metadata=metadata,
            command_syntax=command_syntax,
            key_flags=key_flags,
            examples=examples,
            source_urls=source_urls,
            tool_version=None,
            os_specific_notes=self._os_notes(request),
            error="; ".join(errors) if errors else None,
        )
        chunk.estimate_tokens()
        return chunk

    @staticmethod
    def _os_notes(request: DocRequest) -> str:
        if request.os == "windows":
            return ("On Windows, use `py -m pip` if `pip` is not on PATH. "
                    "Consider using a virtual environment.")
        return ""
