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
        docs_url = self._DOCS_URL.format(operation=request.operation)

        try:
            docs_html = await self._fetch_html(docs_url)
        except Exception as e:
            docs_html = e

        # ── Handle exceptions from gather ──────────────────────────────
        if isinstance(docs_html, BaseException):
            docs_html = ""

        errors: list[str] = []

        # ── Docs ─────────────────────────────────────────────────
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
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="pip",
            operation=request.operation,
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
