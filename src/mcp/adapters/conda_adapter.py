"""Conda adapter.

Registry : https://api.anaconda.org/package/{channel}/{package}
Docs     : https://docs.conda.io/projects/conda/en/stable/commands/{operation}.html
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class CondaAdapter(BaseAdapter):
    tool_name = "conda"
    supported_operations = [
        "install", "remove", "update", "list",
        "create", "search", "info",
    ]

    _REGISTRY_URL = "https://api.anaconda.org/package/{channel}/{package}"
    _DOCS_URL = "https://docs.conda.io/projects/conda/en/stable/commands/{operation}.html"

    async def fetch(self, request: DocRequest) -> DocChunk:
        # Default to conda-forge, fallback to defaults channel
        channels = ["conda-forge", "main", "anaconda"]

        docs_url = self._DOCS_URL.format(operation=request.operation)

        try:
            docs_html_result = await self._fetch_html(docs_url)
        except Exception as e:
            docs_html_result = e

        docs_html = docs_html_result if not isinstance(docs_html_result, BaseException) else ""
        errors: list[str] = []

        # ── Docs ───────────────────────────────────────────────────────
        command_syntax = ""
        key_flags: list[dict[str, str]] = []
        examples: list[str] = []

        if docs_html:
            soup = self._parse_soup(docs_html)
            command_syntax = self._extract_synopsis(soup)
            key_flags = self._extract_options_table(soup)
            examples = self._extract_examples(soup)
        else:
            errors.append("conda docs fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            version_spec = f"={request.version}" if request.version else ""
            fallback = FALLBACK_TEMPLATES["conda"].get(request.operation, "")
            command_syntax = (fallback
                              .replace("{package}", request.package or "{package}")
                              .replace("{version_spec}", version_spec)
                              .replace("{channel_flag}", "")
                              .replace("{env_name}", "{env_name}"))

        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="conda",
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
            return ("On Windows, use Anaconda Prompt or add conda to PATH. "
                    "Alternatively use miniconda.")
        return ""
