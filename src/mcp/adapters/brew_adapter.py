"""Homebrew adapter.

Registry : https://formulae.brew.sh/api/formula/{package}.json
Docs     : https://docs.brew.sh/Manpage
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class BrewAdapter(BaseAdapter):
    tool_name = "brew"
    supported_operations = [
        "install", "uninstall", "update", "upgrade",
        "list", "search", "info",
    ]

    _REGISTRY_URL = "https://formulae.brew.sh/api/formula/{package}.json"
    _CASK_URL = "https://formulae.brew.sh/api/cask/{package}.json"
    _DOCS_URL = "https://docs.brew.sh/Manpage"

    async def fetch(self, request: DocRequest) -> DocChunk:
        docs_url = self._DOCS_URL
        
        try:
            docs_html = await self._fetch_html(docs_url)
        except Exception as e:
            docs_html = e
            
        if isinstance(docs_html, BaseException):
            docs_html = ""

        errors: list[str] = []

        # ── Docs ───────────────────────────────────────────────────────
        command_syntax = ""
        key_flags: list[dict[str, str]] = []
        examples: list[str] = []

        if docs_html:
            soup = self._parse_soup(docs_html)

            # Find the section for the specific operation
            for heading in soup.find_all(["h2", "h3"]):
                heading_text = heading.get_text(" ", strip=True).lower()
                if request.operation in heading_text:
                    # Get content until next heading
                    content_parts: list[str] = []
                    sibling = heading.find_next_sibling()
                    while sibling and sibling.name not in ("h2", "h3"):
                        text = sibling.get_text(" ", strip=True)
                        if text:
                            content_parts.append(text)
                        sibling = sibling.find_next_sibling()
                    if content_parts:
                        command_syntax = content_parts[0][:500]

                    # Extract flags from this section
                    section_soup = self._parse_soup(str(heading.find_next("dl") or ""))
                    key_flags = self._extract_options_table(section_soup)
                    break

            examples = self._extract_examples(soup)
        else:
            errors.append("brew docs fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            fallback = FALLBACK_TEMPLATES["brew"].get(request.operation, "")
            command_syntax = fallback.replace(
                "{package}", request.package or "{package}"
            )


        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []

        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="brew",
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

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _os_notes(request: DocRequest) -> str:
        notes: list[str] = []
        if request.os == "linux":
            notes.append("Using Homebrew on Linux (Linuxbrew). "
                         "Ensure /home/linuxbrew/.linuxbrew/bin is on PATH.")
        if request.os == "windows":
            notes.append("Homebrew is not natively supported on Windows. "
                         "Use WSL or consider winget/chocolatey.")

        return " ".join(notes)
