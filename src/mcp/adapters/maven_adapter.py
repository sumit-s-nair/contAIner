"""Maven adapter.

Registry : https://search.maven.org/solrsearch/select?q={package}&rows=1&wt=json
Docs     : https://maven.apache.org/guides/introduction/introduction-to-the-pom.html
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class MavenAdapter(BaseAdapter):
    tool_name = "maven"
    supported_operations = [
        "add", "install", "clean", "package", "test", "compile",
    ]

    _SEARCH_URL = "https://search.maven.org/solrsearch/select"
    _DOCS_URL = "https://maven.apache.org/guides/introduction/introduction-to-the-pom.html"

    async def fetch(self, request: DocRequest) -> DocChunk:
        # Build search query — package may be "groupId:artifactId" or just a name
        package = request.package
        if ":" in package:
            group_id, artifact_id = package.split(":", 1)
            query = f"g:{group_id} AND a:{artifact_id}"
        else:
            query = package

        search_url = f"{self._SEARCH_URL}?q={query}&rows=5&wt=json"
        docs_url = self._DOCS_URL

        try:
            docs_html = await self._fetch_html(docs_url)
        except Exception as e:
            docs_html = e

        if isinstance(docs_html, BaseException):
            docs_html = ""

        errors: list[str] = []

        # ── Docs ───────────────────────────────────────────────────────
        key_flags: list[dict[str, str]] = []
        examples: list[str] = []

        if docs_html:
            soup = self._parse_soup(docs_html)
            key_flags = self._extract_options_table(soup)
            examples = self._extract_examples(soup)
        else:
            errors.append("Maven docs fetch failed")

        # ── Build command syntax ───────────────────────────────────────
        if request.operation == "add":
            g_id = "{groupId}"
            a_id = "{artifactId}"
            if request.package:
                if ":" in request.package:
                    g_id, a_id = request.package.split(":", 1)
                else:
                    a_id = request.package
            command_syntax = FALLBACK_TEMPLATES["maven"]["add"].format(
                group_id=g_id,
                artifact_id=a_id,
                version="{version}",
            )
        else:
            command_syntax = FALLBACK_TEMPLATES["maven"].get(request.operation, "")

        if not examples:
            examples.append(command_syntax)

        source_urls = []
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="maven",
            operation=request.operation,
            command_syntax=command_syntax,
            key_flags=key_flags,
            examples=examples[:5],
            source_urls=source_urls,
            tool_version=None,
            os_specific_notes=self._os_notes(request),
            error="; ".join(errors) if errors else None,
        )
        chunk.estimate_tokens()
        return chunk

    @staticmethod
    def _os_notes(request: DocRequest) -> str:
        notes: list[str] = []
        if request.os == "windows":
            notes.append("On Windows, use `mvn.cmd` or ensure Maven bin is on PATH.")
        notes.append("Add the <dependency> block to your pom.xml <dependencies> section.")
        return " ".join(notes)
