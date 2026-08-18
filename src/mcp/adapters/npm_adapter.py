"""npm / yarn / pnpm adapter.

Registry : https://registry.npmjs.org/{package}  (shared for all three)
npm docs : https://docs.npmjs.com/cli/v10/commands/npm-{operation}
yarn docs: https://yarnpkg.com/cli/{operation}
pnpm docs: https://pnpm.io/cli/{operation}
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class NpmAdapter(BaseAdapter):
    """Handles npm, yarn, and pnpm via a single adapter."""

    tool_name = "npm"
    supported_operations = ["install", "uninstall", "list", "update", "init", "run"]

    _REGISTRY_URL = "https://registry.npmjs.org/{package}"

    _DOCS_URLS: dict[str, str] = {
        "npm":  "https://docs.npmjs.com/cli/v10/commands/npm-{operation}",
        "yarn": "https://yarnpkg.com/cli/{operation}",
        "pnpm": "https://pnpm.io/cli/{operation}",
    }

    def _resolve_tool(self, request: DocRequest) -> str:
        """Return normalised tool key (npm | yarn | pnpm)."""
        t = request.tool.lower()
        if t in ("yarn", "pnpm"):
            return t
        return "npm"

    async def fetch(self, request: DocRequest) -> DocChunk:
        resolved_tool = self._resolve_tool(request)

        # Map yarn "install" → "add" in docs URL
        op_for_docs = request.operation
        if resolved_tool == "yarn" and request.operation == "install":
            op_for_docs = "add"

        docs_template = self._DOCS_URLS.get(resolved_tool, self._DOCS_URLS["npm"])
        docs_url = docs_template.format(operation=op_for_docs)

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
            command_syntax = self._extract_synopsis(soup)
            key_flags = self._extract_options_table(soup)
            examples = self._extract_examples(soup)
        else:
            errors.append(f"{resolved_tool} docs fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            tool_templates = FALLBACK_TEMPLATES.get(resolved_tool,
                                                     FALLBACK_TEMPLATES["npm"])
            version_spec = f"@{request.version}" if request.version else ""
            fallback = tool_templates.get(request.operation, "")
            command_syntax = (fallback
                              .replace("{package}", request.package or "{package}")
                              .replace("{version_spec}", version_spec))

        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool=resolved_tool,
            operation=request.operation,
            command_syntax=command_syntax,
            key_flags=key_flags,
            examples=examples,
            source_urls=source_urls,
            tool_version=None,
            os_specific_notes=self._os_notes(request, resolved_tool),
            error="; ".join(errors) if errors else None,
        )
        chunk.estimate_tokens()
        return chunk

    @staticmethod
    def _os_notes(request: DocRequest, resolved_tool: str) -> str:
        notes = []
        if request.os == "windows":
            notes.append(f"On Windows, ensure Node.js is installed and "
                         f"`{resolved_tool}` is on PATH.")
        return " ".join(notes)
