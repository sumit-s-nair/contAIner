"""Cargo (Rust) adapter.

Registry : https://crates.io/api/v1/crates/{package}
Docs     : https://doc.rust-lang.org/cargo/commands/cargo-{operation}.html
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class CargoAdapter(BaseAdapter):
    tool_name = "cargo"
    supported_operations = [
        "install", "build", "run", "test", "update", "add", "new",
    ]

    _REGISTRY_URL = "https://crates.io/api/v1/crates/{package}"
    _DOCS_URL = "https://doc.rust-lang.org/cargo/commands/cargo-{operation}.html"

    async def fetch(self, request: DocRequest) -> DocChunk:
        docs_url = self._DOCS_URL.format(operation=request.operation)

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
            errors.append("cargo docs fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            version_spec = f"@{request.version}" if request.version else ""
            fallback = FALLBACK_TEMPLATES["cargo"].get(request.operation, "")
            command_syntax = (fallback
                              .replace("{package}", request.package or "{package}")
                              .replace("{version_spec}", version_spec)
                              .replace("{project_name}", "{project_name}"))

        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="cargo",
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
            return ("On Windows, ensure Visual Studio Build Tools are installed "
                    "for compiling Rust crates with native dependencies.")
        return ""
