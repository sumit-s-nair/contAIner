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
        registry_url = self._REGISTRY_URL.format(package=request.package)
        docs_url = self._DOCS_URL.format(operation=request.operation)

        registry_data, docs_html = await asyncio.gather(
            self._fetch_json(registry_url),
            self._fetch_html(docs_url),
            return_exceptions=True,
        )

        if isinstance(registry_data, BaseException):
            registry_data = {}
        if isinstance(docs_html, BaseException):
            docs_html = ""

        errors: list[str] = []

        # ── Registry metadata ──────────────────────────────────────────
        metadata: dict[str, Any] = {}
        if registry_data and "crate" in registry_data:
            crate = registry_data["crate"]
            metadata = {
                "name": crate.get("name", request.package),
                "description": crate.get("description", ""),
                "max_version": crate.get("max_version", ""),
                "homepage": crate.get("homepage", ""),
                "repository": crate.get("repository", ""),
                "downloads": crate.get("downloads", 0),
                "categories": [
                    c.get("category", "") for c in
                    (registry_data.get("categories") or [])
                ][:5],
                "keywords": [
                    k.get("keyword", "") for k in
                    (registry_data.get("keywords") or [])
                ][:10],
            }
        elif request.package:
            errors.append("crates.io registry fetch failed")

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
        if registry_data and "crate" in registry_data:
            source_urls.append(registry_url)
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="cargo",
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
            return ("On Windows, ensure Visual Studio Build Tools are installed "
                    "for compiling Rust crates with native dependencies.")
        return ""
