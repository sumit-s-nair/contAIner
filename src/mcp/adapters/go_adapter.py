"""Go adapter.

Registry : https://pkg.go.dev/{package} (scrape module info)
Docs     : https://pkg.go.dev/cmd/go  (extract relevant subcommand section)
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class GoAdapter(BaseAdapter):
    tool_name = "go"
    supported_operations = [
        "get", "install", "build", "run", "test", "mod tidy", "mod init",
    ]

    _MODULE_URL = "https://pkg.go.dev/{package}"
    _DOCS_URL = "https://pkg.go.dev/cmd/go"

    # proxy.golang.org JSON endpoint for module metadata
    _PROXY_URL = "https://proxy.golang.org/{package}/@latest"

    async def fetch(self, request: DocRequest) -> DocChunk:
        docs_url = self._DOCS_URL

        # Use Go module proxy for structured metadata
        module_url = self._MODULE_URL.format(package=request.package) if request.package else ""

        try:
            docs_html = await self._fetch_html(docs_url)
        except Exception as e:
            docs_html = e

        errors: list[str] = []

        # ── Docs — extract the relevant subcommand section ─────────────
        command_syntax = ""
        key_flags: list[dict[str, str]] = []
        examples: list[str] = []

        if docs_html:
            soup = self._parse_soup(docs_html)

            # Normalise operation for matching in docs
            op_patterns = [request.operation]
            if " " in request.operation:
                op_patterns.append(request.operation.replace(" ", " "))  # "mod tidy"

            # Scan headings for the relevant subcommand
            for heading in soup.find_all(["h2", "h3"]):
                heading_text = heading.get_text(" ", strip=True).lower()
                if any(p in heading_text for p in op_patterns):
                    # Collect content until next same-level heading
                    parts: list[str] = []
                    sibling = heading.find_next_sibling()
                    while sibling and sibling.name not in ("h2", "h3"):
                        text = sibling.get_text(" ", strip=True)
                        if text:
                            parts.append(text)
                        # Extract code blocks as examples
                        for pre in sibling.find_all("pre") if hasattr(sibling, "find_all") else []:
                            t = pre.get_text(" ", strip=True)
                            if t and len(t) < 300:
                                examples.append(t)
                        sibling = sibling.find_next_sibling()

                    if parts:
                        command_syntax = parts[0][:500]
                    break

            if not key_flags:
                key_flags = self._extract_options_table(soup)
        else:
            errors.append("Go docs fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            version_spec = f"@{request.version}" if request.version else ""
            fallback = FALLBACK_TEMPLATES["go"].get(request.operation, "")
            command_syntax = (fallback
                              .replace("{package}", request.package or "{package}")
                              .replace("{version_spec}", version_spec)
                              .replace("{module}", request.package or "{module}"))

        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []
        if module_url:
            source_urls.append(module_url)
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="go",
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
        if request.os == "windows":
            return ("On Windows, ensure Go is installed and GOPATH/GOBIN "
                    "are configured. Use `go env` to verify.")
        return ""
