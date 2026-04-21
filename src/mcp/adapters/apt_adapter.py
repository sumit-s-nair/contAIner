"""apt adapter.

No registry API — manpages only.
Docs: https://manpages.debian.org/bookworm/apt/apt.8.en.html
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class AptAdapter(BaseAdapter):
    tool_name = "apt"
    supported_operations = [
        "install", "remove", "update", "upgrade", "list",
        "search", "show", "purge", "autoremove",
    ]

    _MANPAGE_URL = "https://manpages.debian.org/bookworm/apt/apt.8.en.html"
    _APT_GET_URL = "https://manpages.debian.org/bookworm/apt/apt-get.8.en.html"
    _APT_CACHE_URL = "https://manpages.debian.org/bookworm/apt/apt-cache.8.en.html"

    def _choose_docs_url(self, operation: str) -> str:
        """Pick the right manpage for the operation."""
        if operation in ("search", "show"):
            return self._APT_CACHE_URL
        if operation in ("install", "remove", "purge", "autoremove"):
            return self._APT_GET_URL
        return self._MANPAGE_URL

    async def fetch(self, request: DocRequest) -> DocChunk:
        docs_url = self._choose_docs_url(request.operation)

        # apt has no registry API, but we can try fetching dpkg info page
        dpkg_url = ""
        if request.package:
            dpkg_url = f"https://packages.debian.org/bookworm/{request.package}"

        tasks = [self._fetch_html(docs_url)]
        if dpkg_url:
            tasks.append(self._fetch_html(dpkg_url))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        docs_html = results[0] if not isinstance(results[0], BaseException) else ""
        pkg_html = (results[1] if len(results) > 1 and
                    not isinstance(results[1], BaseException) else "")

        errors: list[str] = []

        # ── Package metadata from Debian packages page ─────────────────
        metadata: dict[str, Any] = {}
        if pkg_html:
            soup = self._parse_soup(pkg_html)
            title = soup.find("title")
            if title and "Error" not in title.get_text():
                desc_el = soup.find("div", id="pdesc")
                metadata = {
                    "name": request.package,
                    "description": desc_el.get_text(strip=True)[:300] if desc_el else "",
                    "source": "debian-packages",
                }
            else:
                errors.append("Package not found in Debian repos")
        elif request.package:
            errors.append("Debian package page fetch failed")

        # ── Docs ───────────────────────────────────────────────────────
        command_syntax = ""
        key_flags: list[dict[str, str]] = []
        examples: list[str] = []

        if docs_html:
            soup = self._parse_soup(docs_html)
            command_syntax = self._extract_synopsis(soup)
            key_flags = self._extract_options_table(soup)
            examples = self._extract_examples(soup)

            # Try to extract the specific operation section
            if not command_syntax:
                for heading in soup.find_all(["h2", "h3", "dt"]):
                    text = heading.get_text(" ", strip=True).lower()
                    if request.operation in text:
                        next_el = heading.find_next(["p", "pre", "dd"])
                        if next_el:
                            command_syntax = next_el.get_text(" ", strip=True)[:500]
                        break
        else:
            errors.append("apt manpage fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            fallback = FALLBACK_TEMPLATES["apt"].get(request.operation, "")
            command_syntax = fallback.replace(
                "{package}", request.package or "{package}"
            )

        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []
        if docs_html:
            source_urls.append(docs_url)
        if pkg_html and metadata:
            source_urls.append(dpkg_url)

        chunk = DocChunk(
            tool="apt",
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
            return ("apt is not available on Windows natively. "
                    "Use WSL (Windows Subsystem for Linux) to run apt commands.")
        if request.os == "macos":
            return "apt is not available on macOS. Use brew instead."
        return ""
