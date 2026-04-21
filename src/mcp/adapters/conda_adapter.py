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

        # Start docs fetch immediately; try registry channels in sequence
        docs_task = asyncio.create_task(self._fetch_html(docs_url))
        registry_data = await self._try_channels(request.package, channels)
        docs_html_result = await docs_task

        docs_html = docs_html_result if not isinstance(docs_html_result, BaseException) else ""
        errors: list[str] = []

        # ── Registry metadata ──────────────────────────────────────────
        metadata: dict[str, Any] = {}
        if registry_data:
            metadata = {
                "name": registry_data.get("name", request.package),
                "summary": registry_data.get("summary", ""),
                "description": (registry_data.get("description") or "")[:300],
                "home": registry_data.get("home", ""),
                "license": registry_data.get("license", ""),
                "dev_url": registry_data.get("dev_url", ""),
                "latest_version": registry_data.get("latest_version", ""),
                "conda_platforms": registry_data.get("conda_platforms", []),
                "versions": (registry_data.get("versions") or [])[:10],
            }
        elif request.package:
            errors.append("Package not found in any Anaconda channel")

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
        if registry_data:
            source_urls.append(
                self._REGISTRY_URL.format(
                    channel=registry_data.get("_channel", "conda-forge"),
                    package=request.package,
                )
            )
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="conda",
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

    async def _try_channels(self, package: str, channels: list[str]) -> dict:
        """Try each channel until one returns data."""
        for channel in channels:
            url = self._REGISTRY_URL.format(channel=channel, package=package)
            data = await self._fetch_json(url)
            if data:
                data["_channel"] = channel
                return data
        return {}

    @staticmethod
    def _os_notes(request: DocRequest) -> str:
        if request.os == "windows":
            return ("On Windows, use Anaconda Prompt or add conda to PATH. "
                    "Alternatively use miniconda.")
        return ""
