"""Docker / Docker Compose adapter.

Registry : https://hub.docker.com/v2/repositories/library/{package}
CLI docs : https://docs.docker.com/reference/cli/docker/{operation}/
Compose  : https://docs.docker.com/compose/reference/
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseAdapter, FALLBACK_TEMPLATES
from ..models import DocChunk, DocRequest


class DockerAdapter(BaseAdapter):
    tool_name = "docker"
    supported_operations = [
        "run", "pull", "build", "push", "ps", "stop", "rm",
        "images", "exec", "logs",
        "compose up", "compose down", "compose build",
    ]

    _REGISTRY_URL = "https://hub.docker.com/v2/repositories/library/{package}"
    _CLI_DOCS_URL = "https://docs.docker.com/reference/cli/docker/{operation}/"
    _COMPOSE_DOCS_URL = "https://docs.docker.com/compose/reference/"

    def _is_compose(self, operation: str) -> bool:
        """Check if this is a docker compose operation."""
        return operation.startswith("compose")

    def _registry_url(self, package: str) -> str:
        """Build the Docker Hub registry URL.

        Handles both official images (library/) and user images (user/repo).
        """
        if not package:
            return ""
        if "/" in package:
            return f"https://hub.docker.com/v2/repositories/{package}"
        return self._REGISTRY_URL.format(package=package)

    def _docs_url(self, operation: str) -> str:
        """Return the correct docs URL for the operation."""
        if self._is_compose(operation):
            return self._COMPOSE_DOCS_URL
        # Normalize: "exec" → "container/exec", etc.
        op = operation.replace(" ", "/")
        return self._CLI_DOCS_URL.format(operation=op)

    async def fetch(self, request: DocRequest) -> DocChunk:
        reg_url = self._registry_url(request.package)
        docs_url = self._docs_url(request.operation)

        tasks = []
        if reg_url:
            tasks.append(self._fetch_json(reg_url))
        else:
            tasks.append(asyncio.coroutine(lambda: {})())  # noqa — placeholder

        tasks.append(self._fetch_html(docs_url))

        # Use gather directly with URL-based tasks
        if reg_url:
            registry_data, docs_html = await asyncio.gather(
                self._fetch_json(reg_url),
                self._fetch_html(docs_url),
                return_exceptions=True,
            )
        else:
            registry_data = {}
            docs_html = await self._fetch_html(docs_url)

        if isinstance(registry_data, BaseException):
            registry_data = {}
        if isinstance(docs_html, BaseException):
            docs_html = ""

        errors: list[str] = []

        # ── Registry metadata ──────────────────────────────────────────
        metadata: dict[str, Any] = {}
        if registry_data:
            metadata = {
                "name": registry_data.get("name", request.package),
                "namespace": registry_data.get("namespace", "library"),
                "description": registry_data.get("description", ""),
                "star_count": registry_data.get("star_count", 0),
                "pull_count": registry_data.get("pull_count", 0),
                "is_official": registry_data.get("is_official", False),
                "last_updated": registry_data.get("last_updated", ""),
            }
        elif request.package:
            errors.append("Docker Hub registry fetch failed")

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
            errors.append("Docker docs fetch failed")

        # ── Fallback ───────────────────────────────────────────────────
        if not command_syntax:
            version_spec = f":{request.version}" if request.version else ""
            fallback = FALLBACK_TEMPLATES["docker"].get(request.operation, "")
            command_syntax = (fallback
                              .replace("{package}", request.package or "{image}")
                              .replace("{version_spec}", version_spec)
                              .replace("{tag}", request.package or "{tag}")
                              .replace("{container}", "{container}")
                              .replace("{command}", "{command}"))

        if not examples and command_syntax:
            examples = [command_syntax]

        source_urls = []
        if registry_data and reg_url:
            source_urls.append(reg_url)
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="docker" if not self._is_compose(request.operation) else "docker-compose",
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
            return ("On Windows, Docker Desktop must be installed. "
                    "WSL 2 backend is recommended for best performance.")
        if request.os == "linux":
            return ("Ensure the current user is in the 'docker' group, "
                    "or prepend commands with `sudo`.")
        return ""
