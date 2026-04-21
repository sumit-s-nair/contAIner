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

        registry_data, docs_html = await asyncio.gather(
            self._fetch_json(search_url),
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
        dep_xml = ""
        group_id_found = ""
        artifact_id_found = ""
        version_found = ""

        if registry_data:
            response = registry_data.get("response", {})
            docs = response.get("docs", [])
            if docs:
                doc = docs[0]
                group_id_found = doc.get("g", "")
                artifact_id_found = doc.get("a", "")
                version_found = request.version or doc.get("latestVersion", "")

                metadata = {
                    "group_id": group_id_found,
                    "artifact_id": artifact_id_found,
                    "latest_version": doc.get("latestVersion", ""),
                    "packaging": doc.get("p", "jar"),
                    "timestamp": doc.get("timestamp", 0),
                    "version_count": doc.get("versionCount", 0),
                    "repository_id": doc.get("repositoryId", ""),
                }

                # Build the <dependency> XML
                dep_xml = (
                    "<dependency>\n"
                    f"  <groupId>{group_id_found}</groupId>\n"
                    f"  <artifactId>{artifact_id_found}</artifactId>\n"
                    f"  <version>{version_found}</version>\n"
                    "</dependency>"
                )
            else:
                errors.append("No results found on Maven Central")
        else:
            errors.append("Maven Central search failed")

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
        if request.operation == "add" and dep_xml:
            command_syntax = dep_xml
        elif request.operation == "add":
            # Fallback XML template
            command_syntax = FALLBACK_TEMPLATES["maven"]["add"].format(
                group_id=group_id_found or "{groupId}",
                artifact_id=artifact_id_found or "{artifactId}",
                version=version_found or "{version}",
            )
        else:
            fallback = FALLBACK_TEMPLATES["maven"].get(request.operation, "")
            command_syntax = fallback

        if not examples:
            if dep_xml:
                examples.append(dep_xml)
            if request.operation != "add":
                examples.append(command_syntax)

        source_urls = []
        if registry_data and metadata:
            source_urls.append(search_url)
            if group_id_found and artifact_id_found:
                source_urls.append(
                    f"https://search.maven.org/artifact/"
                    f"{group_id_found}/{artifact_id_found}"
                )
        if docs_html:
            source_urls.append(docs_url)

        chunk = DocChunk(
            tool="maven",
            operation=request.operation,
            package_metadata=metadata,
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
