"""Base adapter and shared utilities for all MCP doc adapters.

Every concrete adapter subclasses ``BaseAdapter`` and implements
``async fetch(request) -> DocChunk``.  The base class provides:

* ``_fetch_json``  — GET a URL and return parsed JSON
* ``_fetch_html``  — GET a URL and return raw HTML text
* ``_parse_soup``  — Parse HTML into a BeautifulSoup tree
* ``_extract_options_table`` — Pull flag / description pairs from an HTML table
* ``_estimate_tokens``      — Rough char-based token estimate
* ``FALLBACK_TEMPLATES``    — Hardcoded command templates per tool × operation
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import aiohttp
from bs4 import BeautifulSoup, Tag

from ..models import DocChunk, DocRequest

log = logging.getLogger("mcp.adapters")


# ── Timeout (seconds) for every individual HTTP fetch ──────────────────
FETCH_TIMEOUT = aiohttp.ClientTimeout(total=10)

# User-Agent header so servers don't reject us
_UA = "contAIner-MCP/1.0 (documentation-fetcher)"


# ═══════════════════════════════════════════════════════════════════════
# Fallback command templates — used when online docs scraping fails
# ═══════════════════════════════════════════════════════════════════════

FALLBACK_TEMPLATES: dict[str, dict[str, str]] = {
    "pip": {
        "install":   "pip install {package}{version_spec}",
        "uninstall": "pip uninstall -y {package}",
        "list":      "pip list",
        "show":      "pip show {package}",
        "freeze":    "pip freeze",
        "upgrade":   "pip install --upgrade {package}",
    },
    "npm": {
        "install":   "npm install {package}{version_spec}",
        "uninstall": "npm uninstall {package}",
        "list":      "npm list",
        "update":    "npm update {package}",
        "init":      "npm init -y",
        "run":       "npm run {script}",
    },
    "yarn": {
        "add":       "yarn add {package}{version_spec}",
        "remove":    "yarn remove {package}",
        "list":      "yarn list",
        "upgrade":   "yarn upgrade {package}",
        "init":      "yarn init -y",
        "run":       "yarn run {script}",
        "install":   "yarn add {package}{version_spec}",
    },
    "pnpm": {
        "add":       "pnpm add {package}{version_spec}",
        "remove":    "pnpm remove {package}",
        "list":      "pnpm list",
        "update":    "pnpm update {package}",
        "install":   "pnpm add {package}{version_spec}",
    },
    "apt": {
        "install":   "sudo apt-get install -y {package}",
        "remove":    "sudo apt-get remove -y {package}",
        "update":    "sudo apt-get update",
        "upgrade":   "sudo apt-get upgrade -y",
        "list":      "apt list --installed",
        "search":    "apt-cache search {package}",
        "show":      "apt-cache show {package}",
        "purge":     "sudo apt-get purge -y {package}",
        "autoremove":"sudo apt-get autoremove -y",
    },
    "brew": {
        "install":   "brew install {package}",
        "uninstall": "brew uninstall {package}",
        "update":    "brew update",
        "upgrade":   "brew upgrade {package}",
        "list":      "brew list",
        "search":    "brew search {package}",
        "info":      "brew info {package}",
    },
    "conda": {
        "install":   "conda install {channel_flag}{package}{version_spec}",
        "remove":    "conda remove {package}",
        "update":    "conda update {package}",
        "list":      "conda list",
        "create":    "conda create -n {env_name} {package}",
        "search":    "conda search {package}",
        "info":      "conda info {package}",
    },
    "docker": {
        "run":       "docker run [OPTIONS] {package} [COMMAND]",
        "pull":      "docker pull {package}{version_spec}",
        "build":     "docker build -t {tag} .",
        "push":      "docker push {package}{version_spec}",
        "ps":        "docker ps",
        "stop":      "docker stop {container}",
        "rm":        "docker rm {container}",
        "images":    "docker images",
        "exec":      "docker exec -it {container} {command}",
        "logs":      "docker logs {container}",
        "compose up":    "docker compose up -d",
        "compose down":  "docker compose down",
        "compose build": "docker compose build",
    },
    "cargo": {
        "install":   "cargo install {package}",
        "build":     "cargo build",
        "run":       "cargo run",
        "test":      "cargo test",
        "update":    "cargo update",
        "add":       "cargo add {package}{version_spec}",
        "new":       "cargo new {project_name}",
    },
    "go": {
        "get":       "go get {package}{version_spec}",
        "install":   "go install {package}{version_spec}",
        "build":     "go build ./...",
        "run":       "go run .",
        "test":      "go test ./...",
        "mod tidy":  "go mod tidy",
        "mod init":  "go mod init {module}",
    },
    "maven": {
        "add": (
            "<dependency>\n"
            "  <groupId>{group_id}</groupId>\n"
            "  <artifactId>{artifact_id}</artifactId>\n"
            "  <version>{version}</version>\n"
            "</dependency>"
        ),
        "install":  "mvn install",
        "clean":    "mvn clean",
        "package":  "mvn package",
        "test":     "mvn test",
        "compile":  "mvn compile",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# BaseAdapter
# ═══════════════════════════════════════════════════════════════════════

class BaseAdapter(ABC):
    """Abstract base for every tool-specific documentation adapter.

    Subclasses must implement ``fetch(request)``.  The base class provides
    common HTTP helpers and the parallel-fetch pattern.
    """

    # Every subclass sets this so logs / errors identify the adapter
    tool_name: str = ""
    supported_operations: list[str] = []

    # ── HTTP helpers ───────────────────────────────────────────────────

    @staticmethod
    async def _fetch_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """GET *url* and return parsed JSON.  Returns ``{}`` on any error."""
        hdrs = {"User-Agent": _UA, "Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        try:
            async with aiohttp.ClientSession(timeout=FETCH_TIMEOUT) as session:
                async with session.get(url, headers=hdrs, allow_redirects=True) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)
                    log.warning("JSON fetch %s returned status %d", url, resp.status)
                    return {}
        except Exception as exc:
            log.warning("JSON fetch %s failed: %s", url, exc)
            return {}

    @staticmethod
    async def _fetch_html(url: str) -> str:
        """GET *url* and return raw HTML text.  Returns ``""`` on error."""
        try:
            async with aiohttp.ClientSession(timeout=FETCH_TIMEOUT) as session:
                async with session.get(url, headers={"User-Agent": _UA}, allow_redirects=True) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    log.warning("HTML fetch %s returned status %d", url, resp.status)
                    return ""
        except Exception as exc:
            log.warning("HTML fetch %s failed: %s", url, exc)
            return ""

    @staticmethod
    def _parse_soup(html: str) -> BeautifulSoup:
        """Parse HTML string into a BeautifulSoup tree (lxml parser)."""
        return BeautifulSoup(html, "lxml")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace: collapse runs of spaces/newlines."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_options_table(soup: BeautifulSoup,
                               limit: int = 15) -> list[dict[str, str]]:
        """Extract flag / description pairs from the first suitable HTML table.

        Looks for ``<table>`` elements whose rows contain a code-styled
        flag cell.  Falls back to ``<dt>/<dd>`` definition lists which
        are common in man-page–style docs.
        """
        flags: list[dict[str, str]] = []
        _clean = BaseAdapter._clean_text

        # Strategy 1: <table> rows
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    flag_text = _clean(cells[0].get_text(" ", strip=True))
                    desc_text = _clean(cells[1].get_text(" ", strip=True))
                    if flag_text.startswith("-"):
                        flags.append({"flag": flag_text, "description": desc_text})
                        if len(flags) >= limit:
                            return flags

        # Strategy 2: <dl> definition lists
        if not flags:
            for dt in soup.find_all("dt"):
                flag_text = _clean(dt.get_text(" ", strip=True))
                dd = dt.find_next_sibling("dd")
                if dd and flag_text.startswith("-"):
                    desc_text = _clean(dd.get_text(" ", strip=True))[:200]
                    flags.append({"flag": flag_text, "description": desc_text})
                    if len(flags) >= limit:
                        return flags

        # Strategy 3: code blocks containing flags
        if not flags:
            for code in soup.find_all("code"):
                text = _clean(code.get_text(" ", strip=True))
                if text.startswith("-") and len(text) < 60:
                    # Try to get surrounding description
                    parent = code.find_parent(["li", "p", "td", "dd"])
                    desc = ""
                    if parent:
                        desc = _clean(parent.get_text(" ", strip=True)).replace(text, "").strip()[:200]
                    flags.append({"flag": text, "description": desc})
                    if len(flags) >= limit:
                        return flags

        return flags

    @staticmethod
    def _extract_synopsis(soup: BeautifulSoup) -> str:
        """Try to find a synopsis / usage line from the page."""
        _clean = BaseAdapter._clean_text

        # Look for common patterns
        for selector in ["#synopsis", "#usage", ".synopsis", ".usage",
                         "h2:contains('Synopsis')", "h2:contains('Usage')"]:
            try:
                el = soup.select_one(selector)
                if el:
                    # Get the next sibling or code block
                    code = el.find_next(["pre", "code"])
                    if code:
                        return _clean(code.get_text(" ", strip=True))[:500]
            except Exception:
                continue

        # Fallback — look for <pre> blocks near the top
        for pre in soup.find_all("pre", limit=3):
            text = _clean(pre.get_text(" ", strip=True))
            if len(text) < 500:
                return text

        return ""

    @staticmethod
    def _extract_examples(soup: BeautifulSoup, limit: int = 5) -> list[str]:
        """Pull example command lines from the page."""
        _clean = BaseAdapter._clean_text
        examples: list[str] = []

        # Look for sections titled "Examples"
        for heading in soup.find_all(re.compile(r"^h[2-4]$")):
            if "example" in heading.get_text(" ", strip=True).lower():
                sibling = heading.find_next_sibling()
                while sibling and sibling.name not in ("h1", "h2", "h3", "h4"):
                    if isinstance(sibling, Tag):
                        for pre in ([sibling] if sibling.name == "pre"
                                    else sibling.find_all("pre")):
                            text = _clean(pre.get_text(" ", strip=True))
                            if text and len(text) < 300:
                                examples.append(text)
                                if len(examples) >= limit:
                                    return examples
                    sibling = sibling.find_next_sibling()

        # Fallback: any <pre> / <code> that looks like a shell command
        if not examples:
            for code in soup.find_all(["pre", "code"], limit=20):
                text = _clean(code.get_text(" ", strip=True))
                if text and len(text) < 200 and any(
                    text.startswith(p) for p in ("$", "pip ", "npm ", "apt ",
                                                  "brew ", "conda ", "docker ",
                                                  "cargo ", "go ", "mvn ",
                                                  "yarn ", "pnpm ", "sudo ")
                ):
                    examples.append(text.lstrip("$ ").strip())
                    if len(examples) >= limit:
                        return examples

        return examples

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (≈ 4 chars per token)."""
        return max(1, len(text) // 4)

    def _get_fallback(self, request: DocRequest) -> DocChunk:
        """Return a DocChunk populated from the fallback template."""
        tool_templates = FALLBACK_TEMPLATES.get(request.tool.lower(), {})
        syntax = tool_templates.get(request.operation.lower(), "")

        # Simple placeholder substitution
        version_spec = f"=={request.version}" if request.version else ""
        syntax = (syntax
                  .replace("{package}", request.package or "{package}")
                  .replace("{version_spec}", version_spec)
                  .replace("{version}", request.version or "LATEST"))

        chunk = DocChunk(
            tool=request.tool,
            operation=request.operation,
            command_syntax=syntax,
            examples=[syntax] if syntax else [],
            error="Docs fetch failed — returning fallback template",
            os_specific_notes="",
        )
        chunk.estimate_tokens()
        return chunk

    def _version_spec(self, request: DocRequest, sep: str = "==") -> str:
        """Build a version specifier string like ``==1.26.0`` or ``@1.0.0``."""
        if request.version:
            return f"{sep}{request.version}"
        return ""

    # ── Abstract contract ──────────────────────────────────────────────

    @abstractmethod
    async def fetch(self, request: DocRequest) -> DocChunk:
        """Fetch documentation for *request* and return a DocChunk.

        Implementations must:
        1. Run registry + docs fetches in parallel via ``asyncio.gather``
        2. Gracefully degrade if one or both fail
        3. Fall back to ``_get_fallback`` when everything fails
        """
        ...
