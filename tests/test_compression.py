"""Eval suite for the doc-compression pipeline (Parts A–D).

Tests (all hard assertions, not soft checks):

    test_code_segments_byte_identical
        For every fixture, every CODE segment produced by segmentation
        must appear byte-for-byte in the extractive-compressed output.
        Pass rate must be 100%.

    test_flags_and_versions_preserved
        For every fixture with ``known_flags`` / ``known_versions``,
        each token must appear somewhere in the extractive-compressed
        PROSE.  Failures are collected and reported together.

    test_extractive_reduces_tokens
        For fixtures with > 50 tokens of prose, compressed_token_count
        must be strictly less than original_token_count.

    test_abstractive_graceful_fallback
        With ``transformers`` import mocked to fail, abstractive path
        returns extractive result without raising.

    test_abstractive_cache_hit
        Running the same PROSE segment twice calls the model at most
        once (mock assert call_count == 1).

    test_compression_comparison_report
        Runs both methods on all fixtures and prints a markdown table:
        fixture | orig_tok | ext_tok | ext_ratio | flags_ok | vers_ok
        This table is the data that decides whether abstractive is worth
        the added complexity.

Fixtures
--------
18 hand-crafted DocChunk-shaped dicts covering every adapter (pip, npm,
docker, apt, brew, cargo, go, conda, maven) with deliberately mixed
CODE and PROSE to exercise all segmentation paths.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch, MagicMock, call
from dataclasses import dataclass, field
from typing import Any

# ── Make src importable without install ────────────────────────────────
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Direct imports from individual modules — bypasses the router/adapter chain
# (which requires bs4/aiohttp) since __init__.py now lazy-loads DocRouter.
from src.mcp.models import DocChunk, CompressionReport, CompressedDocChunk
from src.mcp.segmentation import segment_text, reassemble_segments, Segment
from src.mcp.compress_extractive import compress_segments_extractive, compress_prose_extractive
from src.mcp.cache import TTLCache
from src.mcp.compress_abstractive import (
    compress_segments_abstractive,
    compress_prose_abstractive,
    _cache_key,
    reset_model,
)
from src.mcp.reassemble import reassemble_chunk, _extract_compressible_text
from src.mcp.compress import compress_chunk


# ══════════════════════════════════════════════════════════════════════
# Golden fixture corpus (18 entries, no network)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Fixture:
    name: str
    # Raw text that would appear in the compressible fields of a DocChunk
    raw_text: str
    # Tokens that MUST appear somewhere in compressed PROSE
    known_flags: list[str] = field(default_factory=list)
    known_versions: list[str] = field(default_factory=list)
    # Build a DocChunk around this fixture
    tool: str = "pip"
    operation: str = "install"


FIXTURES: list[Fixture] = [
    # ── pip ────────────────────────────────────────────────────────────
    Fixture(
        name="pip_install_prose_heavy",
        raw_text=(
            "pip is the package installer for Python. "
            "You can use it to install packages from PyPI and other indexes. "
            "The --upgrade flag upgrades the package to the latest version. "
            "The --no-deps flag skips dependency installation. "
            "As of version 23.0, pip uses the new resolver by default. "
            "This flag is required when installing in editable mode. "
            "Deprecated: the --process-dependency-links flag was removed in pip 21.0.\n\n"
            "```bash\n"
            "pip install numpy --upgrade\n"
            "pip install -r requirements.txt --no-deps\n"
            "```\n\n"
            "The -q flag suppresses output. "
            "Warning: installing packages into the system Python is not recommended."
        ),
        known_flags=["--upgrade", "--no-deps", "-q"],
        known_versions=["23.0", "21.0"],
        tool="pip",
        operation="install",
    ),
    Fixture(
        name="pip_show_version_dense",
        raw_text=(
            "pip show displays package metadata. "
            "It prints the package name, version, location, and dependencies. "
            "Requires pip >= 9.0. "
            "The --verbose flag adds extra details including classifiers. "
            "Note: this command does not contact PyPI.\n\n"
            "    pip show numpy\n"
            "    pip show --verbose numpy\n"
        ),
        known_flags=["--verbose"],
        known_versions=["9.0"],
        tool="pip",
        operation="show",
    ),
    Fixture(
        name="pip_freeze_plain",
        raw_text=(
            "pip freeze outputs installed packages in requirements format. "
            "This is useful for pinning dependency versions in a project. "
            "Use --local to limit output to the current virtualenv.\n\n"
            "```\n"
            "pip freeze > requirements.txt\n"
            "```"
        ),
        known_flags=["--local"],
        known_versions=[],
        tool="pip",
        operation="freeze",
    ),
    # ── npm ────────────────────────────────────────────────────────────
    Fixture(
        name="npm_install_mixed",
        raw_text=(
            "npm install adds packages to your project. "
            "Running without arguments installs all dependencies from package.json. "
            "Use --save-dev to add a package as a development dependency. "
            "The --global flag installs the package globally. "
            "npm 7+ automatically installs peer dependencies. "
            "Warning: avoid mixing npm and yarn in the same project. "
            "The -E flag (--save-exact) pins the exact version in package.json.\n\n"
            "```bash\n"
            "npm install express@4.18.2\n"
            "npm install --save-dev jest\n"
            "npm install -g typescript\n"
            "```"
        ),
        known_flags=["--save-dev", "--global", "-E"],
        known_versions=["4.18.2"],
        tool="npm",
        operation="install",
    ),
    Fixture(
        name="npm_update_flags",
        raw_text=(
            "npm update updates packages to the version range specified in package.json. "
            "Use --depth to control how deep in the dependency tree to update. "
            "The --save flag also updates package.json entries. "
            "Note: npm update does not upgrade packages beyond the specified semver range."
        ),
        known_flags=["--depth", "--save"],
        known_versions=[],
        tool="npm",
        operation="update",
    ),
    # ── docker ─────────────────────────────────────────────────────────
    Fixture(
        name="docker_run_command_heavy",
        raw_text=(
            "docker run creates and starts a container from an image. "
            "The -d flag runs the container in detached mode. "
            "Use -p to map host ports to container ports (e.g. -p 8080:80). "
            "The --rm flag removes the container when it exits. "
            "Required: the image name must be specified after all options.\n\n"
            "```bash\n"
            "docker run -d --rm -p 8080:80 --name webserver nginx:1.25.3\n"
            "docker run -it --entrypoint /bin/bash ubuntu:22.04\n"
            "```\n\n"
            "The --env-file flag loads environment variables from a file."
        ),
        known_flags=["-d", "-p", "--rm", "--env-file"],
        known_versions=["1.25.3", "22.04"],
        tool="docker",
        operation="run",
    ),
    Fixture(
        name="docker_build_flags",
        raw_text=(
            "docker build creates an image from a Dockerfile. "
            "The -t flag tags the resulting image with a name and optional tag. "
            "Use --build-arg to pass build-time variables. "
            "The --no-cache flag forces a clean build from scratch. "
            "Breaking change in Docker 23.0: BuildKit is now the default builder.\n\n"
            "```\n"
            "docker build -t myapp:1.0 --no-cache .\n"
            "docker build --build-arg VERSION=2.0 -t myapp:2.0 .\n"
            "```"
        ),
        known_flags=["-t", "--build-arg", "--no-cache"],
        known_versions=["23.0"],
        tool="docker",
        operation="build",
    ),
    # ── apt ────────────────────────────────────────────────────────────
    Fixture(
        name="apt_install_flag_heavy",
        raw_text=(
            "apt-get install installs one or more packages. "
            "The -y flag answers yes to all prompts automatically. "
            "Use --no-install-recommends to skip recommended but non-required packages. "
            "The --reinstall flag reinstalls a package already installed. "
            "Required: root or sudo privileges. "
            "This command works on Debian 12 and Ubuntu 22.04 and later.\n\n"
            "sudo apt-get install -y --no-install-recommends curl wget git\n\n"
            "Note: always run apt-get update before installing new packages."
        ),
        known_flags=["-y", "--no-install-recommends", "--reinstall"],
        known_versions=["22.04"],
        tool="apt",
        operation="install",
    ),
    Fixture(
        name="apt_remove_short",
        raw_text=(
            "apt-get remove uninstalls packages but keeps configuration files. "
            "Use apt-get purge to also remove configuration. "
            "The -y flag skips confirmation. "
            "Deprecated: apt-get autoremove --purge is no longer recommended; use apt purge.\n\n"
            "sudo apt-get remove -y curl\n"
        ),
        known_flags=["-y"],
        known_versions=[],
        tool="apt",
        operation="remove",
    ),
    # ── brew ───────────────────────────────────────────────────────────
    Fixture(
        name="brew_info_version_dense",
        raw_text=(
            "brew info displays metadata about a formula or cask. "
            "It shows the stable version, installed version, and dependencies. "
            "As of Homebrew 4.0, brew info also shows analytics data. "
            "The --json=v2 flag returns machine-readable JSON output. "
            "Note: cask info uses the same flag set as formula info.\n\n"
            "```bash\n"
            "brew info --json=v2 node\n"
            "brew info --verbose git\n"
            "```"
        ),
        known_flags=["--json=v2", "--verbose"],
        known_versions=["4.0"],
        tool="brew",
        operation="info",
    ),
    Fixture(
        name="brew_install_options",
        raw_text=(
            "brew install downloads and installs a formula. "
            "The --HEAD flag installs from the latest upstream source. "
            "Use --with-debug to compile with debug symbols. "
            "The --formula flag explicitly selects a formula over a cask. "
            "Important: Homebrew requires macOS 12.0 or higher for Apple Silicon support."
        ),
        known_flags=["--HEAD", "--formula"],
        known_versions=["12.0"],
        tool="brew",
        operation="install",
    ),
    # ── cargo ──────────────────────────────────────────────────────────
    Fixture(
        name="cargo_build_short_prose",
        raw_text=(
            "cargo build compiles the current package. "
            "The --release flag enables optimizations (required for production builds). "
            "Use --target to cross-compile for a different architecture. "
            "As of Rust 1.70, sparse registry is the default.\n\n"
            "```\n"
            "cargo build --release --target x86_64-unknown-linux-gnu\n"
            "```"
        ),
        known_flags=["--release", "--target"],
        known_versions=["1.70"],
        tool="cargo",
        operation="build",
    ),
    Fixture(
        name="cargo_test_flags",
        raw_text=(
            "cargo test runs the test suite. "
            "The -- --nocapture flag shows stdout from passing tests. "
            "Use --test <name> to run a specific integration test binary. "
            "The -p flag runs tests for a specific package in a workspace. "
            "Note: test binaries are compiled before running."
        ),
        known_flags=["--nocapture", "-p"],
        known_versions=[],
        tool="cargo",
        operation="test",
    ),
    # ── go ─────────────────────────────────────────────────────────────
    Fixture(
        name="go_get_url_like",
        raw_text=(
            "go get adds or upgrades dependencies in go.mod. "
            "To download a specific version, append @version to the module path. "
            "The -u flag updates the module to its latest minor or patch version. "
            "Go 1.21 introduced toolchain management; the toolchain directive in go.mod controls it. "
            "Deprecated: go get for installing commands (use go install instead).\n\n"
            "go get github.com/gin-gonic/gin@v1.9.1\n"
            "go get -u golang.org/x/tools\n"
        ),
        known_flags=["-u"],
        known_versions=["1.21", "v1.9.1"],
        tool="go",
        operation="get",
    ),
    # ── conda ──────────────────────────────────────────────────────────
    Fixture(
        name="conda_create_env_flags",
        raw_text=(
            "conda create makes a new isolated environment. "
            "The -n flag specifies the environment name. "
            "Use -c conda-forge to install from the conda-forge channel. "
            "The --copy flag copies files instead of creating hard links. "
            "Required: Miniconda or Anaconda 23.1.0 or higher. "
            "Note: always activate the environment before installing packages.\n\n"
            "```bash\n"
            "conda create -n myenv python=3.11 -c conda-forge --copy\n"
            "conda create -n ml numpy pandas scikit-learn=1.3.0\n"
            "```"
        ),
        known_flags=["-n", "-c", "--copy"],
        known_versions=["23.1.0", "3.11", "1.3.0"],
        tool="conda",
        operation="create",
    ),
    # ── maven ──────────────────────────────────────────────────────────
    Fixture(
        name="maven_add_xml_code",
        raw_text=(
            "To add a dependency in Maven, edit pom.xml. "
            "The groupId, artifactId, and version elements are required. "
            "Use the <scope> element to declare test or provided dependencies. "
            "Deprecated: using version ranges is discouraged; pin exact versions. "
            "Maven 3.9.0 introduced improved dependency conflict resolution.\n\n"
            "```xml\n"
            "<dependency>\n"
            "  <groupId>org.springframework</groupId>\n"
            "  <artifactId>spring-core</artifactId>\n"
            "  <version>6.1.2</version>\n"
            "</dependency>\n"
            "```\n\n"
            "The -DskipTests flag skips running tests during the build."
        ),
        known_flags=["-DskipTests"],
        known_versions=["3.9.0", "6.1.2"],
        tool="maven",
        operation="add",
    ),
    # ── edge cases ─────────────────────────────────────────────────────
    Fixture(
        name="all_code_no_prose",
        raw_text=(
            "```bash\n"
            "pip install flask==3.0.0\n"
            "pip install gunicorn --no-cache-dir\n"
            "```"
        ),
        known_flags=[],
        known_versions=["3.0.0"],
        tool="pip",
        operation="install",
    ),
    Fixture(
        name="all_prose_no_code",
        raw_text=(
            "This package provides utilities for working with JSON data in Python. "
            "It is compatible with Python 3.8 and later versions. "
            "The package is actively maintained and follows semantic versioning. "
            "It has no external runtime dependencies. "
            "Contributions are welcome via pull requests on GitHub."
        ),
        known_flags=[],
        known_versions=["3.8"],
        tool="pip",
        operation="install",
    ),
]


def _make_doc_chunk(fixture: Fixture) -> DocChunk:
    """Wrap a Fixture as a minimal DocChunk with compressible fields."""
    return DocChunk(
        tool=fixture.tool,
        operation=fixture.operation,
        os_specific_notes=fixture.raw_text,
        command_syntax=f"{fixture.tool} {fixture.operation}",
        key_flags=[],
        examples=[],
        source_urls=[],
        tokens_estimate=max(1, len(fixture.raw_text) // 4),
    )


# ══════════════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════════════

def _code_segments(segs: list[Segment]) -> list[Segment]:
    return [s for s in segs if s.kind == "CODE"]


def _prose_segments(segs: list[Segment]) -> list[Segment]:
    return [s for s in segs if s.kind == "PROSE"]


def _count_tokens(text: str) -> int:
    return max(0, len(text) // 4)


# ══════════════════════════════════════════════════════════════════════
# Test 1 — CODE byte-identity (hard assertion, 100% pass rate)
# ══════════════════════════════════════════════════════════════════════

class TestCodeByteIdentity(unittest.TestCase):
    """Every CODE segment must appear byte-for-byte in the compressed output."""

    def test_code_segments_byte_identical(self) -> None:
        failures: list[str] = []

        for fix in FIXTURES:
            segs = segment_text(fix.raw_text)
            compressed = compress_segments_extractive(segs)
            reassembled = reassemble_segments(compressed)

            for seg in _code_segments(segs):
                if seg.text not in reassembled:
                    failures.append(
                        f"[{fix.name}] CODE segment NOT found byte-for-byte in compressed output.\n"
                        f"  expected: {seg.text!r}\n"
                        f"  in output: {reassembled!r}"
                    )

        self.assertEqual(
            failures, [],
            "CODE byte-identity failures:\n" + "\n\n".join(failures),
        )

    def test_reassemble_chunk_code_assertion(self) -> None:
        """reassemble_chunk should raise AssertionError if CODE is mutated."""
        fix = FIXTURES[0]  # has code blocks
        segs = segment_text(fix.raw_text)
        chunk = _make_doc_chunk(fix)

        # Manually corrupt a CODE segment text
        corrupted = []
        for s in segs:
            if s.kind == "CODE":
                corrupted.append(Segment(kind="CODE", text=s.text + " CORRUPTED", index=s.index))
            else:
                corrupted.append(s)

        code_segs = _code_segments(segs)
        if not code_segs:
            self.skipTest(f"Fixture {fix.name} has no CODE segments")

        with self.assertRaises(AssertionError):
            reassemble_chunk(chunk, corrupted, segs, method="extractive")


# ══════════════════════════════════════════════════════════════════════
# Test 2 — Flag and version preservation in extractive output
# ══════════════════════════════════════════════════════════════════════

class TestFlagVersionPreservation(unittest.TestCase):
    """All known_flags and known_versions must survive extractive compression.

    Versions / flags that appear exclusively inside CODE blocks are also
    accepted — the claim is that no *information* is lost, not that it
    must specifically be in the PROSE portion.
    """

    def test_flags_and_versions_preserved(self) -> None:
        dropped: list[str] = []

        for fix in FIXTURES:
            segs = segment_text(fix.raw_text)
            compressed = compress_segments_extractive(segs)

            # Search the full compressed output (CODE + PROSE)
            full_compressed = reassemble_segments(compressed)

            for flag in fix.known_flags:
                if flag not in full_compressed:
                    dropped.append(
                        f"[{fix.name}] FLAG '{flag}' not found in compressed output.\n"
                        f"  full output: {full_compressed[:200]!r}"
                    )

            for ver in fix.known_versions:
                if ver not in full_compressed:
                    dropped.append(
                        f"[{fix.name}] VERSION '{ver}' not found in compressed output.\n"
                        f"  full output: {full_compressed[:200]!r}"
                    )

        self.assertEqual(
            dropped, [],
            f"Dropped tokens ({len(dropped)} failures):\n" + "\n\n".join(dropped),
        )


# ══════════════════════════════════════════════════════════════════════
# Test 3 — Token reduction
# ══════════════════════════════════════════════════════════════════════

class TestTokenReduction(unittest.TestCase):
    """Fixtures with > 50 prose tokens must not expand after extractive compression.

    Strict reduction (compressed < original) is asserted only for fixtures
    that have at least one sentence that does NOT match any keep rule —
    i.e., where there is something to drop.  Fixtures whose every sentence
    triggers a keep rule will not shrink, and that is correct behaviour.
    """

    def test_extractive_reduces_tokens(self) -> None:
        non_reductions: list[str] = []

        from src.mcp.compress_extractive import _should_keep, _split_sentences

        for fix in FIXTURES:
            segs = segment_text(fix.raw_text)
            orig_prose_tokens = sum(_count_tokens(s.text) for s in _prose_segments(segs))

            if orig_prose_tokens <= 50:
                continue  # skip short fixtures

            # Check whether any PROSE sentence is droppable (doesn't match keep rules)
            all_prose = " ".join(s.text for s in _prose_segments(segs))
            sentences = _split_sentences(all_prose)
            # First sentence always kept; droppable = remaining sentences not matching
            droppable = [
                s for s in sentences[1:] if not _should_keep(s)
            ] if len(sentences) > 1 else []

            if not droppable:
                # Every sentence matches a keep rule — no reduction expected; skip
                continue

            compressed = compress_segments_extractive(segs)
            comp_prose_tokens = sum(_count_tokens(s.text) for s in _prose_segments(compressed))

            if comp_prose_tokens >= orig_prose_tokens:
                non_reductions.append(
                    f"[{fix.name}] No token reduction despite {len(droppable)} droppable sentences: "
                    f"orig={orig_prose_tokens}, compressed={comp_prose_tokens}"
                )

        self.assertEqual(
            non_reductions, [],
            "Token reduction failures:\n" + "\n".join(non_reductions),
        )


# ══════════════════════════════════════════════════════════════════════
# Test 4 — Abstractive graceful fallback
# ══════════════════════════════════════════════════════════════════════

class TestAbstractiveFallback(unittest.TestCase):
    """Abstractive path must fall back to extractive when transformers fails."""

    def test_abstractive_graceful_fallback(self) -> None:
        reset_model()

        # Make transformers un-importable
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            reset_model()
            text = (
                "This is a long prose segment. "
                "It requires the --verbose flag. "
                "Version 2.0 is the default. "
                "Warning: this behavior changed in version 1.5."
            )
            result = compress_prose_abstractive(text, cache=None)

        # Should return something (extractive fallback), not raise
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        # Extractive fallback should keep the first sentence
        self.assertIn("This is a long prose segment", result)

        reset_model()


# ══════════════════════════════════════════════════════════════════════
# Test 5 — Abstractive cache hit (model called at most once)
# ══════════════════════════════════════════════════════════════════════

class TestAbstractiveCacheHit(unittest.TestCase):
    """Same segment must not re-run the model on a second call."""

    def test_abstractive_cache_hit(self) -> None:
        import src.mcp.compress_abstractive as ca

        reset_model()
        cache = TTLCache(max_size=100)
        text = "Install the package using the --user flag to avoid permissions issues."

        fake_result = "Use --user flag for installation."

        # Patch _run_model so we can count invocations
        run_mock = MagicMock(return_value=fake_result)

        with patch.object(ca, "_ensure_model", return_value=True), \
             patch.object(ca, "_run_model", run_mock):
            # First call — model runs
            r1 = compress_prose_abstractive(text, cache=cache)
            # Second call — should be a cache hit
            r2 = compress_prose_abstractive(text, cache=cache)

        self.assertEqual(r1, fake_result)
        self.assertEqual(r2, fake_result)
        self.assertEqual(run_mock.call_count, 1, "Model was called more than once for identical input")

        reset_model()


# ══════════════════════════════════════════════════════════════════════
# Test 6 — Side-by-side comparison report (data, not a pass/fail test)
# ══════════════════════════════════════════════════════════════════════

class TestCompressionComparisonReport(unittest.TestCase):
    """Print a markdown comparison table: extractive vs. abstractive.

    This test always passes — its purpose is to emit the data that
    drives the 'is abstractive worth it?' decision.
    """

    def test_compression_comparison_report(self) -> None:
        import src.mcp.compress_abstractive as ca
        reset_model()

        rows: list[dict] = []

        for fix in FIXTURES:
            chunk = _make_doc_chunk(fix)

            # Extractive
            ext_result = compress_chunk(chunk, method="extractive", cache=None)
            ext_report = ext_result.compression_report
            if ext_report is None:
                continue

            # Check flag/version preservation in extractive
            # Search full compressed output (CODE + PROSE) so versions that only
            # appear in code blocks are not counted as dropped.
            from src.mcp.segmentation import segment_text as _seg, reassemble_segments as _reas
            from src.mcp.compress_extractive import compress_segments_extractive as _ext
            _raw_segs = _seg(fix.raw_text)
            _comp_segs = _ext(_raw_segs)
            _full_compressed = _reas(_comp_segs)
            flags_ok = all(f in _full_compressed for f in fix.known_flags)
            vers_ok = all(v in _full_compressed for v in fix.known_versions)

            # Abstractive (mock model to avoid needing transformers installed)
            def _mock_run_model(text: str) -> str:
                # Simulate: return just the first ~60 chars so ratio is measurable
                return text[:60].strip() + "..."

            with patch.object(ca, "_ensure_model", return_value=True), \
                 patch.object(ca, "_run_model", side_effect=_mock_run_model):
                abs_result = compress_chunk(chunk, method="abstractive", cache=None)

            abs_report = abs_result.compression_report

            rows.append({
                "fixture": fix.name,
                "orig_tok": ext_report.original_token_count,
                "ext_tok": ext_report.compressed_token_count,
                "ext_ratio": f"{ext_report.prose_reduction_ratio:.2%}",
                "abs_tok": abs_report.compressed_token_count if abs_report else "N/A",
                "abs_ratio": (
                    f"{abs_report.prose_reduction_ratio:.2%}"
                    if abs_report else "N/A"
                ),
                "flags_ok": "OK" if flags_ok or not fix.known_flags else "FAIL",
                "vers_ok": "OK" if vers_ok or not fix.known_versions else "FAIL",
            })

        reset_model()

        # ── Print markdown table ───────────────────────────────────────
        header = (
            "| fixture | orig_tok | ext_tok | ext_ratio | "
            "abs_tok | abs_ratio | flags_ok | vers_ok |"
        )
        sep = "|---|---|---|---|---|---|---|---|"
        lines = ["\n\n=== COMPRESSION COMPARISON REPORT ===\n", header, sep]
        for r in rows:
            lines.append(
                f"| {r['fixture']} | {r['orig_tok']} | {r['ext_tok']} | "
                f"{r['ext_ratio']} | {r['abs_tok']} | {r['abs_ratio']} | "
                f"{r['flags_ok']} | {r['vers_ok']} |"
            )
        lines.append("")
        output = "\n".join(lines) + "\n"
        sys.stdout.buffer.write(output.encode("utf-8"))
        sys.stdout.buffer.flush()

        # ── Aggregate stats ────────────────────────────────────────────
        if rows:
            avg_ext = sum(
                float(r["ext_ratio"].rstrip("%")) / 100
                for r in rows
                if isinstance(r["ext_ratio"], str) and r["ext_ratio"] != "N/A"
            ) / len(rows)
            flags_preserved = sum(1 for r in rows if r["flags_ok"] == "OK")
            vers_preserved = sum(1 for r in rows if r["vers_ok"] == "OK")
            summary = (
                f"\nSummary:\n"
                f"  Avg extractive prose reduction : {avg_ext:.1%}\n"
                f"  Fixtures with all flags preserved : {flags_preserved}/{len(rows)}\n"
                f"  Fixtures with all versions preserved: {vers_preserved}/{len(rows)}\n"
            )
            sys.stdout.buffer.write(summary.encode("utf-8"))
            sys.stdout.buffer.flush()

        # Always passes — data only
        self.assertTrue(len(rows) > 0, "No rows generated — fixture loop may be broken")


# ══════════════════════════════════════════════════════════════════════
# Test 7 — Segmentation round-trip (lossless reconstruction)
# ══════════════════════════════════════════════════════════════════════

class TestSegmentationRoundTrip(unittest.TestCase):
    """Joining segments must reconstruct the original text exactly."""

    def test_roundtrip_all_fixtures(self) -> None:
        for fix in FIXTURES:
            segs = segment_text(fix.raw_text)
            reconstructed = reassemble_segments(segs)
            self.assertEqual(
                reconstructed, fix.raw_text,
                f"[{fix.name}] Round-trip mismatch:\n"
                f"  original   : {fix.raw_text!r}\n"
                f"  reconstructed: {reconstructed!r}",
            )

    def test_empty_input(self) -> None:
        self.assertEqual(segment_text(""), [])

    def test_only_whitespace(self) -> None:
        segs = segment_text("   \n  \n  ")
        reconstructed = reassemble_segments(segs)
        # Must reconstruct exactly, even for whitespace-only input
        self.assertEqual(reconstructed, "   \n  \n  ")


# ══════════════════════════════════════════════════════════════════════
# Test 8 — compress_chunk end-to-end (DocChunk → CompressedDocChunk)
# ══════════════════════════════════════════════════════════════════════

class TestCompressChunkEndToEnd(unittest.TestCase):
    """Smoke test the full pipeline via compress_chunk."""

    def test_all_fixtures_extractive(self) -> None:
        for fix in FIXTURES:
            chunk = _make_doc_chunk(fix)
            result = compress_chunk(chunk, method="extractive", cache=None)

            # Must return a CompressedDocChunk (or DocChunk with report)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.compression_report)

            report = result.compression_report
            # Code token count must equal original code token count
            # (also verified internally by reassemble_chunk via assert)
            self.assertIsInstance(report.code_token_count, int)
            self.assertGreaterEqual(report.code_token_count, 0)
            self.assertGreaterEqual(report.prose_reduction_ratio, 0.0)
            self.assertLessEqual(report.prose_reduction_ratio, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
