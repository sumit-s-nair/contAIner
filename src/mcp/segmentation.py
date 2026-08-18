"""Deterministic text segmentation for doc-compression.

Splits a raw documentation string into an ordered list of ``Segment``
objects tagged as either ``"CODE"`` or ``"PROSE"``.  The original text
can be reconstructed exactly by joining ``segment.text`` in index order.

Detection rules (applied in precedence order):
    1. Fenced code blocks  — ``` ... ``` (any language tag, multiline)
    2. Indented code blocks — 4-space- or tab-prefixed paragraph
    3. Single-line command patterns — line starts with a known CLI verb,
       or contains ``--flag`` / `` -x`` syntax
    4. Everything else     → PROSE

No third-party dependencies required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ── Known CLI verbs that trigger command-line detection ────────────────
_CLI_VERBS: frozenset[str] = frozenset({
    "pip", "pip3",
    "npm", "npx", "yarn", "pnpm",
    "apt", "apt-get", "apt-cache",
    "brew",
    "conda",
    "docker", "docker-compose",
    "cargo",
    "go",
    "mvn", "maven",
    "sudo",
    "chmod", "chown",
    "curl", "wget",
    "git",
    "cat", "echo", "export", "source",
    "cp", "mv", "rm", "ls", "cd", "mkdir", "touch",
    "python", "python3", "python2",
    "node", "nodejs",
    "bash", "sh", "zsh", "fish",
    "make", "cmake",
    "tar", "gzip", "zip", "unzip",
    "ssh", "scp", "rsync",
    "systemctl", "service",
    "env", "set",
})

# Pattern: --long-flag or -x (short flag), not inside a word
_FLAG_PATTERN = re.compile(r"(?:^|\s)(?:--[a-zA-Z][a-zA-Z0-9_-]*|-[a-zA-Z](?:\s|$))")

# Fenced code block opener/closer
_FENCE_OPEN = re.compile(r"^(?P<fence>`{3,}|~{3,})(?P<lang>[a-zA-Z0-9_+-]*)[ \t]*$")

# Indented code: 4 spaces or a tab at line start
_INDENT_PREFIX = re.compile(r"^(?:    |\t)")


# ══════════════════════════════════════════════════════════════════════
# Public data types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """A single contiguous text block with a CODE or PROSE tag.

    ``text`` preserves the exact original bytes (including trailing
    newlines) so that joining all segments in ``index`` order reconstructs
    the original input character-for-character.
    """
    kind: Literal["CODE", "PROSE"]
    text: str
    index: int  # 0-based insertion order; stable for reassembly


# ══════════════════════════════════════════════════════════════════════
# Segmentation helpers
# ══════════════════════════════════════════════════════════════════════

def _is_command_line(line: str) -> bool:
    """Return True if *line* looks like a standalone shell command.

    Criteria:
    - Length ≤ 200 characters (longer → probably prose with dashes)
    - First non-whitespace token is a known CLI verb, OR
    - Line contains a flag-like token (``--opt`` or `` -x``)
    - Line does NOT end with ``.`` or ``,`` (sentence terminator)
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return False
    # Skip lines that are clearly prose sentences
    if stripped.endswith((".", ",", ";", ":", "?")):
        return False
    # Skip lines that are headings (#) or list bullets (*, -, +) at start
    if re.match(r"^#{1,6}\s", stripped):
        return False

    # First word match against CLI verbs
    first_word = re.split(r"[\s/]", stripped)[0].lstrip("$").strip()
    if first_word in _CLI_VERBS:
        return True

    # Flag syntax match
    if _FLAG_PATTERN.search(stripped):
        return True

    return False


def _classify_indented_block(lines: list[str]) -> bool:
    """Return True if *all* non-empty lines are indented (4 spaces or tab)."""
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False
    return all(_INDENT_PREFIX.match(l) for l in non_empty)


# ══════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════

def segment_text(text: str) -> list[Segment]:
    """Split *text* into an ordered list of CODE / PROSE ``Segment`` objects.

    The original text is losslessly reconstructed by::

        "".join(seg.text for seg in sorted(segments, key=lambda s: s.index))

    Parameters
    ----------
    text:
        Raw documentation string (may contain markdown, plain text, or both).

    Returns
    -------
    list[Segment]
        Ordered segments.  Empty input returns ``[]``.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    segments: list[Segment] = []
    seg_index = 0

    i = 0
    current_prose_lines: list[str] = []

    def _flush_prose() -> None:
        nonlocal seg_index
        if current_prose_lines:
            prose_text = "".join(current_prose_lines)
            if prose_text.strip():  # only emit non-blank prose
                segments.append(Segment(kind="PROSE", text=prose_text, index=seg_index))
                seg_index += 1
            elif prose_text:  # blank/whitespace-only — still preserve for reconstruction
                segments.append(Segment(kind="PROSE", text=prose_text, index=seg_index))
                seg_index += 1
            current_prose_lines.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\r\n")

        # ── Rule 1: Fenced code block ──────────────────────────────────
        fence_match = _FENCE_OPEN.match(stripped)
        if fence_match:
            _flush_prose()
            fence_char = fence_match.group("fence")[0]
            fence_len = len(fence_match.group("fence"))
            close_pattern = re.compile(
                r"^" + re.escape(fence_char * fence_len) + r"\s*$"
            )
            block_lines = [line]
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if close_pattern.match(lines[i].rstrip("\r\n")):
                    i += 1
                    break
                i += 1
            segments.append(Segment(
                kind="CODE",
                text="".join(block_lines),
                index=seg_index,
            ))
            seg_index += 1
            continue

        # ── Rule 2: Indented code block ────────────────────────────────
        # Must be preceded by a blank line (or start of text), and all
        # lines in the block must be indented.
        prev_blank = (not current_prose_lines or
                      current_prose_lines[-1].strip() == "")
        if prev_blank and _INDENT_PREFIX.match(stripped if stripped else " " * 4):
            # Collect all consecutive indented lines
            block_lines = []
            j = i
            while j < len(lines):
                ln = lines[j]
                ln_stripped = ln.strip()
                if not ln_stripped:
                    # blank line — include if next line is also indented
                    if j + 1 < len(lines) and _INDENT_PREFIX.match(lines[j + 1]):
                        block_lines.append(ln)
                        j += 1
                        continue
                    else:
                        break
                if _INDENT_PREFIX.match(ln):
                    block_lines.append(ln)
                    j += 1
                else:
                    break

            if block_lines:
                _flush_prose()
                segments.append(Segment(
                    kind="CODE",
                    text="".join(block_lines),
                    index=seg_index,
                ))
                seg_index += 1
                i = j
                continue

        # ── Rule 3: Single-line command pattern ────────────────────────
        if _is_command_line(stripped):
            _flush_prose()
            segments.append(Segment(
                kind="CODE",
                text=line,
                index=seg_index,
            ))
            seg_index += 1
            i += 1
            continue

        # ── Rule 4: PROSE (accumulate) ─────────────────────────────────
        current_prose_lines.append(line)
        i += 1

    _flush_prose()
    return segments


def reassemble_segments(segments: list[Segment]) -> str:
    """Reconstruct a document from an ordered list of segments.

    Parameters
    ----------
    segments:
        Segments in any order; they will be sorted by ``index`` before joining.

    Returns
    -------
    str
        The reassembled document.
    """
    return "".join(seg.text for seg in sorted(segments, key=lambda s: s.index))
