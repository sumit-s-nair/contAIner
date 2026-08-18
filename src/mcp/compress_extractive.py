"""Extractive baseline for doc-compression (deterministic, no model).

For each PROSE ``Segment``, applies sentence-level keep/drop rules:

Keep rules (any match → sentence is kept):
    1. First sentence in the segment (always kept).
    2. Contains a version-number token  (e.g. ``1.26.0``, ``v3.2``, ``3.x``).
    3. Contains a flag-like token       (e.g. ``--output``, ``-q``).
    4. Contains any signal word from the *safety / importance* vocabulary
       (``required``, ``deprecated``, ``warning``, ``note``, ``caution``,
        ``important``, ``default``, ``must``, ``breaking``).

All other sentences are dropped.

CODE segments are passed through byte-identical (no modification whatsoever).

Usage::

    from src.mcp.segmentation import segment_text
    from src.mcp.compress_extractive import compress_segments_extractive

    segs = segment_text(raw_text)
    compressed = compress_segments_extractive(segs)
    # compressed is a new list[Segment] — same CODE, smaller PROSE
"""

from __future__ import annotations

import re
from typing import Literal

from .segmentation import Segment

# ── Sentence boundary splitter ─────────────────────────────────────────
# Split on   .  !  ?   followed by whitespace and an uppercase letter.
# This is intentionally simple; edge cases like "e.g." are acceptable
# losses for a compression baseline.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")

# ── Keep patterns ──────────────────────────────────────────────────────

# Version number: 1.26.0, v3.2, 3.x, 1.2.3-beta1, 2.0.0rc1, etc.
_VERSION_RE = re.compile(
    r"(?<![a-zA-Z])"          # not part of a larger word
    r"v?\d+\.\d+(?:\.\d+)?"   # core: [v]N.N[.N]
    r"[a-zA-Z0-9._-]*"        # optional pre/post-release suffix
    r"(?![a-zA-Z])"           # not followed by a letter (avoids e.g. "1.5x faster")
)

# Flag-like tokens: --long-flag or -x (not at end of a filename path)
_FLAG_RE = re.compile(r"(?:^|\s)(?:--[a-zA-Z][a-zA-Z0-9_-]*|-[a-zA-Z])(?:\s|$|[,;:.])")

# Signal / safety words (whole-word, case-insensitive)
_SIGNAL_WORDS: frozenset[str] = frozenset({
    "required", "require", "requires",
    "deprecated", "deprecate", "deprecation",
    "warning", "warn", "caution",
    "note", "notice",
    "important",
    "default",
    "must",
    "breaking",
    "mandatory",
    "obsolete",
    "removed",
    "changed",
    "error",
    "fail", "fails", "failed", "failure",
})
_SIGNAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _SIGNAL_WORDS) + r")\b",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════
# Sentence splitting
# ══════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences.

    Uses punctuation-based splitting first; falls back to newline splitting
    when the result is a single long chunk (likely pre-split plain text).
    """
    sentences = _SENT_BOUNDARY.split(text)
    if len(sentences) == 1:
        # Try newline splitting as fallback
        by_newline = [s.strip() for s in text.split("\n") if s.strip()]
        if len(by_newline) > 1:
            return by_newline
    return [s.strip() for s in sentences if s.strip()]


# ══════════════════════════════════════════════════════════════════════
# Per-sentence keep logic
# ══════════════════════════════════════════════════════════════════════

def _should_keep(sentence: str) -> bool:
    """Return True if *sentence* satisfies any keep rule (rules 2–4).

    Rule 1 (first sentence) is handled separately by the caller.
    """
    if _VERSION_RE.search(sentence):
        return True
    if _FLAG_RE.search(sentence):
        return True
    if _SIGNAL_RE.search(sentence):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def compress_prose_extractive(prose_text: str) -> str:
    """Apply extractive compression to a single PROSE block.

    Parameters
    ----------
    prose_text:
        Raw prose string (may span multiple sentences / lines).

    Returns
    -------
    str
        Compressed prose.  At minimum contains the first sentence.
        Returns ``prose_text`` unchanged if it is empty or a single sentence.
    """
    if not prose_text.strip():
        return prose_text

    sentences = _split_sentences(prose_text)
    if len(sentences) <= 1:
        return prose_text  # nothing to drop

    kept: list[str] = [sentences[0]]  # Rule 1: always keep first sentence
    for sent in sentences[1:]:
        if _should_keep(sent):
            kept.append(sent)

    # Reconstruct: join with space, preserve trailing newline if original had one
    compressed = " ".join(kept)
    if prose_text.endswith("\n"):
        compressed += "\n"
    return compressed


def compress_segments_extractive(segments: list[Segment]) -> list[Segment]:
    """Compress a list of segments using the extractive baseline.

    CODE segments are passed through **byte-identical** — they are not
    modified in any way.  PROSE segments are compressed with
    ``compress_prose_extractive``.

    Parameters
    ----------
    segments:
        Output of ``segmentation.segment_text``.

    Returns
    -------
    list[Segment]
        New list of segments with the same ``index`` values (so reassembly
        order is preserved) and CODE text unchanged.
    """
    result: list[Segment] = []
    for seg in segments:
        if seg.kind == "CODE":
            # Byte-identical pass-through — do not touch
            result.append(Segment(kind="CODE", text=seg.text, index=seg.index))
        else:
            compressed_text = compress_prose_extractive(seg.text)
            result.append(Segment(kind="PROSE", text=compressed_text, index=seg.index))
    return result
