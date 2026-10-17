"""Reassembly stage — stitch compressed segments back into a DocChunk.

After compression (extractive or abstractive), this module:
    1. Collects the text fields of a DocChunk that were segmented/compressed.
    2. Puts compressed segments back into those fields in original order.
    3. Builds a ``CompressionReport`` with token counts and reduction ratios.
    4. Wraps the result as a ``CompressedDocChunk``.

The hard invariant (enforced via ``assert``, not a log warning):

    code_token_count_compressed == code_token_count_original

If this assertion ever fails it means a CODE segment was silently mutated
somewhere in the pipeline — that is a bug, not an expected outcome.

Token counting
--------------
All token counts use the same ``≈ 4 chars per token`` heuristic already
used by ``DocChunk.estimate_tokens`` so the numbers are internally
consistent (we are not mixing char-counts and tiktoken counts).
"""

from __future__ import annotations

import logging
from typing import Literal

from .models import DocChunk, CompressedDocChunk, CompressionReport
from .segmentation import Segment, segment_text, reassemble_segments
from .compress_extractive import compress_segments_extractive
from .compress_abstractive import compress_segments_abstractive
from .cache import TTLCache

log = logging.getLogger("mcp.reassemble")

# ── Token count helper (matches DocChunk.estimate_tokens heuristic) ────
def _tok(text: str) -> int:
    return max(0, len(text) // 4)


# ══════════════════════════════════════════════════════════════════════
# DocChunk text field extraction / injection
# ══════════════════════════════════════════════════════════════════════

def _extract_compressible_text(chunk: DocChunk) -> str:
    """Collect the narrative (compressible) text from a DocChunk.

    We compress:
        - ``os_specific_notes``  (main prose field)
        - ``key_flags[].description``  (flag descriptions)

    We treat as CODE (leave in ``command_syntax`` / ``examples``):
        - ``command_syntax``  — already extracted command; compressor would see it as CODE anyway
        - ``examples``        — already extracted examples

    Returns a single concatenated string with a ``\\n\\n`` separator so
    segment boundaries are respected.
    """
    parts: list[str] = []

    if chunk.os_specific_notes:
        parts.append(chunk.os_specific_notes)

    flag_descs = [
        f["description"]
        for f in (chunk.key_flags or [])
        if isinstance(f, dict) and f.get("description")
    ]
    if flag_descs:
        parts.append("\n".join(flag_descs))

    return "\n\n".join(p for p in parts if p.strip())


def _compress_flag_description(text: str) -> str:
    """Compress ONE flag description on its own (extractive, deterministic).

    Never returns an empty string: a flag without its description is worse
    than an uncompressed one.
    """
    if not text or not text.strip():
        return text
    out = reassemble_segments(compress_segments_extractive(segment_text(text))).strip()
    return out or text


def _inject_compressed_text(
    chunk: DocChunk,
    segments: list[Segment],
    method: Literal["extractive", "abstractive"] = "extractive",
    cache: TTLCache | None = None,
) -> DocChunk:
    """Write compressed text back into the chunk's narrative fields.

    ``os_specific_notes`` takes the compressed text of its own segments.
    Flag descriptions are compressed one by one (``_compress_flag_description``)
    so each flag keeps its own description.

    Stage 3A fix: this used to split the compressed notes+flags blob on
    newlines and hand the pieces out by position.  The extractive compressor
    re-joins kept sentences with spaces, so the boundaries vanished: with no
    OS notes, flag 0 received sentences from later flags; with OS notes, the
    notes absorbed flag text.  The MCP snapshot used for System 3 had pip
    flags paired with other flags' descriptions.
    """
    notes_text = ""
    if chunk.os_specific_notes and chunk.os_specific_notes.strip():
        # The notes are the first part of the blob (see
        # _extract_compressible_text).  Compress them on their own so no flag
        # text can leak in.
        notes_segments = segment_text(chunk.os_specific_notes)
        if method == "abstractive":
            notes_segments = compress_segments_abstractive(notes_segments, cache=cache)
        else:
            notes_segments = compress_segments_extractive(notes_segments)
        notes_text = reassemble_segments(notes_segments).strip() or chunk.os_specific_notes

    new_chunk = DocChunk(
        tool=chunk.tool,
        operation=chunk.operation,
        command_syntax=chunk.command_syntax,
        key_flags=[dict(f) for f in chunk.key_flags],
        examples=list(chunk.examples),
        source_urls=list(chunk.source_urls),
        tool_version=chunk.tool_version,
        tokens_estimate=chunk.tokens_estimate,
        os_specific_notes=chunk.os_specific_notes,
        error=chunk.error,
    )

    if notes_text:
        new_chunk.os_specific_notes = notes_text

    for flag in new_chunk.key_flags:
        if isinstance(flag, dict) and flag.get("description"):
            flag["description"] = _compress_flag_description(flag["description"])

    return new_chunk



# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def reassemble_chunk(
    chunk: DocChunk,
    compressed_segments: list[Segment],
    original_segments: list[Segment],
    method: Literal["extractive", "abstractive"],
    cache: TTLCache | None = None,
) -> CompressedDocChunk:
    """Build a ``CompressedDocChunk`` from original and compressed segments.

    Parameters
    ----------
    chunk:
        The original ``DocChunk`` (not mutated).
    compressed_segments:
        Output of ``compress_segments_extractive`` or
        ``compress_segments_abstractive``.
    original_segments:
        The ``Segment`` list from ``segment_text`` *before* compression
        (needed for computing original token counts and verifying CODE
        byte-identity).
    method:
        Which compression path was used.

    Returns
    -------
    CompressedDocChunk

    Raises
    ------
    AssertionError
        If any CODE segment's token count changed between original and
        compressed (i.e., a CODE block was modified — must never happen).
    """
    # ── Build lookup maps ──────────────────────────────────────────────
    orig_by_idx = {s.index: s for s in original_segments}
    comp_by_idx = {s.index: s for s in compressed_segments}

    # ── Verify CODE byte-identity and collect CODE token count ─────────
    orig_code_tokens = 0
    comp_code_tokens = 0
    for idx, orig_seg in orig_by_idx.items():
        if orig_seg.kind == "CODE":
            orig_code_tokens += _tok(orig_seg.text)
            comp_seg = comp_by_idx.get(idx)
            if comp_seg is not None:
                comp_code_tokens += _tok(comp_seg.text)
                # Hard assertion — not a soft check
                assert comp_seg.text == orig_seg.text, (
                    f"CODE segment at index {idx} was mutated during compression!\n"
                    f"  original ({_tok(orig_seg.text)} tok): {orig_seg.text!r}\n"
                    f"  compressed ({_tok(comp_seg.text)} tok): {comp_seg.text!r}"
                )

    assert orig_code_tokens == comp_code_tokens, (
        f"CODE segment token count changed: "
        f"{orig_code_tokens} orig → {comp_code_tokens} compressed"
    )

    # ── Token counts ───────────────────────────────────────────────────
    # Both totals are measured the same way: tokens of the compressible text
    # of the chunk (before / after).
    original_total = _tok(_extract_compressible_text(chunk))
    orig_prose_tokens = max(0, original_total - orig_code_tokens)

    # ── Inject compressed text back into the chunk ─────────────────────
    # Notes are compressed with *method*; flag descriptions one by one
    # (see _inject_compressed_text).  The report is computed from the chunk
    # actually returned, not from the blob-level segments.
    new_chunk = _inject_compressed_text(chunk, compressed_segments, method=method, cache=cache)
    compressed_total = _tok(_extract_compressible_text(new_chunk))
    comp_prose_tokens = max(0, compressed_total - orig_code_tokens)

    prose_reduction_ratio: float = 0.0
    if orig_prose_tokens > 0:
        prose_reduction_ratio = 1.0 - (comp_prose_tokens / orig_prose_tokens)
    # Clamp to [0, 1] — can't reduce below 0 or "expand" with this metric
    prose_reduction_ratio = max(0.0, min(1.0, prose_reduction_ratio))

    report = CompressionReport(
        original_token_count=original_total,
        compressed_token_count=compressed_total,
        code_token_count=orig_code_tokens,
        prose_reduction_ratio=round(prose_reduction_ratio, 4),
        method=method,
    )
    new_chunk.tokens_estimate = compressed_total

    result = CompressedDocChunk.from_doc_chunk(new_chunk, report=report)

    log.debug(
        "reassemble_chunk: orig=%d tok, compressed=%d tok, "
        "code=%d tok, prose_ratio=%.2f, method=%s",
        original_total, compressed_total,
        orig_code_tokens, prose_reduction_ratio, method,
    )
    return result
