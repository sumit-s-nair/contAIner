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


def _inject_compressed_text(
    chunk: DocChunk,
    segments: list[Segment],
) -> DocChunk:
    """Write compressed segment text back into the chunk's narrative fields.

    Because segments may span multiple logical fields, we reconstruct the
    full combined text and then split on the ``\\n\\n`` boundaries used in
    ``_extract_compressible_text``.
    """
    reassembled = reassemble_segments(segments)
    # Re-split by paragraph separator (double newline)
    parts = [p.strip() for p in reassembled.split("\n\n") if p.strip()]

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

    # Distribute compressed parts back into fields in the same order they were added.
    part_idx = 0

    if chunk.os_specific_notes and part_idx < len(parts):
        new_chunk.os_specific_notes = parts[part_idx]
        part_idx += 1

    flag_descs = [
        f for f in (chunk.key_flags or [])
        if isinstance(f, dict) and f.get("description")
    ]
    if flag_descs and part_idx < len(parts):
        # The flag descriptions were joined with \n; split them back
        compressed_descs = [s.strip() for s in parts[part_idx].split("\n") if s.strip()]
        for i, flag in enumerate(new_chunk.key_flags):
            if isinstance(flag, dict) and flag.get("description"):
                if i < len(compressed_descs):
                    flag["description"] = compressed_descs[i]
        part_idx += 1

    return new_chunk


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def reassemble_chunk(
    chunk: DocChunk,
    compressed_segments: list[Segment],
    original_segments: list[Segment],
    method: Literal["extractive", "abstractive"],
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
    orig_prose_tokens = sum(
        _tok(s.text) for s in original_segments if s.kind == "PROSE"
    )
    comp_prose_tokens = sum(
        _tok(s.text) for s in compressed_segments if s.kind == "PROSE"
    )

    original_total = orig_code_tokens + orig_prose_tokens
    compressed_total = comp_code_tokens + comp_prose_tokens

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

    # ── Inject compressed text back into the chunk ─────────────────────
    new_chunk = _inject_compressed_text(chunk, compressed_segments)
    new_chunk.tokens_estimate = compressed_total

    result = CompressedDocChunk.from_doc_chunk(new_chunk, report=report)

    log.debug(
        "reassemble_chunk: orig=%d tok, compressed=%d tok, "
        "code=%d tok, prose_ratio=%.2f, method=%s",
        original_total, compressed_total,
        orig_code_tokens, prose_reduction_ratio, method,
    )
    return result
