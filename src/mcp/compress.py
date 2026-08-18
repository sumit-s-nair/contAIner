"""Public entry point for the doc-compression pipeline.

This module is what ``server.py`` calls after routing and before caching.

Pipeline::

    DocChunk
        │
        ├─ extract narrative text fields
        │
        ├─ segment_text()          [segmentation.py]
        │
        ├─ compress_segments_*()   [compress_extractive.py | compress_abstractive.py]
        │
        └─ reassemble_chunk()      [reassemble.py]
                │
                └─ CompressedDocChunk (with CompressionReport attached)

Usage::

    from src.mcp.compress import compress_chunk
    from src.mcp.cache import TTLCache

    cache = TTLCache()
    result = compress_chunk(chunk, method="extractive", cache=cache)
    # result.compression_report has all the metrics
    # result.os_specific_notes is compressed
    # result.command_syntax / examples are unchanged
"""

from __future__ import annotations

import logging
from typing import Literal

from .models import DocChunk, CompressedDocChunk
from .cache import TTLCache
from .segmentation import segment_text
from .compress_extractive import compress_segments_extractive
from .compress_abstractive import compress_segments_abstractive
from .reassemble import reassemble_chunk

log = logging.getLogger("mcp.compress")


def compress_chunk(
    chunk: DocChunk,
    method: Literal["extractive", "abstractive"] = "extractive",
    cache: TTLCache | None = None,
) -> CompressedDocChunk:
    """Run the full doc-compression pipeline on *chunk*.

    Parameters
    ----------
    chunk:
        Raw ``DocChunk`` returned by an adapter (not mutated).
    method:
        ``"extractive"`` — deterministic, no model, always available.
        ``"abstractive"`` — local Qwen2.5-0.5B-Instruct model, cached,
        falls back to extractive if transformers/torch unavailable.
    cache:
        Shared ``TTLCache`` instance.  Used for abstractive model output
        caching.  Pass ``None`` to disable caching (testing only).

    Returns
    -------
    CompressedDocChunk
        The chunk with narrative fields compressed and a
        ``compression_report`` attached.
    """
    # ── Step 1: collect compressible text ─────────────────────────────
    from .reassemble import _extract_compressible_text  # local import avoids circularity

    raw_text = _extract_compressible_text(chunk)

    # If there is nothing to compress (all fields empty), wrap and return
    if not raw_text.strip():
        log.debug("compress_chunk: no compressible text found — passthrough")
        from .models import CompressionReport
        report = CompressionReport(
            original_token_count=chunk.tokens_estimate,
            compressed_token_count=chunk.tokens_estimate,
            code_token_count=0,
            prose_reduction_ratio=0.0,
            method=method,
        )
        return CompressedDocChunk.from_doc_chunk(chunk, report=report)

    # ── Step 2: segment ────────────────────────────────────────────────
    original_segments = segment_text(raw_text)

    # ── Step 3: compress ───────────────────────────────────────────────
    if method == "abstractive":
        compressed_segments = compress_segments_abstractive(
            original_segments, cache=cache
        )
    else:
        compressed_segments = compress_segments_extractive(original_segments)

    # ── Step 4: reassemble + report ────────────────────────────────────
    result = reassemble_chunk(
        chunk=chunk,
        compressed_segments=compressed_segments,
        original_segments=original_segments,
        method=method,
    )
    return result
