"""Abstractive compression via a small local instruct model (optional).

Falls back to extractive compression if ``transformers`` / ``torch`` is
not installed or if model loading fails.

Caching strategy
----------------
Every compression result is stored in the shared ``TTLCache`` under a
key of the form::

    ("abs_compress", sha256(segment_text + MODEL_VERSION + PROMPT_VERSION)[:16])

Because the key is fully determined by the input text and both version
constants, the same segment always returns the same cached output without
ever re-running the model.

To invalidate the cache for all existing entries (e.g. when changing the
prompt template or upgrading the model), bump either ``MODEL_VERSION`` or
``PROMPT_VERSION`` below.

Usage::

    from src.mcp.cache import TTLCache
    from src.mcp.segmentation import segment_text
    from src.mcp.compress_abstractive import compress_segments_abstractive

    cache = TTLCache()
    segs = segment_text(raw_text)
    compressed = compress_segments_abstractive(segs, cache=cache)
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from .segmentation import Segment
from .compress_extractive import compress_prose_extractive
from .cache import TTLCache, DOCS_TTL

if TYPE_CHECKING:
    pass

log = logging.getLogger("mcp.compress_abstractive")

# ── Cache invalidation sentinels ───────────────────────────────────────
# Bump either string when the model or prompt changes.
MODEL_VERSION: str = "Qwen/Qwen2.5-0.5B-Instruct@v1"
PROMPT_VERSION: str = "v1"

# ── Generation limits ──────────────────────────────────────────────────
_MAX_NEW_TOKENS: int = 256
_MAX_INPUT_CHARS: int = 2_000   # skip model for very long inputs; use extractive

# Fixed prompt template (PROMPT_VERSION must match)
_PROMPT_TEMPLATE: str = (
    "Simplify the following documentation to short plain sentences.\n"
    "Keep all numbers, flags (like --option or -x), package names, and version strings verbatim.\n"
    "Do not add any information not present in the source text.\n"
    "Answer with only the simplified text, no preamble.\n\n"
    "SOURCE:\n{source}"
)

# ── Lazy model state ───────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_load_failed: bool = False


def _cache_key(segment_text: str) -> tuple[str, str]:
    """Build the TTLCache lookup key for *segment_text*."""
    raw = segment_text + MODEL_VERSION + PROMPT_VERSION
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return ("abs_compress", digest)


def _ensure_model() -> bool:
    """Load the model/tokenizer if not already loaded.

    Returns ``True`` on success, ``False`` if unavailable.
    """
    global _model, _tokenizer, _model_load_failed

    if _model is not None:
        return True
    if _model_load_failed:
        return False

    try:
        import torch  # noqa: F401 — check torch availability first
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        log.info("Loading abstractive model: %s", model_id)

        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",      # fp16 on CUDA, fp32 on CPU
            device_map="auto",       # CUDA if available, else CPU
        )
        _model.eval()
        log.info("Abstractive model loaded successfully.")
        return True

    except ImportError:
        log.warning(
            "transformers/torch not installed — abstractive compression unavailable; "
            "falling back to extractive."
        )
        _model_load_failed = True
        return False
    except Exception as exc:
        log.warning(
            "Failed to load abstractive model (%s) — falling back to extractive.", exc
        )
        _model_load_failed = True
        return False


def _run_model(segment_text: str) -> str:
    """Run the instruct model on *segment_text* and return the summary.

    Must only be called after a successful ``_ensure_model()`` call.
    """
    import torch

    prompt = _PROMPT_TEMPLATE.format(source=segment_text)

    # Build chat-formatted input if the tokenizer supports it
    try:
        messages = [{"role": "user", "content": prompt}]
        input_text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        input_text = prompt

    inputs = _tokenizer(input_text, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,        # greedy — deterministic
            temperature=None,       # required when do_sample=False
            top_p=None,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_len:]
    summary = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return summary


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def compress_prose_abstractive(
    prose_text: str,
    cache: TTLCache | None = None,
) -> str:
    """Compress a single PROSE block using the local instruct model.

    Cache hit → returns immediately without touching the model.
    Model unavailable → falls back to extractive compression.

    Parameters
    ----------
    prose_text:
        Raw prose string.
    cache:
        Shared ``TTLCache`` instance.  Pass ``None`` to disable caching
        (model will run every call — only for unit tests).

    Returns
    -------
    str
        Compressed prose.
    """
    if not prose_text.strip():
        return prose_text

    key = _cache_key(prose_text)

    # ── Cache lookup ───────────────────────────────────────────────────
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            log.debug("abs_compress cache HIT %s", key[1])
            return cached

    # ── Fallback: too long or model unavailable ────────────────────────
    if len(prose_text) > _MAX_INPUT_CHARS or not _ensure_model():
        result = compress_prose_extractive(prose_text)
        if cache is not None:
            cache.set(key, result, ttl=DOCS_TTL)
        return result

    # ── Model inference ────────────────────────────────────────────────
    try:
        result = _run_model(prose_text)
        if not result:
            # Empty generation — fall back
            result = compress_prose_extractive(prose_text)
    except Exception as exc:
        log.warning("Abstractive generation failed (%s) — using extractive fallback.", exc)
        result = compress_prose_extractive(prose_text)

    if cache is not None:
        cache.set(key, result, ttl=DOCS_TTL)

    return result


def compress_segments_abstractive(
    segments: list[Segment],
    cache: TTLCache | None = None,
) -> list[Segment]:
    """Compress a list of segments using the abstractive model.

    CODE segments are passed through **byte-identical**.
    PROSE segments are processed with ``compress_prose_abstractive``.

    Parameters
    ----------
    segments:
        Output of ``segmentation.segment_text``.
    cache:
        Shared ``TTLCache`` instance for caching model outputs.

    Returns
    -------
    list[Segment]
        Compressed segments with preserved ``index`` values.
    """
    result: list[Segment] = []
    for seg in segments:
        if seg.kind == "CODE":
            result.append(Segment(kind="CODE", text=seg.text, index=seg.index))
        else:
            compressed_text = compress_prose_abstractive(seg.text, cache=cache)
            result.append(Segment(kind="PROSE", text=compressed_text, index=seg.index))
    return result


def reset_model() -> None:
    """Unload the model and reset the load-failed flag.

    Intended for testing only — allows re-triggering model load logic
    after patching imports.
    """
    global _model, _tokenizer, _model_load_failed
    _model = None
    _tokenizer = None
    _model_load_failed = False
