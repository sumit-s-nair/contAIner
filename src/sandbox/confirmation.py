"""
src/sandbox/confirmation.py
============================
Shared user-confirmation gate for the contAIner sandbox harness.

This is the SINGLE "ask the user" interface used by:

  1. REVIEW-tier risk classification (SandboxExecutor)
  2. System 1 low-confidence disambiguation (bridge/server.py)

Having one function — rather than two separate "ask" code paths — ensures
consistent behaviour, testability, and a single place to extend (e.g. swap
from interactive stdin to an SSE event without changing callers).

Usage
-----
    from src.sandbox.confirmation import request_user_confirmation

    approved = request_user_confirmation(
        reason="This command deletes files permanently.",
        context="rm old_logs/",
        prompt_fn=my_sse_prompt,   # optional; defaults to stdin
    )
    if approved:
        executor.run(command)

Injectable ``prompt_fn``
------------------------
Tests and the SSE bridge should pass a ``prompt_fn`` to avoid blocking on
stdin.  The function signature is:

    def my_prompt(reason: str, context: str) -> bool: ...

If ``prompt_fn`` is ``None``, the gate falls back to an interactive stdin
prompt (suitable for CLI / script use).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

# Type alias for the injectable prompt callable.
PromptFn = Callable[[str, str], bool]


def request_user_confirmation(
    reason: str,
    context: str,
    *,
    prompt_fn: Optional[PromptFn] = None,
) -> bool:
    """
    Present a REVIEW-tier or low-confidence confirmation request and return
    ``True`` if the user approves, ``False`` if they deny.

    This function is synchronous by design — the calling loop is expected to
    block here.  Async contexts (SSE bridge) should supply a ``prompt_fn``
    that internally handles the async coordination and returns a concrete bool
    when it resolves.

    Parameters
    ----------
    reason:
        Plain-English explanation of *why* confirmation is required.
        Sourced from ``ClassificationResult.reason`` (risk gate) or the
        System 1 clarifying question (confidence gate).

    context:
        The literal command string (risk gate) or the user's original prompt
        (confidence gate) — whatever gives the user enough context to make an
        informed decision.

    prompt_fn:
        Optional injectable callback.  If supplied, it is called with
        ``(reason, context)`` and **must** return a ``bool``.  Use this in
        tests and async bridge code to avoid blocking on stdin.

    Returns
    -------
    bool
        ``True`` → user approved; execution may proceed.
        ``False`` → user denied; step is aborted.

    Raises
    ------
    Does not raise.  Any exception inside ``prompt_fn`` is caught and treated
    as a denial (safe default), with the exception message written to stderr.

    Examples
    --------
    Interactive (CLI):

        >>> approved = request_user_confirmation(
        ...     reason="Deletes files permanently.",
        ...     context="rm build/",
        ... )

    Injected (tests):

        >>> approved = request_user_confirmation(
        ...     reason="Deletes files permanently.",
        ...     context="rm build/",
        ...     prompt_fn=lambda r, c: True,   # always approve
        ... )
        >>> assert approved is True
    """
    if prompt_fn is not None:
        try:
            result = prompt_fn(reason, context)
            return bool(result)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[confirmation] prompt_fn raised an exception — treating as denial: {exc}",
                file=sys.stderr,
            )
            return False

    return _interactive_prompt(reason, context)


# ---------------------------------------------------------------------------
# Default interactive (stdin) prompt
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 60


def _interactive_prompt(reason: str, context: str) -> bool:
    """
    Render a formatted REVIEW prompt to stdout and read a y/n response from
    stdin.

    Returns ``True`` for ``y`` / ``yes`` (case-insensitive), ``False`` for
    anything else (including EOF / Ctrl-C, which are caught and treated as
    denial).
    """
    print(f"\n{_SEPARATOR}")
    print("⚠  REVIEW REQUIRED — user confirmation needed before execution")
    print(_SEPARATOR)
    print(f"Reason  : {reason}")
    print(f"Command : {context}")
    print(_SEPARATOR)

    try:
        answer = input("Proceed? [y/N] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[confirmation] Input interrupted — treating as denial.")
        return False

    approved = answer in ("y", "yes")
    if approved:
        print("[confirmation] Approved — executing.")
    else:
        print("[confirmation] Denied — step aborted.")
    return approved
