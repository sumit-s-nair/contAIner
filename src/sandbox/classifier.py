"""
src/sandbox/classifier.py
=========================
Three-tier command risk classifier.

Usage
-----
    from src.sandbox.classifier import CommandRiskClassifier
    from src.sandbox.models import AtomicStep, RiskTier

    clf = CommandRiskClassifier(sandbox_root="/home/user/project")
    result = clf.classify(AtomicStep(command="pip install requests"))
    assert result.tier == RiskTier.SAFE

Design
------
Classification proceeds in three ordered passes, returning as soon as a
tier is determined:

  1. BLOCKED pass  — hard denylist via compiled regex; final, no override.
  2. REVIEW pass   — pattern list check + planner ``destructive`` flag.
  3. SAFE pass     — known-safe action allowlist; unknown commands fall to
                     REVIEW (fail-safe default).

Force-flag scoping (per the design spec)
-----------------------------------------
``--force`` / ``-f`` alone does NOT trigger REVIEW.  The flag is only
escalating when the *verb* of the command is itself in a higher-risk verb
set (rm, remove, delete, purge, push, overwrite, drop, wipe).  This
prevents routine operations such as ``npm install --force`` or
``pip install --force-reinstall`` from generating REVIEW friction while
still catching genuinely risky uses like ``rm --force``, ``git push --force``,
or ``apt-get remove --force``.

Sandbox scope enforcement
--------------------------
Commands that attempt to write to paths *outside* ``sandbox_root`` are
BLOCKED.  Detection uses a simple heuristic — looking for absolute paths
in redirect operators (``>``, ``>>``) or explicit write-flag tool args
(``--output``, ``-o``, ``tee``) that resolve outside the sandbox.
This is intentionally conservative: false negatives (we miss a path) are
possible; false positives are avoided.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from .models import AtomicStep, ClassificationResult, RiskTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tok(pattern: str) -> re.Pattern[str]:
    """Compile *pattern* with IGNORECASE | DOTALL."""
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Tier BLOCKED — hard denylist
# ---------------------------------------------------------------------------
# Each entry is a (compiled_regex, human_reason) tuple.
# Order matters only for the reason string; any match is immediately BLOCKED.

_BLOCKED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- root-level rm -rf ---------------------------------------------------
    # rm -rf / | rm -rf /* | rm -fr / etc.  Anchored to root-level paths.
    (
        _tok(r"\brm\b[^|&;]*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+[\"']?/[\*]*[\"']?(?:\s|$)"),
        "Recursive force-delete targeting root-level path — would destroy the entire filesystem.",
    ),
    (
        _tok(r"\brm\b[^|&;]*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+[\"']?/[\*]*[\"']?(?:\s|$)"),
        "Recursive force-delete targeting root-level path (flags in reverse order) — would destroy the entire filesystem.",
    ),

    # --- disk formatting -----------------------------------------------------
    (
        _tok(r"\bmkfs\b"),
        "Disk formatting command (mkfs) — would permanently erase a filesystem.",
    ),
    (
        _tok(r"\bdd\b[^|&;]*\bof\s*=\s*/dev/"),
        "Raw disk write via dd — would overwrite a block device directly.",
    ),
    (
        _tok(r"\bshred\b[^|&;]*/dev/"),
        "Secure erase targeting a block device — irreversible hardware-level wipe.",
    ),

    # --- fork bombs ----------------------------------------------------------
    # Canonical Bash fork bomb: :(){ :|:& };:
    # Variants: different function name, whitespace, semicolons vs newlines.
    (
        _tok(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:.*&\s*\}"),
        "Fork bomb detected — would crash the system by exhausting process slots.",
    ),
    # Named-function variant: foo(){ foo|foo& };foo
    (
        _tok(r"(\w+)\s*\(\s*\)\s*\{[^}]*\1\s*\|\s*\1.*&\s*\}"),
        "Named-function fork bomb — would crash the system by exhausting process slots.",
    ),
    # .(){ .|.& };. — dot-function variant
    (
        _tok(r"\.\s*\(\s*\)\s*\{[^}]*\.\s*\|\s*\..*&\s*\}"),
        "Dot-function fork bomb variant — would crash the system.",
    ),
    # Extra-space / newline-separated variant guard: function_body with :|:
    (
        _tok(r"\(\s*\)\s*\{[^}]*\|\s*[^\s|&;]+\s*&"),
        "Shell function self-pipe fork pattern — potential fork bomb.",
    ),

    # --- network exfil / reverse shell ---------------------------------------
    (
        _tok(r"\b(curl|wget|fetch)\b[^|&;]*\|\s*(ba)?sh\b"),
        "Piping remote content directly into a shell — remote code execution risk.",
    ),
    (
        _tok(r"\b(curl|wget|fetch)\b[^|&;]*\|\s*sudo\b"),
        "Piping remote content into sudo — privilege-escalated remote code execution risk.",
    ),
    (
        _tok(r"\bnc(at)?\b[^|&;]*-[a-zA-Z]*e\b"),
        "Netcat with -e flag — reverse shell execution channel.",
    ),
    (
        _tok(r"\b(bash|sh|zsh)\b[^|&;]*-i\b[^|&;]*>\s*/dev/tcp/"),
        "Bash TCP redirect — reverse shell technique.",
    ),
    (
        _tok(r"/dev/tcp/\S+/\d+"),
        "Direct /dev/tcp connection — typical reverse shell or exfiltration channel.",
    ),
]


# ---------------------------------------------------------------------------
# Tier REVIEW — force-flag escalation: only with risky verbs
# ---------------------------------------------------------------------------
# ``--force`` / ``-f`` alone is NOT enough.  It must appear in a command
# whose first meaningful verb is in this set.

_FORCE_ESCALATING_VERBS: frozenset[str] = frozenset({
    "rm", "remove", "delete", "del", "purge", "wipe", "drop",
    "push",           # git push --force
    "reset",          # git reset --hard
    "clean",          # git clean -f
    "overwrite",
    "unlink",
})

# Regex to detect a force flag anywhere in the command.
_FORCE_FLAG_RE = _tok(r"(--force|-f\b|--force-with-lease)")

# Regex to extract the first "word token" that looks like a shell verb
# (skips sudo, env, nice, etc. wrapper prefixes up to 3 words).
_VERB_PREFIX_SKIP = frozenset({"sudo", "env", "nice", "ionice", "nohup", "doas",
                                "time", "xargs", "watch", "strace", "ltrace"})


def _extract_verb(command: str) -> str:
    """
    Return the first non-flag, non-prefix token from *command*, lower-cased.

    This is intentionally simple — we only need it for the force-flag
    escalation heuristic, so approximate accuracy is fine.
    """
    tokens = command.strip().split()
    for tok in tokens:
        bare = tok.lstrip("./").split("/")[-1].lower()  # strip path prefix
        if bare in _VERB_PREFIX_SKIP:
            continue
        if bare.startswith("-"):
            continue
        return bare
    return ""


def _extract_all_verbs(command: str) -> list[str]:
    """
    Return all verb-like tokens across the full command (handles pipes and &&).
    Used to catch ``rm --force`` even when preceded by another command.
    """
    # Split on shell separators: | ; && ||
    parts = re.split(r"[|;&]+", command)
    return [_extract_verb(p) for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Tier REVIEW — explicit pattern list (beyond force-flag)
# ---------------------------------------------------------------------------
# Each entry is (compiled_regex, human_reason).

_REVIEW_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- file/directory deletion ---------------------------------------------
    (
        _tok(r"\brm\s"),
        "Deletes files permanently — no Trash/recycle bin.",
    ),
    (
        _tok(r"\brmdir\b"),
        "Removes a directory — fails if non-empty, but permanent.",
    ),
    (
        _tok(r"\bunlink\b"),
        "Removes a filesystem entry — bypasses safety checks.",
    ),
    (
        _tok(r"\bshred\b"),
        "Securely overwrites and deletes a file — unrecoverable.",
    ),

    # --- privilege escalation ------------------------------------------------
    (
        _tok(r"\bsudo\b"),
        "Runs with superuser privileges — can affect system-wide state.",
    ),
    (
        _tok(r"\bsu\b\s+-"),
        "Switches to another user account — elevated privilege.",
    ),
    (
        _tok(r"\bdoas\b"),
        "Runs with delegated superuser privileges (doas).",
    ),

    # --- permission / ownership changes --------------------------------------
    (
        _tok(r"\bchmod\b"),
        "Changes file permissions — can expose sensitive files or break executable safety.",
    ),
    (
        _tok(r"\bchown\b"),
        "Changes file ownership — can escalate privilege or lock out access.",
    ),
    (
        _tok(r"\bchgrp\b"),
        "Changes file group ownership.",
    ),

    # --- package purge (not just remove) ------------------------------------
    (
        _tok(r"\b(apt(-get)?|dpkg|yum|dnf|pacman)\b[^|&;]*--purge\b"),
        "Purges a package including its configuration files — harder to undo than a simple remove.",
    ),
    (
        _tok(r"\bnpm\s+(uninstall|remove|rm)\b"),
        "Removes an npm package — may break dependent code.",
    ),
    (
        _tok(r"\bpip\s+(uninstall|remove)\b"),
        "Removes a Python package — may break dependent code.",
    ),
    (
        _tok(r"\bcargo\s+remove\b"),
        "Removes a Rust crate dependency.",
    ),

    # --- overwrite without backup --------------------------------------------
    # Shell output redirect to a pre-existing file is only heuristic here;
    # we flag "> " followed by a path that doesn't look like /dev/null.
    (
        _tok(r"(?<!>)>\s*(?!/dev/null)[^\s>]"),
        "Output redirect that may overwrite an existing file without a backup.",
    ),
    (
        _tok(r"\bmv\b[^|&;]*--no-clobber\b"),
        # Negative: mv --no-clobber is actually SAFE; but bare mv is risky.
        # We handle bare mv below and skip this — this entry is a reminder.
        None,  # type: ignore[arg-type]  # intentionally skipped; see _classify_review
    ),
    (
        _tok(r"\bmv\b(?![^|&;]*--no-clobber)"),
        "Moves/renames a file — destination will be silently overwritten if it already exists.",
    ),

    # --- git destructive operations ------------------------------------------
    (
        _tok(r"\bgit\s+(push|reset|rebase|clean)\b"),
        "Git operation that can rewrite history or discard local changes.",
    ),
    (
        _tok(r"\bgit\s+branch\s+-[dD]\b"),
        "Deletes a git branch — may lose commits not merged elsewhere.",
    ),
    (
        _tok(r"\bgit\s+stash\s+drop\b"),
        "Permanently discards a git stash entry.",
    ),
]

# Filter out intentional None-reason sentinels from the list.
_REVIEW_PATTERNS = [(p, r) for p, r in _REVIEW_PATTERNS if r is not None]


# ---------------------------------------------------------------------------
# Tier SAFE — known-safe action allowlist
# ---------------------------------------------------------------------------
# A command is SAFE only if it matches at least one of these AND passes both
# BLOCKED and REVIEW checks (i.e., it gets here only after those pass).

_SAFE_PATTERNS: list[re.Pattern[str]] = [
    # package manager install (without destructive force context)
    _tok(r"\b(pip|pip3)\s+install\b"),
    _tok(r"\bnpm\s+(install|ci|i)\b"),
    _tok(r"\byarn\s+(install|add)\b"),
    _tok(r"\bpnpm\s+(install|add)\b"),
    _tok(r"\bapt(-get)?\s+install\b"),
    _tok(r"\bapt(-get)?\s+update\b"),
    _tok(r"\bbrew\s+install\b"),
    _tok(r"\bbrew\s+update\b"),
    _tok(r"\bcargo\s+add\b"),
    _tok(r"\bcargo\s+build\b"),
    _tok(r"\bcargo\s+test\b"),
    _tok(r"\bgo\s+get\b"),
    _tok(r"\bgo\s+build\b"),
    _tok(r"\bgem\s+install\b"),
    _tok(r"\bconda\s+install\b"),
    _tok(r"\bconda\s+update\b"),
    # version / info checks
    _tok(r"--version\b"),
    _tok(r"\bwhich\b"),
    _tok(r"\btype\b"),
    _tok(r"\bwhereis\b"),
    # listing / inspection
    _tok(r"\bpip\s+(list|show|freeze)\b"),
    _tok(r"\bnpm\s+(list|ls|info|view|outdated)\b"),
    _tok(r"\byarn\s+(list|info)\b"),
    _tok(r"\bapt(-get)?\s+(list|show|search)\b"),
    _tok(r"\bcargo\s+(check|clippy|doc|search)\b"),
    _tok(r"\bgo\s+(list|doc|env|version)\b"),
    # read-only filesystem ops
    _tok(r"\bls\b"),
    _tok(r"\bcat\b"),
    _tok(r"\bhead\b"),
    _tok(r"\btail\b"),
    _tok(r"\bgrep\b"),
    _tok(r"\bfind\b(?![^|&;]*-delete)"),   # find without -delete is safe
    _tok(r"\bwc\b"),
    _tok(r"\bstat\b"),
    _tok(r"\bfile\b"),
    _tok(r"\becho\b"),
    _tok(r"^\s*(env|printenv)\b"),  # only if it's the command itself, not --env
    _tok(r"^\s*pwd\b"),
    # package manager update (non-destructive forms)
    _tok(r"\bpip\s+install\b[^|&;]*(--upgrade|-U)\b"),
    _tok(r"\bnpm\s+update\b"),
    _tok(r"\byarn\s+upgrade\b"),
    _tok(r"\bconda\s+update\b"),
    # virtual environment creation
    _tok(r"\bpython\b[^|&;]*-m\s+venv\b"),
    _tok(r"\bpython3\b[^|&;]*-m\s+venv\b"),
    _tok(r"\bconda\s+create\b"),
    _tok(r"\bvirtualenv\b"),
    # git read-only
    _tok(r"\bgit\s+(status|log|diff|show|fetch|clone|pull)\b"),
    _tok(r"\bgit\s+branch\b(?!\s+-[dD])"),  # list branches, not delete
]


# ---------------------------------------------------------------------------
# CommandRiskClassifier
# ---------------------------------------------------------------------------

class CommandRiskClassifier:
    """
    Classifies shell commands into SAFE, REVIEW, or BLOCKED tiers.

    Parameters
    ----------
    sandbox_root:
        Absolute path to the repository / workspace root.  Commands that
        attempt to write to paths outside this tree are BLOCKED.
        Defaults to the current working directory.

    Example
    -------
    >>> clf = CommandRiskClassifier(sandbox_root="/home/user/project")
    >>> result = clf.classify(AtomicStep(command="pip install requests"))
    >>> result.tier
    <RiskTier.SAFE: 'SAFE'>
    """

    def __init__(self, sandbox_root: Optional[str] = None) -> None:
        self.sandbox_root: str = os.path.abspath(sandbox_root or os.getcwd())

    # ------------------------------------------------------------------ public

    def classify(self, step: AtomicStep) -> ClassificationResult:
        """
        Run the three-pass classifier on *step*.

        Returns
        -------
        ClassificationResult
            Always returns a result — never raises.

        Notes
        -----
        Pass order:
          1. BLOCKED (hard denylist, out-of-scope path)
          2. REVIEW (pattern list, force-flag+verb, planner flag)
          3. SAFE (allowlist)
          4. Default → REVIEW (fail-safe for unknown commands)
        """
        command = step.command.strip()

        # --- Pass 1: BLOCKED -------------------------------------------------
        blocked = self._check_blocked(command)
        if blocked is not None:
            return blocked

        # --- Pass 2: REVIEW --------------------------------------------------
        review = self._check_review(command, step)
        if review is not None:
            return review

        # --- Pass 3: SAFE allowlist ------------------------------------------
        safe = self._check_safe(command)
        if safe is not None:
            return safe

        # --- Default: unknown command → REVIEW (fail-safe) -------------------
        return ClassificationResult(
            tier=RiskTier.REVIEW,
            reason=(
                f"Command {command!r} does not match any known-safe action type. "
                "Treating as REVIEW by default (fail-safe)."
            ),
            matched_pattern="<default-fail-safe>",
        )

    # --------------------------------------------------------------- private passes

    def _check_blocked(self, command: str) -> Optional[ClassificationResult]:
        """Return a BLOCKED result if any hard-denylist pattern fires."""
        for pattern, reason in _BLOCKED_PATTERNS:
            if pattern.search(command):
                return ClassificationResult(
                    tier=RiskTier.BLOCKED,
                    reason=reason,
                    matched_pattern=pattern.pattern,
                )

        # Out-of-scope path check
        scope_result = self._check_out_of_scope(command)
        if scope_result is not None:
            return scope_result

        return None

    def _check_review(
        self, command: str, step: AtomicStep
    ) -> Optional[ClassificationResult]:
        """Return a REVIEW result if a REVIEW pattern fires or step is planner-flagged."""

        # 1. Planner explicitly flagged as destructive
        if step.destructive:
            reason = step.risk_reason or (
                f"The planner flagged this step as destructive: {step.description!r}"
            )
            return ClassificationResult(
                tier=RiskTier.REVIEW,
                reason=reason,
                matched_pattern="<planner-destructive-flag>",
            )

        # 2. Scoped force-flag check — only escalate when verb is risky
        if _FORCE_FLAG_RE.search(command):
            verbs = _extract_all_verbs(command)
            for verb in verbs:
                if verb in _FORCE_ESCALATING_VERBS:
                    return ClassificationResult(
                        tier=RiskTier.REVIEW,
                        reason=(
                            f"Force flag used with high-risk verb {verb!r} — "
                            "this can bypass safety checks and cause irreversible changes."
                        ),
                        matched_pattern=f"force-flag+verb:{verb}",
                    )

        # 3. Explicit REVIEW pattern list
        for pattern, reason in _REVIEW_PATTERNS:
            if pattern.search(command):
                return ClassificationResult(
                    tier=RiskTier.REVIEW,
                    reason=reason,
                    matched_pattern=pattern.pattern,
                )

        return None

    def _check_safe(self, command: str) -> Optional[ClassificationResult]:
        """Return a SAFE result if any known-safe pattern matches."""
        for pattern in _SAFE_PATTERNS:
            if pattern.search(command):
                return ClassificationResult(
                    tier=RiskTier.SAFE,
                    reason="Matches known safe action type.",
                    matched_pattern=None,
                )
        return None

    def _check_out_of_scope(self, command: str) -> Optional[ClassificationResult]:
        """
        Heuristically detect absolute paths in write-context positions that
        fall outside ``sandbox_root``.

        This is a best-effort check — it catches the most common patterns
        (redirect operators, ``--output``, ``tee``) but cannot parse all
        possible shell constructs.
        """
        # Find all absolute-looking paths in the command.
        path_candidates = re.findall(r"(?<![\"'])/(?:[^|\s&;\"']+)", command)
        # Contexts that suggest a write: redirect, --output, -o flag, tee destination
        write_context_re = re.compile(
            r"(?:>|>>|--output[= ]|(?<!\w)-o\s+|tee\s+)\s*(/[^\s|&;\"']+)",
            re.IGNORECASE,
        )
        write_paths = write_context_re.findall(command)

        for raw_path in write_paths:
            try:
                abs_path = os.path.normpath(raw_path)
            except Exception:
                continue
            if not abs_path.startswith(self.sandbox_root):
                return ClassificationResult(
                    tier=RiskTier.BLOCKED,
                    reason=(
                        f"Command attempts to write to {raw_path!r}, which is outside "
                        f"the sandbox root {self.sandbox_root!r}."
                    ),
                    matched_pattern="<out-of-scope-write>",
                )
        return None
