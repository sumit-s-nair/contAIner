"""
Grok API helper — generates a single concise clarifying question
when System 1 intent confidence is below the threshold.
"""
from __future__ import annotations
import os
import json
import urllib.request
import urllib.error
from typing import Optional

from .config import GROK_API_BASE, GROK_MODEL, PROJECT_ROOT


def _load_api_key() -> str:
    """Load GROK_API_KEY from project root .env if not already in environment."""
    key = os.environ.get("GROK_API_KEY", "")
    if key:
        return key

    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROK_API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise EnvironmentError(
        "GROK_API_KEY not found in environment or .env file"
    )


_SYSTEM_PROMPT = (
    "You are a helpful assistant for a developer tool called contAIner. "
    "Your only job is to ask ONE short, specific clarifying question to help "
    "understand what the user wants to do with their development environment. "
    "Output only the question itself — no preamble, no explanation, no punctuation beyond the question mark."
)


def get_clarifying_question(
    user_prompt: str,
    top_intent: str,
    confidence: float,
) -> str:
    """
    Call Grok to generate a one-sentence clarifying question.

    Returns the question string, or a sensible fallback on error.
    """
    api_key = _load_api_key()

    user_msg = (
        f"The user said: \"{user_prompt}\"\n"
        f"Our model's best guess is intent '{top_intent}' "
        f"with only {confidence:.0%} confidence.\n"
        "What single clarifying question would best help determine their exact intent?"
    )

    payload = json.dumps({
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 80,
        "temperature": 0.4,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{GROK_API_BASE}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Grok API error {e.code}: {body}") from e
    except Exception as e:
        # Fallback — should not break the demo
        return "Could you clarify what you'd like to do with your environment?"
