"""
Grok API helper — two functions:
  1. get_clarifying_question()   — low-confidence fallback (existing)
  2. generate_command()          — replaces System 2 until training is done,
                                   accepts optional MCP doc context
"""
from __future__ import annotations
import os
import json
import urllib.request
import urllib.error
from typing import Optional

from .config import GROK_API_BASE, GROK_MODEL, PROJECT_ROOT


def _load_api_key() -> str:
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
    raise EnvironmentError("GROK_API_KEY not found in environment or .env file")


def _call_grok(messages: list, max_tokens: int = 120, temperature: float = 0.4) -> str:
    api_key = _load_api_key()
    payload = json.dumps({
        "model": GROK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{GROK_API_BASE}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "contAIner-bridge/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


# ── Clarifying question ───────────────────────────────────────────────────────

_CLARIFY_SYSTEM = (
    "You are a helpful assistant for a developer tool called contAIner. "
    "Your only job is to ask ONE short, specific clarifying question to help "
    "understand what the user wants to do with their development environment. "
    "Output only the question itself — no preamble, no explanation, no punctuation beyond the question mark."
)


def get_clarifying_question(user_prompt: str, top_intent: str, confidence: float) -> str:
    try:
        return _call_grok([
            {"role": "system", "content": _CLARIFY_SYSTEM},
            {"role": "user", "content": (
                f"The user said: \"{user_prompt}\"\n"
                f"Our model's best guess is intent '{top_intent}' with only {confidence:.0%} confidence.\n"
                "What single clarifying question would best help determine their exact intent?"
            )},
        ], max_tokens=80, temperature=0.4)
    except Exception:
        return "Could you clarify what you'd like to do with your environment?"


# ── Command generation (replaces System 2 while model trains) ─────────────────

_CMD_SYSTEM = """You are contAIner, an AI-powered developer environment manager.

The user will give you their original request plus any structured data (intent, pre-extracted entities, OS).
Your job is to produce the correct real shell command AND extract key entities from the user's raw request.

Output ONLY this JSON (no markdown fences, no extra text):
{
  "entities": {
    "package": "exact package name if mentioned, else null",
    "runtime": "runtime/language if mentioned (python, node, rust…), else null",
    "version": "version number if mentioned, else null",
    "virtual_env": "virtualenv/conda env name if mentioned, else null"
  },
  "explanation": "One sentence: what the command does and why.",
  "steps": [
    "Step 1 description",
    "Step 2 description"
  ],
  "command": "exact shell commands here",
  "shell": "bash"
}

Critical rules:
- Set entity fields to null if not mentioned — do not guess.
- ALWAYS use the raw request to identify the exact package/runtime name.
- NEVER use placeholders like <package_name>, <your_package>, [package], etc.
- If the user says \"install python\" the command MUST install python specifically.
- Use the most standard/recommended approach for the given OS and shell.
- explanation: one plain sentence, no markdown.
- steps: 2-4 plain-text action descriptions.
- command: real executable shell commands only.
- Do NOT wrap output in markdown code fences.
"""


def generate_command(
    intent: str,
    entities: dict,
    os_hint: str,
    shell_type: str,
    user_prompt: str = "",
    doc_chunk: Optional[dict] = None,
) -> dict:
    """
    Ask Grok to generate the shell command for the pipeline.

    Returns a dict: { explanation, steps, command, shell }
    Falls back gracefully on parse errors.
    """
    entity_str = ", ".join(f"{k}={v}" for k, v in (entities or {}).items()) or "none"

    doc_section = ""
    if doc_chunk:
        syntax = doc_chunk.get("command_syntax") or ""
        examples = doc_chunk.get("examples") or []
        flags = doc_chunk.get("key_flags") or []
        notes = doc_chunk.get("os_specific_notes") or ""
        parts = []
        if syntax:
            parts.append(f"Syntax: {str(syntax)[:300]}")
        if examples:
            parts.append(f"Example: {str(examples[0])[:200]}")
        if flags:
            parts.append(f"Key flags: {', '.join(str(f) for f in flags[:4])}")
        if notes:
            parts.append(f"Notes: {str(notes)[:200]}")
        if parts:
            doc_section = "\n\nDocumentation:\n" + "\n".join(parts)

    user_msg = (
        f"User request: \"{user_prompt}\"\n"
        f"Intent: {intent}\n"
        f"Entities extracted: {entity_str}\n"
        f"OS: {os_hint}\n"
        f"Shell: {shell_type}"
        f"{doc_section}"
    )

    try:
        raw = _call_grok([
            {"role": "system", "content": _CMD_SYSTEM},
            {"role": "user",   "content": user_msg},
        ], max_tokens=512, temperature=0.3)

        # Try to parse JSON — strip any accidental fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        parsed = json.loads(cleaned)
        raw_entities = parsed.get("entities") or {}
        # Filter out null/empty values
        extracted = {k: v for k, v in raw_entities.items() if v and str(v).lower() not in ("null", "none", "")}
        return {
            "entities": extracted,
            "explanation": str(parsed.get("explanation", "")),
            "steps": [str(s) for s in (parsed.get("steps") or [])],
            "command": str(parsed.get("command", "")),
            "shell": str(parsed.get("shell", shell_type)),
        }
    except Exception as e:
        return {
            "entities": {},
            "explanation": "",
            "steps": [],
            "command": f"# Error: {e}",
            "shell": shell_type,
        }
