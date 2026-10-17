"""Documentation context for System 3 (MCP DocChunk format).

Source of truth: the ``--help`` text of the exact tools the sandbox runs
(``python -m pip install --help``, ``python -m venv --help``,
``python -m pytest --help``), captured from the sandbox image by
``scripts/snapshot_cli_help.py`` into ``datasets/system3-slot/docs/``.

Why not the web MCP snapshot (datasets/command-dataset/data/mcp_snapshot.json):
its pip chunks had flags paired with other flags' descriptions (compression
reassembly bug, fixed in src/mcp/reassemble.py in Stage 3A) and lacked the
flags recovery needs (--only-binary, --prefer-binary, --no-build-isolation).
The help text is version-exact (the sandbox pip is 24.0), offline and
deterministic.  A live MCP server can still be used via ``mcp_url``; its
chunk then goes through the same flag retrieval.

Per step, ``DocProvider.for_step`` returns a DocChunk dict whose ``key_flags``
are the core flags of the operation plus the flags most relevant to the step
target and the previous failure (lexical retrieval with light query
expansion).  The model still decides what to use.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.system2_planner.models import ActionType

log = logging.getLogger("system3.docs")

DEFAULT_DOCS_DIR = Path("datasets/system3-slot/docs")

# (tool, operation) per action type, and the help file holding its docs.
_ACTION_DOCS: Dict[ActionType, List[Tuple[str, str, str]]] = {
    ActionType.INSTALL: [("pip", "install", "pip_install.help.txt")],
    ActionType.ISOLATE: [("venv", "create", "venv.help.txt")],
    ActionType.CHECK: [("pytest", "run", "pytest.help.txt"), ("pip", "install", "pip_install.help.txt")],
}

_CORE_FLAGS: Dict[Tuple[str, str], List[str]] = {
    ("pip", "install"): ["-r", "-e", "-c", "-U", "--no-deps"],
    ("venv", "create"): ["--clear", "--upgrade-deps", "--without-pip"],
    ("pytest", "run"): ["-x", "-q", "-k"],
}

_SYNTAX: Dict[Tuple[str, str], str] = {
    ("pip", "install"): (
        "python -m pip install [options] <requirement specifier> ... | "
        "python -m pip install [options] -r <requirements file> ... | "
        "python -m pip install [options] [-e] <local project path> ..."
    ),
    ("venv", "create"): "python -m venv [options] ENV_DIR",
    ("pytest", "run"): "python -m pytest [options] [file_or_dir ...]",
}

_EXAMPLES: Dict[Tuple[str, str], List[str]] = {
    ("pip", "install"): [
        "python -m pip install -r requirements.txt",
        "python -m pip install .",
        "python -m pip install 'requests>=2.0'",
    ],
    ("venv", "create"): ["python -m venv .venv"],
    ("pytest", "run"): ["python -m pytest tests"],
}

# Query expansion: words in a failure that point at doc vocabulary.
_EXPANSIONS = {
    "wheel": ["binary", "build", "source"],
    "building": ["build", "binary", "source"],
    "build": ["isolation", "binary", "source"],
    "metadata": ["build", "isolation"],
    "conflict": ["constraint", "dependencies", "deps"],
    "conflicting": ["constraint", "dependencies", "deps"],
    "resolutionimpossible": ["constraint", "dependencies", "deps"],
    "python": ["python", "version", "requires"],
    "requires": ["python", "version"],
    "satisfies": ["version", "index", "pre", "release"],
    "version": ["version", "pre", "release", "upgrade"],
    "editable": ["editable", "project"],
    "permission": ["user", "target"],
    "timeout": ["timeout", "retries"],
    "hash": ["hashes", "require"],
}

_STOP = set("""a an the of to in for on and or is be by with from this that it if
not can use used using will default option options when which only all any are as
at its than then into per""".split())

MAX_DOC_CHARS = 1400
MAX_RETRIEVED_FLAGS = 6


@dataclass
class Flag:
    names: List[str]
    arg: str
    description: str

    @property
    def display(self) -> str:
        return ", ".join(self.names) + (f" {self.arg}" if self.arg else "")


_OPT_LINE = re.compile(r"^ {1,6}(-[^\s].*?)(?: {2,}(\S.*))?$")


def parse_help_flags(text: str) -> List[Flag]:
    """Parse optparse/argparse ``--help`` output into flags.

    Handles descriptions that start on the option line or on the next line,
    and multi-line continuation.
    """
    flags: List[Flag] = []
    cur: Optional[Flag] = None
    for line in text.splitlines():
        m = _OPT_LINE.match(line)
        if m and not line.startswith(" " * 8):
            spec, desc = m.group(1).strip(), (m.group(2) or "").strip()
            names, arg = [], ""
            for part in [p.strip() for p in spec.split(",")]:
                bits = part.split(None, 1)
                if bits and bits[0].startswith("-"):
                    names.append(bits[0].split("=")[0])
                    if len(bits) > 1 and not arg:
                        arg = bits[1]
                    elif "=" in bits[0] and not arg:
                        arg = bits[0].split("=", 1)[1]
            if names:
                cur = Flag(names=names, arg=arg, description=desc)
                flags.append(cur)
                continue
        if cur is not None and line.startswith(" " * 8) and line.strip():
            cur.description = (cur.description + " " + line.strip()).strip()
        elif not line.strip() or (line and not line.startswith(" ")):
            cur = None
    for f in flags:
        f.description = re.sub(r"\s+", " ", f.description)
    return flags


def _tokens(text: str) -> List[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 1]
    return [t for t in toks if t not in _STOP]


def _query_terms(*texts: str) -> Dict[str, float]:
    terms: Dict[str, float] = {}
    for t in _tokens(" ".join(x for x in texts if x)):
        terms[t] = terms.get(t, 0.0) + 1.0
        for e in _EXPANSIONS.get(t, []):
            terms[e] = terms.get(e, 0.0) + 0.5
    return terms


def _score(flag: Flag, terms: Dict[str, float]) -> float:
    words = set(_tokens(" ".join(flag.names) + " " + flag.arg + " " + flag.description))
    return sum(w for t, w in terms.items() if t in words)


def _first_sentences(desc: str, max_chars: int = 220) -> str:
    out = ""
    for sent in re.split(r"(?<=[.)])\s+", desc):
        if len(out) + len(sent) + 1 > max_chars and out:
            break
        out = (out + " " + sent).strip()
    return out[:max_chars]


@dataclass
class DocProvider:
    docs_dir: Path = DEFAULT_DOCS_DIR
    mcp_url: Optional[str] = None
    timeout: float = 10.0
    _cache: Dict[str, List[Flag]] = field(default_factory=dict)

    def _flags_from_help(self, filename: str) -> List[Flag]:
        if filename not in self._cache:
            p = Path(self.docs_dir) / filename
            self._cache[filename] = parse_help_flags(p.read_text(encoding="utf-8")) if p.exists() else []
            if not p.exists():
                log.warning("System 3 docs: %s missing (run scripts/snapshot_cli_help.py)", p)
        return self._cache[filename]

    def _flags_from_mcp(self, tool: str, operation: str, package: str) -> Optional[Tuple[List[Flag], str, List[str]]]:
        if not self.mcp_url:
            return None
        body = json.dumps({"tool": tool, "operation": operation, "package": package, "os": "linux"}).encode()
        req = urllib.request.Request(self.mcp_url.rstrip("/") + "/fetch_docs", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                chunk = json.loads(r.read())
        except Exception as exc:  # fall back to help text
            log.warning("MCP fetch failed (%s); using --help docs", exc)
            return None
        if chunk.get("error") and not chunk.get("key_flags"):
            return None
        flags = []
        for f in chunk.get("key_flags") or []:
            names = [n.strip() for n in re.split(r",\s*", re.sub(r"[¶<].*$", "", f.get("flag", ""))) if n.strip()]
            if names:
                flags.append(Flag(names=names, arg="", description=f.get("description", "")))
        return flags, chunk.get("command_syntax", ""), list(chunk.get("examples") or [])

    def for_step(self, action_type: ActionType, target: str = "", description: str = "",
                 stderr: str = "") -> List[Dict[str, Any]]:
        """DocChunk dicts (tool, operation, command_syntax, key_flags, examples)."""
        chunks = []
        for tool, op, helpfile in _ACTION_DOCS.get(action_type, []):
            source = f"{helpfile} (sandbox --help)"
            syntax, examples = _SYNTAX[(tool, op)], _EXAMPLES[(tool, op)]
            mcp = self._flags_from_mcp(tool, op, target) if tool == "pip" else None
            if mcp is not None and mcp[0]:
                flags, syntax, examples = mcp[0], mcp[1] or syntax, mcp[2] or examples
                source = "mcp"
            else:
                flags = self._flags_from_help(helpfile)
            chunks.append({
                "tool": tool, "operation": op, "command_syntax": syntax,
                "key_flags": self._select(flags, (tool, op), target, description, stderr),
                "examples": examples[:3], "source": source,
            })
        return chunks

    def _select(self, flags: List[Flag], key: Tuple[str, str], target: str, description: str,
                stderr: str) -> List[Dict[str, str]]:
        by_name = {n: f for f in flags for n in f.names}
        chosen: List[Flag] = []
        for n in _CORE_FLAGS.get(key, []):
            f = by_name.get(n)
            if f is not None and f not in chosen:
                chosen.append(f)
        terms = _query_terms(target, description, stderr)
        ranked = sorted((f for f in flags if f not in chosen), key=lambda f: -_score(f, terms))
        extra = [f for f in ranked if _score(f, terms) >= 1.0][:MAX_RETRIEVED_FLAGS]
        out, total = [], 0
        for f in chosen + extra:
            d = {"flag": f.display, "description": _first_sentences(f.description)}
            total += len(d["flag"]) + len(d["description"]) + 4
            if total > MAX_DOC_CHARS and out:
                break
            out.append(d)
        return out


def render_docs(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        lines = [f"### {c['tool']} {c['operation']}", f"syntax: {c['command_syntax']}", "flags:"]
        lines += [f"  {f['flag']}: {f['description']}" for f in c["key_flags"]]
        if c.get("examples"):
            lines.append("examples: " + " ; ".join(c["examples"]))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
