"""
Flask bridge server — single file, streaming SSE pipeline.

Runs in WSL (pyg-pip conda env). Electron on Windows calls it via
http://localhost:5050 (WSL port-forwards to Windows automatically).

No CLI args — all paths come from config.py.
"""
from __future__ import annotations

import json
import os
import sys
import time

# ── Allow imports from project root ──────────────────────────────────────────
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
_ELECTRON_DIR = os.path.dirname(_BRIDGE_DIR)
_PROJECT_ROOT = os.path.dirname(_ELECTRON_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from electron_app.bridge.config import (
    SYSTEM1_MODEL_DIR,
    SYSTEM2_MODEL_PATH,
    MCP_URL,
    MCP_TIMEOUT_SECONDS,
    BRIDGE_HOST,
    BRIDGE_PORT,
    CONFIDENCE_THRESHOLD,
)
from electron_app.bridge.system1 import System1Predictor
from electron_app.bridge.grok import get_clarifying_question

# System 2 imports from existing project code
from src.system2_command_generation.models import CommandGenerationModel
from src.system2_command_generation.data_preprocessing import (
    MCPClient,
    format_input,
    parse_model_output,
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow Electron renderer to call across origins

# ── Global model handles (loaded once at startup) ─────────────────────────────
_system1: System1Predictor | None = None
_system2: CommandGenerationModel | None = None
_mcp: MCPClient | None = None


# ─────────────────────────────────────────────────── helpers ──────────────────

RUNTIME_TO_TOOL = {
    "python": "pip", "python3": "pip",
    "node": "npm", "nodejs": "npm",
    "rust": "cargo", "go": "go", "golang": "go",
    "java": "maven", "ruby": "gem",
}
SYSTEM_TOOL_BY_OS = {"linux": "apt", "macos": "brew", "windows": "conda"}
INTENT_TO_OPERATION = {
    "install_package":    "install",
    "install_runtime":    "install",
    "update_package":     "update",
    "update_runtime":     "update",
    "remove_package":     "uninstall",
    "remove_runtime":     "uninstall",
    "list_dependencies":  "list",
    "check_version":      "show",
    "check_installed":    "show",
    "create_isolation":   "create",
}


def _tool_from_intent(intent: str, entities: dict, os_hint: str = "linux") -> str:
    runtime = (entities.get("runtime") or "").strip().lower()
    package = (entities.get("package") or "").strip().lower()

    if intent.endswith("_runtime"):
        return SYSTEM_TOOL_BY_OS.get(os_hint, "apt")
    if runtime:
        return RUNTIME_TO_TOOL.get(runtime, runtime)
    if package in RUNTIME_TO_TOOL:
        return RUNTIME_TO_TOOL[package]
    return ""


def _compact_doc_chunk(chunk: dict | None) -> dict | None:
    if not chunk:
        return None
    compact = dict(chunk)
    syntax = str(compact.get("command_syntax") or "").replace("\n", " ").strip()
    compact["command_syntax"] = syntax[:220]
    examples = compact.get("examples") or []
    if examples:
        compact["examples"] = [str(examples[0]).replace("\n", " ").strip()[:180]]
    flags = compact.get("key_flags")
    if isinstance(flags, list):
        compact["key_flags"] = flags[:4]
    notes = str(compact.get("os_specific_notes") or "").replace("\n", " ").strip()
    compact["os_specific_notes"] = notes[:200]
    return compact


def _doc_chunk_has_context(chunk: dict | None) -> bool:
    if not chunk:
        return False
    return bool(
        str(chunk.get("command_syntax") or "").strip()
        or chunk.get("examples")
        or chunk.get("key_flags")
    )


def _sse(event: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(event)}\n\n"


# ─────────────────────────────────────────────────── endpoints ────────────────

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "system1_loaded": _system1 is not None,
        "system2_loaded": _system2 is not None,
        "mcp_ok": (_mcp is not None and _mcp.is_available()),
        "mcp_url": MCP_URL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    })


@app.post("/run")
def run_pipeline():
    """
    Streaming SSE endpoint.
    Body: { "prompt": str, "os_hint": str?, "shell_type": str?, "clarify_answer": str? }

    Yields SSE events for each pipeline stage so the Electron UI can
    show live "thinking" cards.
    """
    body = request.get_json(force=True) or {}
    prompt: str = (body.get("prompt") or "").strip()
    os_hint: str = (body.get("os_hint") or "linux").strip().lower()
    shell_type: str = (body.get("shell_type") or "bash").strip()
    clarify_answer: str | None = body.get("clarify_answer")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # If user answered a clarifying question, append the answer to the prompt
    # so System 1 has richer context on the second pass.
    effective_prompt = prompt
    if clarify_answer:
        effective_prompt = f"{prompt}. {clarify_answer}"

    def generate():
        # ── Stage 1: System 1 ─────────────────────────────────────────────
        yield _sse({"stage": "system1", "status": "running"})
        try:
            s1_result = _system1.predict(effective_prompt)
        except Exception as e:
            yield _sse({"stage": "error", "status": "done", "message": f"System 1 failed: {e}"})
            return

        intent     = s1_result["intent"]
        confidence = s1_result["confidence"]
        entities   = s1_result["entities"]

        yield _sse({
            "stage": "system1", "status": "done",
            "intent": intent, "confidence": confidence,
            "entities": entities,
            "probabilities": s1_result["probabilities"],
        })

        # ── Stage 2: Confidence gate → Grok clarification ─────────────────
        if confidence < CONFIDENCE_THRESHOLD and not clarify_answer:
            yield _sse({"stage": "clarify", "status": "running"})
            try:
                question = get_clarifying_question(prompt, intent, confidence)
            except Exception as e:
                question = "Could you give me more detail about what you'd like to do?"
            yield _sse({"stage": "clarify", "status": "needed", "question": question})
            # Stop here — renderer will ask the user and POST /run again with clarify_answer
            return

        # ── Stage 3: MCP doc fetch ────────────────────────────────────────
        tool      = _tool_from_intent(intent, entities, os_hint)
        operation = INTENT_TO_OPERATION.get(intent, "install")

        if tool:
            yield _sse({
                "stage": "mcp", "status": "running",
                "tool": tool, "operation": operation,
            })
            try:
                doc_chunk = _mcp.fetch_docs(
                    tool=tool, operation=operation,
                    package=entities.get("package") or entities.get("runtime") or "",
                    os_hint=os_hint,
                    runtime=entities.get("runtime") or "",
                    version=entities.get("version") or "",
                )
                doc_chunk = _compact_doc_chunk(doc_chunk)
            except Exception as e:
                doc_chunk = None

            has_docs = _doc_chunk_has_context(doc_chunk)
            yield _sse({
                "stage": "mcp", "status": "done",
                "tool": tool, "has_docs": has_docs,
                "doc_chunk": doc_chunk if has_docs else None,
            })
        else:
            yield _sse({"stage": "mcp", "status": "skipped", "reason": "could not infer tool"})
            has_docs = False
            doc_chunk = None

        # ── Stage 4: System 2 generation ─────────────────────────────────
        yield _sse({"stage": "system2", "status": "running"})

        row = {
            "intent_type": intent,
            "entities": entities,
            "os": os_hint,
            "shell": shell_type,
        }
        prompt_text = format_input(
            row,
            compact=True,
            doc_chunk=doc_chunk if has_docs else None,
        )

        try:
            # Check if checkpoint was trained with MCP before injecting docs
            checkpoint_trained_with_mcp = getattr(_system2.config, "use_mcp", False)
            if not checkpoint_trained_with_mcp and has_docs:
                # Re-format without docs to avoid distribution mismatch
                prompt_text = format_input(row, compact=True, doc_chunk=None)

            raw_output = _system2.generate_text(
                prompt_text,
                max_length=256,
                num_beams=1,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.3,
            )
        except Exception as e:
            yield _sse({"stage": "error", "status": "done", "message": f"System 2 failed: {e}"})
            return

        command_plan, parse_error = parse_model_output(raw_output)
        yield _sse({
            "stage": "system2", "status": "done",
            "raw": raw_output,
            "command_plan": command_plan,
            "parse_error": parse_error,
        })

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────── startup ──────────────────

def _load_models():
    global _system1, _system2, _mcp

    print(f"[bridge] Loading System 1 from: {SYSTEM1_MODEL_DIR}")
    _system1 = System1Predictor(SYSTEM1_MODEL_DIR)
    print("[bridge] System 1 loaded ✓")

    print(f"[bridge] Loading System 2 from: {SYSTEM2_MODEL_PATH}")
    _system2 = CommandGenerationModel.load(SYSTEM2_MODEL_PATH)
    print("[bridge] System 2 loaded ✓")

    _mcp = MCPClient(url=MCP_URL, timeout=MCP_TIMEOUT_SECONDS)
    mcp_ok = _mcp.is_available()
    print(f"[bridge] MCP server at {MCP_URL}: {'ok' if mcp_ok else 'OFFLINE'}")


if __name__ == "__main__":
    _load_models()
    print(f"[bridge] Starting on http://{BRIDGE_HOST}:{BRIDGE_PORT}")
    # use_reloader=False prevents models loading twice
    app.run(host=BRIDGE_HOST, port=BRIDGE_PORT, threaded=True, use_reloader=False)
