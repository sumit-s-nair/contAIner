"""
Hardcoded configuration for the contAIner pipeline bridge server.
Edit these paths directly instead of using CLI arguments.
"""
import os

# ── Project root (WSL: /home/sumit/contAIner) ─────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── System 1: Intent Classifier ───────────────────────────────────────────────
SYSTEM1_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "intent_classifier", "final_model_v2"
)

# ── System 2: Command Generation ──────────────────────────────────────────────
# Update this path once your MCP-enabled training run finishes.
SYSTEM2_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "system2_command_generation",
    "qwen2_5_coder_1_5b_20260422_232926",
    "checkpoint-400",
)

# ── MCP Documentation Server ──────────────────────────────────────────────────
# Running in WSL on port 11435 — accessible from WSL Python bridge as localhost.
MCP_URL = "http://localhost:11435"
MCP_TIMEOUT_SECONDS = 10

# ── Bridge HTTP Server ─────────────────────────────────────────────────────────
# Electron (Windows) calls this bridge running inside WSL.
# Electron should hit: http://localhost:5050
BRIDGE_HOST = "0.0.0.0"
BRIDGE_PORT = 5050

# ── Grok API ──────────────────────────────────────────────────────────────────
# Read from .env at project root — never hardcoded here.
# NOTE: gsk_ prefix = Groq API key (groq.com), not xAI
GROK_API_BASE = "https://api.groq.com/openai/v1"
GROK_MODEL    = "llama-3.3-70b-versatile"

# ── Demo thresholds ───────────────────────────────────────────────────────────
# Intent confidence below this will trigger a Grok clarifying question.
CONFIDENCE_THRESHOLD = 0.60
