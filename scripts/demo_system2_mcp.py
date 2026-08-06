#!/usr/bin/env python3
"""Interactive System 2 + MCP demo runner.

This script is designed for quick demos:
1) Load a trained System 2 checkpoint/final model.
2) Confirm MCP docs server connectivity.
3) Prompt for CanonicalIntent-style input fields.
4) Generate and print a CommandPlan prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

# Allow direct execution: python scripts/demo_system2_mcp.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.system2_command_generation.config import DEFAULT_SHELL, INTENT_TYPES, OS_TYPES, SHELL_TYPES
from src.system2_command_generation.data_preprocessing import MCPClient, format_input, parse_model_output
from src.system2_command_generation.models import CommandGenerationModel


DEFAULT_MODEL_PATH = (
    "./outputs/system2_command_generation/"
    "qwen2_5_coder_1_5b_20260421_162634/checkpoint-842"
)

RUNTIME_TO_TOOL = {
    "python": "pip",
    "node": "npm",
    "nodejs": "npm",
    "rust": "cargo",
    "go": "go",
    "golang": "go",
    "java": "maven",
    "ruby": "gem",
}

KNOWN_RUNTIMES = {
    "python",
    "python3",
    "node",
    "nodejs",
    "java",
    "go",
    "golang",
    "rust",
    "ruby",
    "dotnet",
    "php",
}

SYSTEM_TOOL_BY_OS = {
    "linux": "apt",
    "macos": "brew",
    "windows": "conda",
}

INTENT_TO_OPERATION = {
    "install_package": "install",
    "install_runtime": "install",
    "update_package": "update",
    "update_runtime": "update",
    "remove_package": "uninstall",
    "remove_runtime": "uninstall",
    "list_packages": "list",
    "check_version": "show",
}


def _prompt(
    label: str,
    default: Optional[str] = None,
    allowed: Optional[list[str]] = None,
    allow_empty: bool = True,
) -> str:
    while True:
        hint = f" [{default}]" if default else ""
        value = input(f"{label}{hint}: ").strip()

        if not value and default is not None:
            value = default

        if not value and allow_empty:
            return ""

        if allowed and value not in allowed:
            print(f"  Invalid value. Choose one of: {', '.join(allowed)}")
            continue

        return value


def _normalize_nullable(text: str) -> Optional[str]:
    cleaned = text.strip()
    return cleaned if cleaned else None


def _tool_from_inputs(
    tool_override: Optional[str],
    runtime: Optional[str],
    package: Optional[str],
    intent_type: str,
    os_hint: str,
) -> str:
    if tool_override:
        return tool_override.lower()

    if intent_type.endswith("_runtime"):
        return SYSTEM_TOOL_BY_OS.get(os_hint.lower(), "")

    runtime_name = (runtime or "").strip().lower()
    if runtime_name:
        return RUNTIME_TO_TOOL.get(runtime_name, runtime_name)

    package_name = (package or "").strip().lower()
    if package_name in RUNTIME_TO_TOOL:
        return RUNTIME_TO_TOOL[package_name]

    return ""


def _version_guardrail(version: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if version is None:
        return None, None

    raw = version.strip()
    if not raw:
        return None, None

    normalized = raw
    if any(ch.isdigit() for ch in raw):
        normalized = raw.replace("o", "0").replace("O", "0")

    if normalized != raw:
        return normalized, f"Normalized version '{raw}' to '{normalized}'."

    if re.fullmatch(r"[0-9]+(\.[0-9]+)*", normalized):
        return normalized, None

    return normalized, (
        "Version format looks unusual. Prefer numeric forms like 3.10, 20.04, or leave blank."
    )


def _apply_input_guardrails(canonical_intent: Dict[str, Any]) -> list[str]:
    notes: list[str] = []
    entities = canonical_intent["entities"]
    intent_type = canonical_intent["intent_type"]

    runtime = entities.get("runtime")
    package = entities.get("package")
    version = entities.get("version")

    normalized_version, version_note = _version_guardrail(version)
    entities["version"] = normalized_version
    if version_note:
        notes.append(version_note)

    runtime_name = (runtime or "").strip().lower()
    package_name = (package or "").strip().lower()

    if intent_type.endswith("_runtime") and runtime_name and package_name:
        entities["package"] = None
        package_name = ""
        notes.append("Runtime intent detected; clearing package field to avoid mixed intent input.")

    if intent_type == "install_runtime" and not runtime_name and package_name in KNOWN_RUNTIMES:
        entities["runtime"] = package_name
        entities["package"] = None
        notes.append(
            "Detected runtime in package field; moved it to runtime for install_runtime intent."
        )

    if intent_type == "install_package" and not runtime_name and package_name in KNOWN_RUNTIMES:
        notes.append(
            "This looks like a runtime install. Better prompt: intent_type=install_runtime, runtime=<value>, package blank."
        )

    if intent_type.endswith("_package") and not entities.get("package"):
        notes.append("Package is empty for a package intent; generation quality may be poor.")

    return notes


def _collect_demo_input() -> Dict[str, Any]:
    print("\nEnter test input details (press Enter to accept defaults).")
    print(f"Intent options: {', '.join(INTENT_TYPES)}")

    intent_type = _prompt("intent_type", default="install_package", allowed=INTENT_TYPES, allow_empty=False)
    os_hint = _prompt("os_hint", default="linux", allowed=OS_TYPES, allow_empty=False)
    shell_default = DEFAULT_SHELL.get(os_hint, "bash")
    shell_type = _prompt("shell_type", default=shell_default, allowed=SHELL_TYPES, allow_empty=False)
    package_default = None if intent_type.endswith("_runtime") else "git"
    package = _prompt("package (blank allowed)", default=package_default)
    runtime = _prompt("runtime (blank allowed)")
    version = _prompt("version (blank allowed)")
    tool_override = _prompt("tool override for MCP (blank=auto)")

    entities = {
        "runtime": _normalize_nullable(runtime),
        "package": _normalize_nullable(package),
        "version": _normalize_nullable(version),
    }

    canonical_intent = {
        "intent_type": intent_type,
        "entities": entities,
        "scope": "user",
        "os_hint": os_hint,
        "shell_type": shell_type,
        "confidence": 0.95,
        "missing_fields": [],
        "needs_clarification": False,
        "clarification_question": None,
    }

    return {
        "canonical_intent": canonical_intent,
        "tool_override": _normalize_nullable(tool_override),
    }


def _build_row(canonical_intent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent_type": canonical_intent["intent_type"],
        "entities": canonical_intent["entities"],
        "os": canonical_intent.get("os_hint") or "linux",
        "shell": canonical_intent["shell_type"],
    }


def _compact_doc_chunk(doc_chunk: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc_chunk or doc_chunk.get("error"):
        return doc_chunk

    compact = dict(doc_chunk)

    syntax = str(compact.get("command_syntax") or "").replace("\n", " ").strip()
    compact["command_syntax"] = syntax[:220]

    examples = compact.get("examples") or []
    if examples:
        first = str(examples[0]).replace("\n", " ").strip()
        compact["examples"] = [first[:180]]

    key_flags = compact.get("key_flags")
    if isinstance(key_flags, list):
        compact["key_flags"] = key_flags[:4]

    notes = str(compact.get("os_specific_notes") or "").replace("\n", " ").strip()
    compact["os_specific_notes"] = notes[:200]

    return compact


def _doc_chunk_has_context(doc_chunk: Optional[Dict[str, Any]]) -> bool:
    if not doc_chunk:
        return False
    syntax = str(doc_chunk.get("command_syntax") or "").strip()
    examples = doc_chunk.get("examples") or []
    key_flags = doc_chunk.get("key_flags") or []
    return bool(syntax or examples or key_flags)


def _input_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    intent_type = args.intent_type or "install_package"
    os_hint = args.os_hint or "linux"
    shell_type = args.shell_type or DEFAULT_SHELL.get(os_hint, "bash")

    entities = {
        "runtime": _normalize_nullable(args.runtime or ""),
        "package": _normalize_nullable(args.package or ""),
        "version": _normalize_nullable(args.version or ""),
    }

    canonical_intent = {
        "intent_type": intent_type,
        "entities": entities,
        "scope": "user",
        "os_hint": os_hint,
        "shell_type": shell_type,
        "confidence": 0.95,
        "missing_fields": [],
        "needs_clarification": False,
        "clarification_question": None,
    }

    return {
        "canonical_intent": canonical_intent,
        "tool_override": _normalize_nullable(args.tool_override or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive System2 command generation demo")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to checkpoint/final model")
    parser.add_argument("--mcp-url", default="http://localhost:11435", help="MCP server base URL")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam size for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max-output-tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                        help="Token repetition penalty (>1.0 reduces loops; 1.0 = disabled)")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP doc attachment")
    parser.add_argument("--single-run", action="store_true", help="Run one non-interactive test using CLI fields")
    parser.add_argument("--intent-type", choices=INTENT_TYPES, help="Intent type for --single-run")
    parser.add_argument("--os-hint", choices=OS_TYPES, help="OS hint for --single-run")
    parser.add_argument("--shell-type", choices=SHELL_TYPES, help="Shell type for --single-run")
    parser.add_argument("--package", help="Package entity for --single-run")
    parser.add_argument("--runtime", help="Runtime entity for --single-run")
    parser.add_argument("--version", help="Version entity for --single-run")
    parser.add_argument("--tool-override", help="Tool override for MCP fetch")
    args = parser.parse_args()

    print(f"\nLoading model from: {args.model_path}")
    model = CommandGenerationModel.load(args.model_path)

    # Detect whether the checkpoint was trained with MCP doc enrichment.
    # If not, injecting the <docs> template block produces unseen noise and
    # causes the model to echo back doc fragments instead of JSON.
    checkpoint_trained_with_mcp = getattr(model.config, "use_mcp", False)

    mcp_client = MCPClient(url=args.mcp_url)
    mcp_available = False
    if not args.no_mcp:
        mcp_available = mcp_client.is_available()
        print(f"MCP health ({args.mcp_url}): {'ok' if mcp_available else 'unreachable'}")
        if not mcp_available:
            print("Continuing without MCP docs (fallback prompt).")
        elif not checkpoint_trained_with_mcp:
            print(
                "Warning: checkpoint was NOT trained with MCP (use_mcp=False in config).\n"
                "         MCP doc injection will be skipped to avoid prompt-distribution mismatch.\n"
                "         Use --no-mcp to silence this warning, or train a new MCP-enabled run."
            )
            mcp_available = False

    while True:
        user_input = _input_from_args(args) if args.single_run else _collect_demo_input()
        canonical_intent = user_input["canonical_intent"]
        input_notes = _apply_input_guardrails(canonical_intent)
        row = _build_row(canonical_intent)

        if input_notes:
            print("\nInput notes:")
            for note in input_notes:
                print(f"  - {note}")

        doc_chunk = None
        if mcp_available and not args.no_mcp:
            entities = canonical_intent["entities"]
            tool = _tool_from_inputs(
                user_input["tool_override"],
                entities.get("runtime"),
                entities.get("package"),
                canonical_intent["intent_type"],
                canonical_intent.get("os_hint") or "linux",
            )
            operation = INTENT_TO_OPERATION.get(canonical_intent["intent_type"], "install")
            if tool:
                print(f"\nMCP lookup: tool={tool}, operation={operation}")
                lookup_package = entities.get("package") or ""
                if canonical_intent["intent_type"].endswith("_runtime") and not lookup_package:
                    lookup_package = entities.get("runtime") or ""
                doc_chunk = mcp_client.fetch_docs(
                    tool=tool,
                    operation=operation,
                    package=lookup_package,
                    os_hint=canonical_intent.get("os_hint") or "linux",
                    runtime=entities.get("runtime") or "",
                    version=entities.get("version") or "",
                )
                doc_chunk = _compact_doc_chunk(doc_chunk)
                if doc_chunk and doc_chunk.get("error"):
                    if _doc_chunk_has_context(doc_chunk):
                        print(f"MCP lookup status: partial ({doc_chunk.get('error')})")
                    else:
                        print(f"MCP lookup status: fallback ({doc_chunk.get('error')})")
                elif doc_chunk:
                    print("MCP lookup status: success")
            else:
                print("\nMCP lookup skipped: could not infer tool. Provide runtime or --tool-override.")

        prompt = format_input(
            row,
            compact=True,
            doc_chunk=doc_chunk if _doc_chunk_has_context(doc_chunk) else None,
        )

        if _doc_chunk_has_context(doc_chunk):
            print("\nMCP attached:")
            print(f"  syntax: {doc_chunk.get('command_syntax', '')}")
            examples = doc_chunk.get("examples") or []
            if examples:
                print(f"  example: {examples[0]}")

        output_text = model.generate_text(
            prompt,
            max_length=args.max_output_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
        )

        command_plan, error = parse_model_output(output_text)

        print("\nInput (CanonicalIntent):")
        print(json.dumps(canonical_intent, indent=2))

        print("\nRaw model output:")
        print(output_text)

        if error:
            print(f"\nParse status: failed ({error})")
        else:
            print("\nParse status: ok")
            print("\nParsed CommandPlan:")
            print(json.dumps(command_plan, indent=2))

        if args.single_run:
            break

        again = _prompt("\nRun another test? (y/n)", default="y", allowed=["y", "n"], allow_empty=False)
        if again == "n":
            break


if __name__ == "__main__":
    main()
