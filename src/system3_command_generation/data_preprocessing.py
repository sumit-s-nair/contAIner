"""Data preprocessing utilities for System 2 command generation.

This module formats command-dataset rows into model input/output text, validates
schema constraints, builds torch datasets, and provides inference helpers.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizer
import jsonschema

log = logging.getLogger("system2.data")

from .config import (
    COMMAND_PLAN_SCHEMA,
    OS_SHELL_COMPATIBILITY,
    TrainingConfig,
)


# =============================================================================
# Dataset Schema (matches command-dataset)
# =============================================================================

DATASET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["instruction", "intent_type", "entities", "os", "shell", "command", "source"],
    "properties": {
        "instruction": {"type": "string"},
        "intent_type": {"type": "string"},
        "entities": {
            "type": "object",
            "properties": {
                "runtime": {"type": ["string", "null"]},
                "package": {"type": ["string", "null"]},
                "version": {"type": ["string", "null"]},
            },
        },
        "os": {"type": "string", "enum": ["linux", "windows", "macos"]},
        "shell": {"type": "string", "enum": ["bash", "powershell", "cmd", "zsh"]},
        "command": {"type": "string"},
        "source": {"type": "string"},
    },
}


# =============================================================================
# Prompt Templates
# =============================================================================

INPUT_TEMPLATE = """Generate a CommandPlan for the following intent:

Intent Type: {intent_type}
Entities: {entities}
Target OS: {os}
Target Shell: {shell}

Output a valid JSON CommandPlan:"""

# Compact prompt format for better token efficiency
INPUT_TEMPLATE_COMPACT = """<intent>{intent_type}</intent>
<entities>{entities}</entities>
<os>{os}</os>
<shell>{shell}</shell>
<command>"""

# MCP-enriched prompt: includes live documentation context fetched from the
# MCP server (syntax, key flags, one example, OS-specific notes).
INPUT_TEMPLATE_WITH_MCP = """<intent>{intent_type}</intent>
<entities>{entities}</entities>
<os>{os}</os>
<shell>{shell}</shell>
<docs>
syntax: {command_syntax}
flags: {key_flags}
example: {example}
notes: {os_notes}
</docs>
<command>"""

# Maps runtime names → package manager tool names (mirrors mcp_doc_fetcher.py)
_RUNTIME_TO_TOOL: Dict[str, str] = {
    "python": "pip",
    "node": "npm",
    "nodejs": "npm",
    "rust": "cargo",
    "go": "go",
    "golang": "go",
    "java": "maven",
    "ruby": "gem",
}

_SYSTEM_TOOL_BY_OS: Dict[str, str] = {
    "linux": "apt",
    "macos": "brew",
    "windows": "conda",
}

# Maps intent_type → MCP operation keyword
_INTENT_TO_OPERATION: Dict[str, str] = {
    "install_package": "install",
    "install_runtime": "install",
    "update_package": "update",
    "update_runtime": "update",
    "remove_package": "uninstall",
    "remove_runtime": "uninstall",
    "list_packages": "list",
    "check_version": "show",
}


def _render_key_flags(key_flags: Any) -> str:
    """Render MCP key_flags as compact prompt text.

    Supports both legacy ``list[str]`` payloads and current
    ``list[{"flag": ..., "description": ...}]`` payloads.
    """
    if key_flags is None:
        return ""

    if isinstance(key_flags, str):
        return key_flags.strip()

    if isinstance(key_flags, dict):
        key_flags = [key_flags]

    if not isinstance(key_flags, list):
        return str(key_flags).strip()

    rendered: List[str] = []
    for item in key_flags:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            flag = str(item.get("flag") or item.get("name") or "").strip()
            description = str(item.get("description") or item.get("desc") or "").strip()
            if flag and description:
                text = f"{flag}: {description}"
            else:
                text = flag or description
        else:
            text = str(item).strip()

        if text:
            rendered.append(text)

    return ", ".join(rendered)


def _doc_chunk_has_context(doc_chunk: Optional[Dict[str, Any]]) -> bool:
    """Return True when a DocChunk has usable prompt context.

    Some adapters may return a non-empty ``error`` string even when they still
    provide useful syntax/examples/flags (for example, docs fetch succeeded but
    registry metadata fetch failed). In that case we still want to inject docs.
    """
    if not doc_chunk:
        return False

    syntax = str(doc_chunk.get("command_syntax") or "").strip()
    examples = doc_chunk.get("examples") or []
    key_flags = doc_chunk.get("key_flags") or []
    return bool(syntax or examples or key_flags)


def _resolve_mcp_tool(intent_type: str, runtime: str, os_hint: str) -> str:
    """Resolve MCP tool from intent, runtime, and OS hint."""
    intent = (intent_type or "").lower()
    runtime_name = (runtime or "").lower()
    os_name = (os_hint or "linux").lower()

    if intent.endswith("_runtime"):
        return _SYSTEM_TOOL_BY_OS.get(os_name, "")

    return _RUNTIME_TO_TOOL.get(runtime_name, runtime_name)


def _resolve_mcp_package(intent_type: str, runtime: str, package: str) -> str:
    """Resolve package field for MCP lookups.

    Runtime intents are typically package-manager level installs, so when no
    package is provided, reuse runtime as the package target.
    """
    intent = (intent_type or "").lower()
    if package:
        return package
    if intent.endswith("_runtime"):
        return runtime
    return package


# =============================================================================
# MCP Client
# =============================================================================

class MCPClient:
    """Lightweight synchronous HTTP client for the MCP documentation server.

    Used during training to pre-fetch command documentation so each training
    sample's input is grounded in real tool syntax, flags, and examples.

    How it fits into training:
        1. For each dataset sample, ``intent_type`` + ``entities.runtime``
           are mapped to a (tool, operation) pair.
        2. This client POSTs to ``/fetch_docs`` on the MCP server.
        3. The returned ``DocChunk`` (syntax, flags, example, notes) is
           injected into the model input via ``INPUT_TEMPLATE_WITH_MCP``.
        4. The model therefore learns to produce commands that are consistent
           with the documentation — not just pattern-matched from the dataset.
    """

    def __init__(self, url: str = "http://localhost:11435", timeout: int = 10):
        self.base_url = url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Return True if the MCP server responds to /health."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except Exception:
            return False

    def fetch_docs(
        self,
        tool: str,
        operation: str,
        package: str = "",
        os_hint: str = "linux",
        runtime: str = "",
        version: str = "",
    ) -> Dict[str, Any]:
        """Fetch a DocChunk from the MCP server.

        Returns the raw response dict on success, or a stub fallback dict
        with ``error`` set when the server is unreachable / returns an error.
        """
        body = {
            "tool": tool,
            "operation": operation,
            "package": package,
            "os": os_hint,
            "runtime": runtime,
            "version": version,
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/fetch_docs",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
                log.debug(
                    "[MCP] tool=%s op=%s pkg=%s  →  syntax: %.60s",
                    tool, operation, package,
                    result.get("command_syntax", "(none)"),
                )
                return result
        except Exception as exc:
            log.debug("[MCP] fetch failed (%s) — stub fallback for %s/%s", exc, tool, operation)
            return self._stub(tool, operation, package)

    @staticmethod
    def _stub(tool: str, operation: str, package: str) -> Dict[str, Any]:
        syntax = f"{tool} {operation} {package}".strip()
        return {
            "command_syntax": syntax,
            "key_flags": [],
            "examples": [syntax],
            "os_specific_notes": "",
            "error": "stub",
        }


def format_input(
    row: Dict[str, Any],
    compact: bool = True,
    doc_chunk: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a dataset row into model input text.

    When ``doc_chunk`` is provided (and has no ``error`` key), the MCP-enriched
    template is used so the model sees live documentation context alongside the
    intent.  Falls back to the compact template when docs are unavailable.
    """
    entities_str = json.dumps(row["entities"], separators=(",", ":"))

    if _doc_chunk_has_context(doc_chunk):
        examples = doc_chunk.get("examples") or []
        return INPUT_TEMPLATE_WITH_MCP.format(
            intent_type=row["intent_type"],
            entities=entities_str,
            os=row["os"],
            shell=row["shell"],
            command_syntax=doc_chunk.get("command_syntax", ""),
            key_flags=_render_key_flags(doc_chunk.get("key_flags")),
            example=examples[0] if examples else "",
            os_notes=doc_chunk.get("os_specific_notes", ""),
        )

    if compact:
        return INPUT_TEMPLATE_COMPACT.format(
            intent_type=row["intent_type"],
            entities=entities_str,
            os=row["os"],
            shell=row["shell"],
        )

    return INPUT_TEMPLATE.format(
        intent_type=row["intent_type"],
        entities=entities_str,
        os=row["os"],
        shell=row["shell"],
    )


def format_output(row: Dict[str, Any]) -> str:
    """Format a dataset row into model output (CommandPlan JSON)."""
    # Determine step type from intent
    intent_to_step_type = {
        "install_runtime": "install",
        "install_package": "install",
        "update_runtime": "update",
        "update_package": "update",
        "remove_runtime": "remove",
        "remove_package": "remove",
        "list_packages": "verify",
        "create_environment": "configure",
        "activate_environment": "configure",
        "deactivate_environment": "configure",
        "configure_setting": "configure",
        "run_script": "verify",
        "check_version": "verify",
    }
    
    step_type = intent_to_step_type.get(row["intent_type"], "configure")
    
    # Check if elevation is required
    command = row["command"]
    requires_elevation = any(kw in command.lower() for kw in [
        "sudo", "admin", "runas", "gsudo", "doas",
        "apt install", "apt-get install", "yum install", 
        "dnf install", "pacman -s", "winget install",
    ])
    
    command_plan = {
        "intent_type": row["intent_type"],
        "entities": row["entities"],
        "os": row["os"],
        "shell": row["shell"],
        "steps": [
            {
                "step_number": 1,
                "type": step_type,
                "command": command,
                "description": row.get("instruction", f"{step_type.capitalize()} operation"),
            }
        ],
        "confidence": 0.95,
        "requires_elevation": requires_elevation,
    }
    
    return json.dumps(command_plan, separators=(",", ":"))


# =============================================================================
# Validation
# =============================================================================

def validate_dataset_row(row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a dataset row against the schema."""
    try:
        jsonschema.validate(row, DATASET_SCHEMA)
        
        # Check OS/shell compatibility
        os_type = row.get("os")
        shell_type = row.get("shell")
        
        if os_type and shell_type:
            compatible = OS_SHELL_COMPATIBILITY.get(os_type, [])
            if shell_type not in compatible:
                return False, f"Shell '{shell_type}' incompatible with OS '{os_type}'"
        
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)


def validate_command_plan(plan: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a generated CommandPlan."""
    try:
        jsonschema.validate(plan, COMMAND_PLAN_SCHEMA)
        
        # Check OS/shell compatibility
        os_type = plan.get("os")
        shell_type = plan.get("shell")
        
        if os_type and shell_type:
            compatible = OS_SHELL_COMPATIBILITY.get(os_type, [])
            if shell_type not in compatible:
                return False, f"Shell '{shell_type}' incompatible with OS '{os_type}'"
        
        # Check step sequence
        steps = plan.get("steps", [])
        for i, step in enumerate(steps, start=1):
            if step.get("step_number") != i:
                return False, f"Step sequence error at position {i}"
        
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)


# =============================================================================
# PyTorch Dataset
# =============================================================================

class CommandGenerationDataset(Dataset):
    """
    PyTorch Dataset for command generation training.
    
    Directly processes command-dataset rows without intermediate transformations.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        max_output_length: int = 512,
        compact_prompts: bool = True,
        is_encoder_decoder: bool = True,
        doc_chunks: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data: List of dataset rows
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            compact_prompts: Use compact prompt format for efficiency
            is_encoder_decoder: Whether the underlying model is seq2seq
            doc_chunks: Pre-fetched MCP DocChunks aligned to ``data`` by index.
                        Pass ``None`` (default) to disable MCP enrichment.
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.compact_prompts = compact_prompts
        self.is_encoder_decoder = is_encoder_decoder

        # Keep only rows that pass schema and OS/shell compatibility checks.
        # Track original indices so doc_chunks stay aligned.
        self.data: List[Dict[str, Any]] = []
        self.doc_chunks: List[Optional[Dict[str, Any]]] = []
        for i, row in enumerate(data):
            is_valid, _ = validate_dataset_row(row)
            if is_valid:
                self.data.append(row)
                chunk = doc_chunks[i] if (doc_chunks is not None and i < len(doc_chunks)) else None
                self.doc_chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data[idx]
        doc_chunk = self.doc_chunks[idx] if idx < len(self.doc_chunks) else None

        # Format input (with MCP context if available) and output
        input_text = format_input(row, compact=self.compact_prompts, doc_chunk=doc_chunk)
        output_text = format_output(row)

        if self.is_encoder_decoder:
            # Standard seq2seq setup: model attends to input and predicts output.
            input_encoding = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            output_encoding = self.tokenizer(
                output_text,
                max_length=self.max_output_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            labels = output_encoding["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_encoding["input_ids"].squeeze(),
                "attention_mask": input_encoding["attention_mask"].squeeze(),
                "labels": labels,
            }

        # Decoder-only setup (e.g., Qwen): train on prompt + target and mask prompt tokens.
        prompt_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )

        full_encoding = self.tokenizer(
            f"{input_text}{output_text}",
            max_length=self.max_input_length + self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"].squeeze()
        attention_mask = full_encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        prompt_len = min(int(prompt_encoding["input_ids"].shape[1]), labels.shape[0])
        labels[:prompt_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Get raw dataset row for evaluation.

        Returns a dict with all keys expected by ``CommandGenerationTrainer.evaluate()``:
        ``input_text``, ``command_plan`` (parsed reference), ``canonical_intent``.
        """
        row = self.data[idx]
        doc_chunk = self.doc_chunks[idx] if idx < len(self.doc_chunks) else None
        input_text = format_input(row, compact=self.compact_prompts, doc_chunk=doc_chunk)
        output_text = format_output(row)
        command_plan, _ = parse_model_output(output_text)
        return {
            "row": row,
            "input_text": input_text,
            "output_text": output_text,
            "command_plan": command_plan or {},
            "canonical_intent": {
                "intent_type": row["intent_type"],
                "entities": row.get("entities", {}),
            },
        }


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class CommandDataCollator:
    """Data collator with padding."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


# =============================================================================
# Data Processor
# =============================================================================

class CommandDataProcessor:
    """
    Main data processor for command generation training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
        mcp_client: Optional[MCPClient] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.mcp_client = mcp_client
    
    def load_huggingface_dataset(
        self,
        dataset_name: str = "sumit-s-nair/command-dataset",
    ) -> DatasetDict:
        """Load dataset from HuggingFace Hub."""
        print(f"📦 Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        print(f"  ✓ Loaded splits: {list(dataset.keys())}")
        return dataset
    
    def load_local_dataset(self, data_dir: str) -> Dict[str, List[Dict]]:
        """Load dataset from local JSONL files."""
        print(f"📦 Loading from: {data_dir}")
        
        splits = {}
        for split in ["train", "validation", "test"]:
            filepath = os.path.join(data_dir, f"{split}.jsonl")
            if os.path.exists(filepath):
                data = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                splits[split] = data
                print(f"  ✓ {split}: {len(data)} samples")
        
        return splits
    
    def _enrich_with_mcp(self, data: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Pre-fetch MCP documentation for every sample in *data*.

        Returns a list of DocChunk dicts (or ``None``) aligned 1-to-1 with
        ``data``.  Server-side caching means repeated (tool, operation, os)
        combos are cheap after the first request.

        What you'll see in the logs explains the MCP flow:
          [MCP] tool=pip   op=install  pkg=requests  →  syntax: pip install requests
          [MCP] tool=npm   op=install  pkg=lodash    →  syntax: npm install lodash
          …
        """
        doc_chunks: List[Optional[Dict[str, Any]]] = []
        total = len(data)
        hits = 0

        print(f"\n[MCP] Pre-fetching documentation for {total} samples "
              f"from {self.mcp_client.base_url} ...")

        for i, row in enumerate(data):
            entities = row.get("entities") or {}
            runtime = (entities.get("runtime") or "").lower()
            package = entities.get("package") or ""
            version = entities.get("version") or ""
            os_hint = row.get("os") or "linux"
            intent_type = row.get("intent_type") or ""

            tool = _resolve_mcp_tool(intent_type, runtime, os_hint)
            operation = _INTENT_TO_OPERATION.get(intent_type, "install")
            lookup_package = _resolve_mcp_package(intent_type, runtime, package)

            if tool and operation:
                chunk = self.mcp_client.fetch_docs(
                    tool=tool,
                    operation=operation,
                    package=lookup_package,
                    os_hint=os_hint,
                    runtime=runtime,
                    version=version,
                )
                if _doc_chunk_has_context(chunk):
                    hits += 1
                doc_chunks.append(chunk)
            else:
                doc_chunks.append(None)

            if (i + 1) % 200 == 0 or (i + 1) == total:
                print(f"  [{i + 1:>5}/{total}]  enriched={hits}  stubs={i + 1 - hits}")

        print(f"[MCP] Enrichment done — {hits}/{total} samples have live doc context\n")
        return doc_chunks

    def create_datasets(
        self,
        raw_data: Union[DatasetDict, Dict[str, List]],
    ) -> Dict[str, CommandGenerationDataset]:
        """Create PyTorch datasets from raw data."""
        datasets = {}

        for split in ["train", "validation", "test"]:
            if split not in raw_data:
                continue

            split_data = raw_data[split]
            if hasattr(split_data, "to_list"):
                split_data = split_data.to_list()
            elif hasattr(split_data, "__iter__") and not isinstance(split_data, list):
                split_data = list(split_data)

            # Pre-fetch MCP docs when client is configured
            doc_chunks = None
            if self.mcp_client is not None:
                doc_chunks = self._enrich_with_mcp(split_data)

            datasets[split] = CommandGenerationDataset(
                data=split_data,
                tokenizer=self.tokenizer,
                max_input_length=self.config.max_input_length,
                max_output_length=self.config.max_output_length,
                is_encoder_decoder=self.config.get_model_config().is_encoder_decoder,
                doc_chunks=doc_chunks,
            )
            mcp_note = " (MCP-enriched)" if doc_chunks is not None else ""
            print(f"  ✓ {split}: {len(datasets[split])} valid samples{mcp_note}")

        return datasets
    
    def get_data_collator(self) -> CommandDataCollator:
        return CommandDataCollator(tokenizer=self.tokenizer)


# =============================================================================
# Inference Utilities
# =============================================================================

def parse_model_output(output_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse model output text into CommandPlan dict."""
    try:
        output_text = output_text.strip()

        # Prefer CommandPlan-like objects when prompt text includes other JSON snippets.
        start = output_text.find('{"intent_type"')
        if start == -1:
            start = output_text.find("{")

        if start == -1:
            return None, "No JSON found"

        decoder = json.JSONDecoder()
        for idx in range(start, len(output_text)):
            if output_text[idx] != "{":
                continue

            try:
                candidate, _ = decoder.raw_decode(output_text[idx:])
            except json.JSONDecodeError:
                continue

            if not isinstance(candidate, dict):
                continue
            if "intent_type" not in candidate or "steps" not in candidate:
                continue

            is_valid, error = validate_command_plan(candidate)
            if not is_valid:
                return None, error

            return candidate, None

        return None, "No valid CommandPlan JSON found"
        
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def prepare_inference_input(
    intent_type: str,
    entities: Dict[str, Any],
    os_type: str,
    shell: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare input for model inference.
    
    This is called at inference time with System 1's CanonicalIntent output.
    
    Args:
        intent_type: From CanonicalIntent.intent_type
        entities: From CanonicalIntent.entities
        os_type: From CanonicalIntent.os_hint (resolved)
        shell: From CanonicalIntent.shell_type
        tokenizer: Model tokenizer
        max_length: Max sequence length
        device: Target device
        
    Returns:
        Tokenized input ready for model.generate()
    """
    row = {
        "intent_type": intent_type,
        "entities": entities,
        "os": os_type,
        "shell": shell,
    }
    
    input_text = format_input(row, compact=True)
    
    encoding = tokenizer(
        input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
    }


def inference_from_canonical_intent(
    canonical_intent: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare input from a full CanonicalIntent object (System 1 output).
    
    This bridges System 1 output → System 2 input at runtime.
    """
    return prepare_inference_input(
        intent_type=canonical_intent["intent_type"],
        entities=canonical_intent["entities"],
        os_type=canonical_intent.get("os_hint") or "linux",  # Default to linux
        shell=canonical_intent["shell_type"],
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )
