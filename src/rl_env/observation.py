"""
src/rl_env/observation.py
=========================
Observation dataclass and serializer for the System 2 planner RL environment.

The observation is the full context the policy receives at each step:

  Observation = CanonicalIntent  +  RepoManifest  +  episode history

``ObservationSerializer.to_context_string()`` emits the **hybrid context format**:

  1. Ecosystem summary block     — one line per ecosystem with dep count + manifest status
  2. Per-dependency explain()    — capped at MAX_DEPS entries via RepoManifest.explain()
  3. Conflicts block             — bulleted list, or "(none)"
  4. Episode history             — step_index/max_steps + prior step descriptions

The serializer outputs a structured JSON string suitable for structured use.

Flat/chat-template serialization
---------------------------------
HARD PREREQUISITE before any training script is drafted:
A ``to_chat_prompt()`` method must be added here that produces a flat string
suitable for direct injection into a Qwen2.5 chat template (system + user turns).
This is explicitly deferred here but must be implemented before any training
script is wired up.  Attempting to wire a training script without this
method present must fail loudly (see ``_CHAT_PROMPT_NOT_IMPLEMENTED`` guard below).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.repo_scan.models import RepoManifest


# ---------------------------------------------------------------------------
# Maximum number of individual dependency explain() entries in the context.
# Remaining deps are summarised as "... and N more".
# ---------------------------------------------------------------------------
MAX_DEPS = 20

# Maximum tokens for the entire prompt (system + user) sent to the model.
# Sections are prioritised: system > ecosystem_summary > conflicts >
# episode_history > dependency_details.  Only the dep list is truncated;
# all other sections are always emitted in full.
# Must match (or be <=) MAX_PROMPT_TOKENS in the training script.
MAX_PROMPT_TOKENS = 512


# ---------------------------------------------------------------------------
# CanonicalIntent — the planner's input goal
# ---------------------------------------------------------------------------

@dataclass
class CanonicalIntent:
    """
    Describes *what* the planner must accomplish in this episode.

    intent_text:
        Natural-language description of the user's intent, e.g.
        "Set up this Python project for development".

    template_name:
        The workflow template selected for this episode
        (one of: "setup_project", "fix_environment", "setup_environment").

    repo_id:
        Identifier of the repo from the corpus (e.g. "owner/name"),
        for logging and reproducibility.
    """
    intent_text:   str
    template_name: str
    repo_id:       str = ""


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    Full observation returned by ``PlannerEnv.reset()`` and ``PlannerEnv.step()``.

    canonical_intent:
        The episode's goal (intent text + template name + repo id).

    repo_manifest:
        The ``RepoManifest`` produced by scanning the episode's repo.

    episode_steps:
        Ordered list of ``AtomicStep.description`` strings for steps taken
        so far in this episode.  Empty at the start of each episode.

    step_index:
        Zero-based index of the *next* step to be taken (0 = first step).

    max_steps:
        Maximum number of steps allowed in this episode.
    """
    canonical_intent: CanonicalIntent
    repo_manifest:    RepoManifest
    episode_steps:    List[str]      = field(default_factory=list)
    step_index:       int            = 0
    max_steps:        int            = 8


# ---------------------------------------------------------------------------
# ObservationSerializer
# ---------------------------------------------------------------------------

class ObservationSerializer:
    """
    Converts an :class:`Observation` into a structured context string for
    the policy model.

    ``to_context_string()``
        Returns a JSON-serializable dict serialized to an indented JSON string.
        This is the canonical format used by the RL loop.

    ``to_chat_prompt()``
        NOT YET IMPLEMENTED — hard prerequisite before any training script
        is wired up.  Calling this method raises ``NotImplementedError`` with
        a clear message so the missing implementation is surfaced loudly at
        integration time rather than silently failing.
    """

    def to_context_string(self, obs: Observation) -> str:
        """
        Serialize *obs* to a structured JSON context string.

        Output structure::

            {
              "intent": {"text": "...", "template": "...", "repo_id": "..."},
              "ecosystem_summary": [
                {"ecosystem": "python", "dep_count": 12,
                 "manifest_present": true, "has_lock": false},
                ...
              ],
              "dependency_details": [
                "numpy: declared as 'numpy>=1.20' in requirements.txt:3",
                ...
              ],
              "dependency_details_truncated": 5,
              "conflicts": ["conflict description 1", ...],
              "episode_history": {
                "step_index": 2,
                "max_steps": 8,
                "steps_taken": ["Detect broken component.", ...]
              }
            }
        """
        manifest = obs.repo_manifest

        # 1. Ecosystem summary
        eco_summary = []
        for eco_name, eco in manifest.ecosystems.items():
            eco_summary.append({
                "ecosystem":        eco_name,
                "dep_count":        len(eco.dependencies) + len(eco.inferred_dependencies),
                "manifest_present": len(eco.manifest_files) > 0,
                "has_lock":         len(eco.lock_files) > 0,
                "inferred_count":   len(eco.inferred_dependencies),
            })

        # 2. Per-dependency explain() — cap at MAX_DEPS
        all_dep_names: List[str] = []
        for eco in manifest.ecosystems.values():
            all_dep_names.extend(eco.dependencies.keys())
            all_dep_names.extend(
                d.guessed_package_name or d.import_name
                for d in eco.inferred_dependencies
            )

        dep_details: List[str] = []
        truncated_count = 0
        for name in all_dep_names[:MAX_DEPS]:
            try:
                dep_details.append(manifest.explain(name))
            except KeyError:
                # explain() raises KeyError if dep not found — skip gracefully
                pass

        if len(all_dep_names) > MAX_DEPS:
            truncated_count = len(all_dep_names) - MAX_DEPS

        # 3. Conflicts
        conflicts = list(manifest.conflicts) if manifest.conflicts else []

        # 4. Episode history
        history = {
            "step_index":  obs.step_index,
            "max_steps":   obs.max_steps,
            "steps_taken": list(obs.episode_steps),
        }

        payload: Dict = {
            "intent": {
                "text":     obs.canonical_intent.intent_text,
                "template": obs.canonical_intent.template_name,
                "repo_id":  obs.canonical_intent.repo_id,
            },
            "ecosystem_summary":           eco_summary,
            "dependency_details":          dep_details,
            "dependency_details_truncated": truncated_count,
            "conflicts":                   conflicts,
            "episode_history":             history,
        }

        return json.dumps(payload, indent=2)

    def to_chat_prompt(
        self,
        obs: Observation,
        tokenizer=None,
        max_tokens: int = MAX_PROMPT_TOKENS,
    ) -> List[Dict[str, str]]:
        """
        Returns a list of message dicts suitable for injection into a Qwen2.5
        chat template via `tokenizer.apply_chat_template()`.

        The output format is:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."}
        ]

        Truncation strategy
        -------------------
        Only the dependency list is truncated to fit within *max_tokens*.
        All other sections (system turn, ecosystem summary, conflicts, and
        episode history) are always emitted in full.  This preserves the
        ``known_conflict`` category's signal even on large repos.

        If *tokenizer* is provided, token counts are exact.  Otherwise a
        conservative character-based heuristic (4 chars ≈ 1 token) is used.
        """
        def _token_len(text: str) -> int:
            if tokenizer is not None:
                return len(tokenizer.encode(text, add_special_tokens=False))
            return max(1, len(text) // 4)  # heuristic fallback

        system_content = (
            "You are the System 2 Planner. Your job is to select the next logical high-level "
            "action to execute in order to accomplish the user's intent. "
            "Output your response strictly as a JSON object representing a PlannedStep with the following keys:\n"
            "- \"action_type\": The high-level action to perform. Must be one of: ISOLATE, INSTALL, CHECK, UPDATE, DETECT, FIX, ESCALATE, BUILD.\n"
            "- \"target\": The specific dependency, file, or target of the action (e.g. 'requests' or 'src/app.py'). Leave empty if not applicable.\n"
            "- \"description\": A brief human-readable explanation of what this step does.\n"
            "- \"rationale\": The reasoning behind selecting this action."
        )

        # ------------------------------------------------------------------ #
        # Build invariant sections first so we know their token cost.         #
        # These are ALWAYS included; only dep_lines is budget-trimmed.        #
        # ------------------------------------------------------------------ #

        header_lines = [
            f"Goal: {obs.canonical_intent.intent_text}",
            f"Template: {obs.canonical_intent.template_name}",
            f"Repository: {obs.canonical_intent.repo_id}",
            "",
        ]

        eco_lines = ["Ecosystem Summary:"]
        for eco_name, eco in obs.repo_manifest.ecosystems.items():
            manifest_status = "present" if eco.manifest_files else "missing"
            eco_lines.append(
                f"- {eco_name}: "
                f"{len(eco.dependencies) + len(eco.inferred_dependencies)} dependencies "
                f"(manifest {manifest_status})"
            )
        eco_lines.append("")

        conflict_lines: List[str] = []
        if obs.repo_manifest.conflicts:
            conflict_lines.append("Conflicts:")
            for conflict in obs.repo_manifest.conflicts:
                conflict_lines.append(f"- {conflict}")
            conflict_lines.append("")

        history_lines = ["Episode History (Steps taken so far):"]
        if not obs.episode_steps:
            history_lines.append("(None)")
        else:
            for i, step_desc in enumerate(obs.episode_steps):
                history_lines.append(f"{i+1}. {step_desc}")
        history_lines.append(f"\nNext Step (Step {obs.step_index + 1} of {obs.max_steps}):")

        # ------------------------------------------------------------------ #
        # Compute token budget remaining for the dependency list.             #
        # ------------------------------------------------------------------ #
        invariant_text = (
            system_content
            + "\n".join(header_lines + eco_lines + conflict_lines + history_lines)
        )
        tokens_used = _token_len(invariant_text)
        # Reserve headroom: 64 tokens for generation + 16-token safety margin
        dep_budget = max(0, max_tokens - tokens_used - 64 - 16)

        # ------------------------------------------------------------------ #
        # Build the dependency list, stopping when the budget is exhausted.   #
        # ------------------------------------------------------------------ #
        all_dep_names: List[str] = []
        for eco in obs.repo_manifest.ecosystems.values():
            all_dep_names.extend(eco.dependencies.keys())
            all_dep_names.extend(
                d.guessed_package_name or d.import_name
                for d in eco.inferred_dependencies
            )

        dep_lines: List[str] = []
        total_deps = len(all_dep_names)
        included = 0
        if all_dep_names and dep_budget > 0:
            dep_lines.append("Dependencies:")
            for name in all_dep_names[:MAX_DEPS]:
                try:
                    line = obs.repo_manifest.explain(name)
                except KeyError:
                    line = name
                line_tokens = _token_len(line + "\n")
                if line_tokens > dep_budget:
                    break
                dep_lines.append(line)
                dep_budget -= line_tokens
                included += 1

            omitted = total_deps - included
            if omitted > 0:
                dep_lines.append(f"... and {omitted} more (truncated to fit token budget).")
            dep_lines.append("")

        # ------------------------------------------------------------------ #
        # Assemble final user content.                                        #
        # ------------------------------------------------------------------ #
        all_user_lines = (
            header_lines
            + eco_lines
            + dep_lines
            + conflict_lines
            + history_lines
        )
        user_content = "\n".join(all_user_lines)

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
