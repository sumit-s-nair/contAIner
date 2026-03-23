"""Evaluation metrics for System 2 command-generation outputs.

Provides exact/normalized matching, contract preservation checks, schema and
syntax validation, compatibility checks, and aggregated reporting helpers.
"""

import json
import re
import subprocess
import shlex
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .config import (
    OS_SHELL_COMPATIBILITY,
    INTENT_TYPES,
    OS_TYPES,
    SHELL_TYPES,
    STEP_TYPES,
)
from .data_preprocessing import validate_command_plan


# =============================================================================
# Metric Results
# =============================================================================

@dataclass
class MetricResults:
    """Container for evaluation metric results."""
    
    # Core metrics
    exact_match: float = 0.0
    normalized_match: float = 0.0
    
    # Preservation metrics
    intent_preservation: float = 0.0
    entity_preservation: float = 0.0
    
    # Validity metrics
    syntax_validity: float = 0.0
    json_validity: float = 0.0
    schema_validity: float = 0.0
    os_shell_compatibility: float = 0.0
    
    # Quality metrics
    confidence_correlation: float = 0.0
    average_confidence: float = 0.0
    
    # Counts
    total_samples: int = 0
    valid_outputs: int = 0
    invalid_outputs: int = 0
    
    # Detailed breakdowns
    per_intent_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_os: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_shell: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exact_match": self.exact_match,
            "normalized_match": self.normalized_match,
            "intent_preservation": self.intent_preservation,
            "entity_preservation": self.entity_preservation,
            "syntax_validity": self.syntax_validity,
            "json_validity": self.json_validity,
            "schema_validity": self.schema_validity,
            "os_shell_compatibility": self.os_shell_compatibility,
            "confidence_correlation": self.confidence_correlation,
            "average_confidence": self.average_confidence,
            "total_samples": self.total_samples,
            "valid_outputs": self.valid_outputs,
            "invalid_outputs": self.invalid_outputs,
            "per_intent_type": dict(self.per_intent_type),
            "per_os": dict(self.per_os),
            "per_shell": dict(self.per_shell),
            "error_types": dict(self.error_types),
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "📊 Evaluation Results",
            "=" * 60,
            "",
            "Core Metrics:",
            f"  Exact Match:        {self.exact_match:.2%}",
            f"  Normalized Match:   {self.normalized_match:.2%}",
            "",
            "Preservation Metrics:",
            f"  Intent Preservation: {self.intent_preservation:.2%}",
            f"  Entity Preservation: {self.entity_preservation:.2%}",
            "",
            "Validity Metrics:",
            f"  JSON Validity:       {self.json_validity:.2%}",
            f"  Schema Validity:     {self.schema_validity:.2%}",
            f"  Syntax Validity:     {self.syntax_validity:.2%}",
            f"  OS/Shell Compat:     {self.os_shell_compatibility:.2%}",
            "",
            "Quality Metrics:",
            f"  Avg Confidence:      {self.average_confidence:.2%}",
            f"  Conf. Correlation:   {self.confidence_correlation:.3f}",
            "",
            "Sample Counts:",
            f"  Total:   {self.total_samples}",
            f"  Valid:   {self.valid_outputs}",
            f"  Invalid: {self.invalid_outputs}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Command Normalization
# =============================================================================

def normalize_command(command: str) -> str:
    """
    Normalize a command for comparison.
    
    - Removes extra whitespace
    - Sorts flags alphabetically
    - Lowercases command names
    - Normalizes path separators
    
    Args:
        command: The command string to normalize
        
    Returns:
        Normalized command string
    """
    # Remove leading/trailing whitespace
    command = command.strip()
    
    # Normalize multiple spaces to single space
    command = re.sub(r'\s+', ' ', command)
    
    # Try to parse and normalize shell commands
    try:
        parts = shlex.split(command)
        if not parts:
            return command
        
        # Separate command, flags, and arguments
        cmd = parts[0].lower()
        flags = []
        args = []
        
        i = 1
        while i < len(parts):
            part = parts[i]
            if part.startswith('-'):
                # It's a flag
                if '=' in part:
                    # Flag with value: --flag=value
                    flags.append(part.lower())
                elif i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                    # Flag with separate value: --flag value
                    flags.append(f"{part.lower()}={parts[i + 1]}")
                    i += 1
                else:
                    flags.append(part.lower())
            else:
                args.append(part)
            i += 1
        
        # Sort flags for consistent comparison
        flags.sort()
        
        # Reconstruct normalized command
        normalized = cmd
        if flags:
            normalized += ' ' + ' '.join(flags)
        if args:
            normalized += ' ' + ' '.join(args)
        
        return normalized
        
    except ValueError:
        # shlex couldn't parse, return simplified version
        return command.lower()


def normalize_json_for_comparison(obj: Any) -> Any:
    """
    Recursively normalize a JSON object for comparison.
    
    - Sorts dictionary keys
    - Normalizes commands in steps
    - Removes whitespace in strings
    """
    if isinstance(obj, dict):
        return {k: normalize_json_for_comparison(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json_for_comparison(item) for item in obj]
    elif isinstance(obj, str):
        # Check if it looks like a command
        if any(c in obj for c in ['install', 'apt', 'pip', 'npm', 'winget', 'brew']):
            return normalize_command(obj)
        return obj.strip()
    else:
        return obj


# =============================================================================
# Individual Metric Functions
# =============================================================================

def compute_exact_match(
    prediction: Dict[str, Any],
    reference: Dict[str, Any],
) -> bool:
    """Check if prediction exactly matches reference."""
    return json.dumps(prediction, sort_keys=True) == json.dumps(reference, sort_keys=True)


def compute_normalized_match(
    prediction: Dict[str, Any],
    reference: Dict[str, Any],
) -> bool:
    """Check if normalized prediction matches normalized reference."""
    norm_pred = normalize_json_for_comparison(prediction)
    norm_ref = normalize_json_for_comparison(reference)
    return json.dumps(norm_pred, sort_keys=True) == json.dumps(norm_ref, sort_keys=True)


def compute_intent_preservation(
    prediction: Dict[str, Any],
    input_intent: Dict[str, Any],
) -> bool:
    """Check if output intent matches input intent."""
    pred_intent = prediction.get("intent_type")
    input_intent_type = input_intent.get("intent_type")
    return pred_intent == input_intent_type


def compute_entity_preservation(
    prediction: Dict[str, Any],
    input_intent: Dict[str, Any],
) -> float:
    """
    Compute entity preservation score.
    
    Returns the fraction of input entities that are preserved in output.
    """
    input_entities = input_intent.get("entities", {})
    pred_entities = prediction.get("entities", {})
    
    if not input_entities:
        return 1.0  # No entities to preserve
    
    matches = 0
    total = 0
    
    for key, value in input_entities.items():
        if value is not None:  # Only count non-null entities
            total += 1
            pred_value = pred_entities.get(key)
            if pred_value == value:
                matches += 1
            elif value is None and pred_value is not None:
                # Allow enrichment of null values
                matches += 1
    
    return matches / total if total > 0 else 1.0


def compute_os_shell_compatibility(prediction: Dict[str, Any]) -> bool:
    """Check if OS and shell are compatible."""
    os_type = prediction.get("os")
    shell_type = prediction.get("shell")
    
    if not os_type or not shell_type:
        return False
    
    compatible_shells = OS_SHELL_COMPATIBILITY.get(os_type, [])
    return shell_type in compatible_shells


def validate_command_syntax(command: str, shell: str) -> Tuple[bool, Optional[str]]:
    """
    Validate command syntax for the given shell.
    
    Note: This is a basic validation. For production, consider using
    actual shell parsers or dry-run validation.
    
    Args:
        command: The command to validate
        shell: Target shell type
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic syntax checks
    if not command or not command.strip():
        return False, "Empty command"
    
    # Check for common syntax errors
    try:
        # Check balanced quotes
        single_quotes = command.count("'")
        double_quotes = command.count('"')
        
        if single_quotes % 2 != 0:
            return False, "Unbalanced single quotes"
        if double_quotes % 2 != 0:
            return False, "Unbalanced double quotes"
        
        # Check balanced parentheses/brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        in_string = False
        string_char = None
        
        for char in command:
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif not in_string:
                if char in brackets:
                    stack.append(brackets[char])
                elif char in brackets.values():
                    if not stack or stack.pop() != char:
                        return False, f"Unbalanced bracket: {char}"
        
        if stack:
            return False, "Unclosed brackets"
        
        # Shell-specific validation
        if shell == "powershell":
            # PowerShell-specific checks
            if command.strip().startswith('$') and '=' not in command:
                # Variable reference is fine
                pass
        elif shell in ["bash", "zsh"]:
            # Bash-specific checks
            if '&&' in command or '||' in command or '|' in command:
                # Ensure operators are used correctly
                parts = re.split(r'\s*(?:\|\||&&|\|)\s*', command)
                if any(not p.strip() for p in parts):
                    return False, "Empty command in pipeline/chain"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def compute_syntax_validity(prediction: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate syntax of all commands in a CommandPlan.
    
    Returns:
        Tuple of (all_valid, first_error)
    """
    shell = prediction.get("shell", "bash")
    steps = prediction.get("steps", [])
    
    if not steps:
        return False, "No steps in CommandPlan"
    
    for step in steps:
        command = step.get("command", "")
        is_valid, error = validate_command_syntax(command, shell)
        if not is_valid:
            return False, f"Step {step.get('step_number', '?')}: {error}"
    
    return True, None


def validate_step_sequence(steps: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate that step numbers are sequential starting from 1."""
    for i, step in enumerate(steps, start=1):
        if step.get("step_number") != i:
            return False, f"Expected step_number {i}, got {step.get('step_number')}"
    return True, None


# =============================================================================
# Main Metrics Class
# =============================================================================

class CommandMetrics:
    """
    Main class for computing evaluation metrics.
    
    Computes all metrics specified in the requirements and aggregates
    results across the dataset.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self._predictions = []
        self._references = []
        self._inputs = []
        self._results = []
    
    def add_prediction(
        self,
        prediction: Dict[str, Any],
        reference: Dict[str, Any],
        input_intent: Dict[str, Any],
    ):
        """
        Add a single prediction for evaluation.
        
        Args:
            prediction: Generated CommandPlan
            reference: Ground truth CommandPlan
            input_intent: Input CanonicalIntent
        """
        self._predictions.append(prediction)
        self._references.append(reference)
        self._inputs.append(input_intent)
    
    def add_batch(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        inputs: List[Dict[str, Any]],
    ):
        """Add a batch of predictions for evaluation."""
        for pred, ref, inp in zip(predictions, references, inputs):
            self.add_prediction(pred, ref, inp)
    
    def compute(self) -> MetricResults:
        """
        Compute all metrics on accumulated predictions.
        
        Returns:
            MetricResults containing all computed metrics
        """
        results = MetricResults()
        results.total_samples = len(self._predictions)
        
        if results.total_samples == 0:
            return results
        
        # Initialize counters
        exact_matches = 0
        normalized_matches = 0
        intent_preserved = 0
        entity_scores = []
        syntax_valid = 0
        json_valid = 0
        schema_valid = 0
        os_shell_compat = 0
        confidences = []
        
        # Per-category tracking
        per_intent = defaultdict(lambda: {"total": 0, "exact": 0, "normalized": 0})
        per_os = defaultdict(lambda: {"total": 0, "exact": 0, "normalized": 0})
        per_shell = defaultdict(lambda: {"total": 0, "exact": 0, "normalized": 0})
        
        for pred, ref, inp in zip(self._predictions, self._references, self._inputs):
            intent_type = inp.get("intent_type", "unknown")
            os_type = ref.get("os", "unknown")
            shell_type = ref.get("shell", "unknown")
            
            # Track by category
            per_intent[intent_type]["total"] += 1
            per_os[os_type]["total"] += 1
            per_shell[shell_type]["total"] += 1
            
            # Check if prediction is valid JSON/dict
            if pred is None or not isinstance(pred, dict):
                results.invalid_outputs += 1
                results.error_types["invalid_output"] += 1
                continue
            
            results.valid_outputs += 1
            json_valid += 1
            
            # Schema validation
            is_valid_schema, schema_error = validate_command_plan(pred)
            if is_valid_schema:
                schema_valid += 1
            else:
                results.error_types[f"schema: {schema_error}"] += 1
            
            # Exact match
            if compute_exact_match(pred, ref):
                exact_matches += 1
                per_intent[intent_type]["exact"] += 1
                per_os[os_type]["exact"] += 1
                per_shell[shell_type]["exact"] += 1
            
            # Normalized match
            if compute_normalized_match(pred, ref):
                normalized_matches += 1
                per_intent[intent_type]["normalized"] += 1
                per_os[os_type]["normalized"] += 1
                per_shell[shell_type]["normalized"] += 1
            
            # Intent preservation
            if compute_intent_preservation(pred, inp):
                intent_preserved += 1
            else:
                results.error_types["intent_mismatch"] += 1
            
            # Entity preservation
            entity_scores.append(compute_entity_preservation(pred, inp))
            
            # OS/shell compatibility
            if compute_os_shell_compatibility(pred):
                os_shell_compat += 1
            else:
                results.error_types["os_shell_incompatible"] += 1
            
            # Syntax validity
            is_syntax_valid, syntax_error = compute_syntax_validity(pred)
            if is_syntax_valid:
                syntax_valid += 1
            else:
                results.error_types[f"syntax: {syntax_error}"] += 1
            
            # Confidence
            if "confidence" in pred:
                confidences.append(pred["confidence"])
        
        # Compute final metrics
        n = results.total_samples
        n_valid = results.valid_outputs
        
        results.exact_match = exact_matches / n if n > 0 else 0
        results.normalized_match = normalized_matches / n if n > 0 else 0
        results.intent_preservation = intent_preserved / n if n > 0 else 0
        results.entity_preservation = sum(entity_scores) / len(entity_scores) if entity_scores else 0
        results.json_validity = json_valid / n if n > 0 else 0
        results.schema_validity = schema_valid / n if n > 0 else 0
        results.syntax_validity = syntax_valid / n_valid if n_valid > 0 else 0
        results.os_shell_compatibility = os_shell_compat / n_valid if n_valid > 0 else 0
        results.average_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Compute confidence correlation (if we have quality scores)
        # For now, use normalized match as the quality indicator
        if confidences and len(confidences) > 1:
            # Simple correlation approximation
            quality_scores = [
                1.0 if compute_normalized_match(p, r) else 0.0
                for p, r in zip(self._predictions[:len(confidences)], self._references[:len(confidences)])
            ]
            results.confidence_correlation = self._compute_correlation(confidences, quality_scores)
        
        # Per-category metrics
        for intent_type, counts in per_intent.items():
            if counts["total"] > 0:
                results.per_intent_type[intent_type] = {
                    "exact_match": counts["exact"] / counts["total"],
                    "normalized_match": counts["normalized"] / counts["total"],
                    "total": counts["total"],
                }
        
        for os_type, counts in per_os.items():
            if counts["total"] > 0:
                results.per_os[os_type] = {
                    "exact_match": counts["exact"] / counts["total"],
                    "normalized_match": counts["normalized"] / counts["total"],
                    "total": counts["total"],
                }
        
        for shell_type, counts in per_shell.items():
            if counts["total"] > 0:
                results.per_shell[shell_type] = {
                    "exact_match": counts["exact"] / counts["total"],
                    "normalized_match": counts["normalized"] / counts["total"],
                    "total": counts["total"],
                }
        
        return results
    
    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = (var_x * var_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# =============================================================================
# HuggingFace Metrics Integration
# =============================================================================

def compute_metrics_for_trainer(
    predictions: List[str],
    references: List[str],
    inputs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute metrics in a format suitable for HuggingFace Trainer.
    
    Args:
        predictions: List of generated text outputs
        references: List of reference text outputs
        inputs: List of input CanonicalIntent dictionaries
        
    Returns:
        Dictionary of metric names to values
    """
    from .data_preprocessing import parse_model_output
    
    metrics = CommandMetrics()
    
    for pred_text, ref_text, input_intent in zip(predictions, references, inputs):
        # Parse prediction
        pred_plan, _ = parse_model_output(pred_text)
        
        # Parse reference
        ref_plan, _ = parse_model_output(ref_text)
        
        if pred_plan and ref_plan:
            metrics.add_prediction(pred_plan, ref_plan, input_intent)
        else:
            # Add None for invalid predictions
            metrics.add_prediction(pred_plan, ref_plan or {}, input_intent)
    
    results = metrics.compute()
    
    return {
        "exact_match": results.exact_match,
        "normalized_match": results.normalized_match,
        "intent_preservation": results.intent_preservation,
        "entity_preservation": results.entity_preservation,
        "syntax_validity": results.syntax_validity,
        "schema_validity": results.schema_validity,
        "os_shell_compatibility": results.os_shell_compatibility,
        "json_validity": results.json_validity,
    }


# =============================================================================
# Exit Criteria Checker
# =============================================================================

def check_exit_criteria(results: MetricResults) -> Tuple[bool, List[str]]:
    """
    Check if the model meets the exit criteria.
    
    Exit Criteria:
    - Exact command match ≥ 70%
    - Normalized command match ≥ 85%
    - Intent preservation: 100%
    - Entity preservation ≥ 95%
    - Syntax validity ≥ 95%
    - OS/shell compatibility: 100%
    
    Args:
        results: Evaluation results
        
    Returns:
        Tuple of (all_passed, list_of_failures)
    """
    criteria = [
        ("Exact Match ≥ 70%", results.exact_match >= 0.70),
        ("Normalized Match ≥ 85%", results.normalized_match >= 0.85),
        ("Intent Preservation = 100%", results.intent_preservation >= 1.00),
        ("Entity Preservation ≥ 95%", results.entity_preservation >= 0.95),
        ("Syntax Validity ≥ 95%", results.syntax_validity >= 0.95),
        ("OS/Shell Compatibility = 100%", results.os_shell_compatibility >= 1.00),
    ]
    
    failures = [name for name, passed in criteria if not passed]
    all_passed = len(failures) == 0
    
    return all_passed, failures
