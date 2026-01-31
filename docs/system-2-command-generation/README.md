# System 2: Command Generation

## Overview

System 2 is responsible for **generating OS-specific, executable shell commands** from structured intents. It receives validated CanonicalIntent objects from System 1 and produces CommandPlan objects with runnable commands.

**Purpose**: Convert structured intent (CanonicalIntent) into executable commands for specific OS/shell combinations.

**Input**: CanonicalIntent (from System 1)  
**Output**: CommandPlan (with executable shell commands)

---

## Architecture

System 2 uses a single trained model (CodeT5+) that has been fine-tuned on multiple command generation datasets.

### Model: CodeT5+ (770M parameters)

- **Base Model**: Salesforce CodeT5+ (pre-trained on code)
- **Fine-tuning Task**: Conditional text generation (intent → command)
- **Training Data**: ~5,000+ samples from normalized datasets
- **Conditioning**: OS type + shell type + intent + entities
- **Output**: Shell commands with proper syntax for target environment

**Why CodeT5+?**
- Pre-trained on code (understands shell syntax)
- Strong at cross-platform generation
- Handles edge cases better than rule-based systems
- Can learn from examples (NL2Bash, NL2SH-ALFA)

---

## Input Contract: CanonicalIntent

System 2 expects a fully validated CanonicalIntent object:

```json
{
  "intent_type": "install_runtime",
  "entities": {
    "runtime": "python",
    "package": null,
    "version": "3.10"
  },
  "scope": "system",
  "os_hint": "windows",
  "shell_type": "powershell",
  "confidence": 0.92,
  "missing_fields": [],
  "needs_clarification": false,
  "clarification_question": null
}
```

**Validation Rules**:
- [x] `needs_clarification` must be `false`
- [x] `missing_fields` must be empty `[]`
- [x] `intent_type` must be atomic (not hierarchical)
- [x] `os_hint` and `shell_type` must be non-null

---

## Output Contract: CommandPlan

System 2 produces a structured CommandPlan:

```json
{
  "intent_type": "install_runtime",
  "entities": {
    "runtime": "python",
    "version": "3.10"
  },
  "os": "windows",
  "shell": "powershell",
  "steps": [
    {
      "step_number": 1,
      "type": "install",
      "command": "winget install Python.Python.3.10",
      "description": "Install Python 3.10 using winget"
    }
  ],
  "confidence": 0.93,
  "requires_elevation": false
}
```

**Field Descriptions**:
- `intent_type`: Matches input intent
- `entities`: Enriched entities (may add defaults)
- `os`: Resolved OS (windows/linux/macos)
- `shell`: Resolved shell (bash/powershell/cmd/zsh)
- `steps`: Ordered array of commands to execute
- `confidence`: Model's confidence in generated commands
- `requires_elevation`: Whether sudo/admin rights needed

---

## Training Pipeline

### Dataset Preparation

**Dataset Normalization**
- **Input**: NL2Bash (~10K samples), NL2SH-ALFA datasets
- **Process**: Convert to canonical format with OS/shell metadata
- **Output**: Normalized dataset (~3,500 samples)

**Dataset Augmentation (Windows)**
- **Input**: Normalized dataset
- **Process**: Add Windows-specific command equivalents
- **Output**: Augmented dataset (~5,000+ samples)

See [Dataset Documentation](../datasets/README.md) for details.

### Model Fine-tuning

**Base Model**: CodeT5+ (Salesforce/codet5p-770m)

**Training Configuration**:
- Learning rate: 5e-5
- Batch size: 8 per device
- Epochs: ~10 with early stopping
- Patience: 3 epochs
- Evaluation metric: Normalized command match

**Prompt Format**:
The model is trained to generate commands from structured prompts containing:
- Intent type
- Entities (runtime, package, version)
- OS hint (windows/linux/macos)
- Shell type (bash/powershell/cmd/zsh)

---

## Inference Pipeline

The model generates commands by:
The model generates commands by:
1. Validating input CanonicalIntent (must have no missing fields or clarifications needed)
2. Creating a structured prompt from intent metadata
3. Generating shell command using fine-tuned CodeT5+ model
4. Validating command syntax for target shell
5. Packaging result into CommandPlan with metadata

**Output**: CommandPlan object with executable commands ready for Stage 3 (execution, out of scope)

---

## Cross-Platform Command Generation

System 2 handles multiple OS/shell combinations:

### Windows
- **PowerShell**: winget-based installations, PowerShell cmdlets
- **CMD**: choco (Chocolatey) for package management

### Linux
- **Bash (apt)**: apt-based installations (Debian/Ubuntu)
- **Bash (yum)**: yum-based installations (RHEL/CentOS)
- **Bash (dnf)**: dnf-based installations (Fedora)

### macOS
- **Bash/Zsh**: brew (Homebrew) for package management

The model learns OS/shell-specific patterns and command structures from the training data.

---

## Command Validation

System 2 includes command validators to ensure generated commands are safe and correct.

### Syntax Validation
- Validates command syntax for target shell (bash/powershell/cmd/zsh)
- Checks for unmatched quotes, invalid redirects, and malformed commands
- Ensures commands use valid flags and arguments

### Safety Checks
- Determines if command requires elevated permissions (sudo/admin)
- Flags potentially dangerous operations (system modifications, deletions)
- Validates package names and versions exist

---

## Performance Characteristics

| Metric | Value | Target |
|--------|-------|--------|
| Model Size | 1.5GB | N/A |
| Inference Latency | 200-400ms | <500ms |
| Exact Command Match | 70%+ | ≥70% |
| Normalized Match | 85%+ | ≥85% |
| Syntax Validity | 95%+ | ≥95% |
| OS Compatibility | 100% | 100% |
| Shell Compatibility | 100% | 100% |

---

## Training Approach

### Dataset Preparation
- Normalize NL2Bash + NL2SH-ALFA datasets into consistent format
- Augment with Windows-specific commands (winget, PowerShell)
- Split into train/val/test (70/15/15) stratified by OS and intent type

### Fine-tuning Strategy
- Base model: Salesforce CodeT5+ (770M params, pre-trained on code)
- Task: Seq2seq conditional generation (CanonicalIntent → command)
- Conditioning: OS type, shell type, intent, entities
- Epochs: ~10 epochs with early stopping
- Evaluation metrics: exact match, normalized match, syntax validity

### Hybrid Approach (Optional)
For common deterministic cases, combine ML model with rule-based templates:
- Fast path: Use templates for frequent intents (install_package on Windows → winget)
- Fallback: Use ML model for complex or ambiguous cases

### Retrieval-Augmented Generation (Optional)
Use vector database to find similar training examples as context for generation:
- Query: Find top-k similar CanonicalIntents from training set
- Context: Provide their commands as examples to the model
- Generate: Produce command with improved accuracy

---

## Integration with System 1

System 2 receives CanonicalIntent objects from System 1 that have been fully validated:
- All required fields are populated
- `needs_clarification` is `false`
- `missing_fields` is empty
- Intent type is atomic (not hierarchical)

System 2 processes each CanonicalIntent independently and returns a CommandPlan ready for execution (Stage 3, out of scope).

---

## Implementation

The training implementation is located at `src/system2_command_generation/`. 

### Quick Start

```bash
# Install dependencies
pip install -r src/system2_command_generation/requirements.txt

# Train CodeT5+ (primary model)
python -m src.system2_command_generation.train --model codet5plus

# Train FLAN-T5 (baseline comparison)
python -m src.system2_command_generation.train --model flan_t5 --baseline
```

### Module Structure

| File | Description |
|------|-------------|
| `config.py` | Training configuration, schemas, and hyperparameters |
| `data_preprocessing.py` | Data loading and CanonicalIntent → CommandPlan transformation |
| `models.py` | CodeT5+ and FLAN-T5 model loading and inference |
| `metrics.py` | Evaluation metrics (exact match, normalized match, etc.) |
| `train.py` | Main training script with HuggingFace Trainer |

See [src/system2_command_generation/README.md](../../src/system2_command_generation/README.md) for detailed usage.

---

## See Also

- [System 1 Documentation](../system-1-intent-understanding/README.md) - Intent understanding
- [Dataset Documentation](../datasets/README.md) - Training data preparation
- [Integration Guide](../integration/README.md) - How systems connect
- [Architecture Diagram](../ARCHITECTURE_DIAGRAM.md) - Visual overview
