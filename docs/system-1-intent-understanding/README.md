# System 1: Intent Understanding

## Overview

System 1 is responsible for **validating and structuring user intent** from natural language input. It does NOT generate commands - that's System 2's job.

**Purpose**: Convert raw user text into structured `CanonicalIntent` objects that System 2 can process.

**Input**: Natural language text (e.g., "install python 3.10")  
**Output**: CanonicalIntent (structured JSON with intent type, entities, OS/shell context)

---

## Components (Development Stages)

System 1 consists of 5 sequential stages:

### Stage 1A: Intent Classification
**Purpose**: Determine WHAT the user wants to do

- **Model**: DistilBERT (110M parameters)
- **Input**: Raw user text
- **Output**: Intent type + confidence score
- **Training**: 7,000 samples from intent parser dataset
- **Intent Types**: install_package, install_runtime, setup_environment, check_version, etc.

**Example**:
```
Input: "install python"
Output: {intent_type: "install_runtime", confidence: 0.92}
```

---

### Stage 1B: Entity Extraction
**Purpose**: Extract structured information from user input

- **Model**: BERT-NER (token classification with BIO tagging)
- **Input**: Raw user text
- **Output**: Entities (runtime, package, version)
- **Training**: 7,000 samples with entity span annotations

**Example**:
```
Input: "install python 3.10"
Output: {
  runtime: "python",
  package: null,
  version: "3.10"
}
```

---

### Stage 1C: OS & Shell Detection
**Purpose**: Determine user's operating system and shell environment

- **Method**: System APIs (100% deterministic, no ML)
- **APIs Used**: 
  - `platform.system()` → OS detection
  - `os.environ["SHELL"]` → Shell detection
  - `PSModulePath` check → PowerShell detection
- **Output**: os_hint + shell_type

**Example**:
```
Windows system with PowerShell:
Output: {os_hint: "windows", shell_type: "powershell"}

Linux system with bash:
Output: {os_hint: "linux", shell_type: "bash"}
```

---

### Stage 1D: Hierarchical Decomposition
**Purpose**: Break down complex intents into atomic steps

- **Method**: Rule-based pattern matching
- **Input**: Hierarchical intent (e.g., "setup_environment")
- **Output**: Array of atomic CanonicalIntents
- **Why**: Some requests like "set up dev environment" need multiple commands

**Example**:
```
Input: "set up python dev environment"
Output: [
  CanonicalIntent(install_runtime, python, 3.10),
  CanonicalIntent(install_package, pip),
  CanonicalIntent(create_virtual_env, python)
]
```

**Decomposition Patterns**:
- `setup_environment` → install runtime + packages + config
- `setup_project` → init directory + install deps + create config files
- `fix_environment` → check versions + reinstall broken deps

---

### Stage 1E: Clarification Handling
**Purpose**: Determine if user input is complete or needs more information

- **Model**: T5-small (question generation)
- **Input**: CanonicalIntent with potential missing fields
- **Output**: needs_clarification flag + optional question
- **Confidence Thresholds**:
  - confidence < 0.70 → always ask for clarification
  - confidence ≥ 0.85 → proceed without clarification
  - 0.70-0.85 + missing fields → ask for clarification

**Example**:
```
Input: "install python" (no version specified)
Output: {
  needs_clarification: true,
  clarification_question: "Which Python version? (3.10, 3.11, latest)"
}

After user responds: "3.10"
Output: {
  needs_clarification: false,
  entities: {runtime: "python", version: "3.10"}
}
```

---

## Output Format: CanonicalIntent

All stages combine to produce a structured CanonicalIntent:

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

**This object is passed to System 2 for command generation.**

---

## Integration with System 2

**Interface Contract**: CanonicalIntent object

System 1 must ensure:
- [x] `needs_clarification` is `false`
- [x] `missing_fields` is empty `[]`
- [x] `intent_type` is atomic (hierarchical intents already decomposed)
- [x] All required fields are populated

---

## Performance Characteristics

| Stage | Model Size | Latency | Accuracy |
|-------|------------|---------|----------|
| 1A: Intent Classification | 260MB | 50ms | 92%+ |
| 1B: Entity Extraction | 420MB | 80ms | 88%+ F1 |
| 1C: OS/Shell Detection | N/A | 5ms | 100% |
| 1D: Decomposition | N/A | 10ms | Rule-based |
| 1E: Clarification | 240MB | 50ms | 80%+ |
| **Total** | **~920MB** | **~200ms** | **N/A** |

---

## Development Stages Summary

1. **Stage 1A**: Intent Classification → Classify user input into intent types
2. **Stage 1B**: Entity Extraction → Extract runtime/package/version
3. **Stage 1C**: OS/Shell Detection → Detect system environment
4. **Stage 1D**: Hierarchical Decomposition → Break complex intents into atomic steps
5. **Stage 1E**: Clarification Handling → Determine if more info needed

**Output**: CanonicalIntent(s) ready for System 2

---

## See Also

- [System 2 Documentation](../system-2-command-generation/README.md) - Command generation
- [Dataset Documentation](../datasets/README.md) - Training data preparation
- [Integration Guide](../integration/README.md) - How systems connect
- [Architecture Diagram](../ARCHITECTURE_DIAGRAM.md) - Visual overview
