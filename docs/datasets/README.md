# Dataset Documentation

## Overview

contAIner uses **two separate datasets** for training its two systems:

1. **Intent Parser Dataset** (7,000 samples) - For System 1
2. **Command Generation Dataset** (5,000+ samples) - For System 2

These datasets are completely separate and serve different purposes.

---

## Dataset 1: Intent Parser Dataset (System 1)

### Purpose
Train System 1 components (intent classification and entity extraction).

### Size
7,000 annotated samples

### Schema

```json
{
  "instruction": "install python 3.10",
  "intent_type": "install_runtime",
  "entities": {
    "runtime": "python",
    "package": null,
    "version": "3.10"
  },
  "entity_spans": {
    "runtime": {"start": 8, "end": 14, "text": "python"},
    "version": {"start": 15, "end": 19, "text": "3.10"}
  },
  "context": {
    "os_hint": "windows",
    "shell_type": "powershell"
  },
  "paraphrase_group": "install_python_001"
}
```

### Data Sources

| Source | Samples | Method |
|--------|---------|--------|
| NL2Bash Dataset | 1,500 | Filter and map to our intent types |
| GPT-4 Synthetic | 3,000 | Generate variations for each intent |
| Manual Curation | 500 | Edge cases and ambiguous phrasings |
| Stack Overflow | 500 | Extract question titles |
| Augmentation | 1,000 | Paraphrase, synonym replacement |
| **Total** | **7,000** | |

### Collection Steps

The intent parser dataset is collected from multiple sources:

1. **NL2Bash Filtering**: Download NL2Bash dataset and filter for relevant intent types (install_package, install_runtime, check_version, etc.)

2. **GPT-4 Synthetic Generation**: Generate 3,000 variations for each intent type using GPT-4 to create diverse phrasings

3. **Manual Curation**: Create 500 edge case samples including ambiguous phrasings, typos, and informal language

4. **Augmentation**: Apply paraphrasing, synonym replacement, and entity removal to expand dataset coverage

5. **Train/Val/Test Split**: Split into 70% train, 15% validation, 15% test, stratified by intent_type with paraphrase groups kept together

### Quality Requirements

- [x] Minimum 500 samples per intent type
- [x] Minimum 300 samples for each entity type
- [x] At least 1,000 samples with missing entities (nulls)
- [x] At least 500 samples with ambiguous phrasing
- [x] No duplicate instructions within same split
- [x] Entity spans must match substring exactly
- [x] Paraphrase groups don't span train/test splits

### File Structure

```
intent_dataset/
├── intent_train.jsonl    (4,900 samples)
├── intent_val.jsonl      (1,050 samples)
├── intent_test.jsonl     (1,050 samples)
├── statistics.json       (intent distribution, entity frequency)
└── README.md             (dataset documentation)
```

### Usage in Training

This dataset trains:
- **Stage 1A**: Intent Classification (DistilBERT)
- **Stage 1B**: Entity Extraction (BERT-NER)
- **Stage 1E**: Clarification Generation (T5-small)

---

## Dataset 2: Command Generation Dataset (System 2)

### Purpose
Train System 2 command generator (CodeT5+).

### Size
5,000+ annotated samples

### Schema

```json
{
  "instruction": "install git",
  "intent_type": "install_package",
  "entities": {
    "package": "git",
    "runtime": null,
    "version": null
  },
  "os": "linux",
  "shell": "bash",
  "command": "sudo apt install git -y",
  "canonical_form": "apt install git"
}
```

### Data Sources

| Source | Samples | Method |
|--------|---------|--------|
| NL2Bash (normalized) | 1,500 | Convert to canonical format |
| NL2SH-ALFA | 2,000 | Multi-shell examples |
| Windows Augmentation | 1,500 | Add PowerShell/winget variants |
| **Total** | **5,000+** | |

### Preparation Steps

1. **Normalize NL2Bash Dataset**: Convert NL2Bash and NL2SH-ALFA datasets to canonical format with OS/shell metadata. Extract intent_type from command patterns and add entities.

2. **Add Windows Commands**: Augment dataset with Windows equivalents using PowerShell and cmd. Map Linux package managers (apt, yum) to Windows alternatives (winget, choco).

3. **Validate Commands**: Check all commands for syntax validity and executability on target platforms.

4. **Train/Val/Test Split**: Split into 70% train, 15% validation, 15% test, stratified by OS and intent_type.

### Quality Requirements

- [x] All commands are valid for target OS/shell
- [x] Commands have been manually verified or tested
- [x] Balanced across OS types (Linux: 60%, Windows: 30%, macOS: 10%)
- [x] Balanced across shell types within each OS
- [x] No duplicate commands within same split
- [x] Intent_type distribution matches real-world usage

### File Structure

```
command_dataset/
├── command_train.jsonl   (3,500 samples)
├── command_val.jsonl     (750 samples)
├── command_test.jsonl    (750 samples)
├── statistics.json       (OS/shell/intent distribution)
└── README.md             (dataset documentation)
```

### Usage in Training

This dataset trains:
- **System 2**: Command Generator (CodeT5+)

The dataset is formatted as prompt-response pairs where the prompt contains intent metadata (intent_type, entities, OS, shell) and the response is the executable shell command.

---

## Dataset Comparison

| Aspect | Intent Parser Dataset | Command Generation Dataset |
|--------|----------------------|---------------------------|
| **Purpose** | Train System 1 (understand intent) | Train System 2 (generate commands) |
| **Size** | 7,000 samples | 5,000+ samples |
| **Key Fields** | instruction, intent_type, entities, entity_spans | instruction, intent_type, entities, command, os, shell |
| **Training** | DistilBERT, BERT-NER, T5-small | CodeT5+ |
| **Sources** | NL2Bash, GPT-4, manual, augmentation | NL2Bash, NL2SH-ALFA, Windows manual |
| **Validation** | Entity spans, no duplicates | Command syntax, executability |

---

## Data Augmentation Techniques

### For Intent Parser Dataset

1. **Paraphrasing**: "install python" → "set up python", "get python"
2. **Entity Removal**: "install python 3.10" → "install python"
3. **Synonym Replacement**: "nodejs" → "node", "py" → "python"
4. **Question Form**: "install git" → "how do I install git"
5. **Context Addition**: "install git" → "install git on my machine"

### For Command Generation Dataset

1. **OS Conversion**: Linux commands → Windows equivalents
2. **Shell Conversion**: bash → powershell, cmd
3. **Package Manager Mapping**: apt → yum, dnf, winget, brew
4. **Version Parameterization**: python3.10 → python3.11, python3.9

---

## See Also

- [System 1 Documentation](../system-1-intent-understanding/README.md) - Uses intent parser dataset
- [System 2 Documentation](../system-2-command-generation/README.md) - Uses command generation dataset
- [Architecture Diagram](../ARCHITECTURE_DIAGRAM.md) - Visual architecture overview
