# System 1: Intent Understanding

## Overview

System 1 is responsible for intent understanding from natural-language setup instructions. It does not generate shell commands directly.

In the current repository, System 1 is implemented as a **joint intent classification + token-level NER training pipeline**.

Input:
- natural-language instruction text

Output from the trained model:
- intent label
- confidence score
- extracted entity tokens

## Current Implementation Status

Implemented in this repository:
- Joint intent + NER training pipeline in `src/system1_intent_understanding/train_intent_classifier.py`
- Model verification utility in `src/system1_intent_understanding/verify_intent_classifier.py`
- Exported trained model artifacts in `outputs/intent_classifier/final_model/`

Not yet implemented in this repository:
- End-to-end runtime orchestrator that packages output into full CanonicalIntent contracts for System 2
- Clarification-loop service for low-confidence or underspecified inputs

---

## Model Design (Current)

Training script uses:
- Base encoder: `distilroberta-base`
- Shared encoder with two heads:
  - intent classification head
  - BIO-tag token classification head for entities

Entity types in model labels:
- `runtime`
- `package`
- `version`
- `virtual_env`
- `package_manager`
- `project`

Intent labels observed in current train split:
- `install_package`
- `update_package`
- `remove_package`
- `install_runtime`
- `update_runtime`
- `remove_runtime`
- `check_version`
- `check_installed`
- `list_dependencies`
- `create_isolation`

---

## Training Data Contract

System 1 training reads:
- `datasets/intent-dataset/data/train.jsonl`
- `datasets/intent-dataset/data/validation.jsonl`
- `datasets/intent-dataset/data/test.jsonl`

Current row counts:
- Train: 3128
- Validation: 392
- Test: 390

Expected row fields:
- `instruction`
- `intent_type`
- `entities`
- `entity_spans`
- `context`
- `paraphrase_group`

---

## CLI Usage

Train model:

```bash
python src/system1_intent_understanding/train_intent_classifier.py \
  --train datasets/intent-dataset/data/train.jsonl \
  --val datasets/intent-dataset/data/validation.jsonl \
  --test datasets/intent-dataset/data/test.jsonl \
  --output outputs/intent_classifier
```

Verify model:

```bash
python src/system1_intent_understanding/verify_intent_classifier.py \
  --model_dir outputs/intent_classifier/final_model
```

---

## See Also

- [System 2 Documentation](../system-2-command-generation/README.md) - Command generation
- [Dataset Documentation](../datasets/README.md) - Training data preparation
- [Integration Guide](../integration/README.md) - How systems connect
- [Architecture Diagram](../ARCHITECTURE_DIAGRAM.md) - Visual overview
