# System 1: Intent Understanding Module

This module contains the implemented training and verification workflow for intent understanding.

## What Is Implemented

- Joint intent classification + NER training script:
  - `train_intent_classifier.py`
- Verification script for interactive and batch checks:
  - `verify_intent_classifier.py`

## Current Scope

- Input: natural-language user instruction
- Output: predicted intent label, confidence, and extracted entities
- Model family: DistilRoBERTa-based shared encoder with dual heads

## Current Labels and Entities

Intent labels used by current train split:
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

Entity types used by the NER head:
- `runtime`
- `package`
- `version`
- `virtual_env`
- `package_manager`
- `project`

## Run Training

```bash
python src/system1_intent_understanding/train_intent_classifier.py \
  --train datasets/intent-dataset/data/train.jsonl \
  --val datasets/intent-dataset/data/validation.jsonl \
  --test datasets/intent-dataset/data/test.jsonl \
  --output outputs/intent_classifier
```

## Verify a Trained Model

```bash
python src/system1_intent_understanding/verify_intent_classifier.py \
  --model_dir outputs/intent_classifier/final_model
```

Batch verification example:

```bash
python src/system1_intent_understanding/verify_intent_classifier.py \
  --model_dir outputs/intent_classifier/final_model \
  --verify_file datasets/intent-dataset/data/test.jsonl \
  --sample_size 100
```

## Artifacts

Expected exported files include:
- `intent_label_map.json`
- `ner_label_map.json`
- `training_config.json`
- tokenizer and model checkpoint files

## Dataset Inputs

Default split files used by training:
- `datasets/intent-dataset/data/train.jsonl` (3128 rows)
- `datasets/intent-dataset/data/validation.jsonl` (392 rows)
- `datasets/intent-dataset/data/test.jsonl` (390 rows)

## Next Steps

- Add a lightweight runtime inference wrapper for integration with System 2.
- Add regression tests for intent/entity behavior across common software setup requests.
