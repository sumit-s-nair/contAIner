# Dataset Documentation

## Overview

contAIner uses two separate datasets:
- Intent dataset for System 1 intent understanding
- Command dataset for System 2 command generation

Both are stored locally under `datasets/` and can be pushed to Hugging Face with `scripts/push_datasets_to_hf.py`.

---

## Intent Dataset (System 1)

Path: `datasets/intent-dataset/`

### Current Local Split Sizes

- `data/train.jsonl`: 3128 rows
- `data/validation.jsonl`: 392 rows
- `data/test.jsonl`: 390 rows
- `data/combined.jsonl`: 12990 rows (source pool)

### Row Structure (Observed)

Each row includes:
- `instruction` (string)
- `intent_type` (string)
- `entities` (object, variable keys by intent)
- `entity_spans` (object, may be partial)
- `context` (object with `os_hint`, `shell_type`)
- `paraphrase_group` (string or null)

Most common entity keys observed in split files:
- `runtime`
- `package`
- `package_manager`
- `virtual_env`
- `project`
- `version`

Intent labels observed in train/validation/test:
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

## Command Dataset (System 2)

Path: `datasets/command-dataset/`

### Purpose

Maps normalized intent metadata plus OS/shell context to executable command strings.

### Row Structure

Each row includes:
- `instruction`
- `intent_type`
- `entities`
- `os`
- `shell`
- `command`
- `source`

---

## Validation

Use per-dataset validation scripts:

```bash
python datasets/intent-dataset/scripts/validate_schema.py datasets/intent-dataset/data/train.jsonl
python datasets/command-dataset/scripts/validate_schema.py datasets/command-dataset/data/train.jsonl
```

---

## See Also

- [System 1 Documentation](../system-1-intent-understanding/README.md)
- [System 2 Documentation](../system-2-command-generation/README.md)
- [Architecture Diagram](../ARCHITECTURE_DIAGRAM.md)
