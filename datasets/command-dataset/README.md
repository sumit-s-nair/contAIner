# Command Dataset

A dataset mapping intents to executable shell commands for container/dev-environment scenarios.

## Purpose

This dataset provides **command-level supervision** for generating shell commands from structured intents. Each row contains a canonicalized instruction, intent metadata, and the corresponding command.

## Schema

Each JSONL row follows:

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | Canonicalized instruction for the intent |
| `intent_type` | string | Classified intent (e.g., `install_package`, `create_environment`) |
| `entities` | object | Extracted entities: `runtime`, `package`, `version` (nullable) |
| `os` | string | Target operating system (e.g., `linux`, `macos`, `windows`) |
| `shell` | string | Target shell (e.g., `bash`, `zsh`, `powershell`) |
| `command` | string | The executable command or serialized step |
| `source` | string | Data origin (e.g., `NL2SH-ALFA`, `manual`, `generated`) |

## Splits

| Split | File |
|-------|------|
| Train | `data/train.jsonl` |
| Validation | `data/validation.jsonl` |
| Test | `data/test.jsonl` |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("sumit-s-nair/command-dataset")
print(dataset["train"][0])
```

## Schema Validation

```bash
python scripts/validate_schema.py data/train.jsonl
```

## Contributing

1. Follow the schema exactly—do not add or rename fields.
2. Use `null` for missing optional entity values.
3. Ensure commands are valid for the specified `os` and `shell`.
4. Validate your additions before submitting.

## License

Apache License 2.0. See [LICENSE](LICENSE).
