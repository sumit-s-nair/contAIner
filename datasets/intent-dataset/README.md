# Intent Dataset

A dataset for intent understanding in container/dev-environment command scenarios.

## Purpose

This dataset captures **natural language instructions** and maps them to structured intents with extracted entities. It does **not** contain commands—only user-facing language and its semantic decomposition.

## Schema

Each JSONL row follows:

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | Raw user instruction |
| `intent_type` | string | Classified intent (e.g., `install_package`, `create_environment`) |
| `entities` | object | Extracted entities: `runtime`, `package`, `version` (nullable) |
| `entity_spans` | object | Character spans for each entity (nullable per key) |
| `context` | object | Optional hints: `os_hint`, `shell_type` |
| `paraphrase_group` | string \| null | Groups paraphrased variants of the same intent |

### Entity Span Format

```json
{
  "start": 10,
  "end": 16,
  "text": "python"
}
```

## Splits

| Split | File |
|-------|------|
| Train | `data/train.jsonl` |
| Validation | `data/validation.jsonl` |
| Test | `data/test.jsonl` |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("sumit-s-nair/intent-dataset")
print(dataset["train"][0])
```

## Schema Validation

```bash
python scripts/validate_schema.py data/train.jsonl
```

## Contributing

1. Follow the schema exactly—do not add or rename fields.
2. Use `null` for missing optional values.
3. Validate your additions before submitting.

## License

Apache License 2.0. See [LICENSE](LICENSE).
