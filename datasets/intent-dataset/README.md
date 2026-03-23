# Intent Dataset

A dataset for intent understanding in container/dev-environment command scenarios.

## Purpose

This dataset captures **natural language instructions** and maps them to structured intents with extracted entities. It does **not** contain commands—only user-facing language and its semantic decomposition.

## Progress

- Train/validation/test JSONL splits are present.
- Validation script is available in `scripts/validate_schema.py`.
- Dataset publishing workflow is available through `scripts/push_datasets_to_hf.py`.

## Current Data Snapshot

Local files in `data/` currently contain:
- `train.jsonl`: 3128 rows
- `validation.jsonl`: 392 rows
- `test.jsonl`: 390 rows
- `combined.jsonl`: 12990 rows (source pool, not a training split)

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

## Schema

Each JSONL row follows:

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | Raw user instruction |
| `intent_type` | string | Classified intent (e.g., `install_package`, `create_environment`) |
| `entities` | object | Extracted entities. Common keys: `runtime`, `package`, `version`, `virtual_env`, `package_manager`, `project` |
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

Notes:
- `entities` can vary by intent and may omit keys that are not relevant.
- `entity_spans` may be partial (for example, only one entity span may be present).
- `context` usually includes `os_hint` and `shell_type`, often set to `null`.

## Splits

| Split | File |
|-------|------|
| Train | `data/train.jsonl` (3128 rows) |
| Validation | `data/validation.jsonl` (392 rows) |
| Test | `data/test.jsonl` (390 rows) |
| Combined (unsplit source pool) | `data/combined.jsonl` (12990 rows) |

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
2. Keep `intent_type` values within the label set used by the training pipeline.
3. Use `null` for missing optional values when keys are present.
4. Validate your additions before submitting.

## License

Apache License 2.0. See [LICENSE](LICENSE).
