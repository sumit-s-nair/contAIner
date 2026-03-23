#!/usr/bin/env python3
"""Validate command-dataset JSONL files against `schema.json`.

The script checks each non-empty line, reports JSON/schema violations, and
returns a non-zero exit code when errors are found.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema package required. Install with: pip install jsonschema")
    sys.exit(1)


def load_schema(schema_path: Path) -> dict:
    """Load JSON schema from file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_jsonl(jsonl_path: Path, schema: dict) -> tuple[int, int, list]:
    """
    Validate each line in a JSONL file against the schema.

    Returns:
        (total_lines, valid_lines, errors)
    """
    errors = []
    total = 0
    valid = 0

    if not jsonl_path.exists():
        return 0, 0, [f"File not found: {jsonl_path}"]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            total += 1
            try:
                row = json.loads(line)
                jsonschema.validate(instance=row, schema=schema)
                valid += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
            except jsonschema.ValidationError as e:
                errors.append(f"Line {line_num}: Schema violation - {e.message}")

    return total, valid, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSONL file against command-dataset schema"
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the JSONL file to validate",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Path to schema.json (default: auto-detect relative to script)",
    )
    args = parser.parse_args()

    # Locate schema
    if args.schema:
        schema_path = args.schema
    else:
        script_dir = Path(__file__).parent
        schema_path = script_dir.parent / "schema.json"

    if not schema_path.exists():
        print(f"Error: Schema not found at {schema_path}")
        sys.exit(1)

    schema = load_schema(schema_path)
    total, valid, errors = validate_jsonl(args.jsonl_file, schema)

    # Report
    print(f"\nValidation Report: {args.jsonl_file}")
    print("=" * 50)
    print(f"Total rows:  {total}")
    print(f"Valid rows:  {valid}")
    print(f"Errors:      {len(errors)}")

    if errors:
        print("\nErrors:")
        for err in errors[:20]:  # limit output
            print(f"  • {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        sys.exit(1)
    else:
        if total == 0:
            print("\n✓ File is empty (no rows to validate)")
        else:
            print("\n✓ All rows valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
