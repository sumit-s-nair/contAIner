#!/usr/bin/env python3
"""
scripts/analysis/validate_parser_on_rollouts.py
Offline parser validation against archive rollouts from run_f02eeb.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_env.parser import parse_planned_step


def main():
    rollouts_path = (
        PROJECT_ROOT
        / "archive"
        / "EXPLOITED_DO_NOT_USE"
        / "run_f02eeb_trailing_comma"
        / "rollouts.jsonl"
    )

    if not rollouts_path.exists():
        print(f"Error: {rollouts_path} not found.")
        sys.exit(1)

    total_completions = 0
    success_a = 0
    success_b = 0

    fixed_by_b = 0
    rejected_by_b_parsed_by_a = []

    b_failures = []
    b_action_types = Counter()

    with open(rollouts_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            completions = entry.get("completions", [])
            for comp in completions:
                total_completions += 1
                raw = comp.get("raw_completion", "")

                # (a) plain json.loads
                parsed_a = None
                try:
                    parsed_a = json.loads(raw)
                    success_a += 1
                except Exception:
                    pass

                # (b) src.rl_env.parser.parse_planned_step
                parsed_b = None
                try:
                    parsed_b = parse_planned_step(raw)
                    success_b += 1
                    b_action_types[parsed_b.action_type.value] += 1
                except Exception as exc:
                    exc_bucket = f"{type(exc).__name__}: {str(exc)}"
                    b_failures.append((exc_bucket, raw))

                # Compare
                if parsed_a is None and parsed_b is not None:
                    fixed_by_b += 1

                if parsed_a is not None and parsed_b is None:
                    rejected_by_b_parsed_by_a.append((parsed_a, raw))

    pct_a = (success_a / total_completions * 100) if total_completions else 0.0
    pct_b = (success_b / total_completions * 100) if total_completions else 0.0

    print("=== OFFLINE PARSER VALIDATION RESULTS ===")
    print(f"Total completions: {total_completions} (matches expected 6472: {total_completions == 6472})")
    print(f"(a) Plain json.loads success: {success_a}/{total_completions} ({pct_a:.2f}%)")
    print(f"(b) parse_planned_step success: {success_b}/{total_completions} ({pct_b:.2f}%)")
    print(f"Completions where (a) failed but (b) fixed: {fixed_by_b}")
    print(f"Completions where (a) parsed but (b) rejected: {len(rejected_by_b_parsed_by_a)}")

    print("\n--- Distribution of action_type among (b) successes ---")
    for act, count in b_action_types.most_common():
        print(f"  {act}: {count} ({count/success_b*100:.2f}%)")

    # Bucketing failures of (b)
    bucket_counts = Counter(b[0] for b in b_failures)
    print("\n--- Top 10 failure buckets for (b) ---")
    for bucket, count in bucket_counts.most_common(10):
        print(f"  [{count} occurrences] {bucket}")

    print("\n--- 3 raw_completion examples from each of the Top 3 failure buckets ---")
    top_3_buckets = [b for b, _ in bucket_counts.most_common(3)]
    for idx, bucket in enumerate(top_3_buckets, 1):
        print(f"\nBucket {idx}: {bucket}")
        examples = [raw for b_name, raw in b_failures if b_name == bucket][:3]
        for e_idx, ex in enumerate(examples, 1):
            print(f"  Example {e_idx}: {repr(ex)}")

    print("\n--- 3 examples where (a) parsed but (b) rejected ---")
    for idx, (a_obj, raw) in enumerate(rejected_by_b_parsed_by_a[:3], 1):
        print(f"  Example {idx}:")
        print(f"    json.loads object: {a_obj}")
        print(f"    raw_completion: {repr(raw)}")


if __name__ == "__main__":
    main()
