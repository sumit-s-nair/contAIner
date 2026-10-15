import json
import collections
from pathlib import Path

rollouts_path = Path("archive/EXPLOITED_DO_NOT_USE/run_f02eeb_trailing_comma/rollouts.jsonl")

entries = []
with open(rollouts_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

all_completions = [c for e in entries for c in e["completions"]]

parsed_ok = [c for c in all_completions if c["action_type"] != "PARSE_FAILED"]
parsed_fail = [c for c in all_completions if c["action_type"] == "PARSE_FAILED"]

print(f"Total parsed OK: {len(parsed_ok)}")
print(f"Total parsed FAIL: {len(parsed_fail)}")

# What did parsed_ok raw completions look like vs parsed_fail?
print("\n--- 5 Examples of parsed OK raw_completion ---")
for c in parsed_ok[:5]:
    print(repr(c["raw_completion"]))

print("\n--- 5 Examples of parsed FAIL raw_completion ---")
for c in parsed_fail[:5]:
    print(repr(c["raw_completion"]))
