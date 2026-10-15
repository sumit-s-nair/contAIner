import json
import collections
from pathlib import Path

rollouts_path = Path("archive/EXPLOITED_DO_NOT_USE/run_f02eeb_trailing_comma/rollouts.jsonl")

with open(rollouts_path, "r", encoding="utf-8") as f:
    first_line = f.readline()
    sample_entry = json.loads(first_line)

sample_completion = sample_entry["completions"][0]
print("COMPLETION_KEYS:", sorted(list(sample_completion.keys())))
print("SAMPLE_COMPLETION:")
print(json.dumps(sample_completion, indent=2))
