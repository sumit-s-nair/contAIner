import json
import collections
from pathlib import Path

rollouts_path = Path("archive/EXPLOITED_DO_NOT_USE/run_f02eeb_trailing_comma/rollouts.jsonl")

step_ok = collections.defaultdict(int)
step_fail = collections.defaultdict(int)

with open(rollouts_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            step_idx = entry["step_idx"]
            for c in entry["completions"]:
                if c["action_type"] == "PARSE_FAILED":
                    step_fail[step_idx] += 1
                else:
                    step_ok[step_idx] += 1

print("Step index breakdown across all episodes:")
for step_idx in sorted(step_ok.keys() | step_fail.keys()):
    print(f"Step {step_idx}: ok={step_ok[step_idx]}, fail={step_fail[step_idx]}")
