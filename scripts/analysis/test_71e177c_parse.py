import json
import collections
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.system2_planner.models import PlannedStep, ActionType

rollouts_path = Path("archive/EXPLOITED_DO_NOT_USE/run_f02eeb_trailing_comma/rollouts.jsonl")

entries = []
with open(rollouts_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

all_completions = [c for e in entries for c in e["completions"]]

# Test the parse block from 71e177c evaluate_completion:
# try:
#     start_idx = comp.find('{')
#     end_idx = comp.rfind('}')
#     if start_idx == -1 or end_idx == -1:
#         raise ValueError("No JSON object found")
#     step_dict = json.loads(comp[start_idx:end_idx+1])
#     atype_str = step_dict.get("action_type", "CHECK").upper()
#     try:
#         action_type = ActionType(atype_str.lower())
#     except ValueError:
#         action_type = ActionType.CHECK
#     action = PlannedStep(...)

parse_success = 0
parse_exceptions = collections.Counter()

for c in all_completions:
    comp = c["raw_completion"]
    try:
        start_idx = comp.find('{')
        end_idx = comp.rfind('}')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found")
        step_dict = json.loads(comp[start_idx:end_idx+1])
        atype_str = step_dict.get("action_type", "CHECK").upper()
        try:
            action_type = ActionType(atype_str.lower())
        except ValueError:
            action_type = ActionType.CHECK
        action = PlannedStep(
            action_type=action_type,
            target=step_dict.get("target", ""),
            description=step_dict.get("description", ""),
            rationale=step_dict.get("rationale", "")
        )
        parse_success += 1
    except Exception as e:
        parse_exceptions[type(e).__name__ + ": " + str(e).split("\n")[0]] += 1

print(f"71e177c parse block success: {parse_success}/{len(all_completions)} ({parse_success/len(all_completions)*100:.2f}%)")
print("71e177c parse block exceptions:")
for k, v in parse_exceptions.most_common(10):
    print(f"  [{v:4d}] {k}")
