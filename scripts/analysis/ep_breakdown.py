import json
import collections
from pathlib import Path

rollouts_path = Path("archive/EXPLOITED_DO_NOT_USE/run_f02eeb_trailing_comma/rollouts.jsonl")

ep_parsed_ok = collections.defaultdict(int)
ep_parsed_fail = collections.defaultdict(int)

with open(rollouts_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            ep = entry["episode"]
            for c in entry["completions"]:
                if c["action_type"] == "PARSE_FAILED":
                    ep_parsed_fail[ep] += 1
                else:
                    ep_parsed_ok[ep] += 1

print("Episode breakdown (first 30 episodes):")
for ep in range(1, 31):
    ok = ep_parsed_ok[ep]
    fail = ep_parsed_fail[ep]
    print(f"Ep {ep:2d}: ok={ok:2d}, fail={fail:2d}")

print("\nSummary across all 202 episodes:")
print("Episodes with AT LEAST ONE ok:", sum(1 for ep in range(1, 203) if ep_parsed_ok[ep] > 0))
print("Episodes with ZERO ok (100% fail):", sum(1 for ep in range(1, 203) if ep_parsed_ok[ep] == 0))
