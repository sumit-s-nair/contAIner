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

for target_atype in ["install", "isolate", "check"]:
    matches = [c for c in all_completions if c["action_type"] == target_atype]
    statuses = collections.Counter(c["exec_status"] for c in matches)
    print(f"\n--- action_type = {target_atype!r} (total {len(matches)}) ---")
    for st, cnt in statuses.most_common():
        print(f"  {st}: {cnt}")

install_matches = [c for c in all_completions if c["action_type"] == "install"]
install_success = [c for c in install_matches if c["exec_status"] in ("success", "SUCCESS")]
install_failed = [c for c in install_matches if c["exec_status"] in ("failed", "FAILED")]

print(f"\nTotal install success: {len(install_success)}")
print(f"Total install failed: {len(install_failed)}")

if install_success:
    print("\n=== 2 Full INSTALL Success Records ===")
    for i, c in enumerate(install_success[:2], 1):
        print(f"\n--- Success Example {i} ---")
        print(json.dumps(c, indent=2))
else:
    print("\nNo install success records found! All statuses for install:")
    print(collections.Counter(c["exec_status"] for c in install_matches))

print("\n=== 2 Full INSTALL Failed Records ===")
for i, c in enumerate(install_failed[:2], 1):
    print(f"\n--- Failure Example {i} ---")
    print(json.dumps(c, indent=2))
