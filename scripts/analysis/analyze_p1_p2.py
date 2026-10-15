import json
import collections
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.system2_planner.models import PlannedStep, ActionType
from src.rl_env.parser import parse_planned_step
from src.rl_env.system3_stub import expand_planned_step

rollouts_path = Path("archive/EXPLOITED_DO_NOT_USE/run_f02eeb_trailing_comma/rollouts.jsonl")

entries = []
with open(rollouts_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

all_completions = []
for entry in entries:
    for c in entry["completions"]:
        all_completions.append(c)

print(f"Total entries (steps): {len(entries)}")
print(f"Total completions: {len(all_completions)}")

# P1.a: Key set of one completion record
print("\n=== P1.a Key set ===")
print("Keys:", sorted(list(all_completions[0].keys())))

# Bucketing by (exec_status, action_type, first 120 chars of execution_error or error field)
buckets = collections.Counter()
bucket_records = collections.defaultdict(list)

for c in all_completions:
    status = c.get("exec_status", "NONE")
    atype = c.get("action_type", "NONE")
    err = c.get("execution_error", "") or c.get("validator_error", "") or ""
    err_prefix = str(err)[:120]
    key = (status, atype, err_prefix)
    buckets[key] += 1
    bucket_records[key].append(c)

print("\n=== P1.a Top 15 Buckets ===")
for i, (key, count) in enumerate(buckets.most_common(15), 1):
    status, atype, err = key
    print(f"#{i:2d} [{count:4d} completions] exec_status={status!r}, action_type={atype!r}, err={err!r}")

# P1.b: Print 3 full failed records from the largest failure bucket
top_key = buckets.most_common(1)[0][0]
print(f"\n=== P1.b 3 Full Records from Largest Bucket {top_key} ===")
for idx, r in enumerate(bucket_records[top_key][:3], 1):
    print(f"\n--- Example {idx} ---")
    print(json.dumps(r, indent=2))

# P1.d: Hypothesis test
success_pydantic = 0
exc_pydantic = collections.Counter()

success_enum_raw = 0
exc_enum_raw = collections.Counter()

for c in all_completions:
    raw = c["raw_completion"]
    
    # (i)
    try:
        PlannedStep.model_validate_json(raw)
        success_pydantic += 1
    except Exception as e:
        msg = type(e).__name__ + ": " + str(e).split("\n")[0]
        exc_pydantic[msg] += 1
        
    # (ii)
    try:
        obj = json.loads(raw)
        at = obj.get("action_type")
        ActionType(at)
        success_enum_raw += 1
    except Exception as e:
        msg = type(e).__name__ + ": " + str(e).split("\n")[0]
        exc_enum_raw[msg] += 1

print("\n=== P1.d Hypothesis Test ===")
print(f"(i) PlannedStep.model_validate_json(raw): {success_pydantic}/{len(all_completions)} ({success_pydantic/len(all_completions)*100:.2f}%)")
print("Top 5 exceptions for (i):")
for msg, cnt in exc_pydantic.most_common(5):
    print(f"  [{cnt:4d}] {msg}")

print(f"\n(ii) json.loads then ActionType(obj['action_type']) [NO case conversion]: {success_enum_raw}/{len(all_completions)} ({success_enum_raw/len(all_completions)*100:.2f}%)")
print("Top 5 exceptions for (ii):")
for msg, cnt in exc_enum_raw.most_common(5):
    print(f"  [{cnt:4d}] {msg}")

# P2: Learning-signal estimate under NEW parser + current stub
# For each logged step's group of G completions in f02eeb, classify each completion using
# parse_planned_step + system3_stub mapping into:
# PARSE_FAIL, UNRECOGNIZED, UNMAPPED (which action_type), MAPPABLE (isolate/install/check)
print("\n=== P2 Group Learning-Signal Analysis ===")

flat_groups_count = 0
has_mappable_count = 0
mappable_hist = collections.Counter()

# Also let's track the rewards for flat groups
flat_groups_rewards = collections.Counter()

for entry in entries:
    group_classes = []
    group_rewards = []
    
    for c in entry["completions"]:
        raw = c["raw_completion"]
        try:
            step = parse_planned_step(raw)
            # check mapping
            atomic_steps = expand_planned_step(step)
            is_unmapped = any(s.command == "__UNMAPPED_ACTION__" for s in atomic_steps)
            if is_unmapped:
                cls_name = f"UNMAPPED({step.action_type.value})"
                # Under current reward.py / env.py, what reward would it get?
                # UNMAPPED action gets execution penalty
                r = -1.0 # execution penalty for UNMAPPED_ACTION
            else:
                cls_name = f"MAPPABLE({step.action_type.value})"
                r = None # depends on execution result
        except ValueError as ve:
            if "Unrecognized action_type" in str(ve):
                cls_name = "UNRECOGNIZED"
                r = -1.0
            elif "No JSON object found" in str(ve):
                cls_name = "PARSE_FAIL"
                r = -1.0
            else:
                cls_name = "PARSE_FAIL"
                r = -1.0
        except json.JSONDecodeError:
            cls_name = "PARSE_FAIL"
            r = -1.0
        except Exception:
            cls_name = "PARSE_FAIL"
            r = -1.0
            
        group_classes.append(cls_name)
        group_rewards.append(r)
        
    num_mappable = sum(1 for cls_name in group_classes if cls_name.startswith("MAPPABLE"))
    mappable_hist[num_mappable] += 1
    
    if num_mappable > 0:
        has_mappable_count += 1
    else:
        # Check if all completions would get identical deterministic rewards
        # All are non-mappable: PARSE_FAIL, UNRECOGNIZED, UNMAPPED
        # In train_system2_grpo.py:
        # PARSE_FAIL -> -1.0
        # UNRECOGNIZED -> -1.0
        # UNMAPPED -> gets execution_error="__UNMAPPED_ACTION__", status=ExecutionStatus.FAILED, returncode=1,
        # validator reward is computed or penalty? Let's check reward logic!
        # If all r == -1.0:
        if all(r == -1.0 for r in group_rewards):
            flat_groups_count += 1

total_groups = len(entries)
print(f"Total groups: {total_groups}")
print(f"Groups with >= 1 MAPPABLE completion: {has_mappable_count}/{total_groups} ({has_mappable_count/total_groups*100:.2f}%)")
print(f"Groups with ALL non-MAPPABLE and identical reward (-1.0): {flat_groups_count}/{total_groups} ({flat_groups_count/total_groups*100:.2f}%)")
print("Histogram of number of MAPPABLE completions per group (0..G):")
for k in range(5):
    cnt = mappable_hist[k]
    print(f"  {k} MAPPABLE completions: {cnt:4d} groups ({cnt/total_groups*100:.2f}%)")
