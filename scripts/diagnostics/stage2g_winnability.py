#!/usr/bin/env python3
"""
scripts/diagnostics/stage2g_winnability.py
=========================================
Stage 2G winnability check (REQUIRED before training).

Runs a scripted oracle planner through the REAL PlannerEnv +
DockerEpisodeExecutor (proxy network, same config as training) on N repos of
the pilot corpus train split:
    [ISOLATE .venv  if the repo has conflicts]
    INSTALL <manifest>  for each Python manifest (requirements*.txt first,
                        then pyproject.toml / setup.py / setup.cfg)
    INSTALL <pkg>       tail: each required package still missing after the
                        manifests (ground truth from pip list), one at a time,
                        while steps remain -- the agent can do the same
and, on the first repo, a CHECK-spam episode for contrast.

Pass criteria (printed): oracle episodes reach episode_complete with a
positive total reward; the spam episode total is negative and below the
oracle's.  Everything is logged to logs/stage2g/winnability_<ts>.json.

--all runs the oracle on every train+val repo of the pilot and writes the
per-repo verdict to datasets/repo_corpus/pilot_winnability.json (updated after
each repo; --resume skips repos already recorded at the same sha).
scripts/build_pilot_corpus.py then excludes non-won repos as
'oracle_unwinnable' (e.g. new92/instatools: its requirements.txt pins are
mutually unsatisfiable, ResolutionImpossible).

Usage:  python scripts/diagnostics/stage2g_winnability.py [--n 6] [--seed 0]
        python scripts/diagnostics/stage2g_winnability.py --all [--resume | --only-lost]
Requires: Docker running, `python scripts/setup_egress_proxy.py` and
`python scripts/build_sandbox_image.py` done.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_env.docker_executor import PYTHON_SANDBOX_IMAGE, DockerExecutorConfig, NetworkConfig  # noqa: E402
from src.rl_env.dry_run_executor import TrainingReviewPolicy  # noqa: E402
from src.rl_env.env import EnvConfig, PlannerEnv  # noqa: E402
from src.rl_env.reward import (WINNABILITY_CRITERION_VERSION, pinned_target,  # noqa: E402
                               required_specifiers, satisfied_required)
from src.system2_planner.models import ActionType, PlannedStep  # noqa: E402

PILOT = "datasets/repo_corpus/corpus_manifest_v3_pilot.json"
WINNABILITY = "datasets/repo_corpus/pilot_winnability.json"
# Stage 2H: same oracle on the (never-filtered) test split, so the paper can
# report completion on the oracle-winnable test subset.  Separate file: the
# test verdicts must never feed build_pilot_corpus.py.
TEST_WINNABILITY = "datasets/repo_corpus/test_winnability.json"
PROJECT_FILES = {"pyproject.toml", "setup.py", "setup.cfg"}


def oracle_plan(scan: dict) -> list:
    plan = []
    if scan["conflicts"]:
        plan.append(PlannedStep(action_type=ActionType.ISOLATE, target=".venv", description="Create virtual environment"))
    files = scan["manifest_files"]
    reqs = sorted(f for f in files if f.lower().endswith(".txt"))
    projs = sorted(f for f in files if Path(f).name in PROJECT_FILES)
    for f in reqs + projs:
        plan.append(PlannedStep(action_type=ActionType.INSTALL, target=f, description=f"Install from {f}"))
    return plan


def scan_via_env(env: PlannerEnv, entry: dict) -> dict:
    """oracle_plan() input for a repo with no pilot_scan entry (the test split):
    the env scans the repo on reset, exactly as it does for the policy."""
    try:
        obs, _ = env.reset(entry)
        return {"conflicts": list(obs.repo_manifest.conflicts or []),
                "manifest_files": list((obs.manifest_paths or {}).get("python", [])),
                "required": []}
    finally:
        env.end_episode()


def _tail_steps(env: PlannerEnv):
    """INSTALL <pkg><declared constraint> for each required package not yet
    satisfied (missing, or installed at a version outside every declared
    constraint)."""
    installed = env._installed_now
    if installed is None:
        return []
    specs = required_specifiers(env._instance)
    missing = sorted(set(specs) - satisfied_required(env._instance, installed))
    return [PlannedStep(action_type=ActionType.INSTALL, target=pinned_target(p, specs[p]),
                        description=f"Install {p}") for p in missing]


def run_episode(env: PlannerEnv, entry: dict, plan: list, label: str, tail: bool = False) -> dict:
    rec = {"label": label, "repo": entry["repo"], "steps": [], "total": 0.0, "complete": False}
    t0 = time.time()
    try:
        env.reset(entry)
        plan = list(plan)
        i = 0
        tail_added = not tail
        while True:
            if i >= len(plan):
                if tail_added:
                    break
                tail_added = True
                extra = _tail_steps(env)
                if extra:
                    print(f"   -- tail: {[s.target for s in extra]}")
                plan += extra
                continue
            step = plan[i]
            ts = time.time()
            _, r, term, trunc, info = env.step(step)
            bd = info["reward_breakdown"]
            rec["steps"].append({
                "action": f"{step.action_type.value} {step.target}",
                "status": info["exec_status"], "verified": info["verified"], "rc": info["returncode"],
                "reward": round(r, 3), "execution_label": bd.get("execution_label"),
                "install_progress": info.get("install_progress"),
                "required": f"{info.get('required_installed')}/{info.get('required_total')}",
                "new_ordering_violations": bd.get("new_ordering_violations"),
                "missing_at_end": bd.get("missing_at_end"),
                "seconds": round(time.time() - ts, 1),
                "stderr_tail": info.get("stderr_tail", "")[-2000:] if info["exec_status"] != "success" else "",
                "pip_check_ok": info.get("dependency_check_ok"),
                "pip_check": info.get("dependency_check_output", ""),
            })
            if info.get("dependency_check_ok") is False:
                print(f"   -- pip check FAILED: {info['dependency_check_output'].strip().splitlines()[:3]}")
            rec["total"] += r
            rec["complete"] = info["episode_complete"]
            print(f"   {i+1}. {step.action_type.value:8} {step.target:30} {info['exec_status']:8} "
                  f"r={r:+.2f} prog={info.get('install_progress')} req={rec['steps'][-1]['required']} "
                  f"({rec['steps'][-1]['seconds']}s)", flush=True)
            if term or trunc:
                rec["ended"] = "terminated" if term else "truncated"
                break
            i += 1
        if "ended" not in rec:
            rec["ended"] = "plan_exhausted"
        if not rec["complete"] and env._installed_now is not None:
            rec["missing_required"] = sorted(set(required_specifiers(env._instance))
                                             - satisfied_required(env._instance, env._installed_now))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        print(f"   ERROR {rec['error']}")
    finally:
        try:
            env.end_episode()
        except Exception as exc:
            rec["end_episode_error"] = repr(exc)
    rec["total"] = round(rec["total"], 3)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def _verdict(entry: dict, rec: dict) -> dict:
    won = bool(rec["complete"] and rec["total"] > 0)
    v = {"sha": entry.get("sha"), "won": won, "total": rec["total"], "ended": rec.get("ended"),
         "seconds": rec["seconds"], "image": rec.get("image"), "criterion": CRITERION_VERSION}
    if not won:
        if "error" in rec:
            v["why"] = rec["error"]
        else:
            failed = [s for s in rec["steps"] if s["status"] != "success"]
            last = (failed[-1]["stderr_tail"].strip().splitlines() or [""]) if failed else [""]
            errs = [ln for ln in last if "ERROR" in ln or "rror:" in ln]
            v["why"] = {
                "failed_steps": [s["action"] for s in failed],
                "missing_required": rec.get("missing_required"),
                "pip_check": next((st["pip_check"][:300] for st in reversed(rec["steps"])
                                   if st.get("pip_check_ok") is False), None),
                "stderr": (errs[0] if errs else last[-1])[:300],
            }
    return v


CRITERION_VERSION = WINNABILITY_CRITERION_VERSION  # see src/rl_env/reward.py


def _image_label() -> str | None:
    """Sandbox image provenance for the verdict file, e.g. python-3.11-git-toolchain-v2."""
    try:
        r = subprocess.run(["docker", "image", "inspect", PYTHON_SANDBOX_IMAGE, "--format",
                            '{{ index .Config.Labels "com.contai.sandbox" }}'],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _select(entries: list, verdicts: dict, *, resume: bool = False, only_lost: bool = False) -> list:
    """Repos to (re)run with --all.

    A verdict is current if it has the same sha AND the same CRITERION_VERSION.
    resume:    skip repos with a current verdict (won or lost).
    only_lost: run only repos whose current verdict is NOT won; repos without a
               current verdict are run too (nothing to keep).
    """
    out = []
    for e in entries:
        v = verdicts.get(e["repo"])
        current = bool(v) and v.get("sha") == e.get("sha") and v.get("criterion") == CRITERION_VERSION
        if only_lost:
            if not current or not v.get("won"):
                out.append(e)
        elif resume:
            if not current:
                out.append(e)
        else:
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="oracle on every train+val repo; write the verdict file")
    ap.add_argument("--resume", action="store_true", help="with --all: skip repos already in the verdict file at the same sha")
    ap.add_argument("--only-lost", action="store_true",
                    help="with --all: rerun only repos whose verdict at the same sha is NOT won "
                         "(use after a sandbox-image fix; won verdicts are kept)")
    ap.add_argument("--split", choices=["pilot", "test"], default="pilot",
                    help="with --all: 'test' runs the oracle on the test split and writes "
                         f"{TEST_WINNABILITY} (evaluation subset only; never used for training data)")
    args = ap.parse_args()
    if args.split == "test" and not args.all:
        ap.error("--split test requires --all")
    pilot = json.loads((PROJECT_ROOT / PILOT).read_text(encoding="utf-8"))
    train = list(pilot["splits"]["train"])
    random.Random(args.seed).shuffle(train)
    # make sure a conflict repo is included if there is one
    conf = [e for e in train if pilot["pilot_scan"][e["repo"]]["conflicts"]]
    chosen = (conf[:1] + [e for e in train if e not in conf[:1]])[: args.n]

    verdict_path = PROJECT_ROOT / (TEST_WINNABILITY if args.split == "test" else WINNABILITY)
    verdicts: dict = {}
    if args.all and args.split == "test":
        chosen = list(pilot["splits"]["test"])
        if verdict_path.exists():
            verdicts = json.loads(verdict_path.read_text(encoding="utf-8")).get("repos", {})
        chosen = _select(chosen, verdicts, resume=args.resume, only_lost=args.only_lost)
        print(f"test split: {len(chosen)} repo(s) to run")
    elif args.all:
        # The oracle must judge every repo that passed the STATIC filters, not
        # just the current splits: the splits already exclude repos an older
        # verdict marked unwinnable, so re-judging only the splits never gives
        # those repos a second chance after an image/criterion fix.
        cand = pilot.get("oracle_candidates")
        if cand is None:
            print("WARNING: pilot has no oracle_candidates (old build_pilot_corpus); "
                  "judging current splits only -- rerun build_pilot_corpus.py first")
            cand = {"train": pilot["splits"]["train"], "val": pilot["splits"]["val"]}
        chosen = list(cand["train"]) + list(cand["val"])
        if verdict_path.exists():
            verdicts = json.loads(verdict_path.read_text(encoding="utf-8")).get("repos", {})
        chosen = _select(chosen, verdicts, resume=args.resume, only_lost=args.only_lost)
        if args.resume or args.only_lost:
            print(f"{'--only-lost' if args.only_lost else '--resume'}: {len(chosen)} repo(s) to run")

    env = PlannerEnv(EnvConfig(
        dry_run=False, review_policy=TrainingReviewPolicy.AUTO_DENY,
        corpus_manifest_path=PILOT,
        docker_config=DockerExecutorConfig(network=NetworkConfig(mode="proxy")),
    ))
    image = _image_label()
    print(f"sandbox image: {PYTHON_SANDBOX_IMAGE} ({image})")
    results = []
    for k, entry in enumerate(chosen):
        if args.split == "test":
            try:
                scan = scan_via_env(env, entry)
            except Exception as exc:  # run_episode's own reset records the error
                print(f"   scan failed ({type(exc).__name__}: {exc}); oracle reset will retry")
                scan = {"conflicts": [], "manifest_files": [], "required": []}
        else:
            scan = pilot["pilot_scan"][entry["repo"]]
        plan = oracle_plan(scan)
        print(f"\n[{k+1}/{len(chosen)}] {entry['repo']}  conflicts={scan['conflicts']}  "
              f"required={len(scan['required'])}  manifests={scan['manifest_files']}")
        rec = run_episode(env, entry, plan, "oracle", tail=True)
        rec["image"] = image
        results.append(rec)
        if args.all:
            verdicts[entry["repo"]] = _verdict(entry, rec)
            verdict_path.write_text(json.dumps({
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "criterion": "oracle (manifests + per-package pinned tail) reaches episode_complete with total > 0; "
                             "required = installed at a version satisfying a declared constraint, and pip check passes",
                "repos": verdicts,
            }, indent=2), encoding="utf-8")
        if k == 0 and not args.all:
            spam = [PlannedStep(action_type=ActionType.CHECK, target=".", description="Run tests")] * env._cfg.max_steps
            print("   -- CHECK-spam contrast episode --")
            results.append(run_episode(env, entry, spam, "check_spam"))

    oracle = [r for r in results if r["label"] == "oracle"]
    won = [r for r in oracle if r["complete"] and r["total"] > 0]
    spam = [r for r in results if r["label"] == "check_spam"]
    summary = {
        "oracle_episodes": len(oracle), "oracle_won": len(won),
        "oracle_totals": {r["repo"]: r["total"] for r in oracle},
        "spam_total": spam[0]["total"] if spam else None,
        "spam_below_oracle": bool(spam) and bool(oracle) and spam[0]["total"] < 0 and spam[0]["total"] < oracle[0]["total"],
    }
    out_dir = PROJECT_ROOT / "logs/stage2g"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"winnability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps(summary, indent=2))
    print(f"log: {out.relative_to(PROJECT_ROOT)}")
    if args.all:
        lost = sorted(r for r, v in verdicts.items() if not v["won"])
        print(f"verdicts: {verdict_path.relative_to(PROJECT_ROOT)}  won {len(verdicts) - len(lost)}/{len(verdicts)}")
        for r in lost:
            print(f"  NOT WON {r}: {verdicts[r].get('why')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
