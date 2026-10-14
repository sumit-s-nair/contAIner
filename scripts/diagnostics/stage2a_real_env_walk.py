#!/usr/bin/env python3
"""
scripts/diagnostics/stage2a_real_env_walk.py
===========================================
Stage 2A: Diagnostic walk of one repo through the real environment.
Uses real PlannerEnv and DockerEpisodeExecutor with the training script configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_env.env import PlannerEnv, EnvConfig
from src.rl_env.dry_run_executor import TrainingReviewPolicy
from src.rl_env.repo_loader import RepoLoader
from src.repo_scan.scanner import scan_repo
from src.system2_planner.models import PlannedStep, ActionType
from src.rl_env.system3_stub import expand_planned_step


def pick_smallest_repo(corpus_path: str) -> Tuple[Dict[str, Any], str, str]:
    """
    Pick ONE repo from the corpus v3 train split:
    the smallest Python repo that has a requirements.txt or pyproject with at least 1 dependency.
    Returns (repo_entry, reason_str, first_dependency_name).
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    train_repos = [r for r in corpus["splits"]["train"] if r.get("size_kb", -1) > 0]
    train_repos.sort(key=lambda x: x["size_kb"])

    cache_dir = Path("cache/repo_clones")
    loader = RepoLoader(reset_between_episodes=False)

    for entry in train_repos:
        repo_name = entry["repo"]
        safe_name = repo_name.replace("/", "_")
        cached_dirs = list(cache_dir.glob(f"{safe_name}_*"))
        if cached_dirs:
            local_path = cached_dirs[0]
            sha_part = local_path.name[len(safe_name) + 1:]
            if not entry.get("sha") and sha_part:
                entry["sha"] = sha_part
            manifest = scan_repo(str(local_path))
        else:
            try:
                manifest, local_path = loader.load(entry)
            except Exception as e:
                continue

        eco = manifest.ecosystems.get("python")
        if not eco:
            continue

        manifest_files = [Path(mf).name.lower() for mf in eco.manifest_files]
        has_req_or_pyproject = any(
            "requirements" in mf or "pyproject" in mf for mf in manifest_files
        )
        if has_req_or_pyproject and len(eco.dependencies) >= 1:
            first_dep = next(iter(eco.dependencies.keys()))
            reason = (
                f"Smallest Python repository in corpus v3 train split (size_kb={entry.get('size_kb')}) "
                f"with a requirements.txt/pyproject.toml declaring at least 1 dependency "
                f"({len(eco.dependencies)} declared dependencies found; first='{first_dep}')."
            )
            return entry, reason, first_dep

    raise RuntimeError("No candidate Python repo found with requirements.txt or pyproject in train split.")


def docker_inspect_fields(container_id: Optional[str]) -> Dict[str, Any]:
    """Query docker inspect for HostConfig.NetworkMode, Config.WorkingDir, and Mounts (with RW flags)."""
    if not container_id:
        return {"error": "No container ID available for inspect"}
    try:
        res = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if res.returncode != 0:
            return {
                "error": f"docker inspect returned {res.returncode}",
                "stderr": res.stderr.strip(),
            }
        data = json.loads(res.stdout)[0]
        mounts_summary = []
        for m in data.get("Mounts", []):
            mounts_summary.append({
                "Type": m.get("Type"),
                "Source": m.get("Source"),
                "Destination": m.get("Destination"),
                "Mode": m.get("Mode"),
                "RW": m.get("RW"),
            })
        return {
            "HostConfig": {
                "NetworkMode": data.get("HostConfig", {}).get("NetworkMode")
            },
            "Config": {
                "WorkingDir": data.get("Config", {}).get("WorkingDir")
            },
            "Mounts": mounts_summary,
        }
    except Exception as exc:
        return {
            "error": f"Exception during docker inspect: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def step_and_record(
    env_instance: PlannerEnv,
    action: PlannedStep,
    step_label: str,
) -> Dict[str, Any]:
    """Execute one step on the given env, record required metrics and container inspect."""
    atomic_steps = expand_planned_step(action)
    expanded_cmds = [s.command for s in atomic_steps]

    record: Dict[str, Any] = {
        "step_label": step_label,
        "action": {
            "action_type": action.action_type.value,
            "target": getattr(action, "target", ""),
            "description": action.description,
        },
        "expanded_command": expanded_cmds,
        "tier": None,
        "status": None,
        "verified": None,
        "reward": None,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "docker_inspect": None,
    }

    try:
        obs, reward, done, truncated, info = env_instance.step(action)
        record["tier"] = info.get("exec_tier")
        record["status"] = info.get("exec_status")
        record["verified"] = info.get("verified")
        record["reward"] = reward
        record["returncode"] = info.get("returncode")
        record["stdout_tail"] = info.get("stdout_tail")
        record["stderr_tail"] = info.get("stderr_tail")
        record["done"] = done
        record["truncated"] = truncated
    except Exception as exc:
        record["status"] = "ENV_ERROR"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()

    # Container inspect
    executor = getattr(env_instance, "_docker_executor", None) or getattr(env_instance, "_executor", None)
    cid = getattr(executor, "_container_id", None)
    record["docker_inspect"] = docker_inspect_fields(cid)
    return record


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs/stage2a")
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = logs_dir / f"{timestamp}.json"

    diagnostic_log: Dict[str, Any] = {
        "timestamp": timestamp,
        "json_path": str(out_json_path),
        "steps": {},
        "summary": [],
    }

    corpus_path = "datasets/repo_corpus/corpus_manifest_v3.json"
    repo_entry, reason, first_dep = pick_smallest_repo(corpus_path)
    print(f"Chosen repo: {repo_entry['repo']}")
    print(f"Why: {reason}")
    diagnostic_log["chosen_repo"] = repo_entry
    diagnostic_log["chosen_reason"] = reason
    diagnostic_log["first_dep"] = first_dep

    # Matching train_system2_grpo.py setup_environment()
    config = EnvConfig(
        dry_run=False,
        review_policy=TrainingReviewPolicy.AUTO_DENY,
        corpus_manifest_path=corpus_path,
    )
    env = PlannerEnv(config)
    forked_env: Optional[PlannerEnv] = None
    snapshot_tag: Optional[str] = None

    # Step a: env.reset(repo)
    try:
        print("[Step a] env.reset(repo)...")
        obs, info = env.reset(repo_entry)
        cid = getattr(getattr(env, "_docker_executor", None) or getattr(env, "_executor", None), "_container_id", None)
        diagnostic_log["steps"]["step_a_reset"] = {
            "status": "SUCCESS",
            "repo": repo_entry["repo"],
            "container_id": cid,
            "docker_inspect": docker_inspect_fields(cid),
        }
    except Exception as exc:
        diagnostic_log["steps"]["step_a_reset"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step b: docker commit the main container, exactly as training script does
    try:
        print("[Step b] docker commit main container...")
        if env._docker_executor is None:
            raise RuntimeError("Main env._docker_executor is None.")
        snapshot_tag = env._docker_executor.snapshot()
        v_res = subprocess.run(
            ["docker", "image", "inspect", snapshot_tag],
            capture_output=True,
            text=True,
            timeout=10,
        )
        diagnostic_log["steps"]["step_b_commit"] = {
            "status": "SUCCESS" if v_res.returncode == 0 else "VERIFY_FAILED",
            "snapshot_tag": snapshot_tag,
            "inspect_returncode": v_res.returncode,
        }
    except Exception as exc:
        diagnostic_log["steps"]["step_b_commit"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step c: env.fork(snapshot_image=tag)
    try:
        print(f"[Step c] env.fork(snapshot_image={snapshot_tag})...")
        forked_env = env.fork(snapshot_image=snapshot_tag)
        fork_cid = getattr(getattr(forked_env, "_docker_executor", None) or getattr(forked_env, "_executor", None), "_container_id", None)
        diagnostic_log["steps"]["step_c_fork"] = {
            "status": "SUCCESS",
            "fork_container_id": fork_cid,
            "docker_inspect": docker_inspect_fields(fork_cid),
        }
    except Exception as exc:
        diagnostic_log["steps"]["step_c_fork"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step d: In FORK, step ISOLATE (target .venv)
    step_isolate = PlannedStep(
        action_type=ActionType.ISOLATE,
        target=".venv",
        description="Create virtual environment in .venv",
    )
    if forked_env is not None:
        try:
            print("[Step d] Fork: step ISOLATE...")
            rec_d = step_and_record(forked_env, step_isolate, "fork_isolate")
            diagnostic_log["steps"]["step_d_fork_isolate"] = rec_d
        except Exception as exc:
            diagnostic_log["steps"]["step_d_fork_isolate"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    else:
        diagnostic_log["steps"]["step_d_fork_isolate"] = {
            "status": "SKIPPED",
            "reason": "forked_env is None",
        }

    # Step e: In FORK, step INSTALL for the first manifest dependency
    step_install = PlannedStep(
        action_type=ActionType.INSTALL,
        target=first_dep,
        description=f"Install first manifest dependency: {first_dep}",
    )
    if forked_env is not None:
        try:
            print(f"[Step e] Fork: step INSTALL ({first_dep})...")
            rec_e = step_and_record(forked_env, step_install, "fork_install")
            diagnostic_log["steps"]["step_e_fork_install"] = rec_e
        except Exception as exc:
            diagnostic_log["steps"]["step_e_fork_install"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    else:
        diagnostic_log["steps"]["step_e_fork_install"] = {
            "status": "SKIPPED",
            "reason": "forked_env is None",
        }

    # Step f: In FORK, step CHECK (target tests or .)
    step_check = PlannedStep(
        action_type=ActionType.CHECK,
        target=".",
        description="Run test check on .",
    )
    if forked_env is not None:
        try:
            print("[Step f] Fork: step CHECK (.)...")
            rec_f = step_and_record(forked_env, step_check, "fork_check")
            diagnostic_log["steps"]["step_f_fork_check"] = rec_f
        except Exception as exc:
            diagnostic_log["steps"]["step_f_fork_check"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    else:
        diagnostic_log["steps"]["step_f_fork_check"] = {
            "status": "SKIPPED",
            "reason": "forked_env is None",
        }

    # Step g: Repeat steps d–f in the MAIN env
    # Main ISOLATE
    try:
        print("[Step g.1] Main: step ISOLATE...")
        rec_g_iso = step_and_record(env, step_isolate, "main_isolate")
        diagnostic_log["steps"]["step_g1_main_isolate"] = rec_g_iso
    except Exception as exc:
        diagnostic_log["steps"]["step_g1_main_isolate"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Main INSTALL
    try:
        print(f"[Step g.2] Main: step INSTALL ({first_dep})...")
        rec_g_inst = step_and_record(env, step_install, "main_install")
        diagnostic_log["steps"]["step_g2_main_install"] = rec_g_inst
    except Exception as exc:
        diagnostic_log["steps"]["step_g2_main_install"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Main CHECK
    try:
        print("[Step g.3] Main: step CHECK (.)...")
        rec_g_chk = step_and_record(env, step_check, "main_check")
        diagnostic_log["steps"]["step_g3_main_check"] = rec_g_chk
    except Exception as exc:
        diagnostic_log["steps"]["step_g3_main_check"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    # Step h: end_episode() on both; docker rmi the snapshot tag
    cleanup_info: Dict[str, Any] = {}
    if forked_env is not None:
        try:
            print("[Step h] forked_env.end_episode()...")
            forked_env.end_episode()
            cleanup_info["fork_end_episode"] = "SUCCESS"
        except Exception as exc:
            cleanup_info["fork_end_episode"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    try:
        print("[Step h] env.end_episode()...")
        env.end_episode()
        cleanup_info["main_end_episode"] = "SUCCESS"
    except Exception as exc:
        cleanup_info["main_end_episode"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    if snapshot_tag:
        try:
            print(f"[Step h] docker rmi {snapshot_tag}...")
            r_rmi = subprocess.run(
                ["docker", "rmi", snapshot_tag],
                capture_output=True,
                text=True,
                timeout=30,
            )
            cleanup_info["docker_rmi"] = {
                "tag": snapshot_tag,
                "returncode": r_rmi.returncode,
                "stdout": r_rmi.stdout.strip(),
                "stderr": r_rmi.stderr.strip(),
            }
        except Exception as exc:
            cleanup_info["docker_rmi"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    diagnostic_log["steps"]["step_h_cleanup"] = cleanup_info

    # Save complete JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_log, f, indent=2)
    print(f"Diagnostics written to: {out_json_path}")

    # Build summary lines
    step_items = [
        ("Fork ISOLATE", diagnostic_log["steps"].get("step_d_fork_isolate")),
        ("Fork INSTALL", diagnostic_log["steps"].get("step_e_fork_install")),
        ("Fork CHECK", diagnostic_log["steps"].get("step_f_fork_check")),
        ("Main ISOLATE", diagnostic_log["steps"].get("step_g1_main_isolate")),
        ("Main INSTALL", diagnostic_log["steps"].get("step_g2_main_install")),
        ("Main CHECK", diagnostic_log["steps"].get("step_g3_main_check")),
    ]

    print("\n=== STEP RESULTS ===")
    for name, s_data in step_items:
        if not s_data:
            print(f"{name}: status=UNKNOWN reward=UNKNOWN returncode=UNKNOWN")
        else:
            status = s_data.get("status")
            reward = s_data.get("reward")
            rc = s_data.get("returncode")
            print(f"{name}: status={status} reward={reward} returncode={rc}")


if __name__ == "__main__":
    main()
