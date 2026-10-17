"""Build the System 3 slot dataset (Stage 3A): execution-verified PlannedStep -> AtomicSteps.

Three phases, run on the 4090 (Docker sandbox; GPU only for `recover`):

  collect   For every repo in the chosen corpus splits, execute canonical
            planner steps with the reference expander (the stub) in the
            sandbox, from snapshots: INSTALL each Python manifest (fresh and
            after ISOLATE), INSTALL single / multi dependency specs, ISOLATE,
            CHECK tests.  Each case records the System 3 prompt inputs
            (step, context, docs) and the sandbox ground truth.  Failed
            manifest installs become recovery seeds.
              python -m scripts.build_system3_dataset --phase collect --splits train val test --workers 3

  recover   For each failed INSTALL seed (train/val by default): reproduce
            the failure, snapshot, sample k candidate expansions from a
            local teacher (Qwen2.5-Coder-7B-Instruct) given the failure +
            docs, execute each candidate (plus the stub's repeat as the
            baseline) on a fork of the snapshot, and keep the candidate that
            installs the most required packages (version-aware ground truth,
            pip check for completion).  Rejection-sampling fine-tuning data.
              python -m scripts.build_system3_dataset --phase recover --k 8 --workers 4

  export    SFT files: train.jsonl / val.jsonl (chat messages + meta) and
            test_cases.jsonl (held-out cases for evaluate_system3), stats.json.
              python -m scripts.build_system3_dataset --phase export

Outputs go to datasets/system3-slot/.  collect/recover append and skip
case_ids already written (safe to restart).  Test-split repos are never in
train/val.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import logging
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from packaging.requirements import InvalidRequirement, Requirement

from src.rl_env.reward import _applies_in_sandbox, satisfied_required
from src.sandbox.models import AtomicStep
from src.system2_planner.models import ActionType, PlannedStep
from src.system3_command_generation.slot import (
    DocProvider,
    Sys3ParseError,
    build_messages,
    context_from_env,
    parse_atomic_steps,
    render_docs,
    steps_policy_violation,
    steps_to_target_json,
)
from src.rl_env.system3_stub import expand_planned_step

log = logging.getLogger("build_system3")

OUT_DIR = Path("datasets/system3-slot")
CORPUS = "datasets/repo_corpus/corpus_manifest_v3.json"
TEACHER = "Qwen/Qwen2.5-Coder-7B-Instruct"

_write_lock = threading.Lock()

# Paraphrases in the style System 2 writes (seen in rollouts); picked by hash
# so a case's description is stable across runs.
_DESCRIPTIONS = {
    "install_manifest": [
        "Install the project's dependencies from {t}.",
        "Install the Python dependencies listed in {t}.",
        "Install dependencies using {t}.",
        "Install all required packages declared in {t}.",
    ],
    "install_specs": [
        "Install {t}.",
        "Install the required package(s): {t}.",
        "Install the dependencies {t}.",
    ],
    "isolate": [
        "Create an isolated virtual environment.",
        "Create a virtual environment in {t}.",
        "Isolate the project's dependencies in a virtual environment.",
    ],
    "check": [
        "Run the test suite in {t}.",
        "Check the setup by running the tests in {t}.",
        "Verify the environment by running tests.",
    ],
}


def _pick(kind: str, key: str, target: str) -> str:
    opts = _DESCRIPTIONS[kind]
    h = int(hashlib.sha1(f"{kind}|{key}".encode()).hexdigest(), 16)
    return opts[h % len(opts)].format(t=target)


def _case_id(repo: str, label: str) -> str:
    return f"{repo}#{label}"


# ------------------------------------------------------------------ io

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("skipping corrupt line in %s", p)
    return rows


def _append(p: Path, row: Dict[str, Any]) -> None:
    with _write_lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _steps_json(steps: List[AtomicStep]) -> List[Dict[str, Any]]:
    return json.loads(steps_to_target_json(steps))["steps"]


# ------------------------------------------------------------------ env

def make_env(corpus: str, network: str, dry_run: bool = False):
    from src.rl_env.docker_executor import DockerExecutorConfig, NetworkConfig
    from src.rl_env.dry_run_executor import TrainingReviewPolicy
    from src.rl_env.env import EnvConfig, PlannerEnv

    if "COMPROMISED" in corpus:
        raise ValueError("refusing a COMPROMISED corpus manifest")
    return PlannerEnv(EnvConfig(
        dry_run=dry_run,
        review_policy=TrainingReviewPolicy.AUTO_DENY,
        corpus_manifest_path=corpus,
        docker_config=DockerExecutorConfig(network=NetworkConfig(mode=network)),
    ))


def _snapshot(env) -> Optional[str]:
    """Docker snapshot of the env's container, or None in dry-run (fork replays)."""
    return env._docker_executor.snapshot() if env._docker_executor is not None else None


def _delete_snapshot(env, tag: Optional[str]) -> None:
    if tag and env._docker_executor is not None:
        try:
            env._docker_executor.delete_snapshot(tag)
        except Exception:
            pass


class FakeTeacher:
    """--teacher fake: deterministic candidates for smoke tests (no GPU)."""

    def generate(self, messages, n=1, temperature=0.0, max_new_tokens=400):
        good = '{"steps": [{"command": "python -m pip install --prefer-binary -r requirements.txt", "description": "retry with wheels", "verify_command": null, "timeout": 600}]}'
        bad = '{"steps": [{"command": "sudo pip install -r requirements.txt"}]}'
        return ([good, bad, "not json", good] * n)[:n]


def _required_now(env) -> int:
    return len(satisfied_required(env._instance, env._installed_now)) if env._installed_now is not None else -1


def _result(info: Dict[str, Any], before: int, seconds: float) -> Dict[str, Any]:
    after = info.get("required_installed", before)
    return {
        "exec_status": info.get("exec_status"),
        "verified": info.get("verified"),
        "returncode": info.get("returncode"),
        "install_progress": info.get("install_progress"),
        "required_before": before,
        "required_installed": after,
        "required_total": info.get("required_total"),
        "gain": (after - before) if (after is not None and before is not None and before >= 0) else None,
        "episode_complete": info.get("episode_complete"),
        "dependency_check_ok": info.get("dependency_check_ok"),
        "blocked": info.get("blocked"),
        "stderr_tail": (info.get("stderr_tail") or "")[-1500:],
        "seconds": round(seconds, 1),
    }


def _load_repos(corpus: str, splits: Iterable[str], limit: int) -> List[Tuple[str, Dict[str, Any]]]:
    data = json.loads(Path(corpus).read_text(encoding="utf-8"))
    out = []
    for split in splits:
        entries = data["splits"][split]
        if limit:
            entries = entries[:limit]
        out += [(split, e) for e in entries]
    return out


# ------------------------------------------------------------------ canonical cases

def _spec(dep) -> Optional[str]:
    """A shell-safe single-token spec (name + specifier, no markers/spaces)."""
    if not _applies_in_sandbox(dep):
        return None
    for c in dep.constraints or [dep.declared_constraint]:
        c = (c or "").split(";")[0].strip()
        if c in ("", "*"):
            continue
        if c[0] in "<>=!~":
            try:
                r = Requirement(f"{dep.name}{c}")
                return f"{r.name}{r.specifier}".replace(" ", "")
            except InvalidRequirement:
                continue
    try:
        return Requirement(dep.name).name
    except InvalidRequirement:
        return None


def canonical_cases(env, local_path: Optional[str]) -> List[Tuple[str, List[PlannedStep], PlannedStep, str]]:
    """(label, prefix_steps, planned, kind) for one repo, deterministic.

    Python repos only: the stub and System 3 target pip; other ecosystems get
    a non-Python image (no `python` binary), so every case would fail for a
    reason no command can fix.
    """
    repo = env._repo_entry["repo"]
    if "python" not in env._instance.manifest.ecosystems:
        return []
    paths = (env._obs.manifest_paths or {}).get("python", [])[:3]
    eco = env._instance.manifest.ecosystems.get("python")
    cases: List[Tuple[str, List[PlannedStep], PlannedStep, str]] = []
    isolate = PlannedStep(ActionType.ISOLATE, _pick("isolate", repo, ".venv"), ".venv")

    for p in paths:
        cases.append((f"install:{p}", [], PlannedStep(ActionType.INSTALL, _pick("install_manifest", repo + p, p), p),
                      "install_manifest"))
    if paths:
        p = paths[0]
        cases.append((f"isolate+install:{p}", [isolate],
                      PlannedStep(ActionType.INSTALL, _pick("install_manifest", repo + "v" + p, p), p),
                      "install_manifest"))

    specs: List[str] = []
    if eco is not None:
        deps = sorted(eco.dependencies.values(), key=lambda d: d.name)
        specs = [s for s in (_spec(d) for d in deps) if s]
        if not specs:
            specs = sorted({d.guessed_package_name or d.import_name for d in eco.inferred_dependencies
                            if (d.guessed_package_name or d.import_name)})
    if specs:
        h = int(hashlib.sha1(repo.encode()).hexdigest(), 16)
        singles = [specs[h % len(specs)]] + ([specs[(h // 7) % len(specs)]] if len(specs) > 3 else [])
        for s in dict.fromkeys(singles):
            cases.append((f"install-spec:{s}", [], PlannedStep(ActionType.INSTALL, _pick("install_specs", repo + s, s), s),
                          "install_spec"))
        if len(specs) >= 2:
            multi = " ".join(specs[:6])
            cases.append((f"install-multi", [], PlannedStep(ActionType.INSTALL, _pick("install_specs", repo + "m", multi), multi),
                          "install_multi"))

    cases.append(("isolate", [], isolate, "isolate"))

    # CHECK runs after the dependencies are installed (as the planner does):
    # from a fresh container every test suite fails at import, which no
    # command can fix.
    install_first: Optional[PlannedStep] = None
    if paths:
        install_first = PlannedStep(ActionType.INSTALL, _pick("install_manifest", repo + paths[0], paths[0]), paths[0])
    elif len(specs) >= 2:
        multi = " ".join(specs[:6])
        install_first = PlannedStep(ActionType.INSTALL, _pick("install_specs", repo + "m", multi), multi)
    if local_path and install_first is not None:
        for d in ("tests", "test"):
            if (Path(local_path) / d).is_dir():
                cases.append((f"install+check:{d}", [install_first],
                              PlannedStep(ActionType.CHECK, _pick("check", repo, d), d), "check"))
                break
    return cases


def collect_repo(split: str, entry: Dict[str, Any], a, docs: DocProvider, done: set, out: Path) -> Dict[str, int]:
    stats = collections.Counter()
    env = make_env(a.corpus, a.network, a.dry_run)
    base = None
    try:
        env.reset(entry)
        cases = canonical_cases(env, env._local_path)
        if not cases:
            stats["skipped_non_python"] += 1
            return stats
        base = _snapshot(env)
        for label, prefix, planned, kind in cases:
            cid = _case_id(entry["repo"], label)
            if cid in done:
                stats["skipped"] += 1
                continue
            f = None
            try:
                f = env.fork(base)
                for p in prefix:
                    f.step(p)
                ctx = context_from_env(f, planned)
                chunks = docs.for_step(planned.action_type, planned.target or "", planned.description)
                before = _required_now(f)
                ref = expand_planned_step(planned)
                t0 = time.time()
                _, _, _, _, info = f.step(planned)
                res = _result(info, before, time.time() - t0)
                ok = res["exec_status"] == "success" and (res["verified"] or all(s.verify_command is None for s in ref))
                row = {
                    "case_id": cid, "repo": entry["repo"], "split": split, "category": entry.get("category"),
                    "kind": kind, "prefix": [{"action_type": p.action_type.value, "target": p.target,
                                              "description": p.description} for p in prefix],
                    "planned": {"action_type": planned.action_type.value, "target": planned.target,
                                "description": planned.description},
                    "context": ctx.to_dict(), "docs": chunks, "reference_steps": _steps_json(ref),
                    "result": res, "accepted": bool(ok),
                    "recovery_seed": planned.action_type == ActionType.INSTALL and res["exec_status"] == "failed",
                }
                _append(out, row)
                stats["accepted" if ok else "rejected"] += 1
                stats["seeds"] += int(row["recovery_seed"])
            except Exception as exc:
                stats["errors"] += 1
                log.error("%s: %s\n%s", cid, exc, traceback.format_exc(limit=3))
            finally:
                if f is not None:
                    try:
                        f.end_episode()
                    except Exception:
                        pass
    except Exception as exc:
        stats["repo_errors"] += 1
        log.error("%s: repo setup failed: %s", entry.get("repo"), exc)
    finally:
        _delete_snapshot(env, base)
        try:
            env.end_episode()
        except Exception:
            pass
    log.info("%s [%s] %s", entry["repo"], split, dict(stats))
    return stats


def phase_collect(a) -> None:
    out = Path(a.out) / "cases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = DocProvider(docs_dir=Path(a.out) / "docs", mcp_url=a.mcp_url)
    done = {r["case_id"] for r in _read_jsonl(out)}
    repos = _load_repos(a.corpus, a.splits, a.limit)
    log.info("collect: %d repos, %d cases already done", len(repos), len(done))
    total = collections.Counter()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(collect_repo, s, e, a, docs, done, out) for s, e in repos]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            total.update(fu.result())
            if i % 10 == 0:
                log.info("collect progress %d/%d %s", i, len(repos), dict(total))
    log.info("collect done: %s", dict(total))


# ------------------------------------------------------------------ recovery

def _rank_key(c: Dict[str, Any]) -> Tuple:
    r = c.get("result") or {}
    return (bool(r.get("episode_complete")), r.get("gain") or 0,
            r.get("exec_status") == "success", -len(c.get("steps") or []),
            -sum(len(s["command"]) for s in (c.get("steps") or [])))


def recover_seed(seed: Dict[str, Any], a, docs: DocProvider, teacher, gen_lock: threading.Lock,
                 out: Path) -> str:
    entry = {"repo": seed["repo"], "category": seed.get("category")}
    corpus = json.loads(Path(a.corpus).read_text(encoding="utf-8"))
    for e in corpus["splits"][seed["split"]]:
        if e["repo"] == seed["repo"]:
            entry = e
            break
    planned = PlannedStep(ActionType(seed["planned"]["action_type"]), seed["planned"]["description"],
                          seed["planned"]["target"])
    prefix = [PlannedStep(ActionType(p["action_type"]), p["description"], p["target"]) for p in seed["prefix"]]

    env = make_env(a.corpus, a.network, a.dry_run)
    snap = None
    try:
        env.reset(entry)
        for p in prefix:
            env.step(p)
        _, _, _, _, first = env.step(planned)           # reproduce the failure with the stub
        if first.get("exec_status") != "failed":
            row = {"case_id": seed["case_id"] + "/recover", "seed": seed["case_id"], "repo": seed["repo"],
                   "split": seed["split"], "status": "not_reproduced", "first": _result(first, -1, 0)}
            _append(out, row)
            return "not_reproduced"
        if env._done:
            _append(out, {"case_id": seed["case_id"] + "/recover", "seed": seed["case_id"], "repo": seed["repo"],
                          "split": seed["split"], "status": "episode_done_after_failure"})
            return "episode_done"
        ctx = context_from_env(env, planned)
        chunks = docs.for_step(planned.action_type, planned.target or "", planned.description,
                               (ctx.previous or {}).get("stderr", ""))
        msgs = build_messages(planned, ctx, render_docs(chunks))
        # a failed install can leave packages behind: measure the real state
        if env._docker_executor is not None:
            now = env._query_installed()
            if now is not None:
                env._installed_now = now
        before = _required_now(env)
        snap = _snapshot(env)

        with gen_lock:
            texts = teacher.generate(msgs, n=a.k, temperature=a.temperature, max_new_tokens=400)

        cands: List[Dict[str, Any]] = [{"source": "stub_repeat", "text": None,
                                        "steps": _steps_json(expand_planned_step(planned))}]
        seen = {json.dumps([s["command"] for s in cands[0]["steps"]])}
        for i, t in enumerate(texts):
            c: Dict[str, Any] = {"source": "teacher", "idx": i, "text": t}
            try:
                steps = parse_atomic_steps(t)
                c["steps"] = _steps_json(steps)
                why = steps_policy_violation(steps)
                if why:
                    c["policy"] = why
            except Sys3ParseError as e:
                c["parse_error"] = e.subtype
            key = json.dumps([s["command"] for s in c.get("steps", [])])
            if c.get("steps") and key in seen:
                c["duplicate"] = True
            elif c.get("steps"):
                seen.add(key)
            cands.append(c)

        def run(c):
            if not c.get("steps") or c.get("policy") or c.get("duplicate"):
                return
            f = None
            try:
                f = env.fork(snap)
                steps = parse_atomic_steps(json.dumps({"steps": c["steps"]}))
                f._expander = lambda action, context, _s=steps: _s
                t0 = time.time()
                _, _, _, _, info = f.step(planned)
                c["result"] = _result(info, before, time.time() - t0)
            except Exception as exc:
                c["env_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if f is not None:
                    try:
                        f.end_episode()
                    except Exception:
                        pass

        with cf.ThreadPoolExecutor(max_workers=a.exec_workers) as ex:
            list(ex.map(run, cands))

        executed = [c for c in cands if c.get("result")]
        best = max((c for c in executed if c["source"] == "teacher"), key=_rank_key, default=None)
        best_r = (best or {}).get("result") or {}
        accepted = bool(best and (best_r.get("episode_complete") or (best_r.get("gain") or 0) > 0))
        if a.dry_run and best is not None:
            # no installed-package probe without Docker: success stands in
            # for gain (pipeline smoke tests only; real runs need gain > 0)
            accepted = best_r.get("exec_status") == "success"
        row = {
            "case_id": seed["case_id"] + "/recover", "seed": seed["case_id"], "repo": seed["repo"],
            "split": seed["split"], "category": seed.get("category"), "kind": "recovery",
            "prefix": seed["prefix"], "planned": seed["planned"], "context": ctx.to_dict(), "docs": chunks,
            "required_before": before, "required_total": first.get("required_total"),
            "candidates": cands, "best_idx": cands.index(best) if best else None,
            "accepted": accepted, "status": "ok", "teacher": a.teacher,
        }
        _append(out, row)
        return "accepted" if accepted else "no_gain"
    except Exception as exc:
        log.error("%s: %s\n%s", seed["case_id"], exc, traceback.format_exc(limit=4))
        return "error"
    finally:
        _delete_snapshot(env, snap)
        try:
            env.end_episode()
        except Exception:
            pass


def phase_recover(a) -> None:
    from src.system3_command_generation.slot.generate import HFGenerator

    cases = _read_jsonl(Path(a.out) / "cases.jsonl")
    out = Path(a.out) / "recovery.jsonl"
    done = {r.get("seed") for r in _read_jsonl(out)}
    seeds = [c for c in cases if c.get("recovery_seed") and c["split"] in a.splits and c["case_id"] not in done]
    # one seed per (repo, target, prefix) is enough; the fresh and after-ISOLATE
    # variants of the same manifest are distinct states, keep both.
    if a.limit:
        seeds = seeds[: a.limit]
    log.info("recover: %d seeds (splits %s), teacher %s, k=%d", len(seeds), a.splits, a.teacher, a.k)
    if not seeds:
        return
    teacher = FakeTeacher() if a.teacher == "fake" else HFGenerator(a.teacher, load_in_4bit=a.teacher_4bit)
    docs = DocProvider(docs_dir=Path(a.out) / "docs", mcp_url=a.mcp_url)
    gen_lock = threading.Lock()
    tally = collections.Counter()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(recover_seed, s, a, docs, teacher, gen_lock, out): s for s in seeds}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            tally[fu.result()] += 1
            log.info("recover %d/%d %s", i, len(seeds), dict(tally))


# ------------------------------------------------------------------ export

def _prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    from src.system3_command_generation.slot import Sys3Context
    return build_messages(row["planned"], Sys3Context.from_dict(row["context"]), render_docs(row["docs"]))


def _signature(stderr: str) -> str:
    m = re.search(r"(ERROR: [A-Za-z][^\n]{0,60}|error: [a-z-]+)", stderr or "")
    sig = m.group(1) if m else (stderr or "").strip().splitlines()[-1:] or [""]
    sig = sig if isinstance(sig, str) else sig[0]
    return re.sub(r"\d[\w.]*", "#", sig)[:70]


def phase_export(a) -> None:
    base = Path(a.out)
    cases = _read_jsonl(base / "cases.jsonl")
    rec = [r for r in _read_jsonl(base / "recovery.jsonl") if r.get("status") == "ok"]
    rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    test_cases: List[Dict[str, Any]] = []
    seen_prompts = set()
    stats: Dict[str, Any] = {"cases": collections.Counter(), "recovery": collections.Counter(),
                             "sft": collections.Counter(), "failure_signatures": collections.Counter(),
                             "stub_repeat_gain": collections.Counter()}

    for c in cases:
        stats["cases"][f"{c['split']}/{c['kind']}/{'acc' if c['accepted'] else 'rej'}"] += 1
        if c.get("recovery_seed"):
            stats["failure_signatures"][_signature(c["result"].get("stderr_tail", ""))] += 1
        if c["split"] == "test":
            test_cases.append({k: c[k] for k in ("case_id", "repo", "split", "category", "kind", "prefix",
                                                 "planned", "result", "recovery_seed")})
            continue
        if not c["accepted"] or c["split"] not in rows:
            continue
        msgs = _prompt_messages(c)
        h = hashlib.sha1(msgs[1]["content"].encode()).hexdigest()
        if h in seen_prompts:
            continue
        seen_prompts.add(h)
        target = json.dumps({"steps": c["reference_steps"]}, ensure_ascii=False)
        rows[c["split"]].append({"id": c["case_id"], "kind": c["kind"], "source": "reference",
                                 "messages": msgs + [{"role": "assistant", "content": target}]})
        stats["sft"][f"{c['split']}/{c['kind']}"] += 1

    for r in rec:
        stub = next((c for c in r["candidates"] if c["source"] == "stub_repeat"), None)
        if stub and stub.get("result"):
            stats["stub_repeat_gain"]["gain>0" if (stub["result"].get("gain") or 0) > 0 else "gain=0"] += 1
        stats["recovery"][f"{r['split']}/{'acc' if r['accepted'] else 'rej'}"] += 1
        if not r["accepted"] or r["split"] not in rows:
            continue
        best = r["candidates"][r["best_idx"]]
        msgs = _prompt_messages(r)
        target = json.dumps({"steps": best["steps"]}, ensure_ascii=False)
        for _ in range(max(1, a.recovery_upsample)):
            rows[r["split"]].append({"id": r["case_id"], "kind": "recovery", "source": "teacher",
                                     "gain": best["result"].get("gain"),
                                     "complete": best["result"].get("episode_complete"),
                                     "messages": msgs + [{"role": "assistant", "content": target}]})
        stats["sft"][f"{r['split']}/recovery"] += 1

    for split, rs in rows.items():
        with (base / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (base / "test_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in test_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {k: dict(v.most_common()) if isinstance(v, collections.Counter) else v for k, v in stats.items()}
    stats["files"] = {"train": len(rows["train"]), "val": len(rows["val"]), "test_cases": len(test_cases)}
    (base / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["files"]), json.dumps(stats["sft"]))


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["collect", "recover", "export"], required=True)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--splits", nargs="+", default=None, choices=["train", "val", "test"],
                    help="collect: default train val test; recover: default train val")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--network", choices=["proxy", "none"], default="proxy")
    ap.add_argument("--workers", type=int, default=3, help="repos (collect) / seeds (recover) in parallel")
    ap.add_argument("--exec_workers", type=int, default=3, help="recover: candidates executed in parallel per seed")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N repos per split / N seeds")
    ap.add_argument("--mcp_url", default=None, help="optional live MCP server for pip docs")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--teacher_4bit", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--recovery_upsample", type=int, default=2)
    ap.add_argument("--dry_run", action="store_true", help="no Docker: synthetic fixtures (pipeline smoke test only)")
    a = ap.parse_args(argv)
    if a.splits is None:
        a.splits = ["train", "val", "test"] if a.phase == "collect" else ["train", "val"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"collect": phase_collect, "recover": phase_recover, "export": phase_export}[a.phase](a)


if __name__ == "__main__":
    main()
