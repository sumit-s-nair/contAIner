"""System 3 case-level evaluation (Stage 3A).

For every held-out case (datasets/system3-slot/test_cases.jsonl, built by
build_system3_dataset from test-split repos): reproduce the state in the
sandbox (reset, prefix steps, and for recovery cases the stub's failed
attempt), snapshot it, then let each System 3 policy expand the planner step
on its own fork and execute it.  Ground truth = the env's (installed packages
satisfying declared constraints, pip check for completion).

Policies (all greedy):
  stub                 the deterministic stub (on recovery = repeat the failed command)
  base                 --model_id without adapter (zero-shot, same prompt + docs)
  <adapter dir>        each --adapters entry
  <name>+nodocs        with --no_docs_ablation: same model, documentation removed

    python -m scripts.evaluate_system3 --adapters outputs/system3_sft_run_XXXX/best --no_docs_ablation

Outputs (--out_dir, default results/system3_eval_<timestamp>): cases.jsonl
(one line per case x policy, full outputs), summary.json, summary.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rl_env.system3_stub import expand_planned_step
from src.system2_planner.models import ActionType, PlannedStep
from src.system3_command_generation.slot import DocProvider
from scripts.build_system3_dataset import (
    _delete_snapshot, _read_jsonl, _required_now, _result, _snapshot, make_env,
)

log = logging.getLogger("evaluate_system3")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


class NoDocs(DocProvider):
    def for_step(self, *a, **k):
        return []


def _stub_policy(action, ctx):
    return expand_planned_step(action)


def _norm(cmds):
    return [" ".join(c.split()) for c in cmds]


def eval_case(case: Dict[str, Any], policies: Dict[str, Any], corpus: Dict[str, Any], a) -> List[Dict[str, Any]]:
    entry = next((e for e in corpus["splits"][case["split"]] if e["repo"] == case["repo"]),
                 {"repo": case["repo"], "category": case.get("category")})
    planned = PlannedStep(ActionType(case["planned"]["action_type"]), case["planned"]["description"],
                          case["planned"]["target"])
    prefix = [PlannedStep(ActionType(p["action_type"]), p["description"], p["target"]) for p in case["prefix"]]
    recovery = bool(case.get("recovery_seed"))
    env = make_env(a.corpus, a.network, a.dry_run)
    snap, rows = None, []
    try:
        env.reset(entry)
        for p in prefix:
            env.step(p)
        failed_cmds = None
        if recovery:
            _, _, _, _, first = env.step(planned)
            if first.get("exec_status") != "failed":
                return [{"case_id": case["case_id"], "status": "not_reproduced"}]
            failed_cmds = _norm([s.command for s in env._expanded_history[-1]])
            if env._docker_executor is not None:
                now = env._query_installed()
                if now is not None:
                    env._installed_now = now
        before = _required_now(env)
        snap = _snapshot(env)
        for name, pol in policies.items():
            row: Dict[str, Any] = {"case_id": case["case_id"], "repo": case["repo"], "kind": case["kind"],
                                   "recovery": recovery, "policy": name, "status": "ok"}
            f = None
            try:
                f = env.fork(snap)
                f._expander = pol
                mark = len(pol.log) if hasattr(pol, "log") else 0
                t0 = time.time()
                _, _, _, _, info = f.step(planned)
                row["result"] = _result(info, before, time.time() - t0)
                gen = pol.log[mark:] if hasattr(pol, "log") else []
                row["generation"] = gen[0] if gen else None
                cmds = _norm([s.command for s in f._expanded_history[-1]])
                row["commands"] = cmds
                row["parse_ok"] = (gen[0]["status"] == "ok") if gen else True
                row["repeats_failed"] = bool(recovery and failed_cmds is not None and cmds == failed_cmds)
            except Exception as exc:
                row["status"] = "env_error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if f is not None:
                    try:
                        f.end_episode()
                    except Exception:
                        pass
            rows.append(row)
    except Exception as exc:
        rows.append({"case_id": case["case_id"], "status": "case_error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        _delete_snapshot(env, snap)
        try:
            env.end_episode()
        except Exception:
            pass
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            by[(r["policy"], "recovery" if r["recovery"] else "canonical")].append(r)
    for (pol, kind), rs in sorted(by.items()):
        n = len(rs)
        res = [r["result"] for r in rs]
        m = {
            "n": n,
            "parse_ok_rate": sum(r["parse_ok"] for r in rs) / n,
            "exec_success_rate": sum(x["exec_status"] == "success" for x in res) / n,
            "verified_success_rate": sum(x["exec_status"] == "success" and bool(x["verified"]) for x in res) / n,
            "blocked": sum(bool(x.get("blocked")) for x in res),
        }
        if kind == "recovery":
            gains = [x.get("gain") or 0 for x in res]
            m.update({
                "gain_rate": sum(g > 0 for g in gains) / n,
                "mean_gain": sum(gains) / n,
                "complete_rate": sum(bool(x.get("episode_complete")) for x in res) / n,
                "repeats_failed_rate": sum(r["repeats_failed"] for r in rs) / n,
            })
        out.setdefault(pol, {})[kind] = m
    return out


def _md(summary: Dict[str, Any]) -> str:
    lines = []
    for kind, cols in (("canonical", ["n", "parse_ok_rate", "exec_success_rate", "verified_success_rate", "blocked"]),
                       ("recovery", ["n", "parse_ok_rate", "exec_success_rate", "gain_rate", "mean_gain",
                                     "complete_rate", "repeats_failed_rate", "blocked"])):
        lines += [f"\n### {kind}\n", "| policy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for pol, per in summary.items():
            if kind in per:
                m = per[kind]
                lines.append(f"| {pol} | " + " | ".join(f"{m[c]:.2f}" if isinstance(m[c], float) else str(m[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="datasets/system3-slot/test_cases.jsonl")
    ap.add_argument("--corpus", default="datasets/repo_corpus/corpus_manifest_v3.json")
    ap.add_argument("--docs_dir", default="datasets/system3-slot/docs")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--no_stub", action="store_true")
    ap.add_argument("--no_docs_ablation", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=["canonical", "recovery"], choices=["canonical", "recovery"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--network", choices=["proxy", "none"], default="proxy")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cases = _read_jsonl(Path(a.cases))
    cases = [c for c in cases if ("recovery" if c.get("recovery_seed") else "canonical") in a.kinds]
    if a.limit:
        cases = cases[: a.limit]
    corpus = json.loads(Path(a.corpus).read_text(encoding="utf-8"))
    out = Path(a.out_dir or f"results/system3_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2) + "\n", encoding="utf-8")

    docs = DocProvider(docs_dir=Path(a.docs_dir))
    policies: Dict[str, Any] = {}
    if not a.no_stub:
        policies["stub"] = _stub_policy
    gens = []
    if not a.no_base:
        gens.append(("base", None))
    gens += [(Path(p).parent.name + "/" + Path(p).name, p) for p in a.adapters]
    if gens:
        from src.system3_command_generation.slot.generate import HFGenerator, Sys3ModelExpander
        for name, adapter in gens:
            g = HFGenerator(a.model_id, adapter=adapter)
            policies[name] = Sys3ModelExpander(generator=g, docs=docs)
            if a.no_docs_ablation:
                policies[name + "+nodocs"] = Sys3ModelExpander(generator=g, docs=NoDocs())
    log.info("%d cases x %d policies: %s", len(cases), len(policies), list(policies))

    rows: List[Dict[str, Any]] = []
    with (out / "cases.jsonl").open("a", encoding="utf-8") as f:
        for i, c in enumerate(cases, 1):
            for r in eval_case(c, policies, corpus, a):
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0:
                log.info("%d/%d cases", i, len(cases))
    summary = summarize(rows)
    summary["_counts"] = dict(collections.Counter(r.get("status") for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = _md({k: v for k, v in summary.items() if not k.startswith("_")})
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
