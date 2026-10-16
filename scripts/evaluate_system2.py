"""
scripts/evaluate_system2.py
===========================
Stage 2H baseline-vs-trained evaluation for the System 2 planner.

Each policy (the untrained base model and any number of LoRA checkpoints) is
rolled out once per repo with greedy decoding, using exactly the training-time
prompt (``to_chat_prompt(obs, tokenizer=...)``), parser (``parse_planned_step``),
environment (``PlannerEnv``, AUTO_DENY review, egress proxy) and step budget
(``EnvConfig.max_steps``).  Unlike the old script, an episode is NOT cut at the
first failed step: training lets the policy recover, so evaluation does too.
A parse error ends the episode (greedy decoding on an unchanged observation
would repeat it until the budget runs out).

Episodes whose env reset/step raised are counted under ``env_errors`` and
excluded from every denominator.

Usage (from the repo root, as a module):
    python -m scripts.evaluate_system2 \\
        --checkpoints outputs/system2_grpo_run_b45f5d/checkpoint-200 \\
                      outputs/system2_grpo_run_b45f5d/checkpoint-300 \\
        --splits val test

Outputs (``--out_dir``, default results/system2_eval_<timestamp>):
    episodes.jsonl   one line per (policy, split, repo) with the full action trace
    summary.json     per-policy, per-split metrics
    summary.md       the same as a table
    demo_traces/     markdown traces for the first ``--demo_n`` repos of each split
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.rl_env.env import PlannerEnv, EnvConfig, TrainingReviewPolicy
from src.rl_env.docker_executor import DockerExecutorConfig, NetworkConfig
from src.rl_env.parser import parse_planned_step
from src.sandbox.models import RiskTier
from src.system2_planner.models import ActionType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MANIFEST = "datasets/repo_corpus/corpus_manifest_v3_pilot.json"
MAX_NEW_TOKENS = 512  # same as training (SAMPLING_CONFIG)


# ---------------------------------------------------------------------------
# Model loading (mirrors train_system2_grpo.py: 4-bit nf4, fp16 compute)
# ---------------------------------------------------------------------------

def load_policy(model_id: str, adapter_path: Optional[str] = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def policy_name(adapter_path: Optional[str]) -> str:
    if not adapter_path:
        return "base"
    p = Path(adapter_path)
    return f"{p.parent.name}/{p.name}"


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _has_conflicts(obs) -> bool:
    try:
        return bool(obs.repo_manifest.conflicts)
    except AttributeError:
        return False


def run_episode(env, model, tokenizer, repo: Dict[str, Any], max_new_tokens: int = MAX_NEW_TOKENS,
                trace: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Roll out one greedy episode.  Returns an episode record; ``env_error`` is
    set (and the record must be excluded from metrics) if reset/step raised.
    """
    rec: Dict[str, Any] = {
        "repo": repo["repo"], "category": repo.get("category", "unknown"),
        "completed": False, "steps": 0, "reward": 0.0, "actions": [],
        "parse_error": False, "parse_error_text": "", "has_conflicts": False,
        "blocked": 0, "review_denied": 0, "env_error": "",
    }
    try:
        obs, _ = env.reset(repo_entry=repo)
    except Exception as e:  # noqa: BLE001
        rec["env_error"] = f"reset: {type(e).__name__}: {e}"
        return rec
    rec["has_conflicts"] = _has_conflicts(obs)

    done = False
    info: Dict[str, Any] = {}
    while not done:
        msgs = env._serializer.to_chat_prompt(obs, tokenizer=tokenizer)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if trace is not None:
            trace.append(f"## Step {rec['steps'] + 1}\n\n**Output:**\n```\n{completion}\n```\n")

        try:
            action = parse_planned_step(completion)
        except Exception as e:  # noqa: BLE001  (any parser failure)
            rec["parse_error"] = True
            rec["parse_error_text"] = f"{type(e).__name__}: {e}"[:300]
            rec["steps"] += 1
            if trace is not None:
                trace.append(f"**Parse error:** {rec['parse_error_text']}\n")
            break

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:  # noqa: BLE001
            rec["env_error"] = f"step {rec['steps'] + 1}: {type(e).__name__}: {e}"
            return rec

        rec["steps"] += 1
        rec["reward"] += float(reward)
        tier = info.get("exec_tier")
        if info.get("blocked") or tier == RiskTier.BLOCKED.value:
            rec["blocked"] += 1
        elif tier == RiskTier.REVIEW.value:
            rec["review_denied"] += 1
        rec["actions"].append({
            "action_type": action.action_type.value,
            "target": action.target or "",
            "reward": float(reward),
            "exec_status": str(info.get("exec_status", "")),
            "install_progress": info.get("install_progress"),
        })
        if trace is not None:
            trace.append(f"**Action:** `{action.action_type.value} {action.target or ''}` "
                         f"-> reward {reward:.2f}, breakdown {info.get('reward_breakdown', {})}\n")
        done = bool(terminated or truncated)

    rec["completed"] = bool(info.get("episode_complete", False))
    return rec


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_ISOLATE = ActionType.ISOLATE.value
_INSTALL = ActionType.INSTALL.value


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics over non-env-error episodes."""
    ok = [r for r in records if not r["env_error"]]
    n = len(ok)
    def rate(k): return (k / n) if n else 0.0
    completed = [r for r in ok if r["completed"]]
    no_conf = [r for r in ok if not r["has_conflicts"]]
    unnec_iso = [r for r in no_conf if any(a["action_type"] == _ISOLATE for a in r["actions"])]
    first = [r["actions"][0]["action_type"] if r["actions"] else None for r in ok]
    total_actions = sum(len(r["actions"]) for r in ok)
    total_isolates = sum(sum(a["action_type"] == _ISOLATE for a in r["actions"]) for r in ok)
    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in ok:
        c = by_cat.setdefault(r["category"], {"total": 0, "completed": 0})
        c["total"] += 1
        c["completed"] += int(r["completed"])
    return {
        "episodes": n,
        "env_errors": len(records) - n,
        "completion_rate": rate(len(completed)),
        "completed": len(completed),
        "mean_steps": (sum(r["steps"] for r in ok) / n) if n else 0.0,
        "mean_steps_completed": (sum(r["steps"] for r in completed) / len(completed)) if completed else 0.0,
        "mean_reward": (sum(r["reward"] for r in ok) / n) if n else 0.0,
        "parse_error_episodes": sum(r["parse_error"] for r in ok),
        "parse_error_rate": rate(sum(r["parse_error"] for r in ok)),
        "no_conflict_episodes": len(no_conf),
        "unnecessary_isolate_rate": (len(unnec_iso) / len(no_conf)) if no_conf else 0.0,
        "first_step_isolate_rate": rate(sum(f == _ISOLATE for f in first)),
        "first_step_install_rate": rate(sum(f == _INSTALL for f in first)),
        "isolate_action_share": (total_isolates / total_actions) if total_actions else 0.0,
        "blocked_steps": sum(r["blocked"] for r in ok),
        "review_denied_steps": sum(r["review_denied"] for r in ok),
        "by_category": by_cat,
    }


def evaluate_model(model_name: str, model, tokenizer, test_repos: List[Dict[str, Any]],
                   demo_repos: List[str], out_dir: Path, network: str = "proxy",
                   split: str = "test", max_new_tokens: int = MAX_NEW_TOKENS,
                   manifest_path: str = DEFAULT_MANIFEST) -> Dict[str, Any]:
    """Evaluate one policy on a list of repos; returns summary metrics plus ``records``."""
    logger.info("Evaluating %s on %s (%d repos, network=%s)", model_name, split, len(test_repos), network)
    env = PlannerEnv(config=EnvConfig(
        dry_run=False,
        review_policy=TrainingReviewPolicy.AUTO_DENY,
        corpus_manifest_path=manifest_path,
        docker_config=DockerExecutorConfig(network=NetworkConfig(mode=network)),
    ))
    out_dir = Path(out_dir)
    demo_dir = out_dir / "demo_traces"
    records: List[Dict[str, Any]] = []
    try:
        for repo in tqdm(test_repos, desc=f"{model_name}:{split}"):
            trace = [f"# {repo['repo']} ({repo.get('category', '')})\n\n**Policy:** {model_name}  **Split:** {split}\n"] \
                if repo["repo"] in demo_repos else None
            t0 = time.time()
            rec = run_episode(env, model, tokenizer, repo, max_new_tokens=max_new_tokens, trace=trace)
            rec.update({"policy": model_name, "split": split, "seconds": round(time.time() - t0, 1)})
            records.append(rec)
            if rec["env_error"]:
                logger.error("ENV ERROR %s: %s", repo["repo"], rec["env_error"])
            if trace is not None:
                trace.append(f"\n## Result: {'COMPLETED' if rec['completed'] else 'NOT COMPLETED'}"
                             f"{' (env error: ' + rec['env_error'] + ')' if rec['env_error'] else ''}\n")
                demo_dir.mkdir(parents=True, exist_ok=True)
                safe = f"{model_name}_{split}_{repo['repo']}".replace("/", "_")
                (demo_dir / f"{safe}.md").write_text("\n".join(trace), encoding="utf-8")
    finally:
        env.close()

    metrics = summarize(records)
    metrics["records"] = records
    logger.info("%s/%s: completion %.2f (%d/%d), steps %.2f, unnecessary ISOLATE %.2f, parse errors %d, env errors %d",
                model_name, split, metrics["completion_rate"], metrics["completed"], metrics["episodes"],
                metrics["mean_steps"], metrics["unnecessary_isolate_rate"],
                metrics["parse_error_episodes"], metrics["env_errors"])
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="System 2 evaluation: base vs GRPO checkpoints")
    p.add_argument("--network", choices=["proxy", "none"], default="proxy", help="Docker network isolation mode")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Base HF model (must match training)")
    p.add_argument("--checkpoints", nargs="*", default=[], help="LoRA checkpoint dirs to evaluate")
    p.add_argument("--no_base", action="store_true", help="Skip the untrained base model")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--limit", type=int, default=0, help="Only the first N repos per split (smoke test)")
    p.add_argument("--demo_n", type=int, default=3, help="Write markdown traces for the first N repos per split")
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--out_dir", default=None, help="Default: results/system2_eval_<timestamp>")
    return p


def _md_table(summary: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    cols = ["episodes", "completion_rate", "mean_steps", "mean_steps_completed", "mean_reward",
            "unnecessary_isolate_rate", "first_step_isolate_rate", "first_step_install_rate",
            "parse_error_rate", "blocked_steps", "env_errors"]
    lines = []
    for split in sorted({s for v in summary.values() for s in v}):
        lines += [f"\n### {split}\n", "| policy | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for pol, per in summary.items():
            if split in per:
                m = per[split]
                cells = [f"{m[c]:.2f}" if isinstance(m[c], float) else str(m[c]) for c in cols]
                lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(cli_args=None):
    args = get_parser().parse_args(cli_args)
    out_dir = Path(args.out_dir or f"results/system2_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    policies: List[Optional[str]] = ([] if args.no_base else [None]) + list(args.checkpoints)
    for ck in args.checkpoints:
        if not (Path(ck) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{ck} has no adapter_config.json")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "policies": [policy_name(p) for p in policies]}, f, indent=2)

    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ep_file = out_dir / "episodes.jsonl"
    for adapter in policies:
        name = policy_name(adapter)
        model, tokenizer = load_policy(args.model_id, adapter)
        summary[name] = {}
        for split in args.splits:
            repos = manifest["splits"][split]
            if args.limit:
                repos = repos[: args.limit]
            demo = [r["repo"] for r in repos[: args.demo_n]]
            m = evaluate_model(name, model, tokenizer, repos, demo, out_dir, network=args.network,
                               split=split, max_new_tokens=args.max_new_tokens, manifest_path=args.manifest)
            with open(ep_file, "a", encoding="utf-8") as f:
                for r in m.pop("records"):
                    f.write(json.dumps(r) + "\n")
            summary[name][split] = m
            with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    (out_dir / "summary.md").write_text(_md_table(summary), encoding="utf-8")
    print(_md_table(summary))
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
