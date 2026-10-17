"""System 3 SFT (Stage 3A): QLoRA on Qwen2.5-Coder-1.5B-Instruct over the slot dataset.

    python -m scripts.train_system3_sft                         # defaults below
    python -m scripts.train_system3_sft --epochs 3 --lr 2e-4 --max_len 2048
    python -m scripts.train_system3_sft --smoke                 # tiny CPU run (tests)

Input: datasets/system3-slot/{train,val}.jsonl from build_system3_dataset
(chat messages: system, user, assistant).  Loss on assistant tokens only.
Each eval writes val loss plus a greedy-generation check on --eval_gen_n val
examples (parse rate, policy violations, exact command match vs target), saves
the adapter to checkpoint-<step>/, and keeps best/ by val loss.  Examples
longer than --max_len are skipped (counted), never truncated: a cut target
would teach the model to stop mid-JSON.

Output: outputs/system3_sft_run_<6hex>/ with config.json, training_log.jsonl,
checkpoint-*/, best/, final/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.system3_command_generation.slot import (
    Sys3ParseError,
    parse_atomic_steps,
    steps_policy_violation,
)

log = logging.getLogger("train_system3_sft")

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_DIR = Path("datasets/system3-slot")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def encode(tokenizer, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
    """input_ids, labels (-100 on the prompt)."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: full conversation does not extend the prompt")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError("tokenization of prompt is not a prefix of the full conversation")
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels


def build_examples(tokenizer, rows, max_len: int) -> Tuple[List[Tuple[List[int], List[int]]], int]:
    out, skipped = [], 0
    for r in rows:
        ids, labels = encode(tokenizer, r["messages"])
        if len(ids) > max_len:
            skipped += 1
            continue
        out.append((ids, labels))
    return out, skipped


def batches(examples, batch_size: int, pad_id: int, shuffle: bool, rng: random.Random):
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [examples[j] for j in idx[i: i + batch_size]]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, : len(x)] = torch.tensor(x)
            lab[k, : len(y)] = torch.tensor(y)
            att[k, : len(x)] = 1
        yield ids, lab, att


def load_model(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(a.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: Dict[str, Any] = {}
    if a.device == "cpu":
        kwargs["dtype"] = torch.float32
    elif a.qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model_id, **kwargs)
    if a.qlora and a.device != "cpu":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif a.device != "cpu":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    if not (a.qlora and a.device != "cpu"):
        model = model.to(a.device)
    return model, tok


@torch.no_grad()
def val_loss(model, examples, a, pad_id) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, att in batches(examples, a.eval_batch_size, pad_id, False, random.Random(0)):
        ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
        ntok = int((lab[:, 1:] != -100).sum())
        tot += float(out.loss) * ntok
        n += ntok
    model.train()
    return tot / max(1, n)


def _norm(cmds: List[str]) -> List[str]:
    return [" ".join(c.split()) for c in cmds]


@torch.no_grad()
def gen_check(model, tok, rows, a) -> Dict[str, Any]:
    model.eval()
    tok.padding_side = "left"
    stats = {"n": 0, "parsed": 0, "policy_ok": 0, "exact": 0, "recovery_n": 0, "recovery_exact": 0,
             "recovery_repeats_failed": 0}
    samples = []
    for r in rows[: a.eval_gen_n]:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        enc = tok([prompt], return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        target = _norm([s["command"] for s in json.loads(r["messages"][-1]["content"])["steps"]])
        stats["n"] += 1
        rec = r.get("kind") == "recovery"
        stats["recovery_n"] += int(rec)
        try:
            steps = parse_atomic_steps(text)
            stats["parsed"] += 1
            stats["policy_ok"] += int(steps_policy_violation(steps) is None)
            got = _norm([s.command for s in steps])
            ex = got == target
            stats["exact"] += int(ex)
            stats["recovery_exact"] += int(ex and rec)
            if rec:
                # did it just repeat the command that already failed?
                prev = r["messages"][1]["content"].split("## Previous attempt (failed)\ncommands:\n", 1)
                if len(prev) == 2:
                    failed = _norm([l.strip() for l in prev[1].split("\nexit code:")[0].splitlines()])
                    stats["recovery_repeats_failed"] += int(got == failed)
        except Sys3ParseError:
            got = None
        if len(samples) < 5:
            samples.append({"id": r.get("id"), "kind": r.get("kind"), "text": text[:600]})
    tok.padding_side = "right"
    model.train()
    n = max(1, stats["n"])
    stats.update({"parse_rate": stats["parsed"] / n, "exact_rate": stats["exact"] / n,
                  "policy_ok_rate": stats["policy_ok"] / n})
    return {"gen": stats, "samples": samples}


def save_adapter(model, tok, path: Path, extra: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "trainer_state.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=0, help="optimizer steps between evals (0 = once per epoch)")
    ap.add_argument("--eval_gen_n", type=int, default=64)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--no_qlora", dest="qlora", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny model, CPU, few steps (pipeline test)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.device = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    if a.smoke:
        a.qlora = False
        a.eval_every, a.eval_gen_n, a.max_new_tokens = 1, 2, 48
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    run_id = uuid.uuid4().hex[:6]
    out = Path(a.out_dir or f"outputs/system3_sft_run_{run_id}")
    out.mkdir(parents=True, exist_ok=True)

    data = Path(a.data_dir)
    train_rows, val_rows = read_jsonl(data / "train.jsonl"), read_jsonl(data / "val.jsonl")
    model, tok = load_model(a)
    tok.padding_side = "right"
    train, skip_t = build_examples(tok, train_rows, a.max_len)
    val, skip_v = build_examples(tok, val_rows, a.max_len)
    if not train:
        raise SystemExit("no training examples (all longer than --max_len?)")

    steps_per_epoch = math.ceil(len(train) / a.batch_size / a.grad_accum)
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    eval_every = a.eval_every or steps_per_epoch
    cfg = {**vars(a), "run_id": run_id, "n_train": len(train), "n_val": len(val), "skipped_train_too_long": skip_t,
           "skipped_val_too_long": skip_v, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
           "data_stats": json.loads((data / "stats.json").read_text()) if (data / "stats.json").exists() else None}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("run %s: %d train (%d too long), %d val, %d optimizer steps", run_id, len(train), skip_t, len(val), total_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    warm = max(1, int(total_steps * a.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

    logf = (out / "training_log.jsonl").open("a", encoding="utf-8")
    best = math.inf
    step, micro, t0 = 0, 0, time.time()
    rng = random.Random(a.seed)
    model.train()
    run_loss, run_n = 0.0, 0
    done = False
    epoch = 0
    while not done:
        for ids, lab, att in batches(train, a.batch_size, tok.pad_token_id, True, rng):
            ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
            loss.backward()
            run_loss += float(loss) * a.grad_accum
            run_n += 1
            micro += 1
            if micro % a.grad_accum:
                continue
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch, "loss": run_loss / max(1, run_n), "grad_norm": gnorm,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            run_loss, run_n = 0.0, 0
            if step % eval_every == 0 or step == total_steps:
                rec["val_loss"] = val_loss(model, val, a, tok.pad_token_id) if val else None
                rec.update(gen_check(model, tok, val_rows, a))
                ck = out / f"checkpoint-{step}"
                save_adapter(model, tok, ck, rec)
                if rec["val_loss"] is not None and rec["val_loss"] < best:
                    best = rec["val_loss"]
                    shutil.rmtree(out / "best", ignore_errors=True)
                    shutil.copytree(ck, out / "best")
                log.info("step %d/%d val_loss %s gen %s", step, total_steps, rec["val_loss"],
                         {k: rec["gen"][k] for k in ("parse_rate", "exact_rate", "policy_ok_rate", "recovery_repeats_failed")})
            elif step % 10 == 0:
                log.info("step %d/%d loss %.4f gnorm %.3f", step, total_steps, rec["loss"], gnorm)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step >= total_steps or (a.smoke and step >= 2):
                done = True
                break
        epoch += 1
    save_adapter(model, tok, out / "final", {"step": step, "best_val_loss": best})
    logf.close()
    log.info("done: %s (best val loss %.4f)", out, best)
    return out


if __name__ == "__main__":
    main()
