"""HF generation for System 3: teacher sampling and the trained-model expander."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.sandbox.models import AtomicStep
from src.system2_planner.models import PlannedStep

from .context import Sys3Context
from .docs import DocProvider, render_docs
from .parse import Sys3ParseError, parse_atomic_steps, steps_policy_violation
from .prompt import build_messages

log = logging.getLogger("system3.generate")


class HFGenerator:
    """Chat-template generation with an optional LoRA adapter."""

    def __init__(self, model_id: str, adapter: Optional[str] = None, dtype: str = "bfloat16",
                 device: Optional[str] = None, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: Dict[str, Any] = {"dtype": getattr(torch, dtype) if self.device != "cpu" else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        if not load_in_4bit:
            model = model.to(self.device)
        model.eval()
        self.model = model

    def generate(self, messages: List[Dict[str, str]], n: int = 1, temperature: float = 0.0,
                 top_p: float = 0.95, max_new_tokens: int = 400) -> List[str]:
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        sample = temperature > 0 and n >= 1
        gen_kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tokenizer.pad_token_id)
        if sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        start = enc["input_ids"].shape[1]
        return [self.tokenizer.decode(o[start:], skip_special_tokens=True) for o in out]


@dataclass
class Sys3ModelExpander:
    """``PlannerEnv`` expander backed by a trained System 3 model.

    Greedy decoding (deterministic, so ``PlannerEnv.fork`` replay matches).
    ``fallback_to_stub``: on a parse error or policy violation, use the stub's
    expansion instead of returning nothing; every fallback is counted so
    evaluation can report raw model behaviour.
    """
    generator: HFGenerator
    docs: DocProvider
    fallback_to_stub: bool = False
    max_new_tokens: int = 400
    log: List[Dict[str, Any]] = field(default_factory=list)

    def __call__(self, action: PlannedStep, ctx: Sys3Context) -> List[AtomicStep]:
        stderr = (ctx.previous or {}).get("stderr", "")
        chunks = self.docs.for_step(action.action_type, action.target or "", action.description or "", stderr)
        msgs = build_messages(action, ctx, render_docs(chunks))
        text = self.generator.generate(msgs, n=1, temperature=0.0, max_new_tokens=self.max_new_tokens)[0]
        rec: Dict[str, Any] = {"action": action.action_type.value, "target": action.target,
                               "attempt": ctx.attempt, "text": text}
        try:
            steps = parse_atomic_steps(text)
            why = steps_policy_violation(steps)
            if why:
                raise Sys3ParseError("POLICY", why)
            rec["status"] = "ok"
            rec["commands"] = [s.command for s in steps]
        except Sys3ParseError as e:
            rec["status"] = e.subtype
            rec["error"] = str(e)
            steps = []
            if self.fallback_to_stub:
                from src.rl_env.system3_stub import expand_planned_step
                steps = expand_planned_step(action)
                rec["fallback"] = True
        self.log.append(rec)
        return steps
