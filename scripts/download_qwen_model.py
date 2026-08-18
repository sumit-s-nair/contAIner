# -*- coding: utf-8 -*-
"""
Download and cache Qwen2.5-0.5B-Instruct for offline abstractive compression.

Run with the project venv:
    .venv\Scripts\python.exe scripts\download_qwen_model.py

The model (~994 MB) will be stored in HuggingFace's local hub cache:
    %USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct

After this script completes successfully, run the eval:
    .venv\Scripts\python.exe -X utf8 <path_to_eval_compression.py>
"""

from __future__ import annotations
import sys
import time

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def main() -> None:
    print(f"=== Downloading {MODEL_ID} ===")
    print("This will take a few minutes depending on your connection speed.")
    print()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed.")
        print("Run: .venv\\Scripts\\pip.exe install transformers>=4.40.0 torch>=2.1.0")
        sys.exit(1)

    # --- Tokenizer ---
    print("[1/2] Downloading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"      Tokenizer ready in {time.time()-t0:.1f}s")

    # --- Model ---
    print("[2/2] Downloading model weights (this is the large step)...")
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    elapsed = time.time() - t1
    print(f"      Model loaded in {elapsed:.1f}s")

    # --- Smoke test ---
    print()
    print("[smoke test] Running a tiny generation to confirm model works...")
    import torch
    prompt = "Hello, world. Summarize in one word:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"      Generated: {result!r}")
    print()
    print("SUCCESS - model is cached and working.")
    print(f"Device: {model.device}")
    print()
    print("You can now run the eval script:")
    print(r"  .venv\Scripts\python.exe -X utf8 <path_to_eval_compression.py>")


if __name__ == "__main__":
    main()
