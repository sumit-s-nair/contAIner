#!/usr/bin/env python3
"""
scripts/clone_missing_corpus.py
===============================
Stage 2G: clone train/val repos of corpus_manifest_v3.json that are not yet
in cache/repo_clones (shallow clone via RepoLoader).  The test split is not
touched.  Failures are printed and listed, not hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.rl_env.repo_loader import RepoLoader  # noqa: E402

corpus = json.loads((PROJECT_ROOT / "datasets/repo_corpus/corpus_manifest_v3.json").read_text(encoding="utf-8"))
cache = PROJECT_ROOT / "cache/repo_clones"
loader = RepoLoader(reset_between_episodes=False)
failed = []
for split in ("train", "val"):
    for e in corpus["splits"][split]:
        if list(cache.glob(e["repo"].replace("/", "_") + "_*")):
            continue
        print(f"[{split}] cloning {e['repo']} ...", flush=True)
        try:
            loader.load(e)
        except Exception as exc:
            print(f"   FAILED: {type(exc).__name__}: {exc}")
            failed.append((split, e["repo"], f"{type(exc).__name__}: {exc}"))
print(f"done; {len(failed)} failed")
for f in failed:
    print("  ", f)
