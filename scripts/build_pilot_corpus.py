#!/usr/bin/env python3
"""
scripts/build_pilot_corpus.py
=============================
Stage 2G: build the pilot training corpus from corpus_manifest_v3.json.

Keeps only train/val repos a perfect planner can actually complete with the
current System 3 stub (pip-only):
  * cloned in cache/repo_clones (scanned from the clone, no network)
  * Python is the only ecosystem (stub has no npm/cargo/... installer)
  * >= 1 *declared* Python dependency that applies on linux/py3.11
    (repos with only import-inferred deps are excluded: inferred names include
    local modules and guesses such as 'kernel', 'model', '-pytest')
  * at least one installable manifest (requirements*.txt / pyproject.toml /
    setup.py / setup.cfg)
  * no heavy framework in the required set, or as the repo itself
    (torch, tensorflow, jax, ...):
    multi-GB wheels through the proxy make episodes minutes long
  * the repo's declared Python constraint admits the sandbox interpreter
    (3.11): [project].requires-python, [tool.poetry.dependencies].python,
    setup.py/setup.cfg python_requires; and not every requirements*.txt line
    is gated off by a PEP 508 marker (pip would "succeed" installing nothing)
  * not recorded as unwinnable in datasets/repo_corpus/pilot_winnability.json
    (written by scripts/diagnostics/stage2g_winnability.py --all; applied only
    when the recorded sha matches): the scripted oracle -- manifests, then each
    still-missing required package -- must reach episode_complete with a
    positive total in the real sandbox.  Catches what static checks cannot,
    e.g. requirements pins that are mutually unsatisfiable.
The test split is copied unchanged and never scanned.  Every excluded repo is
listed with its reason in the output under "pilot_excluded".

Usage:
    python scripts/build_pilot_corpus.py [--scan-cache logs/pilot_scan_cache.jsonl]
Output: datasets/repo_corpus/corpus_manifest_v3_pilot.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repo_scan.scanner import scan_repo  # noqa: E402
from src.rl_env.reward import (WINNABILITY_CRITERION_VERSION, _SANDBOX_MARKER_ENV,  # noqa: E402
                               canonicalize_package, required_packages)
from src.system2_planner.models import TemplateInstance  # noqa: E402
from src.system2_planner.templates import TEMPLATES  # noqa: E402

SRC = PROJECT_ROOT / "datasets/repo_corpus/corpus_manifest_v3.json"
DST = PROJECT_ROOT / "datasets/repo_corpus/corpus_manifest_v3_pilot.json"
CACHE = PROJECT_ROOT / "cache/repo_clones"
WINNABILITY = PROJECT_ROOT / "datasets/repo_corpus/pilot_winnability.json"

# Stage 2G-fix-3: the val split shrank to 2 after the filters (31 -> 2), too
# small to say anything.  Move N_VAL_MOVES kept train repos to val, taken in
# this preference order (first ones still kept win), chosen for coverage:
# a conflict repo (pyproject + requirements), a large pyproject repo, a small
# requirements repo, a repo the manifest install alone does not finish.
# Deterministic and recorded in the output ("pilot_val_moves").  This is a
# pilot smoke-check split, not a paper metric.
VAL_FROM_TRAIN_PREFERENCE = [
    "zhayujie/CowAgent", "browser-use/browser-use", "AlexMathew/scrapple", "nfsi-canada/OrientPy",
    "TauricResearch/TradingAgents", "nvbn/thefuck", "ceccopierangiolieugenio/pyTermTk",
]
N_VAL_MOVES = 4

HEAVY = {
    "torch", "torchvision", "torchaudio", "tensorflow", "tensorflow-gpu", "tensorflow-cpu",
    "tf-nightly", "jax", "jaxlib", "paddlepaddle", "paddlepaddle-gpu", "mxnet", "cupy",
    "triton", "vllm", "xformers", "flash-attn", "deepspeed", "bitsandbytes", "tensorrt",
    "onnxruntime-gpu", "dgl", "torch-geometric",
    "pytorch",  # repo name of pytorch/pytorch (the dist is torch)
}
PROJECT_FILES = {"pyproject.toml", "setup.py", "setup.cfg"}
SANDBOX_PY = _SANDBOX_MARKER_ENV["python_full_version"]  # python:3.11-slim


def _poetry_to_pep440(spec: str) -> str:
    """Convert a Poetry python constraint ('^3.12,<4', '~3.10', '>=3.8', '*', '3.9')
    to a PEP 440 specifier string.  Raises ValueError on syntax it cannot map."""
    out = []
    for part in [p.strip() for p in spec.replace(" || ", "|").split(",")]:
        if not part or part == "*":
            continue
        if "|" in part:
            raise ValueError(f"poetry OR constraint not supported: {spec!r}")
        if part.startswith("^"):
            v = [int(x) for x in part[1:].split(".")]
            v += [0] * (2 - len(v))
            upper = f"{v[0] + 1}.0" if v[0] > 0 else f"0.{v[1] + 1}"
            out += [f">={part[1:]}", f"<{upper}"]
        elif part.startswith("~") and not part.startswith("~="):
            v = [int(x) for x in part[1:].split(".")]
            upper = f"{v[0]}.{v[1] + 1}" if len(v) >= 2 else f"{v[0] + 1}.0"
            out += [f">={part[1:]}", f"<{upper}"]
        elif part[0].isdigit():
            out.append(f"=={part}.*" if part.count(".") < 2 else f"=={part}")
        else:
            out.append(part)
    return ",".join(out)


def _python_constraints(root: Path) -> list:
    """(source, pep440_spec) pairs declared by the repo root."""
    import configparser
    import re
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        import tomli as tomllib
    found = []
    pp = root / "pyproject.toml"
    if pp.is_file():
        try:
            d = tomllib.loads(pp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            d = {}
        rp = (d.get("project") or {}).get("requires-python")
        if isinstance(rp, str) and rp.strip():
            found.append(("pyproject [project].requires-python", rp))
        pv = ((d.get("tool") or {}).get("poetry") or {}).get("dependencies", {}).get("python")
        if isinstance(pv, str) and pv.strip():
            found.append(("pyproject [tool.poetry.dependencies].python", _poetry_to_pep440(pv)))
    sp = root / "setup.py"
    if sp.is_file():
        m = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", sp.read_text(encoding="utf-8", errors="replace"))
        if m:
            found.append(("setup.py python_requires", m.group(1)))
    sc = root / "setup.cfg"
    if sc.is_file():
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read(sc, encoding="utf-8")
            v = cp.get("options", "python_requires", fallback=None)
        except Exception:
            v = None
        if v and v.strip():
            found.append(("setup.cfg python_requires", v.strip()))
    return found


def _python_incompatibility(s: dict):
    """Reason string if the repo cannot install on the sandbox interpreter, else None."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    root = CACHE / s["clone_dir"]
    try:
        cons = _python_constraints(root)
    except ValueError as exc:
        return f"python_constraint_unparsed: {exc}"
    for src, spec in cons:
        try:
            if not SpecifierSet(spec).contains(SANDBOX_PY, prereleases=True):
                return f"python_version_incompatible: {src} {spec!r} excludes {SANDBOX_PY}"
        except InvalidSpecifier:
            return f"python_constraint_unparsed: {src} {spec!r}"
    for f in s["manifest_files"]:
        if not f.lower().endswith(".txt") or not (root / f).is_file():
            continue
        lines = [ln.split(" #")[0].strip() for ln in (root / f).read_text(encoding="utf-8", errors="replace").splitlines()]
        reqs = [ln for ln in lines if ln and not ln.startswith(("#", "-"))]
        live = 0
        for ln in reqs:
            try:
                r = Requirement(ln)
            except InvalidRequirement:
                live += 1  # unknown syntax: assume pip installs something
                continue
            if r.marker is None or r.marker.evaluate(_SANDBOX_MARKER_ENV):
                live += 1
        if reqs and live == 0:
            return f"python_version_incompatible: every line of {f} is gated off by a marker on {SANDBOX_PY}"
    return None


def _scan(entry: dict) -> dict:
    safe = entry["repo"].replace("/", "_")
    clones = sorted(CACHE.glob(f"{safe}_*"))
    if not clones:
        return {"repo": entry["repo"], "reason": "not_cloned"}
    root = clones[0]
    try:
        m = scan_repo(str(root))
    except Exception as exc:  # record, don't hide
        return {"repo": entry["repo"], "reason": f"scan_error: {type(exc).__name__}: {exc}"}
    inst = TemplateInstance(template=TEMPLATES["setup_project"], manifest=m)
    py = m.ecosystems.get("python")
    rel_files = []
    if py:
        for f in py.manifest_files:
            try:
                rel_files.append(Path(f).resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                rel_files.append(Path(f).name)
    req = sorted(required_packages(inst))
    declared = sorted({canonicalize_package(k) for k in (py.dependencies if py else {})})
    return {
        "repo": entry["repo"],
        "clone_dir": root.name,
        "ecosystems": sorted(m.ecosystems),
        "conflicts": bool(m.conflicts),
        "manifest_files": sorted(set(rel_files)),
        "declared": declared,
        "required": req,
    }


def _reason(s: dict):
    if "reason" in s:
        return s["reason"]
    if s["ecosystems"] != ["python"]:
        return f"non_python_ecosystems: {s['ecosystems']}"
    if not s["declared"]:
        return "no_declared_python_deps"
    if not s["required"]:
        return "no_required_after_marker_filter"
    if not any(f.lower().endswith(".txt") or Path(f).name in PROJECT_FILES for f in s["manifest_files"]):
        return f"no_installable_manifest: {s['manifest_files']}"
    # the repo itself counts too: pytorch/pytorch does not list torch as a dep,
    # but `pip install .` would build it from source
    heavy = sorted((set(s["required"]) | {canonicalize_package(s["repo"].split("/")[-1])}) & HEAVY)
    if heavy:
        return f"heavy_packages: {heavy}"
    return _python_incompatibility(s)


def _unwinnable(entry: dict, verdicts: dict):
    """Reason string if the oracle run recorded this repo (same sha, current
    criterion) as not won."""
    v = verdicts.get(entry["repo"])
    if (not v or v.get("sha") != entry.get("sha") or v.get("won")
            or v.get("criterion") != WINNABILITY_CRITERION_VERSION):   # stale criterion: ignored
        return None
    return f"oracle_unwinnable: {json.dumps(v.get('why'))[:300]}"


def _move_to_val(splits: dict, preference=None, n: int = None) -> list:
    """Move up to n kept train repos to val, in preference order (in place)."""
    preference = VAL_FROM_TRAIN_PREFERENCE if preference is None else preference
    n = N_VAL_MOVES if n is None else n
    by_repo = {e["repo"]: e for e in splits["train"]}
    moved = [r for r in preference if r in by_repo][:n]
    splits["train"] = [e for e in splits["train"] if e["repo"] not in moved]
    splits["val"] = list(splits["val"]) + [by_repo[r] for r in moved]
    return moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-cache", default=None, help="jsonl cache of scan results (resumable)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    corpus = json.loads(SRC.read_text(encoding="utf-8"))
    cache: dict = {}
    if args.scan_cache and Path(args.scan_cache).exists():
        for line in Path(args.scan_cache).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["repo"]] = d
    cache_fh = open(args.scan_cache, "a", encoding="utf-8") if args.scan_cache else None

    verdicts = {}
    if WINNABILITY.exists():
        verdicts = json.loads(WINNABILITY.read_text(encoding="utf-8")).get("repos", {})
        print(f"applying oracle verdicts from {WINNABILITY.relative_to(PROJECT_ROOT)} ({len(verdicts)} repos)")

    out = {k: v for k, v in corpus.items() if k != "splits"}
    out["source"] = SRC.name
    out["pilot_generated_at"] = datetime.now(timezone.utc).isoformat()
    out["pilot_filter"] = __doc__.split("Usage:")[0].strip()
    out["splits"] = {"test": corpus["splits"]["test"]}
    out["pilot_excluded"] = {}
    out["pilot_scan"] = {}
    # Every repo that passes the static filters, regardless of oracle verdict:
    # what stage2g_winnability.py --all judges (so a repo excluded by an old
    # verdict is re-judged after an image / criterion change).
    out["oracle_candidates"] = {}
    for split in ("train", "val"):
        kept, excluded, candidates = [], [], []
        for entry in corpus["splits"][split]:
            s = cache.get(entry["repo"])
            if s is None:
                s = _scan(entry)
                if cache_fh:
                    cache_fh.write(json.dumps(s) + "\n")
                    cache_fh.flush()
            r = _reason(s)
            e = dict(entry)
            if not e.get("sha") and "clone_dir" in s:
                e["sha"] = s["clone_dir"].rsplit("_", 1)[-1]  # short sha from clone dir (B7: full SHA pending)
            if r is None:
                candidates.append(e)
                out["pilot_scan"][entry["repo"]] = {k: s[k] for k in ("conflicts", "manifest_files", "required")}
                r = _unwinnable(e, verdicts)
            if r is None:
                kept.append(e)
            else:
                excluded.append({"repo": entry["repo"], "reason": r})
        out["splits"][split] = kept
        out["pilot_excluded"][split] = excluded
        out["oracle_candidates"][split] = candidates
        print(f"{split}: kept {len(kept)} / {len(corpus['splits'][split])}")
        reasons: dict = {}
        for x in excluded:
            key = x["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"  excluded by reason: {reasons}")
    moved = _move_to_val(out["splits"])
    out["pilot_val_moves"] = moved
    print(f"moved train -> val: {moved}")
    print(f"final: train {len(out['splits']['train'])}, val {len(out['splits']['val'])}")
    DST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {DST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
