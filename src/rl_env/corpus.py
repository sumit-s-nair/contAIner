"""
src/rl_env/corpus.py
====================
GitHub API corpus builder for the System 2 planner RL environment.

Fetches a diverse set of real repositories, filters them for safety/size,
stratifies by category, and splits 70/15/15 by repo (never by scenario).

Categories
----------
manifest_present:
    Python repos with requirements.txt or pyproject.toml, and/or Node repos
    with package.json.  The main scan path.

manifest_less:
    Python repos *without* any manifest file.  Exercises the import_scan
    fallback path.

known_conflict:
    Repos that exhibit dependency conflicts.  The ``dependency-conflict``
    GitHub topic is searched first; if it returns fewer than
    ``MIN_CONFLICT_REPOS`` results, the corpus is supplemented with a
    curated list of known-conflict repos (maintained in ``CURATED_CONFLICT_REPOS``
    below).

Filter criteria (applied before split)
---------------------------------------
- 50 KB ≤ repo size ≤ 50 MB
- stars ≥ 10
- not archived
- not a fork
- default branch in {main, master}
- OSI-approved license (spdx_id not in {null, "NOASSERTION", "NONE"})

Split
------
Stratified 70/15/15 by repo.  Within each category the repos are shuffled
with the fixed seed, then allocated proportionally.

Persistence
------------
Written to ``datasets/repo_corpus/corpus_manifest_v{VERSION}.json``.
Re-running always mints a new version — the file is never overwritten.
The manifest includes: version, seed, generated_at, per-split repo lists
with category tags, and per-split stats (total, per-category counts).

The test split is tracked in the manifest but is NEVER loaded by PlannerEnv
during training.  It is accessible only via the manifest file for offline
evaluation.

Usage
------
    python scripts/build_corpus.py --version 1 --seed 42 --per-category 100
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
MIN_SIZE_KB  = 50
MAX_SIZE_KB  = 50 * 1024  # 50 MB
MIN_STARS    = 10
MIN_CONFLICT_REPOS = 20   # if topic search returns fewer, use curated list

# OSI-approved licenses we accept (non-exhaustive; rejects null/NOASSERTION)
_ACCEPTED_LICENSES = {
    "MIT", "Apache-2.0", "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0",
    "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0", "AGPL-3.0", "ISC",
    "Unlicense", "CC0-1.0", "EUPL-1.1", "EUPL-1.2",
}

# Curated fallback for known-conflict repos (supplement when topic search is sparse)
CURATED_CONFLICT_REPOS: List[Dict[str, str]] = [
    # repos with deliberately pinned conflicting requirements or known CVE-related pins
    {"full_name": "pypa/pip",             "category": "known_conflict"},
    {"full_name": "pypa/setuptools",      "category": "known_conflict"},
    {"full_name": "pallets/flask",        "category": "known_conflict"},
    {"full_name": "psf/requests",         "category": "known_conflict"},
    {"full_name": "encode/django-rest-framework", "category": "known_conflict"},
    {"full_name": "django/django",        "category": "known_conflict"},
    {"full_name": "sqlalchemy/sqlalchemy","category": "known_conflict"},
    {"full_name": "boto/boto3",           "category": "known_conflict"},
    {"full_name": "celery/celery",        "category": "known_conflict"},
    {"full_name": "apache/airflow",       "category": "known_conflict"},
    {"full_name": "dask/dask",            "category": "known_conflict"},
    {"full_name": "pandas-dev/pandas",    "category": "known_conflict"},
    {"full_name": "numpy/numpy",          "category": "known_conflict"},
    {"full_name": "scipy/scipy",          "category": "known_conflict"},
    {"full_name": "scikit-learn/scikit-learn", "category": "known_conflict"},
    {"full_name": "pytorch/pytorch",      "category": "known_conflict"},
    {"full_name": "huggingface/transformers", "category": "known_conflict"},
    {"full_name": "keras-team/keras",     "category": "known_conflict"},
    {"full_name": "alembic/alembic",      "category": "known_conflict"},
    {"full_name": "ansible/ansible",      "category": "known_conflict"},
]


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _github_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return s


def _search_repos(
    session: requests.Session,
    query: str,
    per_page: int = 100,
    max_results: int = 300,
) -> List[Dict]:
    """
    Paginate the GitHub search API for *query*.

    Returns raw items (dicts), up to *max_results*.
    Respects rate-limit headers with exponential back-off.
    """
    results = []
    page = 1
    while len(results) < max_results:
        resp = session.get(
            f"{GITHUB_API}/search/repositories",
            params={
                "q":        query,
                "per_page": per_page,
                "page":     page,
                "sort":     "stars",
                "order":    "desc",
            },
            timeout=30,
        )

        # Rate-limit back-off
        if resp.status_code == 403:
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset_ts - time.time(), 1)
            print(f"[corpus] Rate-limited — waiting {wait:.0f}s ...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data  = resp.json()
        items = data.get("items", [])
        results.extend(items)

        if len(items) < per_page or len(results) >= data.get("total_count", 0):
            break
        page += 1

    return results[:max_results]


def _passes_filter(item: Dict) -> bool:
    """Return True if the repo passes all safety/size/provenance filters."""
    size_kb = item.get("size", 0)
    if not (MIN_SIZE_KB <= size_kb <= MAX_SIZE_KB):
        return False
    if item.get("stargazers_count", 0) < MIN_STARS:
        return False
    if item.get("archived", False):
        return False
    if item.get("fork", False):
        return False
    if item.get("default_branch", "") not in {"main", "master"}:
        return False
    lic = item.get("license") or {}
    spdx = lic.get("spdx_id", "NONE") or "NONE"
    if spdx.upper() in {"NONE", "NOASSERTION", "OTHER"}:
        return False
    if spdx not in _ACCEPTED_LICENSES:
        return False
        
    # Security Keyword Denylist
    import re
    import sys
    malware_keywords = r"\b(ransomware|malware|virus|trojan|keylogger|rootkit|exploit|backdoor|rat|botnet|ddos|phishing|stealer|c2)\b"
    pattern = re.compile(malware_keywords, re.IGNORECASE)
    
    text_to_check = f"{item.get('name', '')} {item.get('description', '')} {' '.join(item.get('topics', []))}"
    match = pattern.search(text_to_check)
    if match:
        print(f"[Safety] Blocked repo '{item.get('full_name', '')}' due to keyword: '{match.group(1)}'", file=sys.stderr)
        return False

    return True


def _item_to_entry(item: Dict, category: str) -> Dict:
    """Convert a GitHub API search item to a corpus entry dict."""
    return {
        "repo":           item["full_name"],
        "category":       category,
        "stars":          item.get("stargazers_count", 0),
        "size_kb":        item.get("size", 0),
        "default_branch": item.get("default_branch", "main"),
        "license":        (item.get("license") or {}).get("spdx_id", ""),
        "clone_url":      item.get("clone_url", ""),
        "sha":            "",   # filled by RepoLoader on first clone
    }


# ---------------------------------------------------------------------------
# Per-category fetch logic
# ---------------------------------------------------------------------------

def _fetch_manifest_present(session: requests.Session, per_category: int) -> List[Dict]:
    """Fetch Python + Node repos with manifest files."""
    entries = []
    queries = [
        # Standard popular Python repos (excluding topics typical of manifest_less)
        "language:python stars:>10 -topic:scripts -topic:tutorial -topic:examples -topic:beginner",
        # Standard popular Node repos
        "language:javascript stars:>10 -topic:scripts -topic:tutorial -topic:examples -topic:beginner",
    ]
    seen = set()
    fetched = 0
    passed_filter = 0
    keyword_blocked = 0
    for q in queries:
        for item in _search_repos(session, q, max_results=per_category * 2):
            fetched += 1
            name = item.get("full_name", "")
            if name in seen:
                continue
            
            # Re-implementing a quick keyword check just to count it separately for the funnel
            import re
            malware_keywords = r"\b(ransomware|malware|virus|trojan|keylogger|rootkit|exploit|backdoor|rat|botnet|ddos|phishing|stealer|c2)\b"
            pattern = re.compile(malware_keywords, re.IGNORECASE)
            text_to_check = f"{item.get('name', '')} {item.get('description', '')} {' '.join(item.get('topics', []))}"
            if pattern.search(text_to_check):
                keyword_blocked += 1
                
            if not _passes_filter(item):
                continue
            
            seen.add(name)
            passed_filter += 1
            entries.append(_item_to_entry(item, "manifest_present"))
            if len(entries) >= per_category:
                break
        if len(entries) >= per_category:
            break
            
    print(f"\n[Funnel: manifest_present] API Fetched: {fetched} -> Metadata Blocked: {fetched - passed_filter - keyword_blocked} -> Keyword Blocked: {keyword_blocked} -> Filter Passed: {passed_filter}")
    return entries[:per_category]


def _fetch_manifest_less(session: requests.Session, per_category: int) -> List[Dict]:
    """Fetch Python repos that are *unlikely* to have a manifest."""
    # Search for Python repos explicitly without requirements.txt or pyproject.toml
    # GitHub's code-search can't negate file presence, so we search and post-filter.
    # We look for small/informal Python repos (script collections, tutorials, etc.)
    queries = [
        "language:python topic:scripts stars:>10",
        "language:python topic:tutorial stars:>10",
        "language:python topic:beginner stars:>10",
        "language:python topic:examples stars:>10",
    ]
    entries = []
    seen = set()
    for q in queries:
        for item in _search_repos(session, q, max_results=per_category * 3):
            name = item.get("full_name", "")
            if name in seen or not _passes_filter(item):
                continue
            seen.add(name)
            # Post-filter: the category assignment is tentative; the scanner
            # will determine actual category at clone time.
            entries.append(_item_to_entry(item, "manifest_less"))
            if len(entries) >= per_category:
                break
        if len(entries) >= per_category:
            break
    return entries[:per_category]


def _fetch_known_conflict(
    session: requests.Session,
    per_category: int,
) -> Tuple[List[Dict], int]:
    """
    Fetch repos with known dependency conflicts.

    Tries the ``dependency-conflict`` GitHub topic first.  Reports the count.
    If fewer than ``MIN_CONFLICT_REPOS`` are found, supplements with
    ``CURATED_CONFLICT_REPOS``.

    Returns (entries, topic_count) where topic_count is the raw count
    from the topic search before curation supplementation.
    """
    topic_items = _search_repos(
        session,
        "topic:dependency-conflict",
        max_results=per_category * 2,
    )
    topic_entries = [
        _item_to_entry(item, "known_conflict")
        for item in topic_items
        if _passes_filter(item)
    ]
    topic_count = len(topic_entries)

    if topic_count < MIN_CONFLICT_REPOS:
        print(
            f"[corpus] WARNING: 'dependency-conflict' topic returned only "
            f"{topic_count} repos (< MIN_CONFLICT_REPOS={MIN_CONFLICT_REPOS}). "
            f"Supplementing with {len(CURATED_CONFLICT_REPOS)} curated repos."
        )
        # Merge, dedup by repo name
        seen = {e["repo"] for e in topic_entries}
        for curated in CURATED_CONFLICT_REPOS:
            if curated["full_name"] not in seen:
                topic_entries.append({
                    "repo":           curated["full_name"],
                    "category":       "known_conflict",
                    "stars":          -1,   # unknown; filter not applicable to curated
                    "size_kb":        -1,
                    "default_branch": "main",
                    "license":        "",
                    "clone_url":      f"https://github.com/{curated['full_name']}.git",
                    "sha":            "",
                    "curated":        True,
                })
                seen.add(curated["full_name"])

    return topic_entries[:per_category], topic_count


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(
    entries_by_category: Dict[str, List[Dict]],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Dict[str, List[Dict]]:
    """
    Split repos 70/15/15 within each category, then concatenate.

    The split is BY REPO (not by scenario or episode).  A repo's category
    assignment is fixed at split time.  The test set never leaks into train/val.
    """
    rng = random.Random(seed)
    train, val, test = [], [], []

    for category, entries in entries_by_category.items():
        shuffled = list(entries)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        # Remainder goes to test (slightly more than 15% when rounding)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def _split_stats(split: List[Dict]) -> Dict:
    """Compute per-category counts for a split."""
    counts: Dict[str, int] = {}
    for entry in split:
        cat = entry.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return {"total": len(split), "by_category": counts}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_corpus(
    version:      int  = 1,
    seed:         int  = 42,
    per_category: int  = 100,
    output_dir:   str  = "datasets/repo_corpus",
    github_token: Optional[str] = None,
) -> Path:
    """
    Fetch, filter, split, and persist a versioned corpus manifest.

    Parameters
    ----------
    version:
        Manifest version number.  A file named ``corpus_manifest_v{version}.json``
        is written.  The function raises ``FileExistsError`` if the file already
        exists — increment the version to mint a new manifest.

    seed:
        Fixed random seed for the stratified split.  Stored in the manifest
        for reproducibility.

    per_category:
        Maximum number of repos to fetch per category before splitting.

    output_dir:
        Directory to write the manifest.  Created if absent.

    github_token:
        GitHub personal access token.  If ``None``, reads ``GITHUB_TOKEN``
        from the environment.  Required (no unauthenticated fallback).

    Returns
    -------
    Path
        Absolute path to the written manifest file.
    """
    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is required for the corpus pipeline. "
            "Set it in your .env file or pass it as --token."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"corpus_manifest_v{version}.json"

    if out_path.exists():
        raise FileExistsError(
            f"Manifest {out_path} already exists.  "
            f"Use --version {version + 1} to mint a new version."
        )

    session = _github_session(token)

    print(f"[corpus] Fetching manifest_present repos (target: {per_category}) ...")
    manifest_present = _fetch_manifest_present(session, per_category)
    print(f"[corpus]   → {len(manifest_present)} repos after filter")

    print(f"[corpus] Fetching manifest_less repos (target: {per_category}) ...")
    manifest_less = _fetch_manifest_less(session, per_category)
    print(f"[corpus]   → {len(manifest_less)} repos after filter")

    print(f"[corpus] Fetching known_conflict repos (target: {per_category}) ...")
    known_conflict, topic_count = _fetch_known_conflict(session, per_category)
    print(
        f"[corpus]   → {len(known_conflict)} repos total "
        f"({topic_count} from topic search, rest curated)"
    )

    # --- MALWARE SCAN & MANIFEST VERIFY (Defense in Depth) ---
    print("\n[corpus] Running malware scan & manifest verification on candidate repos...")
    from .repo_loader import RepoLoader, scan_repo
    import subprocess
    import shutil
    import sys
    
    loader = RepoLoader()
    malware_blocked = 0
    manifest_blocked = 0
    clone_failed = 0
    
    def scan_for_malware(repo_entry: Dict) -> bool:
        nonlocal malware_blocked, manifest_blocked, clone_failed
        # OPTION A: Dedicated quarantine directory that gets wiped after every scan.
        # This guarantees that no failed/pending repos sit in the trusted repo_clones cache.
        safe_name = repo_entry["repo"].replace("/", "_")
        quarantine_dir = Path("cache/scan_quarantine")
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        local_path = quarantine_dir / safe_name
        
        try:
            if local_path.exists():
                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(local_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Clone with depth 1, enforcing longpaths for huge repos
            clone_url = repo_entry.get("clone_url") or f"https://github.com/{repo_entry['repo']}.git"
            subprocess.run(
                ["git", "clone", "-c", "core.longpaths=true", "--depth", "1", clone_url, str(local_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            
            # Run Defender Scan
            defender_path = r"C:\Program Files\Windows Defender\MpCmdRun.exe"
            cmd = [defender_path, "-Scan", "-ScanType", "3", "-File", str(local_path.absolute())]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if result.returncode != 0:
                print(f"[Safety] MALWARE DETECTED in '{repo_entry['repo']}'. Blocked.", file=sys.stderr)
                malware_blocked += 1
                return False
                
            # For manifest_present category, enforce manifest exists
            if repo_entry["category"] == "manifest_present":
                manifest = scan_repo(local_path)
                if not manifest.ecosystems:
                    print(f"[Manifest] No manifest found in '{repo_entry['repo']}'. Blocked.", file=sys.stderr)
                    manifest_blocked += 1
                    return False
                
            return True
        except subprocess.CalledProcessError:
            print(f"[Clone] Error cloning '{repo_entry['repo']}'. Blocked.", file=sys.stderr)
            clone_failed += 1
            return False
        except Exception as e:
            print(f"[Safety] Error scanning '{repo_entry['repo']}': {e}", file=sys.stderr)
            return False
        finally:
            if local_path.exists():
                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(local_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    manifest_present_initial = len(manifest_present)
    manifest_present = [r for r in manifest_present if scan_for_malware(r)]
    
    # We only care about the funnel for manifest_present, but apply to all
    manifest_less_initial = len(manifest_less)
    manifest_less = [r for r in manifest_less if scan_for_malware(r)]
    
    known_conflict_initial = len(known_conflict)
    known_conflict = [r for r in known_conflict if scan_for_malware(r)]
    
    print(f"[corpus] Scan & Verify complete.")
    print(f"[corpus] Rejections -> Malware: {malware_blocked}, No-Manifest: {manifest_blocked}, Clone Error: {clone_failed}")

    entries_by_category = {
        "manifest_present": manifest_present,
        "manifest_less":    manifest_less,
        "known_conflict":   known_conflict,
    }

    splits = _stratified_split(entries_by_category, seed=seed)

    manifest = {
        "version":      version,
        "seed":         seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_category": per_category,
        "splits":       splits,
        "stats": {
            "train": _split_stats(splits["train"]),
            "val":   _split_stats(splits["val"]),
            "test":  _split_stats(splits["test"]),
        },
        "topic_search_conflict_count": topic_count,
        "curated_conflict_repos":      len(CURATED_CONFLICT_REPOS),
    }

    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[corpus] Manifest written to {out_path}")
    _print_stats(manifest)

    return out_path.resolve()


def load_corpus_manifest(path: str) -> Dict:
    """Load an existing corpus manifest."""
    p = Path(path)
    if "COMPROMISED" in p.name.upper():
        raise ValueError(f"CRITICAL SAFETY ERROR: Attempted to load known-compromised manifest: {p.name}")
    with p.open() as f:
        return json.load(f)


def _print_stats(manifest: Dict) -> None:
    """Pretty-print split stats to stdout."""
    print("\n=== Corpus split composition ===")
    header = f"{'Split':<8} {'Total':>6}  {'manifest_present':>18}  "
    header += f"{'manifest_less':>13}  {'known_conflict':>14}"
    print(header)
    print("-" * len(header))
    for split_name in ("train", "val", "test"):
        stats = manifest["stats"][split_name]
        by_cat = stats.get("by_category", {})
        print(
            f"{split_name:<8} {stats['total']:>6}  "
            f"{by_cat.get('manifest_present', 0):>18}  "
            f"{by_cat.get('manifest_less', 0):>13}  "
            f"{by_cat.get('known_conflict', 0):>14}"
        )
    print()
