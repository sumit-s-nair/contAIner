"""
scripts/build_corpus.py
=======================
CLI entry-point to build and persist the versioned repo corpus manifest.

Usage
-----
    python scripts/build_corpus.py --version 1 --seed 42 --per-category 100

The script reads GITHUB_TOKEN from the environment (or .env file).
It will raise an error if the target manifest version already exists —
increment --version to mint a fresh manifest without overwriting history.

Output
------
    datasets/repo_corpus/corpus_manifest_v{version}.json

The file includes:
  - Version + seed (reproducibility)
  - Per-split repo lists (train / val / test) with category tags
  - Per-split stats: total repos + per-category breakdown
  - topic_search_conflict_count (how many known_conflict repos came from
    GitHub topic search vs. the curated fallback list)

The test split is written to the manifest but is NEVER loaded by
PlannerEnv during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Load .env before importing corpus (GITHUB_TOKEN may be there)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; token can be set in shell environment

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl_env.corpus import build_corpus, _print_stats, load_corpus_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a versioned repo corpus manifest for System 2 RL training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", "-v",
        type=int,
        default=1,
        help="Manifest version number (default: 1). Increment to avoid overwriting.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for stratified split (default: 42).",
    )
    parser.add_argument(
        "--per-category", "-n",
        type=int,
        default=100,
        dest="per_category",
        help="Max repos to fetch per category before splitting (default: 100).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="datasets/repo_corpus",
        dest="output_dir",
        help="Output directory for the manifest (default: datasets/repo_corpus).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub API token. If not set, reads GITHUB_TOKEN from environment.",
    )

    args = parser.parse_args()

    print(f"Building corpus manifest v{args.version} (seed={args.seed}, "
          f"per_category={args.per_category}) ...")

    try:
        out_path = build_corpus(
            version=args.version,
            seed=args.seed,
            per_category=args.per_category,
            output_dir=args.output_dir,
            github_token=args.token,
        )
        print(f"\n✓ Corpus manifest written: {out_path}")

        # Re-print stats from the written file for confirmation
        manifest = load_corpus_manifest(str(out_path))
        _print_stats(manifest)

    except FileExistsError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as e:
        print(f"\n✗ Environment error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
