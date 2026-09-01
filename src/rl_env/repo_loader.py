"""
src/rl_env/repo_loader.py
=========================
RepoLoader — clones a corpus repo, runs scan_repo(), caches the result.

Design
------
* One ``RepoLoader`` instance can be shared across episodes (cache is on disk).
* Each repo is cloned once into ``cache/repo_clones/{owner}_{name}_{sha[:8]}/``.
* The SHA is resolved at clone time from the default branch HEAD.
* If the cached directory already exists, the clone step is skipped.
* ``scan_repo()`` is called on the cached clone and the ``RepoManifest`` is
  returned along with the local clone path (passed to ``DryRunExecutor`` as
  ``sandbox_root`` for real-execution episodes).

State persistence note
-----------------------
The clone directory is the "container" for real-execution episodes.  All
steps in one episode are executed in the same directory (passed as
``sandbox_root`` to ``DryRunExecutor``), so step N+1 can see the effects
of step N (e.g. installed packages).  The directory is NOT cleaned between
steps within one episode — only between episodes (by resetting to the
original clone).

Between episodes the loader can optionally reset the working tree to the
HEAD state (via ``git checkout -- .`` + ``git clean -fd``) to ensure a
clean starting state for the next episode.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.repo_scan.models import RepoManifest
from src.repo_scan.scanner import scan_repo


DEFAULT_CACHE_DIR = "cache/repo_clones"


class RepoLoader:
    """
    Clone a GitHub repo (once) and return its ``RepoManifest`` + local path.

    Parameters
    ----------
    cache_dir:
        Base directory for cached clones.  Defaults to ``cache/repo_clones/``.
    reset_between_episodes:
        If True, ``git checkout -- .`` and ``git clean -fd`` are run at the
        start of each ``load()`` call to reset any episode-time mutations.
        Default True — ensures a clean working tree for each new episode.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        reset_between_episodes: bool = True,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._reset = reset_between_episodes

    # ------------------------------------------------------------------ public

    def load(self, repo_entry: Dict) -> Tuple[RepoManifest, Path]:
        """
        Ensure the repo is cloned, optionally reset it, scan it, return results.

        Parameters
        ----------
        repo_entry:
            A corpus entry dict with at minimum ``"repo"`` (e.g. ``"owner/name"``)
            and optionally ``"clone_url"`` and ``"default_branch"``.

        Returns
        -------
        (RepoManifest, local_path)
            ``local_path`` is the absolute path to the cloned repo directory.
        """
        repo_name    = repo_entry["repo"]
        clone_url    = repo_entry.get("clone_url") or f"https://github.com/{repo_name}.git"
        default_branch = repo_entry.get("default_branch", "main")

        # Resolve local cache path (use repo name, sha if known)
        sha_tag = repo_entry.get("sha", "")[:8] or "latest"
        safe_name = repo_name.replace("/", "_")
        local_path = self._cache_dir / f"{safe_name}_{sha_tag}"

        # --- Clone (if not cached) ----------------------------------------
        if not local_path.exists():
            self._clone(clone_url, local_path, default_branch)
            # Record the actual HEAD SHA
            actual_sha = self._get_head_sha(local_path)
            # Rename to include actual SHA if we only had "latest"
            if sha_tag == "latest" and actual_sha:
                new_path = self._cache_dir / f"{safe_name}_{actual_sha[:8]}"
                if not new_path.exists():
                    local_path.rename(new_path)
                local_path = new_path
                repo_entry["sha"] = actual_sha  # update in-place for logging

        # --- Reset working tree between episodes ---------------------------
        if self._reset and local_path.exists():
            self._git_reset(local_path)

        # --- Scan ----------------------------------------------------------
        manifest = scan_repo(str(local_path))
        return manifest, local_path.resolve()

    # --------------------------------------------------------------- private

    def _clone(self, clone_url: str, dest: Path, branch: str) -> None:
        """Shallow-clone the default branch into *dest*."""
        print(f"[RepoLoader] Cloning {clone_url} → {dest} ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch,
             clone_url, str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Some repos use "main" vs "master" — try the other branch
            alt_branch = "master" if branch == "main" else "main"
            print(
                f"[RepoLoader] Clone failed (branch={branch!r}), "
                f"retrying with branch={alt_branch!r} ..."
            )
            result2 = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", alt_branch,
                 clone_url, str(dest)],
                capture_output=True,
                text=True,
            )
            if result2.returncode != 0:
                raise RuntimeError(
                    f"Failed to clone {clone_url}:\n"
                    f"  {branch}: {result.stderr.strip()}\n"
                    f"  {alt_branch}: {result2.stderr.strip()}"
                )

    def _get_head_sha(self, repo_path: Path) -> Optional[str]:
        """Return the short HEAD SHA for the cloned repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(repo_path),
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _git_reset(self, repo_path: Path) -> None:
        """
        Reset the working tree to HEAD and remove untracked files.

        This ensures each episode starts from a clean state, even if
        a previous episode installed packages or created files.
        """
        try:
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=str(repo_path),
                capture_output=True,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=str(repo_path),
                capture_output=True,
            )
        except Exception as exc:
            print(f"[RepoLoader] WARNING: git reset failed for {repo_path}: {exc}")
