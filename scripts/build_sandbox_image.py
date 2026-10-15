#!/usr/bin/env python3
"""
scripts/build_sandbox_image.py
==============================
Stage 2G: build the Python sandbox image used by DockerEpisodeExecutor
(``contai-sandbox-python:3.11`` = python:3.11-slim + git + build-essential + pkg-config, see
docker/sandbox_python/Dockerfile) and check that git and the toolchain work inside it.

Run once (and again after editing the Dockerfile), before any real-Docker
episode.  Snapshot images committed during GRPO derive from this image.

Usage:  python scripts/build_sandbox_image.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_env.docker_executor import PYTHON_SANDBOX_IMAGE  # noqa: E402

CONTEXT = PROJECT_ROOT / "docker" / "sandbox_python"


def main() -> int:
    print(f"Building {PYTHON_SANDBOX_IMAGE} from {CONTEXT.relative_to(PROJECT_ROOT)} ...", flush=True)
    r = subprocess.run(["docker", "build", "-t", PYTHON_SANDBOX_IMAGE, str(CONTEXT)])
    if r.returncode != 0:
        print("docker build failed", file=sys.stderr)
        return r.returncode
    chk = subprocess.run(
        ["docker", "run", "--rm", PYTHON_SANDBOX_IMAGE, "sh", "-c",
         "git --version && python --version && git config --system --get-all safe.directory"
         " && gcc --version | head -1 && g++ --version | head -1 && pkg-config --version"],
        capture_output=True, text=True,
    )
    print(chk.stdout.strip())
    if chk.returncode != 0:
        print(chk.stderr.strip(), file=sys.stderr)
        return chk.returncode
    print(f"OK: {PYTHON_SANDBOX_IMAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
