"""Capture the --help text of the tools System 3 drives, from the sandbox image.

    python -m scripts.snapshot_cli_help            # docker (contai-sandbox-python:3.11)
    python -m scripts.snapshot_cli_help --local    # this interpreter (tests / no Docker)

Writes datasets/system3-slot/docs/{pip_install,venv,pytest}.help.txt and
meta.json (tool versions, source).  These are System 3's documentation
source (src/system3_command_generation/slot/docs.py): version-exact for the
pip that executes the commands.  pytest is pinned to the version the stub
installs for CHECK (8.3.5).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.rl_env.docker_executor import PYTHON_SANDBOX_IMAGE

OUT = Path("datasets/system3-slot/docs")
PYTEST_PIN = "pytest==8.3.5"

_SCRIPT = f"""
set -e
python -m pip install -q --disable-pip-version-check {PYTEST_PIN} >/dev/null 2>&1 || true
echo '=====pip_install'; python -m pip install --help
echo '=====venv'; python -m venv --help
echo '=====pytest'; python -m pytest --help
echo '=====versions'; python -m pip --version; python --version; python -m pytest --version 2>&1 | head -1
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use this interpreter instead of the sandbox image")
    ap.add_argument("--image", default=PYTHON_SANDBOX_IMAGE)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    if a.local:
        cmd = ["bash", "-c", _SCRIPT.replace("python ", f"{sys.executable} ")]
        source = f"local:{sys.executable}"
    else:
        cmd = ["docker", "run", "--rm", a.image, "sh", "-c", _SCRIPT]
        source = f"docker:{a.image}"
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        sys.exit(f"failed ({out.returncode}): {out.stderr[-2000:]}")

    parts, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("====="):
            cur = line[5:]
            parts[cur] = []
        elif cur:
            parts[cur].append(line)
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("pip_install", "venv", "pytest"):
        if not parts.get(name):
            sys.exit(f"missing section {name}")
        (dest / f"{name}.help.txt").write_text("\n".join(parts[name]) + "\n", encoding="utf-8")
    meta = {"source": source, "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": parts.get("versions", [])}
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
