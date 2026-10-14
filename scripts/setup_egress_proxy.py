#!/usr/bin/env python3
"""
scripts/setup_egress_proxy.py
=============================
Stage 2C: Sandbox egress setup via allowlisting Squid proxy.
Sets up:
- Docker internal network: contai_sandbox (--internal, no internet routing)
- Container: contai_egress_proxy running ubuntu/squid pinned by digest
- Dual network attachment: contai_sandbox + default bridge
- Mounts docker/egress_proxy/squid.conf and logs/egress_proxy
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Pinned image digest for ubuntu/squid:
SQUID_IMAGE_DIGEST = "sha256:6a097f68bae708cedbabd6188d68c7e2e7a38cedd05a176e1cc0ba29e3bbe029"
SQUID_IMAGE = f"ubuntu/squid@{SQUID_IMAGE_DIGEST}"

NETWORK_NAME = "contai_sandbox"
CONTAINER_NAME = "contai_egress_proxy"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_egress_proxy() -> None:
    # 1. Ensure internal network contai_sandbox exists
    r_net_inspect = subprocess.run(
        ["docker", "network", "inspect", NETWORK_NAME],
        capture_output=True,
        text=True,
    )
    if r_net_inspect.returncode != 0:
        print(f"Creating internal docker network {NETWORK_NAME!r}...")
        r_create = subprocess.run(
            ["docker", "network", "create", "--internal", NETWORK_NAME],
            capture_output=True,
            text=True,
        )
        if r_create.returncode != 0:
            raise RuntimeError(f"Failed to create network {NETWORK_NAME}: {r_create.stderr.strip()}")
    else:
        net_data = json.loads(r_net_inspect.stdout)[0]
        if not net_data.get("Internal"):
            raise RuntimeError(
                f"Existing network {NETWORK_NAME!r} is not an internal network (Internal: {net_data.get('Internal')}). "
                f"Remove it or recreate with --internal."
            )

    # 2. Prepare paths
    conf_path = (PROJECT_ROOT / "docker" / "egress_proxy" / "squid.conf").resolve()
    if not conf_path.exists():
        raise FileNotFoundError(f"Squid configuration not found at {conf_path}")

    log_dir = (PROJECT_ROOT / "logs" / "egress_proxy").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Check container contai_egress_proxy
    r_ps = subprocess.run(
        ["docker", "inspect", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    needs_start = True
    if r_ps.returncode == 0:
        c_info = json.loads(r_ps.stdout)[0]
        is_running = c_info.get("State", {}).get("Running", False)
        if is_running:
            needs_start = False
        else:
            print(f"Container {CONTAINER_NAME!r} exists but is stopped. Recreating...")
            subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

    if needs_start:
        print(f"Starting proxy container {CONTAINER_NAME!r} from {SQUID_IMAGE}...")
        cmd_run = [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--network", NETWORK_NAME,
            "--restart", "unless-stopped",
            "-v", f"{conf_path}:/etc/squid/squid.conf:ro",
            "-v", f"{log_dir}:/var/log/squid",
            SQUID_IMAGE,
        ]
        r_run = subprocess.run(cmd_run, capture_output=True, text=True)
        if r_run.returncode != 0:
            raise RuntimeError(f"Failed to start {CONTAINER_NAME}: {r_run.stderr.strip()}")

    # 4. Ensure attached to default bridge
    r_proxy = subprocess.run(
        ["docker", "inspect", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if r_proxy.returncode != 0:
        raise RuntimeError(f"Could not inspect {CONTAINER_NAME}: {r_proxy.stderr.strip()}")
    proxy_info = json.loads(r_proxy.stdout)[0]
    existing_nets = proxy_info.get("NetworkSettings", {}).get("Networks", {})
    if "bridge" not in existing_nets:
        print(f"Connecting {CONTAINER_NAME!r} to default bridge network...")
        r_conn = subprocess.run(
            ["docker", "network", "connect", "bridge", CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        if r_conn.returncode != 0:
            raise RuntimeError(f"Failed to connect {CONTAINER_NAME} to bridge: {r_conn.stderr.strip()}")

    # 5. Print status
    r_net_inspect2 = subprocess.run(
        ["docker", "network", "inspect", NETWORK_NAME],
        capture_output=True,
        text=True,
    )
    net_data2 = json.loads(r_net_inspect2.stdout)[0]
    internal_flag = net_data2.get("Internal")

    r_proxy2 = subprocess.run(
        ["docker", "inspect", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    proxy_info2 = json.loads(r_proxy2.stdout)[0]
    proxy_networks = list(proxy_info2.get("NetworkSettings", {}).get("Networks", {}).keys())

    print(f"Network {NETWORK_NAME} Internal flag: {internal_flag}")
    print(f"Proxy {CONTAINER_NAME} networks: {proxy_networks}")


if __name__ == "__main__":
    setup_egress_proxy()
