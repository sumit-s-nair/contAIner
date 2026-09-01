#!/usr/bin/env bash
# scripts/setup_docker_network.sh
# =================================
# One-time setup: provision the rl_allowlist Docker network for training.
#
# What this creates
# -----------------
# A Docker bridge network named "rl_allowlist" with iptables OUTPUT rules
# that restrict outbound TCP to a curated set of package-registry IPs.
# All other outbound traffic from episode containers is dropped.
#
# Requirements
# ------------
# - Docker Engine 20.10+ (Linux host or Docker Desktop with iptables)
# - sudo / root (iptables manipulation requires root)
# - The iptables command (pre-installed on most Linux distros)
#
# Usage
# -----
#   chmod +x scripts/setup_docker_network.sh
#   sudo scripts/setup_docker_network.sh
#
# To verify the network exists after setup:
#   docker network inspect rl_allowlist
#
# To tear down:
#   docker network rm rl_allowlist
#   sudo iptables -D FORWARD -i rl_allowlist -j rl_allowlist_filter 2>/dev/null
#   sudo iptables -F rl_allowlist_filter 2>/dev/null
#   sudo iptables -X rl_allowlist_filter 2>/dev/null
#
# IMPORTANT: Verified restriction markers
# ----------------------------------------
# The script adds a custom iptables chain "rl_allowlist_filter" and injects
# a custom label into the network's options that DockerEpisodeExecutor checks
# at start_episode() time. If the label is missing the executor raises
# NetworkNotConfiguredError rather than running with an unconfigured network.
# This ensures misconfigured or manually-created unlabelled networks are
# rejected early.
#
# Label key:   com.contai.rl.network.verified
# Label value: allowlist-v1

set -euo pipefail

NETWORK_NAME="${RL_NETWORK_NAME:-rl_allowlist}"
LABEL_KEY="com.contai.rl.network.verified"
LABEL_VALUE="allowlist-v1"

# Package registry IPs to allow (resolved at setup time).
# These are the same hosts as NetworkConfig.allowed_registry_hosts.
ALLOWED_HOSTS=(
    "pypi.org"
    "files.pythonhosted.org"
    "registry.npmjs.org"
    "registry.yarnpkg.com"
    "rubygems.org"
    "crates.io"
    "static.crates.io"
    "proxy.golang.org"
    "sum.golang.org"
)

echo "=== rl_allowlist Docker network setup ==="
echo ""

# --- Check prerequisites ---------------------------------------------------
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found on PATH."
    exit 1
fi

if ! command -v iptables &>/dev/null; then
    echo "ERROR: iptables not found. Install iptables first."
    exit 1
fi

# --- Remove stale network if it exists ------------------------------------
if docker network inspect "$NETWORK_NAME" &>/dev/null; then
    echo "Network '$NETWORK_NAME' already exists. Removing to recreate..."
    docker network rm "$NETWORK_NAME"
fi

# --- Resolve registry IPs --------------------------------------------------
echo "Resolving registry hostnames..."
ALLOWED_IPS=()
for host in "${ALLOWED_HOSTS[@]}"; do
    ip=$(getent hosts "$host" 2>/dev/null | awk '{print $1; exit}' || \
         python3 -c "import socket; print(socket.gethostbyname('$host'))" 2>/dev/null || \
         echo "")
    if [[ -n "$ip" ]]; then
        echo "  $host -> $ip"
        ALLOWED_IPS+=("$ip")
    else
        echo "  WARNING: could not resolve $host; skipping"
    fi
done

if [[ ${#ALLOWED_IPS[@]} -eq 0 ]]; then
    echo "ERROR: No registry IPs resolved. Check network connectivity."
    exit 1
fi

# --- Create the Docker bridge network with verification label --------------
echo ""
echo "Creating Docker network '$NETWORK_NAME'..."
docker network create \
    --driver bridge \
    --opt com.docker.network.bridge.name="$NETWORK_NAME" \
    --label "$LABEL_KEY=$LABEL_VALUE" \
    "$NETWORK_NAME"

echo "Network '$NETWORK_NAME' created."

# --- Add iptables rules ---------------------------------------------------
echo ""
echo "Adding iptables rules..."

# Create custom chain (idempotent)
iptables -N rl_allowlist_filter 2>/dev/null || \
    iptables -F rl_allowlist_filter   # flush if already exists

# Allow traffic to each resolved registry IP
for ip in "${ALLOWED_IPS[@]}"; do
    iptables -A rl_allowlist_filter -d "$ip" -j ACCEPT
done

# Allow DNS (port 53) so hostname resolution inside the container works
iptables -A rl_allowlist_filter -p udp --dport 53 -j ACCEPT
iptables -A rl_allowlist_filter -p tcp --dport 53 -j ACCEPT

# Allow established/related connections (return traffic)
iptables -A rl_allowlist_filter -m state --state ESTABLISHED,RELATED -j ACCEPT

# Drop everything else from this bridge
iptables -A rl_allowlist_filter -j DROP

# Jump into our chain from FORWARD for traffic from this bridge
iptables -D FORWARD -i "$NETWORK_NAME" -j rl_allowlist_filter 2>/dev/null || true
iptables -I FORWARD 1 -i "$NETWORK_NAME" -j rl_allowlist_filter

echo "iptables rules applied."
echo ""
echo "=== Setup complete ==="
echo ""
echo "Allowed registries:"
for host in "${ALLOWED_HOSTS[@]}"; do echo "  - $host"; done
echo ""
echo "Network label for DockerEpisodeExecutor:"
echo "  $LABEL_KEY=$LABEL_VALUE"
echo ""
echo "Set in your DockerExecutorConfig:"
echo "  NetworkConfig(mode='allowlist', docker_network_name='$NETWORK_NAME')"
echo ""
echo "Verify with:"
echo "  docker network inspect $NETWORK_NAME | grep -A2 Labels"
