"""
src/rl_env/docker_executor.py
==============================
DockerEpisodeExecutor — container-isolated execution backend for RL training.

Lifecycle
---------
One Docker container per episode:

    executor = DockerEpisodeExecutor()
    executor.start_episode(
        repo_path="/path/to/clone",
        repo_name="owner/repo",
        ecosystems=["python"],
    )
    result = executor.execute(step)   # repeated for each step
    executor.end_episode()            # container destroyed

``PlannerEnv`` calls ``end_episode()`` + ``start_episode()`` in ``reset()`` to
enforce the "one fresh container per episode" invariant.  The container is
never reused across episodes.

Container design
----------------
* Repo bind-mounted **read-only** at ``/workspace`` — policy can read source
  files but cannot corrupt the cached clone on the host.
* Writable overlayfs at ``/workspace_rw`` for episode side-effects:
  - ``pip install --user`` respects ``PYTHONUSERBASE=/workspace_rw``
  - ``npm`` respects ``NPM_CONFIG_PREFIX=/workspace_rw``
  Packages installed during an episode are visible to verify commands in the
  **same** episode (e.g. ``python -c 'import requests'`` works after
  ``pip install requests``) without modifying the host or the read-only mount.
  Unlike tmpfs, the overlayfs layer is captured by ``docker commit``, enabling
  snapshotting of in-progress episode state for GRPO branch evaluation.
* **Package cache mounts**: a host-side pip/npm cache directory is bind-mounted
  read-write into every container (``/cache/pip``, ``/cache/npm``).  This
  means packages downloaded by one container are reused by all subsequent
  containers without hitting the network.  pip and npm are both safe for
  concurrent cache access (file-level locking).
* **Snapshot/restore**: ``snapshot()`` commits the running container's
  overlayfs layer to a named local image.  ``start_episode(snapshot_image=...)``
  boots from that image instead of the base image, so group-evaluation
  containers inherit the main trajectory's installed packages without replaying
  history.  Snapshots are cleaned up in ``end_episode()`` via ``try/finally``.
* Resource limits: memory, CPU quota, PID count — all configurable via
  :class:`DockerExecutorConfig`.
* Network: ``"none"`` by default; ``"allowlist"`` mode resolves package-registry
  hostnames to IPs and injects them as ``--add-host`` entries on a named network
  with operator-managed iptables rules (see :class:`NetworkConfig`).

Three-tier gate
---------------
:class:`~src.sandbox.classifier.CommandRiskClassifier` sits in front of every
``docker exec`` call — identical to the gate in ``DryRunExecutor`` and
``SandboxExecutor``.  BLOCKED steps **never** reach the container.  REVIEW
steps are auto-denied.  Only SAFE steps run via ``docker exec``.

Per-step timeout and process kill
-----------------------------------
Each step runs via ``subprocess.Popen`` (not ``subprocess.run``) so the
``docker exec`` process can be explicitly killed on timeout.  The kill sequence
on :exc:`subprocess.TimeoutExpired`:

  1. ``proc.kill()``   — SIGKILL to the ``docker exec`` host-side process.
  2. ``proc.wait()``   — reap the zombie to release OS resources.
  3. ``kill -9 -1`` inside the container — kill every remaining process in the
     container's PID namespace so no runaway child survives into the next step.

The result returned has ``returncode=-1`` and a ``[timeout]`` prefix in stderr.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from src.sandbox.classifier import CommandRiskClassifier
from src.sandbox.models import (
    AtomicStep,
    ClassificationResult,
    ExecutionResult,
    ExecutionStatus,
    RiskTier,
)
from src.rl_env.dry_run_executor import TrainingReviewPolicy
from src.rl_env.verify_table import inject_verify_command

# ---------------------------------------------------------------------------
# Package cache configuration
# ---------------------------------------------------------------------------

# Host-side directories for persistent pip/npm caches, shared across all
# episode containers.  Created on first use.  The cache directories are
# bind-mounted read-write into every container so packages downloaded once
# are reused without hitting the network again.
_DEFAULT_PKG_CACHE_ROOT = Path("cache/docker_pkg_cache")
_PIP_CACHE_HOST  = _DEFAULT_PKG_CACHE_ROOT / "pip"
_NPM_CACHE_HOST  = _DEFAULT_PKG_CACHE_ROOT / "npm"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DockerNotAvailableError(RuntimeError):
    """
    Raised when the Docker daemon is not reachable or the ``docker`` binary
    is missing from PATH.

    This is a subclass of :exc:`RuntimeError` so callers can catch it without
    importing this module if they handle the generic base class.
    """


class NetworkNotConfiguredError(RuntimeError):
    """
    Raised when ``mode="allowlist"`` is requested but ``docker_network_name``
    has not been set in :class:`NetworkConfig`.

    The allowlist network mode requires a **pre-created** Docker network with
    operator-configured iptables rules (see :class:`NetworkConfig`).  Without
    that network the traffic-restriction guarantee cannot be enforced, so
    :class:`DockerEpisodeExecutor` refuses to start rather than silently
    falling back to unrestricted access.

    Resolution
    ----------
    Run the one-time setup script to provision the restricted network::

        scripts/setup_docker_network.sh

    Then set ``NetworkConfig.docker_network_name = "rl_allowlist"`` in your
    :class:`DockerExecutorConfig`.
    """


# ---------------------------------------------------------------------------
# Internal result carrier for raw docker exec output
# ---------------------------------------------------------------------------

class _ExecResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Verification label set by scripts/setup_docker_network.sh
# ---------------------------------------------------------------------------

#: Label that ``setup_docker_network.sh`` adds to the allowlist network.
#: :meth:`DockerEpisodeExecutor._verify_network_label` checks for this label
#: to confirm the network was provisioned correctly with iptables rules.
_NETWORK_VERIFIED_LABEL_KEY   = "com.contai.rl.network.verified"
_NETWORK_VERIFIED_LABEL_VALUE = "allowlist-v1"

# ---------------------------------------------------------------------------
# Per-ecosystem image selection
# ---------------------------------------------------------------------------

#: Mapping from detected ecosystem name → Docker base image.
#: Extend this dict to add support for additional runtimes.
ECOSYSTEM_IMAGES: dict[str, str] = {
    "python":     "python:3.11-slim",
    "javascript": "node:20-slim",
    "node":       "node:20-slim",
    "typescript": "node:20-slim",
    "rust":       "rust:1.75-slim",
    "go":         "golang:1.21-bookworm",
    "java":       "eclipse-temurin:21-jre-jammy",
    "ruby":       "ruby:3.2-slim",
}

# Combined Python+Node image for repos with both ecosystems.
_MULTI_PYTHON_NODE_IMAGE = "nikolaik/python-nodejs:python3.11-nodejs20"
_DEFAULT_IMAGE = "python:3.11-slim"

# Ordered priority when multiple ecosystems are detected but no combined image exists.
_ECO_PRIORITY = (
    "python", "node", "javascript", "typescript",
    "rust", "go", "java", "ruby",
)


def select_image(ecosystems: Optional[List[str]]) -> str:
    """
    Choose the appropriate Docker base image for the detected ecosystems.

    Parameters
    ----------
    ecosystems:
        List of ecosystem names detected by ``scan_repo()`` (e.g.
        ``["python"]``, ``["javascript", "python"]``).  May be ``None`` or
        empty.

    Returns
    -------
    str
        Docker image name.  Falls back to ``python:3.11-slim`` for unknown or
        empty ecosystem lists.

    Notes
    -----
    Multi-ecosystem combos without a dedicated combined image fall back to the
    highest-priority ecosystem in ``_ECO_PRIORITY``.  A warning is printed so
    the fallback is never silent — operators can then build a combined image
    and add it to :data:`ECOSYSTEM_IMAGES`.

    Examples
    --------
    >>> select_image(["python"])
    'python:3.11-slim'
    >>> select_image(["javascript"])
    'node:20-slim'
    >>> select_image(["python", "javascript"])
    'nikolaik/python-nodejs:python3.11-nodejs20'
    >>> select_image(None)
    'python:3.11-slim'
    """
    if not ecosystems:
        return _DEFAULT_IMAGE

    eco_set = {e.lower() for e in ecosystems}

    python_present = "python" in eco_set
    node_present   = bool(eco_set & {"javascript", "node", "typescript"})

    if python_present and node_present:
        return _MULTI_PYTHON_NODE_IMAGE

    if len(eco_set) == 1:
        return ECOSYSTEM_IMAGES.get(next(iter(eco_set)), _DEFAULT_IMAGE)

    # Multi-ecosystem with no combined image: pick highest-priority ecosystem
    # and log explicitly — never silently choose a possibly-wrong image.
    for eco in _ECO_PRIORITY:
        if eco in eco_set:
            chosen = ECOSYSTEM_IMAGES.get(eco, _DEFAULT_IMAGE)
            remaining = sorted(eco_set - {eco})
            print(
                f"[DockerEpisodeExecutor] INFO: multi-ecosystem repo "
                f"{sorted(eco_set)} has no combined image; "
                f"using {chosen!r} (matched ecosystem: {eco!r}). "
                f"Unsupported ecosystems for this episode: {remaining}. "
                f"Add a combined image to ECOSYSTEM_IMAGES to resolve this.",
                file=sys.stderr,
            )
            return chosen

    return _DEFAULT_IMAGE


# ---------------------------------------------------------------------------
# NetworkConfig
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """
    Network isolation configuration for episode containers.

    Modes
    -----
    ``"none"`` (default)
        Container has no network access.  Safest option for unattended
        rollouts.  Package install commands will fail unless all required
        packages are pre-baked into the Docker image.

    ``"allowlist"``
        Restricts outbound access to a curated list of package-registry hosts.

        At ``start_episode()`` time each hostname in ``allowed_registry_hosts``
        is resolved to its current IP address.  These IPs are injected into the
        container as ``--add-host`` entries (``/etc/hosts`` overrides).

        If ``docker_network_name`` is provided, that pre-created Docker network
        is used for the ``--network`` flag.  The expectation is that this
        network has iptables ``FORWARD`` / ``OUTPUT`` rules that block all
        traffic except to the resolved registry IPs.  Create the network once
        before training::

            docker network create \\
                --driver bridge \\
                --opt com.docker.network.bridge.name=rl_allowlist \\
                rl_allowlist
            # Then add iptables rules to restrict FORWARD to allowed IPs only.

        If ``docker_network_name`` is ``None``, the default bridge network is
        used with ``--add-host`` overrides but **no iptables enforcement**.
        A warning is printed.  This mode is suitable only for development;
        production training **must** use a pre-created network with firewall rules.

    ``"bridge"``
        Full internet access.  Not recommended for unattended RL rollouts —
        a misbehaving policy could make arbitrary external requests or exfiltrate
        repository data.  Use only in controlled environments where the policy
        is already trusted.
    """

    mode: str = "none"
    """Network mode: ``"none"`` | ``"allowlist"`` | ``"bridge"``."""

    allowed_registry_hosts: List[str] = field(default_factory=lambda: [
        # PyPI (Python)
        "pypi.org",
        "files.pythonhosted.org",
        # npm (JavaScript / Node)
        "registry.npmjs.org",
        "registry.yarnpkg.com",
        # RubyGems
        "rubygems.org",
        # crates.io (Rust)
        "crates.io",
        "static.crates.io",
        # Go module proxy
        "proxy.golang.org",
        "sum.golang.org",
    ])
    """
    Hostnames to resolve and inject as ``--add-host`` entries when
    ``mode="allowlist"``.  Only effective when ``docker_network_name`` points
    to a network with matching iptables rules; without that, these entries only
    affect ``/etc/hosts`` name resolution.
    """

    docker_network_name: Optional[str] = None
    """
    Name of the pre-created Docker network to use in ``"allowlist"`` mode.

    **Required when** ``mode="allowlist"``; leaving it ``None`` causes
    :meth:`~DockerEpisodeExecutor.start_episode` to raise
    :exc:`NetworkNotConfiguredError` rather than silently falling back to
    unrestricted access.  This is intentional fail-closed behaviour.

    Create the network and its firewall rules once before training::

        scripts/setup_docker_network.sh

    Then pass the network name here::

        NetworkConfig(mode="allowlist", docker_network_name="rl_allowlist")
    """


# ---------------------------------------------------------------------------
# DockerExecutorConfig
# ---------------------------------------------------------------------------

@dataclass
class DockerExecutorConfig:
    """
    Full configuration for :class:`DockerEpisodeExecutor`.

    All defaults are conservative — suitable for multiple parallel rollouts on
    a single training host.  Profile actual usage before increasing limits.

    Parameters
    ----------
    memory_limit:
        Maximum RAM per episode container.  Docker format: ``"512m"``,
        ``"2g"``, etc.  Memory swap is set equal to this limit (swap disabled).
    cpu_quota:
        Docker CPU quota in microseconds per 100 ms scheduling period.
        100 000 = 1 full CPU core.  Default 50 000 = 50 % of one core.
    pids_limit:
        Maximum concurrent processes in the container.  Prevents fork-bomb
        patterns that might evade the classifier.
    step_timeout:
        Wall-clock seconds per main command before SIGKILL.  Default 30 s.
    verify_timeout:
        Wall-clock seconds per verify command before SIGKILL.  Default 10 s.
    pull_policy:
        ``"always"``  — always pull the image before starting the container.
        ``"missing"`` — pull only if the image is not in the local cache.
        ``"never"``   — never pull; fail loudly if the image is absent.
    network:
        Network isolation configuration.  See :class:`NetworkConfig`.
    """

    memory_limit:   str           = "512m"
    cpu_quota:      int           = 50_000   # 50 % of one CPU core
    pids_limit:     int           = 64
    step_timeout:   int           = 30       # seconds
    verify_timeout: int           = 10       # seconds
    pull_policy:    str           = "missing"
    network:        NetworkConfig = field(default_factory=NetworkConfig)


# ---------------------------------------------------------------------------
# DockerEpisodeExecutor
# ---------------------------------------------------------------------------

class DockerEpisodeExecutor:
    """
    Container-isolated step executor for RL training episodes.

    One Docker container persists across ALL steps within a single episode.
    A fresh container is created for each new episode.  This design gives
    later steps access to the side-effects of earlier steps (e.g. an
    INSTALL step's packages are visible to a subsequent CHECK step via
    ``/workspace_rw``), while guaranteeing complete isolation between episodes.

    Interface mirrors :class:`~src.rl_env.dry_run_executor.DryRunExecutor`:
    both expose ``execute(step) -> ExecutionResult``.  The additional
    ``start_episode()`` / ``end_episode()`` calls are managed by
    :class:`~src.rl_env.env.PlannerEnv`.

    Parameters
    ----------
    config:
        :class:`DockerExecutorConfig`.  Defaults to conservative resource limits
        with ``network.mode="none"`` (no outbound network).
    review_policy:
        How REVIEW-tier steps are handled.  In a container environment there is
        no human present, so ``AUTO_DENY`` (default) and ``BLOCK_AS_HARD_STOP``
        are the only meaningful values.  ``AUTO_APPROVE`` is not supported
        (treated as ``AUTO_DENY`` with a warning).

    Raises
    ------
    DockerNotAvailableError
        Raised from :meth:`start_episode` if the Docker daemon is unreachable
        or the ``docker`` binary is not on PATH.
    RuntimeError
        Raised from :meth:`start_episode` if ``docker run`` fails.
    """

    def __init__(
        self,
        config: Optional[DockerExecutorConfig] = None,
        *,
        review_policy: TrainingReviewPolicy = TrainingReviewPolicy.AUTO_DENY,
        pkg_cache_root: Optional[Path] = None,
    ) -> None:
        self._config        = config or DockerExecutorConfig()
        self._review_policy = review_policy
        # Classifier uses /workspace_rw as sandbox_root: the writable episode
        # directory inside the container.  Out-of-scope path checks target that
        # directory, not the host filesystem.
        self._classifier    = CommandRiskClassifier(sandbox_root="/workspace_rw")
        self._container_id: Optional[str] = None
        self._repo_name:    str           = ""
        # Snapshot tracking: the current live snapshot image tag for this
        # executor instance (set by snapshot(), cleared by delete_snapshot()).
        self._snapshot_tag: Optional[str] = None
        # Host-side package cache root (absolute).  Created on first start_episode.
        _root = pkg_cache_root or _DEFAULT_PKG_CACHE_ROOT
        self._pip_cache_host = (_root / "pip").resolve()
        self._npm_cache_host = (_root / "npm").resolve()

    # ---------------------------------------------------------------- public

    def start_episode(
        self,
        repo_path:   str,
        repo_name:   str                 = "",
        ecosystems:  Optional[List[str]] = None,
        *,
        snapshot_image: Optional[str]   = None,
    ) -> None:
        """
        Start the episode container.

        Parameters
        ----------
        repo_path:
            Absolute path to the git-cloned repo directory on the host.
            Bind-mounted read-only at ``/workspace`` inside the container.
        repo_name:
            ``"owner/repo"`` identifier — used to name the container for
            easier debugging (``docker ps`` shows meaningful names).
        ecosystems:
            Detected ecosystem list (e.g. ``["python"]``, ``["python", "node"]``).
            Drives Docker image selection.  ``None`` or empty → uses the Python
            3.11-slim default.  Ignored when ``snapshot_image`` is set (the
            snapshot already encodes the correct base image).
        snapshot_image:
            If provided, boot from this pre-committed snapshot image instead of
            the base ecosystem image.  Used by GRPO group-evaluation containers
            to inherit the main trajectory's installed packages without replaying
            history.  The snapshot must have been produced by :meth:`snapshot`.
        """
        # Guard: tear down any container leaked from a previous episode.
        if self._container_id is not None:
            self.end_episode()

        self._repo_name = repo_name

        # Use snapshot image if provided; otherwise select by ecosystem.
        if snapshot_image:
            image = snapshot_image
            # Snapshot images already have the correct layers; no pull needed.
        else:
            image = select_image(ecosystems)
            self._check_docker_available()
            self._pull_image_if_needed(image)

        if not snapshot_image:
            self._check_docker_available()

        network_flags, extra_host_flags = self._build_network_flags()

        # Verify the allowlist network was provisioned by the setup script.
        if self._config.network.mode == "allowlist":
            self._verify_network_label(self._config.network.docker_network_name)

        # Ensure host-side package cache directories exist before mounting.
        self._pip_cache_host.mkdir(parents=True, exist_ok=True)
        self._npm_cache_host.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [
            "docker", "run",
            "--detach",
            "--rm=false",                        # we control removal in end_episode()
            "--name", self._make_container_name(),
            # --- Resource limits -------------------------------------------------
            "--memory", self._config.memory_limit,
            "--memory-swap", self._config.memory_limit,  # disable swap entirely
            "--cpu-quota", str(self._config.cpu_quota),
            "--cpu-period", "100000",
            "--pids-limit", str(self._config.pids_limit),
            # --- Filesystem -------------------------------------------------------
            # Repo: read-only so the policy cannot modify the cached clone.
            "--volume", f"{repo_path}:/workspace:ro",
            # Episode working area: overlayfs (NOT tmpfs) so docker commit captures
            # installed packages.  The directory is initialised to /workspace_rw
            # inside the container; the overlayfs layer is discarded when the
            # container is removed (rm=false + explicit docker rm in end_episode).
            "--workdir", "/workspace",
            # --- Persistent package caches (host-side, shared across containers) --
            # Bind-mounted read-write so packages downloaded once are reused by all
            # subsequent containers without network round-trips.  pip and npm both
            # handle concurrent cache access via file-level locking.
            "--volume", f"{self._pip_cache_host}:/cache/pip",
            "--volume", f"{self._npm_cache_host}:/cache/npm",
            # --- Package manager environment --------------------------------------
            "--env", "PYTHONUSERBASE=/workspace_rw",
            "--env", "PIP_CACHE_DIR=/cache/pip",      # persistent host-side cache
            "--env", "NPM_CONFIG_PREFIX=/workspace_rw",
            "--env", "NPM_CONFIG_CACHE=/cache/npm",   # persistent host-side cache
            "--env", "NODE_PATH=/workspace_rw/lib/node_modules",
            "--env", "HOME=/workspace_rw",
            "--env", "PIP_ROOT_USER_ACTION=ignore",
            # --- Network ---------------------------------------------------------
            *network_flags,
            *extra_host_flags,
            # --- Image + entrypoint ----------------------------------------------
            image,
            "sleep", "infinity",    # container stays alive; steps use docker exec
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"[DockerEpisodeExecutor] docker run failed for image {image!r}.\n"
                f"  stdout: {result.stdout.strip()}\n"
                f"  stderr: {result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()

        # Ensure /workspace_rw exists inside the container.
        # With overlayfs (no tmpfs), the directory must be created explicitly
        # on the base image.  For snapshot containers it will already exist
        # (inherited from the committed layer), so mkdir -p is idempotent.
        subprocess.run(
            ["docker", "exec", self._container_id, "mkdir", "-p", "/workspace_rw"],
            capture_output=True,
            timeout=10,
        )

    def execute(self, step: AtomicStep) -> ExecutionResult:
        """
        Execute one :class:`~src.sandbox.models.AtomicStep` inside the episode container.

        The three-tier gate runs first:

        * BLOCKED → returned immediately with :attr:`~src.sandbox.models.ExecutionStatus.BLOCKED`;
          the container is never contacted.
        * REVIEW  → auto-denied per :attr:`review_policy`; the container is never contacted.
        * SAFE    → run via ``docker exec``; verify command also run if present.

        Parameters
        ----------
        step:
            The step to execute.  ``step.verify_command``, if set, is run inside
            the same container after the main command exits 0.  The
            ``PYTHONUSERBASE=/workspace_rw`` environment variable (set at
            container-start time) ensures that packages installed during this
            episode are visible to verify commands without any extra flags.

        Returns
        -------
        ExecutionResult

        Raises
        ------
        RuntimeError
            If called before :meth:`start_episode`.
        """
        if self._container_id is None:
            raise RuntimeError(
                "[DockerEpisodeExecutor] No container running. "
                "Call start_episode() before execute()."
            )

        classification = self._classifier.classify(step)

        if classification.tier == RiskTier.BLOCKED:
            return self._handle_blocked(step, classification)

        if classification.tier == RiskTier.REVIEW:
            return self._handle_review(step, classification)

        return self._handle_safe(step, classification)

    def snapshot(self, tag: Optional[str] = None) -> str:
        """
        Commit the running container's overlayfs state to a new local image.

        Used by GRPO training to capture the main trajectory's state after
        each successful step so group-evaluation containers can branch from
        it without replaying history.

        Ordering guarantee
        ------------------
        The caller (``train_system2_grpo.py``) must follow
        commit → verify → delete-previous, never delete → commit.  This
        ensures there is always at least one valid snapshot available even
        if the process crashes between steps.

        Parameters
        ----------
        tag:
            Optional image tag.  Defaults to ``grpo-snapshot-<uuid8>``.
            Must be a valid Docker image name (lowercase, no spaces).

        Returns
        -------
        str
            The image tag that was created.  Pass this to
            :meth:`start_episode` as ``snapshot_image`` on group containers.

        Raises
        ------
        RuntimeError
            If no container is running or if ``docker commit`` fails.
        """
        if self._container_id is None:
            raise RuntimeError(
                "[DockerEpisodeExecutor] snapshot() called with no running container."
            )

        new_tag = tag or f"grpo-snapshot-{uuid.uuid4().hex[:8]}"
        result = subprocess.run(
            ["docker", "commit", self._container_id, new_tag],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[DockerEpisodeExecutor] docker commit failed for container "
                f"{self._container_id!r} -> {new_tag!r}.\n"
                f"  stderr: {result.stderr.strip()}"
            )

        print(
            f"[DockerEpisodeExecutor] Snapshot committed: {new_tag}",
            file=sys.stderr,
        )
        return new_tag

    def delete_snapshot(self, tag: str) -> None:
        """
        Remove a snapshot image created by :meth:`snapshot`.

        Safe to call even if the image has already been removed (idempotent).
        Errors are logged as warnings, never raised — cleanup must not abort
        a training run.

        Parameters
        ----------
        tag:
            The image tag returned by a previous :meth:`snapshot` call.
        """
        try:
            result = subprocess.run(
                ["docker", "rmi", "-f", tag],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                print(
                    f"[DockerEpisodeExecutor] WARNING: docker rmi {tag!r} failed "
                    f"(may already be removed): {result.stderr.strip()}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[DockerEpisodeExecutor] Snapshot deleted: {tag}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"[DockerEpisodeExecutor] WARNING: delete_snapshot({tag!r}) raised: {exc}",
                file=sys.stderr,
            )

    def end_episode(self) -> None:
        """
        Stop and remove the episode container, then clean up any snapshot.

        Called on every episode exit path — success, failure, truncation,
        and exception — via ``try/finally`` in :class:`~src.rl_env.env.PlannerEnv`.

        Safe to call if the container has already been destroyed externally
        (OOM kill, manual ``docker stop``).  Always resets ``_container_id``
        to ``None`` so subsequent :meth:`start_episode` calls start clean.

        Snapshot cleanup
        ----------------
        Any snapshot image tracked in ``self._snapshot_tag`` is deleted here
        unconditionally, guaranteeing no dangling images are left behind after
        an aborted episode.
        """
        try:
            if self._container_id is None:
                return

            cid = self._container_id
            self._container_id = None  # clear before stop so we don't retry on error

            try:
                subprocess.run(
                    ["docker", "stop", cid],
                    capture_output=True,
                    timeout=15,
                )
            except Exception as exc:
                print(
                    f"[DockerEpisodeExecutor] WARNING: docker stop {cid!r} failed: {exc}",
                    file=sys.stderr,
                )

            try:
                subprocess.run(
                    ["docker", "rm", "-f", cid],
                    capture_output=True,
                    timeout=10,
                )
            except Exception as exc:
                print(
                    f"[DockerEpisodeExecutor] WARNING: docker rm -f {cid!r} failed: {exc}",
                    file=sys.stderr,
                )
        finally:
            # Always clean up any tracked snapshot image, regardless of whether
            # the container teardown above succeeded.  This fires on every exit
            # path: success, failure, truncation, and unhandled exception.
            if self._snapshot_tag is not None:
                snap = self._snapshot_tag
                self._snapshot_tag = None  # clear first so a crash here doesn't retry
                self.delete_snapshot(snap)

    @property
    def container_id(self) -> Optional[str]:
        """The running container's ID, or ``None`` if no episode is active."""
        return self._container_id

    def __enter__(self) -> "DockerEpisodeExecutor":
        return self

    def __exit__(self, *_: object) -> None:
        self.end_episode()

    # ---------------------------------------------------------------- private

    def _handle_blocked(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        return ExecutionResult(
            step=step,
            status=ExecutionStatus.BLOCKED,
            tier=RiskTier.BLOCKED,
            classification=classification,
            verified=False,
        )

    def _handle_review(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        if self._review_policy == TrainingReviewPolicy.BLOCK_AS_HARD_STOP:
            return ExecutionResult(
                step=step,
                status=ExecutionStatus.BLOCKED,
                tier=RiskTier.REVIEW,
                classification=classification,
                verified=False,
                stderr=(
                    f"[DockerEpisodeExecutor] REVIEW step escalated to BLOCKED "
                    f"by policy '{self._review_policy}': {step.command!r}"
                ),
            )

        # AUTO_DENY (default) and AUTO_APPROVE both deny inside the container:
        # there is no human confirmation path in an unattended container
        # environment. AUTO_APPROVE is silently treated as AUTO_DENY here.
        if self._review_policy == TrainingReviewPolicy.AUTO_APPROVE:
            print(
                f"[DockerEpisodeExecutor] WARNING: AUTO_APPROVE is not supported "
                f"in container execution — treating as AUTO_DENY for step: "
                f"{step.command!r}",
                file=sys.stderr,
            )

        return ExecutionResult(
            step=step,
            status=ExecutionStatus.ABORTED,
            tier=RiskTier.REVIEW,
            classification=classification,
            verified=False,
            stderr=(
                f"[DockerEpisodeExecutor] REVIEW step auto-denied "
                f"(effective policy: AUTO_DENY): {step.command!r}"
            ),
        )

    def _handle_safe(
        self,
        step:           AtomicStep,
        classification: ClassificationResult,
    ) -> ExecutionResult:
        """
        Run a SAFE step via docker exec and optionally run its verify command.

        The verify command is always derived from the deterministic lookup
        table in :mod:`src.rl_env.verify_table` (``inject_verify_command``).
        Any ``verify_command`` the policy may have placed on the step is
        **unconditionally replaced** here — this is the structural fix that
        closes the self-certification exploit (verify_command="echo ok" etc.).

        The injected verify command is run inside the same container, in the
        same environment, with ``PYTHONUSERBASE=/workspace_rw`` already set
        so packages installed during this episode are visible.
        """
        # --- Overwrite verify_command from the lookup table (not policy) ----
        step = inject_verify_command(step)

        main = self._run_docker_exec(step.command, timeout=self._config.step_timeout)
        main_status = (
            ExecutionStatus.SUCCESS if main.returncode == 0
            else ExecutionStatus.FAILED
        )

        verified:           bool         = False
        verify_stdout:      str          = ""
        verify_returncode:  Optional[int] = None

        if main.returncode == 0 and step.verify_command:
            # PYTHONUSERBASE=/workspace_rw is already in the container environment
            # (set at start_episode time), so packages installed via
            # "pip install --user ..." are visible to "python -c 'import pkg'"
            # without any extra flags or path manipulation here.
            verify = self._run_docker_exec(
                step.verify_command,
                timeout=self._config.verify_timeout,
            )
            verify_returncode = verify.returncode
            verify_stdout     = verify.stdout
            verified          = (verify.returncode == 0)

        return ExecutionResult(
            step=step,
            status=main_status,
            tier=RiskTier.SAFE,
            classification=classification,
            stdout=main.stdout,
            stderr=main.stderr,
            returncode=main.returncode,
            verified=verified,
            verify_returncode=verify_returncode,
            verify_stdout=verify_stdout,
        )

    def _run_docker_exec(self, command: str, timeout: int) -> _ExecResult:
        """
        Run *command* inside the episode container via ``docker exec``.

        Uses :class:`subprocess.Popen` (not ``subprocess.run``) so the process
        can be explicitly killed on timeout.

        On :exc:`subprocess.TimeoutExpired`:

        1. ``proc.kill()``  — SIGKILL the ``docker exec`` host-side process.
        2. ``proc.wait()``  — reap the zombie to avoid an OS resource leak.
        3. ``kill -9 -1`` inside the container — terminate every remaining
           process in the container's PID namespace so no runaway child
           survives into the next step.

        Returns
        -------
        _ExecResult
            On timeout: ``returncode=-1``, ``stderr`` starts with ``"[timeout]"``.
        """
        proc = subprocess.Popen(
            [
                "docker", "exec",
                self._container_id,
                "/bin/sh", "-c", command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return _ExecResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            # --- Kill sequence ---------------------------------------------------
            proc.kill()   # SIGKILL the docker exec process (host side)
            proc.wait()   # reap zombie; avoids resource leak

            # Kill all processes still alive in the container's PID namespace.
            # "kill -9 -1" signals every process the container user can reach.
            # Errors here are benign (container may already be in bad state).
            try:
                subprocess.run(
                    [
                        "docker", "exec",
                        self._container_id,
                        "/bin/sh", "-c", "kill -9 -1 2>/dev/null || true",
                    ],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

            return _ExecResult(
                returncode=-1,
                stdout="",
                stderr=(
                    f"[timeout] Command exceeded {timeout}s wall-clock limit "
                    f"and was killed: {command!r}"
                ),
            )

    def _check_docker_available(self) -> None:
        """
        Verify Docker is installed and the daemon is responsive.

        Raises
        ------
        DockerNotAvailableError
        """
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise DockerNotAvailableError(
                    "Docker daemon is not running or not accessible.\n"
                    f"  docker version stderr: {result.stderr.strip()}\n"
                    "Start the Docker daemon and retry.\n"
                    "Install guide: https://docs.docker.com/get-docker/"
                )
        except FileNotFoundError:
            raise DockerNotAvailableError(
                "docker binary not found on PATH.\n"
                "Install Docker: https://docs.docker.com/get-docker/"
            ) from None
        except subprocess.TimeoutExpired:
            raise DockerNotAvailableError(
                "docker version timed out after 5 s — daemon may be hung.\n"
                "Restart the Docker daemon and retry."
            ) from None

    def _pull_image_if_needed(self, image: str) -> None:
        """Pull *image* according to the configured pull_policy."""
        policy = self._config.pull_policy

        if policy == "never":
            return

        if policy == "missing":
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return   # image already cached locally

        # "always" path, or image was missing in the "missing" branch.
        print(f"[DockerEpisodeExecutor] Pulling image {image!r} ...")
        result = subprocess.run(
            ["docker", "pull", image],
            timeout=300,   # 5-minute pull budget
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[DockerEpisodeExecutor] Failed to pull image {image!r}. "
                "Check your network connection and that the image name is correct."
            )

    def _build_network_flags(self) -> Tuple[List[str], List[str]]:
        """
        Build ``--network`` and ``--add-host`` flags for ``docker run``.

        Returns
        -------
        (network_flags, extra_host_flags)
            Both are lists of strings ready to be splatted into the docker
            run command.  Returned separately so tests can inspect each group.
        """
        net = self._config.network

        if net.mode == "none":
            return ["--network", "none"], []

        if net.mode == "bridge":
            return ["--network", "bridge"], []

        if net.mode == "allowlist":
            # --- Fail-closed: require a pre-created network name ----------
            # Allowlist mode without a named Docker network cannot enforce
            # iptables traffic restrictions.  Silently falling back to the
            # default bridge would give a false sense of isolation while
            # running the container with full internet access.
            if not net.docker_network_name:
                raise NetworkNotConfiguredError(
                    "NetworkConfig.mode='allowlist' requires "
                    "docker_network_name to be set to a pre-created Docker "
                    "network with iptables firewall rules.\n"
                    "Without a named network the traffic-restriction guarantee "
                    "cannot be enforced; refusing to start rather than silently "
                    "falling back to unrestricted access.\n\n"
                    "Run the one-time setup script:\n"
                    "    scripts/setup_docker_network.sh\n\n"
                    "Then set:\n"
                    "    NetworkConfig("
                    "mode='allowlist', "
                    "docker_network_name='rl_allowlist')"
                )

            # Resolve allowed hostnames to current IPs for --add-host injection.
            extra_hosts: List[str] = []
            unresolvable: List[str] = []
            for host in net.allowed_registry_hosts:
                try:
                    ip = socket.gethostbyname(host)
                    extra_hosts.extend(["--add-host", f"{host}:{ip}"])
                except socket.gaierror:
                    unresolvable.append(host)

            if unresolvable:
                print(
                    f"[DockerEpisodeExecutor] WARNING: could not resolve hosts "
                    f"{unresolvable} for allowlist — skipping those entries.",
                    file=sys.stderr,
                )

            return ["--network", net.docker_network_name], extra_hosts

        raise ValueError(
            f"[DockerEpisodeExecutor] Unknown network mode: {net.mode!r}.  "
            "Expected one of: 'none', 'allowlist', 'bridge'."
        )

    def _make_container_name(self) -> str:
        """Generate a unique, Docker-safe container name for this episode."""
        # Sanitise repo name: Docker allows [a-zA-Z0-9][a-zA-Z0-9_.-]*
        safe_repo = re.sub(r"[^a-zA-Z0-9_.\-]", "_", self._repo_name or "episode")
        return f"rl_ep_{safe_repo}_{uuid.uuid4().hex[:8]}"

    def _verify_network_label(self, network_name: Optional[str]) -> None:
        """
        Check that *network_name* carries the expected setup-script label.

        The label ``com.contai.rl.network.verified=allowlist-v1`` is injected
        by ``scripts/setup_docker_network.sh`` after it applies iptables rules.
        Its presence confirms the network was provisioned correctly, not just
        created bare with ``docker network create``.

        Raises
        ------
        NetworkNotConfiguredError
            If the network is not found, or if the expected label is absent.
        """
        if not network_name:
            # _build_network_flags already raised; this is a safety guard.
            raise NetworkNotConfiguredError(
                "allowlist mode requires docker_network_name to be set."
            )

        import json as _json

        try:
            result = subprocess.run(
                ["docker", "network", "inspect", network_name,
                 "--format", "{{json .Labels}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            # docker not on PATH — _check_docker_available caught this first;
            # this guard handles a race condition.
            raise DockerNotAvailableError(
                "docker binary not found on PATH."
            ) from None

        if result.returncode != 0:
            raise NetworkNotConfiguredError(
                f"Docker network {network_name!r} does not exist.  "
                "Run scripts/setup_docker_network.sh to create it.\n"
                f"  docker network inspect stderr: {result.stderr.strip()}"
            )

        try:
            labels: dict = _json.loads(result.stdout.strip() or "{}")
        except _json.JSONDecodeError:
            labels = {}

        actual_value = labels.get(_NETWORK_VERIFIED_LABEL_KEY)
        if actual_value != _NETWORK_VERIFIED_LABEL_VALUE:
            raise NetworkNotConfiguredError(
                f"Docker network {network_name!r} exists but is missing the "
                f"required setup-script label.  "
                f"Expected: {_NETWORK_VERIFIED_LABEL_KEY}={_NETWORK_VERIFIED_LABEL_VALUE!r}.  "
                f"Got: {actual_value!r}.\n\n"
                "This means the network was created manually without running "
                "setup_docker_network.sh, so iptables firewall rules may not be "
                "in place.  Re-create the network using the setup script:\n"
                "    sudo scripts/setup_docker_network.sh\n\n"
                "If you intentionally created the network differently and have "
                "verified iptables rules are correct, add the label manually:\n"
                f"    docker network inspect {network_name} --format '{{{{.Labels}}}}'\n"
                f"    # Then: docker network update is not available for labels;\n"
                f"    # recreate with: docker network rm {network_name} && "
                f"scripts/setup_docker_network.sh"
            )

