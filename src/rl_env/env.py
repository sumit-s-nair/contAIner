"""
src/rl_env/env.py
=================
PlannerEnv — OpenEnv-compatible RL environment for System 2 planner training.

Interface contract (OpenEnv standard)
--------------------------------------
    env = PlannerEnv(config)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    text = env.render()

``action`` is an ``AtomicStep``.
``obs`` is an ``Observation`` (also serializable via ``ObservationSerializer``).

Episode lifecycle
------------------
1. ``reset()`` — sample a repo from the corpus, scan it, pick a template,
   build the initial Observation.
2. ``step(action)`` — execute the action, compute reward, check termination.
3. Episode terminates when:
   a. All expected template actions have been taken AND the last one was
      verified (``terminated=True``).
   b. ``max_steps`` reached without completion (``truncated=True``).
   c. A BLOCKED step was encountered (``terminated=True``, hard penalty logged).

Execution state persistence (within one episode)
-------------------------------------------------
One ``DryRunExecutor`` instance is created per episode in ``reset()`` and
reused for all ``step()`` calls.  This means all steps share the same
``sandbox_root`` directory, so later steps see the side effects of earlier
steps (e.g. an INSTALL step's packages are visible to a CHECK step that
follows).  In ``dry_run=True`` mode this is trivially true; in real mode
the executor's subprocess calls all use the same ``sandbox_root``.

Corpus loading
--------------
``PlannerEnv`` loads a corpus manifest (produced by ``scripts/build_corpus.py``)
and samples from the requested split (``"train"`` or ``"val"``).  The test
split is never accessible from ``PlannerEnv`` — load it directly for offline
evaluation only.

If no corpus manifest is provided, ``PlannerEnv`` falls back to synthetic
fixture repos (for smoke testing without a real corpus).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.repo_scan.models import Dependency, EcosystemManifest, RepoManifest, SourceRef
from src.sandbox.models import AtomicStep, ExecutionResult, ExecutionStatus, RiskTier
from src.system2_planner.models import PlannedStep, TemplateInstance
from src.rl_env.system3_stub import expand_planned_step
from src.system2_planner.templates import TEMPLATES

from .docker_executor import DockerEpisodeExecutor, DockerExecutorConfig
from .dry_run_executor import DryRunExecutor, TrainingReviewPolicy
from .observation import CanonicalIntent, Observation, ObservationSerializer
from .reward import RewardConfig, compute_reward


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """
    Configuration for ``PlannerEnv``.

    corpus_manifest_path:
        Path to the versioned corpus manifest JSON produced by
        ``scripts/build_corpus.py``.  If ``None``, synthetic fixtures are used.

    split:
        Corpus split to sample from during training.  One of:
        ``"train"`` or ``"val"``.
        The ``"test"`` split is NOT accessible from this class.

    max_steps:
        Maximum number of steps per episode before truncation.

    reward_config:
        All reward weights and per-outcome values.  Defaults to ``RewardConfig()``.

    review_policy:
        How to handle REVIEW-tier steps during rollouts.  Default: AUTO_DENY.

    dry_run:
        If True, SAFE steps return synthetic SUCCESS without real subprocesses.
        Useful for smoke testing and debugging.

    seed:
        Random seed for reproducible episode sampling.  ``None`` → unseeded.
    """

    corpus_manifest_path: Optional[str]           = None
    split:                str                     = "train"
    max_steps:            int                     = 8
    reward_config:        RewardConfig            = field(default_factory=RewardConfig)
    review_policy:        TrainingReviewPolicy    = TrainingReviewPolicy.AUTO_DENY
    dry_run:              bool                    = True
    seed:                 Optional[int]           = None
    docker_config:        Optional[DockerExecutorConfig] = None
    """
    Docker executor configuration for real (non-dry-run) episodes.
    If ``None`` and ``dry_run=False``, :class:`DockerExecutorConfig` defaults
    are used (512 MB memory, 50 % CPU, no network access).
    """


# ---------------------------------------------------------------------------
# Synthetic fixtures (used when no corpus manifest is provided)
# ---------------------------------------------------------------------------

def _make_synthetic_manifest(category: str = "manifest_present") -> RepoManifest:
    """Build a minimal RepoManifest fixture for smoke testing."""
    manifest = RepoManifest()
    eco = EcosystemManifest(
        ecosystem="python",
        manifest_files=["requirements.txt"] if category != "manifest_less" else [],
    )
    if category == "manifest_less":
        # Simulates import-scan fallback — no declared deps
        from src.repo_scan.models import InferredDependency
        eco.inferred_dependencies = [
            InferredDependency(
                import_name="requests",
                guessed_package_name="requests",
                confidence="unmapped_guess",
                sources=[SourceRef(file="src/app.py", line=1)],
            )
        ]
    else:
        eco.add_dependency(Dependency(
            name="requests",
            declared_constraint=">=2.0",
            sources=[SourceRef(file="requirements.txt", line=1,
                               raw_line="requests>=2.0")],
        ))
        if category == "known_conflict":
            eco.add_dependency(Dependency(
                name="flask",
                declared_constraint=">=2.0",
                sources=[SourceRef(file="requirements.txt", line=2,
                                   raw_line="flask>=2.0")],
            ))
            manifest.conflicts = [
                "Version conflict: 'requests' required >=2.0 but 'flask' pins 1.x"
            ]
    manifest.ecosystems["python"] = eco
    return manifest


_SYNTHETIC_CORPUS = [
    {"repo": "fixture/manifest_present", "category": "manifest_present"},
    {"repo": "fixture/manifest_less",    "category": "manifest_less"},
    {"repo": "fixture/known_conflict",   "category": "known_conflict"},
]

_TEMPLATE_FOR_CATEGORY = {
    "manifest_present": "setup_project",
    "manifest_less":    "setup_project",
    "known_conflict":   "setup_project",
}

_INTENT_TEXTS = {
    "setup_project":    "Set up this repository project for development.",
    "fix_environment":  "Diagnose and fix the broken environment component.",
    "setup_environment":"Ensure the required environment component is available.",
}


# ---------------------------------------------------------------------------
# PlannerEnv
# ---------------------------------------------------------------------------

class PlannerEnv:
    """
    OpenEnv-compatible environment for System 2 planner RL training.

    See module docstring for full lifecycle and design notes.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self._cfg       = config or EnvConfig()
        self._rng       = random.Random(self._cfg.seed)
        self._serializer = ObservationSerializer()

        # Loaded lazily on first reset()
        self._corpus:   Optional[List[Dict]] = None

        # Episode state (set by reset(), mutated by step())
        self._obs:             Optional[Observation]            = None
        self._instance:        Optional[TemplateInstance]       = None
        self._plan_so_far:     List[AtomicStep]                 = []
        self._executor:        Optional[DryRunExecutor]         = None
        self._docker_executor: Optional[DockerEpisodeExecutor]  = None
        self._done:            bool                             = False

    # ------------------------------------------------------------------ public

    def reset(
        self,
        repo_entry: Optional[Dict] = None,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[Observation, Dict]:
        """
        Start a new episode.

        Parameters
        ----------
        repo_entry:
            If provided, use this corpus entry instead of sampling.
            Must have keys ``"repo"`` and ``"category"``.
        seed:
            If provided, reseeds the RNG for this reset (useful for
            reproducible evaluation).

        Returns
        -------
        (observation, info)
        """
        if seed is not None:
            self._rng = random.Random(seed)

        # --- Sample repo ---------------------------------------------------
        if repo_entry is None:
            corpus = self._load_corpus()
            repo_entry = self._rng.choice(corpus)

        # --- Prepare repo (clone + scan in real mode) ---------------------
        manifest, local_path = self._prepare_repo(repo_entry)
        self._local_path = local_path

        # --- Pick template ------------------------------------------------
        template_name = _TEMPLATE_FOR_CATEGORY.get(
            repo_entry.get("category", "manifest_present"),
            "setup_project",
        )
        template = TEMPLATES[template_name]
        self._instance = TemplateInstance(template=template, manifest=manifest)

        # --- Build observation --------------------------------------------
        intent = CanonicalIntent(
            intent_text=_INTENT_TEXTS[template_name],
            template_name=template_name,
            repo_id=repo_entry["repo"],
        )
        self._obs = Observation(
            canonical_intent=intent,
            repo_manifest=manifest,
            episode_steps=[],
            step_index=0,
            max_steps=self._cfg.max_steps,
        )

        # --- Fresh executor for this episode ------------------------------
        # dry_run=True  → DryRunExecutor (no Docker, no subprocess, CI-safe)
        # dry_run=False → DockerEpisodeExecutor (isolated container)
        self._executor = self._make_executor(
            repo_entry=repo_entry,
            manifest=manifest,
            local_path=local_path,
        )

        self._plan_so_far = []
        self._done = False

        info = {
            "repo":     repo_entry["repo"],
            "category": repo_entry.get("category", ""),
            "template": template_name,
        }
        return self._obs, info

    def step(
        self,
        action: PlannedStep,
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Take one step in the episode.

        Parameters
        ----------
        action:
            A ``PlannedStep`` produced by the policy.

        Returns
        -------
        (observation, reward, terminated, truncated, info)
            Standard OpenEnv 5-tuple.

            * ``terminated`` — natural episode end (plan complete or BLOCKED halt).
            * ``truncated``  — episode cut short by max_steps limit.
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done.  Call reset() before stepping again."
            )
        if self._executor is None or self._obs is None or self._instance is None:
            raise RuntimeError("Call reset() before step().")

        # --- Execute action via System 3 Stub -----------------------------
        atomic_steps = expand_planned_step(action)
        
        # All-must-succeed semantics for multi-command expansion
        aggregated_status = ExecutionStatus.SUCCESS
        aggregated_tier = RiskTier.SAFE
        all_verified = True
        final_returncode = 0
        
        for a_step in atomic_steps:
            res = self._executor.execute(a_step)
            # Tier escalates to highest risk
            if res.tier == RiskTier.BLOCKED:
                aggregated_tier = RiskTier.BLOCKED
            elif res.tier == RiskTier.REVIEW and aggregated_tier != RiskTier.BLOCKED:
                aggregated_tier = RiskTier.REVIEW
                
            if res.status != ExecutionStatus.SUCCESS:
                aggregated_status = res.status
                all_verified = False
                final_returncode = res.returncode
                break
                
            if a_step.verify_command and not res.verified:
                all_verified = False
        
        
        final_atomic_step = atomic_steps[-1] if atomic_steps else AtomicStep(command="noop", description="Empty expansion")
        
        exec_result = ExecutionResult(
            step=final_atomic_step,
            status=aggregated_status,
            tier=aggregated_tier,
            verified=all_verified,
            returncode=final_returncode
        )

        # --- Accumulate plan ----------------------------------------------
        self._plan_so_far.append(action)
        self._obs.episode_steps.append(action.description)
        self._obs.step_index += 1

        # --- Check termination conditions ---------------------------------
        blocked = (
            exec_result.tier == RiskTier.BLOCKED
            or exec_result.status == ExecutionStatus.BLOCKED
        )

        # Episode is "complete" when all expected actions verified + not blocked
        expected_actions = self._instance.get_expected_actions()
        all_steps_taken  = len(self._plan_so_far) >= len(expected_actions)
        episode_complete = all_steps_taken and not blocked and exec_result.verified

        terminated = blocked or episode_complete
        truncated  = (not terminated) and (self._obs.step_index >= self._cfg.max_steps)

        # --- Compute reward -----------------------------------------------
        reward, breakdown = compute_reward(
            step=action,
            plan_so_far=self._plan_so_far,
            instance=self._instance,
            execution_result=exec_result,
            episode_complete=episode_complete,
            config=self._cfg.reward_config,
        )

        if terminated or truncated:
            self._done = True

        info = {
            "reward_breakdown": breakdown,
            "exec_status":      exec_result.status.value,
            "exec_tier":        exec_result.tier.value,
            "verified":         exec_result.verified,
            "blocked":          blocked,
            "episode_complete": all_verified,
            "step_index":       self._obs.step_index,
        }

        return self._obs, reward, terminated, truncated, info

    def render(self) -> str:
        """Return a human-readable text representation of the current episode state."""
        if self._obs is None:
            return "[PlannerEnv] No active episode. Call reset() first."
        return self._serializer.to_context_string(self._obs)

    def observation_as_json(self) -> Optional[str]:
        """Convenience accessor for the current observation as a JSON string."""
        if self._obs is None:
            return None
        return self._serializer.to_context_string(self._obs)

    # --------------------------------------------------------------- private

    def _load_corpus(self) -> List[Dict]:
        """Load and cache the corpus split (train or val only)."""
        if self._corpus is not None:
            return self._corpus

        path = self._cfg.corpus_manifest_path
        if not path:
            # Smoke-test mode: use synthetic fixtures
            self._corpus = _SYNTHETIC_CORPUS
            return self._corpus

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Corpus manifest not found: {path}\n"
                "Run `python scripts/build_corpus.py` to generate it."
            )

        with p.open() as f:
            data = json.load(f)

        if "COMPROMISED" in p.name.upper():
            raise ValueError(f"CRITICAL SAFETY ERROR: Attempted to load known-compromised manifest: {p.name}")

        split = self._cfg.split
        if split == "test":
            raise ValueError(
                "PlannerEnv does not expose the 'test' split.  "
                "The held-out test set must only be used via offline evaluation — "
                "never loaded into the training environment."
            )
        if split not in data.get("splits", {}):
            raise KeyError(f"Split '{split}' not found in corpus manifest.")

        self._corpus = data["splits"][split]
        return self._corpus

    def close(self) -> None:
        """
        Release environment resources.

        Stops and removes the running episode container (if any).  Call this
        when the environment is no longer needed — e.g. at the end of a
        training run — to avoid leaving orphaned Docker containers.
        """
        if self._docker_executor is not None:
            self._docker_executor.end_episode()
            self._docker_executor = None

    def _make_executor(
        self,
        repo_entry: Dict,
        manifest: RepoManifest,
        local_path: Optional[str],
    ) -> DryRunExecutor:
        """
        Create the appropriate executor for a new episode.

        ``dry_run=True``  → :class:`DryRunExecutor` (no Docker, no subprocesses).
        ``dry_run=False`` → :class:`DockerEpisodeExecutor` (isolated container).

        For the Docker path, any container from the **previous** episode is
        torn down before the new one is started.
        """
        if self._cfg.dry_run:
            return DryRunExecutor(
                sandbox_root=local_path,
                review_policy=self._cfg.review_policy,
                dry_run=True,
            )

        # Real mode — Docker-backed execution.
        # Tear down the previous episode's container first.
        if self._docker_executor is not None:
            self._docker_executor.end_episode()

        if local_path is None:
            raise RuntimeError(
                "[PlannerEnv] DockerEpisodeExecutor requires a local repo path "
                "but _prepare_repo() returned None.  "
                "This is a bug — real-mode episodes always produce a local_path "
                "via RepoLoader."
            )

        docker_exec = DockerEpisodeExecutor(
            config=self._cfg.docker_config,
            review_policy=self._cfg.review_policy,
        )
        docker_exec.start_episode(
            repo_path=local_path,
            repo_name=repo_entry["repo"],
            ecosystems=list(manifest.ecosystems.keys()),
        )
        self._docker_executor = docker_exec
        return docker_exec

    def _prepare_repo(
        self, repo_entry: Dict
    ) -> Tuple[RepoManifest, Optional[str]]:
        """
        Prepare the episode repo and return ``(manifest, local_path)``.

        In **dry-run** mode: returns a synthetic fixture manifest and
        ``None`` for the local path (no filesystem access needed).

        In **real** mode: clones or resets the repo via :class:`RepoLoader`,
        scans it with ``scan_repo()``, and returns the real manifest together
        with the absolute path to the cloned directory.  The local path is
        passed to :class:`DockerEpisodeExecutor` as the read-only bind-mount
        source.
        """
        category = repo_entry.get("category", "manifest_present")

        if not self._cfg.dry_run:
            from .repo_loader import RepoLoader
            loader = RepoLoader()
            manifest, local_path = loader.load(repo_entry)
            return manifest, str(local_path)

        # Dry-run / smoke-test: return synthetic fixture by category.
        return _make_synthetic_manifest(category), None
