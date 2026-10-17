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

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.repo_scan.models import Dependency, EcosystemManifest, RepoManifest, SourceRef
from src.sandbox.models import AtomicStep, ExecutionResult, ExecutionStatus, RiskTier
from src.system2_planner.models import PlannedStep, TemplateInstance, ActionType
from src.rl_env.system3_stub import expand_planned_step
from src.system2_planner.templates import TEMPLATES

from .docker_executor import DockerEpisodeExecutor, DockerExecutorConfig
from .dry_run_executor import DryRunExecutor, TrainingReviewPolicy
from .observation import CanonicalIntent, Observation, ObservationSerializer
from .reward import (RewardConfig, canonicalize_package, compute_reward, required_packages,
                     satisfied_required)


class EnvForkError(RuntimeError):
    """Raised when environment forking fails or is attempted on an un-reset environment."""
    pass


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


def _rel_manifest_paths(manifest, local_path) -> Dict[str, List[str]]:
    """Stage 2G: manifest files per ecosystem, relative to the repo root
    (= the container working dir /workspace_rw), POSIX style."""
    root = Path(local_path).resolve() if local_path else None
    out: Dict[str, List[str]] = {}
    for eco_name, eco in manifest.ecosystems.items():
        rels = set()
        for f in eco.manifest_files:
            p = Path(f)
            try:
                rels.add(p.resolve().relative_to(root).as_posix() if root else p.name)
            except (ValueError, OSError):
                rels.add(p.name)
        out[eco_name] = sorted(rels)
    return out


# ---------------------------------------------------------------------------
# PlannerEnv
# ---------------------------------------------------------------------------

class PlannerEnv:
    """
    OpenEnv-compatible environment for System 2 planner RL training.

    See module docstring for full lifecycle and design notes.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(self, config: Optional[EnvConfig] = None, expander: Optional[Any] = None) -> None:
        """
        expander:
            System 3.  ``None`` = the deterministic stub
            (``expand_planned_step(action)``).  Otherwise a callable
            ``expander(action, context) -> List[AtomicStep]`` where *context*
            is the ``Sys3Context`` built from this episode
            (``src.system3_command_generation.slot.context_from_env``).
            Stage 3A.
        """
        self._expander = expander
        self._cfg       = config or EnvConfig()
        self._rng       = random.Random(self._cfg.seed)
        self._serializer = ObservationSerializer()

        # Loaded lazily on first reset()
        self._corpus:   Optional[List[Dict]] = None

        # Episode state (set by reset(), mutated by step())
        self._obs:             Optional[Observation]            = None
        self._instance:        Optional[TemplateInstance]       = None
        self._plan_so_far:     List[PlannedStep]                = []
        self._execution_history: List[ExecutionResult]          = []
        self._executor:        Optional[DryRunExecutor]         = None
        self._docker_executor: Optional[DockerEpisodeExecutor]  = None
        self._done:            bool                             = False
        self._repo_entry:      Optional[Dict]                   = None
        self._local_path:      Optional[str]                    = None
        # What each planned step expanded to (System 3 output), parallel to
        # _plan_so_far / _execution_history.  System 3 context reads it.
        self._expanded_history: List[List[AtomicStep]]          = []

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
        self._repo_entry = repo_entry

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
            manifest_paths=_rel_manifest_paths(manifest, local_path),
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
        self._execution_history = []
        self._expanded_history = []
        self._done = False
        # Stage 2G ground truth: installed packages in the container
        # (name -> version) and the high-water set of required packages already
        # credited.  A required package counts only at a version satisfying one
        # of the repo's declared constraints (satisfied_required).  Packages
        # satisfied at reset (base image) are pre-credited: no free reward.
        self._installed_now = self._query_installed()
        self._credited_required = satisfied_required(self._instance, self._installed_now)

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
        atomic_steps = self._expand(action)
        
        # All-must-succeed semantics for multi-command expansion
        aggregated_status = ExecutionStatus.SUCCESS
        aggregated_tier = RiskTier.SAFE
        all_verified = True
        final_returncode = 0
        last_res = None
        step_tier_override = None  # any atomic step's REVIEW->SAFE override
        
        for a_step in atomic_steps:
            if a_step.command == "__UNMAPPED_ACTION__":
                res = ExecutionResult(
                    step=a_step,
                    status=ExecutionStatus.UNMAPPED,
                    tier=RiskTier.SAFE,
                    verified=False,
                    stderr="Unmapped action type executed."
                )
            else:
                res = self._executor.execute(a_step)
            last_res = res
            res_override = getattr(res, "tier_override", None) or (res.extra or {}).get("tier_override")
            if res_override and step_tier_override is None:
                step_tier_override = res_override
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
            unverified_rcs = getattr(a_step, "unverified_returncodes", frozenset()) or frozenset()
            if res.returncode is not None and res.returncode in unverified_rcs:
                all_verified = False
        
        final_atomic_step = atomic_steps[-1] if atomic_steps else AtomicStep(command="noop", description="Empty expansion")
        final_stdout = last_res.stdout if last_res is not None else ""
        final_stderr = last_res.stderr if last_res is not None else ""
        if final_returncode == 0 and last_res is not None and last_res.returncode is not None:
            final_returncode = last_res.returncode
        final_verify_stdout = last_res.verify_stdout if last_res is not None else ""
        final_verify_returncode = last_res.verify_returncode if last_res is not None else None
        
        exec_result = ExecutionResult(
            step=final_atomic_step,
            status=aggregated_status,
            tier=aggregated_tier,
            verified=all_verified,
            returncode=final_returncode,
            stdout=final_stdout,
            stderr=final_stderr,
            verify_stdout=final_verify_stdout,
            verify_returncode=final_verify_returncode,
            tier_override=step_tier_override,
        )

        # --- Ground truth (Stage 2G) -------------------------------------
        install_progress: Optional[float] = None
        if (exec_result.status == ExecutionStatus.SUCCESS
                and action.action_type in (ActionType.INSTALL, ActionType.ISOLATE)):
            now = self._query_installed()
            if now is not None:
                self._installed_now = now
        if action.action_type == ActionType.INSTALL and self._installed_now is not None:
            required = required_packages(self._instance)
            if required:
                newly = set()
                if exec_result.status == ExecutionStatus.SUCCESS:
                    newly = satisfied_required(self._instance, self._installed_now) - self._credited_required
                    self._credited_required |= newly
                install_progress = len(newly) / len(required)

        # --- Accumulate plan ----------------------------------------------
        self._plan_so_far.append(action)
        outcome = exec_result.status.value
        if exec_result.status == ExecutionStatus.SUCCESS:
            outcome += ", verified" if exec_result.verified else ", not verified"
        # Stage 2H: show the action and target, not only the model's own prose,
        # so the policy can see what it already did (it repeated ISOLATE .venv).
        self._obs.episode_steps.append(
            f"{action.action_type.value.upper()} {action.target}: {action.description} [{outcome}]"
        )
        self._obs.step_index += 1
        self._execution_history.append(exec_result)
        self._expanded_history.append(list(atomic_steps))

        # --- Check termination conditions ---------------------------------
        blocked = (
            exec_result.tier == RiskTier.BLOCKED
            or exec_result.status == ExecutionStatus.BLOCKED
        )

        dep_check: Optional[Tuple[bool, str]] = None
        expected_actions = self._instance.get_expected_actions()
        expected_types = [a.action_type for a in expected_actions]
        
        verified_types = [
            p.action_type
            for p, e in zip(self._plan_so_far, self._execution_history)
            if e.verified and e.status == ExecutionStatus.SUCCESS
        ]
        
        import collections
        expected_counts = collections.Counter(expected_types)
        verified_counts = collections.Counter(verified_types)
        
        episode_complete = True
        for atype, count in expected_counts.items():
            if verified_counts[atype] < count:
                episode_complete = False
                break
                
        if episode_complete and ActionType.INSTALL in expected_types:
            # Stage 2F: shared PEP 508 / PEP 503 canonicalisation (reward.py),
            # so 'requests>=2.0' counts as 'requests'.
            required = required_packages(self._instance)
            if required and self._installed_now is not None:
                # Ground truth: everything required is installed right now, at
                # a version satisfying a declared constraint.
                if not required.issubset(satisfied_required(self._instance, self._installed_now)):
                    episode_complete = False
                else:
                    # ... and the installed set is consistent (pip check).
                    dep_check = self._dependency_check()
                    if dep_check is not None and not dep_check[0]:
                        episode_complete = False
            elif required:
                verified_installs = {
                    canonicalize_package(p.target)
                    for p, e in zip(self._plan_so_far, self._execution_history)
                    if p.action_type == ActionType.INSTALL 
                    and e.verified 
                    and e.status == ExecutionStatus.SUCCESS
                    and p.target
                }
                if not required.issubset(verified_installs):
                    episode_complete = False

        episode_complete = episode_complete and not blocked

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
            history=self._execution_history[:-1],
            episode_done=terminated or truncated,
            install_progress=install_progress,
        )

        if terminated or truncated:
            self._done = True

        info = {
            "reward_breakdown": breakdown,
            "exec_status":      exec_result.status.value,
            "exec_tier":        exec_result.tier.value,
            "verified":         exec_result.verified,
            "blocked":          blocked,
            "episode_complete": episode_complete,
            "step_index":       self._obs.step_index,
            "stdout_tail":      (exec_result.stdout or "")[-2000:],
            "stderr_tail":      (exec_result.stderr or "")[-2000:],
            "returncode":       exec_result.returncode,
        }
        if getattr(exec_result, "tier_override", None):
            info["tier_override"] = exec_result.tier_override
        elif "tier_override" in getattr(exec_result, "extra", {}):
            info["tier_override"] = exec_result.extra["tier_override"]
        if install_progress is not None:
            info["install_progress"] = install_progress
        if dep_check is not None:
            info["dependency_check_ok"] = dep_check[0]
            if not dep_check[0]:
                info["dependency_check_output"] = dep_check[1]
        if self._installed_now is not None:
            req = required_packages(self._instance)
            info["required_installed"] = len(satisfied_required(self._instance, self._installed_now))
            info["required_total"] = len(req)
        if exec_result.verify_stdout:
            info["verify_stdout_tail"] = exec_result.verify_stdout[-2000:]
        if exec_result.verify_returncode is not None:
            info["verify_returncode"] = exec_result.verify_returncode

        return self._obs, reward, terminated, truncated, info

    def _expand(self, action: PlannedStep) -> List[AtomicStep]:
        """System 3: PlannedStep -> AtomicSteps (stub unless an expander is set)."""
        if self._expander is None:
            return expand_planned_step(action)
        from src.system3_command_generation.slot.context import context_from_env
        steps = self._expander(action, context_from_env(self, action))
        return list(steps) if steps else [AtomicStep(command="__UNMAPPED_ACTION__",
                                                     description="System 3 produced no command",
                                                     destructive=False)]

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
        elif self._executor is not None and hasattr(self._executor, "end_episode"):
            self._executor.end_episode()
            self._executor = None

    def _dependency_check(self) -> Optional[Tuple[bool, str]]:
        """``pip check`` via the executor, or None (unsupported / dry run / failure)."""
        fn = getattr(self._executor, "dependency_check", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as exc:  # never let the probe crash a rollout
            print(f"[PlannerEnv] WARNING: dependency_check() failed: {type(exc).__name__}: {exc}")
            return None

    def _query_installed(self) -> Optional[Dict[str, Optional[str]]]:
        """Container ground truth via the executor: canonical name -> version,
        or None (dry run / failure).  Executors exposing only
        installed_packages() (a set) give version None, which satisfies only
        unconstrained requirements."""
        fn = getattr(self._executor, "installed_versions", None)
        name = "installed_versions"
        if fn is None:
            fn = getattr(self._executor, "installed_packages", None)
            name = "installed_packages"
        if fn is None:
            return None
        try:
            got = fn()
        except Exception as exc:  # never let the probe crash a rollout
            print(f"[PlannerEnv] WARNING: {name}() failed: {type(exc).__name__}: {exc}")
            return None
        if got is None:
            return None
        return dict(got) if isinstance(got, dict) else {n: None for n in got}

    def end_episode(self) -> None:
        """Release episode resources (delegates to close())."""
        self.close()

    def fork(self, snapshot_image: Optional[str] = None) -> PlannerEnv:
        """
        Create a new, independent PlannerEnv branching from the current state.

        Parameters
        ----------
        snapshot_image:
            If provided, starts the new environment from this pre-committed
            Docker snapshot image without re-running past steps.
            If None, starts from the base state and replays self._plan_so_far.
        """
        # Detect if called on an un-reset env via _instance being None
        if self._instance is None:
            raise EnvForkError("Cannot fork an environment that has not been reset(). Call reset() first.")

        # If self._repo_entry is None, raise EnvForkError
        if self._repo_entry is None:
            raise EnvForkError("Cannot fork an environment with _repo_entry=None.")

        # Cannot fork if episode is already done
        if self._done:
            raise EnvForkError("Cannot fork an environment that has finished (_done=True).")

        # 1. New PlannerEnv using the exact same config object
        forked = PlannerEnv(config=self._cfg, expander=self._expander)

        # 2. Deep copies of mutable state
        forked._instance = copy.deepcopy(self._instance)
        forked._obs = copy.deepcopy(self._obs)
        forked._plan_so_far = copy.deepcopy(self._plan_so_far)
        forked._execution_history = copy.deepcopy(self._execution_history)
        forked._expanded_history = copy.deepcopy(getattr(self, "_expanded_history", []))
        forked._local_path = copy.deepcopy(self._local_path)
        forked._done = copy.deepcopy(self._done)
        forked._repo_entry = copy.deepcopy(self._repo_entry)
        forked._installed_now = copy.deepcopy(getattr(self, "_installed_now", None))
        forked._credited_required = copy.deepcopy(getattr(self, "_credited_required", set()))

        repo_entry = forked._repo_entry
        manifest = forked._instance.manifest
        local_path = forked._local_path

        # 3. Create executor via the same factory reset() uses, with cleanup guard
        try:
            if snapshot_image is not None:
                # Starts from snapshot_image. No re-clone, no replay.
                forked._executor = forked._make_executor(
                    repo_entry=repo_entry,
                    manifest=manifest,
                    local_path=local_path,
                    snapshot_image=snapshot_image,
                )
            else:
                # Starts fresh from base state and replays parent's _plan_so_far
                forked._executor = forked._make_executor(
                    repo_entry=repo_entry,
                    manifest=manifest,
                    local_path=local_path,
                )
                parent_plan = copy.deepcopy(self._plan_so_far)
                parent_tuples = [
                    (e.status, e.verified, e.tier) for e in self._execution_history
                ]

                # Clear plan & history on the fork before replaying through step()
                forked._plan_so_far = []
                forked._execution_history = []
                forked._expanded_history = []
                forked._obs.episode_steps = []
                forked._obs.step_index = 0
                forked._done = False
                forked._installed_now = forked._query_installed()
                forked._credited_required = satisfied_required(forked._instance, forked._installed_now)

                for past_step in parent_plan:
                    forked.step(past_step)

                replayed_tuples = [
                    (e.status, e.verified, e.tier) for e in forked._execution_history
                ]

                if len(replayed_tuples) != len(parent_tuples):
                    raise RuntimeError(
                        f"Replayed execution history length mismatch: parent={len(parent_tuples)}, replayed={len(replayed_tuples)}"
                    )

                for step_idx, (p_tup, r_tup) in enumerate(zip(parent_tuples, replayed_tuples)):
                    if p_tup != r_tup:
                        raise RuntimeError(
                            f"Replayed execution history does not match parent history at step {step_idx}: "
                            f"parent=(status={p_tup[0]}, verified={p_tup[1]}, tier={p_tup[2]}), "
                            f"replayed=(status={r_tup[0]}, verified={r_tup[1]}, tier={r_tup[2]})"
                        )
        except Exception as exc:
            try:
                forked.end_episode()
            except Exception:
                pass
            raise EnvForkError(f"Failed to fork environment: {exc}") from exc

        return forked

    def _make_executor(
        self,
        repo_entry: Dict,
        manifest: RepoManifest,
        local_path: Optional[str],
        *,
        snapshot_image: Optional[str] = None,
    ) -> Union[DryRunExecutor, DockerEpisodeExecutor]:
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
            self._docker_executor = None

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
            snapshot_image=snapshot_image,
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
