"""
src/rl_env
==========
OpenEnv-compatible RL training environment for the System 2 planner.

Public surface
--------------
    from src.rl_env import PlannerEnv
    from src.rl_env import DockerEpisodeExecutor, DockerExecutorConfig
    from src.rl_env.observation import Observation, ObservationSerializer
    from src.rl_env.reward import RewardConfig, compute_reward
    from src.rl_env.dry_run_executor import DryRunExecutor, TrainingReviewPolicy
    from src.rl_env.docker_executor import NetworkConfig, DockerNotAvailableError
    from src.rl_env.corpus import build_corpus, load_corpus_manifest
    from src.rl_env.repo_loader import RepoLoader
"""

from .env import PlannerEnv
from .docker_executor import (
    DockerEpisodeExecutor,
    DockerExecutorConfig,
    DockerNotAvailableError,
    NetworkNotConfiguredError,
    NetworkConfig,
    select_image,
)

__all__ = [
    "PlannerEnv",
    "DockerEpisodeExecutor",
    "DockerExecutorConfig",
    "DockerNotAvailableError",
    "NetworkNotConfiguredError",
    "NetworkConfig",
    "select_image",
]
