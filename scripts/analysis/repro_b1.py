import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_system2_grpo import setup_environment

# 1. Call setup_environment() exactly as evaluate_completion does (line 415)
env = setup_environment()

# 2. Print type(env._docker_executor) and repr(env._docker_executor) BEFORE reset()
print("type(env._docker_executor):", type(env._docker_executor))
print("repr(env._docker_executor):", repr(env._docker_executor))

# 3. Make the same start_episode(...) call the snapshot branch makes (lines 425-429),
# NOT wrapped in try/except, so the full traceback prints.
# In evaluate_completion:
# snap = current_snapshot_tag
# env_clone._docker_executor.start_episode(
#     repo_path=env._local_path,
#     repo_name=repo_entry["repo"],
#     snapshot_image=snap,
# )
snap = "grpo-snapshot-test"
repo_path = "/workspace/dummy_repo"
repo_name = "test/dummy_repo"

env._docker_executor.start_episode(
    repo_path=repo_path,
    repo_name=repo_name,
    snapshot_image=snap,
)
