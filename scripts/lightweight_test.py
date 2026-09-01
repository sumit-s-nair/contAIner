import logging
import json
import random
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RunState:
    def __init__(self):
        self.episode_idx = 0
        self.step_idx = 0
        self.corpus_cursor = 0
        self.global_step = 0

    def state_dict(self):
        return {
            "episode_idx": self.episode_idx,
            "step_idx": self.step_idx,
            "corpus_cursor": self.corpus_cursor,
            "global_step": self.global_step,
        }

    def load_state_dict(self, sd: dict):
        self.episode_idx = sd.get("episode_idx", 0)
        self.step_idx = sd.get("step_idx", 0)
        self.corpus_cursor = sd.get("corpus_cursor", 0)
        self.global_step = sd.get("global_step", 0)

def save_checkpoint(output_dir, run_state):
    ckpt_dir = output_dir / f"checkpoint-{run_state.episode_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "run_state.json", "w") as f:
        json.dump(run_state.state_dict(), f, indent=2)
    logger.info(f"Checkpoint saved to {ckpt_dir}")

def load_checkpoint(ckpt_dir):
    logger.info(f"Resuming from checkpoint {ckpt_dir}")
    run_state = RunState()
    state_path = ckpt_dir / "run_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            run_state.load_state_dict(json.load(f))
    return run_state

def _run_dry_run_test():
    logger.info("Running dry-run checkpoint test...")
    state = RunState()
    state.episode_idx = 42
    state.corpus_cursor = 100
    
    tmp_dir = Path("outputs/dry_run_test")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    save_checkpoint(tmp_dir, state)
    
    logger.info(f"BEFORE: episode_idx={state.episode_idx}, corpus_cursor={state.corpus_cursor}")

    new_state = load_checkpoint(tmp_dir / "checkpoint-42")
    
    logger.info(f"AFTER: episode_idx={new_state.episode_idx}, corpus_cursor={new_state.corpus_cursor}")

    assert new_state.episode_idx == 42
    assert new_state.corpus_cursor == 100
    logger.info("Dry-run checkpoint test passed.")

if __name__ == "__main__":
    _run_dry_run_test()
