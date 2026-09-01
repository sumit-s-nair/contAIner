# System 2: RL Environment

## PlannerEnv OpenEnv Contract
The System 2 planner environment (`PlannerEnv`) adheres to the standard OpenEnv interface:
- `obs, info = env.reset()`
- `obs, reward, terminated, truncated, info = env.step(action)`
- `text = env.render()`

## Observation Format
The observation uses a hybrid format combining:
1. Ecosystem summary block (dependency counts, manifest status)
2. Per-dependency `explain()` output (capped at 20 entries)
3. Conflicts block
4. Episode history (prior steps taken)

**`to_context_string()` Example Output**:
```json
{
  "intent": {
    "text": "Set up this repository project for development.",
    "template": "setup_project",
    "repo_id": "owner/repo"
  },
  "ecosystem_summary": [
    {
      "ecosystem": "python",
      "dep_count": 5,
      "manifest_present": true,
      "has_lock": false,
      "inferred_count": 0
    }
  ],
  "dependency_details": [
    "requests: declared as 'requests>=2.0' in requirements.txt:3"
  ],
  "dependency_details_truncated": 0,
  "conflicts": [],
  "episode_history": {
    "step_index": 1,
    "max_steps": 8,
    "steps_taken": ["isolate"]
  }
}
```
**Note on `to_chat_prompt()`**: This method is explicitly marked with `NotImplementedError` and serves as a hard prerequisite guard before any training scripts can be wired up.

## Reward Function
The reward formula balances validation constraints against execution success and plan length:
```text
total = (w_validator  * step_validator_reward)
      + (w_execution  * step_execution_reward)
      + (w_completion * episode_completion_bonus)
      - (w_step_penalty)
```
- **Why `step_penalty`?** (w_step_penalty = 0.1). Unconditionally subtracted every step to discourage plan padding.
- **Why the `verified` flag?** Full credit (`r_success = 1.0`) requires BOTH an exit-0 execution and a successful verification. An exit-0 step that fails verification gets `r_success_unverified = 0.2`, effectively penalizing syntactic success without semantic success.

## The `verify_command` Exploit and Fix
**The Exploit**: This exploit was identified during design review of the reward function specification, before any training or execution occurred. (No training has run yet). If unmitigated, the RL policy could have trivially self-certified by generating an `AtomicStep` with `command="echo nothing"` but `verify_command="true"`. Because `true` always exits with 0, the step would be flagged as `verified=True`, effectively hacking the reward loop to gain full credit (`r_success = 1.0`) and the episode completion bonus (`5.0`) while doing no actual work.
**The Fix (Option A)**: The policy no longer controls its own verification. The `DockerEpisodeExecutor` uses a deterministic lookup table (`src/rl_env/verify_table.py`) keyed on `(action_type, target)` to inject a fixed verify command prior to execution. For example, if the action is `ActionType.INSTALL` and the target ecosystem is `python`, the injected verify command is strictly `python -c "import <guessed_package_name>"`. The policy has no way to override this.

## Docker Executor
The `DockerEpisodeExecutor` enforces a strict container lifecycle:
- **Container Lifecycle**: One `docker run` per episode. Subsequent steps use `docker exec` in the same container. `end_episode()` destroys the container via `docker stop` and `docker rm -f`.
- **Image Selection**: Selected per ecosystem (e.g., `python:3.11-slim`, `node:20-slim`). Mixed ecosystems use combined images.
- **Filesystem**: The repository is bind-mounted read-only at `/workspace` while a writable `tmpfs` is placed at `/workspace_rw` to catch side-effects.
- **Resource Limits**: CPU (`--cpu-quota=25000`), Memory (`--memory=1g`), and Process limits (`--pids-limit=32`) are applied.
- **Network Allowlist**: Fail-closed by default (network = `none`). Allowlist mode resolves specific registry IPs and injects them via `--add-host` over an isolated bridge network.
- **Timeout Handling**: Timeout `TimeoutExpired` triggers `proc.kill()` (SIGKILL) on the host-side `docker exec` process followed by `proc.wait()` to reap the zombie, and then an in-container `kill -9 -1` step that purges surviving child processes in the container's PID namespace so no runaway child survives.

## Training-Time REVIEW Policy
During unattended RL training rollouts, human-in-the-loop confirmation is impossible. The environment enforces an `AUTO_DENY` policy, returning `ABORTED` (with a slight negative reward of `-0.5`) whenever a step is classified as `REVIEW`.

## Corpus Pipeline
- **Fetch/Filter Criteria**: 50 KB ≤ repo size ≤ 50 MB, ≥10 stars, not archived, not a fork, default branch `main/master`, OSI-approved license.
- **The Malware Incident**: During corpus construction via the Windows Defender scan step, one confirmed incident occurred: the repository `HalilDeniz/RansomwareSim` was flagged by Windows Defender as `Trojan:Python/FileCoder.AG!MTB` via a file named `Encoder.py`. This posed a severe risk to the local RL executor. This was resolved with a defense-in-depth fix:
  1. **Keyword Blocklist**: A regex blocklist now drops repos whose metadata contains terms like `ransomware`, `trojan`, or `exploit`.
  2. **Defender Quarantine Scan**: Repos are first cloned to a temporary quarantine folder. We invoke `MpCmdRun.exe -Scan -ScanType 3 -File <quarantine_path>` (Windows Defender). If a threat is detected, the clone is purged and the repo is permanently excluded from the corpus.
- **The `manifest_present` GitHub API Bug**: Relying entirely on GitHub code-search `filename:requirements.txt` quietly missed or falsely categorized repos. The fix ensures a manifest file is concretely present by explicitly cloning the repository and validating with `scan_repo()`.
- **v3 Composition Table**: Stratified dynamically by repository (70/15/15) across three primary categories: `manifest_present`, `manifest_less`, and `known_conflict`.

## Concrete Example: Dry-Run Episode Trace
*Dry Run `setup_project` with `manifest_present`*
1. **Reset**: Samples repo. Produces intent `Set up this repository project for development.`
2. **Action Taken**: `pip install requests`
3. **Reward Breakdown**:
   - `validator`: `1.0` (Valid ordering)
   - `execution`: `1.0` (Synthetic dry-run `SUCCESS` + verified)
   - `completion_bonus`: `5.0` (Episode complete)
   - `step_penalty`: `0.1`
   - **Total Reward**: `6.9`
4. **Termination**: Episode completes (`terminated=True`).
