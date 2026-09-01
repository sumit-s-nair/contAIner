# Open Items and Unresolved Gaps

This is an explicit list of components and investigations that remain incomplete before RL training and end-to-end integration can be finalized.

## System 1 (Intent Analysis & NER)
- **9 Uncategorized E2E Mismatches**: The end-to-end strict string-matching evaluation logged 9 software mismatches (`artifacts/results/e2e_mismatches.json`). These must be manually categorized to determine if they are genuine model errors or simply reconstruction/formatting artifacts. System 1 is not fully closed until this is categorized.

## System 2 (RL Environment and Planner)
- **`to_chat_prompt()` Missing Implementation**: The `ObservationSerializer.to_chat_prompt()` method is explicitly marked as `NotImplementedError` in `src/rl_env/observation.py`. It is a hard prerequisite to draft the training script (`scripts/train_system2_grpo.py`) and must be implemented so that the model can receive observations in a flat Qwen2.5 chat template format.
- **AST Fallback Generalization**: The `repo_scan` AST fallback currently checks stdlib membership against the running Python interpreter, making it vulnerable to version mismatches (e.g., Python 2 `urlparse`). It also remains untested on conditional or deferred imports.
- **Live Pipeline Wiring**: The robust `RepoManifest` produced by `repo_scan` is not yet completely wired to live MCP document lookups in the deployed end-to-end pipeline.

## System 3 (Command Generation)
- **Training Completion Status**: The local System 2 (System 3 Command Generation) QLoRA training run was halted at 400 / 28,810 steps before its first evaluation. The local model remains incomplete, and the Electron bridge currently uses a temporary Groq API fallback (`llama-3.3-70b-versatile`) for demo purposes.
- **`NEEDS_REVIEW` Dataset Backlog**: The command-dataset audit was explicitly deferred and the `NEEDS_REVIEW` backlog (unannotatable TLDR-style descriptions) remains open.

## MCP Pipeline
- **Live Network End-to-End Metrics**: The documentation retrieval pipeline's latency and reliability have only been benchmarked in mocked, isolated tests. Live network conditions and end-to-end concurrency behavior remain unmeasured.
