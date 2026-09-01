# contAIner Pre-Training System Documentation

This folder contains a comprehensive, component-by-component snapshot of the **contAIner** architecture and its implementations as they exist right now, prior to any RL training. 

## Pipeline Overview
The system operates as a four-stage agentic pipeline designed to safely set up, diagnose, and manage software projects. 

1. [System 1: Intent Analysis & NER](01_system1_intent_ner.md) - Analyzes user queries to extract semantic intent and structured entities.
2. [System 2: Repository Scanner](02_repo_scan.md) - Deterministically parses and infers repository state into a canonical `RepoManifest`.
3. [System 2: Planner](05_system2_planner.md) - Employs DAG-based `WorkflowTemplates` and structural validators to map intents to high-level abstract `PlannedSteps`.
4. [MCP Pipeline](03_mcp_pipeline.md) - Retrieves and compresses targeted documentation from remote registries to ground command generation.
5. System 3: Command Generation - Generates concrete `AtomicStep` commands (currently relying on a temporary hosted API). 
6. [System 4: Sandbox Safety Gate](04_sandbox_safety_gate.md) - Enforces a strict three-tier classification (BLOCKED, REVIEW, SAFE) gate prior to execution.
7. [System 2: RL Environment](06_rl_environment.md) - The OpenEnv-compatible wrapper defining the training environment, rewards, and docker-isolated execution lifecycles for future RL training.

## What Exists vs. What's Still Planned
Currently, the pipeline architecture, structural validation, dataset schemas, and isolated RL environment are fully functional and heavily tested. System 1 (Intent Analysis) is stable and robustly evaluated. However, **RL training has not yet been run**. The local System 3 (Command Generation) QLoRA training run was halted early at 400 steps, meaning the live Electron bridge is currently pivoting to a Groq API proxy for demonstration purposes. Furthermore, crucial final-integration pieces, such as wiring the `RepoManifest` to live MCP document lookups and implementing the `to_chat_prompt()` formatting for the System 2 RL environment, remain explicitly [open items](07_open_items.md).
