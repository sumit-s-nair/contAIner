# contAIner Project Status Report — 2026-08-18

This checkpoint represents the "where are we really" state of the project, covering the full arc of recent work across System 1 NER validation, the new repo scanner, and the MCP document compression/adapter cleanup.

## 1. Architecture Snapshot

contAIner is a two-stage local command-generation pipeline, currently bridged by an Electron UI for demonstration purposes.

- **System 1 (Intent & Entity Understanding)**: DistilRoBERTa joint intent classification + NER. **[Functionally Complete]**
- **Repo Scanner (Context Retrieval)**: Local directory parsing to infer dependency context, falling back to AST static analysis for manifest-less codebases. **[Functionally Complete, unintegrated]**
- **System 2 (Context-Grounded Command Generation)**: Qwen2.5-Coder-1.5B (QLoRA) generation. **[Incomplete/Blocked]**
  - **MCP Document Retrieval**: Fetches live tool documentation based on inferred intent to inject into the System 2 prompt. **[Functionally Complete]**
  - **Fallback Generation**: Hosted Groq Llama-3.3-70b inference is temporarily in place of local System 2.

## 2. Progress Since Last Checkpoint

### System 1: NER Bug Fixed and Validated End-to-End
- Diagnosed the contradictory NER scores (1.00 vs 0.1355 F1). Discovered the root cause was a substring-search bug in the label construction path for entities with diverging surface vs. canonical forms (e.g. "node" vs "nodejs").
- Re-annotated the training data using an offset-based builder. Evaluated across 5 random seeds (variance is negligible: Intent Accuracy 0.9953 ± 0.0010, NER F1 0.9973 ± 0.0006).
- Added a permanent `e2e_entity_check.py` to test the actual served `System1Predictor` path, confirming 98.91% exact string match on the `software` entity class (820/829). 

### Context: Repo Scanner & AST Import Inference Added
- Built `src/repo_scan` to deterministically parse `package.json` and `pyproject.toml` files, providing source line traceability (`SourceRef`).
- Implemented an AST-based static import inference fallback (`import_scan.py`) for Python scripts lacking manifests. Validated against 3 real public repos and a PyPI mapping spot-check. 

### Context: MCP Compression Pipeline Audited
- Completed a full comparative evaluation of extractive vs abstractive compression for retrieved tool documentation across 18 fixtures.
- Found that abstractive (Qwen-0.5B) hallucinated flags in 13.6% of fixtures, making it unsafe for CLI command grounding.
- Standardized on the extractive pipeline (regex-based sentence scoring), which achieved 100% flag preservation while reliably discarding 20-30% of irrelevant prose.

### Context: MCP Adapter Cleanup and Test Coverage
- Removed the unused and bloated `package_metadata` field from `DocChunk` across all 9 adapters.
- Discovered that the bulk edit broke 7 of 9 adapters because they still executed an `asyncio.gather` for the registry fetch, wasting a full HTTP round-trip per request. The breakage went undetected due to 0% adapter test coverage.
- Wrote an 11-test suite with mocked HTTP to enforce the `adapter -> DocChunk` contract and verify edge cases (like Maven string splitting).
- Benchmarked post-fix adapter latency: pure adapter logic is now under 1.5ms overhead, with an inferred live-network saving of 1 HTTP RTT (~150-800ms) per fetch.

## 3. Current Blockers & Unfinished Work

Be explicit: what is *not* done.

1. **System 2 Local Training**: The Qwen2.5-Coder-1.5B local model training is still incomplete (stopped at 400/28,810 steps previously). The Electron bridge continues to rely on the Groq API fallback.
2. **Command Dataset Verification**: The surface/canonical divergence bug that affected System 1 NER may also affect the System 2 command-generation training data. A separate audit is required before resuming training.
3. **Repo Scanner Integration**: `RepoManifest` and the import scanner are built and unit-tested but are *not yet wired into* the MCP lookups or the main Electron application pipeline.
4. **System 1 Mismatches**: The 9 exact-match failures (out of 829) from the System 1 end-to-end check are still uncategorized (model error vs. formatting artifact).

## 4. Novelty Log (Patent / Paper Candidates)

1. **AST-based Declaration-Free Dependency Inference**: Extracting dependency context via static import analysis with confidence tagging (mapped vs. unmapped guess). Differentiates contAIner from manifest-dependent tools (Devbox, Codespaces).
2. **Evaluation Stratification Methodology**: The diagnostic batch technique of stratifying by surface/canonical divergence to isolate label engineering bugs from model architecture capabilities.
3. **Served-Pipeline Validation vs Token F1**: Demonstrating that high token-level F1 is insufficient for live validation, and employing a strict exact-match check on the actual production inference class to catch subword-join and schema issues.

## 5. Results & Metrics Checkpoint

- **System 1 Intent Accuracy**: `0.9953 ± 0.0010` (5 seeds)
- **System 1 NER F1**: `0.9973 ± 0.0006` (5 seeds)
- **System 1 End-to-End Exact Match (`software`)**: `98.91%` (820/829)
- **Repo Scanner Unit Tests**: 95/95 passing
- **Extractive Doc Compression Safety**: `100%` flag preservation across 18 fixtures
- **MCP Adapter Tests**: 11/11 passing (covering all 9 adapters)
- **MCP Adapter Pure Logic Latency**: `<1.5ms` per adapter

## 6. Open Questions / Next Steps

1. **Are the 9 System 1 mismatches genuine errors or string artifacts?** We need to categorize `artifacts/results/e2e_mismatches.json` to truly close out System 1.
2. **Does the command dataset suffer from the same labelling bugs?** Audit the command dataset before attempting to train System 2 again.
3. **How do we handle conditional imports in the repo scanner?** The AST scanner assumes unconditional top-level imports. Do we expand it to function-scoped/conditional imports?
4. **When do we integrate the repo scanner?** The scanner is isolated. It needs to be wired into the prompt generation phase of System 2.
