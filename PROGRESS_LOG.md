## [2026-01-29] Dataset foundations, schemas, validation, and Hub tooling added

**Question/Hypothesis:** How to structure semantic intent versus command targets to cleanly separate understanding from generation?
**Method:** Created schema-validated local JSONL datasets with `.env` loaders and Hugging Face Hub upload utilities, separating intent models from command models by OS/shell.
**Result:** Established normalized intent and command datasets (`20c2f94`, `ea34f1f`, `8baf9f1`), keeping command targets versioned locally.
**Interpretation:** Clean data foundations and validation are in place, unblocking model training.
**Novelty/contribution relevance:** Canonical intent/entity/OS/shell records separate semantic understanding from command targets; novelty uncertain.
**Threats to validity:** No evaluation results or model training yet.
**Open question:** Can this normalized data be effectively used for training both the understanding and generation models?

## [2026-01-31] Initial CodeT5+ command-generation training stack added

**Question/Hypothesis:** Can a code-specialized seq2seq model (CodeT5+) serve as a viable System 2 for structured command generation?
**Method:** Added CodeT5+ command models and a structured training pipeline (`1ce3c14`).
**Result:** Initial training stack established but no completed runs or evaluations yet.
**Interpretation:** A baseline command generation architecture is in place but remains unproven without end-to-end testing.
**Novelty/contribution relevance:** Structured intent-to-command generation conditioned on OS/shell; likely a standard fine-tuning application.
**Threats to validity:** End-to-end inference and execution validation remained open. No committed evaluation artifact.
**Open question:** How well will this seq2seq model perform once integrated with the intent understanding stage?

## [2026-03-22] System 1 intent classification and split datasets established

**Question/Hypothesis:** Can intent understanding be effectively separated from command generation?
**Method:** Added System 1A classification/training code (`0d4aef0`) and updated intent data into formal train/validation/test splits (`3469fb5`).
**Result:** Successfully separated intent datasets and built the classification pipeline, with follow-on training/docs landed (`0799976`, `45997dc`).
**Interpretation:** Establishing a two-stage pipeline (System 1 and System 2) is a viable architectural choice.
**Novelty/contribution relevance:** Joint structured intent/entity representation is an explicit pipeline contract, not a demonstrated novel method.
**Threats to validity:** Entity extraction, runtime orchestration, and System 1→System 2 integration remained open. No committed evaluation numbers at this checkpoint.
**Open question:** Will the joint intent/NER model integrate smoothly with command planning downstream?

## [2026-04-21] MCP documentation grounding and QLoRA-oriented System 2 added

**Question/Hypothesis:** Can live documentation improve command generation compared to static training examples alone?
**Method:** Ground System 2 with live package-manager/registry documentation using an MCP HTTP server, tool adapters, and a TTL cache (`0fc4cd5`). Added Qwen2.5-Coder-1.5B/QLoRA training support.
**Result:** Added MCP documentation fetcher and QLoRA training pipeline to the workflow.
**Interpretation:** Documentation retrieval is successfully integrated into the prompt to enrich generation context.
**Novelty/contribution relevance:** Retrieving package-manager documentation by inferred operation and injecting compact evidence into generation prompts; novelty uncertain.
**Threats to validity:** Local command generation still requires a completed run and dry-run validation. No completed System 2 evaluation committed.
**Open question:** Will stabilizing QLoRA training with retrieved documentation significantly improve command accuracy?

## [2026-04-22] Training robustness, Electron demo, and intent results added

**Question/Hypothesis:** Can we provide a demonstrable desktop surface while model work continues?
**Method:** Improved training behavior (`35d5456`), added an Electron + Flask/SSE bridge (`9295b1d`), and evaluated the System 1 intent classifier (`a488f5e`).
**Result:** Deployed a streaming UI and achieved a 0.9939 test intent accuracy on System 1 (`artifacts/results/intent_classifier_research_stats.json`).
**Interpretation:** The demonstration surface is functional and correctly exposes intent, docs, and generation stages separately.
**Novelty/contribution relevance:** Streaming UI exposes stages separately; engineering differentiation only.
**Threats to validity:** System 2 evaluation/final model and real execution safety remain open.
**Open question:** How will the local System 2 inference perform once fully trained and integrated to replace the demo?

## [2026-07-20] Electron bridge temporarily pivots from unfinished local System 2 to Groq-compatible generation

**Question/Hypothesis:** How can we enable a functional UI demo while the local System 2 Qwen training remains incomplete?
**Method:** Replaced local `CommandGenerationModel` inference with hosted Groq-compatible generation (`llama-3.3-70b-versatile`) in the Electron bridge (`config.py`, `bridge/grok.py`) as a temporary fallback.
**Result:** System 1 intent accuracy remained 0.9939. System 2 QLoRA run stopped at 400/28,810 steps before first eval. Contradictory NER F1 emerged (1.00 vs 0.1355) in `artifacts/results/ner_research_stats.json`.
**Interpretation:** Hosted generation serves as a functional demo path, but local-only deployment remains blocked by incomplete training and the unresolved NER discrepancy.
**Novelty/contribution relevance:** MCP documentation chunks keyed by inferred package-manager operation are injected into command-generation prompts.
**Threats to validity:** Local System 2 has no final model or eval artifact; codebase scanner/onboarding generator missing. The contradictory NER metrics indicate a severe flaw in evaluation.
**Open question:** How do we reconcile the contradictory NER metrics and finalize the local System 2 model for external reporting?

## [2026-08-06] Two independent NER evaluations disagree by 86 points — is the model or the measurement broken?

**Question/Hypothesis:** A previously reported NER F1 of 1.00 seemed implausibly high for a 4-entity BIO tagging task on informal instruction text. A second, independently-implemented eval script (`generate_ner_stats.py`) reported 0.1355. Hypothesis: the label-building code derives BIO tags by searching for the canonical entity value as a text substring, which fails whenever the canonical form doesn't literally appear in the sentence (e.g. "node" -> "nodejs"), silently producing all-O labels for a large class of examples.

**Method:** Constructed a 40-example diagnostic batch stratified by surface-vs-canonical divergence (identical, abbreviation, informal synonym, multi-token, boundary position, multi-entity). Ran both the existing and a hypothesis-driven offset-based label builder against it. Then audited the full 9,769-row intent-dataset the same way.

**Result:** Diagnostic batch confirmed the pattern (old builder: near-zero recall on divergent-surface-form categories; new builder: correct on all categories including the identical-form control). Full-dataset audit found the data itself was 99%+ clean (only 230/9,769 rows had offset errors, 0 needed re-annotation) — the defect was entirely in label construction, not annotation quality. Old logic was shown to silently mislabel ~2,215 rows dataset-wide. After retraining with offset-derived labels, both eval scripts reconciled to ~99.84%/1.00 F1, with per-entity F1 of 1.00 across all four entity types including `software` (823 support, the dominant class).

**Interpretation:** The 13.55% figure, not the 1.00 figure, was the true pre-fix model performance — the original 1.00 was itself an artifact of the same substring-search bug being present in that eval path too, and only ever scoring on the subset of entities (`project`, `version`) that happened to have surface-equal canonical forms. Confirms the model architecture (DistilRoBERTa joint intent+NER) is not the limiting factor; label engineering was.

**Novelty/contribution relevance:** Not a claimed novel technique, but a documented, quantified failure mode in canonical-vs-surface-form entity labeling that's a reusable methodological note — the diagnostic-batch technique (stratifying by surface/canonical divergence to isolate a labeling bug before touching the full dataset) is worth stating explicitly as a validation method in the paper's methodology section.

**Threats to validity:** Single train/eval run post-fix — no repeated-seed variance reported. Fix validated on intent-dataset only; command-dataset (different schema, different entity semantics) was audited separately and explicitly deferred, not yet resolved. Manual/production inference path initially reproduced a related-but-distinct bug (subword-join spacing, stale fallback label schema) not caught by either eval script — indicates eval-script agreement alone does not guarantee correctness end-to-end.

**Open question:** Does the same surface/canonical divergence issue affect System 2's command-generation training data, where entities are consumed as generation targets rather than token labels? Command-dataset audit (separate, deferred) suggests a different failure mode there (unannotatable TLDR-style descriptions) rather than this one, but not yet confirmed.
