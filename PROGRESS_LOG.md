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

## [2026-08-07] System 1 robustly validated across seeds; served-pipeline error rate quantified but not yet categorized

**Question/Hypothesis:** Is the high NER performance (1.00 F1) robust across random initialization, or a fluke of a lucky seed? Does eval-script agreement guarantee the live served pipeline produces correct output?

**Method:** Ran a 5-seed repeated training variance study (seeds: 42, 100, 888, 1234, 2026), reporting mean±std for intent accuracy and per-entity NER F1. Separately, built a permanent end-to-end check (`tools/e2e_entity_check.py`) that runs the full test set through the actual live `System1Predictor` path used by the Electron bridge, and compares exact-string entity output against ground truth — a stricter, different check than token-level F1, designed specifically to catch bugs invisible to it (as happened previously with subword-join spacing and a stale fallback label schema).

**Result:** Seed variance is negligible: Intent Accuracy 0.9953 ± 0.0010, NER F1 (overall) 0.9973 ± 0.0006; `version`/`project`/`file` at 1.0000 ± 0.0000, `software` (highest-support, most linguistically diverse class) at 0.9968 ± 0.0007. End-to-end exact-match: software 820/829 (98.91%), project/version/file 100%. The 9 software mismatches are logged in `artifacts/results/e2e_mismatches.json` but have not yet been categorized as genuine model errors vs. reconstruction/formatting artifacts.

**Interpretation:** The architecture (DistilRoBERTa joint intent+NER) is stable and not seed-sensitive. The end-to-end check did its job — it's the right tool for this kind of validation regardless of what the 9 mismatches turn out to be — but the result should not yet be read as "100% validated end-to-end"; a residual, uncategorized error exists.

**Novelty/contribution relevance:** Repeated-seed variance reporting plus a dedicated served-pipeline exact-match check (distinct from and stricter than aggregate F1) is a reusable validation methodology worth stating explicitly — the earlier NER bug demonstrated that eval-script agreement alone is insufficient, and this closes that specific gap with a permanent, re-runnable check rather than one-off manual testing.

**Threats to validity:** The 9 software mismatches are unclassified — until categorized, the "98.91%" figure can't be split into "acceptable residual model error" vs. "a fixable bug," and shouldn't be reported as a finished number.

**Open question:** Are the 9 mismatches genuine model errors or artifacts? Categorize before treating System 1 as fully closed.

## [2026-08-07] Repo scanner extended with AST-based import inference for manifest-less Python repos

**Question/Hypothesis:** Can dependency context be recovered deterministically from manifest-backed repos (package.json, pyproject.toml) and, as a fallback, inferred for manifest-less Python codebases via static import analysis?

**Method:** Implemented `src/repo_scan/` with deterministic manifest/lockfile parsing (Python, Node) plus line-level source traceability (`SourceRef`). Added `import_scan.py` as a Python-only fallback: AST-based extraction of top-level imports, filtered against stdlib and local modules/packages, mapped to PyPI package names via a ~200-entry table where known. Validated via 95 unit tests against hand-built fixtures, then separately against 3 real public no-manifest Python repos, plus a 15-entry random spot-check of the mapping table against the live PyPI API.

**Result:** All 95 unit tests pass. Real-repo validation: correctly inferred and mapped dependencies (e.g. `cv2` → `opencv-python`) across all 3 repos; correctly filtered stdlib and local imports; one false positive found (`urlparse`, valid stdlib in Python 2, flagged as external when scanned under a Python 3 interpreter). 15/15 randomly sampled mapping-table entries (of ~200 total) confirmed valid against PyPI — this is a spot-check of the sample, not a full-table validation. Several `unmapped_guess` entries in real repos (`tweepy`, `twitter`, `hurry`, `wand`) happened to resolve correctly because import name equals package name — mapped-entry accuracy was validated, unmapped-guess accuracy was not separately measured.

**Interpretation:** The AST fallback scanner is functionally correct for the cases exercised and provides dependency-context recovery for exactly the class of repo (informal/legacy scripts with no manifest) that manifest-only tools cannot handle at all.

**Novelty/contribution relevance:** Declaration-free dependency inference via static import analysis, with source-line traceability and explicit confidence tagging (mapped vs. unmapped guess), differentiates this from every manifest-dependent onboarding tool (Nix, Devbox, Codespaces) and is a concrete candidate for the patent's novelty claims.

**Threats to validity:** Stdlib-membership accuracy is dependent on the scanning interpreter's Python version matching the target repo's era (confirmed false positive on Python 2 code). The "no false negatives" claim is scoped to top-level, unconditional imports across the 3 repos tested — conditional imports (`try/except ImportError`), function-scoped imports, and star imports were not present in these repos and remain unverified by this test. Unmapped-guess accuracy on real code has not been separately measured.

**Open question:** Does accuracy hold on conditional/deferred imports and a larger, more diverse repo sample? Should stdlib membership be checked against a bundled versioned list rather than the running interpreter's, to remove the version-dependency limitation?

## [2026-08-18] MCP adapter cleanup, test coverage, and latency benchmarking

**Question/Hypothesis:** Can we cleanly remove the unused `package_metadata` field from the MCP `DocChunk` to reduce token overhead and context window bloat, without breaking the adapter pipeline?

**Method:** Applied a bulk removal of the `package_metadata` field from `DocChunk` and all 9 adapter implementations. After the bulk edit, audited the adapters to discover that 7/9 were broken because they still called `asyncio.gather(registry_fetch, docs_fetch)` and discarded the registry result, wasting an entire HTTP round-trip per request. The root cause of this breakage surviving was zero adapter test coverage. Wrote an 11-test suite (`tests/test_adapters.py`) with mocked HTTP to enforce the `adapter -> DocChunk` contract across all 9 adapters, including specific string-parsing verification for Maven's colon-split and bare-name fallbacks. Finally, benchmarked adapter latency (post-fix) using the mocked harness.

**Result:** The unused metadata is removed. Test coverage went from 0 to 11 passing tests, fully covering the fetch paths of all 9 adapters. The mocked latency benchmark confirms the pure adapter logic overhead (URL construction, HTML parsing, template filling) is sub-millisecond to ~1.4ms per adapter. 

**Interpretation:** The codebase was fragile due to lack of tests at the adapter boundary, allowing a simple schema change to silently break 78% of the fetching layer. The test suite now guards this. The latency measurements isolate the pure compute cost (which is negligible); the real-world latency improvement from dropping the registry fetch is structurally ~1 fewer HTTP RTT (~150-800ms saved depending on registry), though this live-network delta is inferred rather than directly measured here.

**Novelty/contribution relevance:** Methodological note on the compounding risk of schema changes in unchecked I/O boundaries. The adapter latency measurements also establish a baseline compute ceiling for the documentation retrieval stage.

**Threats to validity:** Live network latency was not measured (all benchmarks used mocked HTTP); the "1 fewer RTT" saving is inferred from the structural change. The test suite mocks the network layer, meaning it cannot catch changes in the upstream registry APIs or HTML structures if they diverge from the mocks.

**Open question:** How does the full documentation retrieval pipeline (adapter fetch + segmentation + compression) perform end-to-end under live network conditions and concurrency?
