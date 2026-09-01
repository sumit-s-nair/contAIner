# System 1: Intent Analysis & NER

## Architecture
The System 1 model uses a **DistilRoBERTa joint intent+NER** architecture. It is deployed locally to classify semantic intent and extract entities simultaneously.

## The NER Bug
During the evaluation of the NER module (documented `2026-08-06`), two independent scripts disagreed on the F1 score: `generate_ner_stats.py` reported **13.55%**, while the original evaluation pipeline reported **1.00** (100%).

- **Root Cause**: The label-building logic was deriving BIO tags by searching for the canonical entity value as a exact text substring. This failed silently whenever the surface form diverged from the canonical form (e.g., "node" vs "nodejs"), producing all-O labels for those examples. The original 1.00 score was an artifact of the exact same bug existing in the evaluation path.
- **The Fix**: The label generation was updated to use proper token offsets rather than string-matching the canonical substring. 
- **Impact**: True pre-fix performance was 13.55%. Post-fix F1 improved to **99.84%** (with 1.00 per-entity F1 across all four entity types, including `software`).

## Seed Variance Results
A 5-seed repeated training variance study was conducted to ensure robustness:
- **Seeds Used**: 42, 100, 888, 1234, 2026
- **Intent Accuracy**: 0.9953 ± 0.0010
- **NER F1 (overall)**: 0.9973 ± 0.0006
  - `software`: 0.9968 ± 0.0007 (highest-support class)
  - `project`/`version`/`file`: 1.0000 ± 0.0000

## End-to-End Exact-Match Validation
An end-to-end check (`tools/e2e_entity_check.py`) was run through the live `System1Predictor` path.
- **Results**: Software entities matched 820/829 (**98.91%**). Project/version/file achieved 100%.
- **Bugs Caught**: The exact-string output check revealed prior issues with subword-join spacing and a stale fallback label schema that token-level F1 metrics had masked. The 9 remaining mismatches in software entities are logged in `artifacts/results/e2e_mismatches.json`.

## Concrete Example: Pipeline Walkthrough
Instruction: *"upgrade git to v1.5.1 on my windows machine"*
1. **Intent Classification**: Evaluated through DistilRoBERTa. `Intent: UPDATE` (Confidence: `0.995`)
2. **Entity Extraction**:
   - `[upgrade]` (O)
   - `[git]` (B-software, Confidence `0.998`)
   - `[to]` (O)
   - `[v1.5.1]` (B-version, Confidence `0.999`)
   - `[on my windows machine]` (O)
3. **Final Structured Output**:
   ```json
   {
     "intent": "UPDATE",
     "entities": {
       "software": "git",
       "version": "1.5.1"
     }
   }
   ```
   *(Note: The above numbers are illustrative structural examples as specific JSON output was not captured in the progress log for this query).*
