# MCP Adapter Evaluation Results

Results from the MCP adapter cleanup work: `package_metadata` removal,
broken-adapter discovery and fix, new test suite, Maven parsing verification,
and per-adapter latency benchmarking.

**Date:** 2026-08-18  
**Corresponding PROGRESS_LOG.md entry:** `[2026-08-18] MCP adapter cleanup…`

## Files

| File | Contents |
|------|----------|
| `latency_benchmark.md` | Per-adapter wall-clock latency (mocked HTTP), pre-fix vs post-fix, with interpretation |
| `latency_benchmark_postfix.json` | Raw JSON for post-fix benchmark (n=20 per adapter) |
| `test_coverage.md` | Coverage before/after: 0 → 11 tests passing, broken adapter audit |

## Key numbers

- **Pre-fix adapter test coverage:** 0 tests
- **Adapters broken by bulk package_metadata removal:** 7/9
- **Post-fix tests:** 11/11 passing
- **Post-fix adapter logic latency (mocked):** 0.3–1.4 ms mean per adapter
- **Expected real-world improvement:** ~1 fewer HTTP RTT per call (~150–800ms saved)
  — this figure is structural/inferred, not a live-network measurement

## What was tested, what was not

✅ Adapter → DocChunk contract (all 9 adapters)  
✅ Maven groupId/artifactId colon-split parsing  
✅ Maven bare-name fallback (no colon → `{groupId}` placeholder)  
❌ Live network latency (all benchmarks used mocked HTTP)  
❌ Adapter integration with router/server under load  
❌ Compression pipeline latency with real adapter output  
