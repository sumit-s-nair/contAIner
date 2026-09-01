# Model Context Protocol (MCP) Pipeline

## Adapter Architecture
The MCP documentation retrieval pipeline uses a three-tier architecture (`fetch -> compress -> cache`). 
It relies on **9 distinct adapters** responsible for fetching remote package manager and registry documentation based on the inferred operation.

## The `package_metadata` Removal Incident
During cleanup, the unused `package_metadata` field was removed from the `DocChunk` object and all 9 adapters.
- **Incident**: The bulk removal silently broke 7 out of 9 adapters. 
- **Root Cause**: The adapters were calling `asyncio.gather(registry_fetch, docs_fetch)` and discarding the registry result without `package_metadata`, which wasted an entire HTTP round-trip per request and caused structural failure. This regression survived because the adapter boundary had **zero test coverage**.
- **The Fix**: Removed the unnecessary registry fetch calls, recovering ~1 fewer HTTP RTT (~150-800ms saved). Added an 11-test suite (`tests/test_adapters.py`) using mocked HTTP to strictly enforce the `adapter -> DocChunk` contract across all 9 adapters, including nuanced string-parsing verification (e.g., Maven's colon-split and bare-name fallbacks). Post-fix, pure adapter logic overhead is benchmarked at sub-millisecond to ~1.4ms per adapter.

## Compression
To manage context window bloat, retrieved documentation chunks are compressed before injection:
- **Segmentation**: Content is segmented into `CODE` vs. `PROSE` blocks.
- **Extractive Rules**: Unnecessary flags, filler text, and boilerplate are stripped out using rule-based heuristics.
- **Abstractive Rejection**: Early experiments tested an abstractive LLM compression step. However, it only achieved a 15.0% reduction compared to the 10.2% reduction of the extractive rules alone, while taking ~4 seconds per fixture latency and suffering 3/18 flag-preservation failures (accidentally stripping critical command flags). The abstractive step was rejected in favor of the purely extractive rules.

## Per-Adapter Density Findings
Analysis of documentation density by registry:
- **Dense Registries**: Docker, Cargo, Conda (typically provide direct, command-focused docs).
- **Compressible Registries**: Pip, Brew, Go, Maven (typically contain verbose prose, tutorial sections, and high boilerplate ratios).

## Concrete Example: Compression
Here is a real `DocChunk` before and after extractive compression:

**Before Compression**:
```text
The pip install command is used to install packages from the Python Package Index.
Use the --no-deps flag to ensure that package dependencies are not installed.
This is highly recommended if you are managing dependencies manually.
```

**After Extractive Compression**:
```text
pip install
--no-deps: ensure package dependencies are not installed.
```
*(Notice the critical `--no-deps` flag is preserved, while the verbose filler sentence at the end is completely dropped).*
