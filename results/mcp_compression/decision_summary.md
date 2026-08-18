# Decision Summary

**DECISION: Do NOT enable abstractive compression — not by default, not even for verbose adapters. Shipped Extractive only.**

### The numbers that drove it

| factor | value | verdict |
|---|---|---|
| Abstractive avg prose reduction | 10.2% | **WORSE than extractive 15.0%** |
| Abstractive gain over extractive | **-4.8%** | Negative — extractive wins outright |
| Flag preservation failures | **3/18** | Correctness bug, not a tradeoff |
| Avg inference latency | **4,073 ms/fixture** | 4 seconds per segment on CUDA |

### Why Extractive is the right default
Extractive compression achieved **15.0% prose reduction with 100% flag and version preservation**, no GPU, no model weight, sub-millisecond latency, and fully deterministic output. That cost/quality profile dominates abstractive on every axis at this model size.
