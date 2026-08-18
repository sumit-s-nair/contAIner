# Test Coverage: Before/After Adapter Fix

## Context

On 2026-08-18, `package_metadata` was removed from `DocChunk` as part of
the MCP compression pipeline work (the field was unused downstream, added
~10–40% token overhead to every chunk, and bloated context windows).

The removal was applied as a bulk edit across all 9 adapters simultaneously.
The edit was correct for the model (`DocChunk`), but **7 of 9 adapters had their
`fetch()` method broken** by the change: they were still calling
`asyncio.gather(registry_fetch, docs_fetch)` — now the registry fetch's result
was discarded and the call wasted a full RTT with no consumer.

**Root cause of the breakage surviving undetected:** zero adapter test coverage at
the time of the edit. The only existing test was a schema-level import check.

---

## Before Fix (at commit 6b77c14, prior to test file creation)

| State | Value |
|-------|-------|
| Adapter test files | 0 |
| Tests for adapter fetch() paths | 0 |
| Coverage of adapter → DocChunk contract | 0% |
| Broken adapters (passing wrong args to DocChunk) | 7/9 |
| Adapters returning garbage/crashing | 7/9 |

### Which adapters were broken and how

| Adapter | Breakage type |
|---------|--------------|
| pip_adapter | Still called `asyncio.gather(registry_url, docs_url)` → unused result |
| npm_adapter | Same — registry fetch discarded |
| maven_adapter | Same — registry fetch discarded, `dep_xml` built from stale metadata |
| go_adapter | Same — registry fetch discarded |
| docker_adapter | Same — registry fetch discarded |
| brew_adapter | Same — registry fetch discarded |
| cargo_adapter | Same — registry fetch discarded |
| conda_adapter | ✅ Already only fetched docs (no registry URL in original) |
| apt_adapter | ✅ Already only fetched docs (no registry URL in original) |

---

## After Fix (current state — tests/test_adapters.py added)

| State | Value |
|-------|-------|
| Test file | `tests/test_adapters.py` |
| Total tests | 11 |
| Tests passing | 11/11 |
| Adapters smoke-tested | 9/9 |
| Maven-specific parsing tests | 2 (colon-split and bare-name) |
| HTTP I/O mocked | Yes (no real network calls) |

### Test breakdown

| Test | What it verifies |
|------|-----------------|
| `test_pip_adapter` | fetch() returns DocChunk, calls _fetch_html exactly once, _fetch_json never called |
| `test_npm_adapter` | Same contract |
| `test_go_adapter` | Same contract |
| `test_docker_adapter` | Same contract |
| `test_cargo_adapter` | Same contract |
| `test_conda_adapter` | Same contract |
| `test_brew_adapter` | Same contract |
| `test_apt_adapter` | Same contract |
| `test_maven_adapter` | Same contract |
| `test_maven_adapter_parsing_with_colon` | `com.google.guava:guava` → correct `<groupId>/<artifactId>` split in command_syntax |
| `test_maven_adapter_parsing_without_colon` | `guava` → `<groupId>{groupId}</groupId>` placeholder, `<artifactId>guava</artifactId>` |

### Run command
```
.venv/Scripts/python.exe -m pytest tests/test_adapters.py -v
```

### Output (2026-08-18)
```
collected 11 items
tests/test_adapters.py::test_pip_adapter PASSED                          [  9%]
tests/test_adapters.py::test_npm_adapter PASSED                          [ 18%]
tests/test_adapters.py::test_go_adapter PASSED                           [ 27%]
tests/test_adapters.py::test_docker_adapter PASSED                       [ 36%]
tests/test_adapters.py::test_cargo_adapter PASSED                        [ 45%]
tests/test_adapters.py::test_conda_adapter PASSED                        [ 54%]
tests/test_adapters.py::test_brew_adapter PASSED                         [ 63%]
tests/test_adapters.py::test_apt_adapter PASSED                          [ 72%]
tests/test_adapters.py::test_maven_adapter PASSED                        [ 81%]
tests/test_adapters.py::test_maven_adapter_parsing_with_colon PASSED     [ 90%]
tests/test_adapters.py::test_maven_adapter_parsing_without_colon PASSED  [100%]
======================== 11 passed, 1 warning in 1.57s ========================
```

(The 1 warning is an unrelated FutureWarning from soupsieve about `:contains`.)
