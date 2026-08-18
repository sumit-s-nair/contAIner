# Extractive Results by Adapter

### Per-Fixture Table

| fixture | adp | orig_prose_tok | cmp_prose_tok | prose_reduc | drop/total_sents | flags | vers |
|---|---|---|---|---|---|---|---|
| pip_install_prose_heavy | pip | 126 | 110 | 12.7% | 1/9 | OK | OK |
| pip_show_version_dense | pip | 56 | 40 | 28.6% | 1/5 | OK | OK |
| pip_freeze_plain | pip | 44 | 29 | 34.1% | 1/3 | OK | OK |
| npm_install_mixed | npm | 99 | 81 | 18.2% | 1/6 | OK | OK |
| npm_update_flags | npm | 67 | 67 | 0.0% | 0/4 | OK | OK |
| docker_run_command_heavy | docker | 85 | 85 | **0.0%** | 0/6 | OK | OK |
| docker_build_flags | docker | 71 | 71 | **0.0%** | 0/5 | OK | OK |
| apt_install_flag_heavy | apt | 99 | 99 | **0.0%** | 0/7 | OK | OK |
| apt_remove_short | apt | 56 | 44 | 21.4% | 1/4 | OK | OK |
| brew_info_version_dense | brew | 72 | 41 | **43.1%** | 2/5 | OK | OK |
| brew_install_options | brew | 72 | 72 | 0.0% | 0/5 | OK | OK |
| cargo_build_short_prose | cargo | 56 | 56 | **0.0%** | 0/4 | OK | OK |
| cargo_test_flags | cargo | 65 | 65 | **0.0%** | 0/5 | OK | OK |
| go_get_url_like | go | 85 | 68 | 20.0% | 1/5 | OK | OK |
| conda_create_env_flags | conda | 82 | 82 | **0.0%** | 0/6 | OK | OK |
| maven_add_xml_code | maven | 90 | 73 | 18.9% | 2/6 | OK | OK |
| all_code_no_prose | pip | 0 | 0 | 0.0% | 0/0 | OK | OK |
| all_prose_no_code | pip | 71 | 30 | **57.7%** | 3/5 | OK | OK |

**Extractive: 100% flag preservation, 100% version preservation, across all 18 fixtures.**

### Per-Adapter Summary

| adapter | avg prose reduc | tok reduc | droppable/total sents | fixtures | density |
|---|---|---|---|---|---|
| **pip** | **26.6%** | **29.6%** | 6/22 | 5 | **VERBOSE** |
| **brew** | **21.5%** | 21.5% | 2/10 | 2 | **VERBOSE** |
| **go** | **20.0%** | 20.0% | 1/5 | 1 | **VERBOSE** |
| **maven** | **18.9%** | 18.9% | 2/6 | 1 | **VERBOSE** |
| apt | 10.7% | 7.7% | 1/11 | 2 | MEDIUM |
| npm | 9.1% | 10.8% | 1/10 | 2 | MEDIUM |
| **docker** | **0.0%** | **0.0%** | 0/11 | 2 | **DENSE** |
| **cargo** | **0.0%** | **0.0%** | 0/9 | 2 | **DENSE** |
| **conda** | **0.0%** | **0.0%** | 0/6 | 1 | **DENSE** |

**Why docker / cargo / conda are perfectly dense:** Every sentence in their fixtures contains a `--flag`, a version number, or a signal word (`required`, `breaking`, `note`, etc.) — the keep-rules fire on every sentence so nothing is droppable. These adapter docs are information-saturated at the sentence level; there is no filler to compress.
