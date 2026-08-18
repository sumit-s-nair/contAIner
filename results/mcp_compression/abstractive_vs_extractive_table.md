# Abstractive vs Extractive Comparison

Real Abstractive Inference (Qwen2.5-0.5B-Instruct, cuda:0). Max_new_tokens=256, greedy decoding.

### Per-Fixture Results

| fixture | adp | orig | ext_p | ext% | abs_p | abs% | e_flags | e_vers | **a_flags** | **a_vers** | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pip_install_prose_heavy | pip | 126 | 110 | 12.7% | 92 | 27.0% | OK | OK | **FAIL** | OK | 6335 |
| pip_show_version_dense | pip | 56 | 40 | 28.6% | 59 | -5.4% | OK | OK | OK | OK | 3547 |
| pip_freeze_plain | pip | 44 | 29 | 34.1% | 39 | 11.4% | OK | OK | OK | OK | 2072 |
| npm_install_mixed | npm | 99 | 81 | 18.2% | 79 | 20.2% | OK | OK | **FAIL** | OK | 5001 |
| npm_update_flags | npm | 67 | 67 | 0.0% | 55 | 17.9% | OK | OK | OK | OK | 2891 |
| docker_run_command_heavy | docker | 85 | 85 | 0.0% | 72 | 15.3% | OK | OK | OK | OK | 4872 |
| docker_build_flags | docker | 71 | 71 | 0.0% | 70 | 1.4% | OK | OK | OK | OK | 4813 |
| apt_install_flag_heavy | apt | 99 | 99 | 0.0% | 96 | 3.0% | OK | OK | OK | OK | 6666 |
| apt_remove_short | apt | 56 | 44 | 21.4% | 56 | 0.0% | OK | OK | OK | OK | 3511 |
| brew_info_version_dense | brew | 72 | 41 | 43.1% | 56 | 22.2% | OK | OK | OK | OK | 3408 |
| brew_install_options | brew | 72 | 72 | 0.0% | 73 | -1.4% | OK | OK | OK | OK | 4209 |
| cargo_build_short_prose | cargo | 56 | 56 | 0.0% | 46 | 17.9% | OK | OK | OK | OK | 2945 |
| cargo_test_flags | cargo | 65 | 65 | 0.0% | 61 | 6.2% | OK | OK | OK | OK | 3756 |
| go_get_url_like | go | 85 | 68 | 20.0% | 83 | 2.4% | OK | OK | OK | OK | 6299 |
| conda_create_env_flags | conda | 82 | 82 | 0.0% | 80 | 2.4% | OK | OK | OK | OK | 5420 |
| maven_add_xml_code | maven | 90 | 73 | 18.9% | 79 | 12.2% | OK | OK | **FAIL** | OK | 4572 |
| all_code_no_prose | pip | 0 | 0 | 0.0% | 0 | 0.0% | OK | OK | OK | OK | 0 |
| all_prose_no_code | pip | 71 | 30 | 57.7% | 56 | 21.1% | OK | OK | OK | OK | 2991 |

### Aggregate

| metric | extractive | abstractive |
|---|---|---|
| Avg prose reduction | **15.0%** | 10.2% |
| Flag preservation | 18/18 (100%) | **15/18 (83.3%)** |
| Version preservation | 18/18 (100%) | 18/18 (100%) |
| Avg latency | ~0ms | **4,073 ms/fixture** |
| Deterministic | Yes | Yes (greedy) |
| Model weight on disk | 0 MB | 988 MB |
