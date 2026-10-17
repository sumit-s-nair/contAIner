
### test

| policy | episodes | completion_rate | mean_steps | mean_steps_completed | mean_reward | unnecessary_isolate_rate | first_step_isolate_rate | first_step_install_rate | parse_error_rate | blocked_steps | env_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 32 | 0.31 | 6.75 | 4.00 | -4.97 | 0.97 | 0.97 | 0.00 | 0.00 | 0 | 2 |
| system2_grpo_run_b45f5d/checkpoint-200 | 33 | 0.33 | 5.67 | 1.00 | -4.73 | 0.03 | 0.06 | 0.94 | 0.00 | 0 | 1 |
| system2_grpo_run_b45f5d/checkpoint-300 | 33 | 0.33 | 5.67 | 1.00 | -4.61 | 0.29 | 0.09 | 0.91 | 0.00 | 0 | 1 |

### val

| policy | episodes | completion_rate | mean_steps | mean_steps_completed | mean_reward | unnecessary_isolate_rate | first_step_isolate_rate | first_step_install_rate | parse_error_rate | blocked_steps | env_errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 6 | 0.67 | 5.50 | 4.25 | -0.56 | 0.80 | 0.83 | 0.00 | 0.00 | 0 | 0 |
| system2_grpo_run_b45f5d/checkpoint-200 | 6 | 0.83 | 2.17 | 1.00 | 6.48 | 0.00 | 0.17 | 0.83 | 0.00 | 0 | 0 |
| system2_grpo_run_b45f5d/checkpoint-300 | 6 | 0.83 | 2.17 | 1.00 | 6.48 | 0.00 | 0.17 | 0.83 | 0.00 | 0 | 0 |
