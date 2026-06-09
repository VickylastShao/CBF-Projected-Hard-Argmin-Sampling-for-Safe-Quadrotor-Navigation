# TRM-only / PD / TRM+PD K-sweep diagnostic

task=**narrow**, n_trials=10, n_steps=150, model=cl_trm_model.pt

## Summary table
| method | K | success | TErr mean | TErr median |
|---|---|---|---|---|
| Expert_K1 | 1 | 0/10 | 3.14 | 3.01 |
| TRM_Rollout_K1 | 1 | 0/10 | 7.45 | 7.49 |
| PD_Rollout_K1 | 1 | 0/10 | 4.52 | 4.53 |
| TRM_PD_Rollout_a095_K1 | 1 | 0/10 | 4.52 | 4.52 |
| TRM_Rollout_K5 | 5 | 0/10 | 5.85 | 5.74 |
| PD_Rollout_K5 | 5 | 4/10 | 0.72 | 0.43 |
| TRM_PD_Rollout_a095_K5 | 5 | 6/10 | 0.73 | 0.17 |
| TRM_Rollout_K10 | 10 | 0/10 | 5.49 | 5.26 |
| PD_Rollout_K10 | 10 | 7/10 | 0.39 | 0.20 |
| TRM_PD_Rollout_a095_K10 | 10 | 8/10 | 0.24 | 0.14 |
| TRM_Rollout_K20 | 20 | 0/10 | 5.10 | 5.10 |
| PD_Rollout_K20 | 20 | 7/10 | 0.44 | 0.08 |
| TRM_PD_Rollout_a095_K20 | 20 | 9/10 | 0.16 | 0.12 |