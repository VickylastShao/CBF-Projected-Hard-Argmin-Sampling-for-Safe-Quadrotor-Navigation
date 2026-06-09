# TRM-only failure diagnostic

task=**narrow**, n_trials=10, closed-loop steps=150, model=cl_trm_model.pt

## Per-trial summary
| trial | mag(TRM)/mag(exp) | cos_first(TRM,exp) | cos_mean | CL TRM succ | CL TRM TErr | CBF interv. | failure mode |
|---|---|---|---|---|---|---|---|
| 0 | 1.40 | 0.97 | 0.91 | False | 7.070 | 0/150 | other |
| 1 | 1.46 | 0.97 | 0.86 | False | 7.245 | 0/150 | other |
| 2 | 1.46 | 0.97 | 0.91 | False | 8.138 | 0/150 | other |
| 3 | 1.33 | 0.97 | 0.89 | False | 7.695 | 0/150 | other |
| 4 | 1.36 | 0.97 | 0.92 | False | 7.515 | 0/150 | other |
| 5 | 1.48 | 0.97 | 0.91 | False | 6.417 | 0/150 | other |
| 6 | 1.41 | 0.97 | 0.88 | False | 7.455 | 0/150 | other |
| 7 | 1.34 | 0.97 | 0.87 | False | 8.004 | 0/150 | other |
| 8 | 1.42 | 0.97 | 0.94 | False | 7.382 | 0/150 | other |
| 9 | 1.38 | 0.97 | 0.95 | False | 7.551 | 0/150 | other |

## Reference closed-loop (same trials, same CBF)
| trial | PD succ | PD TErr | Expert succ | Expert TErr |
|---|---|---|---|---|
| 0 | False | 4.526 | False | 1.925 |
| 1 | False | 4.491 | False | 2.444 |
| 2 | False | 4.524 | False | 2.539 |
| 3 | False | 4.540 | False | 4.541 |
| 4 | False | 4.514 | False | 2.903 |
| 5 | False | 4.477 | False | 1.654 |
| 6 | False | 4.533 | False | 4.489 |
| 7 | False | 4.514 | False | 3.114 |
| 8 | False | 4.530 | False | 4.508 |
| 9 | False | 4.531 | False | 3.277 |

## Aggregates
- TRM closed-loop success: 0/10
- PD  closed-loop success: 0/10
- Expert closed-loop success: 0/10
- TRM mag ratio mean=1.40 std=0.05
- TRM cos_first mean=0.97 std=0.00

Figures: `experiments/figures_diag/trm_trial_*_{u,traj}.png`