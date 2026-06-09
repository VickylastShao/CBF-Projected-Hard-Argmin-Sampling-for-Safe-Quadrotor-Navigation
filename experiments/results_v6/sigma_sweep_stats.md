# Probes A statistical comparison

input: `experiments/results_v6/sigma_sweep_n40.json`  n_mc per cell = 40

## R1_s1 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 30/40 (75.0%) | 27/40 (67.5%) | (5,2) | 0.453 | -0.043 [-0.070,-0.014] | -0.430 [-0.649,-0.227] | 25 |

## R1_s2 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 36/40 (90.0%) | 27/40 (67.5%) | (9,0) | **0.004** | -0.065 [-0.092,-0.040] | -0.874 [-1.071,-0.672] | 27 |

## R1_s3 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 34/40 (85.0%) | 27/40 (67.5%) | (9,2) | 0.065 | -0.065 [-0.094,-0.039] | -1.169 [-1.399,-0.947] | 25 |

## R1_s5 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 40/40 (100.0%) | 27/40 (67.5%) | (13,0) | **0.000** | -0.094 [-0.125,-0.065] | -2.023 [-2.233,-1.809] | 27 |

## R1_s8 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 40/40 (100.0%) | 27/40 (67.5%) | (13,0) | **0.000** | -0.094 [-0.126,-0.063] | -2.689 [-2.900,-2.465] | 27 |

## Notes
- ΔX = mean(A) - mean(B) on paired both-success trials. Negative ΔTErr/ΔIAE = A 更好.
- McNemar 双侧精确检验; p<0.05 显著（粗体）。
- Wilson 95% CI for individual rates.