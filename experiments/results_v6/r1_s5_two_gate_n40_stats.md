# Probes A statistical comparison

input: `experiments/results_v6/r1_s5_two_gate_n40.json`  n_mc per cell = 40

## R1_s5 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 40/40 (100.0%) | 40/40 (100.0%) | (0,0) | 1.000 | -0.019 [-0.022,-0.015] | -0.277 [-0.329,-0.224] | 40 |

## Notes
- ΔX = mean(A) - mean(B) on paired both-success trials. Negative ΔTErr/ΔIAE = A 更好.
- McNemar 双侧精确检验; p<0.05 显著（粗体）。
- Wilson 95% CI for individual rates.