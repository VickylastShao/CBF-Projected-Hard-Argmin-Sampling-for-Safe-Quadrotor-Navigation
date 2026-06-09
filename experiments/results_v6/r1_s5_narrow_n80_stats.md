# Probes A statistical comparison

input: `experiments/results_v6/r1_s5_narrow_n80.json`  n_mc per cell = 80

## R1_s5 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 5 | 70/80 (87.5%) | 46/80 (57.5%) | (27,3) | **0.000** | -0.095 [-0.121,-0.068] | -2.055 [-2.277,-1.817] | 43 |
| 10 | 79/80 (98.8%) | 53/80 (66.2%) | (27,1) | **0.000** | -0.081 [-0.101,-0.061] | -1.928 [-2.149,-1.700] | 52 |
| 20 | 77/80 (96.2%) | 59/80 (73.8%) | (19,1) | **0.000** | -0.062 [-0.078,-0.049] | -1.979 [-2.167,-1.800] | 58 |

## Notes
- ΔX = mean(A) - mean(B) on paired both-success trials. Negative ΔTErr/ΔIAE = A 更好.
- McNemar 双侧精确检验; p<0.05 显著（粗体）。
- Wilson 95% CI for individual rates.