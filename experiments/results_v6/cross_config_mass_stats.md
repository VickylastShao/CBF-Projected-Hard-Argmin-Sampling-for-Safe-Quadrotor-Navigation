# Probes A statistical comparison

input: `experiments/results_v6/cross_config_mass_n40.json`  n_mc per cell = 40

## R1_s5 vs PD

| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |
|---|---|---|---|---|---|---|---|
| 10 | 31/40 (77.5%) | 0/40 (0.0%) | (31,0) | **0.000** | +nan [+nan,+nan] | +nan [+nan,+nan] | 0 |

## Notes
- ΔX = mean(A) - mean(B) on paired both-success trials. Negative ΔTErr/ΔIAE = A 更好.
- McNemar 双侧精确检验; p<0.05 显著（粗体）。
- Wilson 95% CI for individual rates.