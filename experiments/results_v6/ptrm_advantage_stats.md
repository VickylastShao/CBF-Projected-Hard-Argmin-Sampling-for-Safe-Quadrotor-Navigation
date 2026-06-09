# PTRM advantage paired statistics

## narrow n=100, alpha_blend=0.95, alt=TRM+PD+Rollout vs base=PD+Rollout

### Paired comparison (narrow n=100)

| K / D | n | Baseline succ | Alt succ | McNemar (b, c, p) | ΔTErr (alt−base) | ΔIAE (alt−base) | n_both |
|---|---|---|---|---|---|---|---|
| K=5, D=16 | 100 | 64.0% [54.2, 72.7] | 73.0% [63.6, 80.7] | (16, 7, p=0.093) | +0.017m [-0.009, +0.040] | -0.049 [-0.097, +0.001] | 57 |
| K=10, D=16 | 100 | 79.0% [70.0, 85.8] | 82.0% [73.3, 88.3] | (10, 7, p=0.629) | +0.019m [-0.007, +0.045] | -0.080 [-0.129, -0.030] * | 72 |

`*` 表示 95% CI 不含 0（i.e. statistically significant at α=0.05）。

### CEM vs PD+Rollout (same trials, n=100)

| K / D | n | Baseline succ | Alt succ | McNemar (b, c, p) | ΔTErr (alt−base) | ΔIAE (alt−base) | n_both |
|---|---|---|---|---|---|---|---|
| K=5 (CEM eff=15) | 100 | 64.0% [54.2, 72.7] | 65.0% [55.3, 73.6] | (12, 11, p=1.000) | +0.073m [+0.044, +0.103] * | -0.077 [-0.138, -0.015] * | 53 |
| K=10 (CEM eff=30) | 100 | 79.0% [70.0, 85.8] | 91.0% [83.8, 95.2] | (16, 4, p=0.012) | +0.006m [-0.023, +0.035] | -0.396 [-0.446, -0.346] * | 75 |

`*` 表示 95% CI 不含 0（i.e. statistically significant at α=0.05）。

## narrow D-sweep n=50, K=10, alpha_blend=0.95, alt=TRM+PD+Rollout vs base=PD+Rollout

### Paired D-sweep

| K / D | n | Baseline succ | Alt succ | McNemar (b, c, p) | ΔTErr (alt−base) | ΔIAE (alt−base) | n_both |
|---|---|---|---|---|---|---|---|
| D=8, K=10 | 50 | 84.0% [71.5, 91.7] | 88.0% [76.2, 94.4] | (5, 3, p=0.727) | +0.032m [+0.006, +0.055] * | -0.056 [-0.113, -0.002] * | 39 |
| D=16, K=10 | 50 | 80.0% [67.0, 88.8] | 84.0% [71.5, 91.7] | (5, 3, p=0.727) | +0.026m [-0.006, +0.055] | -0.082 [-0.155, -0.013] * | 37 |
| D=24, K=10 | 50 | 80.0% [67.0, 88.8] | 92.0% [81.2, 96.8] | (6, 0, p=0.031) | +0.037m [+0.004, +0.068] * | -0.056 [-0.122, +0.009] | 40 |

`*` 表示 95% CI 不含 0（i.e. statistically significant at α=0.05）。
