# Experiments

This directory contains the scripts that produce every numerical table and figure in the manuscript, plus the raw JSON result files cited in each.

## Authoritative result files

| Manuscript table | JSON file (in `results_v6/`) |
|---|---|
| Table II — narrow $K{=}10$, $N{=}80$ paired McNemar | `r0_vs_r1_paired_narrow_n80_s2026.json` |
| Table III — $\sigma$-sweep | `sigma_sweep_n40.json` |
| Table IV — $+50\%$ mass mismatch | `r0_r1_pd_mass_n40_s2026.json`, `casadi_nmpc_mass_n40_s2026.json` |
| Table V — MPPI temperature sweep | `mppi_lambda_sweep_paired_stats.json` |
| Table VI — CEM / iCEM | `mce_mass_n40_cem_icem.json` |
| Table VII — CasADi+IPOPT $H{=}20$ parity | `casadi_nmpc_narrow_n80_s7777.json`, `casadi_vs_tsh_kscale_paired_stats.json` |
| Table VIII — negative ablation (random vs. learned TRM) | `negabl_unified_n80_s7777.json` |
| Table IX — Dryden wind robustness | `dryden_wind_benchmark.json` |
| Table X — dynamic obstacle | `dynamic_obstacle_benchmark.json` |
| Table XI — latency profile | `casadi_latency_distribution_n40_s7777.json` |
| Holm-Bonferroni summary (Appendix) | recomputed by `compute_holm_m13.py` |

## Compute scripts

These scripts read the JSON files above and re-derive the summary statistics:

| Script | Role |
|---|---|
| `audit_manuscript_vs_json.py` | **Full audit** — checks every cell in the manuscript LaTeX against the corresponding JSON; prints any disagreement |
| `compute_r0_vs_r1_stats.py` | Reproduces Table II (R0 vs. R1 paired equivalence) |
| `compute_mass_three_seed.py` | Reproduces Table IV (+50% mass across three seeds) |
| `compute_mppi_lambda_paired.py` | Reproduces Table V (MPPI $\lambda$ sweep) |
| `compute_matched_paired.py` | Reproduces matched-budget paired comparisons |
| `compute_casadi_vs_tsh_kscale.py` | Reproduces Table VII (CasADi+IPOPT parity) |
| `compute_marginal_stats.py` | Wilson 95% intervals + marginal means |
| `compute_holm_m13.py` | Holm-Bonferroni correction across the $m=13$ primary hypothesis tests |
| `compute_matched_cem.py` | CEM / iCEM matched-budget paired comparison |
| `compute_penalty_sensitivity.py` | Hyperparameter sensitivity (rollout cost weights) |

## Monte Carlo simulators (re-run from scratch)

| Script | What it produces |
|---|---|
| `casadi_nmpc_benchmark.py` | CasADi+IPOPT NMPC baseline on a chosen task |
| `casadi_latency_distribution.py` | Latency CDF + tail statistics for CasADi vs. TSH |
| `cbf_fallback_analysis.py` | DT-CCBF fallback rate audit |
| `dryden_wind_benchmark.py` | Dryden turbulence robustness sweep |
| `dynamic_obstacle_benchmark.py` | Moving obstacle (1 m/s) benchmark |
| `gap_width_sensitivity.py` | Gap-width sensitivity sweep |
| `adaptive_pd_experiment.py` | Adaptive-gain PD baseline |
| `b1_generate_narrow_dataset.py` + `b1_retrain_narrow_trm.py` | Negative-ablation: retrain TRM on task-matched data |

## Plotting

| Script | Output |
|---|---|
| `plot_ieee_figures.py` | Regenerates `fig1` … `fig5` PDFs from JSON inputs |
| `generate_fig5_tasks.py`, `generate_fig5_trajectories.py` | Task-family visualization |
| `generate_latex_tables.py` | Auto-generates the manuscript's LaTeX table source from JSON |

## Baselines (`baselines/`)

| File | Method |
|---|---|
| `mppi_controller.py` | MPPI (importance-weighted averaging) |
| `cem_controller.py` | Cross-Entropy Method (3 iterations) |
| `mlp_controller.py` | MLP + CBF supervised baseline |
| `casadi_nmpc_controller.py` | CasADi + IPOPT collocation NMPC (H=20) |

All baselines accept the same `QuadrotorDynamics` object and obstacle list as TSH-NMPC, so paired comparisons share identical task seeds and initial conditions.

## Reproducibility notes

- Every Monte Carlo script accepts `--seed` and `--n_mc` flags. The manuscript uses `seed=2026` (primary), `seed=7777` (cross-validation), `seed=42` (third seed for Table IV).
- All paired comparisons share initial state across methods (no method gets a free easier seed).
- McNemar exact two-sided test is used throughout (`scipy.stats.binomtest(min(b,c), b+c, 0.5)`).
- Bootstrap CIs use $10^4$ resamples with `np.random.default_rng(2026)`.
- Wilson 95% intervals via Newcombe's correction.

## Hardware used in the manuscript

Latency numbers (Table XI, Section V-J) were measured on:

- CPU: AMD Ryzen 9 7900X (12C / 24T)
- RAM: 64 GB DDR5-5600
- OS: Ubuntu 22.04 LTS in WSL2 on Windows 11
- Python 3.11.7, PyTorch 2.2.0 (CPU-only path), CasADi 3.6.5 (IPOPT 3.14)

GPU was used during TRM training only; the controller itself runs CPU-only.
