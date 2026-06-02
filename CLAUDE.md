# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PTRM-NMPC: Probabilistic Tiny Recursive Model-based Nonlinear Model Predictive Control — a test-time compute scaling approach for quadrotor 3D obstacle avoidance. Target venues: IEEE TAC or Automatica.

## Commands

```bash
# Run the full pipeline (dataset generation → training → Monte Carlo experiments)
python quadrotor_core_simulation.py

# Run the extended experiment suite (v5: 6 experiments with ablation studies)
python experiments/ptrm_nmpc_v5_experiments.py

# Generate IEEE-quality figures for manuscript
python experiments/plot_ieee_figures.py

# The scripts auto-detect CUDA/CPU; figures saved to respective results directories
```

No test runner, linter, or build system exists yet. Dependencies: PyTorch, NumPy, Matplotlib (available in base conda environment).

## Code Architecture

### Main simulation: `quadrotor_core_simulation.py`

The core simulation lives in a single monolithic file, organized into 7 sections:

1. **`QuadrotorDynamics`** — 6D quadrotor translational dynamics + DT-CCBF safety filter
   - `step_discrete()`: RK4 integration with optional +50% mass/drag mismatch and Gaussian process noise
   - `apply_cbf_projection()`: Log-Sum-Exponential smooth CBF via relative-degree-2 Jacobian; constructs linear safety constraint A^T u ≤ b
   - `_project_control()`: Bisection Lagrange multiplier solver for CBF QP; box-constrained emergency fallback when infeasible

2. **`GoldenNMPCSolver`** — Offline expert baseline (L-BFGS, horizon H=10, Q=diag(15,15,15,1,1,1), R=0.02)
   - Generates training data via `solve()` and Q-head labels via `evaluate_cost()`

3. **`TRMNMPC(nn.Module)`** — Weight-shared latent recursive network (27,935 trainable params)
   - Input: concatenated [x_init(6), x_sp(6)] → 12D
   - Latent: 64D with D recursive steps sharing W_z, M_z, M_y weights
   - Output: f_O decoder (64→30, i.e. horizon-10 × 3D control), f_Q Q-head (64→32→1)

4. **`PTRMNMPCPredictor`** — Online probabilistic inference (K=50 candidates, D=16 recursion, σ=0.25 noise)
   - `predict_action()`: K parallel noise-injected rollouts → Q-head ranking → trajectory-space hysteresis → optional CBF

### Extended experiments: `experiments/ptrm_nmpc_v5_experiments.py`

6 experiments with 100 MC runs each:
- **exp1**: K-scaling under 3 CBF conditions (NoCBF/WeakCBF/StrongCBF), K∈{1,5,10,20,50,100}
- **exp2**: σ-scaling ablation, σ∈{0.5,1.0,1.5,2.0,3.0,4.0}
- **exp3**: Model mismatch robustness (4 conditions × K∈{1,5,10,20,50,100})
- **exp4**: Noise robustness
- **exp5**: Ablation (rollout horizon T + obstacle weight w_obs)
- **exp6**: Runtime profiling

### IEEE figure generation: `experiments/plot_ieee_figures.py`

Generates Figure 1–5 for manuscript:
- Fig 1: Success rate vs K (3 CBF conditions)
- Fig 2: IAE vs K (3 CBF conditions)
- Fig 3: Robustness under mismatch (grouped bar charts)
- Fig 4: Ablation studies (σ scaling + rollout horizon)
- Fig 5: 3D trajectory visualization

Output: `experiments/figures_ieee/` (PDF + PNG)

### Manuscript ↔ Code Mapping

`PTRM_NMPC_manuscript.md` is the paper draft. **All numerical result tables (Tables 1–3 in Section 6) are now filled with real experimental data** from `experiments/results_v5/raw_results.json`. The data has been verified for consistency between the manuscript and the JSON source.

## Critical Implementation Details

- **Parameter count must be exactly 27,935**: The `__main__` block audits this; any architecture change must preserve this count or update the manuscript's Table 4
- **Q-head labels use decoded candidate costs**, NOT expert optimal — this is a deliberate design choice for test-time consistency
- **6D abstract dynamics do NOT re-subtract gravity**: Gravity pre-compensation is handled in the differential flatness mapping (cascaded architecture), not in the abstract model
- **Hysteresis penalty**: L2 distance to *shifted* previous optimal trajectory (shift by one timestep), with η_hyst=0.05
- **DT-CCBF uses Log-Sum-Exponential smooth approximation** of the min-barrier (not hard min), enabling gradient-based safety projection
- **CBF fallback**: When QP is infeasible (A^T u ≤ b has no solution within box constraints), the solver falls back to the closest feasible control within actuator limits plus safety buffer δ_buffer=0.15m
- **Code comments are in Chinese**; manuscript is in English
- **No-CBF IAE increases with K** (163.9→190.8): This is expected — K=1 collisions terminate early (low cumulative error), while higher-K surviving trajectories traverse longer detour paths
- **Process Noise K=100 IAE (217.6) ≥ K=50 IAE (217.1)**: Within MC variance; zero-mean noise doesn't systematically benefit from more candidates

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| dt | 0.02s | `QuadrotorDynamics.__init__` |
| mass / drag | 1.5 kg / 0.1 | `QuadrotorDynamics.__init__` |
| u_max | 15.0 N | `QuadrotorDynamics.__init__` |
| α_d / γ_d | 0.8 / 0.2 | `QuadrotorDynamics.__init__` |
| δ_buffer | 0.15m | `QuadrotorDynamics.__init__` |
| K (candidates) | 50 | `PTRMNMPCPredictor.__init__` |
| D (recursion depth) | 16 | `PTRMNMPCPredictor.__init__` |
| σ (noise scale) | 0.25 | `PTRMNMPCPredictor.__init__` |
| σ (online perturbation) | 2.0 | Experiment config |
| rollout horizon T | 20 steps (0.4s) | Experiment config |
| latent_dim | 64 | `TRMNMPC.__init__` |
| mpc_horizon | 30 (10 steps × 3D) | `TRMNMPC.__init__` |
| Training epochs | 35 | `train_trm_jointly` |
| Adam lr | 0.0025 | `train_trm_jointly` |
| Dataset size | 150 | `generate_quadrotor_dataset` |

## Phase Progress

- [x] Phase 1: Core simulation framework (quadrotor dynamics + DT-CCBF + PTRM network)
- [x] Phase 2: Manuscript drafting (English, targeting IEEE TAC / Automatica)
- [x] Phase 3: Full experiments (Monte Carlo across multiple scenarios, 6 experiments, 100 MC runs each)
- [x] Phase 3a: Data backfill to manuscript Tables 1–3
- [x] Phase 3b: IEEE-quality figure generation (Fig 1–5)
- [ ] Phase 4: Paper revision and submission

## Directory Structure

```
PTRM-NMPC/
├── CLAUDE.md                          # This file
├── PTRM_NMPC_manuscript.md            # Paper draft (English)
├── quadrotor_core_simulation.py       # Main simulation script
├── quadrotor_core/                    # Modularized core (v2-v4 iterations)
├── experiments/
│   ├── ptrm_nmpc_v5_experiments.py    # Extended experiment suite
│   ├── plot_ieee_figures.py           # IEEE figure generation
│   ├── results_v5/                    # V5 experiment results + LaTeX tables
│   │   ├── raw_results.json           # All numerical data
│   │   ├── *.tex                      # Auto-generated LaTeX tables
│   │   └── *.pdf / *.png              # Auto-generated figures
│   └── figures_ieee/                  # IEEE-quality figures for manuscript
│       ├── fig1_success_rate_vs_K.pdf
│       ├── fig2_iae_vs_K.pdf
│       ├── fig3_robustness_mismatch.pdf
│       ├── fig4_ablation_studies.pdf
│       └── fig5_trajectories_3d.pdf
├── plot_3d_trajectories.py            # Standalone 3D trajectory plotter
└── requirements.txt
```
