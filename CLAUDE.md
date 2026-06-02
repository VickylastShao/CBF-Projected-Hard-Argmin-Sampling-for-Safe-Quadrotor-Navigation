# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PTRM-NMPC: Probabilistic Tiny Recursive Model-based Nonlinear Model Predictive Control — a test-time compute scaling approach for quadrotor 3D obstacle avoidance. Target venues: IEEE TAC or Automatica.

## Commands

```bash
# Activate environment and run the full pipeline (dataset generation → training → Monte Carlo experiments)
conda activate pytorch
python quadrotor_core_simulation.py

# The script auto-detects CUDA/CPU; output figure saved as ptrm_nmpc_advanced_experiments.png
```

No test runner, linter, or build system exists yet. Dependencies are implicit: PyTorch, NumPy, Matplotlib (install via conda `pytorch` environment).

## Code Architecture

The entire simulation lives in a single monolithic file `quadrotor_core_simulation.py` (755 lines), organized into 7 sections:

### Class Pipeline (execution order)

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

### Training & Evaluation Functions

- `generate_quadrotor_dataset()`: 150 expert NMPC trajectories
- `train_trm_jointly()`: Joint policy (deep supervision, γ=0.95) + Q-head (cost regression, V_max=150, λ_Q=0.1), 35 epochs, Adam lr=0.0025
- `run_monte_carlo_experiments()`: 3 experiments (obstacle avoidance, robustness under mismatch, latency profiling across K=[1,10,50,100])

### Manuscript ↔ Code Mapping

`PTRM_NMPC_manuscript.md` is the paper draft. All numerical result tables (Tables 1–3 in Section 6) contain "script-reported" placeholders — they are filled by running `quadrotor_core_simulation.py`. When updating the manuscript with results, copy values from the script's stdout output.

## Critical Implementation Details

- **Parameter count must be exactly 27,935**: The `__main__` block audits this; any architecture change must preserve this count or update the manuscript's Table 4
- **Q-head labels use decoded candidate costs**, NOT expert optimal — this is a deliberate design choice for test-time consistency
- **6D abstract dynamics do NOT re-subtract gravity**: Gravity pre-compensation is handled in the differential flatness mapping (cascaded architecture), not in the abstract model
- **Hysteresis penalty**: L2 distance to *shifted* previous optimal trajectory (shift by one timestep), with η_hyst=0.05
- **DT-CCBF uses Log-Sum-Exponential smooth approximation** of the min-barrier (not hard min), enabling gradient-based safety projection
- **CBF fallback**: When QP is infeasible (A^T u ≤ b has no solution within box constraints), the solver falls back to the closest feasible control within actuator limits plus safety buffer δ_buffer=0.15m
- **Code comments are in Chinese**; manuscript is in English

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
| latent_dim | 64 | `TRMNMPC.__init__` |
| mpc_horizon | 30 (10 steps × 3D) | `TRMNMPC.__init__` |
| Training epochs | 35 | `train_trm_jointly` |
| Adam lr | 0.0025 | `train_trm_jointly` |
| Dataset size | 150 | `generate_quadrotor_dataset` |

## Phase Progress

- [x] Phase 1: Core simulation framework (quadrotor dynamics + DT-CCBF + PTRM network)
- [x] Phase 2: Manuscript drafting (English, targeting IEEE TAC / Automatica)
- [ ] Phase 3: Full experiments (Monte Carlo across multiple scenarios)
- [ ] Phase 4: Paper revision and submission

## Planned Directories (not yet created)

- `docs/` — Design specs, wiki, and plans
- `experiments/` — Training and evaluation scripts
- `tests/` — Unit and integration tests
