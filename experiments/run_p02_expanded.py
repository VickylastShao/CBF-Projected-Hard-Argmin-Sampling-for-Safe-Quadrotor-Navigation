# -*- coding: utf-8 -*-
"""
P0-2: Expanded experiments with N_MC=100, multiple seeds, Wilson CIs,
additional setpoints, and dynamic obstacle scenario.
"""

import sys
import os
import time
import json
import numpy as np
import torch

# Dual output: write to both stdout and a log file
_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v6', 'p02_progress.log')
_log_f = open(_log_path, 'w', buffering=1)  # line-buffered

_builtin_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _builtin_print(*args, **kwargs)
    _log_f.write(' '.join(str(a) for a in args) + '\n')
    _log_f.flush()

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_core import (
    QuadrotorDynamics, TRMNMPC, PTRMNMPCPredictor
)
from baselines import MPPIController, CEMController, MLPController, MLPPredictor

# ========== Configuration ==========
SEEDS = [2026, 2027, 2028]
N_MC = 100
N_STEPS = 300
DT = 0.02
X_SP_DEFAULT = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5

OBSTACLE_CONFIGS = {
    'Corridor': [
        {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
        {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
        {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
    ],
    'Dense-5': [
        {"p": np.array([0.8, 1.0, 0.8]), "r": 0.4},
        {"p": np.array([1.5, 0.8, 1.5]), "r": 0.35},
        {"p": np.array([2.0, 1.5, 1.0]), "r": 0.4},
        {"p": np.array([1.2, 2.0, 2.0]), "r": 0.35},
        {"p": np.array([1.8, 2.5, 1.5]), "r": 0.3}
    ],
}

ENV_TARGETS = {
    'Corridor': torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    'Dense-5': torch.tensor([2.5, 3.0, 2.5, 0.0, 0.0, 0.0], dtype=torch.float32),
}

ADDITIONAL_SETPOINTS = {
    'SP_A': torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    'SP_B': torch.tensor([3.0, 1.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32),
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init():
    return torch.tensor([
        np.random.uniform(-0.5, 1.5), np.random.uniform(-1.0, 0.0),
        np.random.uniform(-0.5, 1.5), np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4), np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    spread = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    lo = max(0.0, center - spread) * 100
    hi = min(1.0, center + spread) * 100
    return lo, hi


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True,
                  use_mismatch=False, process_noise=0.0, predictor_type='ptrm',
                  env_factory=None):
    """Run N_MC Monte Carlo trials and return aggregated results."""
    results = []
    for _ in range(n_mc):
        if env_factory is not None:
            env = env_factory()
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()
        collision = False
        min_dist = float('inf')
        iae = 0.0

        for step in range(N_STEPS):
            if predictor_type == 'ptrm':
                u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'baseline':
                u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'mlp':
                u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)

            x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch,
                                  process_noise=process_noise)
            p_np = x[0:3].detach().numpy()
            for obs in env.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                min_dist = min(min_dist, d)
                if d < 0:
                    collision = True
            iae += torch.norm(x[0:3] - x_sp[0:3]).item()

        iae = iae / N_STEPS
        terr = torch.norm(x[0:3] - x_sp[0:3]).item()
        results.append({
            'success': (not collision) and (terr < TErr_THRESH),
            'collision': collision,
            'terminal_error': terr,
            'iae': iae,
        })

    succs = [r['success'] for r in results]
    colls = [r['collision'] for r in results]
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]

    n_success = sum(succs)
    ci_lo, ci_hi = wilson_ci(n_success, n_mc)

    return {
        'success_rate': float(np.mean(succs) * 100),
        'success_ci_lo': float(ci_lo),
        'success_ci_hi': float(ci_hi),
        'collision_rate': float(np.mean(colls) * 100),
        'terminal_error_mean': float(np.mean(terrs)),
        'terminal_error_std': float(np.std(terrs)),
        'iae_mean': float(np.mean(iaes)),
        'iae_std': float(np.std(iaes)),
    }


def run_multi_seed(env, predictor_factory, x_sp, seeds=SEEDS, **kwargs):
    """Run experiments across multiple seeds and aggregate results."""
    seed_results = []
    for seed in seeds:
        set_seed(seed)
        predictor = predictor_factory()
        r = run_mc_trials(env, predictor, x_sp, **kwargs)
        seed_results.append(r)


    # Aggregate across seeds
    succ_rates = [r['success_rate'] for r in seed_results]
    terr_means = [r['terminal_error_mean'] for r in seed_results]
    terr_stds = [r['terminal_error_std'] for r in seed_results]
    iae_means = [r['iae_mean'] for r in seed_results]
    iae_stds = [r['iae_std'] for r in seed_results]
    coll_rates = [r['collision_rate'] for r in seed_results]

    # Aggregate Wilson CIs: take union (min lo, max hi) across seeds
    ci_los = [r['success_ci_lo'] for r in seed_results]
    ci_his = [r['success_ci_hi'] for r in seed_results]

    return {
        'success_mean': float(np.mean(succ_rates)),
        'success_std_across_seeds': float(np.std(succ_rates)),
        'success_ci_lo': float(np.min(ci_los)),
        'success_ci_hi': float(np.max(ci_his)),
        'collision_mean': float(np.mean(coll_rates)),
        'terr_mean': float(np.mean(terr_means)),
        'terr_std_within': float(np.mean(terr_stds)),
        'terr_std_across_seeds': float(np.std(terr_means)),
        'iae_mean': float(np.mean(iae_means)),
        'iae_std_within': float(np.mean(iae_stds)),
        'iae_std_across_seeds': float(np.std(iae_means)),
        'per_seed': seed_results,
    }


def make_ptrm_factory(trm_model, env, K, sigma=0.25, alpha_blend=0.3,
                      candidate_mode='pd', pd_sigma=2.0):
    """Factory for PTRM predictor (fresh per seed)."""
    def factory():
        return PTRMNMPCPredictor(
            trm_model, env, K=K, D=16, sigma=sigma,
            alpha_blend=alpha_blend, candidate_mode=candidate_mode,
            pd_sigma=pd_sigma, use_rollout_cost=True
        )
    return factory


def make_mppi_factory(env, K, sigma=2.0):
    """Factory for MPPI predictor."""
    def factory():
        return MPPIController(env, K=K, sigma=sigma)
    return factory


def make_cem_factory(env, K=50, n_iter=3, sigma=2.0):
    """Factory for CEM predictor."""
    def factory():
        return CEMController(env, K=K, n_iter=n_iter, sigma=sigma)
    return factory


def make_pd_factory(env, K=1, sigma=0.0):
    """Factory for PD-only predictor (MPPI with K=1, sigma=0)."""
    def factory():
        return MPPIController(env, K=K, sigma=sigma)
    return factory


# ========== Dynamic Obstacle Environment ==========

class DynamicObstacleEnv(QuadrotorDynamics):
    """Obstacle configuration changes at a trigger step."""

    def __init__(self, obstacles, trigger_step=150, new_obstacles=None):
        super().__init__(obstacles=obstacles)
        self.trigger_step = trigger_step
        self.new_obstacles = new_obstacles
        self.step_count = 0

    def step_discrete(self, x, u, **kwargs):
        if self.step_count == self.trigger_step and self.new_obstacles is not None:
            self.obstacles = self.new_obstacles
        self.step_count += 1
        return super().step_discrete(x, u, **kwargs)

    def reset_counter(self):
        self.step_count = 0


# ========== Main ==========

def main():
    t_start = time.time()
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    mlp_path = os.path.join(save_dir, 'mlp_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"P0-2 Expanded Experiments: N_MC={N_MC}, Seeds={SEEDS}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load models
    print("Loading TRM model...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    print("Loading MLP model...")
    mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    mlp_model.eval()

    all_results = {'config': {'n_mc': N_MC, 'seeds': SEEDS, 'n_steps': N_STEPS, 'dt': DT}}

    # ========== Exp A: K-Scaling (Table 1) ==========
    print("\n" + "=" * 80)
    print("Exp A: K-Scaling (Table 1)")
    print("=" * 80)

    env = QuadrotorDynamics()
    exp_a = {}
    for K in [1, 10, 50]:
        sigma = 0.25 if K > 1 else 0.0
        pd_sigma = 2.0 if K > 1 else 0.0
        factory = make_ptrm_factory(trm_model, env, K=K, sigma=sigma, pd_sigma=pd_sigma)
        r = run_multi_seed(env, factory, X_SP_DEFAULT)
        exp_a[f'K={K}'] = r
        print(f"  K={K:3d}: Succ={r['success_mean']:.1f}% "
              f"[{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
              f"TErr={r['terr_mean']:.4f}±{r['terr_std_across_seeds']:.4f}m, "
              f"IAE={r['iae_mean']:.2f}±{r['iae_std_across_seeds']:.2f}")

    all_results['exp_a_k_scaling'] = exp_a

    # ========== Exp B: Baselines (Table 2) ==========
    print("\n" + "=" * 80)
    print("Exp B: Baselines (Table 2, K=50)")
    print("=" * 80)

    exp_b = {}

    # PTRM-NMPC
    factory = make_ptrm_factory(trm_model, env, K=50)
    r = run_multi_seed(env, factory, X_SP_DEFAULT)
    exp_b['PTRM_K50'] = r
    print(f"  PTRM K=50:  Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # MPPI
    factory = make_mppi_factory(env, K=50)
    r = run_multi_seed(env, factory, X_SP_DEFAULT, predictor_type='baseline')
    exp_b['MPPI_K50'] = r
    print(f"  MPPI K=50:  Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # CEM
    factory = make_cem_factory(env, K=50)
    r = run_multi_seed(env, factory, X_SP_DEFAULT, predictor_type='baseline')
    exp_b['CEM_K50'] = r
    print(f"  CEM K=50:   Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # PD K=1
    factory = make_pd_factory(env, K=1)
    r = run_multi_seed(env, factory, X_SP_DEFAULT, predictor_type='baseline')
    exp_b['PD_K1'] = r
    print(f"  PD K=1:     Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # MLP+CBF
    def mlp_factory():
        return MLPPredictor(mlp_model, env, alpha_blend=0.3)
    r = run_multi_seed(env, mlp_factory, X_SP_DEFAULT, predictor_type='mlp')
    exp_b['MLP_CBF'] = r
    print(f"  MLP+CBF:    Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    all_results['exp_b_baselines'] = exp_b

    # ========== Exp C: Noise Robustness (Table 4) ==========
    print("\n" + "=" * 80)
    print("Exp C: Noise Robustness (Table 4, K=50, mismatch)")
    print("=" * 80)

    exp_c = {}

    # Nominal (no noise, no mismatch)
    factory = make_ptrm_factory(trm_model, env, K=50)
    r = run_multi_seed(env, factory, X_SP_DEFAULT, use_mismatch=False, process_noise=0.0)
    exp_c['nominal'] = r
    print(f"  Nominal:       Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # Mismatch only
    factory = make_ptrm_factory(trm_model, env, K=50)
    r = run_multi_seed(env, factory, X_SP_DEFAULT, use_mismatch=True, process_noise=0.0)
    exp_c['mismatch_only'] = r
    print(f"  Mismatch only: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # Mismatch + noise
    for noise in [0.01, 0.05]:
        factory = make_ptrm_factory(trm_model, env, K=50)
        r = run_multi_seed(env, factory, X_SP_DEFAULT, use_mismatch=True, process_noise=noise)
        exp_c[f'noise_{noise}'] = r
        print(f"  Noise {noise:.2f}:      Succ={r['success_mean']:.1f}% "
              f"[{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
              f"TErr={r['terr_mean']:.4f}m")

    all_results['exp_c_noise'] = exp_c

    # ========== Exp D: Multi-Obstacle (Table 6) ==========
    print("\n" + "=" * 80)
    print("Exp D: Multi-Obstacle (Table 6, K=50)")
    print("=" * 80)

    exp_d = {}
    for env_name, obstacles in OBSTACLE_CONFIGS.items():
        print(f"\n--- {env_name} ---")
        env_obs = QuadrotorDynamics(obstacles=obstacles)
        x_sp_env = ENV_TARGETS[env_name]

        # PTRM
        factory = make_ptrm_factory(trm_model, env_obs, K=50)
        r = run_multi_seed(env_obs, factory, x_sp_env)
        ptrm_r = r
        print(f"  PTRM: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # MPPI
        factory = make_mppi_factory(env_obs, K=50)
        r = run_multi_seed(env_obs, factory, x_sp_env, predictor_type='baseline')
        mppi_r = r
        print(f"  MPPI: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # CEM
        factory = make_cem_factory(env_obs, K=50)
        r = run_multi_seed(env_obs, factory, x_sp_env, predictor_type='baseline')
        cem_r = r
        print(f"  CEM:  Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        exp_d[env_name] = {'PTRM': ptrm_r, 'MPPI': mppi_r, 'CEM': cem_r}

    all_results['exp_d_multi_obstacle'] = exp_d

    # ========== Exp E: Additional Setpoints ==========
    print("\n" + "=" * 80)
    print("Exp E: Additional Setpoints (K=50)")
    print("=" * 80)

    exp_e = {}
    for sp_name, x_sp in ADDITIONAL_SETPOINTS.items():
        print(f"\n--- {sp_name}: {x_sp[:3].tolist()} ---")

        # PTRM
        factory = make_ptrm_factory(trm_model, env, K=50)
        r = run_multi_seed(env, factory, x_sp)
        ptrm_r = r
        print(f"  PTRM: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # MPPI
        factory = make_mppi_factory(env, K=50)
        r = run_multi_seed(env, factory, x_sp, predictor_type='baseline')
        mppi_r = r
        print(f"  MPPI: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # PD K=1
        factory = make_pd_factory(env, K=1)
        r = run_multi_seed(env, factory, x_sp, predictor_type='baseline')
        pd_r = r
        print(f"  PD K=1: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        exp_e[sp_name] = {'PTRM': ptrm_r, 'MPPI': mppi_r, 'PD_K1': pd_r}

    all_results['exp_e_setpoints'] = exp_e

    # ========== Exp F: Dynamic Obstacle ==========
    print("\n" + "=" * 80)
    print("Exp F: Dynamic Obstacle (obstacle moves at t=150)")
    print("=" * 80)

    # Obstacle 2 moves from [2.0, 1.5, 2.0] to [2.0, 2.0, 2.0] at step 150
    obstacles_initial = OBSTACLE_CONFIGS['Corridor']
    obstacles_moved = [
        {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
        {"p": np.array([2.0, 2.0, 2.0]), "r": 0.5},  # moved up
        {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
    ]

    exp_f = {}

    def make_dyn_env():
        return DynamicObstacleEnv(obstacles_initial, trigger_step=150,
                                  new_obstacles=obstacles_moved)

    # PTRM K=50
    def ptrm_dyn_factory():
        dyn_env = make_dyn_env()
        return PTRMNMPCPredictor(
            trm_model, dyn_env, K=50, D=16, sigma=0.25,
            alpha_blend=0.3, candidate_mode='pd',
            pd_sigma=2.0, use_rollout_cost=True
        )
    r = run_multi_seed(
        make_dyn_env(), ptrm_dyn_factory, X_SP_DEFAULT,
        env_factory=make_dyn_env
    )
    exp_f['PTRM_K50'] = r
    print(f"  PTRM K=50: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # MPPI K=50
    def mppi_dyn_factory():
        dyn_env = make_dyn_env()
        return MPPIController(dyn_env, K=50, sigma=2.0)
    r = run_multi_seed(
        make_dyn_env(), mppi_dyn_factory, X_SP_DEFAULT,
        predictor_type='baseline', env_factory=make_dyn_env
    )
    exp_f['MPPI_K50'] = r
    print(f"  MPPI K=50: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # PD K=1
    def pd_dyn_factory():
        dyn_env = make_dyn_env()
        return MPPIController(dyn_env, K=1, sigma=0.0)
    r = run_multi_seed(
        make_dyn_env(), pd_dyn_factory, X_SP_DEFAULT,
        predictor_type='baseline', env_factory=make_dyn_env
    )
    exp_f['PD_K1'] = r
    print(f"  PD K=1:    Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    all_results['exp_f_dynamic_obstacle'] = exp_f

    # ========== Save Results ==========
    out_path = os.path.join(save_dir, 'p02_expanded_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    t_total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    print("P0-2 expanded experiments complete!")


if __name__ == '__main__':
    main()
