# -*- coding: utf-8 -*-
"""
P0-3: Ablation experiments for Phase 2 (B1, B3, B4, B5, B7)

B1: Selection mechanism ablation (argmin vs weighted avg vs random)
B3: PD+CBF under mismatch/noise (missing Table 4 rows)
B4: MPPI argmin vs importance-weighted (fair comparison)
B5: Failure case analysis (noise_0.05)
B7: Random obstacle scenarios
"""

import sys
import os
import time
import json
import numpy as np
import torch

# Dual output
_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v6', 'p03_progress.log')
_log_f = open(_log_path, 'w', buffering=1)
_builtin_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _builtin_print(*args, **kwargs)
    _log_f.write(' '.join(str(a) for a in args) + '\n')
    _log_f.flush()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_core import QuadrotorDynamics, TRMNMPC, PTRMNMPCPredictor
from baselines import MPPIController, CEMController

# ========== Configuration ==========
SEEDS = [2026, 2027, 2028]
N_MC = 100
N_STEPS = 300
DT = 0.02
X_SP_DEFAULT = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5


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
        trajectory = []

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
            trajectory.append(p_np.copy())

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
            'min_dist': min_dist,
            'trajectory': [t.tolist() for t in trajectory] if len(results) == 0 else None,
        })

    succs = [r['success'] for r in results]
    colls = [r['collision'] for r in results]
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    min_dists = [r['min_dist'] for r in results]

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
        'min_dist_mean': float(np.mean(min_dists)),
        'min_dist_std': float(np.std(min_dists)),
        'trajectory': results[0]['trajectory'],
        'per_trial_terrs': [r['terminal_error'] for r in results],
        'per_trial_collisions': [r['collision'] for r in results],
    }


def run_multi_seed(env, predictor_factory, x_sp, seeds=SEEDS, **kwargs):
    seed_results = []
    for seed in seeds:
        set_seed(seed)
        predictor = predictor_factory()
        r = run_mc_trials(env, predictor, x_sp, **kwargs)
        seed_results.append(r)

    succ_rates = [r['success_rate'] for r in seed_results]
    terr_means = [r['terminal_error_mean'] for r in seed_results]
    terr_stds = [r['terminal_error_std'] for r in seed_results]
    iae_means = [r['iae_mean'] for r in seed_results]
    iae_stds = [r['iae_std'] for r in seed_results]
    coll_rates = [r['collision_rate'] for r in seed_results]
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


# ========== Ablation Controllers ==========

class ArgminMPPIController(MPPIController):
    """MPPI with argmin selection instead of importance-weighted averaging."""
    def predict_action(self, x_init, x_sp, enable_cbf=True):
        u_pd = self._compute_pd_baseline(x_init, x_sp)

        if self.K == 1:
            u_nominal = u_pd
        else:
            noise = torch.randn(self.K, 3) * self.sigma
            u_candidates = u_pd.unsqueeze(0) + noise
            cost = self._batch_rollout_cost(x_init, u_candidates, x_sp)

            if self.last_u is not None:
                dist = torch.sum((u_candidates - self.last_u.unsqueeze(0)) ** 2, dim=1)
                cost = cost + self.eta_hyst * dist

            # Argmin selection (like PTRM)
            best_idx = torch.argmin(cost)
            u_nominal = u_candidates[best_idx]

        self.last_u = u_nominal.clone()

        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x_init, u_nominal)
        else:
            u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
        return u_safe


class RandomSelectionMPPI(MPPIController):
    """MPPI with random selection (no cost evaluation)."""
    def predict_action(self, x_init, x_sp, enable_cbf=True):
        u_pd = self._compute_pd_baseline(x_init, x_sp)

        if self.K == 1:
            u_nominal = u_pd
        else:
            noise = torch.randn(self.K, 3) * self.sigma
            u_candidates = u_pd.unsqueeze(0) + noise
            # Random selection
            idx = torch.randint(0, self.K, (1,)).item()
            u_nominal = u_candidates[idx]

        self.last_u = u_nominal.clone()

        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x_init, u_nominal)
        else:
            u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
        return u_safe


class WeightedAvgMPPI(MPPIController):
    """Standard MPPI with importance-weighted averaging (original)."""
    pass  # Already implemented in base class


# ========== Random Obstacle Generator ==========

def generate_random_obstacles(n_obstacles, x_init, x_sp, seed=42):
    """Generate random obstacles that don't block start or goal."""
    rng = np.random.RandomState(seed)
    obstacles = []
    for _ in range(n_obstacles * 3):  # oversample, reject invalid
        if len(obstacles) >= n_obstacles:
            break
        p = rng.uniform(0.0, 3.5, 3)
        r = rng.uniform(0.25, 0.5)
        # Reject if too close to start or goal
        d_start = np.linalg.norm(p - x_init[:3].numpy())
        d_goal = np.linalg.norm(p - x_sp[:3].numpy())
        if d_start < r + 0.5 or d_goal < r + 0.5:
            continue
        # Reject if too close to existing obstacles
        too_close = False
        for obs in obstacles:
            if np.linalg.norm(p - obs['p']) < r + obs['r'] + 0.3:
                too_close = True
                break
        if too_close:
            continue
        obstacles.append({"p": p, "r": r})
    return obstacles


# ========== Main ==========

def main():
    t_start = time.time()
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"P0-3 Ablation Experiments: N_MC={N_MC}, Seeds={SEEDS}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load TRM model (needed for PTRM predictor)
    print("Loading TRM model...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    all_results = {'config': {'n_mc': N_MC, 'seeds': SEEDS, 'n_steps': N_STEPS, 'dt': DT}}
    env = QuadrotorDynamics()

    # ========== B1: Selection Mechanism Ablation ==========
    print("\n" + "=" * 80)
    print("B1: Selection Mechanism Ablation (K=50)")
    print("=" * 80)

    b1 = {}

    # PTRM-NMPC (argmin, proposed)
    def ptrm_factory():
        return PTRMNMPCPredictor(
            trm_model, env, K=50, D=16, sigma=0.25,
            alpha_blend=0.3, candidate_mode='pd',
            pd_sigma=2.0, use_rollout_cost=True
        )
    r = run_multi_seed(env, ptrm_factory, X_SP_DEFAULT)
    b1['PTRM_argmin'] = r
    print(f"  PTRM (argmin):     Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}±{r['terr_std_across_seeds']:.4f}m")

    # MPPI (importance-weighted, baseline)
    def mppi_factory():
        return WeightedAvgMPPI(env, K=50, sigma=2.0)
    r = run_multi_seed(env, mppi_factory, X_SP_DEFAULT, predictor_type='baseline')
    b1['MPPI_weighted'] = r
    print(f"  MPPI (weighted):   Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}±{r['terr_std_across_seeds']:.4f}m")

    # Argmin-MPPI (argmin selection, no TRM)
    def argmin_factory():
        return ArgminMPPIController(env, K=50, sigma=2.0)
    r = run_multi_seed(env, argmin_factory, X_SP_DEFAULT, predictor_type='baseline')
    b1['Argmin_MPPI'] = r
    print(f"  Argmin-MPPI:       Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}±{r['terr_std_across_seeds']:.4f}m")

    # Random selection (no cost evaluation)
    def random_factory():
        return RandomSelectionMPPI(env, K=50, sigma=2.0)
    r = run_multi_seed(env, random_factory, X_SP_DEFAULT, predictor_type='baseline')
    b1['Random_K50'] = r
    print(f"  Random K=50:       Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}±{r['terr_std_across_seeds']:.4f}m")

    # PD K=1 (no sampling)
    def pd_factory():
        return WeightedAvgMPPI(env, K=1, sigma=0.0)
    r = run_multi_seed(env, pd_factory, X_SP_DEFAULT, predictor_type='baseline')
    b1['PD_K1'] = r
    print(f"  PD K=1:            Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}±{r['terr_std_across_seeds']:.4f}m")

    all_results['b1_selection_ablation'] = b1

    # ========== B3: PD+CBF under mismatch/noise ==========
    print("\n" + "=" * 80)
    print("B3: PD+CBF under mismatch/noise (missing Table 4 rows)")
    print("=" * 80)

    b3 = {}

    # PD K=1 nominal
    def pd_factory():
        return WeightedAvgMPPI(env, K=1, sigma=0.0)
    r = run_multi_seed(env, pd_factory, X_SP_DEFAULT, predictor_type='baseline',
                       use_mismatch=False, process_noise=0.0)
    b3['PD_nominal'] = r
    print(f"  PD K=1 Nominal:       Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # PD K=1 mismatch only
    r = run_multi_seed(env, pd_factory, X_SP_DEFAULT, predictor_type='baseline',
                       use_mismatch=True, process_noise=0.0)
    b3['PD_mismatch'] = r
    print(f"  PD K=1 Mismatch:      Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    # PD K=1 mismatch + noise 0.01
    r = run_multi_seed(env, pd_factory, X_SP_DEFAULT, predictor_type='baseline',
                       use_mismatch=True, process_noise=0.01)
    b3['PD_noise_0.01'] = r
    print(f"  PD K=1 Noise 0.01:    Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}m")

    # PD K=1 mismatch + noise 0.05
    r = run_multi_seed(env, pd_factory, X_SP_DEFAULT, predictor_type='baseline',
                       use_mismatch=True, process_noise=0.05)
    b3['PD_noise_0.05'] = r
    print(f"  PD K=1 Noise 0.05:    Succ={r['success_mean']:.1f}% [{r['success_ci_lo']:.1f}, {r['success_ci_hi']:.1f}], "
          f"TErr={r['terr_mean']:.4f}m")

    # PTRM K=50 mismatch only (for comparison)
    r = run_multi_seed(env, ptrm_factory, X_SP_DEFAULT,
                       use_mismatch=True, process_noise=0.0)
    b3['PTRM_mismatch'] = r
    print(f"  PTRM K=50 Mismatch:   Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    all_results['b3_pd_noise'] = b3

    # ========== B4: MPPI Argmin vs Importance-Weighted ==========
    print("\n" + "=" * 80)
    print("B4: MPPI Selection Mechanism (argmin vs weighted)")
    print("=" * 80)

    b4 = {}

    for K in [10, 50]:
        print(f"\n--- K={K} ---")

        # Weighted avg
        def weighted_factory(K=K):
            return WeightedAvgMPPI(env, K=K, sigma=2.0)
        r = run_multi_seed(env, weighted_factory, X_SP_DEFAULT, predictor_type='baseline')
        b4[f'weighted_K{K}'] = r
        print(f"  Weighted:  Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # Argmin
        def argmin_factory_k(K=K):
            return ArgminMPPIController(env, K=K, sigma=2.0)
        r = run_multi_seed(env, argmin_factory_k, X_SP_DEFAULT, predictor_type='baseline')
        b4[f'argmin_K{K}'] = r
        print(f"  Argmin:    Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

    all_results['b4_selection_comparison'] = b4

    # ========== B5: Failure Case Analysis ==========
    print("\n" + "=" * 80)
    print("B5: Failure Case Analysis (noise_0.05)")
    print("=" * 80)

    # Collect detailed failure data from PTRM K=50 with noise_0.05
    set_seed(2026)
    failure_trials = []
    n_fail_trials = 30  # Run 30 trials to collect failure examples

    for trial_idx in range(n_fail_trials):
        x_init = random_x_init()
        predictor = PTRMNMPCPredictor(
            trm_model, env, K=50, D=16, sigma=0.25,
            alpha_blend=0.3, candidate_mode='pd',
            pd_sigma=2.0, use_rollout_cost=True
        )
        predictor.reset()
        x = x_init.clone()
        collision = False
        collision_step = -1
        trajectory = [x[0:3].detach().numpy().copy()]

        for step in range(N_STEPS):
            u_safe, _ = predictor.predict_action(x, X_SP_DEFAULT, enable_cbf=True)
            x = env.step_discrete(x, u_safe, use_mismatch=True, process_noise=0.05)
            p_np = x[0:3].detach().numpy()
            trajectory.append(p_np.copy())

            for obs in env.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                if d < 0 and not collision:
                    collision = True
                    collision_step = step

        terr = torch.norm(x[0:3] - X_SP_DEFAULT[0:3]).item()
        success = (not collision) and (terr < TErr_THRESH)

        if not success:
            failure_trials.append({
                'trial': trial_idx,
                'collision': collision,
                'collision_step': collision_step,
                'terminal_error': terr,
                'x_init': x_init.tolist(),
                'trajectory': [t.tolist() for t in trajectory[::10]],  # subsample
            })

    b5 = {
        'n_trials': n_fail_trials,
        'n_failures': len(failure_trials),
        'failure_rate': len(failure_trials) / n_fail_trials * 100,
        'failure_examples': failure_trials[:5],  # Keep top 5 examples
    }
    print(f"  Failures: {len(failure_trials)}/{n_fail_trials} ({b5['failure_rate']:.1f}%)")
    if failure_trials:
        coll_fails = sum(1 for f in failure_trials if f['collision'])
        terr_fails = sum(1 for f in failure_trials if not f['collision'])
        print(f"  Collision failures: {coll_fails}, TErr failures: {terr_fails}")
        if coll_fails > 0:
            coll_steps = [f['collision_step'] for f in failure_trials if f['collision']]
            print(f"  Collision step range: [{min(coll_steps)}, {max(coll_steps)}]")

    all_results['b5_failure_analysis'] = b5

    # ========== B7: Random Obstacle Scenarios ==========
    print("\n" + "=" * 80)
    print("B7: Random Obstacle Scenarios")
    print("=" * 80)

    b7 = {}
    x_init_ref = torch.tensor([0.0, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    for n_obs in [3, 5, 7]:
        print(f"\n--- {n_obs} obstacles ---")
        obstacles = generate_random_obstacles(n_obs, x_init_ref, X_SP_DEFAULT, seed=42 + n_obs)
        print(f"  Generated {len(obstacles)} obstacles")
        env_obs = QuadrotorDynamics(obstacles=obstacles)

        # PTRM K=50
        def ptrm_factory_obs(env_obs=env_obs):
            return PTRMNMPCPredictor(
                trm_model, env_obs, K=50, D=16, sigma=0.25,
                alpha_blend=0.3, candidate_mode='pd',
                pd_sigma=2.0, use_rollout_cost=True
            )
        r = run_multi_seed(env_obs, ptrm_factory_obs, X_SP_DEFAULT)
        ptrm_r = r
        print(f"  PTRM: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # MPPI K=50
        def mppi_factory_obs(env_obs=env_obs):
            return WeightedAvgMPPI(env_obs, K=50, sigma=2.0)
        r = run_multi_seed(env_obs, mppi_factory_obs, X_SP_DEFAULT, predictor_type='baseline')
        mppi_r = r
        print(f"  MPPI: Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        # CEM K=50
        def cem_factory_obs(env_obs=env_obs):
            return CEMController(env_obs, K=50, n_iter=3, sigma=2.0)
        r = run_multi_seed(env_obs, cem_factory_obs, X_SP_DEFAULT, predictor_type='baseline')
        cem_r = r
        print(f"  CEM:  Succ={r['success_mean']:.1f}%, TErr={r['terr_mean']:.4f}m")

        b7[f'{n_obs}_obs'] = {
            'PTRM': ptrm_r, 'MPPI': mppi_r, 'CEM': cem_r,
            'obstacles': [{'p': o['p'].tolist(), 'r': o['r']} for o in obstacles]
        }

    all_results['b7_random_obstacles'] = b7

    # ========== Save Results ==========
    out_path = os.path.join(save_dir, 'p03_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    t_total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    print("P0-3 ablation experiments complete!")


if __name__ == '__main__':
    main()
