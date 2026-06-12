# -*- coding: utf-8 -*-
"""
E.1 ablation: Candidate Generation Mode with candidate_mode='trm_pd'
"""

import sys
import os
import time
import json
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_core import (
    QuadrotorDynamics, TRMNMPC, PTRMNMPCPredictor
)

SEED = 2026
N_MC = 20
N_STEPS = 300
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
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


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True):
    results = []
    for _ in range(n_mc):
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()
        collision = False; min_dist = float('inf'); iae = 0.0

        for step in range(N_STEPS):
            u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            x = env.step_discrete(x, u_safe)
            p_np = x[0:3].detach().numpy()
            for obs in env.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                min_dist = min(min_dist, d)
                if d < 0: collision = True
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
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    return {
        'success_rate': float(np.mean(succs) * 100),
        'collision_rate': float(np.mean([r['collision'] for r in results]) * 100),
        'terminal_error_mean': float(np.mean(terrs)),
        'terminal_error_std': float(np.std(terrs)),
        'iae_mean': float(np.mean(iaes)),
        'iae_std': float(np.std(iaes)),
    }


def main():
    set_seed(SEED)
    t_start = time.time()

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    device = torch.device('cpu')

    print("Loading retrained TRM model...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    env = QuadrotorDynamics()
    e1 = {}

    # CL-TRM+PD(alpha=0.5)+Rollout
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.5, candidate_mode='trm_pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    e1['trm_pd_a05'] = r
    print(f"  CL-TRM+PD(a=0.5): Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # CL-TRM+PD(alpha=0.8)+Rollout
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.8, candidate_mode='trm_pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    e1['trm_pd_a08'] = r
    print(f"  CL-TRM+PD(a=0.8): Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # PD+Rollout (alpha=1.0)
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=1.0, candidate_mode='trm_pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    e1['pd_rollout'] = r
    print(f"  PD+Rollout(a=1.0): Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # CL-TRM+Rollout (alpha=0.0) K=50
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.0, candidate_mode='trm_pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    e1['trm_rollout_k50'] = r
    print(f"  CL-TRM+Rollout K=50: Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # CL-TRM+Rollout (alpha=0.0) K=1
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=1, D=16, sigma=0.0,
                                   alpha_blend=0.0, candidate_mode='trm_pd',
                                   pd_sigma=0.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    e1['trm_rollout_k1'] = r
    print(f"  CL-TRM+Rollout K=1:  Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    out_path = os.path.join(save_dir, 'retrain_e1_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(e1, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - t_start:.0f}s ({(time.time() - t_start)/60:.1f}min)")


if __name__ == '__main__':
    main()
