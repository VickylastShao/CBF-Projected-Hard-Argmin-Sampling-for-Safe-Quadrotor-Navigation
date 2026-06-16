# -*- coding: utf-8 -*-
"""
P0-1.3: 重训模型后，针对剩余表格 (Table 2-4, 6) 运行针对性实验。
使用 K=1/10/50, N_MC=20 快速获取数据。
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
    QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
    PTRMNMPCPredictor, generate_quadrotor_dataset, train_trm_jointly
)
from baselines import MPPIController, MLPController, MLPPredictor, train_mlp, CEMController

SEED = 2026
N_MC = 20
N_STEPS = 300
DT = 0.02
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
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

K_VALUES = [1, 10, 50]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init():
    return torch.tensor([
        np.random.uniform(-0.5, 1.5), np.random.uniform(-1.0, 0.0),
        np.random.uniform(-0.5, 1.5), np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4), np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True,
                  use_mismatch=False, process_noise=0.0, predictor_type='ptrm'):
    results = []
    for _ in range(n_mc):
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()
        collision = False; min_dist = float('inf'); iae = 0.0

        for step in range(N_STEPS):
            if predictor_type == 'ptrm':
                u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'baseline':
                u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'mlp':
                u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)

            x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)
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
            'min_distance': min_dist,
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
    mlp_path = os.path.join(save_dir, 'mlp_model.pt')
    device = torch.device('cpu')

    # Load TRM model
    print("加载重训后的 TRM 模型...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    # Load MLP model
    if os.path.exists(mlp_path):
        print("加载 MLP 模型...")
        mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        mlp_model.eval()
    else:
        print("训练 MLP 模型...")
        env_train = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env_train, horizon=10)
        dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                              x_sp=X_SP, pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])
        mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
        mlp_model = train_mlp(mlp_model, dataset, env_train, epochs=100, lr=0.001, patience=20, verbose=True)
        torch.save(mlp_model.state_dict(), mlp_path)
        mlp_model.eval()

    all_results = {}

    # ========== Table 2: Baseline Comparison ==========
    print("\n" + "=" * 80)
    print("Table 2: Baseline Comparison (Strong CBF, K=50)")
    print("=" * 80)

    env = QuadrotorDynamics()
    table2 = {}

    # PTRM K=50
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    table2['PTRM_K50'] = r
    print(f"  PTRM K=50:    Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # MPPI K=50
    set_seed(SEED)
    mppi = MPPIController(env, K=50, sigma=2.0)
    r = run_mc_trials(env, mppi, X_SP, enable_cbf=True, predictor_type='baseline')
    table2['MPPI_K50'] = r
    print(f"  MPPI K=50:    Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # CEM K=50
    set_seed(SEED)
    cem = CEMController(env, K=50, n_iter=3, sigma=2.0)
    r = run_mc_trials(env, cem, X_SP, enable_cbf=True, predictor_type='baseline')
    table2['CEM_K50'] = r
    print(f"  CEM K=50:     Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # MLP+CBF
    set_seed(SEED)
    mlp_pred = MLPPredictor(mlp_model, env, alpha_blend=0.3)
    r = run_mc_trials(env, mlp_pred, X_SP, enable_cbf=True, predictor_type='mlp')
    table2['MLP_CBF'] = r
    print(f"  MLP+CBF:      Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # PD+CBF K=1
    set_seed(SEED)
    pd_mppi = MPPIController(env, K=1, sigma=0.0)
    r = run_mc_trials(env, pd_mppi, X_SP, enable_cbf=True, predictor_type='baseline')
    table2['PD_CBF_K1'] = r
    print(f"  PD+CBF K=1:   Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    all_results['table2'] = table2

    # ========== Table 3: MPPI vs PTRM K-Scaling ==========
    print("\n" + "=" * 80)
    print("Table 3: MPPI vs PTRM K-Scaling")
    print("=" * 80)

    table3 = {'PTRM': {}, 'MPPI': {}}
    for k in K_VALUES:
        # PTRM
        set_seed(SEED)
        sigma = 0.25 if k > 1 else 0.0
        predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
        table3['PTRM'][k] = r
        print(f"  PTRM K={k:3d}: Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

        # MPPI
        set_seed(SEED)
        sigma_mppi = 2.0 if k > 1 else 0.0
        mppi = MPPIController(env, K=k, sigma=sigma_mppi)
        r = run_mc_trials(env, mppi, X_SP, enable_cbf=True, predictor_type='baseline')
        table3['MPPI'][k] = r
        print(f"  MPPI K={k:3d}: Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    all_results['table3'] = table3

    # ========== Table 4: Robustness (Mismatch + Noise) ==========
    print("\n" + "=" * 80)
    print("Table 4: Robustness (Mismatch + Noise, K=50)")
    print("=" * 80)

    table4 = {}

    # Nominal
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
    table4['nominal'] = r
    print(f"  Nominal:           Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # Mass mismatch
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, predictor, X_SP, enable_cbf=True, use_mismatch=True)
    table4['mismatch'] = r
    print(f"  Mass×1.5+Drag×2:  Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    # Process noise
    for noise in [0.005, 0.01, 0.02, 0.05]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        r = run_mc_trials(env, predictor, X_SP, enable_cbf=True, process_noise=noise)
        table4[f'noise_{noise}'] = r
        print(f"  Noise σ={noise}:    Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}±{r['terminal_error_std']:.4f}m, IAE={r['iae_mean']:.1f}")

    all_results['table4'] = table4

    # ========== Table 6: Multi-Obstacle ==========
    print("\n" + "=" * 80)
    print("Table 6: Multi-Obstacle (K=50, Strong CBF)")
    print("=" * 80)

    table6 = {}
    for env_name, obstacles in OBSTACLE_CONFIGS.items():
        print(f"\n--- {env_name} ---")
        env_obs = QuadrotorDynamics(obstacles=obstacles)
        x_sp_env = ENV_TARGETS[env_name]
        env_results = {}

        # PTRM
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env_obs, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        r = run_mc_trials(env_obs, predictor, x_sp_env, enable_cbf=True)
        env_results['PTRM'] = r
        print(f"  PTRM: Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m")

        # MPPI
        set_seed(SEED)
        mppi = MPPIController(env_obs, K=50, sigma=2.0)
        r = run_mc_trials(env_obs, mppi, x_sp_env, enable_cbf=True, predictor_type='baseline')
        env_results['MPPI'] = r
        print(f"  MPPI: Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m")

        # CEM
        set_seed(SEED)
        cem = CEMController(env_obs, K=50, n_iter=3, sigma=2.0)
        r = run_mc_trials(env_obs, cem, x_sp_env, enable_cbf=True, predictor_type='baseline')
        env_results['CEM'] = r
        print(f"  CEM:  Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m")

        table6[env_name] = env_results

    all_results['table6'] = table6

    # Save all results
    out_path = os.path.join(save_dir, 'retrain_remaining_tables.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至 {out_path}")
    print(f"总耗时: {time.time() - t_start:.0f}s ({(time.time() - t_start)/60:.1f}min)")


if __name__ == '__main__':
    main()
