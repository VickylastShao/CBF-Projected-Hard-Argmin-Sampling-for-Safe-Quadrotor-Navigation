# -*- coding: utf-8 -*-
"""
v6 快速验证脚本 — 减少MC次数和K值，验证完整pipeline
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
N_MC = 20          # 快速验证用
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

        # IAE归一化: IAE = (1/T) * Σ ||e|| * dt = Σ ||e|| / N
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
    colls = [r['collision'] for r in results]
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    return {
        'success_rate': np.mean(succs) * 100,
        'collision_rate': np.mean(colls) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'min_distance_mean': np.mean([r['min_distance'] for r in results]),
    }


def main():
    set_seed(SEED)
    t_start = time.time()

    # 加载/训练模型
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    os.makedirs(save_dir, exist_ok=True)

    trm_path = os.path.join(save_dir, 'trm_model.pt')
    mlp_path = os.path.join(save_dir, 'mlp_model.pt')
    device = torch.device('cpu')

    if os.path.exists(trm_path):
        print("加载TRM模型...")
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    else:
        print("训练TRM模型...")
        env_train = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env_train, horizon=10)
        dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                              x_sp=X_SP, pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model, _ = train_trm_jointly(trm_model, dataset, env_train, epochs=100, lr=0.001, patience=20, verbose=True)
        torch.save(trm_model.state_dict(), trm_path)

    if os.path.exists(mlp_path):
        print("加载MLP模型...")
        mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
    else:
        print("训练MLP模型...")
        env_train = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env_train, horizon=10)
        dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                              x_sp=X_SP, pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])
        mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
        mlp_model = train_mlp(mlp_model, dataset, env_train, epochs=100, lr=0.001, patience=20, verbose=True)
        torch.save(mlp_model.state_dict(), mlp_path)

    trm_model.eval()
    mlp_model.eval()

    # ========== Exp 1: K-Scaling × CBF ==========
    print("\n" + "=" * 80)
    print("实验 1: K-Scaling × CBF 强度消融 (PD候选 + TRM评估)")
    print("=" * 80)

    cbf_configs = {
        'StrongCBF': {'alpha_d': 0.8, 'gamma_d': 0.2, 'enable_cbf': True},
        'NoCBF': {'alpha_d': 0.0, 'gamma_d': 0.0, 'enable_cbf': False},
    }

    exp1 = {}
    for cbf_name, cbf_cfg in cbf_configs.items():
        print(f"\n--- {cbf_name} ---")
        env = QuadrotorDynamics()
        env.alpha_d = cbf_cfg['alpha_d']
        env.gamma_d = cbf_cfg['gamma_d']
        cbf_results = {}
        for k in K_VALUES:
            set_seed(SEED)
            sigma = 0.25 if k > 1 else 0.0
            predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                           alpha_blend=0.3, candidate_mode='pd',
                                           pd_sigma=2.0, use_rollout_cost=True)
            result = run_mc_trials(env, predictor, X_SP, enable_cbf=cbf_cfg['enable_cbf'])
            cbf_results[k] = result
            print(f"  K={k:3d}: Succ={result['success_rate']:.0f}%, TErr={result['terminal_error_mean']:.4f}m, IAE={result['iae_mean']:.1f}")
        exp1[cbf_name] = cbf_results

    # ========== Exp 5a: 候选模式消融 ==========
    print("\n" + "=" * 80)
    print("实验 5a: 候选模式消融 (K=50, Strong CBF)")
    print("=" * 80)

    env = QuadrotorDynamics()
    exp5a = {}
    for cmode, label in [('pd', 'PD+TRM-Eval'), ('trm', 'TRM+PD-Corr')]:
        for alpha in [0.3, 0.5, 0.9]:
            set_seed(SEED)
            predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                           alpha_blend=alpha, noise_mode='both',
                                           candidate_mode=cmode,
                                           pd_sigma=2.0 if cmode == 'pd' else 0.0,
                                           use_rollout_cost=True)
            result = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
            exp5a[f'{label}_a{alpha}'] = result
            print(f"  {label:14s} α={alpha:.1f}: Succ={result['success_rate']:.0f}%, IAE={result['iae_mean']:.1f}")

    # ========== Exp 5c: Q-head vs Rollout 消融 ==========
    print("\n" + "=" * 80)
    print("实验 5c: Q-head vs Rollout 评估消融 (K=50, PD候选, Strong CBF)")
    print("=" * 80)

    exp5c = {}
    for use_rollout, label in [(True, 'Q+Rollout'), (False, 'Q-only')]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=use_rollout)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
        exp5c[label] = result
        print(f"  {label:10s}: Succ={result['success_rate']:.0f}%, TErr={result['terminal_error_mean']:.4f}m, IAE={result['iae_mean']:.1f}")

    # ========== Exp 7: Baseline Comparison ==========
    print("\n" + "=" * 80)
    print("实验 7: 基线对比 (Strong CBF)")
    print("=" * 80)

    env = QuadrotorDynamics()
    exp7 = {}

    # PTRM
    ptrm_results = {}
    for k in K_VALUES:
        set_seed(SEED)
        sigma = 0.25 if k > 1 else 0.0
        predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
        ptrm_results[k] = result
        print(f"  PTRM K={k:3d}: Succ={result['success_rate']:.0f}%, TErr={result['terminal_error_mean']:.4f}m")
    exp7['PTRM'] = ptrm_results

    # MPPI
    mppi_results = {}
    for k in K_VALUES:
        set_seed(SEED)
        sigma_mppi = 2.0 if k > 1 else 0.0
        mppi = MPPIController(env, K=k, sigma=sigma_mppi)
        result = run_mc_trials(env, mppi, X_SP, enable_cbf=True, predictor_type='baseline')
        mppi_results[k] = result
        print(f"  MPPI K={k:3d}: Succ={result['success_rate']:.0f}%, TErr={result['terminal_error_mean']:.4f}m")
    exp7['MPPI'] = mppi_results

    # MLP+CBF
    set_seed(SEED)
    mlp_pred = MLPPredictor(mlp_model, env, alpha_blend=0.3)
    mlp_result = run_mc_trials(env, mlp_pred, X_SP, enable_cbf=True, predictor_type='mlp')
    exp7['MLP+CBF'] = mlp_result
    print(f"  MLP+CBF:  Succ={mlp_result['success_rate']:.0f}%, TErr={mlp_result['terminal_error_mean']:.4f}m")

    # PD+CBF
    set_seed(SEED)
    pd_mppi = MPPIController(env, K=1, sigma=0.0)
    pd_result = run_mc_trials(env, pd_mppi, X_SP, enable_cbf=True, predictor_type='baseline')
    exp7['PD+CBF'] = pd_result
    print(f"  PD+CBF:   Succ={pd_result['success_rate']:.0f}%, TErr={pd_result['terminal_error_mean']:.4f}m")

    # CEM
    set_seed(SEED)
    cem = CEMController(env, K=50, n_iter=3, sigma=2.0)
    cem_result = run_mc_trials(env, cem, X_SP, enable_cbf=True, predictor_type='baseline')
    exp7['CEM'] = cem_result
    print(f"  CEM K=50: Succ={cem_result['success_rate']:.0f}%, TErr={cem_result['terminal_error_mean']:.4f}m")

    # ========== Exp 8: Multi-Obstacle ==========
    print("\n" + "=" * 80)
    print("实验 8: 多障碍物配置 (K=50, Strong CBF)")
    print("=" * 80)

    exp8 = {}
    for env_name, obstacles in OBSTACLE_CONFIGS.items():
        print(f"\n--- {env_name} ---")
        env_obs = QuadrotorDynamics(obstacles=obstacles)
        x_sp_env = ENV_TARGETS[env_name]

        # PTRM
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env_obs, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        ptrm_res = run_mc_trials(env_obs, predictor, x_sp_env, enable_cbf=True)

        # MPPI
        set_seed(SEED)
        mppi = MPPIController(env_obs, K=50, sigma=2.0)
        mppi_res = run_mc_trials(env_obs, mppi, x_sp_env, enable_cbf=True, predictor_type='baseline')

        # MLP+CBF
        set_seed(SEED)
        mlp_pred = MLPPredictor(mlp_model, env_obs, alpha_blend=0.3)
        mlp_res = run_mc_trials(env_obs, mlp_pred, x_sp_env, enable_cbf=True, predictor_type='mlp')

        exp8[env_name] = {'ptrm': ptrm_res, 'mppi': mppi_res, 'mlp': mlp_res}
        print(f"  PTRM: Succ={ptrm_res['success_rate']:.0f}%, TErr={ptrm_res['terminal_error_mean']:.4f}m")
        print(f"  MPPI: Succ={mppi_res['success_rate']:.0f}%, TErr={mppi_res['terminal_error_mean']:.4f}m")
        print(f"  MLP:  Succ={mlp_res['success_rate']:.0f}%, TErr={mlp_res['terminal_error_mean']:.4f}m")

    # ========== Exp 9: Theory Verification ==========
    print("\n" + "=" * 80)
    print("实验 9: Q-head 排序相关性")
    print("=" * 80)

    from scipy.stats import spearmanr, pearsonr
    trm_model_eval = trm_model
    trm_model_eval.eval()

    q_scores_all = []
    rollout_costs_all = []

    for _ in range(50):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)
        with torch.no_grad():
            X = torch.cat([x_init, x_sp_dev]).unsqueeze(0).repeat(50, 1)
            y_history = trm_model_eval.forward_steps(X, D=16, noise_scale=0.25)
        u_candidates, final_latent_y = y_history[-1]
        q_scores = trm_model_eval.f_Q(final_latent_y).squeeze(-1).detach().cpu().numpy()

        env_temp = QuadrotorDynamics()
        q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        rollout_costs = []
        for k_idx in range(50):
            u_k = u_candidates[k_idx].cpu()
            cost = 0.0
            x_r = x_init.cpu().clone()
            for i in range(10):
                u_i = torch.clamp(u_k[i*3:(i+1)*3], env_temp.u_min, env_temp.u_max)
                x_r = env_temp.step_discrete(x_r, u_i)
                error = x_r - X_SP
                cost += torch.sum(q_diag * error * error).item() + 0.02 * torch.sum(u_i * u_i).item()
            rollout_costs.append(cost)

        q_scores_all.extend(q_scores.tolist())
        rollout_costs_all.extend(rollout_costs)

    q_arr = np.array(q_scores_all)
    c_arr = np.array(rollout_costs_all)
    spearman_rho, sp_p = spearmanr(q_arr, c_arr)
    pearson_r, pe_p = pearsonr(q_arr, c_arr)
    print(f"  Spearman ρ = {spearman_rho:.4f} (p={sp_p:.2e})")
    print(f"  Pearson r  = {pearson_r:.4f} (p={pe_p:.2e})")

    # ========== 汇总 ==========
    t_total = time.time() - t_start
    print(f"\n总实验时间: {t_total:.1f}s ({t_total/60:.1f}min)")

    # 保存结果
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results = {
        'exp1_k_scaling': strip(exp1),
        'exp5a_candidate_mode': strip(exp5a),
        'exp5c_eval_ablation': strip(exp5c),
        'exp7_baselines': strip(exp7),
        'exp8_multi_obstacle': strip(exp8),
        'exp9_qhead_corr': {
            'spearman_rho': float(spearman_rho),
            'pearson_r': float(pearson_r),
        }
    }

    with open(os.path.join(save_dir, 'quick_test_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存至 {save_dir}/quick_test_results.json")
    print("v6 快速验证完成!")


if __name__ == '__main__':
    main()
