# -*- coding: utf-8 -*-
"""
路径D验证：去Q-head架构 — TRM基线+纯Rollout评估

核心架构：
  TRM递归 → 确定性策略基线 u_base
  u_base + Gaussian噪声 → K个候选
  纯Rollout评估 → 选最优
  CBF安全过滤

对比实验：
  1. TRM-Rollout vs MPPI vs CEM vs PD+Rollout (不同基线的效果)
  2. K-scaling (10, 25, 50, 100)
  3. TRM基线 vs PD基线 + 相同Rollout (证明TRM递归的价值)
  4. 噪声鲁棒性 (模型失配+过程噪声)
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
    PTRMNMPCPredictor, generate_quadrotor_dataset,
    train_trm_jointly,
)
from experiments.baselines.mppi_controller import MPPIController
from experiments.baselines.cem_controller import CEMController

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


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True,
                   use_mismatch=False, process_noise=0.0):
    results = []
    for _ in range(n_mc):
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()
        collision = False; min_dist = float('inf'); iae = 0.0

        for step in range(N_STEPS):
            ret = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            u_safe = ret[0] if isinstance(ret, tuple) else ret
            x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)
            p_np = x[0:3].detach().numpy()
            for obs in env.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                min_dist = min(min_dist, d)
                if d < 0: collision = True
            iae += torch.norm(x[0:3] - x_sp[0:3]).item()

        # IAE归一化
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
        'success_rate': np.mean(succs) * 100,
        'collision_rate': np.mean([r['collision'] for r in results]) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'min_distance_mean': np.mean([r['min_distance'] for r in results]),
    }


def main():
    set_seed(SEED)
    t_start = time.time()

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    device = torch.device('cpu')

    # ==================================================================
    # 加载已训练模型
    # ==================================================================
    print("=" * 80)
    print("路径D验证：去Q-head架构 — TRM基线+纯Rollout评估")
    print("=" * 80)

    env = QuadrotorDynamics()
    trm = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    model_path = os.path.join(save_dir, 'trm_model.pt')
    trm.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    trm.eval()
    print(f"TRM模型已加载: {model_path}")

    all_results = {}

    # ==================================================================
    # 实验1: 控制性能对比
    # ==================================================================
    print("\n" + "=" * 60)
    print("实验1: 控制性能对比 (TRM-Rollout vs 基线)")
    print("=" * 60)

    # 1a. TRM-Rollout (路径D: TRM基线 + 纯Rollout排序)
    set_seed(SEED)
    pred_trm_rollout = PTRMNMPCPredictor(trm, env, K=50, D=16, sigma=0.25,
                                           candidate_mode='trm_rollout',
                                           pd_sigma=2.0,
                                           use_rollout_cost=True, ranking_mode='q_head')
    r = run_mc_trials(env, pred_trm_rollout, X_SP)
    all_results['TRM-Rollout K=50'] = r
    print(f"  TRM-Rollout K=50:  Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 1b. PD+Rollout (PD基线 + 纯Rollout排序 — 消融：基线来源)
    set_seed(SEED)
    pred_pd_rollout = PTRMNMPCPredictor(trm, env, K=50, D=16, sigma=0.25,
                                          candidate_mode='pd',
                                          pd_sigma=2.0,
                                          use_rollout_cost=True, ranking_mode='rollout_all')
    r = run_mc_trials(env, pred_pd_rollout, X_SP)
    all_results['PD-Rollout K=50'] = r
    print(f"  PD-Rollout K=50:   Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 1c. 原PTRM-v1 (PD+Gaussian候选 + Q-head粗筛+Rollout精排)
    set_seed(SEED)
    pred_ptrm_v1 = PTRMNMPCPredictor(trm, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
    r = run_mc_trials(env, pred_ptrm_v1, X_SP)
    all_results['PTRM-v1 K=50'] = r
    print(f"  PTRM-v1 K=50:      Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 1d. MPPI K=50
    set_seed(SEED)
    mppi = MPPIController(env, K=50, sigma=2.0, lam=0.1, rollout_steps=20)
    r = run_mc_trials(env, mppi, X_SP)
    all_results['MPPI K=50'] = r
    print(f"  MPPI K=50:         Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 1e. CEM K=50
    set_seed(SEED)
    cem = CEMController(env, K=50, n_iter=3, alpha=0.1, sigma_init=2.0, rollout_steps=20)
    r = run_mc_trials(env, cem, X_SP)
    all_results['CEM K=50'] = r
    print(f"  CEM K=50:          Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 1f. TRM确定性 (K=1，无test-time scaling)
    set_seed(SEED)
    pred_trm_k1 = PTRMNMPCPredictor(trm, env, K=1, D=16, sigma=0.0,
                                      candidate_mode='trm_rollout',
                                      use_rollout_cost=False)
    r = run_mc_trials(env, pred_trm_k1, X_SP)
    all_results['TRM K=1 (det)'] = r
    print(f"  TRM K=1 (det):     Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 1g. PD+CBF (K=1，纯PD基线)
    set_seed(SEED)
    pred_pd_k1 = PTRMNMPCPredictor(trm, env, K=1, D=16, sigma=0.0,
                                     candidate_mode='pd',
                                     use_rollout_cost=False)
    r = run_mc_trials(env, pred_pd_k1, X_SP)
    all_results['PD+CBF K=1'] = r
    print(f"  PD+CBF K=1:        Succ={r['success_rate']:.0f}%, "
          f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # ==================================================================
    # 实验2: K-scaling (test-time compute scaling)
    # ==================================================================
    print("\n" + "=" * 60)
    print("实验2: K-scaling (TRM-Rollout vs MPPI)")
    print("=" * 60)

    k_values = [1, 5, 10, 25, 50, 100]
    k_scaling_results = {}

    for K in k_values:
        # TRM-Rollout
        set_seed(SEED)
        pred = PTRMNMPCPredictor(trm, env, K=K, D=16, sigma=0.25,
                                  candidate_mode='trm_rollout',
                                  pd_sigma=2.0,
                                  use_rollout_cost=True if K > 1 else False)
        r_trm = run_mc_trials(env, pred, X_SP)

        # MPPI
        set_seed(SEED)
        mppi_k = MPPIController(env, K=K, sigma=2.0, lam=0.1, rollout_steps=20)
        r_mppi = run_mc_trials(env, mppi_k, X_SP)

        k_scaling_results[K] = {'TRM-Rollout': r_trm, 'MPPI': r_mppi}
        print(f"  K={K:3d} | TRM-Rollout: Succ={r_trm['success_rate']:.0f}%, "
              f"TErr={r_trm['terminal_error_mean']:.4f}m, IAE={r_trm['iae_mean']:.1f} | "
              f"MPPI: Succ={r_mppi['success_rate']:.0f}%, "
              f"TErr={r_mppi['terminal_error_mean']:.4f}m, IAE={r_mppi['iae_mean']:.1f}")

    # ==================================================================
    # 实验3: TRM基线 vs PD基线 + 相同Rollout排序
    # (证明TRM递归对策略基线的贡献)
    # ==================================================================
    print("\n" + "=" * 60)
    print("实验3: 基线消融 (TRM基线 vs PD基线，相同Rollout排序)")
    print("=" * 60)

    # 两组都用纯Rollout排序，唯一差异是候选基线来源
    set_seed(SEED)
    pred_trm_base = PTRMNMPCPredictor(trm, env, K=50, D=16, sigma=0.25,
                                       candidate_mode='trm_rollout',
                                       pd_sigma=2.0,
                                       use_rollout_cost=True)
    r_trm_base = run_mc_trials(env, pred_trm_base, X_SP)

    set_seed(SEED)
    pred_pd_base = PTRMNMPCPredictor(trm, env, K=50, D=16, sigma=0.25,
                                      candidate_mode='pd',
                                      pd_sigma=2.0,
                                      use_rollout_cost=True, ranking_mode='rollout_all')
    r_pd_base = run_mc_trials(env, pred_pd_base, X_SP)

    all_results['基线消融-TRM'] = r_trm_base
    all_results['基线消融-PD'] = r_pd_base

    print(f"  TRM基线+Rollout: Succ={r_trm_base['success_rate']:.0f}%, "
          f"TErr={r_trm_base['terminal_error_mean']:.4f}m, IAE={r_trm_base['iae_mean']:.1f}")
    print(f"  PD基线+Rollout:  Succ={r_pd_base['success_rate']:.0f}%, "
          f"TErr={r_pd_base['terminal_error_mean']:.4f}m, IAE={r_pd_base['iae_mean']:.1f}")

    if r_trm_base['iae_mean'] > 0 and r_pd_base['iae_mean'] > 0:
        trm_adv = (r_pd_base['iae_mean'] - r_trm_base['iae_mean']) / r_trm_base['iae_mean'] * 100
        print(f"  TRM vs PD基线: ΔIAE = {trm_adv:+.1f}% ({'TRM更优' if trm_adv > 0 else 'PD更优'})")

    # ==================================================================
    # 实验4: 噪声鲁棒性 (模型失配 + 过程噪声)
    # ==================================================================
    print("\n" + "=" * 60)
    print("实验4: 噪声鲁棒性")
    print("=" * 60)

    noise_results = {}

    for use_mismatch, process_noise, label in [
        (False, 0.0, 'Nominal'),
        (True, 0.0, '+50% Mass/Drag'),
        (False, 0.3, 'ProcessNoise=0.3'),
        (True, 0.3, 'Both'),
    ]:
        set_seed(SEED)
        pred = PTRMNMPCPredictor(trm, env, K=50, D=16, sigma=0.25,
                                  candidate_mode='trm_rollout',
                                  pd_sigma=2.0,
                                  use_rollout_cost=True)
        r = run_mc_trials(env, pred, X_SP, use_mismatch=use_mismatch, process_noise=process_noise)
        noise_results[label] = r
        print(f"  {label:20s}: Succ={r['success_rate']:.0f}%, "
              f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # ==================================================================
    # 最终结论
    # ==================================================================
    print("\n" + "=" * 80)
    print("路径D验证结论")
    print("=" * 80)

    trm_r = all_results.get('TRM-Rollout K=50', {})
    mppi_r = all_results.get('MPPI K=50', {})
    pd_r = all_results.get('PD-Rollout K=50', {})

    print(f"\n  1. TRM-Rollout vs MPPI:")
    print(f"     TRM-Rollout: Succ={trm_r.get('success_rate',0):.0f}%, "
          f"TErr={trm_r.get('terminal_error_mean',0):.4f}m, IAE={trm_r.get('iae_mean',0):.1f}")
    print(f"     MPPI:         Succ={mppi_r.get('success_rate',0):.0f}%, "
          f"TErr={mppi_r.get('terminal_error_mean',0):.4f}m, IAE={mppi_r.get('iae_mean',0):.1f}")

    print(f"\n  2. TRM基线 vs PD基线 (相同Rollout排序):")
    print(f"     TRM基线: Succ={r_trm_base['success_rate']:.0f}%, "
          f"TErr={r_trm_base['terminal_error_mean']:.4f}m, IAE={r_trm_base['iae_mean']:.1f}")
    print(f"     PD基线:  Succ={r_pd_base['success_rate']:.0f}%, "
          f"TErr={r_pd_base['terminal_error_mean']:.4f}m, IAE={r_pd_base['iae_mean']:.1f}")

    # K-scaling 总结
    print(f"\n  3. K-scaling (TRM-Rollout):")
    for K in k_values:
        ks = k_scaling_results.get(K, {}).get('TRM-Rollout', {})
        ms = k_scaling_results.get(K, {}).get('MPPI', {})
        print(f"     K={K:3d}: TRM TE={ks.get('terminal_error_mean',0):.4f}m, "
              f"MPPI TE={ms.get('terminal_error_mean',0):.4f}m")

    # 噪声鲁棒性
    print(f"\n  4. 噪声鲁棒性:")
    for label, r in noise_results.items():
        print(f"     {label:20s}: Succ={r['success_rate']:.0f}%, IAE={r['iae_mean']:.1f}")

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

    save_data = {
        'exp1_performance_comparison': strip(all_results),
        'exp2_k_scaling': strip(k_scaling_results),
        'exp3_baseline_ablation': {
            'TRM_baseline': strip(r_trm_base),
            'PD_baseline': strip(r_pd_base),
        },
        'exp4_noise_robustness': strip(noise_results),
    }

    results_path = os.path.join(save_dir, 'path_d_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n结果已保存至 {results_path}")


if __name__ == '__main__':
    main()
