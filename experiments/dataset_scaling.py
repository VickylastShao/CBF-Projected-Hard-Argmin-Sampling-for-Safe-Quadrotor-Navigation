# -*- coding: utf-8 -*-
"""
S6: 数据集大小缩放实验 — 训练不同数据集大小的TRM模型并评估性能
Dataset size scaling curve: success rate vs. N_samples
"""

import sys
import os
import time
import json
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_core import (
    QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
    PTRMNMPCPredictor, generate_quadrotor_dataset, train_trm_jointly
)

SEED = 2026
N_MC = 20
N_STEPS = 300
DT = 0.02
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5

# 测试的数据集大小
DATASET_SIZES = [25, 50, 100, 200, 500]

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
    """运行蒙特卡洛试验"""
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
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    return {
        'success_rate': np.mean(succs) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
    }


def main():
    set_seed(SEED)
    t_start = time.time()
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cpu')

    # 先生成最大数据集
    print("=" * 60)
    print("生成专家数据集 (N=500)...")
    print("=" * 60)
    env_train = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env_train, horizon=10)
    full_dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                               x_sp=X_SP,
                                               pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])

    all_results = {}

    for n_samples in DATASET_SIZES:
        print(f"\n{'=' * 60}")
        print(f"训练数据集大小 N={n_samples}...")
        print(f"{'=' * 60}")

        set_seed(SEED)

        # 截取子数据集
        sub_dataset = full_dataset[:n_samples]

        # 训练TRM模型
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model, _ = train_trm_jointly(trm_model, sub_dataset, env_train,
                                           epochs=100, lr=0.001, patience=20, verbose=False)
        trm_model.eval()

        # 创建predictor（必须指定candidate_mode='pd'，默认'trm'模式失败）
        env_eval = QuadrotorDynamics()
        predictor = PTRMNMPCPredictor(trm_model, env_eval, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)

        # 评估 (K=50, Strong CBF)
        print(f"  评估 N={n_samples}, K=50, Strong CBF...")
        res_strong = run_mc_trials(env_eval, predictor, X_SP, enable_cbf=True)

        # 评估 (K=50, No CBF)
        print(f"  评估 N={n_samples}, K=50, No CBF...")
        res_no_cbf = run_mc_trials(env_eval, predictor, X_SP, enable_cbf=False)

        # 评估 (K=1, Strong CBF)
        predictor_k1 = PTRMNMPCPredictor(trm_model, env_eval, K=1, D=16, sigma=0.0,
                                          alpha_blend=0.3, candidate_mode='pd',
                                          pd_sigma=2.0, use_rollout_cost=True)
        print(f"  评估 N={n_samples}, K=1, Strong CBF...")
        res_k1 = run_mc_trials(env_eval, predictor_k1, X_SP, enable_cbf=True)

        all_results[f"N{n_samples}_K50_cbf"] = res_strong
        all_results[f"N{n_samples}_K50_nocbf"] = res_no_cbf
        all_results[f"N{n_samples}_K1_cbf"] = res_k1

        print(f"  N={n_samples} K=50 CBF: Succ={res_strong['success_rate']:.0f}%, "
              f"TErr={res_strong['terminal_error_mean']:.4f}, IAE={res_strong['iae_mean']:.1f}")
        print(f"  N={n_samples} K=50 NoCBF: Succ={res_no_cbf['success_rate']:.0f}%, "
              f"TErr={res_no_cbf['terminal_error_mean']:.4f}, IAE={res_no_cbf['iae_mean']:.1f}")
        print(f"  N={n_samples} K=1 CBF: Succ={res_k1['success_rate']:.0f}%, "
              f"TErr={res_k1['terminal_error_mean']:.4f}, IAE={res_k1['iae_mean']:.1f}")

    # 保存结果
    output_path = os.path.join(save_dir, 'dataset_scaling_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"数据集缩放实验完成! 耗时 {elapsed:.1f}s")
    print(f"结果保存到: {output_path}")
    print(f"{'=' * 60}")

    # 打印汇总表格
    print("\n| N_samples | K | CBF | Succ (%) | TErr (m) | IAE |")
    print("|-----------|-----|-----|----------|----------|-----|")
    for n in DATASET_SIZES:
        for key_label, key_suffix in [("50", "K50_cbf"), ("50-No", "K50_nocbf"), ("1", "K1_cbf")]:
            k = f"N{n}_{key_suffix}"
            if k in all_results:
                r = all_results[k]
                print(f"| {n} | {key_label} | {'Yes' if 'nocbf' not in key_suffix else 'No'} | "
                      f"{r['success_rate']:.0f} | {r['terminal_error_mean']:.4f} | {r['iae_mean']:.1f} |")


if __name__ == '__main__':
    main()
