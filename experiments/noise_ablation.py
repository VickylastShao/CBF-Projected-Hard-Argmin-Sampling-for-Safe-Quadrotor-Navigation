# -*- coding: utf-8 -*-
"""
噪声消融实验 (R2) — 量化双噪声通道的贡献

在 candidate_mode='pd' 下测试三种噪声模式:
  'both'   — 潜在空间 + 输出空间双通道 (默认)
  'latent' — 仅潜在空间扰动
  'output' — 仅输出空间扰动
  'none'   — 无噪声 (确定性TRM推理)

以及不同 K 值下的噪声-K交互效应
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
    QuadrotorDynamics, TRMNMPC,
    PTRMNMPCPredictor
)

SEED = 2026
N_MC = 30          # 30次MC (与v6快速验证一致)
N_STEPS = 300
DT = 0.02
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'results_v6', 'trm_model.pt')


def random_x_init():
    """随机初始状态（6D抽象动力学空间）"""
    p0 = np.random.uniform(-0.1, 0.1, 3)
    v0 = np.random.uniform(-0.05, 0.05, 3)
    return torch.tensor(np.concatenate([p0, v0]), dtype=torch.float32)


def run_single_trial(env, predictor, x_init, x_sp, n_steps):
    """运行单次蒙特卡洛试验，返回指标字典"""
    x = x_init.clone()
    predictor.reset()
    total_iae = 0.0
    success = True

    for step in range(n_steps):
        u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=True)
        x = env.step_discrete(x, u_safe)

        iae_step = torch.norm(x[0:3] - x_sp[0:3]).item()
        total_iae += iae_step

        # 碰撞检测
        p = x[0:3].detach().numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(p - obs['p'])
            if d < obs['r']:
                success = False
                break
        if not success:
            break

    # IAE归一化
    total_iae = total_iae / n_steps

    terminal_err = torch.norm(x[0:3] - x_sp[0:3]).item()
    return {
        'success': success,
        'iae': total_iae,
        'terr': terminal_err,
        'steps_completed': step + 1 if not success else n_steps
    }


def run_noise_ablation():
    """噪声通道消融实验"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载训练好的TRM模型
    print("加载TRM模型...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    trm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    trm_model.to(device)
    trm_model.eval()
    print("模型加载完成")

    env = QuadrotorDynamics()
    noise_modes = ['both', 'latent', 'output', 'none']
    K_values = [1, 10, 50]

    results = {}

    # ===== 实验1: 噪声模式消融 (K=50) =====
    print("\n" + "=" * 80)
    print("实验1: 噪声通道消融 (K=50, Strong CBF)")
    print("=" * 80)

    K = 50
    for nm in noise_modes:
        predictor = PTRMNMPCPredictor(
            model=trm_model, env=env, K=K, D=16, sigma=0.25,
            candidate_mode='pd', noise_mode=nm,
            use_rollout_cost=True, rollout_top_m=10,
            alpha_blend=0.3, pd_sigma=2.0
        )

        successes = 0
        iae_list = []
        terr_list = []

        for trial in range(N_MC):
            x0 = random_x_init()
            res = run_single_trial(env, predictor, x0, X_SP, N_STEPS)
            if res['success']:
                successes += 1
                iae_list.append(res['iae'])
                terr_list.append(res['terr'])

        succ_rate = 100.0 * successes / N_MC
        mean_iae = np.mean(iae_list) if iae_list else float('inf')
        mean_terr = np.mean(terr_list) if terr_list else float('inf')

        print(f"  noise_mode={nm:8s}: Succ={succ_rate:.0f}%, "
              f"TErr={mean_terr:.4f}m, IAE={mean_iae:.1f}")

        results[f'noise_K{K}_{nm}'] = {
            'success_rate': succ_rate,
            'terr': mean_terr,
            'iae': mean_iae,
            'n_success': successes,
            'n_total': N_MC
        }

    # ===== 实验2: 噪声模式 × K 交互消融 =====
    print("\n" + "=" * 80)
    print("实验2: 噪声通道 × K 交互消融")
    print("=" * 80)

    for K in K_values:
        print(f"\n--- K={K} ---")
        for nm in ['both', 'latent', 'output', 'none']:
            predictor = PTRMNMPCPredictor(
                model=trm_model, env=env, K=K, D=16, sigma=0.25,
                candidate_mode='pd', noise_mode=nm,
                use_rollout_cost=True if K > 1 else False,
                rollout_top_m=min(10, K),
                alpha_blend=0.3, pd_sigma=2.0
            )

            successes = 0
            iae_list = []
            terr_list = []

            for trial in range(N_MC):
                x0 = random_x_init()
                res = run_single_trial(env, predictor, x0, X_SP, N_STEPS)
                if res['success']:
                    successes += 1
                    iae_list.append(res['iae'])
                    terr_list.append(res['terr'])

            succ_rate = 100.0 * successes / N_MC
            mean_iae = np.mean(iae_list) if iae_list else float('inf')
            mean_terr = np.mean(terr_list) if terr_list else float('inf')

            print(f"  noise_mode={nm:8s}: Succ={succ_rate:.0f}%, "
                  f"TErr={mean_terr:.4f}m, IAE={mean_iae:.1f}")

            results[f'noise_K{K}_{nm}'] = {
                'success_rate': succ_rate,
                'terr': mean_terr,
                'iae': mean_iae,
                'n_success': successes,
                'n_total': N_MC
            }

    # 保存结果
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

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

    with open(os.path.join(save_dir, 'noise_ablation_results.json'), 'w') as f:
        json.dump(strip(results), f, indent=2)

    print(f"\n结果已保存至 {save_dir}/noise_ablation_results.json")
    return results


if __name__ == '__main__':
    t0 = time.time()
    results = run_noise_ablation()
    elapsed = time.time() - t0
    print(f"\n总实验时间: {elapsed:.1f}s ({elapsed/60:.1f}min)")
