# -*- coding: utf-8 -*-
"""
R5 扩展多障碍物实验 — 5种障碍物配置，包含5-10个障碍物和多种同伦类
Extended multi-obstacle experiments per reviewer R5:
- Dense-8: 8 spheres (5-10 obstacles)
- Dense-10: 10 spheres (5-10 obstacles)
- Multi-Homotopy: 6 spheres forcing K-candidate path discovery
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
from baselines import MPPIController, CEMController

SEED = 2026
N_MC = 30
N_STEPS = 300
DT = 0.02
TErr_THRESH = 0.5

# 扩展障碍物配置（包含5-10个障碍物和多种同伦类）
OBSTACLE_CONFIGS = {
    'Corridor': {
        'obstacles': [
            {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
            {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
            {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
        ],
        'x_sp': torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        'x_init_range': [(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)],
        'description': '3 overlapping spheres (default)'
    },
    'Dense-5': {
        'obstacles': [
            {"p": np.array([0.8, 1.0, 0.8]), "r": 0.4},
            {"p": np.array([1.5, 0.8, 1.5]), "r": 0.35},
            {"p": np.array([2.0, 1.5, 1.0]), "r": 0.4},
            {"p": np.array([1.2, 2.0, 2.0]), "r": 0.35},
            {"p": np.array([1.8, 2.5, 1.5]), "r": 0.3}
        ],
        'x_sp': torch.tensor([2.5, 3.0, 2.5, 0.0, 0.0, 0.0], dtype=torch.float32),
        'x_init_range': [(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)],
        'description': '5 spheres (r=0.3-0.4)'
    },
    'Dense-8': {
        'obstacles': [
            {"p": np.array([0.6, 0.7, 0.6]), "r": 0.3},
            {"p": np.array([1.0, 1.0, 1.2]), "r": 0.35},
            {"p": np.array([1.4, 0.6, 0.8]), "r": 0.3},
            {"p": np.array([1.8, 1.2, 1.6]), "r": 0.35},
            {"p": np.array([2.2, 1.8, 1.0]), "r": 0.3},
            {"p": np.array([1.0, 2.2, 2.0]), "r": 0.3},
            {"p": np.array([1.6, 2.5, 1.8]), "r": 0.3},
            {"p": np.array([2.0, 2.8, 2.2]), "r": 0.3}
        ],
        'x_sp': torch.tensor([2.8, 3.5, 2.8, 0.0, 0.0, 0.0], dtype=torch.float32),
        'x_init_range': [(-0.5, 1.0), (-1.0, 0.0), (-0.5, 1.0)],
        'description': '8 spheres (r=0.3-0.35)'
    },
    'Dense-10': {
        'obstacles': [
            {"p": np.array([0.5, 0.6, 0.5]), "r": 0.25},
            {"p": np.array([0.8, 1.0, 1.0]), "r": 0.3},
            {"p": np.array([1.2, 0.5, 0.7]), "r": 0.25},
            {"p": np.array([1.5, 1.2, 1.4]), "r": 0.3},
            {"p": np.array([1.8, 0.8, 0.9]), "r": 0.25},
            {"p": np.array([2.1, 1.6, 1.8]), "r": 0.3},
            {"p": np.array([1.0, 2.0, 2.0]), "r": 0.25},
            {"p": np.array([1.5, 2.5, 1.5]), "r": 0.3},
            {"p": np.array([2.0, 2.2, 2.2]), "r": 0.25},
            {"p": np.array([2.3, 2.8, 2.5]), "r": 0.25}
        ],
        'x_sp': torch.tensor([3.0, 3.5, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        'x_init_range': [(-0.5, 1.0), (-1.0, 0.0), (-0.5, 1.0)],
        'description': '10 spheres (r=0.25-0.3)'
    },
    'Multi-Homotopy': {
        # 多同伦类：两个独立通道，K候选机制需发现根本不同的路径
        # 通道1：y < 1.0（下方通道）
        # 通道2：y > 2.0（上方通道）
        # 中间区域被障碍物填充
        'obstacles': [
            # 上方通道边界
            {"p": np.array([1.0, 1.5, 1.0]), "r": 0.45},
            {"p": np.array([1.8, 1.5, 1.5]), "r": 0.45},
            # 下方通道边界
            {"p": np.array([1.0, 0.5, 2.0]), "r": 0.4},
            {"p": np.array([1.8, 0.5, 1.5]), "r": 0.4},
            # 中间障碍物
            {"p": np.array([1.4, 1.0, 1.2]), "r": 0.35},
        ],
        'x_sp': torch.tensor([2.5, 1.0, 1.5, 0.0, 0.0, 0.0], dtype=torch.float32),
        'x_init_range': [(-0.5, 0.5), (-0.5, 0.5), (0.5, 2.5)],
        'description': '6 spheres, dual-channel homotopy (upper/lower path)'
    },
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init(x_init_range):
    return torch.tensor([
        np.random.uniform(*x_init_range[0]),
        np.random.uniform(*x_init_range[1]),
        np.random.uniform(*x_init_range[2]),
        np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


def run_mc_trials(env, predictor, x_sp, x_init_range, n_mc=N_MC, enable_cbf=True):
    """运行蒙特卡洛试验"""
    results = []
    for _ in range(n_mc):
        x_init = random_x_init(x_init_range)
        predictor.reset()
        x = x_init.clone()
        collision = False; min_dist = float('inf'); iae = 0.0

        for step in range(N_STEPS):
            result = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            if isinstance(result, tuple):
                u_safe = result[0]
            else:
                u_safe = result
            x = env.step_discrete(x, u_safe)
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
        'success_rate_std': np.std(succs) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'min_distance_mean': np.mean([r['min_distance'] for r in results]),
        'n_success': sum(succs),
        'n_total': len(results),
    }


def main():
    set_seed(SEED)
    t_start = time.time()
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cpu')

    # 加载TRM模型
    trm_path = os.path.join(os.path.dirname(__file__), 'results_v6', 'trm_model.pt')
    if not os.path.exists(trm_path):
        trm_path = os.path.join(os.path.dirname(__file__), 'results', 'trm_model.pt')

    if os.path.exists(trm_path):
        print("加载TRM模型...")
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    else:
        print("训练TRM模型...")
        env_train = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env_train, horizon=10)
        X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                              x_sp=X_SP,
                                              pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model, _ = train_trm_jointly(trm_model, dataset, env_train,
                                           epochs=100, lr=0.001, patience=20, verbose=True)
        torch.save(trm_model.state_dict(), os.path.join(save_dir, 'trm_model.pt'))

    trm_model.eval()

    all_results = {}

    for env_name, config in OBSTACLE_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"环境: {env_name} — {config['description']}")
        print(f"{'=' * 60}")

        # 创建环境
        env = QuadrotorDynamics(obstacles=config['obstacles'])
        x_sp = config['x_sp']
        x_init_range = config['x_init_range']

        # PTRM K=50 Strong CBF
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        print(f"  PTRM K=50 CBF...")
        res_ptrm = run_mc_trials(env, predictor, x_sp, x_init_range, enable_cbf=True)
        all_results[f"{env_name}_ptrm_K50_cbf"] = res_ptrm
        print(f"    Succ={res_ptrm['success_rate']:.0f}%  TErr={res_ptrm['terminal_error_mean']:.4f}  "
              f"IAE={res_ptrm['iae_mean']:.1f}")

        # MPPI K=50 Strong CBF
        mppi = MPPIController(env, K=50)
        print(f"  MPPI K=50 CBF...")
        res_mppi = run_mc_trials(env, mppi, x_sp, x_init_range, enable_cbf=True)
        all_results[f"{env_name}_mppi_K50_cbf"] = res_mppi
        print(f"    Succ={res_mppi['success_rate']:.0f}%  TErr={res_mppi['terminal_error_mean']:.4f}  "
              f"IAE={res_mppi['iae_mean']:.1f}")

        # CEM K=50 Strong CBF
        cem = CEMController(env, K=50, n_iter=3)
        print(f"  CEM K=50 CBF...")
        res_cem = run_mc_trials(env, cem, x_sp, x_init_range, enable_cbf=True)
        all_results[f"{env_name}_cem_K50_cbf"] = res_cem
        print(f"    Succ={res_cem['success_rate']:.0f}%  TErr={res_cem['terminal_error_mean']:.4f}  "
              f"IAE={res_cem['iae_mean']:.1f}")

    # 保存结果
    output_path = os.path.join(save_dir, 'multi_obstacle_extended_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"扩展多障碍物实验完成! 耗时 {elapsed:.1f}s")
    print(f"结果保存到: {output_path}")
    print(f"{'=' * 60}")

    # 打印汇总表格
    print("\n| Environment | Obstacles | PTRM Succ (%) | PTRM TErr | PTRM IAE | MPPI Succ (%) | MPPI TErr | CEM Succ (%) | CEM TErr |")
    print("|-------------|-----------|--------------|-----------|----------|--------------|-----------|-------------|----------|")
    for env_name, config in OBSTACLE_CONFIGS.items():
        n_obs = len(config['obstacles'])
        p = all_results[f"{env_name}_ptrm_K50_cbf"]
        m = all_results[f"{env_name}_mppi_K50_cbf"]
        c = all_results[f"{env_name}_cem_K50_cbf"]
        print(f"| {env_name} | {n_obs} | {p['success_rate']:.0f} | {p['terminal_error_mean']:.4f} | {p['iae_mean']:.1f} | "
              f"{m['success_rate']:.0f} | {m['terminal_error_mean']:.4f} | {c['success_rate']:.0f} | {c['terminal_error_mean']:.4f} |")


if __name__ == '__main__':
    main()
