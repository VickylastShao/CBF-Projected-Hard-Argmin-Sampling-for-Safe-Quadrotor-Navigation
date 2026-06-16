# -*- coding: utf-8 -*-
"""
TRM-PD混合架构实验（v2：与v6实验参数对齐）

与v6_quick_test.py对齐的关键参数：
- 随机初始条件 (random_x_init)
- 单一设定点 [2.0, 3.0, 2.0]
- 300步仿真 (6s)
- 成功阈值: TErr < 0.5m
- N_MC = 20
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from quadrotor_core.dynamics import QuadrotorDynamics
from quadrotor_core.trm_network import TRMNMPC
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor

# ============================================================
# 与v6对齐的配置
# ============================================================
CL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v6', 'cl_trm_model.pt')

N_MC = 20
N_STEPS = 300       # 6s仿真
DT = 0.02
TErr_THRESH = 0.5   # v6成功阈值

SEED = 2026

# 单一设定点（与v6一致）
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)

# 障碍物环境
OBSTACLES = [
    {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
    {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
    {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init():
    """与v6一致的随机初始条件"""
    return torch.tensor([
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(-1.0, 0.0),
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


def load_cl_trm_model(model_path, device='cpu'):
    """加载闭环训练的TRM模型"""
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC,
                   use_mismatch=False, process_noise=0.0):
    """与v6对齐的MC试验"""
    results = []

    for mc in range(n_mc):
        set_seed(SEED + mc)
        x = random_x_init()
        predictor.reset()

        collision = False
        min_dist = float('inf')

        for step in range(N_STEPS):
            u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=True)
            x = env.step_discrete(x, u_safe,
                                    use_mismatch=use_mismatch,
                                    process_noise=process_noise)

            # 检查碰撞
            pos = x[0:3].numpy()
            for obs in env.obstacles:
                d = np.linalg.norm(pos - obs['p']) - obs['r']
                min_dist = min(min_dist, d)
                if d < -0.05:
                    collision = True

        terr = torch.norm(x[0:3] - x_sp[0:3]).item()
        # IAE: 位置误差积分
        # 近似用最终状态计算（完整IAE需要记录每步，这里用最终误差和最终速度估计）
        # 简化: 只返回TErr和成功率

        results.append({
            'TErr': terr,
            'success': (not collision) and (terr < TErr_THRESH),
            'collision': collision,
            'min_dist': min_dist
        })

    t_errs = [r['TErr'] for r in results]
    n_success = sum(1 for r in results if r['success'])
    n_collision = sum(1 for r in results if r['collision'])

    return {
        'TErr_mean': np.mean(t_errs),
        'TErr_std': np.std(t_errs),
        'TErr_median': np.median(t_errs),
        'success_rate': n_success / n_mc * 100,
        'collision_rate': n_collision / n_mc * 100,
    }


def main():
    print("=" * 80)
    print("TRM-PD混合架构实验 (v2: v6参数对齐)")
    print(f"设定点: {X_SP.numpy()[:3]}")
    print(f"N_MC={N_MC}, N_STEPS={N_STEPS}, TErr阈值={TErr_THRESH}")
    print("=" * 80)

    device = 'cpu'
    cl_model = load_cl_trm_model(CL_MODEL_PATH, device)
    n_params = sum(p.numel() for p in cl_model.parameters() if p.requires_grad)
    print(f"CL-TRM模型: {n_params} 参数")

    env = QuadrotorDynamics(obstacles=OBSTACLES)

    # ============================================================
    # 实验1: alpha_blend消融 (K=50, 无失配)
    # ============================================================
    print("\n" + "=" * 80)
    print("实验1: alpha_blend消融 (K=50, 无失配, v6参数)")
    print("=" * 80)

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    alpha_results = {}

    for alpha in alphas:
        label = f"α={alpha:.1f}"
        if alpha == 1.0:
            label += "(纯PD基线)"
        elif alpha == 0.0:
            label += "(纯TRM基线)"

        predictor = PTRMNMPCPredictor(
            model=cl_model, env=env, K=50, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='trm_pd',
            alpha_blend=alpha,
            use_rollout_cost=True, rollout_top_m=10,
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_trials(env, predictor, X_SP, N_MC)
        elapsed = time.time() - t0

        alpha_results[alpha] = result
        print(f"  {label:25s} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| median={result['TErr_median']:.4f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 碰撞={result['collision_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 实验2: K消融 (Test-time compute scaling)
    # ============================================================
    # 找最优alpha（排除0.0和1.0的极端值）
    intermediate_alphas = {a: r for a, r in alpha_results.items() if 0.2 <= a <= 0.8}
    best_alpha = min(intermediate_alphas, key=lambda a: intermediate_alphas[a]['TErr_mean'])
    # 也测试α=1.0作为参照
    print(f"\n  最优中间α = {best_alpha:.1f}")

    print("\n" + "=" * 80)
    print(f"实验2: K消融 (CL-TRM+PD α={best_alpha:.1f}, 无失配)")
    print("=" * 80)

    k_values = [1, 5, 10, 20, 50]
    k_results_alpha = {}

    for K in k_values:
        predictor = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='trm_pd',
            alpha_blend=best_alpha,
            use_rollout_cost=True, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_trials(env, predictor, X_SP, N_MC)
        elapsed = time.time() - t0

        k_results_alpha[K] = result
        print(f"  K={K:3d} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # α=1.0 (纯PD基线)的K消融
    print(f"\n  --- α=1.0 (纯PD基线) K消融 ---")
    k_results_pd = {}
    for K in k_values:
        predictor = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='trm_pd',
            alpha_blend=1.0,
            use_rollout_cost=True, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_trials(env, predictor, X_SP, N_MC)
        elapsed = time.time() - t0

        k_results_pd[K] = result
        print(f"  K={K:3d} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 实验3: 模型失配鲁棒性
    # ============================================================
    print("\n" + "=" * 80)
    print("实验3: 模型失配鲁棒性 (50% mass/drag, noise=0.01)")
    print("=" * 80)

    mismatch_configs = [
        (f"CL-TRM+PD(α={best_alpha:.1f}) K=50", 'trm_pd', 50, best_alpha),
        (f"CL-TRM+PD(α={best_alpha:.1f}) K=1", 'trm_pd', 1, best_alpha),
        ("CL-TRM+PD(α=1.0) K=50", 'trm_pd', 50, 1.0),
        ("CL-TRM+PD(α=1.0) K=1", 'trm_pd', 1, 1.0),
        ("CL-TRM-Rollout K=50", 'trm_rollout', 50, 0.0),
        ("CL-TRM-Rollout K=1", 'trm_rollout', 1, 0.0),
    ]

    mismatch_results = {}
    for label, mode, K, alpha in mismatch_configs:
        predictor = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode=mode,
            alpha_blend=alpha,
            use_rollout_cost=True, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_trials(env, predictor, X_SP, N_MC,
                                use_mismatch=True, process_noise=0.01)
        elapsed = time.time() - t0

        mismatch_results[label] = result
        print(f"  {label:30s} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 碰撞={result['collision_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 实验4: 评估策略消融 (Q-head vs 纯Rollout)
    # ============================================================
    print("\n" + "=" * 80)
    print("实验4: 评估策略消融 (K=50, α=0.5, 无失配)")
    print("=" * 80)

    # PD模式 + Q-head+Rollout评估
    predictor_qr = PTRMNMPCPredictor(
        model=cl_model, env=env, K=50, D=16,
        sigma=0.25, pd_sigma=2.0,
        candidate_mode='pd',
        alpha_blend=0.3,
        use_rollout_cost=True, rollout_top_m=10,
        rollout_steps=20, obs_weight=2000.0
    )
    r_qr = run_mc_trials(env, predictor_qr, X_SP, N_MC)
    print(f"  PD+Q+Rollout          | TErr={r_qr['TErr_mean']:.4f}±{r_qr['TErr_std']:.4f} "
          f"| 成功={r_qr['success_rate']:.0f}%")

    # PD模式 + 纯Rollout评估 (ranking_mode=rollout_all)
    predictor_ro = PTRMNMPCPredictor(
        model=cl_model, env=env, K=50, D=16,
        sigma=0.25, pd_sigma=2.0,
        candidate_mode='pd',
        alpha_blend=0.3,
        use_rollout_cost=True, rollout_top_m=10,
        rollout_steps=20, obs_weight=2000.0,
        ranking_mode='rollout_all'
    )
    r_ro = run_mc_trials(env, predictor_ro, X_SP, N_MC)
    print(f"  PD+纯Rollout          | TErr={r_ro['TErr_mean']:.4f}±{r_ro['TErr_std']:.4f} "
          f"| 成功={r_ro['success_rate']:.0f}%")

    # TRM-PD混合 + 纯Rollout评估
    predictor_hybrid = PTRMNMPCPredictor(
        model=cl_model, env=env, K=50, D=16,
        sigma=0.25, pd_sigma=2.0,
        candidate_mode='trm_pd',
        alpha_blend=0.5,
        use_rollout_cost=True, rollout_top_m=10,
        rollout_steps=20, obs_weight=2000.0
    )
    r_hybrid = run_mc_trials(env, predictor_hybrid, X_SP, N_MC)
    print(f"  TRM+PD(α=0.5)+Rollout | TErr={r_hybrid['TErr_mean']:.4f}±{r_hybrid['TErr_std']:.4f} "
          f"| 成功={r_hybrid['success_rate']:.0f}%")

    eval_results = {
        'PD+Q+Rollout': r_qr,
        'PD+纯Rollout': r_ro,
        'TRM+PD(α=0.5)+Rollout': r_hybrid
    }

    # ============================================================
    # 汇总表格
    # ============================================================
    print("\n" + "=" * 80)
    print("实验汇总")
    print("=" * 80)

    print("\n--- 实验1: alpha_blend消融 (K=50, 无失配) ---")
    print(f"{'α':>8s} | {'TErr_mean':>10s} | {'TErr_std':>10s} | {'TErr_median':>12s} | {'成功率':>6s}")
    print("-" * 60)
    for alpha in alphas:
        r = alpha_results[alpha]
        tag = ""
        if alpha == 0.0:
            tag = " (TRM)"
        elif alpha == 1.0:
            tag = " (PD)"
        print(f"  {alpha:.1f}{tag:>6s} | {r['TErr_mean']:10.4f} | {r['TErr_std']:10.4f} | "
              f"{r['TErr_median']:12.4f} | {r['success_rate']:5.0f}%")

    print(f"\n--- 实验2a: K消融 (α={best_alpha:.1f}, 混合基线) ---")
    print(f"{'K':>5s} | {'TErr_mean':>10s} | {'TErr_std':>10s} | {'成功率':>6s} | vs K=1")
    print("-" * 55)
    base = k_results_alpha[1]['TErr_mean']
    for K in k_values:
        r = k_results_alpha[K]
        ratio = base / max(r['TErr_mean'], 1e-6)
        print(f"{K:>5d} | {r['TErr_mean']:10.4f} | {r['TErr_std']:10.4f} | "
              f"{r['success_rate']:5.0f}% | {ratio:.1f}×")

    print(f"\n--- 实验2b: K消融 (α=1.0, 纯PD基线) ---")
    print(f"{'K':>5s} | {'TErr_mean':>10s} | {'TErr_std':>10s} | {'成功率':>6s} | vs K=1")
    print("-" * 55)
    base_pd = k_results_pd[1]['TErr_mean']
    for K in k_values:
        r = k_results_pd[K]
        ratio = base_pd / max(r['TErr_mean'], 1e-6)
        print(f"{K:>5d} | {r['TErr_mean']:10.4f} | {r['TErr_std']:10.4f} | "
              f"{r['success_rate']:5.0f}% | {ratio:.1f}×")

    print(f"\n--- 实验3: 模型失配鲁棒性 ---")
    print(f"{'方法':>30s} | {'TErr_mean':>10s} | {'TErr_std':>10s} | {'成功率':>6s} | {'碰撞率':>6s}")
    print("-" * 75)
    for label, r in mismatch_results.items():
        print(f"{label:>30s} | {r['TErr_mean']:10.4f} | {r['TErr_std']:10.4f} | "
              f"{r['success_rate']:5.0f}% | {r['collision_rate']:5.0f}%")

    print(f"\n--- 实验4: 评估策略消融 ---")
    print(f"{'策略':>25s} | {'TErr_mean':>10s} | {'TErr_std':>10s} | {'成功率':>6s}")
    print("-" * 60)
    for label, r in eval_results.items():
        print(f"{label:>25s} | {r['TErr_mean']:10.4f} | {r['TErr_std']:10.4f} | {r['success_rate']:5.0f}%")

    # 保存结果
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_hybrid')
    os.makedirs(save_dir, exist_ok=True)

    all_results = {
        'alpha_ablation': {str(k): v for k, v in alpha_results.items()},
        'k_scaling_hybrid': {str(k): v for k, v in k_results_alpha.items()},
        'k_scaling_pd': {str(k): v for k, v in k_results_pd.items()},
        'mismatch': mismatch_results,
        'eval_ablation': eval_results,
        'best_alpha': best_alpha,
    }
    torch.save(all_results, os.path.join(save_dir, 'hybrid_v2_results.pt'))
    print(f"\n结果已保存到: {save_dir}/hybrid_v2_results.pt")


if __name__ == '__main__':
    main()
