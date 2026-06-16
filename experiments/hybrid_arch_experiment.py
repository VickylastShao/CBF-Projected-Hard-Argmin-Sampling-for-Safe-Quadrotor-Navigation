# -*- coding: utf-8 -*-
"""
TRM-PD混合架构实验：闭环训练TRM + PD混合基线 + Rollout评估

实验目标：
1. 验证TRM-PD混合基线比纯PD基线更具信息量
2. 验证test-time compute scaling (K候选)的收益
3. 确定最优alpha_blend配置
4. 与纯PD、纯TRM、MPPI、CEM基线对比

架构设计：
- 闭环训练TRM提供状态依赖策略先验（学习到的障碍物回避、制动力等）
- PD反馈提供实时稳定性（无模型、无训练依赖、保证收敛方向）
- 混合基线: u_base = (1-α)*TRM + α*PD
- K候选 + Rollout评估提供test-time compute scaling
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
# 配置
# ============================================================
CL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v6', 'cl_trm_model.pt')
ORIG_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'experiments', 'results_v6', 'trm_model.pt')

N_MC = 20              # Monte Carlo 试验次数
SEED_BASE = 42

# 设定点（与v6实验一致）
SETPOINTS = [
    torch.tensor([2.5, 2.5, 2.5, 0.0, 0.0, 0.0]),
    torch.tensor([3.0, 1.0, 2.5, 0.0, 0.0, 0.0]),
    torch.tensor([1.0, 3.0, 2.0, 0.0, 0.0, 0.0]),
]

# 障碍物环境（与v6一致）
OBSTACLES = [
    {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
    {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
    {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
]

MAX_STEPS = 200  # 每个设定点最大步数
DT = 0.02


def load_cl_trm_model(model_path, device='cpu'):
    """加载闭环训练的TRM模型"""
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    state_dict = torch.load(model_path, map_location=device)
    # 处理可能的键名差异
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  加载CL-TRM模型: {n_params} 参数")
    return model


def load_orig_trm_model(model_path, device='cpu'):
    """加载原始开环训练的TRM模型"""
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  加载原始TRM模型: {n_params} 参数")
    return model


def run_single_trial(env, predictor, x0, x_sp, use_mismatch=False, process_noise=0.0):
    """
    单次闭环仿真试验

    Returns:
        dict: 包含TErr, IAE, success, collision等指标
    """
    x = x0.clone()
    predictor.reset()

    total_error = 0.0
    min_dist_to_obs = float('inf')
    collision = False
    reached = False
    actual_steps = 0

    for step in range(MAX_STEPS):
        u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=True)

        # 仿真步进
        x_next = env.step_discrete(x, u_safe,
                                    use_mismatch=use_mismatch,
                                    process_noise=process_noise)

        # 检查障碍物距离
        pos = x_next[0:3].numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(pos - obs['p']) - obs['r']
            min_dist_to_obs = min(min_dist_to_obs, d)
            if d < -0.05:  # 允许小数值误差
                collision = True

        # 累积跟踪误差: IAE = (1/T) * Σ ||e|| * dt
        err = (x_sp[0:3] - x_next[0:3])
        total_error += torch.norm(err).item() * DT
        actual_steps += 1

        x = x_next

        # 检查是否到达目标
        if torch.norm(x_sp[0:3] - x[0:3]) < 0.05:
            reached = True
            # 继续仿真计算剩余IAE
            for _ in range(min(50, MAX_STEPS - step - 1)):
                u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=True)
                x = env.step_discrete(x, u_safe,
                                       use_mismatch=use_mismatch,
                                       process_noise=process_noise)
                err = (x_sp[0:3] - x[0:3])
                total_error += torch.norm(err).item() * DT
                actual_steps += 1
            break

    # IAE归一化: 除以总仿真时间 T
    T_actual = actual_steps * DT
    iae_normalized = total_error / T_actual if T_actual > 0 else 0.0

    # 最终跟踪误差
    t_err = torch.norm(x_sp[0:3] - x[0:3]).item()
    success = not collision and t_err < 0.1

    return {
        'TErr': t_err,
        'IAE': iae_normalized,
        'success': success,
        'collision': collision,
        'reached': reached,
        'min_dist': min_dist_to_obs,
        'final_state': x.numpy().copy()
    }


def run_mc_experiment(env, predictor, setpoints, n_mc=20,
                       use_mismatch=False, process_noise=0.0):
    """运行Monte Carlo实验"""
    results = {sp_idx: [] for sp_idx in range(len(setpoints))}

    for mc in range(n_mc):
        torch.manual_seed(SEED_BASE + mc)
        np.random.seed(SEED_BASE + mc)

        x0 = torch.zeros(6)  # 初始状态: 原点静止

        for sp_idx, x_sp in enumerate(setpoints):
            result = run_single_trial(env, predictor, x0, x_sp,
                                       use_mismatch=use_mismatch,
                                       process_noise=process_noise)
            results[sp_idx].append(result)

    # 汇总统计
    all_t_err = []
    all_iae = []
    n_success = 0
    n_collision = 0

    for sp_idx in range(len(setpoints)):
        for r in results[sp_idx]:
            all_t_err.append(r['TErr'])
            all_iae.append(r['IAE'])
            if r['success']:
                n_success += 1
            if r['collision']:
                n_collision += 1

    total_trials = n_mc * len(setpoints)
    return {
        'TErr_mean': np.mean(all_t_err),
        'TErr_std': np.std(all_t_err),
        'IAE_mean': np.mean(all_iae),
        'IAE_std': np.std(all_iae),
        'success_rate': n_success / total_trials * 100,
        'collision_rate': n_collision / total_trials * 100,
        'n_trials': total_trials
    }


def main():
    print("=" * 80)
    print("TRM-PD混合架构实验")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    # 加载模型
    print("\n--- 加载模型 ---")
    cl_model = load_cl_trm_model(CL_MODEL_PATH, device)

    # 检查原始模型是否存在
    has_orig = os.path.exists(ORIG_MODEL_PATH)
    if has_orig:
        orig_model = load_orig_trm_model(ORIG_MODEL_PATH, device)

    # 创建动力学环境
    env = QuadrotorDynamics(obstacles=OBSTACLES)

    # ============================================================
    # 实验1: alpha_blend消融（TRM-PD混合比例）
    # ============================================================
    print("\n" + "=" * 80)
    print("实验1: alpha_blend消融 (CL-TRM + PD混合比例)")
    print("K=50, N_MC=20, 无失配")
    print("=" * 80)

    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    alpha_results = {}

    for alpha in alphas:
        mode_label = f"α={alpha:.1f}"
        if alpha == 1.0:
            mode_label += " (纯PD)"
        elif alpha == 0.0:
            mode_label += " (纯TRM)"

        predictor = PTRMNMPCPredictor(
            model=cl_model, env=env, K=50, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='trm_pd',
            alpha_blend=alpha,
            use_rollout_cost=True, rollout_top_m=10,
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_experiment(env, predictor, SETPOINTS, N_MC,
                                    use_mismatch=False, process_noise=0.0)
        elapsed = time.time() - t0

        alpha_results[alpha] = result
        print(f"  {mode_label:25s} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| IAE={result['IAE_mean']:.2f}±{result['IAE_std']:.2f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 碰撞={result['collision_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 实验2: Test-time compute scaling (K消融)
    # ============================================================
    print("\n" + "=" * 80)
    print("实验2: Test-time compute scaling (K消融)")
    print("CL-TRM+PD (最优α), N_MC=20, 无失配")
    print("=" * 80)

    # 从实验1中找到最优alpha
    best_alpha = min(alpha_results, key=lambda a: alpha_results[a]['TErr_mean'])
    print(f"  最优alpha_blend = {best_alpha:.1f}")

    k_values = [1, 5, 10, 20, 50]
    k_results = {}

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
        result = run_mc_experiment(env, predictor, SETPOINTS, N_MC,
                                    use_mismatch=False, process_noise=0.0)
        elapsed = time.time() - t0

        k_results[K] = result
        print(f"  K={K:3d} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| IAE={result['IAE_mean']:.2f}±{result['IAE_std']:.2f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 实验3: 模型失配鲁棒性
    # ============================================================
    print("\n" + "=" * 80)
    print("实验3: 模型失配鲁棒性 (50% mass/drag, noise=0.01)")
    print("N_MC=20")
    print("=" * 80)

    mismatch_configs = [
        ("CL-TRM+PD K=50", 'trm_pd', cl_model, 50, best_alpha),
        ("CL-TRM+PD K=1",  'trm_pd', cl_model, 1,  best_alpha),
        ("CL-TRM-Rollout K=50", 'trm_rollout', cl_model, 50, 0.0),
        ("CL-TRM-Rollout K=1",  'trm_rollout', cl_model, 1,  0.0),
        ("PD+CBF K=1",  'pd', cl_model, 1,  0.3),
        ("PD+CBF K=50", 'pd', cl_model, 50, 0.3),
    ]

    mismatch_results = {}
    for label, mode, model, K, alpha in mismatch_configs:
        predictor = PTRMNMPCPredictor(
            model=model, env=env, K=K, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode=mode,
            alpha_blend=alpha,
            use_rollout_cost=True, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_experiment(env, predictor, SETPOINTS, N_MC,
                                    use_mismatch=True, process_noise=0.01)
        elapsed = time.time() - t0

        mismatch_results[label] = result
        print(f"  {label:25s} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| IAE={result['IAE_mean']:.2f}±{result['IAE_std']:.2f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 碰撞={result['collision_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 实验4: 候选模式消融（混合 vs 纯PD vs 纯TRM基线）
    # ============================================================
    print("\n" + "=" * 80)
    print("实验4: 候选模式消融 (K=50, 无失配)")
    print("=" * 80)

    mode_configs = [
        ("PD+Noise+Rollout", 'pd', cl_model, 0.3),
        ("CL-TRM-Rollout", 'trm_rollout', cl_model, 0.0),
        (f"CL-TRM+PD(α={best_alpha:.1f})", 'trm_pd', cl_model, best_alpha),
    ]

    mode_results = {}
    for label, mode, model, alpha in mode_configs:
        predictor = PTRMNMPCPredictor(
            model=model, env=env, K=50, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode=mode,
            alpha_blend=alpha,
            use_rollout_cost=True, rollout_top_m=10,
            rollout_steps=20, obs_weight=2000.0
        )

        t0 = time.time()
        result = run_mc_experiment(env, predictor, SETPOINTS, N_MC,
                                    use_mismatch=False, process_noise=0.0)
        elapsed = time.time() - t0

        mode_results[label] = result
        print(f"  {label:25s} | TErr={result['TErr_mean']:.4f}±{result['TErr_std']:.4f} "
              f"| IAE={result['IAE_mean']:.2f}±{result['IAE_std']:.2f} "
              f"| 成功={result['success_rate']:.0f}% "
              f"| 碰撞={result['collision_rate']:.0f}% "
              f"| 耗时={elapsed:.1f}s")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 80)
    print("实验汇总")
    print("=" * 80)

    print("\n--- 实验1: alpha_blend消融 (K=50, 无失配) ---")
    print(f"{'α':>5s} | {'TErr':>12s} | {'IAE':>12s} | {'成功率':>6s}")
    print("-" * 50)
    for alpha in alphas:
        r = alpha_results[alpha]
        label = f"{alpha:.1f}"
        if alpha == 1.0:
            label += "(PD)"
        elif alpha == 0.0:
            label += "(TRM)"
        print(f"{label:>5s} | {r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | "
              f"{r['IAE_mean']:.2f}±{r['IAE_std']:.2f} | {r['success_rate']:.0f}%")

    print(f"\n最优α = {best_alpha:.1f}")

    print("\n--- 实验2: K消融 (CL-TRM+PD, 最优α, 无失配) ---")
    print(f"{'K':>5s} | {'TErr':>12s} | {'IAE':>12s} | {'成功率':>6s} | K=1提升倍数")
    print("-" * 60)
    baseline_t_err = k_results[1]['TErr_mean']
    for K in k_values:
        r = k_results[K]
        improvement = baseline_t_err / max(r['TErr_mean'], 1e-6)
        print(f"{K:>5d} | {r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | "
              f"{r['IAE_mean']:.2f}±{r['IAE_std']:.2f} | {r['success_rate']:.0f}% | {improvement:.1f}×")

    print("\n--- 实验3: 模型失配鲁棒性 ---")
    print(f"{'方法':>25s} | {'TErr':>12s} | {'IAE':>12s} | {'成功率':>6s}")
    print("-" * 65)
    for label, r in mismatch_results.items():
        print(f"{label:>25s} | {r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | "
              f"{r['IAE_mean']:.2f}±{r['IAE_std']:.2f} | {r['success_rate']:.0f}%")

    print("\n--- 实验4: 候选模式消融 ---")
    print(f"{'模式':>25s} | {'TErr':>12s} | {'IAE':>12s} | {'成功率':>6s}")
    print("-" * 65)
    for label, r in mode_results.items():
        print(f"{label:>25s} | {r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | "
              f"{r['IAE_mean']:.2f}±{r['IAE_std']:.2f} | {r['success_rate']:.0f}%")

    # 保存结果
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_hybrid')
    os.makedirs(save_dir, exist_ok=True)

    # 转换为可序列化格式
    all_results = {
        'alpha_ablation': {str(k): v for k, v in alpha_results.items()},
        'k_scaling': {str(k): v for k, v in k_results.items()},
        'mismatch': mismatch_results,
        'mode_ablation': mode_results,
        'best_alpha': best_alpha,
    }
    torch.save(all_results, os.path.join(save_dir, 'hybrid_experiment_results.pt'))
    print(f"\n结果已保存到: {save_dir}/hybrid_experiment_results.pt")


if __name__ == '__main__':
    main()
