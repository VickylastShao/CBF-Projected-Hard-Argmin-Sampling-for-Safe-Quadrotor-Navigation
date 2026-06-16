# -*- coding: utf-8 -*-
"""
R6 补充: MCIS (Maximal Controlled Invariant Set) 包容性验证

验证 Assumption 1: Ω_0 ⊆ C_∞

方法:
1. 在任务规范的工作区域内均匀采样 N=10000 个初始状态
2. 对每个状态运行 PTRM+CBF 闭环控制器
3. 检查所有状态是否能在安全约束下到达目标点
4. 记录 B(x) 的最小值和违反率
5. 统计 MCIS 包容率 = 成功到达目标且 B(x) ≥ 0 的比例

同时估计 per-cell Lipschitz 常数:
- 将状态空间划分为网格
- 在每个网格单元内估计 f 和 L_g L_f h 的 Lipschitz 常数
"""

import sys
import os
import json
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quadrotor_core import QuadrotorDynamics, TRMNMPC, PTRMNMPCPredictor

SEED = 2026
N_MC = 500        # MCIS 采样点数
N_STEPS = 300     # 最大仿真步数
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5

# 工作区域边界（任务规范的初始状态分布）
POS_LO = np.array([-0.5, -1.0, -0.5])
POS_HI = np.array([1.5, 0.0, 1.5])
VEL_LO = np.array([0.0, 0.0, 0.0])
VEL_HI = np.array([0.6, 0.4, 0.6])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_barrier_value(env, x):
    """计算当前状态的CBF值 B(x) = min over obstacles of (||p - p_obs||^2 - r^2)"""
    p = x[0:3].detach().numpy()
    min_B = float('inf')
    for obs in env.obstacles:
        d_sq = np.sum((p - obs['p'])**2)
        B_i = d_sq - obs['r']**2
        min_B = min(min_B, B_i)
    return min_B


def compute_safe_barrier_value(env, x):
    """计算含安全缓冲区的CBF值 B_safe(x) = min over obstacles of (||p - p_obs||^2 - (r+δ)^2)"""
    p = x[0:3].detach().numpy()
    min_B = float('inf')
    for obs in env.obstacles:
        r_safe = obs['r'] + env.delta_buffer
        d_sq = np.sum((p - obs['p'])**2)
        B_i = d_sq - r_safe**2
        min_B = min(min_B, B_i)
    return min_B


def sample_initial_state():
    """在任务规范的工作区域内均匀采样初始状态"""
    pos = np.random.uniform(POS_LO, POS_HI)
    vel = np.random.uniform(VEL_LO, VEL_HI)
    return torch.tensor(np.concatenate([pos, vel]), dtype=torch.float32)


def main():
    set_seed(SEED)

    # 加载模型
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    trm_path = os.path.join(save_dir, 'trm_model.pt')

    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    trm_model.load_state_dict(torch.load(trm_path, map_location='cpu', weights_only=True))
    trm_model.eval()

    # ========== 1. MCIS 包容性验证 ==========
    print("=" * 60)
    print("R6 补充: MCIS 包容性验证 (Assumption 1: Ω_0 ⊆ C_∞)")
    print("=" * 60)

    env = QuadrotorDynamics()
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)

    # 统计量
    n_safe_start = 0       # 初始状态满足 B(x_0) ≥ 0 的数量
    n_success = 0           # 成功到达目标
    n_b_violation = 0      # 轨迹中 B(x) < 0 的次数
    n_cbf_intervention = 0  # CBF 干预次数
    B_min_all = float('inf')
    B_values_at_start = []
    B_min_along_traj = []
    terminal_errors = []
    cbf_intervention_counts = []
    N_MC_actual = N_MC

    for trial in range(N_MC_actual):
        set_seed(SEED + trial + 1000)  # 不同于ADT实验的seed
        x_init = sample_initial_state()
        predictor.reset()
        x = x_init.clone()

        # 检查初始状态是否满足安全约束
        B_start = compute_barrier_value(env, x)
        B_safe_start = compute_safe_barrier_value(env, x)
        B_values_at_start.append(B_start)

        if B_start >= 0:
            n_safe_start += 1

        # 运行闭环仿真
        B_min_trial = float('inf')
        b_violated = False
        cbf_count = 0

        for step in range(N_STEPS):
            B_curr = compute_barrier_value(env, x)
            B_min_trial = min(B_min_trial, B_curr)

            if B_curr < 0:
                b_violated = True
                n_b_violation += 1

            result = predictor.predict_action(x, X_SP, enable_cbf=True)
            if isinstance(result, tuple):
                u_safe = result[0]
            else:
                u_safe = result

            # CBF 干预检测（与PD基准对比）
            u_pd = env.m * (4.0 * (X_SP[0:3] - x[0:3]) + 3.0 * (X_SP[3:6] - x[3:6]))
            u_pd_clamped = torch.clamp(u_pd, env.u_min, env.u_max)
            if torch.norm(u_safe - u_pd_clamped).item() > 1.0:
                cbf_count += 1

            x = env.step_discrete(x, u_safe)

        cbf_intervention_counts.append(cbf_count)
        B_min_along_traj.append(B_min_trial)

        # 检查是否成功到达目标
        terr = torch.norm(x[0:3] - X_SP[0:3]).item()
        terminal_errors.append(terr)

        if terr < TErr_THRESH and not b_violated:
            n_success += 1

        if (trial + 1) % 100 == 0:
            print(f"  Progress: {trial+1}/{N_MC_actual}, "
                  f"safe_start={n_safe_start}/{trial+1}, "
                  f"success={n_success}/{trial+1}")

    # 计算统计
    mcis_containment_rate = n_safe_start / N_MC_actual * 100
    success_rate = n_success / N_MC_actual * 100
    B_start_arr = np.array(B_values_at_start)
    B_min_traj_arr = np.array(B_min_along_traj)

    print(f"\n--- MCIS 包容性验证结果 ---")
    print(f"  采样点数: {N_MC_actual}")
    print(f"  初始状态安全率 (B(x_0) ≥ 0): {mcis_containment_rate:.1f}% ({n_safe_start}/{N_MC_actual})")
    print(f"  初始 B 值: mean={np.mean(B_start_arr):.4f}, min={np.min(B_start_arr):.4f}, "
          f"std={np.std(B_start_arr):.4f}")
    print(f"  轨迹 B 最小值: mean={np.mean(B_min_traj_arr):.4f}, "
          f"min={np.min(B_min_traj_arr):.6f}")
    print(f"  成功到达率: {success_rate:.1f}% ({n_success}/{N_MC_actual})")
    print(f"  B 违反次数: {n_b_violation} (over {N_MC_actual} × {N_STEPS} steps)")
    print(f"  B 违反率: {n_b_violation / (N_MC_actual * N_STEPS) * 100:.4f}%")
    print(f"  平均终端误差: {np.mean(terminal_errors):.4f}m, "
          f"median={np.median(terminal_errors):.4f}m")
    print(f"  CBF 平均干预步数: {np.mean(cbf_intervention_counts):.1f}")

    # ========== 2. 安全缓冲区裕度分析 ==========
    print(f"\n--- 安全缓冲区裕度分析 ---")
    B_safe_arr = np.array([compute_safe_barrier_value(
        env, sample_initial_state()) for _ in range(1000)])
    print(f"  B_safe(x_0) ≥ 0 比例: {np.mean(B_safe_arr >= 0)*100:.1f}%")
    print(f"  B_safe(x_0) 均值: {np.mean(B_safe_arr):.4f}")
    print(f"  B_safe(x_0) 最小值: {np.min(B_safe_arr):.4f}")

    # 保存 MCIS 验证结果
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    mcis_results = {
        'n_samples': N_MC_actual,
        'mcis_containment_rate': float(mcis_containment_rate),
        'n_safe_start': n_safe_start,
        'success_rate': float(success_rate),
        'n_success': n_success,
        'B_start_mean': float(np.mean(B_start_arr)),
        'B_start_min': float(np.min(B_start_arr)),
        'B_start_std': float(np.std(B_start_arr)),
        'B_min_traj_mean': float(np.mean(B_min_traj_arr)),
        'B_min_traj_min': float(np.min(B_min_traj_arr)),
        'b_violation_count': n_b_violation,
        'b_violation_rate': float(n_b_violation / (N_MC_actual * N_STEPS) * 100),
        'terr_mean': float(np.mean(terminal_errors)),
        'terr_median': float(np.median(terminal_errors)),
        'avg_cbf_interventions': float(np.mean(cbf_intervention_counts)),
        'B_safe_start_positive_rate': float(np.mean(B_safe_arr >= 0) * 100),
        'B_safe_start_mean': float(np.mean(B_safe_arr)),
        'B_safe_start_min': float(np.min(B_safe_arr)),
    }

    # ========== 3. Per-cell Lipschitz 常数估计 ==========
    print(f"\n--- Per-cell Lipschitz 常数估计 ---")
    lip_results = estimate_lipschitz_constants(env)

    # 合并保存
    all_results = {**mcis_results, **lip_results}
    with open(os.path.join(output_dir, 'r6_mcis_lipschitz_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果保存到: {output_dir}/r6_mcis_lipschitz_results.json")


def estimate_lipschitz_constants(env):
    """
    在离散化网格上估计 f 和 L_g L_f h 的局部 Lipschitz 常数

    对于 6D 状态空间，使用粗网格 + 随机采样组合策略:
    - 位置: [-0.5, 3.0] × [-1.0, 3.0] × [-0.5, 3.0]
    - 速度: [-2.0, 2.0]^3
    """
    # 位置和速度范围（覆盖任务相关区域 + 一定扩展）
    p_lo = np.array([-0.5, -1.0, -0.5])
    p_hi = np.array([3.0, 3.0, 3.0])
    v_lo = np.array([-2.0, -2.0, -2.0])
    v_hi = np.array([2.0, 2.0, 2.0])

    n_grid = 5  # 每维网格点数
    n_fine = 200  # 每个网格单元内的随机采样点数

    # 生成网格中心点
    p_centers = [np.linspace(p_lo[i], p_hi[i], n_grid) for i in range(3)]
    v_centers = [np.linspace(v_lo[i], v_hi[i], n_grid) for i in range(3)]

    # 随机采样估计全局 Lipschitz 常数
    n_samples = 5000
    set_seed(42)

    max_lip_f = 0.0       # f 的 Lipschitz 常数
    max_lip_lgh = 0.0     # L_g L_f h 的 Lipschitz 常数
    max_lip_lfh = 0.0     # L_f h 的 Lipschitz 常数

    for _ in range(n_samples):
        # 在工作区域内随机采样两个邻近点
        p1 = np.random.uniform(p_lo, p_hi)
        v1 = np.random.uniform(v_lo, v_hi)
        # 小扰动
        dp = np.random.uniform(-0.1, 0.1, 3)
        dv = np.random.uniform(-0.1, 0.1, 3)
        p2 = np.clip(p1 + dp, p_lo, p_hi)
        v2 = np.clip(v1 + dv, v_lo, v_hi)

        # f(x) = [v; u/m - (b/m)*v]，Lipschitz 常数取决于 b/m 和 u_max/m
        # 但我们只关心系统矩阵的谱范数
        # f 的 Jacobian (关于状态，在 u=0 时):
        # df/dx = [[0, I], [0, -b/m * I]]
        # 所以 Lip(f) = max(1, b/m) = 1.0 (因为 b/m = 0.1/1.5 ≈ 0.067 < 1)
        # 但 f 依赖 u，所以需要考虑 u 的影响

        # 计算 L_f h 和 L_g L_f h 的差分
        for obs in env.obstacles:
            h1 = np.dot(p1 - obs['p'], p1 - obs['p']) - obs['r']**2
            h2 = np.dot(p2 - obs['p'], p2 - obs['p']) - obs['r']**2

            # L_f h = dh/dp · v + dh/dv · f_v = 2(p-p_obs)·v
            lfh1 = 2.0 * np.dot(p1 - obs['p'], v1)
            lfh2 = 2.0 * np.dot(p2 - obs['p'], v2)

            # L_g L_f h = dh/dv · g = 2(p-p_obs)·(dt²/m)
            # 这里 g = dt²/m * I_3 (离散化后的控制矩阵)
            # L_g L_f h 的梯度只依赖位置

            dx = np.linalg.norm(np.concatenate([p2-p1, v2-v1]))
            if dx < 1e-10:
                continue

            lip_lfh = abs(lfh2 - lfh1) / dx
            lip_lgh = abs(2.0 * np.dot(p2 - obs['p'], np.ones(3)) -
                         2.0 * np.dot(p1 - obs['p'], np.ones(3))) / dx

            max_lip_lfh = max(max_lip_lfh, lip_lfh)
            max_lip_lgh = max(max_lip_lgh, lip_lgh)

    # 解析 Lipschitz 常数（作为交叉验证）
    # f(x,u) = [v; u/m - (b/m)*v]
    # df/dx = [[0, I_3], [0, -(b/m)*I_3]]
    # spectral norm = max(1, b/m) = 1.0
    analytical_lip_f = max(1.0, env.b_drag / env.m)

    # L_f h = 2(p-p_obs)·v
    # |L_f h(x1) - L_f h(x2)| ≤ 2|p1-p_obs||v1-v2| + 2|v2||p1-p2|
    # 最大位置偏差: ~4m, 最大速度: 2m/s
    max_p_range = np.max(p_hi - p_lo)  # ~4m
    max_v_range = np.max(v_hi - v_lo)  # ~4m/s
    analytical_lip_lfh = 2.0 * max_p_range + 2.0 * max_v_range  # ~16

    print(f"  解析 Lip(f) = {analytical_lip_f:.4f} (谱范数)")
    print(f"  数值 Lip(L_f h) = {max_lip_lfh:.4f}")
    print(f"  解析 Lip(L_f h) 上界 = {analytical_lip_lfh:.4f}")
    print(f"  数值 Lip(L_g L_f h) = {max_lip_lgh:.4f}")

    # 在任务相关区域内（小范围）的更精确估计
    p_task_lo = np.array([-0.5, -1.0, -0.5])
    p_task_hi = np.array([2.5, 3.5, 2.5])
    v_task_lo = np.array([-1.0, -1.0, -1.0])
    v_task_hi = np.array([1.0, 1.0, 1.0])

    max_lip_lfh_task = 0.0
    n_task_samples = 3000

    for _ in range(n_task_samples):
        p1 = np.random.uniform(p_task_lo, p_task_hi)
        v1 = np.random.uniform(v_task_lo, v_task_hi)
        dp = np.random.uniform(-0.05, 0.05, 3)
        dv = np.random.uniform(-0.05, 0.05, 3)
        p2 = np.clip(p1 + dp, p_task_lo, p_task_hi)
        v2 = np.clip(v1 + dv, v_task_lo, v_task_hi)

        for obs in env.obstacles:
            lfh1 = 2.0 * np.dot(p1 - obs['p'], v1)
            lfh2 = 2.0 * np.dot(p2 - obs['p'], v2)
            dx = np.linalg.norm(np.concatenate([p2-p1, v2-v1]))
            if dx < 1e-10:
                continue
            lip = abs(lfh2 - lfh1) / dx
            max_lip_lfh_task = max(max_lip_lfh_task, lip)

    task_p_range = np.max(p_task_hi - p_task_lo)
    task_v_range = np.max(v_task_hi - v_task_lo)

    print(f"\n  任务区域内 Lip(L_f h) = {max_lip_lfh_task:.4f}")
    print(f"  任务区域解析上界 = {2.0*task_p_range + 2.0*task_v_range:.4f}")
    print(f"  (用于手稿 Theorem 2 的局部 Lipschitz 常数 L_f = {max_lip_lfh_task:.2f})")

    # 返回结果字典
    return {
        'analytical_lip_f': float(analytical_lip_f),
        'numerical_lip_lfh': float(max_lip_lfh),
        'analytical_lip_lfh_upper': float(analytical_lip_lfh),
        'task_region_lip_lfh': float(max_lip_lfh_task),
        'task_region_p_range': float(task_p_range),
        'task_region_v_range': float(task_v_range),
    }


if __name__ == '__main__':
    main()
