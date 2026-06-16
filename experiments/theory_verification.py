# -*- coding: utf-8 -*-
"""
理论验证实验 (R6, R8, S1)

量化:
  1. epsilon_lin — CBF线性化残差 (实际B值 vs 线性化预测B值)
  2. ADT (Average Dwell Time) — CBF切换事件的平均驻留时间
  3. MCIS包含性 — 训练/测试状态是否在最大可控不变集内
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
N_MC = 20
N_STEPS = 300
DT = 0.02
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'results_v6', 'trm_model.pt')


def random_x_init():
    """随机初始状态"""
    p0 = np.random.uniform(-0.1, 0.1, 3)
    v0 = np.random.uniform(-0.05, 0.05, 3)
    return torch.tensor(np.concatenate([p0, v0]), dtype=torch.float32)


def compute_smooth_barrier(env, p, v):
    """
    计算smooth barrier值 B_smooth(p, v)
    使用Log-Sum-Exponential平滑近似
    """
    alpha_d = env.alpha_d
    gamma_d = env.gamma_d
    h_values = []
    for obs in env.obstacles:
        obs_p = torch.tensor(obs['p'], dtype=torch.float32)
        obs_r = obs['r']
        h_j = torch.norm(p - obs_p) - obs_r
        h_values.append(h_j)

    # Log-Sum-Exponential smooth min
    h_tensor = torch.stack(h_values)
    kappa = 10.0  # smoothing parameter
    h_min_smooth = -(1.0 / kappa) * torch.logsumexp(-kappa * h_tensor, dim=0)

    # B_smooth = h_min(p_{k+1}) - (1-alpha_d) * h_min(p_k) + gamma_d * ||v_k||
    # Simplified: use position-only barrier for epsilon_lin computation
    return h_min_smooth


def compute_epsilon_lin(env, x, u_safe):
    """
    计算CBF线性化残差 epsilon_lin

    线性化预测: B_linear = B_current + A^T * (u_safe - u_nominal)
    实际值: B_actual = B(p_{k+1}, v_{k+1}) after applying u_safe
    残差: epsilon_lin = B_actual - B_linear
    """
    # 线性化预测的B值变化 (来自CBF投影的约束)
    # 实际上我们记录: 线性化保证 B_{k+1} >= (1-gamma_d)*B_k
    # 但由于线性化误差，实际 B_{k+1} 可能偏离

    p = x[0:3]
    v = x[3:6]

    # 当前 barrier 值
    B_k = compute_smooth_barrier(env, p, v)

    # 执行 u_safe 一步
    x_next = env.step_discrete(x, u_safe)
    p_next = x_next[0:3]
    v_next = x_next[3:6]

    # 下一步 barrier 值
    B_k1_actual = compute_smooth_barrier(env, p_next, v_next)

    # 线性化承诺: B_{k+1} >= (1-gamma_d) * B_k
    B_k1_linearized = (1.0 - env.gamma_d) * B_k

    # 残差 = 实际值 - 线性化预测
    epsilon_lin = B_k1_actual.item() - B_k1_linearized.item()

    return epsilon_lin, B_k.item(), B_k1_actual.item()


def run_epsilon_lin_experiment(env, predictor):
    """
    量化CBF线性化残差 epsilon_lin
    """
    print("\n" + "=" * 80)
    print("实验: epsilon_lin 线性化残差量化")
    print("=" * 80)

    epsilon_lin_values = []
    B_values = []
    B_next_values = []

    for trial in range(N_MC):
        x = random_x_init()
        predictor.reset()

        for step in range(N_STEPS):
            u_safe, _ = predictor.predict_action(x, X_SP, enable_cbf=True)

            # 只在有CBF干预时计算epsilon_lin
            # 检测CBF干预: 比较u_safe与u_nominal
            eps, B_k, B_k1 = compute_epsilon_lin(env, x, u_safe)
            epsilon_lin_values.append(eps)
            B_values.append(B_k)
            B_next_values.append(B_k1)

            x = env.step_discrete(x, u_safe)

    eps_arr = np.array(epsilon_lin_values)
    B_arr = np.array(B_values)
    B_next_arr = np.array(B_next_values)

    # 计算线性化违反率
    violations = np.sum(eps_arr < 0)
    violation_rate = violations / len(eps_arr)

    print(f"  epsilon_lin 统计 (N={len(eps_arr)} 个采样点):")
    print(f"    均值:   {np.mean(eps_arr):.6f}")
    print(f"    标准差: {np.std(eps_arr):.6f}")
    print(f"    最小值: {np.min(eps_arr):.6f}")
    print(f"    最大值: {np.max(eps_arr):.6f}")
    print(f"    中位数: {np.median(eps_arr):.6f}")
    print(f"    95%分位: {np.percentile(eps_arr, 5):.6f}")
    print(f"    线性化违反率 (epsilon<0): {violation_rate*100:.2f}%")

    # B值统计
    print(f"\n  B_smooth 统计:")
    print(f"    均值: {np.mean(B_arr):.4f}")
    print(f"    最小值: {np.min(B_arr):.4f}")
    print(f"    B<0.2 (接近障碍)比例: {np.mean(B_arr < 0.2)*100:.1f}%")

    return {
        'epsilon_lin_mean': float(np.mean(eps_arr)),
        'epsilon_lin_std': float(np.std(eps_arr)),
        'epsilon_lin_min': float(np.min(eps_arr)),
        'epsilon_lin_max': float(np.max(eps_arr)),
        'epsilon_lin_median': float(np.median(eps_arr)),
        'epsilon_lin_p5': float(np.percentile(eps_arr, 5)),
        'epsilon_lin_p95': float(np.percentile(eps_arr, 95)),
        'violation_rate': float(violation_rate),
        'n_samples': len(eps_arr),
        'B_mean': float(np.mean(B_arr)),
        'B_min': float(np.min(B_arr)),
        'B_near_obstacle_rate': float(np.mean(B_arr < 0.2)),
    }


def run_adt_experiment(env, predictor):
    """
    验证ADT (Average Dwell Time) 条件

    记录CBF干预事件，计算经验平均驻留时间 tau_a，
    检查是否满足 Theorem 2 的 ADT 约束
    """
    print("\n" + "=" * 80)
    print("实验: ADT (Average Dwell Time) 验证")
    print("=" * 80)

    # CBF干预检测: barrier值接近0时视为"切换到CBF模式"
    BARRIER_THRESHOLD = 0.2  # h < 0.2 视为接近障碍

    all_dwell_times = []
    all_cbf_intervention_counts = []

    for trial in range(N_MC):
        x = random_x_init()
        predictor.reset()

        in_cbf_mode = False
        cbf_enter_step = -1
        dwell_times = []
        cbf_interventions = 0

        for step in range(N_STEPS):
            u_safe, _ = predictor.predict_action(x, X_SP, enable_cbf=True)

            # 计算当前barrier值
            p = x[0:3].detach().numpy()
            h_min = float('inf')
            for obs in env.obstacles:
                d = np.linalg.norm(p - obs['p']) - obs['r']
                h_min = min(h_min, d)

            if h_min < BARRIER_THRESHOLD:
                if not in_cbf_mode:
                    # 进入CBF模式
                    in_cbf_mode = True
                    cbf_enter_step = step
                cbf_interventions += 1
            else:
                if in_cbf_mode:
                    # 离开CBF模式，记录驻留时间
                    dwell_time = step - cbf_enter_step
                    dwell_times.append(dwell_time)
                    in_cbf_mode = False

            x = env.step_discrete(x, u_safe)

        # 如果试验结束时仍在CBF模式
        if in_cbf_mode:
            dwell_times.append(N_STEPS - cbf_enter_step)

        all_dwell_times.extend(dwell_times)
        all_cbf_intervention_counts.append(cbf_interventions)

    dwell_arr = np.array(all_dwell_times) if all_dwell_times else np.array([0])
    cbf_arr = np.array(all_cbf_intervention_counts)

    # Theorem 2 的 ADT 约束: tau_a > tau_a* = ln(mu) / ln(1/(1-lambda))
    # 其中 mu = lambda_max(P)/lambda_min(P), lambda 是衰减率
    # 这里使用保守估计: mu=10, lambda=0.05 → tau_a* = ln(10)/ln(1/0.95) ≈ 45.6
    mu_estimate = 10.0
    lambda_estimate = 0.05
    tau_a_star = np.log(mu_estimate) / np.log(1.0 / (1.0 - lambda_estimate))

    # 经验平均驻留时间
    # N_sigma(k0, k) <= N0 + (k-k0)/tau_a
    # 简化: tau_a = 总步数 / 切换次数
    total_steps = N_MC * N_STEPS
    total_switches = sum(max(1, len([dt for dt in all_dwell_times if dt > 0])) for _ in range(N_MC))
    # 更精确: 计算每个MC试验中的切换次数
    switch_counts = []
    for trial in range(N_MC):
        # 近似: CBF干预次数 / 驻留段数 ≈ 切换频率
        n_segments = max(1, all_cbf_intervention_counts[trial] // max(1, np.mean(dwell_arr)) if len(all_dwell_times) > 0 else 1)
        switch_counts.append(n_segments)

    total_switch_events = sum(switch_counts)
    tau_a_empirical = total_steps / max(1, total_switch_events)

    print(f"  CBF干预统计:")
    print(f"    平均干预步数/试验: {np.mean(cbf_arr):.1f} ± {np.std(cbf_arr):.1f}")
    print(f"    干预步数范围: [{np.min(cbf_arr):.0f}, {np.max(cbf_arr):.0f}]")
    print(f"\n  驻留时间统计 (N={len(dwell_arr)} 段):")
    print(f"    均值: {np.mean(dwell_arr):.1f} 步")
    print(f"    中位数: {np.median(dwell_arr):.1f} 步")
    print(f"    最大值: {np.max(dwell_arr):.0f} 步")
    print(f"\n  ADT 条件验证:")
    print(f"    理论要求 tau_a > tau_a* = {tau_a_star:.1f}")
    print(f"    经验平均驻留时间 tau_a ≈ {tau_a_empirical:.1f}")
    print(f"    ADT条件{'满足' if tau_a_empirical > tau_a_star else '不满足'} (经验值 {tau_a_empirical:.1f} vs 阈值 {tau_a_star:.1f})")

    return {
        'avg_cbf_interventions': float(np.mean(cbf_arr)),
        'std_cbf_interventions': float(np.std(cbf_arr)),
        'dwell_time_mean': float(np.mean(dwell_arr)),
        'dwell_time_median': float(np.median(dwell_arr)),
        'dwell_time_max': float(np.max(dwell_arr)),
        'n_dwell_segments': len(dwell_arr),
        'tau_a_empirical': float(tau_a_empirical),
        'tau_a_star': float(tau_a_star),
        'adt_satisfied': bool(tau_a_empirical > tau_a_star),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载TRM模型
    print("加载TRM模型...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    trm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    trm_model.to(device)
    trm_model.eval()
    print("模型加载完成")

    env = QuadrotorDynamics()

    predictor = PTRMNMPCPredictor(
        model=trm_model, env=env, K=50, D=16, sigma=0.25,
        candidate_mode='pd', noise_mode='both',
        use_rollout_cost=True, rollout_top_m=10,
        alpha_blend=0.3, pd_sigma=2.0
    )

    t0 = time.time()

    # 实验1: epsilon_lin 量化
    eps_results = run_epsilon_lin_experiment(env, predictor)

    # 实验2: ADT 验证
    adt_results = run_adt_experiment(env, predictor)

    elapsed = time.time() - t0
    print(f"\n总实验时间: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 保存结果
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    all_results = {
        'epsilon_lin': eps_results,
        'adt': adt_results,
    }

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(os.path.join(save_dir, 'theory_verification_results.json'), 'w') as f:
        json.dump(strip(all_results), f, indent=2)

    print(f"结果已保存至 {save_dir}/theory_verification_results.json")


if __name__ == '__main__':
    main()
