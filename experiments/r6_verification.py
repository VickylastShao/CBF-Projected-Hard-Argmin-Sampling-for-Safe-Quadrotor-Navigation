# -*- coding: utf-8 -*-
"""
R6: ADT (Average Dwell Time) 和 epsilon_lin (CBF线性化残差) 验证实验

1. ADT验证: 记录CBF干预切换事件，计算经验平均驻留时间 τ_a
   检查: τ_a > τ_a* (Eq. 5.8 理论阈值)
   
2. epsilon_lin量化: 在CBF投影中记录线性化残差
   epsilon_lin = B_{k+1}(actual) - [(1-γ_d)*B_k + α_d*B_k] (linearized prediction)
   即 epsilon_lin = actual_B_next - linearized_B_next
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


def compute_barrier_value(env, x):
    """计算当前状态的CBF值 B(x) = min over obstacles of (||p - p_obs||^2 - r^2)"""
    p = x[0:3].detach().numpy()
    min_B = float('inf')
    for obs in env.obstacles:
        d_sq = np.sum((p - obs['p'])**2)
        B_i = d_sq - obs['r']**2
        min_B = min(min_B, B_i)
    return min_B


def run_adt_epsilon_experiment(env, predictor, x_sp, n_mc=N_MC):
    """
    收集ADT和epsilon_lin数据

    ADT（手稿 Theorem 2 定义）: 同伦类切换的平均驻留时间
    同伦类切换 ≈ 在非CBF干预期间的控制器大幅跳变

    检测策略:
    1. 记录每一步的CBF激活状态
    2. 只在非CBF干预步之间检测同伦类切换
    3. 使用位置空间的中分面穿越作为辅助检测

    epsilon_lin: CBF线性化残差
    """
    all_dwell_times = []
    all_epsilon_lin = []
    all_B_values = []
    all_cbf_interventions = []
    all_homotopy_switches = []

    # 同伦类切换检测方法：
    # 方法1: CBF非激活期间的控制器大幅跳变（阈值8N ≈ 53% u_max）
    # 方法2: 位置穿越障碍物对之间的中分面
    HOMOTOPY_JUMP_THRESH = 8.0  # N，约 53% 的 u_max

    for trial in range(n_mc):
        set_seed(SEED + trial)
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()

        epsilon_lin_trial = []
        B_values_trial = []
        cbf_intervention_count = 0
        u_prev_noncbf = None  # 只追踪非CBF步的控制
        homotopy_switches = 0
        last_switch_step = 0
        dwell_intervals = []
        positions = []  # 记录位置轨迹

        for step in range(N_STEPS):
            # 记录当前B值
            B_k = compute_barrier_value(env, x)
            B_values_trial.append(B_k)

            # 获取控制（单次调用，避免状态污染）
            result = predictor.predict_action(x, x_sp, enable_cbf=True)
            if isinstance(result, tuple):
                u_safe = result[0]
            else:
                u_safe = result

            # 检测CBF是否干预：用 PD 控制器作为参考
            u_pd = env.m * (4.0 * (x_sp[0:3] - x[0:3]) + 3.0 * (x_sp[3:6] - x[3:6]))
            u_pd_clamped = torch.clamp(u_pd, env.u_min, env.u_max)
            cbf_active = torch.norm(u_safe - u_pd_clamped).item() > 1.0  # CBF使控制偏离PD超过1N

            if cbf_active:
                cbf_intervention_count += 1

            # 同伦类切换检测：只在非CBF干预步之间检测
            # 理由：CBF干预导致的控制跳变是同一避障路径内的安全修正，
            # 不是真正的同伦类切换。远离障碍物时的控制跳变更可能
            # 表示路径选择的变化。
            if not cbf_active:
                if u_prev_noncbf is not None:
                    delta_u = torch.norm(u_safe - u_prev_noncbf).item()
                    if delta_u > HOMOTOPY_JUMP_THRESH:
                        homotopy_switches += 1
                        dwell_interval = step - last_switch_step
                        if dwell_interval > 0:
                            dwell_intervals.append(dwell_interval)
                        last_switch_step = step
                u_prev_noncbf = u_safe.clone()

            # 记录位置
            positions.append(x[0:3].detach().numpy().copy())

            # 前进一步
            x_next = env.step_discrete(x, u_safe)

            # 计算epsilon_lin: 线性化残差
            B_k1_actual = compute_barrier_value(env, x_next)
            B_k1_linearized = (1 - env.gamma_d) * B_k
            eps = B_k1_actual - B_k1_linearized
            epsilon_lin_trial.append(eps)

            x = x_next

        # 最后一段区间（从最后一次切换到试验结束）
        dwell_intervals.append(N_STEPS - last_switch_step)

        all_dwell_times.extend(dwell_intervals)
        all_epsilon_lin.extend(epsilon_lin_trial)
        all_B_values.extend(B_values_trial)
        all_cbf_interventions.append(cbf_intervention_count)
        all_homotopy_switches.append(homotopy_switches)
    
    # 计算统计量
    # ADT: 同伦类切换的平均驻留时间
    if len(all_dwell_times) > 0:
        tau_a = np.mean(all_dwell_times)
        tau_a_std = np.std(all_dwell_times)
        tau_a_min = np.min(all_dwell_times)
    else:
        tau_a = float('inf')
        tau_a_std = 0
        tau_a_min = float('inf')

    # 理论阈值 τ_a* (Eq. 5.8)
    tau_a_star = 44.9  # 手稿 Eq. 5.8

    # epsilon_lin 统计
    eps_arr = np.array(all_epsilon_lin)
    eps_positive_rate = np.mean(eps_arr >= 0) * 100

    return {
        'tau_a': float(tau_a),
        'tau_a_std': float(tau_a_std),
        'tau_a_min': float(tau_a_min),
        'tau_a_star': float(tau_a_star),
        'tau_a_exceeds_threshold': bool(tau_a > tau_a_star),
        'n_dwell_intervals': len(all_dwell_times),
        'n_homotopy_switches_mean': float(np.mean(all_homotopy_switches)),
        'n_homotopy_switches_total': int(sum(all_homotopy_switches)),
        'epsilon_lin_mean': float(np.mean(eps_arr)),
        'epsilon_lin_std': float(np.std(eps_arr)),
        'epsilon_lin_max': float(np.max(eps_arr)),
        'epsilon_lin_min': float(np.min(eps_arr)),
        'epsilon_lin_positive_rate': float(eps_positive_rate),
        'B_mean': float(np.mean(all_B_values)),
        'B_min': float(np.min(all_B_values)),
        'avg_cbf_interventions_per_trial': float(np.mean(all_cbf_interventions)),
    }


def main():
    set_seed(SEED)
    
    # 加载模型
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    trm_model.load_state_dict(torch.load(trm_path, map_location='cpu', weights_only=True))
    trm_model.eval()
    
    # ========== 1. PTRM K=50 ADT & epsilon_lin ==========
    print("=" * 60)
    print("R6 验证: ADT & epsilon_lin (PTRM K=50, Strong CBF)")
    print("=" * 60)
    
    env = QuadrotorDynamics()
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    
    ptrm_results = run_adt_epsilon_experiment(env, predictor, X_SP)
    
    print(f"\n--- ADT 验证 (同伦类切换) ---")
    print(f"  经验 τ_a = {ptrm_results['tau_a']:.1f} ± {ptrm_results['tau_a_std']:.1f}")
    print(f"  最小 τ_a = {ptrm_results['tau_a_min']:.0f}")
    print(f"  理论阈值 τ_a* = {ptrm_results['tau_a_star']:.1f}")
    print(f"  τ_a > τ_a* ? {'✓ YES' if ptrm_results['tau_a_exceeds_threshold'] else '✗ NO'}")
    print(f"  驻留区间数: {ptrm_results['n_dwell_intervals']}")
    print(f"  平均同伦切换次数/trial: {ptrm_results['n_homotopy_switches_mean']:.1f}")
    print(f"  总同伦切换次数: {ptrm_results['n_homotopy_switches_total']}")
    
    print(f"\n--- epsilon_lin 验证 ---")
    print(f"  ε_lin 均值 = {ptrm_results['epsilon_lin_mean']:.6f}")
    print(f"  ε_lin 标准差 = {ptrm_results['epsilon_lin_std']:.6f}")
    print(f"  ε_lin 最大值 = {ptrm_results['epsilon_lin_max']:.6f}")
    print(f"  ε_lin 最小值 = {ptrm_results['epsilon_lin_min']:.6f}")
    print(f"  ε_lin ≥ 0 比例 = {ptrm_results['epsilon_lin_positive_rate']:.1f}%")
    
    print(f"\n--- CBF 活动统计 ---")
    print(f"  平均CBF干预步数/trial = {ptrm_results['avg_cbf_interventions_per_trial']:.1f}")
    print(f"  B值均值 = {ptrm_results['B_mean']:.4f}")
    print(f"  B值最小值 = {ptrm_results['B_min']:.6f}")
    
    # ========== 2. 不同K值的对比 ==========
    print("\n" + "=" * 60)
    print("ADT & epsilon_lin vs K值")
    print("=" * 60)
    
    all_results = {'ptrm_K50': ptrm_results}
    
    for k in [1, 10, 100]:
        env_k = QuadrotorDynamics()
        sigma = 0.25 if k > 1 else 0.0
        predictor_k = PTRMNMPCPredictor(trm_model, env_k, K=k, D=16, sigma=sigma,
                                         alpha_blend=0.3, candidate_mode='pd',
                                         pd_sigma=2.0, use_rollout_cost=True)
        res_k = run_adt_epsilon_experiment(env_k, predictor_k, X_SP)
        all_results[f'ptrm_K{k}'] = res_k
        print(f"  K={k:3d}: τ_a={res_k['tau_a']:.1f}, ε_lin_mean={res_k['epsilon_lin_mean']:.6f}, "
              f"ε_lin≥0: {res_k['epsilon_lin_positive_rate']:.1f}%, "
              f"CBF干预: {res_k['avg_cbf_interventions_per_trial']:.1f}")
    
    # ========== 3. 模型失配下的验证 ==========
    print("\n" + "=" * 60)
    print("ADT & epsilon_lin with model mismatch (+50% mass/drag)")
    print("=" * 60)

    env_mm = QuadrotorDynamics()
    predictor_mm = PTRMNMPCPredictor(trm_model, env_mm, K=50, D=16, sigma=0.25,
                                      alpha_blend=0.3, candidate_mode='pd',
                                      pd_sigma=2.0, use_rollout_cost=True)

    HOMOTOPY_JUMP_THRESH = 8.0
    mismatch_results = {'tau_a': [], 'epsilon_lin_mean': [], 'epsilon_lin_positive_rate': [],
                         'homotopy_switches': []}

    for trial in range(N_MC):
        set_seed(SEED + trial)
        x_init = random_x_init()
        predictor_mm.reset()
        x = x_init.clone()

        epsilon_lin_trial = []
        u_prev_noncbf = None
        homotopy_switches = 0
        last_switch_step = 0
        dwell_intervals = []

        for step in range(N_STEPS):
            B_k = compute_barrier_value(env_mm, x)

            result = predictor_mm.predict_action(x, X_SP, enable_cbf=True)
            if isinstance(result, tuple):
                u_safe = result[0]
            else:
                u_safe = result

            # CBF干预检测
            u_pd = env_mm.m * (4.0 * (X_SP[0:3] - x[0:3]) + 3.0 * (X_SP[3:6] - x[3:6]))
            u_pd_clamped = torch.clamp(u_pd, env_mm.u_min, env_mm.u_max)
            cbf_active = torch.norm(u_safe - u_pd_clamped).item() > 1.0

            # 同伦类切换检测：只在非CBF干预步检测
            if not cbf_active:
                if u_prev_noncbf is not None:
                    delta_u = torch.norm(u_safe - u_prev_noncbf).item()
                    if delta_u > HOMOTOPY_JUMP_THRESH:
                        homotopy_switches += 1
                        dwell_interval = step - last_switch_step
                        if dwell_interval > 0:
                            dwell_intervals.append(dwell_interval)
                        last_switch_step = step
                u_prev_noncbf = u_safe.clone()

            x_next = env_mm.step_discrete(x, u_safe, use_mismatch=True)
            B_k1_actual = compute_barrier_value(env_mm, x_next)
            B_k1_linearized = (1 - env_mm.gamma_d) * B_k
            eps = B_k1_actual - B_k1_linearized
            epsilon_lin_trial.append(eps)

            x = x_next

        dwell_intervals.append(N_STEPS - last_switch_step)

        if len(dwell_intervals) > 0:
            mismatch_results['tau_a'].append(np.mean(dwell_intervals))
        eps_arr = np.array(epsilon_lin_trial)
        mismatch_results['epsilon_lin_mean'].append(float(np.mean(eps_arr)))
        mismatch_results['epsilon_lin_positive_rate'].append(float(np.mean(eps_arr >= 0) * 100))
        mismatch_results['homotopy_switches'].append(homotopy_switches)

    mm_tau_a = np.mean(mismatch_results['tau_a']) if mismatch_results['tau_a'] else float('inf')
    mm_eps_mean = np.mean(mismatch_results['epsilon_lin_mean'])
    mm_eps_pos = np.mean(mismatch_results['epsilon_lin_positive_rate'])
    mm_homotopy_mean = np.mean(mismatch_results['homotopy_switches']) if mismatch_results['homotopy_switches'] else 0

    print(f"  Mismatch: τ_a={mm_tau_a:.1f}, ε_lin_mean={mm_eps_mean:.6f}, ε_lin≥0: {mm_eps_pos:.1f}%")
    print(f"  Mismatch: homotopy switches mean={mm_homotopy_mean:.1f}, total={sum(mismatch_results['homotopy_switches'])}")

    all_results['mismatch_K50'] = {
        'tau_a': float(mm_tau_a),
        'epsilon_lin_mean': float(mm_eps_mean),
        'epsilon_lin_positive_rate': float(mm_eps_pos),
        'homotopy_switches_mean': float(mm_homotopy_mean),
        'homotopy_switches_total': int(sum(mismatch_results['homotopy_switches'])),
    }
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'r6_verification_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}/r6_verification_results.json")


if __name__ == '__main__':
    main()
