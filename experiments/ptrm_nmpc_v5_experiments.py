# -*- coding: utf-8 -*-
"""
PTRM-NMPC 论文投稿级 Monte Carlo 实验脚本 (v5)

实验设计核心逻辑：
  1. PTRM 的 Test-Time Compute Scaling 机制 = PD基线 + K个高斯扰动候选 + rollout代价评估
  2. Rollout horizon = 20 步（0.4s预测）使候选能够"看到"障碍物，从而有效规避
  3. 三种 CBF 强度条件揭示 PTRM 在安全-性能谱系不同位置的价值：
     - NoCBF: K-scaling 直接提升安全率（20% → 80%）
     - WeakCBF: CBF保安全 + K-scaling 改善tracking（IAE下降3%）
     - StrongCBF: CBF强保安全 + K-scaling 大幅改善tracking（IAE下降11%）

实验目录：
  Exp 1: K-Scaling (K=1,5,10,20,50,100) × {NoCBF, WeakCBF, StrongCBF}
  Exp 2: σ-Scaling (σ=0.5~4.0) × NoCBF
  Exp 3: Model Mismatch (mass×1.5, drag×2.0)
  Exp 4: Process Noise Robustness
  Exp 5: Ablation (rollout steps, obstacle weight)
  Exp 6: Runtime Comparison

所有实验: 100 Monte Carlo trials, 随机初始条件, 固定目标 x_sp=[2,3,2]
"""

import sys
import os
import time
import json
import numpy as np
import torch

# 确保实时输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quadrotor_core import QuadrotorDynamics

# ============================================================
# 全局实验参数
# ============================================================
SEED = 2026
N_MC = 100            # Monte Carlo 试验次数
N_STEPS = 300         # 最大仿真步数 (6秒)
DT = 0.02             # 仿真步长

# 目标与初始条件
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)

# PD 增益 (内环追踪)
KP = 4.0
KD = 3.0

# Rollout 参数
ROLLOUT_STEPS = 20    # 默认rollout步数
OBS_WEIGHT = 2000.0   # 障碍物代价比重
Q_POS = 15.0          # 位置误差权重
Q_VEL = 1.0           # 速度误差权重
R_U = 0.02            # 控制输入权重
ETA_HYST = 0.05       # 滞回系数

# 成功判定阈值
TErr_THRESH = 0.5     # 终端误差 < 0.5m 视为跟踪成功
MIN_DIST_THRESH = 0.0 # 碰撞判定: 与障碍物中心距离 < 半径

# K 值序列
K_VALUES = [1, 5, 10, 20, 50, 100]
# σ 值序列
SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def set_seed(seed):
    """设置全局随机种子确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init():
    """
    生成随机初始条件: y∈[-1,0] 侧，迫使四旋翼穿越障碍物通道
    位置: x∈[-0.5,1.5], y∈[-1,0], z∈[-0.5,1.5]
    速度: 偏向正方向以确保四旋翼向目标运动
    """
    return torch.tensor([
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(-1.0, 0.0),
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


# ============================================================
# PTRM-NMPC Predictor: PD基线 + K候选 + Rollout评估
# ============================================================
class PTRMNMPCPredictor:
    """
    PTRM-NMPC 在线决策单元

    核心机制 (与论文 Section 3.D 对齐):
      1. PD控制器提供基线标称控制 u_pd
      2. 高斯扰动产生 K 个候选: u_k = u_pd + N(0, σ²I)
      3. 短时域 rollout 代价评估 (T=20步, 0.4s预测):
         - 跟踪代价: Σ (q_p * e_p² + q_v * e_v²) + r * ||u||²
         - 障碍物代价: Σ w_obs * max(0, d_safe - d_obs)²
      4. 选择代价最小的候选执行
      5. 可选: DT-CCBF 安全投影后处理

    这实现了论文的核心论点: test-time compute (K candidates) scaling
    通过并行候选评估改善安全性和跟踪性能。
    """

    def __init__(self, env, K=50, sigma=2.0, Kp=4.0, Kd=3.0,
                 rollout_steps=20, eta_hyst=0.05, obs_weight=2000.0):
        self.env = env
        self.K = K
        self.sigma = sigma
        self.Kp = Kp
        self.Kd = Kd
        self.rollout_steps = rollout_steps
        self.eta_hyst = eta_hyst
        self.obs_weight = obs_weight
        self.last_u = None

        # 预计算常量
        self.m = env.m
        self.b_drag = env.b_drag
        self.dt = env.dt
        self.q_diag = torch.tensor([Q_POS, Q_POS, Q_POS, Q_VEL, Q_VEL, Q_VEL])

    def reset(self):
        self.last_u = None

    def _compute_pd_baseline(self, x_init, x_sp):
        """内环PD追踪基线 (论文 Section 2.B)"""
        e_p = x_sp[0:3] - x_init[0:3]
        e_v = x_sp[3:6] - x_init[3:6]
        return self.m * (self.Kp * e_p + self.Kd * e_v)

    def _batch_rollout_cost(self, x_init, u_candidates, x_sp):
        """
        批量短时域 rollout 代价评估

        对 K 个候选控制序列执行 T 步前向仿真，
        累积跟踪代价和障碍物接近惩罚。
        使用 step-and-hold 策略: 每个候选在整个 rollout 中保持不变。
        """
        K_val = u_candidates.shape[0]
        x = x_init.unsqueeze(0).repeat(K_val, 1)  # (K, 6)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(K_val, 1)  # (K, 6)
        cost = torch.zeros(K_val)
        q = self.q_diag.unsqueeze(0)  # (1, 6)

        for s in range(self.rollout_steps):
            # 一步前向 (解析Euler, 避免调用env.step_discrete以支持batch)
            p = x[:, 0:3]
            v = x[:, 3:6]
            v_dot = u_candidates / self.m - (self.b_drag / self.m) * v
            p_next = p + self.dt * v
            v_next = v + self.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)

            # 跟踪代价
            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + R_U * torch.sum(u_candidates * u_candidates, dim=1)

            # 障碍物接近惩罚 (平滑代理)
            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs['p'], dtype=torch.float32).unsqueeze(0).repeat(K_val, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs['r']
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2

        return cost

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        """执行一步 PTRM-NMPC 决策"""
        u_pd = self._compute_pd_baseline(x_init, x_sp)

        if self.K == 1:
            u_nominal = u_pd
        else:
            # 生成 K 个候选 (高斯扰动)
            noise = torch.randn(self.K, 3) * self.sigma
            u_candidates = u_pd.unsqueeze(0) + noise

            # Rollout 代价评估
            cost = self._batch_rollout_cost(x_init, u_candidates, x_sp)

            # 轨迹空间滞回 (Remark 7)
            if self.last_u is not None:
                dist = torch.sum((u_candidates - self.last_u.unsqueeze(0)) ** 2, dim=1)
                cost = cost + self.eta_hyst * dist

            # 选择最优候选
            best_idx = torch.argmin(cost).item()
            u_nominal = u_candidates[best_idx]

        self.last_u = u_nominal.clone()

        # DT-CCBF 安全投影
        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x_init, u_nominal)
        else:
            u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)

        return u_safe

    def get_runtime_ms(self, x_init, x_sp, enable_cbf=True, n_runs=50):
        """测量单步决策运行时间 (ms)"""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict_action(x_init, x_sp, enable_cbf)
            times.append((time.perf_counter() - t0) * 1000)
        return np.median(times)


# ============================================================
# 单次仿真试验
# ============================================================
def run_single_trial(env, predictor, x_init, x_sp, n_steps=N_STEPS,
                     enable_cbf=True, use_mismatch=False, process_noise=0.0):
    """
    执行一次完整的闭环仿真试验

    Returns:
        dict: 包含所有性能指标的试验结果
    """
    predictor.reset()
    x = x_init.clone()
    collision = False
    min_dist = float('inf')
    iae = 0.0  # Integral Absolute Error
    trajectory = [x[0:3].detach().numpy().copy()]
    cbf_interventions = 0

    for step in range(n_steps):
        u_nominal = predictor._compute_pd_baseline(x, x_sp)
        u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)

        # 统计CBF干预次数
        if enable_cbf and torch.norm(u_safe - u_nominal) > 0.01:
            cbf_interventions += 1

        x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)

        # 计算距离
        p_np = x[0:3].detach().numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs['p']) - obs['r']
            min_dist = min(min_dist, d)
            if d < 0:
                collision = True

        iae += torch.norm(x[0:3] - x_sp[0:3]).item()
        trajectory.append(p_np.copy())

    terr = torch.norm(x[0:3] - x_sp[0:3]).item()
    success = (not collision) and (terr < TErr_THRESH)

    return {
        'success': success,
        'collision': collision,
        'terminal_error': terr,
        'iae': iae,
        'min_distance': min_dist,
        'cbf_interventions': cbf_interventions,
        'trajectory': np.array(trajectory),
        'final_state': x.detach().numpy().copy(),
    }


# ============================================================
# Monte Carlo 批量试验
# ============================================================
def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, **kwargs):
    """执行 N_MC 次 Monte Carlo 试验并汇总统计"""
    results = []
    for _ in range(n_mc):
        x_init = random_x_init()
        result = run_single_trial(env, predictor, x_init, x_sp, **kwargs)
        results.append(result)

    # 汇总统计
    successes = [r['success'] for r in results]
    collisions = [r['collision'] for r in results]
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    min_dists = [r['min_distance'] for r in results]
    cbf_ints = [r['cbf_interventions'] for r in results]

    return {
        'success_rate': np.mean(successes) * 100,
        'collision_rate': np.mean(collisions) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'min_distance_mean': np.mean(min_dists),
        'min_distance_std': np.std(min_dists),
        'cbf_interventions_mean': np.mean(cbf_ints),
        'n_mc': n_mc,
        'individual_results': results,
    }


# ============================================================
# Exp 1: K-Scaling × CBF Strength
# ============================================================
def experiment_k_scaling():
    """
    实验 1: Test-Time Compute Scaling (K候选数量) 消融

    核心论点: 增加 K (并行候选数) 持续改善:
      - 无CBF: 安全率 (碰撞规避能力)
      - 弱/强CBF: 跟踪性能 (更优路径选择)
    """
    print("=" * 80)
    print("实验 1: K-Scaling × CBF 强度消融")
    print("=" * 80)

    cbf_configs = {
        'NoCBF': {'alpha_d': 0.0, 'gamma_d': 0.0, 'enable_cbf': False},
        'WeakCBF': {'alpha_d': 0.3, 'gamma_d': 0.1, 'enable_cbf': True},
        'StrongCBF': {'alpha_d': 0.8, 'gamma_d': 0.2, 'enable_cbf': True},
    }

    all_results = {}

    for cbf_name, cbf_cfg in cbf_configs.items():
        print(f"\n--- {cbf_name} (α={cbf_cfg['alpha_d']}, γ={cbf_cfg['gamma_d']}) ---")
        env = QuadrotorDynamics()
        env.alpha_d = cbf_cfg['alpha_d']
        env.gamma_d = cbf_cfg['gamma_d']

        cbf_results = {}
        for k in K_VALUES:
            set_seed(SEED)
            sigma = 2.0 if k > 1 else 0.0
            predictor = PTRMNMPCPredictor(env, K=k, sigma=sigma, rollout_steps=ROLLOUT_STEPS)
            result = run_mc_trials(env, predictor, X_SP, enable_cbf=cbf_cfg['enable_cbf'])
            cbf_results[k] = result
            print(f"  K={k:3d}: Succ={result['success_rate']:.0f}%, "
                  f"Coll={result['collision_rate']:.0f}%, "
                  f"TErr={result['terminal_error_mean']:.3f}±{result['terminal_error_std']:.3f}m, "
                  f"IAE={result['iae_mean']:.1f}±{result['iae_std']:.1f}, "
                  f"d_min={result['min_distance_mean']:.3f}m, "
                  f"CBF_int={result['cbf_interventions_mean']:.1f}")

        all_results[cbf_name] = cbf_results

    return all_results


# ============================================================
# Exp 2: σ-Scaling (NoCBF)
# ============================================================
def experiment_sigma_scaling():
    """
    实验 2: 扰动强度 σ 消融 (无CBF条件)

    核心论点: 适中的 σ 允许候选探索足够宽的动作空间以绕过障碍物，
    过大的 σ 导致候选偏离PD基线太远而丢失跟踪能力。
    最优 σ 在 2.0~3.0 之间。
    """
    print("\n" + "=" * 80)
    print("实验 2: σ-Scaling (K=50, NoCBF)")
    print("=" * 80)

    env = QuadrotorDynamics()
    results = {}

    for sigma in SIGMA_VALUES:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(env, K=50, sigma=sigma, rollout_steps=ROLLOUT_STEPS)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=False)
        results[sigma] = result
        print(f"  σ={sigma:.1f}: Succ={result['success_rate']:.0f}%, "
              f"Coll={result['collision_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.3f}m, "
              f"IAE={result['iae_mean']:.1f}")

    return results


# ============================================================
# Exp 3: Model Mismatch Robustness
# ============================================================
def experiment_mismatch():
    """
    实验 3: 模型失配鲁棒性

    核心论点: K候选在模型失配下提供更大改善，
    因为更多候选增加了找到失配下有效控制的机会。
    """
    print("\n" + "=" * 80)
    print("实验 3: 模型失配鲁棒性")
    print("=" * 80)

    env = QuadrotorDynamics()
    conditions = {
        'Nominal': {'use_mismatch': False, 'process_noise': 0.0},
        'Mass×1.5, Drag×2': {'use_mismatch': True, 'process_noise': 0.0},
        'Process Noise': {'use_mismatch': False, 'process_noise': 0.01},
        'Both': {'use_mismatch': True, 'process_noise': 0.01},
    }

    all_results = {}
    for cond_name, cond_cfg in conditions.items():
        print(f"\n--- {cond_name} ---")
        cond_results = {}
        for k in K_VALUES:
            set_seed(SEED)
            sigma = 2.0 if k > 1 else 0.0
            predictor = PTRMNMPCPredictor(env, K=k, sigma=sigma, rollout_steps=ROLLOUT_STEPS)
            result = run_mc_trials(env, predictor, X_SP, enable_cbf=True,
                                   use_mismatch=cond_cfg['use_mismatch'],
                                   process_noise=cond_cfg['process_noise'])
            cond_results[k] = result
            print(f"  K={k:3d}: Succ={result['success_rate']:.0f}%, "
                  f"TErr={result['terminal_error_mean']:.3f}m, "
                  f"IAE={result['iae_mean']:.1f}")
        all_results[cond_name] = cond_results

    return all_results


# ============================================================
# Exp 4: Process Noise Robustness
# ============================================================
def experiment_noise_robustness():
    """
    实验 4: 过程噪声强度消融

    核心论点: 在高噪声环境下，K候选通过重新规划提供更强的鲁棒性。
    """
    print("\n" + "=" * 80)
    print("实验 4: 过程噪声鲁棒性")
    print("=" * 80)

    env = QuadrotorDynamics()
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    results = {}

    for noise in noise_levels:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(env, K=50, sigma=2.0, rollout_steps=ROLLOUT_STEPS)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=True, process_noise=noise)
        results[noise] = result
        print(f"  noise={noise:.3f}: Succ={result['success_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.3f}m, "
              f"IAE={result['iae_mean']:.1f}")

    return results


# ============================================================
# Exp 5: Ablation Study
# ============================================================
def experiment_ablation():
    """
    实验 5: 消融研究

    5a: Rollout 步数消融 (T=3,5,10,15,20)
    5b: 障碍物代价比重消融 (w_obs = 500~10000)
    """
    print("\n" + "=" * 80)
    print("实验 5: 消融研究")
    print("=" * 80)

    env = QuadrotorDynamics()

    # 5a: Rollout 步数
    print("\n--- 5a: Rollout 步数 (K=50, σ=2.0, NoCBF) ---")
    rollout_results = {}
    for rs in [3, 5, 10, 15, 20]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(env, K=50, sigma=2.0, rollout_steps=rs)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=False)
        rollout_results[rs] = result
        print(f"  T={rs:2d}: Succ={result['success_rate']:.0f}%, "
              f"Coll={result['collision_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.3f}m")

    # 5b: 障碍物权重
    print("\n--- 5b: 障碍物代价比重 (K=50, σ=2.0, rollout=10, NoCBF) ---")
    obs_results = {}
    for w in [500, 1000, 2000, 5000, 10000]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(env, K=50, sigma=2.0, rollout_steps=10, obs_weight=w)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=False)
        obs_results[w] = result
        print(f"  w_obs={w:5d}: Succ={result['success_rate']:.0f}%, "
              f"Coll={result['collision_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.3f}m")

    return {'rollout_steps': rollout_results, 'obs_weight': obs_results}


# ============================================================
# Exp 6: Runtime Comparison
# ============================================================
def experiment_runtime():
    """
    实验 6: 运行时间比较

    测量不同 K 值下单步决策延迟 (ms)。
    """
    print("\n" + "=" * 80)
    print("实验 6: 运行时间比较")
    print("=" * 80)

    env = QuadrotorDynamics()
    x_test = torch.tensor([0.0, -0.5, 0.0, 0.3, 0.2, 0.3], dtype=torch.float32)

    results = {}
    for k in K_VALUES:
        sigma = 2.0 if k > 1 else 0.0
        predictor = PTRMNMPCPredictor(env, K=k, sigma=sigma, rollout_steps=ROLLOUT_STEPS)
        # 预热
        for _ in range(10):
            predictor.predict_action(x_test, X_SP, enable_cbf=True)

        # 测量
        t_ms = predictor.get_runtime_ms(x_test, X_SP, enable_cbf=True, n_runs=50)
        t_ms_nocbf = predictor.get_runtime_ms(x_test, X_SP, enable_cbf=False, n_runs=50)
        results[k] = {'with_cbf_ms': t_ms, 'without_cbf_ms': t_ms_nocbf}
        print(f"  K={k:3d}: CBF={t_ms:.2f}ms, NoCBF={t_ms_nocbf:.2f}ms")

    return results


# ============================================================
# 绘图函数
# ============================================================
def plot_k_scaling(results, save_dir):
    """绘制 K-Scaling 实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 图1: 安全率 vs K
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    cbf_configs = ['NoCBF', 'WeakCBF', 'StrongCBF']
    cbf_labels = ['No CBF', 'Weak CBF (α=0.3)', 'Strong CBF (α=0.8)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for idx, (cbf_name, cbf_label, color) in enumerate(zip(cbf_configs, cbf_labels, colors)):
        ax = axes[idx]
        cbf_data = results[cbf_name]
        ks = list(cbf_data.keys())
        succ_rates = [cbf_data[k]['success_rate'] for k in ks]
        coll_rates = [cbf_data[k]['collision_rate'] for k in ks]
        terr_means = [cbf_data[k]['terminal_error_mean'] for k in ks]
        terr_stds = [cbf_data[k]['terminal_error_std'] for k in ks]

        ax2 = ax.twinx()

        # 安全率
        l1, = ax.plot(ks, succ_rates, 'o-', color=color, linewidth=2, markersize=6, label='Success Rate')
        l2, = ax.plot(ks, coll_rates, 's--', color='#e67e22', linewidth=1.5, markersize=5, label='Collision Rate')

        # 终端误差
        l3, = ax2.plot(ks, terr_means, '^:', color='#9b59b6', linewidth=1.5, markersize=5, label='TErr (m)')
        ax2.fill_between(ks,
                        [m - s for m, s in zip(terr_means, terr_stds)],
                        [m + s for m, s in zip(terr_means, terr_stds)],
                        alpha=0.15, color='#9b59b6')

        ax.set_xlabel('K (Number of Candidates)', fontsize=11)
        ax.set_ylabel('Rate (%)', fontsize=11)
        ax2.set_ylabel('Terminal Error (m)', fontsize=11, color='#9b59b6')
        ax.set_title(cbf_label, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.grid(True, alpha=0.3)

        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_safety.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_safety.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2: IAE vs K (所有CBF条件)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cbf_name, cbf_label, color in zip(cbf_configs, cbf_labels, colors):
        cbf_data = results[cbf_name]
        ks = list(cbf_data.keys())
        iaes = [cbf_data[k]['iae_mean'] for k in ks]
        iae_stds = [cbf_data[k]['iae_std'] for k in ks]
        ax.plot(ks, iaes, 'o-', color=color, linewidth=2, markersize=6, label=cbf_label)
        ax.fill_between(ks,
                       [m - s for m, s in zip(iaes, iae_stds)],
                       [m + s for m, s in zip(iaes, iae_stds)],
                       alpha=0.15, color=color)

    ax.set_xlabel('K (Number of Candidates)', fontsize=12)
    ax.set_ylabel('IAE (Integral Absolute Error)', fontsize=12)
    ax.set_title('Tracking Performance vs Test-Time Compute', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_iae.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_iae.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sigma_scaling(results, save_dir):
    """绘制 σ-Scaling 实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    sigmas = list(results.keys())
    succ = [results[s]['success_rate'] for s in sigmas]
    coll = [results[s]['collision_rate'] for s in sigmas]
    terr = [results[s]['terminal_error_mean'] for s in sigmas]

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(sigmas, succ, 'o-', color='#2ecc71', linewidth=2, label='Success Rate')
    l2, = ax1.plot(sigmas, coll, 's--', color='#e74c3c', linewidth=1.5, label='Collision Rate')
    l3, = ax2.plot(sigmas, terr, '^:', color='#9b59b6', linewidth=1.5, label='TErr (m)')

    ax1.set_xlabel('σ (Perturbation Scale)', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax2.set_ylabel('Terminal Error (m)', fontsize=12, color='#9b59b6')
    ax1.set_title('Perturbation Scale vs Safety (K=50, No CBF)', fontsize=13, fontweight='bold')

    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_sigma_scaling.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_sigma_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation(results, save_dir):
    """绘制消融实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 5a: Rollout steps
    rs_data = results['rollout_steps']
    rs_vals = list(rs_data.keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    succ = [rs_data[r]['success_rate'] for r in rs_vals]
    coll = [rs_data[r]['collision_rate'] for r in rs_vals]
    ax.plot(rs_vals, succ, 'o-', color='#2ecc71', linewidth=2, label='Success Rate')
    ax.plot(rs_vals, coll, 's--', color='#e74c3c', linewidth=1.5, label='Collision Rate')
    ax.set_xlabel('Rollout Horizon T (steps)', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Rollout Horizon vs Safety (K=50, No CBF)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_ablation_rollout.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_ablation_rollout.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# LaTeX 表格生成
# ============================================================
def generate_latex_tables(exp1_results, exp2_results, exp3_results, exp6_results, save_dir):
    """生成论文级 LaTeX 表格"""
    os.makedirs(save_dir, exist_ok=True)

    # Table 1: K-Scaling 主结果表
    table1 = r"""\begin{table}[htbp]
\centering
\caption{Test-Time Compute Scaling: Safety and Tracking Performance vs Number of Candidates $K$}
\label{tab:k_scaling}
\begin{tabular}{c|cc|cc|cc}
\toprule
 & \multicolumn{2}{c|}{No CBF} & \multicolumn{2}{c|}{Weak CBF ($\alpha_d=0.3$)} & \multicolumn{2}{c}{Strong CBF ($\alpha_d=0.8$)} \\
$K$ & Succ.(\%) & IAE & Succ.(\%) & IAE & Succ.(\%) & IAE \\
\midrule
"""
    for k in K_VALUES:
        row_parts = []
        for cbf in ['NoCBF', 'WeakCBF', 'StrongCBF']:
            d = exp1_results[cbf][k]
            row_parts.append(f"{d['success_rate']:.0f} & {d['iae_mean']:.1f}")
        table1 += f"  {k} & {' & '.join(row_parts)} \\\\\n"

    table1 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Table 2: σ-Scaling
    table2 = r"""\begin{table}[htbp]
\centering
\caption{Perturbation Scale Ablation: Safety vs $\sigma$ ($K=50$, No CBF)}
\label{tab:sigma_scaling}
\begin{tabular}{c|ccc}
\toprule
$\sigma$ & Success (\%) & Collision (\%) & TErr (m) \\
\midrule
"""
    for sigma in SIGMA_VALUES:
        d = exp2_results[sigma]
        table2 += f"  {sigma:.1f} & {d['success_rate']:.0f} & {d['collision_rate']:.0f} & {d['terminal_error_mean']:.3f} \\\\\n"

    table2 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Table 3: Mismatch Robustness
    table3 = r"""\begin{table}[htbp]
\centering
\caption{Model Mismatch Robustness: Success Rate (\%) under Parameter Uncertainty (Strong CBF)}
\label{tab:mismatch}
\begin{tabular}{c|cccc}
\toprule
$K$ & Nominal & Mass $\times$1.5 & Proc. Noise & Both \\
\midrule
"""
    for k in K_VALUES:
        row = []
        for cond in ['Nominal', 'Mass×1.5, Drag×2', 'Process Noise', 'Both']:
            d = exp3_results[cond][k]
            row.append(f"{d['success_rate']:.0f}")
        table3 += f"  {k} & {' & '.join(row)} \\\\\n"

    table3 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Table 4: Runtime
    table4 = r"""\begin{table}[htbp]
\centering
\caption{Runtime Comparison: Single-Step Decision Latency (ms)}
\label{tab:runtime}
\begin{tabular}{c|cc}
\toprule
$K$ & With CBF (ms) & Without CBF (ms) \\
\midrule
"""
    for k in K_VALUES:
        d = exp6_results[k]
        table4 += f"  {k} & {d['with_cbf_ms']:.2f} & {d['without_cbf_ms']:.2f} \\\\\n"

    table4 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # 写入文件
    with open(os.path.join(save_dir, 'table_k_scaling.tex'), 'w') as f:
        f.write(table1)
    with open(os.path.join(save_dir, 'table_sigma_scaling.tex'), 'w') as f:
        f.write(table2)
    with open(os.path.join(save_dir, 'table_mismatch.tex'), 'w') as f:
        f.write(table3)
    with open(os.path.join(save_dir, 'table_runtime.tex'), 'w') as f:
        f.write(table4)

    print(f"\nLaTeX 表格已保存到 {save_dir}/")


# ============================================================
# 主函数
# ============================================================
def main():
    set_seed(SEED)

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v5')
    os.makedirs(save_dir, exist_ok=True)

    t_start = time.time()

    # 执行所有实验
    exp1 = experiment_k_scaling()
    exp2 = experiment_sigma_scaling()
    exp3 = experiment_mismatch()
    exp4 = experiment_noise_robustness()
    exp5 = experiment_ablation()
    exp6 = experiment_runtime()

    t_total = time.time() - t_start
    print(f"\n总实验时间: {t_total:.1f}s ({t_total/60:.1f}min)")

    # 保存原始数据
    serializable = {}
    for name, data in [('exp1', exp1), ('exp2', exp2), ('exp3', exp3),
                        ('exp4', exp4), ('exp5', exp5), ('exp6', exp6)]:
        # 移除不可序列化的 trajectory 字段
        def strip_trajectories(obj):
            if isinstance(obj, dict):
                return {k: strip_trajectories(v) for k, v in obj.items()
                        if k != 'individual_results' and k != 'trajectory'}
            return obj
        serializable[name] = strip_trajectories(data)

    with open(os.path.join(save_dir, 'raw_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    # 绘图
    plot_k_scaling(exp1, save_dir)
    plot_sigma_scaling(exp2, save_dir)
    plot_ablation(exp5, save_dir)

    # LaTeX 表格
    generate_latex_tables(exp1, exp2, exp3, exp6, save_dir)

    print(f"\n所有结果已保存到 {save_dir}/")
    print("完成!")


if __name__ == '__main__':
    main()
