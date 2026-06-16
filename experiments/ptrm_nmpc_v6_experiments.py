# -*- coding: utf-8 -*-
"""
PTRM-NMPC 论文投稿级 Monte Carlo 实验脚本 (v6)

v6 修订核心变更（针对审稿意见 R1-R9, S1-S15）:
  1. TRM网络参与候选生成与评估（Q-head粗筛 + rollout精排）
  2. 添加 PD 修正消融 (R1): alpha_blend ∈ {0.0, 0.1, 0.2, 0.3, 0.5}
  3. 添加噪声通道消融 (R2): noise_mode ∈ {both, latent, output}
  4. 添加 MPPI 基线 (R3)
  5. 添加 MLP+CBF 基线 (R4)
  6. 添加 CEM 基线 (S3)
  7. 多障碍物配置实验 (R5): Corridor, Dense-5, Dense-8, Multi-Homotopy
  8. 理论验证指标 (R6, R8): ADT统计, epsilon_lin, Q-head rank correlation
  9. 代码-论文一致性修复 (R7): 训练配置统一

架构设计（修正后）:
  PTRM决策流程:
    1) TRM网络前向推理产生K个候选方向（潜在空间扰动提供多样性）
    2) PD基线修正确保基本稳定性（alpha_blend融合）
    3) Q-head粗筛top-M候选（廉价代理评估）
    4) Rollout代价精排top-M中选出最优（精确评估）
    5) 滞回正则化（轨迹空间级抗路径抖动）
    6) DT-CCBF安全投影

  vs MPPI:
    1) PD+高斯扰动产生K个候选
    2) Rollout代价评估所有K个候选
    3) 重要性加权平均

  PTRM优势:
    - Q-head预筛减少rollout计算量（K→M, M≪K）
    - 潜在空间扰动提供结构化多样性（vs 纯高斯随机）
    - TRM递归提供时间一致性

实验目录:
  Exp 1: K-Scaling × CBF Strength (TRM网络)
  Exp 2: σ-Scaling (TRM网络)
  Exp 3: Model Mismatch Robustness
  Exp 4: Process Noise Robustness
  Exp 5: Ablation (PD correction, noise channel, Q-head vs rollout)
  Exp 6: Runtime Comparison (所有控制器)
  Exp 7: Baseline Comparison (PTRM vs MPPI vs MLP+CBF vs CEM vs PD+CBF)
  Exp 8: Multi-Obstacle Configuration
  Exp 9: Theoretical Verification (ADT, epsilon_lin, Q-head correlation)
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
from quadrotor_core import (
    QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
    PTRMNMPCPredictor, generate_quadrotor_dataset,
    evaluate_batch_decoded_trajectory_cost, train_trm_jointly
)
from baselines import MPPIController, MLPController, MLPPredictor, train_mlp, CEMController

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

# 成功判定阈值
TErr_THRESH = 0.5

# K 值序列
K_VALUES = [1, 5, 10, 20, 50, 100]
# σ 值序列 (潜在空间量纲)
SIGMA_VALUES = [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]

# TRM 训练参数（与手稿统一：500轨迹/100 epochs+早停）
TRAIN_DATASET_SIZE = 500
TRAIN_EPOCHS = 100
TRAIN_PATIENCE = 20
TRAIN_LR = 0.001
TRAIN_BATCH_SIZE = 32

# ============================================================
# 多障碍物环境配置 (审稿修订 R5)
# ============================================================
OBSTACLE_CONFIGS = {
    'Corridor': [
        {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
        {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
        {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
    ],
    'Dense-5': [
        {"p": np.array([0.8, 1.0, 0.8]), "r": 0.4},
        {"p": np.array([1.5, 0.8, 1.5]), "r": 0.35},
        {"p": np.array([2.0, 1.5, 1.0]), "r": 0.4},
        {"p": np.array([1.2, 2.0, 2.0]), "r": 0.35},
        {"p": np.array([1.8, 2.5, 1.5]), "r": 0.3}
    ],
    'Dense-8': [
        {"p": np.array([0.6, 0.8, 0.7]), "r": 0.3},
        {"p": np.array([1.0, 1.5, 1.2]), "r": 0.35},
        {"p": np.array([1.4, 0.6, 1.8]), "r": 0.3},
        {"p": np.array([1.8, 1.8, 0.8]), "r": 0.35},
        {"p": np.array([2.2, 1.2, 1.5]), "r": 0.3},
        {"p": np.array([0.9, 2.3, 1.6]), "r": 0.25},
        {"p": np.array([1.6, 2.6, 2.2]), "r": 0.3},
        {"p": np.array([2.5, 2.0, 2.0]), "r": 0.25}
    ],
    'Multi-Homotopy': [
        {"p": np.array([1.0, 1.5, 1.5]), "r": 0.45},
        {"p": np.array([2.0, 1.5, 1.5]), "r": 0.45},
        {"p": np.array([1.5, 1.0, 0.5]), "r": 0.25},
        {"p": np.array([1.5, 2.0, 2.5]), "r": 0.25},
    ]
}

# 环境对应的目标点
ENV_TARGETS = {
    'Corridor': torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    'Dense-5': torch.tensor([2.5, 3.0, 2.5, 0.0, 0.0, 0.0], dtype=torch.float32),
    'Dense-8': torch.tensor([3.0, 3.0, 2.5, 0.0, 0.0, 0.0], dtype=torch.float32),
    'Multi-Homotopy': torch.tensor([2.5, 3.0, 1.5, 0.0, 0.0, 0.0], dtype=torch.float32),
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init(env_name='Corridor'):
    """生成随机初始条件"""
    if env_name == 'Dense-8':
        return torch.tensor([
            np.random.uniform(-0.5, 1.0),
            np.random.uniform(-1.0, 0.0),
            np.random.uniform(-0.5, 1.0),
            np.random.uniform(0.0, 0.5),
            np.random.uniform(0.0, 0.4),
            np.random.uniform(0.0, 0.5),
        ], dtype=torch.float32)
    else:
        return torch.tensor([
            np.random.uniform(-0.5, 1.5),
            np.random.uniform(-1.0, 0.0),
            np.random.uniform(-0.5, 1.5),
            np.random.uniform(0.0, 0.6),
            np.random.uniform(0.0, 0.4),
            np.random.uniform(0.0, 0.6),
        ], dtype=torch.float32)


# ============================================================
# 通用仿真试验函数
# ============================================================
def run_single_trial_trm(env, predictor, x_init, x_sp, n_steps=N_STEPS,
                          enable_cbf=True, use_mismatch=False, process_noise=0.0,
                          record_theory_metrics=False):
    """执行一次完整的 TRM-PTRM 闭环仿真试验"""
    predictor.reset()
    x = x_init.clone()
    collision = False
    min_dist = float('inf')
    iae = 0.0
    trajectory = [x[0:3].detach().numpy().copy()]
    cbf_interventions = 0

    theory_metrics = {
        'cbf_switches': [],
        'q_scores': [],
        'rollout_costs': [],
    }

    for step in range(n_steps):
        # 获取CBF干预前的名义控制
        if predictor.candidate_mode == 'pd':
            u_nominal_pre = predictor._compute_tracking_correction(x.cpu(), x_sp.cpu())
        else:
            u_nominal_pre = predictor._compute_tracking_correction(x.cpu(), x_sp.cpu()) if predictor.alpha_blend > 0 else torch.zeros(3)

        u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)

        if enable_cbf and torch.norm(u_safe - u_nominal_pre) > 0.01:
            cbf_interventions += 1
            theory_metrics['cbf_switches'].append(step)

        x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)

        p_np = x[0:3].detach().numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs['p']) - obs['r']
            min_dist = min(min_dist, d)
            if d < 0:
                collision = True

        iae += torch.norm(x[0:3] - x_sp[0:3]).item()
        trajectory.append(p_np.copy())

    # IAE归一化: IAE = (1/T) * Σ ||e|| * dt = Σ ||e|| / N
    iae = iae / n_steps

    terr = torch.norm(x[0:3] - x_sp[0:3]).item()
    success = (not collision) and (terr < TErr_THRESH)

    result = {
        'success': success,
        'collision': collision,
        'terminal_error': terr,
        'iae': iae,
        'min_distance': min_dist,
        'cbf_interventions': cbf_interventions,
        'trajectory': np.array(trajectory),
        'final_state': x.detach().numpy().copy(),
    }

    if record_theory_metrics:
        result['theory_metrics'] = theory_metrics

    return result


def run_single_trial_baseline(env, controller, x_init, x_sp, n_steps=N_STEPS,
                               enable_cbf=True, use_mismatch=False, process_noise=0.0):
    """执行一次基线控制器的闭环仿真试验"""
    controller.reset()
    x = x_init.clone()
    collision = False
    min_dist = float('inf')
    iae = 0.0
    trajectory = [x[0:3].detach().numpy().copy()]
    cbf_interventions = 0

    for step in range(n_steps):
        u_safe = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)

        if enable_cbf:
            u_nominal = controller._compute_pd_baseline(x, x_sp)
            if torch.norm(u_safe - u_nominal) > 0.01:
                cbf_interventions += 1

        x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)

        p_np = x[0:3].detach().numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs['p']) - obs['r']
            min_dist = min(min_dist, d)
            if d < 0:
                collision = True

        iae += torch.norm(x[0:3] - x_sp[0:3]).item()
        trajectory.append(p_np.copy())

    # IAE归一化
    iae = iae / n_steps

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


def run_single_trial_mlp(env, predictor, x_init, x_sp, n_steps=N_STEPS,
                          enable_cbf=True, use_mismatch=False, process_noise=0.0):
    """执行一次MLP控制器的闭环仿真试验"""
    predictor.reset()
    x = x_init.clone()
    collision = False
    min_dist = float('inf')
    iae = 0.0
    trajectory = [x[0:3].detach().numpy().copy()]
    cbf_interventions = 0

    for step in range(n_steps):
        u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
        u_nominal = predictor.last_u if predictor.last_u is not None else u_safe

        if enable_cbf and torch.norm(u_safe - u_nominal) > 0.01:
            cbf_interventions += 1

        x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)

        p_np = x[0:3].detach().numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs['p']) - obs['r']
            min_dist = min(min_dist, d)
            if d < 0:
                collision = True

        iae += torch.norm(x[0:3] - x_sp[0:3]).item()
        trajectory.append(p_np.copy())

    # IAE归一化
    iae = iae / n_steps

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
def run_mc_trials_trm(env, predictor, x_sp, n_mc=N_MC, env_name='Corridor',
                       record_theory=False, **kwargs):
    results = []
    for _ in range(n_mc):
        x_init = random_x_init(env_name)
        result = run_single_trial_trm(env, predictor, x_init, x_sp,
                                       record_theory_metrics=record_theory, **kwargs)
        results.append(result)
    return _aggregate_results(results, n_mc)


def run_mc_trials_baseline(env, controller, x_sp, n_mc=N_MC, env_name='Corridor', **kwargs):
    results = []
    for _ in range(n_mc):
        x_init = random_x_init(env_name)
        result = run_single_trial_baseline(env, controller, x_init, x_sp, **kwargs)
        results.append(result)
    return _aggregate_results(results, n_mc)


def run_mc_trials_mlp(env, predictor, x_sp, n_mc=N_MC, env_name='Corridor', **kwargs):
    results = []
    for _ in range(n_mc):
        x_init = random_x_init(env_name)
        result = run_single_trial_mlp(env, predictor, x_init, x_sp, **kwargs)
        results.append(result)
    return _aggregate_results(results, n_mc)


def _aggregate_results(results, n_mc):
    """汇总统计"""
    successes = [r['success'] for r in results]
    collisions = [r['collision'] for r in results]
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    min_dists = [r['min_distance'] for r in results]
    cbf_ints = [r['cbf_interventions'] for r in results]

    success_iaes = [r['iae'] for r in results if r['success']]
    collision_iaes = [r['iae'] for r in results if r['collision']]

    agg = {
        'success_rate': np.mean(successes) * 100,
        'collision_rate': np.mean(collisions) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'iae_success_mean': np.mean(success_iaes) if success_iaes else 0.0,
        'iae_collision_mean': np.mean(collision_iaes) if collision_iaes else 0.0,
        'min_distance_mean': np.mean(min_dists),
        'min_distance_std': np.std(min_dists),
        'cbf_interventions_mean': np.mean(cbf_ints),
        'n_mc': n_mc,
        'individual_results': results,
    }

    all_theory = [r.get('theory_metrics', {}) for r in results if 'theory_metrics' in r]
    if all_theory:
        agg['theory'] = {
            'cbf_switch_events': sum(len(t.get('cbf_switches', [])) for t in all_theory),
            'avg_dwell_time': _compute_avg_dwell_time(all_theory),
        }

    return agg


def _compute_avg_dwell_time(theory_metrics_list):
    """计算经验平均驻留时间"""
    all_intervals = []
    for tm in theory_metrics_list:
        switches = tm.get('cbf_switches', [])
        if len(switches) > 1:
            intervals = [switches[i+1] - switches[i] for i in range(len(switches)-1)]
            all_intervals.extend(intervals)
    if all_intervals:
        return np.mean(all_intervals) * DT
    return float('inf')


# ============================================================
# 训练流程
# ============================================================
def train_all_models(x_sp, save_dir):
    """训练TRM网络和MLP基线"""
    print("\n" + "=" * 80)
    print("阶段 1: 模型训练")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    env_train = QuadrotorDynamics()

    print(f"\n生成专家数据集 (size={TRAIN_DATASET_SIZE})...")
    solver = GoldenNMPCSolver(env_train, horizon=10)
    pos_range = [(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)]
    dataset = generate_quadrotor_dataset(env_train, solver, size=TRAIN_DATASET_SIZE,
                                          x_sp=x_sp, pos_range=pos_range)

    print(f"\n训练 TRM 网络 (epochs={TRAIN_EPOCHS}, patience={TRAIN_PATIENCE})...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    n_params = sum(p.numel() for p in trm_model.parameters() if p.requires_grad)
    print(f"TRM 参数量: {n_params} (应为 27,935)")

    trm_model, trm_history = train_trm_jointly(
        trm_model, dataset, env_train,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        lr=TRAIN_LR,
        patience=TRAIN_PATIENCE,
        verbose=True
    )

    trm_path = os.path.join(save_dir, 'trm_model.pt')
    torch.save(trm_model.state_dict(), trm_path)
    print(f"TRM 模型已保存至: {trm_path}")

    print(f"\n训练 MLP 基线...")
    mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
    n_mlp_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"MLP 参数量: {n_mlp_params}")

    mlp_model = train_mlp(mlp_model, dataset, env_train,
                           epochs=TRAIN_EPOCHS,
                           batch_size=TRAIN_BATCH_SIZE,
                           lr=TRAIN_LR,
                           patience=TRAIN_PATIENCE,
                           verbose=True)

    mlp_path = os.path.join(save_dir, 'mlp_model.pt')
    torch.save(mlp_model.state_dict(), mlp_path)
    print(f"MLP 模型已保存至: {mlp_path}")

    return trm_model, mlp_model, dataset


def load_or_train_models(x_sp, save_dir):
    """加载已有模型或重新训练"""
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    mlp_path = os.path.join(save_dir, 'mlp_model.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(trm_path) and os.path.exists(mlp_path):
        print("检测到已保存的模型，加载中...")
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
        mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3).to(device)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        print("模型加载完成。")
        return trm_model, mlp_model, None
    else:
        return train_all_models(x_sp, save_dir)


# ============================================================
# Exp 1: K-Scaling × CBF Strength (TRM网络)
# ============================================================
def experiment_k_scaling(trm_model):
    """实验 1: Test-Time Compute Scaling × CBF 强度消融"""
    print("\n" + "=" * 80)
    print("实验 1: K-Scaling × CBF 强度消融 (PD候选 + TRM评估)")
    print("=" * 80)

    cbf_configs = {
        'NoCBF': {'alpha_d': 0.0, 'gamma_d': 0.0, 'enable_cbf': False},
        'WeakCBF': {'alpha_d': 0.3, 'gamma_d': 0.1, 'enable_cbf': True},
        'StrongCBF': {'alpha_d': 0.8, 'gamma_d': 0.2, 'enable_cbf': True},
    }

    all_results = {}
    for cbf_name, cbf_cfg in cbf_configs.items():
        print(f"\n--- {cbf_name} (α_d={cbf_cfg['alpha_d']}, γ_d={cbf_cfg['gamma_d']}) ---")
        env = QuadrotorDynamics()
        env.alpha_d = cbf_cfg['alpha_d']
        env.gamma_d = cbf_cfg['gamma_d']

        cbf_results = {}
        for k in K_VALUES:
            set_seed(SEED)
            sigma = 0.25 if k > 1 else 0.0
            # PD候选模式: PD+高斯扰动产生候选，TRM Q-head/rollout评估排序
            predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                           alpha_blend=0.3, noise_mode='both',
                                           candidate_mode='pd', pd_sigma=2.0,
                                           use_rollout_cost=True, rollout_top_m=10)
            result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=cbf_cfg['enable_cbf'])
            cbf_results[k] = result
            print(f"  K={k:3d}: Succ={result['success_rate']:.0f}%, "
                  f"Coll={result['collision_rate']:.0f}%, "
                  f"TErr={result['terminal_error_mean']:.3f}±{result['terminal_error_std']:.3f}m, "
                  f"IAE={result['iae_mean']:.1f}±{result['iae_std']:.1f}, "
                  f"d_min={result['min_distance_mean']:.3f}m")

        all_results[cbf_name] = cbf_results

    return all_results


# ============================================================
# Exp 2: σ-Scaling
# ============================================================
def experiment_sigma_scaling(trm_model):
    """实验 2: PD候选扰动强度 σ 消融"""
    print("\n" + "=" * 80)
    print("实验 2: σ-Scaling (K=50, NoCBF, PD候选 + TRM评估)")
    print("=" * 80)

    env = QuadrotorDynamics()
    results = {}

    for sigma in SIGMA_VALUES:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=sigma,
                                       alpha_blend=0.3, noise_mode='both',
                                       candidate_mode='pd', pd_sigma=2.0,
                                       use_rollout_cost=True)
        result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=False)
        results[sigma] = result
        print(f"  σ={sigma:.2f}: Succ={result['success_rate']:.0f}%, "
              f"Coll={result['collision_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.3f}m, "
              f"IAE={result['iae_mean']:.1f}")

    return results


# ============================================================
# Exp 3: Model Mismatch Robustness
# ============================================================
def experiment_mismatch(trm_model):
    """实验 3: 模型失配鲁棒性"""
    print("\n" + "=" * 80)
    print("实验 3: 模型失配鲁棒性 (PD候选 + TRM评估)")
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
            sigma = 0.25 if k > 1 else 0.0
            predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                           alpha_blend=0.3, candidate_mode='pd',
                                           pd_sigma=2.0, use_rollout_cost=True)
            result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True,
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
def experiment_noise_robustness(trm_model):
    """实验 4: 过程噪声鲁棒性"""
    print("\n" + "=" * 80)
    print("实验 4: 过程噪声鲁棒性 (TRM网络)")
    print("=" * 80)

    env = QuadrotorDynamics()
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]
    results = {}

    for noise in noise_levels:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True, process_noise=noise)
        results[noise] = result
        print(f"  noise={noise:.3f}: Succ={result['success_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.3f}m, "
              f"IAE={result['iae_mean']:.1f}")

    return results


# ============================================================
# Exp 5: Ablation Study
# ============================================================
def experiment_ablation(trm_model):
    """实验 5: 消融研究"""
    print("\n" + "=" * 80)
    print("实验 5: 消融研究")
    print("=" * 80)

    env = QuadrotorDynamics()
    all_results = {}

    # 5a: 候选模式消融 (PD候选 vs TRM候选，审稿修订 R1/R3)
    print("\n--- 5a: 候选模式消融 (K=50, Strong CBF) ---")
    mode_results = {}
    for cmode, label in [('pd', 'PD+TRM-Eval'), ('trm', 'TRM+PD-Corr')]:
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            set_seed(SEED)
            predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                           alpha_blend=alpha, noise_mode='both',
                                           candidate_mode=cmode,
                                           pd_sigma=2.0 if cmode == 'pd' else 0.0,
                                           use_rollout_cost=True)
            result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True)
            mode_results[f'{label}_a{alpha}'] = result
            print(f"  {label:14s} α={alpha:.1f}: Succ={result['success_rate']:.0f}%, "
                  f"IAE={result['iae_mean']:.1f}")
    all_results['candidate_mode_ablation'] = mode_results

    # 5b: PD修正消融 — TRM模式下 (审稿修订 R1)
    print("\n--- 5b: PD修正消融 — TRM候选模式 (K=50, σ=0.25, Strong CBF) ---")
    pd_results = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=alpha, noise_mode='both',
                                       candidate_mode='trm',
                                       use_rollout_cost=True)
        result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True)
        pd_results[alpha] = result
        print(f"  α_blend={alpha:.1f}: Succ={result['success_rate']:.0f}%, "
              f"IAE={result['iae_mean']:.1f}, "
              f"CBF_int={result['cbf_interventions_mean']:.1f}")
    all_results['pd_ablation'] = pd_results

    # 5c: 噪声通道消融 (审稿修订 R2) — PD候选模式
    print("\n--- 5c: 噪声通道消融 — PD候选模式 (K=50, σ=0.25, NoCBF) ---")
    noise_results = {}
    for mode in ['both', 'latent', 'output']:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, noise_mode=mode,
                                       candidate_mode='pd', pd_sigma=2.0,
                                       use_rollout_cost=True)
        result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=False)
        noise_results[mode] = result
        print(f"  noise_mode={mode:8s}: Succ={result['success_rate']:.0f}%, "
              f"Coll={result['collision_rate']:.0f}%, "
              f"IAE={result['iae_mean']:.1f}")
    all_results['noise_ablation'] = noise_results

    # 5d: Q-head vs Rollout 评估消融（审稿修订 S1: 隔离Q-head贡献）
    print("\n--- 5d: Q-head vs Rollout 评估消融 (K=50, PD候选, Strong CBF) ---")
    eval_results = {}
    for use_rollout, label in [(True, 'Q+Rollout'), (False, 'Q-only')]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, noise_mode='both',
                                       candidate_mode='pd', pd_sigma=2.0,
                                       use_rollout_cost=use_rollout, rollout_top_m=10)
        result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True)
        eval_results[label] = result
        print(f"  {label:10s}: Succ={result['success_rate']:.0f}%, "
              f"IAE={result['iae_mean']:.1f}")
    all_results['eval_ablation'] = eval_results

    # 5e: 噪声通道 × K-Scaling 消融 — PD候选模式
    print("\n--- 5e: 噪声通道 × K-Scaling 消融 — PD候选 (NoCBF) ---")
    noise_k_results = {}
    for mode in ['both', 'latent', 'output']:
        mode_results = {}
        for k in [1, 10, 50, 100]:
            set_seed(SEED)
            sigma = 0.25 if k > 1 else 0.0
            predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                           alpha_blend=0.3, noise_mode=mode,
                                           candidate_mode='pd', pd_sigma=2.0,
                                           use_rollout_cost=True)
            result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=False)
            mode_results[k] = result
            print(f"  [{mode:8s}] K={k:3d}: Succ={result['success_rate']:.0f}%")
        noise_k_results[mode] = mode_results
    all_results['noise_k_ablation'] = noise_k_results

    return all_results


# ============================================================
# Exp 6: Runtime Comparison
# ============================================================
def experiment_runtime(trm_model, mlp_model):
    """实验 6: 运行时间比较"""
    print("\n" + "=" * 80)
    print("实验 6: 运行时间比较")
    print("=" * 80)

    env = QuadrotorDynamics()
    x_test = torch.tensor([0.0, -0.5, 0.0, 0.3, 0.2, 0.3], dtype=torch.float32)

    results = {}

    # TRM 各K值 (PD候选模式)
    print("\n--- PTRM (PD候选 + TRM评估) ---")
    trm_results = {}
    for k in K_VALUES:
        sigma = 0.25 if k > 1 else 0.0
        predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                       candidate_mode='pd', pd_sigma=2.0,
                                       use_rollout_cost=True)
        for _ in range(10):
            predictor.predict_action(x_test, X_SP, enable_cbf=True)
        times_cbf = []
        times_nocbf = []
        for _ in range(50):
            t0 = time.perf_counter()
            predictor.predict_action(x_test, X_SP, enable_cbf=True)
            times_cbf.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            predictor.predict_action(x_test, X_SP, enable_cbf=False)
            times_nocbf.append((time.perf_counter() - t0) * 1000)
        trm_results[k] = {
            'with_cbf_ms': np.median(times_cbf),
            'without_cbf_ms': np.median(times_nocbf)
        }
        print(f"  TRM K={k:3d}: CBF={np.median(times_cbf):.2f}ms, NoCBF={np.median(times_nocbf):.2f}ms")
    results['trm'] = trm_results

    # MPPI 各K值
    print("\n--- MPPI ---")
    mppi_results = {}
    for k in K_VALUES:
        sigma_mppi = 2.0 if k > 1 else 0.0
        mppi = MPPIController(env, K=k, sigma=sigma_mppi)
        t_ms = mppi.get_runtime_ms(x_test, X_SP, enable_cbf=True)
        t_ms_nc = mppi.get_runtime_ms(x_test, X_SP, enable_cbf=False)
        mppi_results[k] = {'with_cbf_ms': t_ms, 'without_cbf_ms': t_ms_nc}
        print(f"  MPPI K={k:3d}: CBF={t_ms:.2f}ms, NoCBF={t_ms_nc:.2f}ms")
    results['mppi'] = mppi_results

    # CEM
    print("\n--- CEM ---")
    cem_results = {}
    for n_iter in [1, 3, 5]:
        cem = CEMController(env, K=50, n_iter=n_iter, sigma=2.0)
        t_ms = cem.get_runtime_ms(x_test, X_SP, enable_cbf=True)
        cem_results[n_iter] = {'with_cbf_ms': t_ms, 'total_samples': 50 * n_iter}
        print(f"  CEM iter={n_iter}: CBF={t_ms:.2f}ms (total={50*n_iter} samples)")
    results['cem'] = cem_results

    # MLP
    print("\n--- MLP ---")
    mlp_pred = MLPPredictor(mlp_model, env, alpha_blend=0.3)
    t_mlp = mlp_pred.get_runtime_ms(x_test, X_SP, enable_cbf=True)
    results['mlp'] = {'with_cbf_ms': t_mlp}
    print(f"  MLP: CBF={t_mlp:.2f}ms")

    return results


# ============================================================
# Exp 7: Baseline Comparison
# ============================================================
def experiment_baselines(trm_model, mlp_model):
    """实验 7: 基线对比"""
    print("\n" + "=" * 80)
    print("实验 7: 基线对比")
    print("=" * 80)

    env = QuadrotorDynamics()
    all_results = {}

    print("\n--- 基线对比 × K-Scaling (Strong CBF) ---")
    methods = {}

    # PTRM (PD候选 + TRM评估)
    ptrm_results = {}
    for k in K_VALUES:
        set_seed(SEED)
        sigma = 0.25 if k > 1 else 0.0
        predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True)
        result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True)
        ptrm_results[k] = result
    methods['PTRM'] = ptrm_results

    # MPPI
    mppi_results = {}
    for k in K_VALUES:
        set_seed(SEED)
        sigma_mppi = 2.0 if k > 1 else 0.0
        mppi = MPPIController(env, K=k, sigma=sigma_mppi)
        result = run_mc_trials_baseline(env, mppi, X_SP, enable_cbf=True)
        mppi_results[k] = result
    methods['MPPI'] = mppi_results

    # CEM
    cem_results = {}
    for n_iter in [1, 3, 5]:
        set_seed(SEED)
        cem = CEMController(env, K=50, n_iter=n_iter, sigma=2.0)
        result = run_mc_trials_baseline(env, cem, X_SP, enable_cbf=True)
        cem_results[f'iter={n_iter}'] = result
    methods['CEM'] = cem_results

    # MLP+CBF
    set_seed(SEED)
    mlp_pred = MLPPredictor(mlp_model, env, alpha_blend=0.3)
    mlp_result = run_mc_trials_mlp(env, mlp_pred, X_SP, enable_cbf=True)
    methods['MLP+CBF'] = {'K=1': mlp_result}

    # PD+CBF
    set_seed(SEED)
    pd_mppi = MPPIController(env, K=1, sigma=0.0)
    pd_result = run_mc_trials_baseline(env, pd_mppi, X_SP, enable_cbf=True)
    methods['PD+CBF'] = {'K=1': pd_result}

    # 打印对比表
    print("\n--- 基线对比汇总 (Strong CBF) ---")
    print(f"{'Method':<12} {'K':<5} {'Succ%':<8} {'IAE':<10} {'TErr':<10} {'d_min':<8}")
    print("-" * 55)

    for method_name, method_data in methods.items():
        for k_key, data in method_data.items():
            print(f"{method_name:<12} {k_key:<5} {data['success_rate']:<8.0f} "
                  f"{data['iae_mean']:<10.1f} {data['terminal_error_mean']:<10.3f} "
                  f"{data['min_distance_mean']:<8.3f}")

    all_results['methods'] = methods
    return all_results


# ============================================================
# Exp 8: Multi-Obstacle Configuration
# ============================================================
def experiment_multi_obstacle(trm_model, mlp_model):
    """实验 8: 多障碍物配置"""
    print("\n" + "=" * 80)
    print("实验 8: 多障碍物配置实验")
    print("=" * 80)

    all_results = {}

    for env_name, obstacles in OBSTACLE_CONFIGS.items():
        print(f"\n--- 环境: {env_name} ({len(obstacles)}个障碍物) ---")
        env = QuadrotorDynamics(obstacles=obstacles)
        x_sp = ENV_TARGETS[env_name]

        env_results = {}

        # PTRM K-Scaling
        ptrm_env_results = {}
        for k in [1, 10, 50, 100]:
            set_seed(SEED)
            sigma = 0.25 if k > 1 else 0.0
            predictor = PTRMNMPCPredictor(trm_model, env, K=k, D=16, sigma=sigma,
                                           alpha_blend=0.3, candidate_mode='pd',
                                           pd_sigma=2.0, use_rollout_cost=True)
            result = run_mc_trials_trm(env, predictor, x_sp, env_name=env_name, enable_cbf=True)
            ptrm_env_results[k] = result
            print(f"  PTRM K={k:3d}: Succ={result['success_rate']:.0f}%, "
                  f"IAE={result['iae_mean']:.1f}")
        env_results['ptrm'] = ptrm_env_results

        # MPPI K=50
        set_seed(SEED)
        mppi = MPPIController(env, K=50, sigma=2.0)
        mppi_result = run_mc_trials_baseline(env, mppi, x_sp, env_name=env_name, enable_cbf=True)
        env_results['mppi_k50'] = mppi_result
        print(f"  MPPI K=50: Succ={mppi_result['success_rate']:.0f}%, "
              f"IAE={mppi_result['iae_mean']:.1f}")

        # MLP+CBF
        set_seed(SEED)
        mlp_pred = MLPPredictor(mlp_model, env, alpha_blend=0.3)
        mlp_result = run_mc_trials_mlp(env, mlp_pred, x_sp, env_name=env_name, enable_cbf=True)
        env_results['mlp'] = mlp_result
        print(f"  MLP+CBF:  Succ={mlp_result['success_rate']:.0f}%, "
              f"IAE={mlp_result['iae_mean']:.1f}")

        # PD+CBF
        set_seed(SEED)
        pd_mppi = MPPIController(env, K=1, sigma=0.0)
        pd_result = run_mc_trials_baseline(env, pd_mppi, x_sp, env_name=env_name, enable_cbf=True)
        env_results['pd_cbf'] = pd_result
        print(f"  PD+CBF:   Succ={pd_result['success_rate']:.0f}%, "
              f"IAE={pd_result['iae_mean']:.1f}")

        all_results[env_name] = env_results

    return all_results


# ============================================================
# Exp 9: Theoretical Verification
# ============================================================
def experiment_theory(trm_model):
    """实验 9: 理论验证"""
    print("\n" + "=" * 80)
    print("实验 9: 理论验证")
    print("=" * 80)

    env = QuadrotorDynamics()
    results = {}

    # 9a: ADT验证
    print("\n--- 9a: ADT (平均驻留时间) 验证 ---")
    set_seed(SEED)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    result = run_mc_trials_trm(env, predictor, X_SP, enable_cbf=True,
                                record_theory=True)
    if 'theory' in result:
        tau_a = result['theory']['avg_dwell_time']
        tau_a_star = 1.0
        print(f"  经验平均驻留时间 τ_a = {tau_a:.4f}s")
        print(f"  理论下界 τ_a* ≈ {tau_a_star:.4f}s")
        print(f"  τ_a > τ_a*: {'满足' if tau_a > tau_a_star else '不满足'}")
        results['adt'] = {
            'tau_a_empirical': tau_a,
            'tau_a_star': tau_a_star,
            'satisfied': tau_a > tau_a_star,
            'n_switch_events': result['theory']['cbf_switch_events']
        }

    # 9b: CBF线性化残差
    print("\n--- 9b: CBF 线性化残差 ε_lin 量化 ---")
    epsilon_lin_vals = _quantify_epsilon_lin(env, n_samples=1000)
    print(f"  ε_lin 最大值: {epsilon_lin_vals['max']:.6f}")
    print(f"  ε_lin 均值:   {epsilon_lin_vals['mean']:.6f}")
    print(f"  ε_lin 95%分位: {epsilon_lin_vals['p95']:.6f}")
    results['epsilon_lin'] = epsilon_lin_vals

    # 9c: Q-head 排序相关性
    print("\n--- 9c: Q-head 排序相关性 (Spearman ρ) ---")
    q_corr = _compute_qhead_rank_correlation(trm_model, env, n_samples=200)
    print(f"  Spearman ρ = {q_corr['spearman_rho']:.4f}")
    print(f"  Pearson r  = {q_corr['pearson_r']:.4f}")
    results['qhead_correlation'] = q_corr

    return results


def _quantify_epsilon_lin(env, n_samples=1000):
    """量化 CBF 线性化残差"""
    residuals = []

    for _ in range(n_samples):
        p = np.random.uniform(-1, 3, 3)
        v = np.random.uniform(-1, 1, 3)
        x = torch.tensor(np.concatenate([p, v]), dtype=torch.float32)
        u_nominal = torch.tensor(np.random.uniform(-10, 10, 3), dtype=torch.float32)

        u_safe = env.apply_cbf_projection(x, u_nominal)

        p_val = x[0:3].detach().cpu().numpy()
        v_val = x[3:6].detach().cpu().numpy()
        p_k1 = p_val + env.dt * v_val

        v_k1_drift = v_val + env.dt * (-(env.b_drag / env.m) * v_val)
        p_k2_drift = p_k1 + env.dt * v_k1_drift

        u_np = u_safe.numpy()
        v_k1_ctrl = v_val + env.dt * (u_np / env.m - (env.b_drag / env.m) * v_val)
        p_k2_ctrl = p_k1 + env.dt * v_k1_ctrl

        for obs in env.obstacles:
            r_safe = obs["r"] + env.delta_buffer
            h_k1_val = np.dot(p_k1 - obs["p"], p_k1 - obs["p"]) - r_safe**2
            h_k2_drift = np.dot(p_k2_drift - obs["p"], p_k2_drift - obs["p"]) - r_safe**2
            h_k2_ctrl = np.dot(p_k2_ctrl - obs["p"], p_k2_ctrl - obs["p"]) - r_safe**2

            B_drift = h_k2_drift - (1 - env.alpha_d) * h_k1_val
            B_ctrl = h_k2_ctrl - (1 - env.alpha_d) * h_k1_val

            dp_du = (env.dt**2) / env.m
            dB_du = -2.0 * dp_du * (p_k2_drift - obs["p"])
            B_linearized = B_drift + np.dot(dB_du, u_np)

            residual = abs(B_ctrl - B_linearized)
            residuals.append(residual)

    residuals = np.array(residuals)
    return {
        'max': float(np.max(residuals)),
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'p95': float(np.percentile(residuals, 95)),
        'p99': float(np.percentile(residuals, 99)),
    }


def _compute_qhead_rank_correlation(trm_model, env, n_samples=200):
    """计算 Q-head 分数与真实 rollout 代价的排序相关性"""
    from scipy.stats import spearmanr, pearsonr

    trm_model.eval()
    device = next(trm_model.parameters()).device

    q_scores_all = []
    rollout_costs_all = []

    x_sp = X_SP.to(device)

    for _ in range(n_samples):
        x_init = random_x_init().to(device)

        with torch.no_grad():
            X = torch.cat([x_init, x_sp]).unsqueeze(0).repeat(50, 1)
            y_history = trm_model.forward_steps(X, D=16, noise_scale=0.25)

        u_candidates, final_latent_y = y_history[-1]
        q_scores = trm_model.f_Q(final_latent_y).squeeze(-1).detach().cpu().numpy()

        rollout_costs = []
        for k_idx in range(50):
            u_k = u_candidates[k_idx].cpu()
            cost = _single_rollout_cost(env, x_init.cpu(), u_k, X_SP)
            rollout_costs.append(cost)

        q_scores_all.extend(q_scores.tolist())
        rollout_costs_all.extend(rollout_costs)

    q_arr = np.array(q_scores_all)
    c_arr = np.array(rollout_costs_all)

    spearman_rho, sp_p = spearmanr(q_arr, c_arr)
    pearson_r, pe_p = pearsonr(q_arr, c_arr)

    return {
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(sp_p),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pe_p),
        'n_samples': n_samples * 50,
    }


def _single_rollout_cost(env, x_init, u_sequence, x_sp, horizon=10):
    """计算单条轨迹的NMPC代价"""
    x = x_init.clone()
    cost = 0.0
    q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
    steps = min(u_sequence.shape[0] // 3, horizon)

    for i in range(steps):
        u = torch.clamp(u_sequence[i*3:(i+1)*3], env.u_min, env.u_max)
        x = env.step_discrete(x, u)
        error = x - x_sp
        cost += torch.sum(q_diag * error * error).item() + 0.02 * torch.sum(u * u).item()

    return cost


# ============================================================
# 绘图函数
# ============================================================
def plot_k_scaling_v6(results, save_dir):
    """绘制 K-Scaling 实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    cbf_configs = ['NoCBF', 'WeakCBF', 'StrongCBF']
    cbf_labels = ['No CBF', 'Weak CBF ($\\alpha_d$=0.3)', 'Strong CBF ($\\alpha_d$=0.8)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (cbf_name, cbf_label, color) in enumerate(zip(cbf_configs, cbf_labels, colors)):
        ax = axes[idx]
        cbf_data = results[cbf_name]
        ks = list(cbf_data.keys())
        succ_rates = [cbf_data[k]['success_rate'] for k in ks]
        coll_rates = [cbf_data[k]['collision_rate'] for k in ks]
        terr_means = [cbf_data[k]['terminal_error_mean'] for k in ks]
        terr_stds = [cbf_data[k]['terminal_error_std'] for k in ks]

        ax2 = ax.twinx()
        l1, = ax.plot(ks, succ_rates, 'o-', color=color, linewidth=2, markersize=6, label='Success Rate')
        l2, = ax.plot(ks, coll_rates, 's--', color='#e67e22', linewidth=1.5, markersize=5, label='Collision Rate')
        l3, = ax2.plot(ks, terr_means, '^:', color='#9b59b6', linewidth=1.5, markersize=5, label='TErr (m)')
        ax2.fill_between(ks, [m-s for m,s in zip(terr_means, terr_stds)],
                         [m+s for m,s in zip(terr_means, terr_stds)], alpha=0.15, color='#9b59b6')

        ax.set_xlabel('K (Candidates)')
        ax.set_ylabel('Rate (%)')
        ax2.set_ylabel('Terminal Error (m)', color='#9b59b6')
        ax.set_title(cbf_label, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.grid(True, alpha=0.3)
        ax.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()], loc='center right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_safety.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_safety.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # IAE vs K
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cbf_name, cbf_label, color in zip(cbf_configs, cbf_labels, colors):
        cbf_data = results[cbf_name]
        ks = list(cbf_data.keys())
        iaes = [cbf_data[k]['iae_mean'] for k in ks]
        iae_stds = [cbf_data[k]['iae_std'] for k in ks]
        ax.plot(ks, iaes, 'o-', color=color, linewidth=2, markersize=6, label=cbf_label)
        ax.fill_between(ks, [m-s for m,s in zip(iaes, iae_stds)],
                        [m+s for m,s in zip(iaes, iae_stds)], alpha=0.15, color=color)
    ax.set_xlabel('K (Candidates)')
    ax.set_ylabel('IAE')
    ax.set_title('Tracking Performance vs Test-Time Compute (TRM)', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_iae.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_k_scaling_iae.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_baselines_v6(results, save_dir):
    """绘制基线对比图"""
    os.makedirs(save_dir, exist_ok=True)

    methods = results['methods']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    if 'PTRM' in methods:
        ptrm = methods['PTRM']
        ks = sorted([k for k in ptrm.keys() if isinstance(k, int)])
        ax1.plot(ks, [ptrm[k]['success_rate'] for k in ks], 'o-', color='#2ecc71', linewidth=2, label='PTRM')
        ax2.plot(ks, [ptrm[k]['iae_mean'] for k in ks], 'o-', color='#2ecc71', linewidth=2, label='PTRM')

    if 'MPPI' in methods:
        mppi = methods['MPPI']
        ks = sorted([k for k in mppi.keys() if isinstance(k, int)])
        ax1.plot(ks, [mppi[k]['success_rate'] for k in ks], 's--', color='#3498db', linewidth=2, label='MPPI')
        ax2.plot(ks, [mppi[k]['iae_mean'] for k in ks], 's--', color='#3498db', linewidth=2, label='MPPI')

    if 'MLP+CBF' in methods:
        mlp_data = list(methods['MLP+CBF'].values())[0]
        ax1.axhline(mlp_data['success_rate'], color='#e67e22', linestyle=':', linewidth=1.5, label='MLP+CBF')
        ax2.axhline(mlp_data['iae_mean'], color='#e67e22', linestyle=':', linewidth=1.5, label='MLP+CBF')

    if 'PD+CBF' in methods:
        pd_data = list(methods['PD+CBF'].values())[0]
        ax1.axhline(pd_data['success_rate'], color='#e74c3c', linestyle='-.', linewidth=1.5, label='PD+CBF')
        ax2.axhline(pd_data['iae_mean'], color='#e74c3c', linestyle='-.', linewidth=1.5, label='PD+CBF')

    ax1.set_xlabel('K (Candidates)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Safety Performance', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    ax2.set_xlabel('K (Candidates)')
    ax2.set_ylabel('IAE')
    ax2.set_title('Tracking Performance', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_baselines.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_baselines.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_v6(results, save_dir):
    """绘制消融实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    pd_data = results.get('pd_ablation', {})
    if pd_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        alphas = sorted(pd_data.keys())
        ax1.plot(alphas, [pd_data[a]['success_rate'] for a in alphas], 'o-', color='#2ecc71', linewidth=2)
        ax1.set_xlabel('$\\alpha_{blend}$')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('PD Correction Ablation: Safety', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(alphas, [pd_data[a]['iae_mean'] for a in alphas], 'o-', color='#3498db', linewidth=2)
        ax2.set_xlabel('$\\alpha_{blend}$')
        ax2.set_ylabel('IAE')
        ax2.set_title('PD Correction Ablation: Tracking', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig_ablation_pd.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    noise_data = results.get('noise_ablation', {})
    if noise_data:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        modes = list(noise_data.keys())
        succ = [noise_data[m]['success_rate'] for m in modes]
        ax.bar(range(len(modes)), succ, color=['#2ecc71', '#3498db', '#e67e22'])
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Noise Channel Ablation (K=50, NoCBF)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig_ablation_noise.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    noise_k_data = results.get('noise_k_ablation', {})
    if noise_k_data:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = {'both': '#2ecc71', 'latent': '#3498db', 'output': '#e67e22'}
        for mode in ['both', 'latent', 'output']:
            if mode in noise_k_data:
                ks = sorted(noise_k_data[mode].keys())
                succ = [noise_k_data[mode][k]['success_rate'] for k in ks]
                ax.plot(ks, succ, 'o-', color=colors[mode], linewidth=2, label=mode)
        ax.set_xlabel('K (Candidates)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Noise Channel × K-Scaling (NoCBF)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig_ablation_noise_k.pdf'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_multi_obstacle_v6(results, save_dir):
    """绘制多障碍物配置实验结果"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    env_names = list(results.keys())
    x = np.arange(len(env_names))
    width = 0.2

    methods_to_plot = [('ptrm', 'PTRM K=50', 50), ('mppi_k50', 'MPPI K=50', None),
                       ('mlp', 'MLP+CBF', None), ('pd_cbf', 'PD+CBF', None)]

    for i, (key, label, k_val) in enumerate(methods_to_plot):
        succ_vals = []
        iae_vals = []
        for env_name in env_names:
            if key == 'ptrm' and 'ptrm' in results[env_name]:
                data = results[env_name]['ptrm'].get(k_val, {})
            elif key in results[env_name]:
                data = results[env_name][key]
            else:
                data = {}
            succ_vals.append(data.get('success_rate', 0))
            iae_vals.append(data.get('iae_mean', 0))

        axes[0].bar(x + i*width, succ_vals, width, label=label)
        axes[1].bar(x + i*width, iae_vals, width, label=label)

    for ax_idx, (ax, ylabel, title) in enumerate(zip(
        axes,
        ['Success Rate (%)', 'IAE'],
        ['Safety Across Environments', 'Tracking Across Environments']
    )):
        ax.set_xlabel('Environment')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(env_names, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig_multi_obstacle.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig_multi_obstacle.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 主函数
# ============================================================
def main():
    set_seed(SEED)

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    os.makedirs(save_dir, exist_ok=True)

    t_start = time.time()

    # 阶段 1: 训练模型
    trm_model, mlp_model, dataset = load_or_train_models(X_SP, save_dir)

    # 阶段 2: 执行所有实验
    print("\n" + "=" * 80)
    print("阶段 2: Monte Carlo 实验")
    print("=" * 80)

    exp1 = experiment_k_scaling(trm_model)
    exp2 = experiment_sigma_scaling(trm_model)
    exp3 = experiment_mismatch(trm_model)
    exp4 = experiment_noise_robustness(trm_model)
    exp5 = experiment_ablation(trm_model)
    exp6 = experiment_runtime(trm_model, mlp_model)
    exp7 = experiment_baselines(trm_model, mlp_model)
    exp8 = experiment_multi_obstacle(trm_model, mlp_model)
    exp9 = experiment_theory(trm_model)

    t_total = time.time() - t_start
    print(f"\n总实验时间: {t_total:.1f}s ({t_total/60:.1f}min)")

    # 保存原始数据
    serializable = {}
    for name, data in [('exp1', exp1), ('exp2', exp2), ('exp3', exp3),
                        ('exp4', exp4), ('exp5', exp5), ('exp6', exp6),
                        ('exp7', exp7), ('exp8', exp8), ('exp9', exp9)]:
        def strip_trajectories(obj):
            if isinstance(obj, dict):
                return {k: strip_trajectories(v) for k, v in obj.items()
                        if k not in ('individual_results', 'trajectory', 'theory_metrics')}
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        serializable[name] = strip_trajectories(data)

    with open(os.path.join(save_dir, 'raw_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    # 绘图
    plot_k_scaling_v6(exp1, save_dir)
    plot_ablation_v6(exp5, save_dir)
    plot_baselines_v6(exp7, save_dir)
    plot_multi_obstacle_v6(exp8, save_dir)

    print(f"\n所有结果已保存到 {save_dir}/")
    print("v6 实验完成!")


if __name__ == '__main__':
    main()
