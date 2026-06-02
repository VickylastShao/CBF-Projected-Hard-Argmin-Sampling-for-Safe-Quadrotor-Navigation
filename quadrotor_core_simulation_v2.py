# -*- coding: utf-8 -*-
"""
PTRM-NMPC 改进实验：修正实验设计以公平展示PTRM的附加价值

核心修改：
1. 实验一：Det TRM 和 PTRM 均开启 CBF（公平比较），分离 PTRM 在 CBF 之上的贡献
2. 新增窄通道场景（5个障碍物，更紧凑），CBF 独力不足时 PTRM 的附加价值
3. 消融实验：部分配置关闭 CBF 以展示不同 K/σ/D 在"CBF 边界条件"下的差异
4. 统一 success 定义：碰撞 + 终端误差双重判定
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from quadrotor_core import (
    QuadrotorDynamics,
    GoldenNMPCSolver,
    TRMNMPC,
    PTRMNMPCPredictor,
    generate_quadrotor_dataset,
    evaluate_batch_decoded_trajectory_cost,
    train_trm_jointly,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class NarrowCorridorDynamics(QuadrotorDynamics):
    """窄通道场景：5个障碍物形成更紧凑的非凸通道，CBF独力难以通过"""

    def __init__(self, m=1.5, b_drag=0.1, dt=0.02):
        super().__init__(m=m, b_drag=b_drag, dt=dt)
        # 窄通道：更大的障碍物 + 更多数量，形成窄缝通道
        self.obstacles = [
            {"p": np.array([0.8, 0.8, 0.8]), "r": 0.55},   # 障碍1：增大
            {"p": np.array([1.5, 1.8, 1.5]), "r": 0.50},   # 障碍2：移近通道中心
            {"p": np.array([2.2, 1.2, 2.2]), "r": 0.50},   # 障碍3：新增
            {"p": np.array([1.2, 2.5, 1.0]), "r": 0.45},   # 障碍4：新增
            {"p": np.array([2.0, 2.0, 2.5]), "r": 0.45},   # 障碍5：新增
        ]


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def save_experiment_data(name, metrics, config=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(RESULTS_DIR, f"{name}_{timestamp}")
    if metrics is not None:
        np.save(f"{prefix}_metrics.npy", metrics)
        with open(f"{prefix}_metrics.json", 'w') as f:
            json.dump(make_serializable(metrics), f, indent=2)
    if config is not None:
        with open(f"{prefix}_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
    print(f"实验数据已保存至: {prefix}_*")
    return prefix


def run_single_trial(env, predictor, x_init, x_sp, sim_steps, enable_cbf=True,
                     use_mismatch=False, process_noise=0.008):
    """运行单次闭环仿真试验，返回详细指标"""
    predictor.reset()
    x_curr = x_init.clone()
    p_iae, v_iae = 0.0, 0.0
    collision_flag = False
    cbf_intervention_count = 0
    min_obstacle_dist = float('inf')
    trajectory = [x_curr[0:3].detach().cpu().numpy().copy()]

    for step in range(sim_steps):
        u, u_seq = predictor.predict_action(x_curr, x_sp, enable_cbf=enable_cbf)
        # 检测 CBF 是否介入
        u_clamp = torch.clamp(u_seq[0:3].cpu(), env.u_min, env.u_max)
        if torch.norm(u.cpu() - u_clamp) > 1e-4:
            cbf_intervention_count += 1
        x_curr = env.step_discrete(x_curr, u, use_mismatch=use_mismatch, process_noise=process_noise)
        trajectory.append(x_curr[0:3].detach().cpu().numpy().copy())
        p_iae += np.linalg.norm(x_curr[0:3].detach().cpu().numpy() - x_sp[0:3].detach().cpu().numpy()) * env.dt
        v_iae += np.linalg.norm(x_curr[3:6].detach().cpu().numpy()) * env.dt
        p_np = x_curr[0:3].detach().cpu().numpy()
        if any(np.linalg.norm(p_np - obs["p"]) < obs["r"] for obs in env.obstacles):
            collision_flag = True
        # 记录最小障碍物距离
        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs["p"]) - obs["r"]
            min_obstacle_dist = min(min_obstacle_dist, d)

    terminal_err = np.linalg.norm(x_curr[0:3].detach().cpu().numpy() - x_sp[0:3].detach().cpu().numpy())
    success = (not collision_flag) and (terminal_err < 0.3)

    return {
        'p_iae': p_iae,
        'v_iae': v_iae,
        'collision': collision_flag,
        'success': success,
        'terminal_err': terminal_err,
        'cbf_interventions': cbf_intervention_count,
        'min_obstacle_dist': min_obstacle_dist,
        'trajectory': np.array(trajectory),
    }


def run_revised_experiments(env_narrow, env_wide, solver_narrow, solver_wide,
                            trm_model, num_trials=200, sim_steps=100):
    """运行修正后的核心实验"""

    x_sp = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    all_results = {}

    # ==================================================================
    # 实验一（修正）：3D 非凸窄通道避障 — 公平比较（所有方法均开启CBF）
    # ==================================================================
    print("\n" + "="*60)
    print("实验一（修正版）：非凸3D窄通道避障 — 所有方法均开启CBF")
    print("="*60)

    exp1_configs = [
        {'label': 'NMPC+CBF', 'predictor': None, 'solver': solver_narrow,
         'env': env_narrow, 'enable_cbf': True, 'use_mismatch': False},
        {'label': 'DetTRM+CBF(K=1)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=1, D=16, sigma=0.0),
         'solver': None, 'env': env_narrow, 'enable_cbf': True, 'use_mismatch': False},
        {'label': 'PTRM+CBF(K=10)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=10, D=16, sigma=0.25),
         'solver': None, 'env': env_narrow, 'enable_cbf': True, 'use_mismatch': False},
        {'label': 'PTRM+CBF(K=50)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=50, D=16, sigma=0.25),
         'solver': None, 'env': env_narrow, 'enable_cbf': True, 'use_mismatch': False},
        {'label': 'PTRM+CBF(K=100)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=100, D=16, sigma=0.25),
         'solver': None, 'env': env_narrow, 'enable_cbf': True, 'use_mismatch': False},
    ]

    exp1_results = {}
    for cfg in exp1_configs:
        label = cfg['label']
        is_nmpc = (cfg['predictor'] is None)
        n_trials_this = 50 if is_nmpc else num_trials  # NMPC 只跑 50 次（太慢）
        print(f"  运行 {label} ({n_trials_this} trials) ...")
        trial_results = []
        for trial in range(n_trials_this):
            init_px = 0.0 + np.random.normal(0, 0.02)
            init_py = 0.0 + np.random.normal(0, 0.02)
            init_pz = 0.0 + np.random.normal(0, 0.02)
            init_vx = 0.5 + np.random.normal(0, 0.01)
            init_vy = 0.5 + np.random.normal(0, 0.01)
            init_vz = 0.5 + np.random.normal(0, 0.01)
            x_init = torch.tensor([init_px, init_py, init_pz, init_vx, init_vy, init_vz], dtype=torch.float32)

            if is_nmpc:
                # NMPC baseline — 用一个简单的包装器适配接口
                class NMPCWrapper:
                    def __init__(self, solver, env):
                        self.solver = solver
                        self.env = env
                        self.last_u_seq = None
                    def reset(self):
                        self.last_u_seq = None
                    def predict_action(self, x, x_sp, enable_cbf=True):
                        return _nmpc_step(self.solver, self.env, x, x_sp, enable_cbf)
                pred_obj = NMPCWrapper(cfg['solver'], cfg['env'])
                result = run_single_trial(cfg['env'], pred_obj, x_init, x_sp, sim_steps,
                                         enable_cbf=cfg['enable_cbf'], use_mismatch=cfg['use_mismatch'])
            else:
                result = run_single_trial(cfg['env'], cfg['predictor'], x_init, x_sp, sim_steps,
                                         enable_cbf=cfg['enable_cbf'], use_mismatch=cfg['use_mismatch'])
            trial_results.append(result)
            if (trial + 1) % 50 == 0:
                print(f"    进度: {trial+1}/{n_trials_this}")

        # 汇总
        summary = {
            'success_rate': sum(r['success'] for r in trial_results) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trial_results) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trial_results]),
            'p_iae_std': np.std([r['p_iae'] for r in trial_results]),
            'v_iae_mean': np.mean([r['v_iae'] for r in trial_results]),
            'v_iae_std': np.std([r['v_iae'] for r in trial_results]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trial_results]),
            'terminal_err_std': np.std([r['terminal_err'] for r in trial_results]),
            'cbf_interventions_mean': np.mean([r['cbf_interventions'] for r in trial_results]),
            'min_obstacle_dist_mean': np.mean([r['min_obstacle_dist'] for r in trial_results]),
        }
        exp1_results[label] = summary
        print(f"    {label}: Success={summary['success_rate']:.1f}% | Collision={summary['collision_rate']:.1f}% | "
              f"Pos IAE={summary['p_iae_mean']:.2f}±{summary['p_iae_std']:.2f} | "
              f"CBF介入={summary['cbf_interventions_mean']:.1f}次 | "
              f"MinDist={summary['min_obstacle_dist_mean']:.3f}m")

    all_results['exp1_narrow_corridor'] = exp1_results

    # ==================================================================
    # 实验一b：宽通道场景（原始3障碍物），作为对比
    # ==================================================================
    print("\n" + "="*60)
    print("实验一b（对比）：原始宽通道 — 所有方法均开启CBF")
    print("="*60)

    exp1b_configs = [
        {'label': 'DetTRM+CBF(K=1)', 'predictor': PTRMNMPCPredictor(trm_model, env_wide, K=1, D=16, sigma=0.0),
         'env': env_wide, 'enable_cbf': True},
        {'label': 'PTRM+CBF(K=50)', 'predictor': PTRMNMPCPredictor(trm_model, env_wide, K=50, D=16, sigma=0.25),
         'env': env_wide, 'enable_cbf': True},
    ]

    exp1b_results = {}
    for cfg in exp1b_configs:
        label = cfg['label']
        print(f"  运行 {label} (宽通道) ...")
        trial_results = []
        for trial in range(num_trials):
            init_px = 0.0 + np.random.normal(0, 0.02)
            init_py = 0.0 + np.random.normal(0, 0.02)
            init_pz = 0.0 + np.random.normal(0, 0.02)
            init_vx = 0.5 + np.random.normal(0, 0.01)
            init_vy = 0.5 + np.random.normal(0, 0.01)
            init_vz = 0.5 + np.random.normal(0, 0.01)
            x_init = torch.tensor([init_px, init_py, init_pz, init_vx, init_vy, init_vz], dtype=torch.float32)
            result = run_single_trial(cfg['env'], cfg['predictor'], x_init, x_sp, sim_steps,
                                     enable_cbf=cfg['enable_cbf'])
            trial_results.append(result)

        summary = {
            'success_rate': sum(r['success'] for r in trial_results) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trial_results) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trial_results]),
            'p_iae_std': np.std([r['p_iae'] for r in trial_results]),
            'v_iae_mean': np.mean([r['v_iae'] for r in trial_results]),
            'v_iae_std': np.std([r['v_iae'] for r in trial_results]),
            'cbf_interventions_mean': np.mean([r['cbf_interventions'] for r in trial_results]),
        }
        exp1b_results[label] = summary
        print(f"    {label}: Success={summary['success_rate']:.1f}% | Collision={summary['collision_rate']:.1f}% | "
              f"Pos IAE={summary['p_iae_mean']:.2f}±{summary['p_iae_std']:.2f} | "
              f"CBF介入={summary['cbf_interventions_mean']:.1f}次")

    all_results['exp1b_wide_corridor'] = exp1b_results

    # ==================================================================
    # 实验二（修正）：参数失配下的鲁棒性 — 所有方法均开启CBF
    # ==================================================================
    print("\n" + "="*60)
    print("实验二（修正版）：+50%质量失配鲁棒性 — 所有方法均开启CBF")
    print("="*60)

    exp2_configs = [
        {'label': 'DetTRM+CBF(K=1)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=1, D=16, sigma=0.0),
         'env': env_narrow, 'enable_cbf': True, 'use_mismatch': True},
        {'label': 'PTRM+CBF(K=10)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=10, D=16, sigma=0.25),
         'env': env_narrow, 'enable_cbf': True, 'use_mismatch': True},
        {'label': 'PTRM+CBF(K=50)', 'predictor': PTRMNMPCPredictor(trm_model, env_narrow, K=50, D=16, sigma=0.25),
         'env': env_narrow, 'enable_cbf': True, 'use_mismatch': True},
    ]

    exp2_results = {}
    for cfg in exp2_configs:
        label = cfg['label']
        print(f"  运行 {label} (+50%质量失配) ...")
        trial_results = []
        for trial in range(num_trials):
            init_px = 0.0 + np.random.normal(0, 0.02)
            init_py = 0.0 + np.random.normal(0, 0.02)
            init_pz = 0.0 + np.random.normal(0, 0.02)
            init_vx = 0.5 + np.random.normal(0, 0.01)
            init_vy = 0.5 + np.random.normal(0, 0.01)
            init_vz = 0.5 + np.random.normal(0, 0.01)
            x_init = torch.tensor([init_px, init_py, init_pz, init_vx, init_vy, init_vz], dtype=torch.float32)
            result = run_single_trial(cfg['env'], cfg['predictor'], x_init, x_sp, sim_steps,
                                     enable_cbf=cfg['enable_cbf'], use_mismatch=cfg['use_mismatch'])
            trial_results.append(result)

        summary = {
            'success_rate': sum(r['success'] for r in trial_results) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trial_results) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trial_results]),
            'p_iae_std': np.std([r['p_iae'] for r in trial_results]),
            'v_iae_mean': np.mean([r['v_iae'] for r in trial_results]),
            'v_iae_std': np.std([r['v_iae'] for r in trial_results]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trial_results]),
            'terminal_err_std': np.std([r['terminal_err'] for r in trial_results]),
            'cbf_interventions_mean': np.mean([r['cbf_interventions'] for r in trial_results]),
        }
        exp2_results[label] = summary
        print(f"    {label}: Success={summary['success_rate']:.1f}% | Collision={summary['collision_rate']:.1f}% | "
              f"Pos IAE={summary['p_iae_mean']:.2f}±{summary['p_iae_std']:.2f} | "
              f"TerminalErr={summary['terminal_err_mean']:.3f}±{summary['terminal_err_std']:.3f}m | "
              f"CBF介入={summary['cbf_interventions_mean']:.1f}次")

    all_results['exp2_mismatch'] = exp2_results

    # ==================================================================
    # 实验三：计算效率与宽度扩展延迟（不变，但补充K=5, K=20）
    # ==================================================================
    print("\n" + "="*60)
    print("实验三：计算效率与宽度扩展延迟")
    print("="*60)

    widths_K = [1, 5, 10, 20, 50, 100]
    latencies = []
    cost_lowerings = []
    x_test = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=torch.float32)

    ref_predictor = PTRMNMPCPredictor(trm_model, env_narrow, K=1, D=16, sigma=0.0)
    ref_predictor.reset()
    ref_cost = 0.0
    x_curr_ref = x_test.clone()
    for step in range(sim_steps):
        u, _ = ref_predictor.predict_action(x_curr_ref, x_sp, enable_cbf=True)
        x_curr_ref = env_narrow.step_discrete(x_curr_ref, u, process_noise=0.008)
        ref_cost += np.linalg.norm(x_curr_ref[0:3].detach().cpu().numpy() - x_sp[0:3].detach().cpu().numpy())

    for k in widths_K:
        tester = PTRMNMPCPredictor(trm_model, env_narrow, K=k, D=16, sigma=0.2)
        start_t = time.time()
        for _ in range(50):
            _, _ = tester.predict_action(x_test, x_sp, enable_cbf=True)
        avg_latency = (time.time() - start_t) / 50.0 * 1000.0
        latencies.append(avg_latency)

        tester.reset()
        total_cost = 0.0
        x_curr_k = x_test.clone()
        for step in range(sim_steps):
            u, _ = tester.predict_action(x_curr_k, x_sp, enable_cbf=True)
            x_curr_k = env_narrow.step_discrete(x_curr_k, u, process_noise=0.008)
            total_cost += np.linalg.norm(x_curr_k[0:3].detach().cpu().numpy() - x_sp[0:3].detach().cpu().numpy())

        if ref_cost > 1e-6:
            cost_change_pct = ((ref_cost - total_cost) / ref_cost) * 100.0
        else:
            cost_change_pct = 0.0
        cost_lowerings.append(cost_change_pct)
        print(f"  K={k:3d} | Latency: {avg_latency:.3f} ms | Cost Change vs K=1: {cost_change_pct:+.1f}%")

    solver_times = []
    for _ in range(30):
        start_t = time.time()
        _ = solver_narrow.solve(x_test, x_sp)
        solver_times.append((time.time() - start_t) * 1000.0)
    expert_latency = np.mean(solver_times)
    print(f"  Expert NMPC | Latency: {expert_latency:.3f} ms")

    all_results['exp3_runtime'] = {
        'widths_K': widths_K,
        'latencies_ms': latencies,
        'cost_change_pct': cost_lowerings,
        'expert_latency_ms': expert_latency,
    }

    # ==================================================================
    # 消融实验（修正）：K/σ/D 消融，区分 CBF 开启和关闭
    # ==================================================================
    print("\n" + "="*60)
    print("消融实验（修正版）：K/σ/D 消融 — 窄通道场景")
    print("="*60)

    ablation_results = {}

    # A. K 消融 — 窄通道 + CBF
    print("\n--- K 消融 (窄通道 + CBF) ---")
    k_values = [1, 5, 10, 20, 50, 100]
    k_ablation = {}
    for k in k_values:
        pred = PTRMNMPCPredictor(trm_model, env_narrow, K=k, D=16, sigma=0.25)
        trials = []
        for trial in range(num_trials):
            x_init = torch.tensor([
                np.random.normal(0, 0.02), np.random.normal(0, 0.02), np.random.normal(0, 0.02),
                0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01)
            ], dtype=torch.float32)
            result = run_single_trial(env_narrow, pred, x_init, x_sp, sim_steps, enable_cbf=True)
            trials.append(result)
        s = {
            'success_rate': sum(r['success'] for r in trials) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trials) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trials]),
            'p_iae_std': np.std([r['p_iae'] for r in trials]),
            'v_iae_mean': np.mean([r['v_iae'] for r in trials]),
            'v_iae_std': np.std([r['v_iae'] for r in trials]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trials]),
            'min_obstacle_dist_mean': np.mean([r['min_obstacle_dist'] for r in trials]),
        }
        k_ablation[f'K={k}'] = s
        print(f"  K={k:3d}: Success={s['success_rate']:.1f}% | Collision={s['collision_rate']:.1f}% | "
              f"Pos IAE={s['p_iae_mean']:.2f}±{s['p_iae_std']:.2f} | "
              f"TerminalErr={s['terminal_err_mean']:.3f}m | MinDist={s['min_obstacle_dist_mean']:.3f}m")
    ablation_results['K_with_CBF'] = k_ablation

    # B. σ 消融 — 窄通道 + CBF
    print("\n--- σ 消融 (窄通道 + CBF) ---")
    sigma_values = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    sigma_ablation = {}
    for sig in sigma_values:
        pred = PTRMNMPCPredictor(trm_model, env_narrow, K=50, D=16, sigma=sig)
        trials = []
        for trial in range(num_trials):
            x_init = torch.tensor([
                np.random.normal(0, 0.02), np.random.normal(0, 0.02), np.random.normal(0, 0.02),
                0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01)
            ], dtype=torch.float32)
            result = run_single_trial(env_narrow, pred, x_init, x_sp, sim_steps, enable_cbf=True)
            trials.append(result)
        s = {
            'success_rate': sum(r['success'] for r in trials) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trials) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trials]),
            'p_iae_std': np.std([r['p_iae'] for r in trials]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trials]),
            'min_obstacle_dist_mean': np.mean([r['min_obstacle_dist'] for r in trials]),
        }
        sigma_ablation[f'σ={sig:.2f}'] = s
        print(f"  σ={sig:.2f}: Success={s['success_rate']:.1f}% | Collision={s['collision_rate']:.1f}% | "
              f"Pos IAE={s['p_iae_mean']:.2f}±{s['p_iae_std']:.2f} | MinDist={s['min_obstacle_dist_mean']:.3f}m")
    ablation_results['sigma_with_CBF'] = sigma_ablation

    # C. D 消融 — 窄通道 + CBF
    print("\n--- D 消融 (窄通道 + CBF) ---")
    d_values = [4, 8, 12, 16, 20, 24]
    d_ablation = {}
    for d in d_values:
        pred = PTRMNMPCPredictor(trm_model, env_narrow, K=50, D=d, sigma=0.25)
        trials = []
        for trial in range(num_trials):
            x_init = torch.tensor([
                np.random.normal(0, 0.02), np.random.normal(0, 0.02), np.random.normal(0, 0.02),
                0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01)
            ], dtype=torch.float32)
            result = run_single_trial(env_narrow, pred, x_init, x_sp, sim_steps, enable_cbf=True)
            trials.append(result)
        s = {
            'success_rate': sum(r['success'] for r in trials) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trials) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trials]),
            'p_iae_std': np.std([r['p_iae'] for r in trials]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trials]),
            'min_obstacle_dist_mean': np.mean([r['min_obstacle_dist'] for r in trials]),
        }
        d_ablation[f'D={d}'] = s
        print(f"  D={d:2d}: Success={s['success_rate']:.1f}% | Collision={s['collision_rate']:.1f}% | "
              f"Pos IAE={s['p_iae_mean']:.2f}±{s['p_iae_std']:.2f} | MinDist={s['min_obstacle_dist_mean']:.3f}m")
    ablation_results['D_with_CBF'] = d_ablation

    # D. CBF 消融 — 窄通道 + 无 CBF（展示 CBF 必要性）
    print("\n--- CBF 消融 (窄通道, 无CBF vs 有CBF) ---")
    cbf_ablation = {}
    for cbf_label, cbf_flag in [('CBF=ON', True), ('CBF=OFF', False)]:
        pred = PTRMNMPCPredictor(trm_model, env_narrow, K=50, D=16, sigma=0.25)
        trials = []
        for trial in range(num_trials):
            x_init = torch.tensor([
                np.random.normal(0, 0.02), np.random.normal(0, 0.02), np.random.normal(0, 0.02),
                0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01)
            ], dtype=torch.float32)
            result = run_single_trial(env_narrow, pred, x_init, x_sp, sim_steps, enable_cbf=cbf_flag)
            trials.append(result)
        s = {
            'success_rate': sum(r['success'] for r in trials) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trials) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trials]),
            'p_iae_std': np.std([r['p_iae'] for r in trials]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trials]),
            'min_obstacle_dist_mean': np.mean([r['min_obstacle_dist'] for r in trials]),
        }
        cbf_ablation[cbf_label] = s
        print(f"  {cbf_label}: Success={s['success_rate']:.1f}% | Collision={s['collision_rate']:.1f}% | "
              f"Pos IAE={s['p_iae_mean']:.2f}±{s['p_iae_std']:.2f} | MinDist={s['min_obstacle_dist_mean']:.3f}m")
    ablation_results['CBF_ablation'] = cbf_ablation

    # E. 滞回消融
    print("\n--- 滞回消融 (窄通道) ---")
    hyst_ablation = {}
    for eta_label, eta_val in [('η=0.00', 0.0), ('η=0.05', 0.05), ('η=0.10', 0.10), ('η=0.20', 0.20)]:
        pred = PTRMNMPCPredictor(trm_model, env_narrow, K=50, D=16, sigma=0.25, eta_hyst=eta_val)
        trials = []
        for trial in range(num_trials):
            x_init = torch.tensor([
                np.random.normal(0, 0.02), np.random.normal(0, 0.02), np.random.normal(0, 0.02),
                0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01), 0.5 + np.random.normal(0, 0.01)
            ], dtype=torch.float32)
            result = run_single_trial(env_narrow, pred, x_init, x_sp, sim_steps, enable_cbf=True)
            trials.append(result)
        s = {
            'success_rate': sum(r['success'] for r in trials) / num_trials * 100,
            'collision_rate': sum(r['collision'] for r in trials) / num_trials * 100,
            'p_iae_mean': np.mean([r['p_iae'] for r in trials]),
            'p_iae_std': np.std([r['p_iae'] for r in trials]),
            'terminal_err_mean': np.mean([r['terminal_err'] for r in trials]),
        }
        hyst_ablation[eta_label] = s
        print(f"  {eta_label}: Success={s['success_rate']:.1f}% | Collision={s['collision_rate']:.1f}% | "
              f"Pos IAE={s['p_iae_mean']:.2f}±{s['p_iae_std']:.2f} | TerminalErr={s['terminal_err_mean']:.3f}m")
    ablation_results['hysteresis'] = hyst_ablation

    all_results['ablation'] = ablation_results

    return all_results


def _nmpc_step(solver, env, x, x_sp, enable_cbf):
    """NMPC 单步辅助函数（适配 run_single_trial 接口）"""
    u_full = solver.solve(x, x_sp)
    u_nominal = u_full[0:3]
    if enable_cbf:
        u_safe = env.apply_cbf_projection(x, u_nominal)
    else:
        u_safe = torch.clamp(u_nominal, env.u_min, env.u_max)
    u_seq = torch.zeros(30, dtype=torch.float32)
    u_seq[0:3] = u_safe
    return u_safe, u_seq


def plot_revised_results(results, env_narrow, env_wide):
    """绘制修正后的实验结果"""

    # --- 图1: 窄通道实验对比 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Exp1 窄通道
    exp1 = results['exp1_narrow_corridor']
    methods = list(exp1.keys())
    success_rates = [exp1[m]['success_rate'] for m in methods]
    collision_rates = [exp1[m]['collision_rate'] for m in methods]

    ax = axes[0, 0]
    x_pos = np.arange(len(methods))
    width = 0.35
    ax.bar(x_pos - width/2, success_rates, width, label='Success Rate (%)', color='green', alpha=0.7)
    ax.bar(x_pos + width/2, collision_rates, width, label='Collision Rate (%)', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Exp I: Narrow Corridor (5 obstacles, CBF enabled for all)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Exp1 位置IAE
    ax = axes[0, 1]
    p_iae_means = [exp1[m]['p_iae_mean'] for m in methods]
    p_iae_stds = [exp1[m]['p_iae_std'] for m in methods]
    ax.bar(x_pos, p_iae_means, yerr=p_iae_stds, capsize=3, color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Position IAE (m·s)')
    ax.set_title('Exp I: Position Tracking Error (Narrow Corridor)')
    ax.grid(True, alpha=0.3)

    # Exp2 失配鲁棒性
    exp2 = results['exp2_mismatch']
    methods2 = list(exp2.keys())
    success_rates2 = [exp2[m]['success_rate'] for m in methods2]
    terminal_errs = [exp2[m]['terminal_err_mean'] for m in methods2]
    terminal_stds = [exp2[m]['terminal_err_std'] for m in methods2]

    ax = axes[1, 0]
    x_pos2 = np.arange(len(methods2))
    ax.bar(x_pos2, success_rates2, color='green', alpha=0.7)
    ax.set_xticks(x_pos2)
    ax.set_xticklabels(methods2, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Exp II: Robustness under +50% Mass Mismatch (CBF enabled)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x_pos2, terminal_errs, yerr=terminal_stds, capsize=3, color='coral', alpha=0.7)
    ax.set_xticks(x_pos2)
    ax.set_xticklabels(methods2, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Terminal Position Error (m)')
    ax.set_title('Exp II: Terminal Accuracy under Mismatch')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'ptrm_nmpc_revised_exp1_exp2.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig_path_pdf = os.path.join(RESULTS_DIR, 'ptrm_nmpc_revised_exp1_exp2.pdf')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"修正实验图已保存: {fig_path}")
    plt.close(fig)

    # --- 图2: Runtime + Ablation ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Runtime
    exp3 = results['exp3_runtime']
    ax = axes[0]
    ax.plot(exp3['widths_K'], exp3['latencies_ms'], 'b-o', linewidth=2, markersize=8, label='PTRM Step Latency')
    ax.axhline(y=exp3['expert_latency_ms'], color='red', linestyle='--', label='Expert NMPC Latency')
    ax.set_xlabel('Parallel Width K')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Exp III: Compute Latency vs. Width K')
    ax.legend()
    ax.grid(True)

    # K ablation
    abl_k = results['ablation']['K_with_CBF']
    ax = axes[1]
    k_labels = list(abl_k.keys())
    k_success = [abl_k[k]['success_rate'] for k in k_labels]
    k_p_iae = [abl_k[k]['p_iae_mean'] for k in k_labels]
    k_min_dist = [abl_k[k]['min_obstacle_dist_mean'] for k in k_labels]
    ax2 = ax.twinx()
    l1 = ax.bar(k_labels, k_success, color='green', alpha=0.6, label='Success Rate (%)')
    l2 = ax2.plot(k_labels, k_min_dist, 'r-s', linewidth=2, markersize=8, label='Min Obstacle Dist (m)')
    ax.set_ylabel('Success Rate (%)', color='green')
    ax2.set_ylabel('Min Obstacle Distance (m)', color='red')
    ax.set_title('Ablation: K (Narrow Corridor + CBF)')
    lines = [l1] + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right')
    ax.grid(True, alpha=0.3)

    # σ ablation
    abl_s = results['ablation']['sigma_with_CBF']
    ax = axes[2]
    s_labels = list(abl_s.keys())
    s_success = [abl_s[k]['success_rate'] for k in s_labels]
    s_p_iae = [abl_s[k]['p_iae_mean'] for k in s_labels]
    s_min_dist = [abl_s[k]['min_obstacle_dist_mean'] for k in s_labels]
    ax2 = ax.twinx()
    l1 = ax.bar(s_labels, s_success, color='green', alpha=0.6, label='Success Rate (%)')
    l2 = ax2.plot(s_labels, s_min_dist, 'r-s', linewidth=2, markersize=8, label='Min Obstacle Dist (m)')
    ax.set_ylabel('Success Rate (%)', color='green')
    ax2.set_ylabel('Min Obstacle Distance (m)', color='red')
    ax.set_title('Ablation: σ (Narrow Corridor + CBF)')
    ax.tick_params(axis='x', rotation=30)
    lines = [l1] + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'ptrm_nmpc_revised_runtime_ablation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig_path_pdf = os.path.join(RESULTS_DIR, 'ptrm_nmpc_revised_runtime_ablation.pdf')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"消融实验图已保存: {fig_path}")
    plt.close(fig)

    # --- 图3: CBF消融对比 ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cbf_abl = results['ablation']['CBF_ablation']
    cbf_labels = list(cbf_abl.keys())
    cbf_success = [cbf_abl[k]['success_rate'] for k in cbf_labels]
    cbf_collision = [cbf_abl[k]['collision_rate'] for k in cbf_labels]
    x_pos = np.arange(len(cbf_labels))
    ax.bar(x_pos - 0.15, cbf_success, 0.3, label='Success Rate (%)', color='green', alpha=0.7)
    ax.bar(x_pos + 0.15, cbf_collision, 0.3, label='Collision Rate (%)', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cbf_labels)
    ax.set_ylabel('Rate (%)')
    ax.set_title('CBF Ablation: Narrow Corridor (K=50, D=16, σ=0.25)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(RESULTS_DIR, 'ptrm_nmpc_revised_cbf_ablation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"CBF消融图已保存: {fig_path}")
    plt.close(fig)


def main():
    device = torch.device("cpu")
    print(f"设备: {device}")

    # 创建两个环境
    env_wide = QuadrotorDynamics()   # 原始3障碍物宽通道
    env_narrow = NarrowCorridorDynamics()  # 5障碍物窄通道

    solver_wide = GoldenNMPCSolver(env_wide, horizon=10)
    solver_narrow = GoldenNMPCSolver(env_narrow, horizon=10)

    # 训练模型：使用宽通道环境生成数据（模拟实际部署时训练与测试分布差异）
    print("\n训练 PTRM-NMPC 模型（宽通道数据集）...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    dataset = generate_quadrotor_dataset(env_wide, solver_wide, size=500)
    trm_model, history = train_trm_jointly(trm_model, dataset, env_wide, epochs=100, patience=15, verbose=True)

    # 运行修正后的实验
    results = run_revised_experiments(
        env_narrow, env_wide, solver_narrow, solver_wide,
        trm_model, num_trials=200, sim_steps=100
    )

    # 绘制结果
    plot_revised_results(results, env_narrow, env_wide)

    # 保存数据
    save_experiment_data("revised_experiments", results, config={
        'num_trials': 200,
        'sim_steps': 100,
        'seed': SEED,
        'narrow_corridor_obstacles': [
            {'p': o['p'].tolist(), 'r': float(o['r'])} for o in env_narrow.obstacles
        ],
        'wide_corridor_obstacles': [
            {'p': o['p'].tolist(), 'r': float(o['r'])} for o in env_wide.obstacles
        ],
    })

    print("\n" + "="*60)
    print("全部修正实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
