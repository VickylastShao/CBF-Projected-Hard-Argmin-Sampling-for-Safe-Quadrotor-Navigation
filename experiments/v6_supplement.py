# -*- coding: utf-8 -*-
"""
v6 补充实验 — 收集审稿修订所需的数据:
  - S5: 分别报告成功/碰撞试验的IAE
  - S8: 失败案例分析
  - S9: Expert NMPC IAE对比
  - S14: 延迟分解
  - S15: 自愈步数（CBF恢复步数）
"""

import sys
import os
import time
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_core import (
    QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
    PTRMNMPCPredictor, generate_quadrotor_dataset
)
from baselines import MPPIController

SEED = 2026
N_MC = 20
N_STEPS = 300
DT = 0.02
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
TErr_THRESH = 0.5

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
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def random_x_init():
    return torch.tensor([
        np.random.uniform(-0.5, 1.5), np.random.uniform(-1.0, 0.0),
        np.random.uniform(-0.5, 1.5), np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4), np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


def run_detailed_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True,
                         use_mismatch=False, process_noise=0.0, predictor_type='ptrm'):
    """收集详细试验数据：成功/失败分开统计IAE，记录CBF干预和自愈步数"""
    results = []
    for trial_idx in range(n_mc):
        set_seed(SEED + trial_idx)
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()
        collision = False
        min_dist = float('inf')
        iae = 0.0
        cbf_interventions = 0
        # 自愈步数：CBF violation后的恢复步数
        in_violation = False
        heal_steps = 0
        max_heal_steps = 0
        current_heal_count = 0
        collision_step = -1

        for step in range(N_STEPS):
            if predictor_type == 'ptrm':
                u_safe, safe_u_seq = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'baseline':
                u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'expert':
                u_safe = predictor.solve(x, x_sp)

            x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)
            p_np = x[0:3].detach().numpy()

            # 安全性检查 + CBF干预检测
            min_d = float('inf')
            for obs in env.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                min_d = min(min_d, d)
                if d < 0:
                    collision = True
                    if collision_step < 0:
                        collision_step = step

            min_dist = min(min_dist, min_d)

            # CBF干预检测：检查smooth barrier值是否接近0
            # 当障碍物距离 < buffer + margin 时，CBF很可能干预
            if enable_cbf:
                buffer_d = env.delta_buffer if hasattr(env, 'delta_buffer') else 0.15
                for obs in env.obstacles:
                    h_j = np.linalg.norm(p_np - obs['p']) - (obs['r'] + buffer_d)
                    if h_j < 0.2:  # 接近CBF激活边界
                        cbf_interventions += 1
                        break

            # 自愈检测：CBF barrier < 0 然后恢复 > 0
            if min_d < 0 and not in_violation:
                in_violation = True
                current_heal_count = 0
            elif min_d < 0 and in_violation:
                current_heal_count += 1
            elif min_d >= 0 and in_violation:
                # 恢复了！
                heal_steps = current_heal_count + 1  # 包括violation步
                max_heal_steps = max(max_heal_steps, heal_steps)
                in_violation = False
                current_heal_count = 0

            iae += torch.norm(x[0:3] - x_sp[0:3]).item()

        # IAE归一化: IAE = (1/T) * Σ ||e|| * dt = Σ ||e|| / N
        iae = iae / N_STEPS

        # 如果结束时仍在violation
        if in_violation:
            heal_steps = current_heal_count + 1
            max_heal_steps = max(max_heal_steps, heal_steps)

        terr = torch.norm(x[0:3] - x_sp[0:3]).item()
        success = (not collision) and (terr < TErr_THRESH)

        results.append({
            'trial': trial_idx,
            'success': success,
            'collision': collision,
            'terminal_error': terr,
            'iae': iae,
            'min_distance': min_dist,
            'cbf_interventions': cbf_interventions,
            'max_heal_steps': max_heal_steps,
            'collision_step': collision_step,
        })

    # 分组统计
    succ_results = [r for r in results if r['success']]
    coll_results = [r for r in results if r['collision']]

    stats = {
        'n_total': len(results),
        'n_success': len(succ_results),
        'n_collision': len(coll_results),
        'success_rate': len(succ_results) / len(results) * 100,
        # 全体统计
        'terr_mean': np.mean([r['terminal_error'] for r in results]),
        'terr_std': np.std([r['terminal_error'] for r in results]),
        'iae_mean': np.mean([r['iae'] for r in results]),
        'iae_std': np.std([r['iae'] for r in results]),
    }

    # S5: 分别报告
    if succ_results:
        stats['iae_success_mean'] = np.mean([r['iae'] for r in succ_results])
        stats['iae_success_std'] = np.std([r['iae'] for r in succ_results])
        stats['terr_success_mean'] = np.mean([r['terminal_error'] for r in succ_results])
    else:
        stats['iae_success_mean'] = float('nan')
        stats['iae_success_std'] = float('nan')
        stats['terr_success_mean'] = float('nan')

    if coll_results:
        stats['iae_collision_mean'] = np.mean([r['iae'] for r in coll_results])
        stats['iae_collision_std'] = np.std([r['iae'] for r in coll_results])
        stats['terr_collision_mean'] = np.mean([r['terminal_error'] for r in coll_results])
        # S8: 失败案例分析
        stats['collision_steps'] = [r['collision_step'] for r in coll_results]
        stats['collision_min_dists'] = [r['min_distance'] for r in coll_results]
    else:
        stats['iae_collision_mean'] = float('nan')
        stats['iae_collision_std'] = float('nan')
        stats['terr_collision_mean'] = float('nan')

    # S15: 自愈步数
    heal_steps_list = [r['max_heal_steps'] for r in results if r['max_heal_steps'] > 0]
    if heal_steps_list:
        stats['max_heal_steps_observed'] = max(heal_steps_list)
        stats['mean_heal_steps'] = np.mean(heal_steps_list)
        stats['n_heal_events'] = len(heal_steps_list)
    else:
        stats['max_heal_steps_observed'] = 0
        stats['mean_heal_steps'] = 0
        stats['n_heal_events'] = 0

    # CBF干预统计
    stats['cbf_interventions_mean'] = np.mean([r['cbf_interventions'] for r in results])
    stats['cbf_interventions_max'] = max([r['cbf_interventions'] for r in results])

    return stats, results


def run_latency_breakdown(trm_model, env, x_sp, K=50):
    """S14: 延迟分解"""
    import time

    predictor = PTRMNMPCPredictor(trm_model, env, K=K, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)

    x = random_x_init()
    x_sp_dev = x_sp

    # Warmup
    for _ in range(5):
        predictor.predict_action(x, x_sp_dev, enable_cbf=True)

    N_TRIALS = 20
    times = {
        'candidate_gen': [],
        'trm_forward': [],
        'q_head_eval': [],
        'rollout_eval': [],
        'cbf_projection': [],
        'total': [],
    }

    for _ in range(N_TRIALS):
        t_total_start = time.perf_counter()

        # 1. Candidate generation
        t0 = time.perf_counter()
        predictor.reset()
        x_input = torch.cat([x, x_sp_dev]).unsqueeze(0)

        # PD baseline
        K_p = torch.tensor([5.0, 5.0, 5.0])
        K_d = torch.tensor([2.0, 2.0, 2.0])
        u_pd = K_p * (x_sp_dev[0:3] - x[0:3]) + K_d * (x_sp_dev[3:6] - x[3:6])
        u_pd = torch.clamp(u_pd, -env.u_max, env.u_max)

        # Generate K Gaussian perturbed candidates
        candidates = []
        for _ in range(K):
            noise = torch.randn(30) * 2.0
            u_cand = u_pd.unsqueeze(0).repeat(10, 1) + noise.reshape(10, 3)
            u_cand = torch.clamp(u_cand, -env.u_max, env.u_max)
            candidates.append(u_cand)
        t1 = time.perf_counter()
        times['candidate_gen'].append((t1 - t0) * 1000)

        # 2. TRM forward pass (Q-head computation)
        t2 = time.perf_counter()
        with torch.no_grad():
            X_batch = x_input.repeat(K, 1)
            y_history = trm_model.forward_steps(X_batch, D=16, noise_scale=0.25)
            u_trm, final_y = y_history[-1]
            q_scores = trm_model.f_Q(final_y).squeeze(-1)
        t3 = time.perf_counter()
        times['trm_forward'].append((t3 - t2) * 1000)

        # 3. Q-head screening (top-M)
        t4 = time.perf_counter()
        top_M = 10
        _, top_indices = torch.topk(q_scores, min(top_M, K))
        t5 = time.perf_counter()
        times['q_head_eval'].append((t5 - t4) * 1000)

        # 4. Rollout evaluation
        t6 = time.perf_counter()
        rollout_costs = []
        for idx in top_indices:
            u_cand = candidates[idx.item()]
            cost = 0.0
            x_r = x.clone()
            for i in range(10):
                u_i = torch.clamp(u_cand[i], -env.u_max, env.u_max)
                x_r = env.step_discrete(x_r, u_i)
                error = x_r - x_sp_dev
                cost += torch.sum(error * error).item() + 0.02 * torch.sum(u_i * u_i).item()
            rollout_costs.append(cost)
        t7 = time.perf_counter()
        times['rollout_eval'].append((t7 - t6) * 1000)

        # 5. CBF projection
        t8 = time.perf_counter()
        best_idx = top_indices[np.argmin(rollout_costs)]
        u_nominal = candidates[best_idx.item()][0]
        u_safe = env.apply_cbf_projection(x, u_nominal)
        t9 = time.perf_counter()
        times['cbf_projection'].append((t9 - t8) * 1000)

        t_total_end = time.perf_counter()
        times['total'].append((t_total_end - t_total_start) * 1000)

    latency_stats = {}
    for name, vals in times.items():
        latency_stats[name] = {
            'mean_ms': np.mean(vals),
            'std_ms': np.std(vals),
            'max_ms': np.max(vals),
        }

    return latency_stats


def main():
    set_seed(SEED)

    # 加载模型
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    device = torch.device('cpu')

    print("加载TRM模型...")
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    # ==================================================================
    # S5/S8: 分组IAE统计 + 失败案例分析
    # ==================================================================
    print("\n" + "=" * 80)
    print("S5/S8: 分组IAE统计 + 失败案例分析")
    print("=" * 80)

    env = QuadrotorDynamics()

    # PTRM K=50 with CBF
    print("\n--- PTRM K=50 Strong CBF ---")
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   pd_sigma=2.0, use_rollout_cost=True)
    ptrm_stats, ptrm_details = run_detailed_trials(env, predictor, X_SP, enable_cbf=True)
    print(f"  成功率: {ptrm_stats['success_rate']:.0f}% ({ptrm_stats['n_success']}/{ptrm_stats['n_total']})")
    print(f"  全体 IAE: {ptrm_stats['iae_mean']:.1f} ± {ptrm_stats['iae_std']:.1f}")
    if not np.isnan(ptrm_stats['iae_success_mean']):
        print(f"  成功 IAE: {ptrm_stats['iae_success_mean']:.1f} ± {ptrm_stats['iae_success_std']:.1f}")
    if not np.isnan(ptrm_stats['iae_collision_mean']):
        print(f"  碰撞 IAE: {ptrm_stats['iae_collision_mean']:.1f} ± {ptrm_stats['iae_collision_std']:.1f}")
    print(f"  CBF干预次数: {ptrm_stats['cbf_interventions_mean']:.1f} (max={ptrm_stats['cbf_interventions_max']})")
    print(f"  自愈事件数: {ptrm_stats['n_heal_events']}, 最大自愈步数: {ptrm_stats['max_heal_steps_observed']}")

    # PTRM K=50 without CBF (有失败案例)
    print("\n--- PTRM K=50 No CBF ---")
    set_seed(SEED)
    predictor_nocbf = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                          alpha_blend=0.3, candidate_mode='pd',
                                          pd_sigma=2.0, use_rollout_cost=True)
    nocbf_stats, nocbf_details = run_detailed_trials(env, predictor_nocbf, X_SP, enable_cbf=False)
    print(f"  成功率: {nocbf_stats['success_rate']:.0f}% ({nocbf_stats['n_success']}/{nocbf_stats['n_total']})")
    print(f"  全体 IAE: {nocbf_stats['iae_mean']:.1f} ± {nocbf_stats['iae_std']:.1f}")
    if not np.isnan(nocbf_stats['iae_success_mean']):
        print(f"  成功 IAE: {nocbf_stats['iae_success_mean']:.1f} ± {nocbf_stats['iae_success_std']:.1f}")
    if not np.isnan(nocbf_stats['iae_collision_mean']):
        print(f"  碰撞 IAE: {nocbf_stats['iae_collision_mean']:.1f} ± {nocbf_stats['iae_collision_std']:.1f}")
        # S8: 失败案例分析
        print(f"  碰撞步骤: {nocbf_stats['collision_steps']}")
        print(f"  碰撞最小距离: {[f'{d:.3f}' for d in nocbf_stats['collision_min_dists']]}")

    # MPPI K=50 with CBF
    print("\n--- MPPI K=50 Strong CBF ---")
    set_seed(SEED)
    mppi = MPPIController(env, K=50, sigma=2.0)
    mppi_stats, mppi_details = run_detailed_trials(env, mppi, X_SP, enable_cbf=True, predictor_type='baseline')
    print(f"  成功率: {mppi_stats['success_rate']:.0f}% ({mppi_stats['n_success']}/{mppi_stats['n_total']})")
    print(f"  全体 IAE: {mppi_stats['iae_mean']:.1f} ± {mppi_stats['iae_std']:.1f}")
    if not np.isnan(mppi_stats['iae_success_mean']):
        print(f"  成功 IAE: {mppi_stats['iae_success_mean']:.1f} ± {mppi_stats['iae_success_std']:.1f}")

    # ==================================================================
    # S9: Expert NMPC IAE对比
    # ==================================================================
    print("\n" + "=" * 80)
    print("S9: Expert NMPC IAE对比")
    print("=" * 80)

    env_expert = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env_expert, horizon=10)

    expert_results = []
    N_MC_EXPERT = 5  # Expert NMPC非常慢，减少试验次数
    N_STEPS_EXPERT = 150  # 减少步数
    for trial_idx in range(N_MC_EXPERT):
        set_seed(SEED + trial_idx)
        x_init = random_x_init()
        x = x_init.clone()
        collision = False
        iae = 0.0

        for step in range(N_STEPS_EXPERT):
            u_seq = solver.solve(x, X_SP)
            if u_seq is None:
                # Fallback to PD
                K_p = torch.tensor([5.0, 5.0, 5.0])
                K_d = torch.tensor([2.0, 2.0, 2.0])
                u = K_p * (X_SP[0:3] - x[0:3]) + K_d * (X_SP[3:6] - x[3:6])
                u = torch.clamp(u, -env_expert.u_max, env_expert.u_max)
            else:
                u = u_seq[0:3]  # 取第一步控制
            x = env_expert.step_discrete(x, u)
            p_np = x[0:3].detach().numpy()
            for obs in env_expert.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                if d < 0: collision = True
            iae += torch.norm(x[0:3] - X_SP[0:3]).item()

        # IAE归一化
        iae = iae / N_STEPS_EXPERT

        terr = torch.norm(x[0:3] - X_SP[0:3]).item()
        expert_results.append({
            'success': (not collision) and (terr < TErr_THRESH),
            'collision': collision,
            'terminal_error': terr,
            'iae': iae,
        })

    expert_succ = [r for r in expert_results if r['success']]
    expert_coll = [r for r in expert_results if r['collision']]
    print(f"  Expert NMPC:")
    print(f"    成功率: {len(expert_succ)/len(expert_results)*100:.0f}%")
    print(f"    TErr: {np.mean([r['terminal_error'] for r in expert_results]):.4f}m")
    print(f"    IAE (all): {np.mean([r['iae'] for r in expert_results]):.1f} ± {np.std([r['iae'] for r in expert_results]):.1f}")
    if expert_succ:
        print(f"    IAE (success): {np.mean([r['iae'] for r in expert_succ]):.1f}")

    # ==================================================================
    # S14: 延迟分解
    # ==================================================================
    print("\n" + "=" * 80)
    print("S14: 延迟分解 (K=50)")
    print("=" * 80)

    latency = run_latency_breakdown(trm_model, env, X_SP, K=50)
    total_est = sum(v['mean_ms'] for k, v in latency.items() if k != 'total')
    print(f"  Candidate generation: {latency['candidate_gen']['mean_ms']:.2f} ± {latency['candidate_gen']['std_ms']:.2f} ms")
    print(f"  TRM forward pass:     {latency['trm_forward']['mean_ms']:.2f} ± {latency['trm_forward']['std_ms']:.2f} ms")
    print(f"  Q-head screening:     {latency['q_head_eval']['mean_ms']:.2f} ± {latency['q_head_eval']['std_ms']:.2f} ms")
    print(f"  Rollout evaluation:   {latency['rollout_eval']['mean_ms']:.2f} ± {latency['rollout_eval']['std_ms']:.2f} ms")
    print(f"  CBF projection:       {latency['cbf_projection']['mean_ms']:.2f} ± {latency['cbf_projection']['std_ms']:.2f} ms")
    print(f"  Sum of components:    {total_est:.2f} ms")
    print(f"  End-to-end total:     {latency['total']['mean_ms']:.2f} ± {latency['total']['std_ms']:.2f} ms")

    # ==================================================================
    # S15: 自愈步数 (多场景)
    # ==================================================================
    print("\n" + "=" * 80)
    print("S15: 自愈步数分析 (多个K值和障碍物配置)")
    print("=" * 80)

    for K_val in [1, 10, 50]:
        env_heal = QuadrotorDynamics()
        predictor_heal = PTRMNMPCPredictor(trm_model, env_heal, K=K_val, D=16,
                                            sigma=0.25 if K_val > 1 else 0.0,
                                            alpha_blend=0.3, candidate_mode='pd',
                                            pd_sigma=2.0, use_rollout_cost=True)
        # 用更强噪声触发CBF violation
        stats_heal, _ = run_detailed_trials(env_heal, predictor_heal, X_SP,
                                             enable_cbf=True, use_mismatch=True,
                                             process_noise=0.03)
        print(f"  K={K_val:3d}: CBF干预={stats_heal['cbf_interventions_mean']:.1f}, "
              f"自愈事件={stats_heal['n_heal_events']}, "
              f"最大自愈步数={stats_heal['max_heal_steps_observed']}")

    print("\n补充实验完成!")


if __name__ == '__main__':
    main()
