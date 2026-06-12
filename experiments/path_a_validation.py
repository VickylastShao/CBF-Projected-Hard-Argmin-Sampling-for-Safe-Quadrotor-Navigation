# -*- coding: utf-8 -*-
"""
路径A验证实验：证明TRM架构独特价值

三个关键实验：
  实验1: Q-head vs Random vs Rollout-All 排序对比
  实验2: Q-head 对 PD 候选的排序相关性（vs 对 TRM 候选的排序相关性）
  实验3: Simple Encoder 消融（训练 + MC 评估）

目标：验证 Q-head 排序是否比随机选择更有信息量，
      以及 TRM 递归结构是否比简单编码器产生更好的 Q-head 特征。
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
    PTRMNMPCPredictor, SimpleEncoderQHead, generate_quadrotor_dataset,
    train_trm_jointly, train_simple_encoder_qhead
)
from baselines import MPPIController

SEED = 2026
N_MC = 20
N_STEPS = 300
DT = 0.02
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


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True,
                   use_mismatch=False, process_noise=0.0, predictor_type='ptrm'):
    """标准MC试验（与v6_quick_test相同）"""
    results = []
    for _ in range(n_mc):
        x_init = random_x_init()
        predictor.reset()
        x = x_init.clone()
        collision = False; min_dist = float('inf'); iae = 0.0

        for step in range(N_STEPS):
            if predictor_type == 'ptrm':
                u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'baseline':
                u_safe = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
            elif predictor_type == 'simple_encoder':
                u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)

            x = env.step_discrete(x, u_safe, use_mismatch=use_mismatch, process_noise=process_noise)
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
    colls = [r['collision'] for r in results]
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    return {
        'success_rate': np.mean(succs) * 100,
        'collision_rate': np.mean(colls) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'min_distance_mean': np.mean([r['min_distance'] for r in results]),
    }


class SimpleEncoderPredictor:
    """
    简单编码器Q-head预测器（路径A实验3专用）

    替代PTRMNMPCPredictor，使用SimpleEncoderQHead进行Q-head评分。
    候选生成逻辑与PD模式相同，但Q-head特征来自简单编码器而非TRM。
    """

    def __init__(self, model, env, K=50, sigma=0.25,
                 tracking_Kp=4.0, tracking_Kd=3.0,
                 eta_hyst=0.05, pd_sigma=2.0,
                 rollout_top_m=10, rollout_steps=20, obs_weight=2000.0):
        self.model = model
        self.env = env
        self.K = K
        self.sigma = sigma
        self.tracking_Kp = tracking_Kp
        self.tracking_Kd = tracking_Kd
        self.eta_hyst = eta_hyst
        self.pd_sigma = pd_sigma
        self.rollout_top_m = min(rollout_top_m, K)
        self.rollout_steps = rollout_steps
        self.obs_weight = obs_weight
        self.last_u_seq = None
        self.q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        self.R_U = 0.02

    def reset(self):
        self.last_u_seq = None

    def _compute_tracking_correction(self, x_init, x_sp):
        e_p = x_sp[0:3] - x_init[0:3]
        e_v = x_sp[3:6] - x_init[3:6]
        u_corr = self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)
        return u_corr

    def _batch_rollout_cost(self, x_init, u_first_candidates, x_sp):
        M = u_first_candidates.shape[0]
        x = x_init.unsqueeze(0).repeat(M, 1)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(M, 1)
        cost = torch.zeros(M)
        q = self.q_diag.unsqueeze(0)

        for s in range(self.rollout_steps):
            if s == 0:
                u = u_first_candidates
            else:
                e_p = x_sp6[:, 0:3] - x[:, 0:3]
                e_v = x_sp6[:, 3:6] - x[:, 3:6]
                u = self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)

            u = torch.clamp(u, self.env.u_min, self.env.u_max)
            p = x[:, 0:3]
            v = x[:, 3:6]
            v_dot = u / self.env.m - (self.env.b_drag / self.env.m) * v
            p_next = p + self.env.dt * v
            v_next = v + self.env.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)

            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + self.R_U * torch.sum(u * u, dim=1)

            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs['p'], dtype=torch.float32).unsqueeze(0).repeat(M, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs['r']
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2

        return cost

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        """与PTRMNMPCPredictor PD模式相同的决策流程，但Q-head来自简单编码器"""
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            # PD候选生成（与PTRMNMPCPredictor._generate_candidates_pd相同）
            u_pd = self._compute_tracking_correction(x_init.cpu(), x_sp.cpu())

            if self.K == 1:
                u_candidates = u_pd.unsqueeze(0).repeat(1, 30).to(device)
                X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
                _, q_scores = self.model(X_single)
                q_scores = q_scores.squeeze(-1)
            else:
                noise = torch.randn(self.K, 3) * self.pd_sigma
                u_first_candidates = u_pd.unsqueeze(0) + noise
                u_pd_full = u_pd.unsqueeze(0).repeat(self.K, 1)
                u_candidates = torch.zeros(self.K, 30, device=device)
                u_candidates[:, 0:3] = u_first_candidates.to(device)
                for i in range(10):
                    u_candidates[:, i*3:(i+1)*3] = u_pd_full.to(device) if i > 0 else u_first_candidates.to(device)

                # 简单编码器Q-head评分（替代TRM Q-head）
                X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
                X_parallel = X_single.repeat(self.K, 1)
                # 加入噪声以产生不同的评分（模拟TRM的潜在空间扰动）
                if self.sigma > 0:
                    X_perturbed = X_parallel + torch.randn_like(X_parallel) * self.sigma * 0.1
                else:
                    X_perturbed = X_parallel
                latent_y, q_all = self.model(X_perturbed)
                q_scores = q_all.squeeze(-1)  # (K,)

            # 滞回正则化
            if self.K > 1 and self.last_u_seq is not None:
                u_shift = torch.cat([self.last_u_seq[3:], self.last_u_seq[-3:]]).to(device)
                u_shift_batch = u_shift.unsqueeze(0).repeat(self.K, 1)
                dist = torch.sum((u_candidates - u_shift_batch) ** 2, dim=1)
                q_scores = q_scores - self.eta_hyst * dist

            # 两阶段评估
            if self.K > 1:
                M = min(self.rollout_top_m, self.K)
                _, top_indices = torch.topk(q_scores, min(M, self.K))
                top_indices = top_indices.sort()[0]
                u_first_top = u_candidates[top_indices, 0:3].cpu()
                rollout_costs = self._batch_rollout_cost(x_init.cpu(), u_first_top, x_sp.cpu())
                best_in_top = torch.argmin(rollout_costs).item()
                best_idx = top_indices[best_in_top].item()
            else:
                best_idx = 0

            best_u_sequence = u_candidates[best_idx]
            self.last_u_seq = best_u_sequence.clone()
            u_nominal = best_u_sequence[0:3].cpu()

            if enable_cbf:
                u_safe = self.env.apply_cbf_projection(x_init.cpu(), u_nominal)
                safe_u_sequence = best_u_sequence.clone().cpu()
                safe_u_sequence[0:3] = u_safe
            else:
                u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
                safe_u_sequence = torch.clamp(best_u_sequence.cpu(), self.env.u_min, self.env.u_max)

            return u_safe, safe_u_sequence


def main():
    set_seed(SEED)
    t_start = time.time()

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    os.makedirs(save_dir, exist_ok=True)

    # ===== 加载/训练模型 =====
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    simple_enc_path = os.path.join(save_dir, 'simple_encoder_model.pt')
    device = torch.device('cpu')

    # TRM模型
    if os.path.exists(trm_path):
        print("加载TRM模型...")
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    else:
        print("训练TRM模型...")
        env_train = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env_train, horizon=10)
        dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                              x_sp=X_SP, pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model, _ = train_trm_jointly(trm_model, dataset, env_train, epochs=100, lr=0.001, patience=20, verbose=True)
        torch.save(trm_model.state_dict(), trm_path)
    trm_model.eval()

    # 数据集（为Simple Encoder训练复用）
    print("生成训练数据集...")
    env_train = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env_train, horizon=10)
    dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                          x_sp=X_SP, pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])

    # Simple Encoder模型
    if os.path.exists(simple_enc_path):
        print("加载Simple Encoder模型...")
        simple_model = SimpleEncoderQHead(input_dim=12, latent_dim=64).to(device)
        simple_model.load_state_dict(torch.load(simple_enc_path, map_location=device, weights_only=True))
    else:
        print("训练Simple Encoder模型...")
        simple_model = SimpleEncoderQHead(input_dim=12, latent_dim=64).to(device)
        simple_model, _ = train_simple_encoder_qhead(simple_model, dataset, env_train,
                                                      epochs=100, lr=0.001, patience=20, verbose=True)
        torch.save(simple_model.state_dict(), simple_enc_path)
    simple_model.eval()

    # ====================================================================
    # 实验1: Q-head vs Random vs Rollout-All 排序对比
    # ====================================================================
    print("\n" + "=" * 80)
    print("实验1: Q-head vs Random vs Rollout-All 排序对比 (K=50, Strong CBF)")
    print("=" * 80)

    env = QuadrotorDynamics()
    exp1_results = {}

    for ranking_mode, label in [
        ('q_head', 'Q-head排序'),
        ('random', '随机排序'),
        ('rollout_all', '全量Rollout排序')
    ]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True,
                                       ranking_mode=ranking_mode)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
        exp1_results[label] = result
        print(f"  {label:14s}: Succ={result['success_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.4f}m, "
              f"IAE={result['iae_mean']:.1f}")

    # K=10也测试（不同K下排序模式的效果差异）
    print("\n--- K=10 排序对比 ---")
    for ranking_mode, label in [
        ('q_head', 'Q-head排序'),
        ('random', '随机排序'),
    ]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=10, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True,
                                       ranking_mode=ranking_mode)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
        exp1_results[f'{label}_K10'] = result
        print(f"  {label:14s} K=10: Succ={result['success_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.4f}m, "
              f"IAE={result['iae_mean']:.1f}")

    # ====================================================================
    # 实验2: Q-head 对 PD 候选的排序相关性
    # ====================================================================
    print("\n" + "=" * 80)
    print("实验2: Q-head 排序相关性 — PD候选 vs TRM候选")
    print("=" * 80)

    from scipy.stats import spearmanr, pearsonr

    N_SAMPLES = 100  # 100个不同初始状态
    K_CORR = 50      # 每个状态50个候选

    # 2a: Q-head 对 PD 候选的排序相关性（关键实验）
    q_pd_scores_all = []
    pd_rollout_costs_all = []

    # 2b: Q-head 对 TRM 候选的排序相关性（Exp 9 复现，作为对比）
    q_trm_scores_all = []
    trm_rollout_costs_all = []

    for sample_idx in range(N_SAMPLES):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)

        with torch.no_grad():
            # --- PD候选的Q-head评分和rollout代价 ---
            # 生成PD候选
            env_temp = QuadrotorDynamics()
            e_p = x_sp_dev[0:3] - x_init[0:3]
            e_v = x_sp_dev[3:6] - x_init[3:6]
            u_pd = env_temp.m * (4.0 * e_p + 3.0 * e_v)

            noise_pd = torch.randn(K_CORR, 3) * 2.0
            u_first_pd = u_pd.unsqueeze(0) + noise_pd  # (K, 3)

            # PD候选的rollout代价
            pd_costs = []
            q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
            for k_idx in range(K_CORR):
                x_r = x_init.cpu().clone()
                cost = 0.0
                for s in range(20):  # rollout_steps=20
                    if s == 0:
                        u_s = torch.clamp(u_first_pd[k_idx].cpu(), env_temp.u_min, env_temp.u_max)
                    else:
                        e_p_s = X_SP[0:3] - x_r[0:3]
                        e_v_s = X_SP[3:6] - x_r[3:6]
                        u_s = torch.clamp(env_temp.m * (4.0 * e_p_s + 3.0 * e_v_s),
                                          env_temp.u_min, env_temp.u_max)
                    # Euler积分
                    p = x_r[0:3]; v = x_r[3:6]
                    v_dot = u_s / env_temp.m - (env_temp.b_drag / env_temp.m) * v
                    x_r = torch.cat([p + env_temp.dt * v, v + env_temp.dt * v_dot])
                    err = x_r - X_SP
                    cost += torch.sum(q_diag * err * err).item() + 0.02 * torch.sum(u_s * u_s).item()
                    # 障碍物代价
                    for obs in env_temp.obstacles:
                        d = torch.norm(x_r[0:3] - torch.tensor(obs['p'])) - obs['r']
                        cost += 2000.0 * max(0.0, 0.3 - d.item()) ** 2
                pd_costs.append(cost)

            # PD候选的Q-head评分（通过TRM forward pass获取latent，再用f_Q评分）
            X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
            X_parallel = X_single.repeat(K_CORR, 1)
            y_history = trm_model.forward_steps(X_parallel, D=16, noise_scale=0.25)
            _, final_latent_y = y_history[-1]
            q_scores_pd = trm_model.f_Q(final_latent_y).squeeze(-1).cpu().numpy()

            q_pd_scores_all.extend(q_scores_pd.tolist())
            pd_rollout_costs_all.extend(pd_costs)

            # --- TRM候选的Q-head评分和rollout代价（Exp 9 复现） ---
            X_trm = torch.cat([x_init, x_sp_dev]).unsqueeze(0).repeat(K_CORR, 1)
            y_history_trm = trm_model.forward_steps(X_trm, D=16, noise_scale=0.25)
            u_trm_candidates, final_latent_y_trm = y_history_trm[-1]
            q_scores_trm = trm_model.f_Q(final_latent_y_trm).squeeze(-1).cpu().numpy()

            trm_costs = []
            for k_idx in range(K_CORR):
                u_k = u_trm_candidates[k_idx].cpu()
                cost = 0.0
                x_r = x_init.cpu().clone()
                for i in range(10):
                    u_i = torch.clamp(u_k[i*3:(i+1)*3], env_temp.u_min, env_temp.u_max)
                    x_r = env_temp.step_discrete(x_r, u_i)
                    error = x_r - X_SP
                    cost += torch.sum(q_diag * error * error).item() + 0.02 * torch.sum(u_i * u_i).item()
                trm_costs.append(cost)

            q_trm_scores_all.extend(q_scores_trm.tolist())
            trm_rollout_costs_all.extend(trm_costs)

        if (sample_idx + 1) % 20 == 0:
            print(f"  采样进度: {sample_idx+1}/{N_SAMPLES}")

    # 计算相关性
    q_pd_arr = np.array(q_pd_scores_all)
    pd_cost_arr = np.array(pd_rollout_costs_all)
    q_trm_arr = np.array(q_trm_scores_all)
    trm_cost_arr = np.array(trm_rollout_costs_all)

    # Q-head vs PD候选rollout代价
    rho_pd, p_pd = spearmanr(q_pd_arr, pd_cost_arr)
    r_pd, rp_pd = pearsonr(q_pd_arr, pd_cost_arr)

    # Q-head vs TRM候选rollout代价
    rho_trm, p_trm = spearmanr(q_trm_arr, trm_cost_arr)
    r_trm, rp_trm = pearsonr(q_trm_arr, trm_cost_arr)

    print(f"\n  Q-head 对 PD 候选排序相关性:")
    print(f"    Spearman ρ = {rho_pd:.4f} (p={p_pd:.2e})")
    print(f"    Pearson  r = {r_pd:.4f} (p={rp_pd:.2e})")
    print(f"\n  Q-head 对 TRM 候选排序相关性:")
    print(f"    Spearman ρ = {rho_trm:.4f} (p={p_trm:.2e})")
    print(f"    Pearson  r = {r_trm:.4f} (p={rp_trm:.2e})")
    print(f"\n  差异 (PD vs TRM): Δρ = {rho_pd - rho_trm:.4f}")

    # 随机基线：N次随机排序的平均Spearman ρ
    N_RANDOM = 1000
    random_rho_list = []
    for _ in range(N_RANDOM):
        rand_scores = np.random.randn(K_CORR)
        rand_costs = np.random.randn(K_CORR)
        random_rho_list.append(abs(spearmanr(rand_scores, rand_costs)[0]))
    random_rho_mean = np.mean(random_rho_list)
    print(f"\n  随机基线: |ρ|_random = {random_rho_mean:.4f} (期望≈0.08 for K=50)")

    exp2_results = {
        'q_head_vs_pd': {'spearman_rho': float(rho_pd), 'pearson_r': float(r_pd),
                          'spearman_p': float(p_pd), 'pearson_p': float(rp_pd)},
        'q_head_vs_trm': {'spearman_rho': float(rho_trm), 'pearson_r': float(r_trm),
                           'spearman_p': float(p_trm), 'pearson_p': float(rp_trm)},
        'random_baseline_abs_rho': float(random_rho_mean),
        'delta_rho_pd_vs_trm': float(rho_pd - rho_trm),
    }

    # ====================================================================
    # 实验3: Simple Encoder 消融
    # ====================================================================
    print("\n" + "=" * 80)
    print("实验3: Simple Encoder 消融 (K=50, Strong CBF)")
    print("=" * 80)

    env = QuadrotorDynamics()

    # 3a: TRM Q-head 基线
    set_seed(SEED)
    ptrm_predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                        alpha_blend=0.3, candidate_mode='pd',
                                        pd_sigma=2.0, use_rollout_cost=True,
                                        ranking_mode='q_head')
    trm_result = run_mc_trials(env, ptrm_predictor, X_SP, enable_cbf=True)
    print(f"  TRM Q-head:    Succ={trm_result['success_rate']:.0f}%, "
          f"TErr={trm_result['terminal_error_mean']:.4f}m, "
          f"IAE={trm_result['iae_mean']:.1f}")

    # 3b: Simple Encoder Q-head
    set_seed(SEED)
    simple_predictor = SimpleEncoderPredictor(simple_model, env, K=50, sigma=0.25,
                                               pd_sigma=2.0)
    simple_result = run_mc_trials(env, simple_predictor, X_SP, enable_cbf=True,
                                   predictor_type='simple_encoder')
    print(f"  SimpleEncoder: Succ={simple_result['success_rate']:.0f}%, "
          f"TErr={simple_result['terminal_error_mean']:.4f}m, "
          f"IAE={simple_result['iae_mean']:.1f}")

    # 3c: 随机排序基线（复用实验1数据）
    random_result = exp1_results['随机排序']
    print(f"  Random排序:    Succ={random_result['success_rate']:.0f}%, "
          f"TErr={random_result['terminal_error_mean']:.4f}m, "
          f"IAE={random_result['iae_mean']:.1f}")

    # 3d: 全量Rollout上界（复用实验1数据）
    rollout_all_result = exp1_results['全量Rollout排序']
    print(f"  全量Rollout:   Succ={rollout_all_result['success_rate']:.0f}%, "
          f"TErr={rollout_all_result['terminal_error_mean']:.4f}m, "
          f"IAE={rollout_all_result['iae_mean']:.1f}")

    # 3e: Simple Encoder的Q-head排序相关性
    q_simple_scores_all = []
    pd_simple_costs_all = []

    for sample_idx in range(N_SAMPLES):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)

        with torch.no_grad():
            X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
            X_parallel = X_single.repeat(K_CORR, 1)

            # Simple Encoder评分
            X_perturbed = X_parallel + torch.randn_like(X_parallel) * 0.025
            _, q_simple = simple_model(X_perturbed)
            q_scores_simple = q_simple.squeeze(-1).cpu().numpy()

            # PD候选rollout代价（复用上面已有的逻辑）
            env_temp = QuadrotorDynamics()
            e_p = x_sp_dev[0:3] - x_init[0:3]
            e_v = x_sp_dev[3:6] - x_init[3:6]
            u_pd = env_temp.m * (4.0 * e_p + 3.0 * e_v)
            noise_pd = torch.randn(K_CORR, 3) * 2.0
            u_first_pd = u_pd.unsqueeze(0) + noise_pd

            pd_costs = []
            q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
            for k_idx in range(K_CORR):
                x_r = x_init.cpu().clone()
                cost = 0.0
                for s in range(20):
                    if s == 0:
                        u_s = torch.clamp(u_first_pd[k_idx].cpu(), env_temp.u_min, env_temp.u_max)
                    else:
                        e_p_s = X_SP[0:3] - x_r[0:3]
                        e_v_s = X_SP[3:6] - x_r[3:6]
                        u_s = torch.clamp(env_temp.m * (4.0 * e_p_s + 3.0 * e_v_s),
                                          env_temp.u_min, env_temp.u_max)
                    p = x_r[0:3]; v = x_r[3:6]
                    v_dot = u_s / env_temp.m - (env_temp.b_drag / env_temp.m) * v
                    x_r = torch.cat([p + env_temp.dt * v, v + env_temp.dt * v_dot])
                    err = x_r - X_SP
                    cost += torch.sum(q_diag * err * err).item() + 0.02 * torch.sum(u_s * u_s).item()
                    for obs in env_temp.obstacles:
                        d = torch.norm(x_r[0:3] - torch.tensor(obs['p'])) - obs['r']
                        cost += 2000.0 * max(0.0, 0.3 - d.item()) ** 2
                pd_costs.append(cost)

            q_simple_scores_all.extend(q_scores_simple.tolist())
            pd_simple_costs_all.extend(pd_costs)

    q_simple_arr = np.array(q_simple_scores_all)
    pd_simple_cost_arr = np.array(pd_simple_costs_all)
    rho_simple, p_simple = spearmanr(q_simple_arr, pd_simple_cost_arr)
    r_simple, rp_simple = pearsonr(q_simple_arr, pd_simple_cost_arr)

    print(f"\n  Simple Encoder 对 PD 候选排序相关性:")
    print(f"    Spearman ρ = {rho_simple:.4f} (p={p_simple:.2e})")
    print(f"    Pearson  r = {r_simple:.4f} (p={rp_simple:.2e})")

    # ====================================================================
    # 汇总与结论
    # ====================================================================
    print("\n" + "=" * 80)
    print("路径A验证实验汇总")
    print("=" * 80)

    print("\n--- 实验1: 排序模式对比 (K=50) ---")
    for label in ['Q-head排序', '随机排序', '全量Rollout排序']:
        r = exp1_results[label]
        print(f"  {label:14s}: Succ={r['success_rate']:.0f}%, "
              f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    print("\n--- 实验2: Q-head排序相关性 ---")
    print(f"  Q-head vs PD候选:  ρ = {rho_pd:.4f}")
    print(f"  Q-head vs TRM候选: ρ = {rho_trm:.4f}")
    print(f"  SimpleEnc vs PD:   ρ = {rho_simple:.4f}")

    print("\n--- 实验3: 编码器消融 ---")
    print(f"  TRM Q-head (27935 params): IAE={trm_result['iae_mean']:.1f}")
    print(f"  SimpleEncoder (2913 params): IAE={simple_result['iae_mean']:.1f}")
    print(f"  ΔIAE = {trm_result['iae_mean'] - simple_result['iae_mean']:.1f} "
          f"({(trm_result['iae_mean'] - simple_result['iae_mean']) / simple_result['iae_mean'] * 100:.1f}%)")

    # 判断结论
    print("\n--- 关键结论 ---")

    # 实验1判断
    qh_iae = exp1_results['Q-head排序']['iae_mean']
    rand_iae = exp1_results['随机排序']['iae_mean']
    rollout_iae = exp1_results['全量Rollout排序']['iae_mean']
    iae_diff_pct = (rand_iae - qh_iae) / qh_iae * 100

    if abs(iae_diff_pct) < 3:
        print(f"  ⚠ 实验1: Q-head排序 vs 随机排序 IAE差异仅{iae_diff_pct:.1f}%，"
              f"Q-head可能不提供有信息量的排序信号")
    else:
        print(f"  ✓ 实验1: Q-head排序 vs 随机排序 IAE差异{iae_diff_pct:.1f}%，"
              f"Q-head提供了有意义的排序信号")

    # 实验2判断
    if abs(rho_pd) > 0.15 and abs(rho_pd) > 3 * random_rho_mean:
        print(f"  ✓ 实验2: Q-head对PD候选排序相关性强(ρ={rho_pd:.4f})，"
              f"显著高于随机基线(|ρ|_rand={random_rho_mean:.4f})")
    else:
        print(f"  ⚠ 实验2: Q-head对PD候选排序相关性弱(ρ={rho_pd:.4f})，"
              f"接近随机基线(|ρ|_rand={random_rho_mean:.4f})")

    # 实验3判断
    iae_trm_vs_simple = (simple_result['iae_mean'] - trm_result['iae_mean']) / trm_result['iae_mean'] * 100
    if iae_trm_vs_simple > 3:
        print(f"  ✓ 实验3: TRM Q-head比SimpleEncoder Q-head好{iae_trm_vs_simple:.1f}%，"
              f"TRM递归结构有独特价值")
    elif iae_trm_vs_simple > 0:
        print(f"  ⚠ 实验3: TRM Q-head比SimpleEncoder Q-head仅好{iae_trm_vs_simple:.1f}%，"
              f"优势微弱")
    else:
        print(f"  ✗ 实验3: SimpleEncoder Q-head反而优于TRM Q-head "
              f"({abs(iae_trm_vs_simple):.1f}%)，TRM递归结构无独特价值")

    t_total = time.time() - t_start
    print(f"\n总实验时间: {t_total:.1f}s ({t_total/60:.1f}min)")

    # 保存结果
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results = {
        'exp1_ranking_comparison': strip(exp1_results),
        'exp2_qhead_correlation': exp2_results,
        'exp3_simple_encoder_ablation': {
            'trm_qhead': strip(trm_result),
            'simple_encoder': strip(simple_result),
            'random_ranking': strip(random_result),
            'rollout_all': strip(rollout_all_result),
            'simple_encoder_correlation': {
                'spearman_rho': float(rho_simple),
                'pearson_r': float(r_simple),
            }
        }
    }

    results_path = os.path.join(save_dir, 'path_a_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至 {results_path}")


if __name__ == '__main__':
    main()
