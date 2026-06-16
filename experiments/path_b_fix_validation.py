# -*- coding: utf-8 -*-
"""
架构修复验证实验：Q-head条件化于PD候选

修复方案：在forward_steps()中通过W_y将PD候选序列编码到TRM潜在空间，
使Q-head评分基于PD候选而非TRM自身解码输出。

重复路径A的三组实验，验证修复后：
1. Q-head排序是否优于随机排序
2. Q-head对PD候选的排序相关性是否保持/提升
3. TRM Q-head是否优于SimpleEncoder Q-head
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


def run_mc_trials(env, predictor, x_sp, n_mc=N_MC, enable_cbf=True,
                   use_mismatch=False, process_noise=0.0, predictor_type='ptrm'):
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
    terrs = [r['terminal_error'] for r in results]
    iaes = [r['iae'] for r in results]
    return {
        'success_rate': np.mean(succs) * 100,
        'collision_rate': np.mean([r['collision'] for r in results]) * 100,
        'terminal_error_mean': np.mean(terrs),
        'terminal_error_std': np.std(terrs),
        'iae_mean': np.mean(iaes),
        'iae_std': np.std(iaes),
        'min_distance_mean': np.mean([r['min_distance'] for r in results]),
    }


class SimpleEncoderPredictor:
    """简单编码器Q-head预测器（与path_a_validation.py相同）"""

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
        return self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)

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
            p = x[:, 0:3]; v = x[:, 3:6]
            v_dot = u / self.env.m - (self.env.b_drag / self.env.m) * v
            x = torch.cat([p + self.env.dt * v, v + self.env.dt * v_dot], dim=1)
            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + self.R_U * torch.sum(u * u, dim=1)
            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs['p'], dtype=torch.float32).unsqueeze(0).repeat(M, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs['r']
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2
        return cost

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
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
                X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
                X_parallel = X_single.repeat(self.K, 1)
                if self.sigma > 0:
                    X_perturbed = X_parallel + torch.randn_like(X_parallel) * self.sigma * 0.1
                else:
                    X_perturbed = X_parallel
                latent_y, q_all = self.model(X_perturbed)
                q_scores = q_all.squeeze(-1)

            if self.K > 1 and self.last_u_seq is not None:
                u_shift = torch.cat([self.last_u_seq[3:], self.last_u_seq[-3:]]).to(device)
                u_shift_batch = u_shift.unsqueeze(0).repeat(self.K, 1)
                dist = torch.sum((u_candidates - u_shift_batch) ** 2, dim=1)
                q_scores = q_scores - self.eta_hyst * dist

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
    device = torch.device('cpu')

    # 加载TRM模型
    trm_path = os.path.join(save_dir, 'trm_model.pt')
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    # 加载Simple Encoder模型
    simple_enc_path = os.path.join(save_dir, 'simple_encoder_model.pt')
    simple_model = SimpleEncoderQHead(input_dim=12, latent_dim=64).to(device)
    simple_model.load_state_dict(torch.load(simple_enc_path, map_location=device, weights_only=True))
    simple_model.eval()

    print("模型加载完成")

    # ====================================================================
    # 实验1: 修复后 Q-head vs Random vs Rollout-All 排序对比
    # ====================================================================
    print("\n" + "=" * 80)
    print("实验1: 修复后 Q-head vs Random vs Rollout-All 排序对比 (K=50)")
    print("=" * 80)

    env = QuadrotorDynamics()
    exp1_results = {}

    for ranking_mode, label in [
        ('q_head', 'Q-head排序(修复)'),
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
        print(f"  {label:18s}: Succ={result['success_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.4f}m, "
              f"IAE={result['iae_mean']:.1f}")

    # K=10
    print("\n--- K=10 排序对比 ---")
    for ranking_mode, label in [
        ('q_head', 'Q-head排序(修复)'),
        ('random', '随机排序'),
    ]:
        set_seed(SEED)
        predictor = PTRMNMPCPredictor(trm_model, env, K=10, D=16, sigma=0.25,
                                       alpha_blend=0.3, candidate_mode='pd',
                                       pd_sigma=2.0, use_rollout_cost=True,
                                       ranking_mode=ranking_mode)
        result = run_mc_trials(env, predictor, X_SP, enable_cbf=True)
        exp1_results[f'{label}_K10'] = result
        print(f"  {label:18s} K=10: Succ={result['success_rate']:.0f}%, "
              f"TErr={result['terminal_error_mean']:.4f}m, "
              f"IAE={result['iae_mean']:.1f}")

    # ====================================================================
    # 实验2: 修复后 Q-head 对 PD 候选的排序相关性
    # ====================================================================
    print("\n" + "=" * 80)
    print("实验2: 修复后 Q-head 排序相关性 — PD候选 vs TRM候选")
    print("=" * 80)

    from scipy.stats import spearmanr, pearsonr

    N_SAMPLES = 100
    K_CORR = 50

    q_pd_scores_all = []
    pd_rollout_costs_all = []
    q_trm_scores_all = []
    trm_rollout_costs_all = []

    for sample_idx in range(N_SAMPLES):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)

        with torch.no_grad():
            env_temp = QuadrotorDynamics()
            e_p = x_sp_dev[0:3] - x_init[0:3]
            e_v = x_sp_dev[3:6] - x_init[3:6]
            u_pd = env_temp.m * (4.0 * e_p + 3.0 * e_v)

            # --- PD候选 ---
            noise_pd = torch.randn(K_CORR, 3) * 2.0
            u_first_pd = u_pd.unsqueeze(0) + noise_pd
            u_pd_full = u_pd.unsqueeze(0).repeat(K_CORR, 1)
            u_pd_seq = torch.zeros(K_CORR, 30)
            u_pd_seq[:, 0:3] = u_first_pd
            for i in range(10):
                u_pd_seq[:, i*3:(i+1)*3] = u_pd_full if i > 0 else u_first_pd

            # PD候选的rollout代价
            pd_costs = []
            q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
            for k_idx in range(K_CORR):
                x_r = x_init.cpu().clone()
                cost = 0.0
                for s in range(20):
                    if s == 0:
                        u_s = torch.clamp(u_first_pd[k_idx], env_temp.u_min, env_temp.u_max)
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

            # 修复后：Q-head评分基于PD候选（通过W_y注入）
            X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
            X_parallel = X_single.repeat(K_CORR, 1)
            y_history = trm_model.forward_steps(X_parallel, D=16, noise_scale=0.25,
                                                 u_seq_external=u_pd_seq.to(device))
            _, final_latent_y = y_history[-1]
            q_scores_pd = trm_model.f_Q(final_latent_y).squeeze(-1).cpu().numpy()

            q_pd_scores_all.extend(q_scores_pd.tolist())
            pd_rollout_costs_all.extend(pd_costs)

            # --- TRM候选（无外部输入，标准模式）---
            y_history_trm = trm_model.forward_steps(X_parallel, D=16, noise_scale=0.25)
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
    rho_pd, p_pd = spearmanr(np.array(q_pd_scores_all), np.array(pd_rollout_costs_all))
    r_pd, rp_pd = pearsonr(np.array(q_pd_scores_all), np.array(pd_rollout_costs_all))
    rho_trm, p_trm = spearmanr(np.array(q_trm_scores_all), np.array(trm_rollout_costs_all))
    r_trm, rp_trm = pearsonr(np.array(q_trm_scores_all), np.array(trm_rollout_costs_all))

    print(f"\n  修复后 Q-head 对 PD 候选排序相关性:")
    print(f"    Spearman ρ = {rho_pd:.4f} (p={p_pd:.2e})")
    print(f"    Pearson  r = {r_pd:.4f} (p={rp_pd:.2e})")
    print(f"\n  Q-head 对 TRM 候选排序相关性（标准模式）:")
    print(f"    Spearman ρ = {rho_trm:.4f} (p={p_trm:.2e})")
    print(f"    Pearson  r = {r_trm:.4f} (p={rp_trm:.2e})")

    # Simple Encoder 对比
    q_simple_scores_all = []
    pd_simple_costs_all = []
    for sample_idx in range(N_SAMPLES):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)
        with torch.no_grad():
            env_temp = QuadrotorDynamics()
            e_p = x_sp_dev[0:3] - x_init[0:3]
            e_v = x_sp_dev[3:6] - x_init[3:6]
            u_pd = env_temp.m * (4.0 * e_p + 3.0 * e_v)
            noise_pd = torch.randn(K_CORR, 3) * 2.0
            u_first_pd = u_pd.unsqueeze(0) + noise_pd
            X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
            X_parallel = X_single.repeat(K_CORR, 1)
            X_perturbed = X_parallel + torch.randn_like(X_parallel) * 0.025
            _, q_simple = simple_model(X_perturbed)
            q_scores_simple = q_simple.squeeze(-1).cpu().numpy()
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

    rho_simple, p_simple = spearmanr(np.array(q_simple_scores_all), np.array(pd_simple_costs_all))
    r_simple, rp_simple = pearsonr(np.array(q_simple_scores_all), np.array(pd_simple_costs_all))
    print(f"\n  Simple Encoder 对 PD 候选排序相关性:")
    print(f"    Spearman ρ = {rho_simple:.4f} (p={p_simple:.2e})")

    # ====================================================================
    # 实验3: 修复后 Simple Encoder 消融
    # ====================================================================
    print("\n" + "=" * 80)
    print("实验3: 修复后编码器消融 (K=50, Strong CBF)")
    print("=" * 80)

    # TRM Q-head (修复后)
    set_seed(SEED)
    ptrm_predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                        alpha_blend=0.3, candidate_mode='pd',
                                        pd_sigma=2.0, use_rollout_cost=True,
                                        ranking_mode='q_head')
    trm_result = run_mc_trials(env, ptrm_predictor, X_SP, enable_cbf=True)
    print(f"  TRM Q-head(修复): Succ={trm_result['success_rate']:.0f}%, "
          f"TErr={trm_result['terminal_error_mean']:.4f}m, IAE={trm_result['iae_mean']:.1f}")

    # Simple Encoder
    set_seed(SEED)
    simple_predictor = SimpleEncoderPredictor(simple_model, env, K=50, sigma=0.25, pd_sigma=2.0)
    simple_result = run_mc_trials(env, simple_predictor, X_SP, enable_cbf=True,
                                   predictor_type='simple_encoder')
    print(f"  SimpleEncoder:     Succ={simple_result['success_rate']:.0f}%, "
          f"TErr={simple_result['terminal_error_mean']:.4f}m, IAE={simple_result['iae_mean']:.1f}")

    # Random
    random_result = exp1_results['随机排序']
    print(f"  Random排序:        Succ={random_result['success_rate']:.0f}%, "
          f"TErr={random_result['terminal_error_mean']:.4f}m, IAE={random_result['iae_mean']:.1f}")

    # Rollout-All
    rollout_result = exp1_results['全量Rollout排序']
    print(f"  全量Rollout:       Succ={rollout_result['success_rate']:.0f}%, "
          f"TErr={rollout_result['terminal_error_mean']:.4f}m, IAE={rollout_result['iae_mean']:.1f}")

    # ====================================================================
    # 与修复前结果对比
    # ====================================================================
    print("\n" + "=" * 80)
    print("修复前后对比")
    print("=" * 80)

    # 加载修复前结果
    prev_path = os.path.join(save_dir, 'path_a_validation_results.json')
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)
        prev_exp1 = prev['exp1_ranking_comparison']
        prev_exp2 = prev['exp2_qhead_correlation']

        print("\n--- 实验1: 排序模式对比 ---")
        print(f"  {'':20s} {'修复前IAE':>10s} {'修复后IAE':>10s} {'差异':>8s}")
        for label, new_key in [('Q-head排序', 'Q-head排序(修复)'), ('随机排序', '随机排序')]:
            old_iae = prev_exp1[label]['iae_mean']
            new_iae = exp1_results[new_key]['iae_mean']
            diff = new_iae - old_iae
            print(f"  {label:20s} {old_iae:10.1f} {new_iae:10.1f} {diff:+8.1f}")

        print(f"\n--- 实验2: Q-head对PD候选排序相关性 ---")
        print(f"  修复前: ρ = {prev_exp2['q_head_vs_pd']['spearman_rho']:.4f}")
        print(f"  修复后: ρ = {rho_pd:.4f}")
        print(f"  变化: Δρ = {rho_pd - prev_exp2['q_head_vs_pd']['spearman_rho']:.4f}")
    else:
        print("  未找到修复前结果文件，跳过对比")

    # ====================================================================
    # 结论
    # ====================================================================
    print("\n" + "=" * 80)
    print("架构修复验证结论")
    print("=" * 80)

    qh_iae = exp1_results['Q-head排序(修复)']['iae_mean']
    rand_iae = exp1_results['随机排序']['iae_mean']
    rollout_iae = exp1_results['全量Rollout排序']['iae_mean']
    iae_diff_pct = (rand_iae - qh_iae) / qh_iae * 100

    print(f"\n  1. Q-head vs 随机排序 IAE差异: {iae_diff_pct:+.1f}%")
    if iae_diff_pct > 3:
        print(f"     ✓ Q-head排序优于随机排序，修复有效")
    elif iae_diff_pct > 0:
        print(f"     △ Q-head排序略优于随机排序，优势微弱")
    else:
        print(f"     ✗ Q-head排序仍不优于随机排序")

    print(f"\n  2. Q-head对PD候选排序相关性: ρ = {rho_pd:.4f}")
    if abs(rho_pd) > 0.3:
        print(f"     ✓ 相关性显著，Q-head能有效评估PD候选")
    elif abs(rho_pd) > 0.15:
        print(f"     △ 相关性中等，Q-head部分有效")
    else:
        print(f"     ✗ 相关性弱，Q-head无法有效评估PD候选")

    iae_trm_vs_simple = (simple_result['iae_mean'] - trm_result['iae_mean']) / trm_result['iae_mean'] * 100
    print(f"\n  3. TRM Q-head vs SimpleEncoder: ΔIAE = {iae_trm_vs_simple:+.1f}%")
    if iae_trm_vs_simple > 3:
        print(f"     ✓ TRM递归结构为Q-head提供独特价值")
    elif iae_trm_vs_simple > 0:
        print(f"     △ TRM略有优势")
    else:
        print(f"     ✗ TRM无优势")

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
        'exp2_qhead_correlation': {
            'q_head_vs_pd_fixed': {'spearman_rho': float(rho_pd), 'pearson_r': float(r_pd)},
            'q_head_vs_trm': {'spearman_rho': float(rho_trm), 'pearson_r': float(r_trm)},
            'simple_encoder_vs_pd': {'spearman_rho': float(rho_simple)},
        },
        'exp3_encoder_ablation': {
            'trm_qhead_fixed': strip(trm_result),
            'simple_encoder': strip(simple_result),
            'random': strip(random_result),
            'rollout_all': strip(rollout_result),
        }
    }

    results_path = os.path.join(save_dir, 'path_b_fix_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至 {results_path}")


if __name__ == '__main__':
    main()
