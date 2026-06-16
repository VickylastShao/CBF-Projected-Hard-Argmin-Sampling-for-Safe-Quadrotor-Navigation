# -*- coding: utf-8 -*-
"""
路径C验证：候选条件化TRM训练 + 完整实验验证

流程：
1. 生成专家数据集
2. 使用 train_trm_candidate_conditioned() 训练 v2 模型
3. 运行4组验证实验
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
    train_trm_candidate_conditioned, train_simple_encoder_qhead,
    evaluate_batch_decoded_trajectory_cost
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
    """简单编码器Q-head预测器（消融对照）"""

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

    # ==================================================================
    # Phase 1: 训练 v2 模型（候选条件化训练）
    # ==================================================================
    print("=" * 80)
    print("Phase 1: 候选条件化TRM训练")
    print("=" * 80)

    env = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env)

    # 生成数据集（与v6相同，500样本）
    x_sp_train = X_SP.clone()
    dataset = generate_quadrotor_dataset(
        env, solver, size=500,
        x_sp=x_sp_train,
        pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)]
    )
    print(f"数据集生成完成: {len(dataset)} 样本")

    # 初始化 v2 模型
    trm_v2 = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    print(f"TRM v2 参数量: {sum(p.numel() for p in trm_v2.parameters())}")

    # 候选条件化训练
    trm_v2, train_history = train_trm_candidate_conditioned(
        trm_v2, dataset, env,
        epochs=80,
        batch_size=32,
        lr=0.002,
        gamma=0.95,
        lambda_Q=0.3,        # 比标准训练增大（Q-head任务更难）
        V_max=150.0,
        K_train=10,           # 每样本生成10个候选
        output_noise_sigma=2.0,  # 与推理时匹配
        patience=25,
        verbose=True
    )

    # 保存 v2 模型
    v2_path = os.path.join(save_dir, 'trm_model_v2.pt')
    torch.save(trm_v2.state_dict(), v2_path)
    print(f"\nv2 模型已保存至 {v2_path}")

    # 同时训练 SimpleEncoder 对照（用相同数据集）
    print("\n--- 训练 SimpleEncoder Q-head 对照 ---")
    simple_model = SimpleEncoderQHead(input_dim=12, latent_dim=64).to(device)
    simple_model, _ = train_simple_encoder_qhead(
        simple_model, dataset, env,
        epochs=50, batch_size=32, lr=0.002, V_max=150.0,
        patience=20, verbose=True
    )
    simple_path = os.path.join(save_dir, 'simple_encoder_model_v2.pt')
    torch.save(simple_model.state_dict(), simple_path)
    print(f"SimpleEncoder v2 模型已保存至 {simple_path}")

    # ==================================================================
    # Phase 2: 验证实验
    # ==================================================================
    print("\n" + "=" * 80)
    print("Phase 2: 验证实验")
    print("=" * 80)

    trm_v2.eval()

    # ------------------------------------------------------------------
    # 实验1: 性能对比 (TRM-v2 vs MPPI vs 原PTRM vs PD+CBF)
    # ------------------------------------------------------------------
    print("\n--- 实验1: 控制性能对比 ---")

    # 加载原v1模型用于对比
    trm_v1 = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    v1_path = os.path.join(save_dir, 'trm_model.pt')
    if os.path.exists(v1_path):
        trm_v1.load_state_dict(torch.load(v1_path, map_location=device, weights_only=True))
        trm_v1.eval()

    exp1_results = {}

    # TRM-v2 (candidate_mode='trm_v2')
    set_seed(SEED)
    pred_v2 = PTRMNMPCPredictor(trm_v2, env, K=50, D=16, sigma=0.25,
                                 candidate_mode='trm_v2',
                                 pd_sigma=2.0,
                                 use_rollout_cost=True, ranking_mode='q_head')
    r = run_mc_trials(env, pred_v2, X_SP, enable_cbf=True)
    exp1_results['TRM-v2 (K=50)'] = r
    print(f"  TRM-v2 K=50:  Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # TRM-v2 K=10
    set_seed(SEED)
    pred_v2_k10 = PTRMNMPCPredictor(trm_v2, env, K=10, D=16, sigma=0.25,
                                      candidate_mode='trm_v2',
                                      pd_sigma=2.0,
                                      use_rollout_cost=True, ranking_mode='q_head')
    r = run_mc_trials(env, pred_v2_k10, X_SP, enable_cbf=True)
    exp1_results['TRM-v2 (K=10)'] = r
    print(f"  TRM-v2 K=10:  Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # TRM-v2 K=1 (确定性基线)
    set_seed(SEED)
    pred_v2_k1 = PTRMNMPCPredictor(trm_v2, env, K=1, D=16, sigma=0.0,
                                     candidate_mode='trm_v2',
                                     use_rollout_cost=False, ranking_mode='q_head')
    r = run_mc_trials(env, pred_v2_k1, X_SP, enable_cbf=True)
    exp1_results['TRM-v2 (K=1)'] = r
    print(f"  TRM-v2 K=1:   Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # 原PTRM (PD+TRM-Eval, v1模型)
    if os.path.exists(v1_path):
        set_seed(SEED)
        pred_v1 = PTRMNMPCPredictor(trm_v1, env, K=50, D=16, sigma=0.25,
                                     alpha_blend=0.3, candidate_mode='pd',
                                     pd_sigma=2.0, use_rollout_cost=True)
        r = run_mc_trials(env, pred_v1, X_SP, enable_cbf=True)
        exp1_results['PTRM-v1 (K=50)'] = r
        print(f"  PTRM-v1 K=50: Succ={r['success_rate']:.0f}%, TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # ------------------------------------------------------------------
    # 实验2: Q-head 同一样本内排序相关性（关键指标！）
    # ------------------------------------------------------------------
    print("\n--- 实验2: Q-head 同一样本内排序相关性 ---")

    from scipy.stats import spearmanr, pearsonr

    N_SAMPLES = 50
    K_CORR = 50

    # TRM-v2 的 Q-head 排序相关性
    v2_rhos = []
    v2_r_within = []

    for sample_idx in range(N_SAMPLES):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)

        with torch.no_grad():
            X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
            X_K = X_single.repeat(K_CORR, 1)

            # 确定性推理获取基线
            y_det = trm_v2.forward_steps(X_single, D=16, noise_scale=0.0, noise_mode='none')
            u_base = y_det[-1][0].squeeze()

            # 生成K个输出噪声候选
            u_cands = u_base.unsqueeze(0).repeat(K_CORR, 1)
            first_noise = torch.randn(K_CORR, 3) * 2.0
            u_cands[:, 0:3] = u_cands[:, 0:3] + first_noise
            for si in range(1, 10):
                decay = max(0.3, 1.0 - si * 0.1)
                u_cands[:, si*3:(si+1)*3] = u_cands[:, si*3:(si+1)*3] + torch.randn(K_CORR, 3) * 2.0 * decay

            # Q-head评估
            y_c = trm_v2.forward_steps(X_K, D=16, noise_scale=0.0, noise_mode='none',
                                        u_seq_external=u_cands)
            q_scores = trm_v2.f_Q(y_c[-1][1]).squeeze(-1).cpu().numpy()

            # Rollout cost（用第一步+PD反馈，与推理一致）
            q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
            costs = []
            for k in range(K_CORR):
                x_r = x_init.cpu().clone()
                cost = 0.0
                for s in range(20):
                    if s == 0:
                        u_s = torch.clamp(u_cands[k, 0:3].cpu(), -15.0, 15.0)
                    else:
                        e_p_s = X_SP[0:3] - x_r[0:3]
                        e_v_s = X_SP[3:6] - x_r[3:6]
                        u_s = torch.clamp(env.m * (4.0 * e_p_s + 3.0 * e_v_s), -15.0, 15.0)
                    x_r = env.step_discrete(x_r, u_s)
                    err = x_r - X_SP
                    cost += torch.sum(q_diag * err * err).item() + 0.02 * torch.sum(u_s * u_s).item()
                    for obs in env.obstacles:
                        d = torch.norm(x_r[0:3] - torch.tensor(obs['p'])) - obs['r']
                        cost += 2000.0 * max(0.0, 0.3 - d.item()) ** 2
                costs.append(cost)

            rho, _ = spearmanr(q_scores, costs)
            r_w, _ = pearsonr(q_scores, costs)
            if not np.isnan(rho):
                v2_rhos.append(rho)
                v2_r_within.append(r_w)

        if (sample_idx + 1) % 10 == 0:
            print(f"  采样进度: {sample_idx+1}/{N_SAMPLES}, 当前平均ρ={np.mean(v2_rhos):.4f}")

    v2_rho_arr = np.array(v2_rhos)
    print(f"\n  TRM-v2 Q-head 同一样本内排序相关性:")
    print(f"    Spearman ρ: mean={v2_rho_arr.mean():.4f}, std={v2_rho_arr.std():.4f}, "
          f"min={v2_rho_arr.min():.4f}, max={v2_rho_arr.max():.4f}")

    # 对比：v1模型的Q-head排序能力
    v1_rhos = []
    if os.path.exists(v1_path):
        for sample_idx in range(N_SAMPLES):
            x_init = random_x_init().to(device)
            x_sp_dev = X_SP.to(device)

            with torch.no_grad():
                X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
                X_K = X_single.repeat(K_CORR, 1)

                # v1: TRM噪声推理
                y = trm_v1.forward_steps(X_K, D=16, noise_scale=0.25, noise_mode='both')
                q_scores = trm_v1.f_Q(y[-1][1]).squeeze(-1).cpu().numpy()
                u_cands = y[-1][0]

                q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
                costs = []
                for k in range(K_CORR):
                    x_r = x_init.cpu().clone()
                    cost = 0.0
                    for s in range(20):
                        if s == 0:
                            u_s = torch.clamp(u_cands[k, 0:3].cpu(), -15.0, 15.0)
                        else:
                            e_p_s = X_SP[0:3] - x_r[0:3]
                            e_v_s = X_SP[3:6] - x_r[3:6]
                            u_s = torch.clamp(env.m * (4.0 * e_p_s + 3.0 * e_v_s), -15.0, 15.0)
                        x_r = env.step_discrete(x_r, u_s)
                        err = x_r - X_SP
                        cost += torch.sum(q_diag * err * err).item() + 0.02 * torch.sum(u_s * u_s).item()
                        for obs in env.obstacles:
                            d = torch.norm(x_r[0:3] - torch.tensor(obs['p'])) - obs['r']
                            cost += 2000.0 * max(0.0, 0.3 - d.item()) ** 2
                    costs.append(cost)

                rho, _ = spearmanr(q_scores, costs)
                if not np.isnan(rho):
                    v1_rhos.append(rho)

        v1_rho_arr = np.array(v1_rhos)
        print(f"\n  PTRM-v1 Q-head 同一样本内排序相关性 (TRM噪声推理):")
        print(f"    Spearman ρ: mean={v1_rho_arr.mean():.4f}, std={v1_rho_arr.std():.4f}")

    # ------------------------------------------------------------------
    # 实验3: Q-head vs 随机 vs 全量rollout排序对比
    # ------------------------------------------------------------------
    print("\n--- 实验3: 排序模式对比 ---")

    for ranking_mode, label in [
        ('q_head', 'Q-head排序(v2)'),
        ('random', '随机排序'),
        ('rollout_all', '全量Rollout排序')
    ]:
        set_seed(SEED)
        pred = PTRMNMPCPredictor(trm_v2, env, K=50, D=16, sigma=0.25,
                                  candidate_mode='trm_v2',
                                  pd_sigma=2.0,
                                  use_rollout_cost=True,
                                  ranking_mode=ranking_mode)
        r = run_mc_trials(env, pred, X_SP, enable_cbf=True)
        exp1_results[label] = r
        print(f"  {label:20s}: Succ={r['success_rate']:.0f}%, "
              f"TErr={r['terminal_error_mean']:.4f}m, IAE={r['iae_mean']:.1f}")

    # ------------------------------------------------------------------
    # 实验4: 编码器消融 (TRM-v2 vs SimpleEncoder)
    # ------------------------------------------------------------------
    print("\n--- 实验4: 编码器消融 ---")

    # TRM-v2 Q-head
    set_seed(SEED)
    pred_trm_v2 = PTRMNMPCPredictor(trm_v2, env, K=50, D=16, sigma=0.25,
                                      candidate_mode='trm_v2',
                                      pd_sigma=2.0,
                                      use_rollout_cost=True)
    r_trm = run_mc_trials(env, pred_trm_v2, X_SP, enable_cbf=True)
    print(f"  TRM-v2 Q-head:  Succ={r_trm['success_rate']:.0f}%, "
          f"TErr={r_trm['terminal_error_mean']:.4f}m, IAE={r_trm['iae_mean']:.1f}")

    # SimpleEncoder
    set_seed(SEED)
    simple_pred = SimpleEncoderPredictor(simple_model, env, K=50, sigma=0.25, pd_sigma=2.0)
    r_simple = run_mc_trials(env, simple_pred, X_SP, enable_cbf=True, predictor_type='simple_encoder')
    print(f"  SimpleEncoder:   Succ={r_simple['success_rate']:.0f}%, "
          f"TErr={r_simple['terminal_error_mean']:.4f}m, IAE={r_simple['iae_mean']:.1f}")

    # SimpleEncoder 排序相关性
    se_rhos = []
    for sample_idx in range(N_SAMPLES):
        x_init = random_x_init().to(device)
        x_sp_dev = X_SP.to(device)
        with torch.no_grad():
            e_p = x_sp_dev[0:3] - x_init[0:3]
            e_v = x_sp_dev[3:6] - x_init[3:6]
            u_pd = env.m * (4.0 * e_p + 3.0 * e_v)
            noise_pd = torch.randn(K_CORR, 3) * 2.0
            u_first_pd = u_pd.unsqueeze(0) + noise_pd
            X_single = torch.cat([x_init, x_sp_dev]).unsqueeze(0)
            X_parallel = X_single.repeat(K_CORR, 1)
            X_perturbed = X_parallel + torch.randn_like(X_parallel) * 0.025
            _, q_simple = simple_model(X_perturbed)
            q_scores = q_simple.squeeze(-1).cpu().numpy()
            q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
            costs = []
            for k in range(K_CORR):
                x_r = x_init.cpu().clone()
                cost = 0.0
                for s in range(20):
                    if s == 0:
                        u_s = torch.clamp(u_first_pd[k].cpu(), -15.0, 15.0)
                    else:
                        e_p_s = X_SP[0:3] - x_r[0:3]
                        e_v_s = X_SP[3:6] - x_r[3:6]
                        u_s = torch.clamp(env.m * (4.0 * e_p_s + 3.0 * e_v_s), -15.0, 15.0)
                    x_r = env.step_discrete(x_r, u_s)
                    err = x_r - X_SP
                    cost += torch.sum(q_diag * err * err).item() + 0.02 * torch.sum(u_s * u_s).item()
                    for obs in env.obstacles:
                        d = torch.norm(x_r[0:3] - torch.tensor(obs['p'])) - obs['r']
                        cost += 2000.0 * max(0.0, 0.3 - d.item()) ** 2
                costs.append(cost)
            rho, _ = spearmanr(q_scores, costs)
            if not np.isnan(rho):
                se_rhos.append(rho)

    se_rho_arr = np.array(se_rhos)
    print(f"\n  SimpleEncoder 同一样本内排序相关性:")
    print(f"    Spearman ρ: mean={se_rho_arr.mean():.4f}, std={se_rho_arr.std():.4f}")

    # ------------------------------------------------------------------
    # 最终结论
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("路径C验证结论")
    print("=" * 80)

    qh_iae = exp1_results.get('Q-head排序(v2)', {}).get('iae_mean', 999)
    rand_iae = exp1_results.get('随机排序', {}).get('iae_mean', 999)
    rollout_iae = exp1_results.get('全量Rollout排序', {}).get('iae_mean', 999)

    print(f"\n  1. Q-head vs 随机排序:")
    if rand_iae > 0 and qh_iae > 0:
        diff_pct = (rand_iae - qh_iae) / qh_iae * 100
        print(f"     IAE差异: {diff_pct:+.1f}%")
        if diff_pct > 3:
            print(f"     ✓ Q-head排序优于随机排序")
        elif diff_pct > 0:
            print(f"     △ Q-head排序略优")
        else:
            print(f"     ✗ Q-head排序不优于随机")

    print(f"\n  2. Q-head同一样本内排序相关性:")
    print(f"     TRM-v2: ρ = {v2_rho_arr.mean():.4f}")
    print(f"     v1模型: ρ = {v1_rho_arr.mean():.4f}" if len(v1_rhos) > 0 else "")
    print(f"     SimpleEncoder: ρ = {se_rho_arr.mean():.4f}")
    if abs(v2_rho_arr.mean()) > 0.3:
        print(f"     ✓ TRM-v2 Q-head能有效排序候选")
    elif abs(v2_rho_arr.mean()) > 0.15:
        print(f"     △ TRM-v2 Q-head排序能力中等")
    else:
        print(f"     ✗ TRM-v2 Q-head无法有效排序候选")

    print(f"\n  3. TRM-v2 vs SimpleEncoder:")
    if r_trm['iae_mean'] > 0 and r_simple['iae_mean'] > 0:
        trm_vs_se = (r_simple['iae_mean'] - r_trm['iae_mean']) / r_trm['iae_mean'] * 100
        print(f"     ΔIAE = {trm_vs_se:+.1f}%")
        if trm_vs_se > 3:
            print(f"     ✓ TRM递归结构+候选条件化Q-head有独特价值")
        elif trm_vs_se > 0:
            print(f"     △ TRM略有优势")
        else:
            print(f"     ✗ TRM无优势")

    print(f"\n  4. W_y编码候选 vs SimpleEncoder无候选信息:")
    print(f"     TRM-v2 ρ = {v2_rho_arr.mean():.4f} (W_y编码30维候选)")
    print(f"     SimpleEnc ρ = {se_rho_arr.mean():.4f} (仅12维状态，无候选信息)")
    if abs(v2_rho_arr.mean()) > abs(se_rho_arr.mean()) + 0.1:
        print(f"     ✓ W_y编码候选信息显著提升Q-head排序能力")
    else:
        print(f"     △ W_y编码候选信息未显著提升排序能力")

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
        'exp1_performance_comparison': strip(exp1_results),
        'exp2_qhead_ranking': {
            'trm_v2_within_sample_rho': {
                'mean': float(v2_rho_arr.mean()),
                'std': float(v2_rho_arr.std()),
                'min': float(v2_rho_arr.min()),
                'max': float(v2_rho_arr.max()),
            },
            'v1_within_sample_rho': {
                'mean': float(v1_rho_arr.mean()),
                'std': float(v1_rho_arr.std()),
            } if len(v1_rhos) > 0 else None,
            'simple_encoder_within_sample_rho': {
                'mean': float(se_rho_arr.mean()),
                'std': float(se_rho_arr.std()),
            },
        },
        'exp3_ranking_comparison': strip({
            k: exp1_results[k] for k in ['Q-head排序(v2)', '随机排序', '全量Rollout排序']
            if k in exp1_results
        }),
        'exp4_encoder_ablation': {
            'trm_v2': strip(r_trm),
            'simple_encoder': strip(r_simple),
        },
        'training_history': {
            'q_rank_correlation': [float(x) for x in train_history.get('q_rank_correlation', [])],
        }
    }

    results_path = os.path.join(save_dir, 'path_c_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至 {results_path}")


if __name__ == '__main__':
    main()
