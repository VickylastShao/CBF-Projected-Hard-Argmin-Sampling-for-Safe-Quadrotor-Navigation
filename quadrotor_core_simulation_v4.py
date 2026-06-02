# -*- coding: utf-8 -*-
"""
PTRM-NMPC 论文投稿级实验 v4（最终修正版）

v3→v4 关键修正：
  1. 目标[2,2,2]恰在障碍物2表面上 → 改用x_sp=[2,3,2]（y>2，需穿越障碍物通道）
  2. 训练数据x_sp=[3,3,3]与测试不一致 → 训练数据也用[2,3,2]
  3. 初始位置在y<0一侧 → 必须穿过三个球体障碍物形成的非凸通道到达y>2的目标
  4. CBF完全主导 → 添加弱CBF实验(alpha_d=0.3)，让K有机会展示安全增益
  5. sigma=0.5多样性不足 → sigma=0.8 + 更宽output perturbation
  6. 仿真300步(6s) → 确保穿越通道有足够时间

实验结构：
  Exp I:  标准避障（全部强CBF）— 展示tracking性能
  Exp II: 弱CBF避障（alpha_d=0.3）— 核心实验，展示K的安全增益
  Exp III: 参数失配鲁棒性（强CBF）
  Exp IV: 无CBF安全性比较 — 展示K的内在安全增益
  Exp V:  消融实验（K / σ / D）
  Exp VI: 计算效率与宽度缩放
"""

import torch
import torch.nn as nn
import torch.optim as optim
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


# ==============================================================
# 关键修正：Q-head 多样性微调
# ==============================================================
def finetune_qhead_with_diversity(model, dataset, env, epochs=60, batch_size=32,
                                   lr=5e-4, K_diverse=20, D_steps=16,
                                   sigma_diverse=0.5, lambda_Q=1.0, V_max=5000.0,
                                   val_ratio=0.2, patience=15, verbose=True):
    """
    Q-head 多样性微调阶段。

    核心思路：对每个训练样本，生成 K_diverse 个带噪声候选，
    计算每个候选的真实轨迹代价，训练 Q-head 预测 (V_max - cost)。
    这使得 Q-head 学会在同一状态下区分不同候选的优劣。
    """
    device = next(model.parameters()).device

    # 冻结策略网络，只训练 Q-head
    for name, param in model.named_parameters():
        if 'f_Q' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = optim.Adam(model.f_Q.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 数据拆分
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:dataset_size - val_size]
    val_indices = indices[dataset_size - val_size:]

    X_all = torch.stack([d[0] for d in dataset]).to(device)

    X_train = X_all[train_indices]
    X_val = X_all[val_indices]

    best_val_loss = float('inf')
    best_q_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for name, param in model.named_parameters():
            if 'f_Q' not in name:
                param.requires_grad = False

        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_x = X_train[batch_idx]
            bs = batch_x.shape[0]

            X_expanded = batch_x.unsqueeze(1).repeat(1, K_diverse, 1).reshape(bs * K_diverse, -1)

            optimizer.zero_grad()
            with torch.no_grad():
                y_history = model.forward_steps(X_expanded, D=D_steps, noise_scale=sigma_diverse)
                final_u_seq, final_latent_y = y_history[-1]

            q_predicted = model.f_Q(final_latent_y).squeeze(-1)

            with torch.no_grad():
                decoded_cost = evaluate_batch_decoded_trajectory_cost(env, X_expanded, final_u_seq)
                q_target = torch.clamp(V_max - decoded_cost, min=0.0)

            loss = mse_loss(q_predicted, q_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # 验证
        model.eval()
        with torch.no_grad():
            X_val_expanded = X_val.unsqueeze(1).repeat(1, K_diverse, 1).reshape(len(X_val) * K_diverse, -1)
            y_val = model.forward_steps(X_val_expanded, D=D_steps, noise_scale=sigma_diverse)
            _, val_latent = y_val[-1]
            q_val_pred = model.f_Q(val_latent).squeeze(-1)
            val_cost = evaluate_batch_decoded_trajectory_cost(env, X_val_expanded, y_val[-1][0])
            q_val_target = torch.clamp(V_max - val_cost, min=0.0)
            val_loss = mse_loss(q_val_pred, q_val_target).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_q_state = {k: v.clone() for k, v in model.f_Q.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Q-微调 Epoch [{epoch+1}/{epochs}] | 训练损失: {epoch_loss/n_batches:.4f} | "
                  f"验证损失: {val_loss:.4f} | 早停: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Q-微调早停：连续 {patience} 轮未改善")
            break

    if best_q_state is not None:
        model.f_Q.load_state_dict(best_q_state)

    for param in model.parameters():
        param.requires_grad = True

    if verbose:
        print(f"  Q-head 多样性微调完成。最佳验证损失: {best_val_loss:.4f}")

    return model


# ==============================================================
# 实验辅助函数
# ==============================================================
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


class NMPCWrapper:
    """NMPC 适配器"""
    def __init__(self, solver, env):
        self.solver = solver
        self.env = env
        self.last_u_seq = None
    def reset(self):
        self.last_u_seq = None
    def predict_action(self, x, x_sp, enable_cbf=True):
        u_full = self.solver.solve(x, x_sp)
        u_nominal = u_full[0:3]
        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x, u_nominal)
        else:
            u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
        u_seq = torch.zeros(30, dtype=torch.float32)
        u_seq[0:3] = u_safe
        return u_safe, u_seq


def run_single_trial(env, predictor, x_init, x_sp, sim_steps, enable_cbf=True,
                     use_mismatch=False, process_noise=0.008):
    """运行单次闭环仿真，返回详细指标"""
    predictor.reset()
    x_curr = x_init.clone()
    p_iae, v_iae = 0.0, 0.0
    collision_flag = False
    cbf_intervention_count = 0
    min_obstacle_dist = float('inf')
    step_costs = []

    q_diag = np.array([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
    R_cost = 0.02

    for step in range(sim_steps):
        u, u_seq = predictor.predict_action(x_curr, x_sp, enable_cbf=enable_cbf)

        # CBF 介入检测
        u_clamp = torch.clamp(u_seq[0:3].cpu(), env.u_min, env.u_max)
        if torch.norm(u.cpu() - u_clamp) > 1e-4:
            cbf_intervention_count += 1

        x_curr = env.step_discrete(x_curr, u, use_mismatch=use_mismatch, process_noise=process_noise)

        # 累计指标
        err = x_curr[0:6].detach().cpu().numpy() - x_sp[0:6].detach().cpu().numpy()
        step_cost = float(np.dot(q_diag * err, err) + R_cost * float(torch.sum(u.cpu()**2)))
        step_costs.append(step_cost)

        p_iae += np.linalg.norm(x_curr[0:3].detach().cpu().numpy() - x_sp[0:3].detach().cpu().numpy()) * env.dt
        v_iae += np.linalg.norm(x_curr[3:6].detach().cpu().numpy()) * env.dt

        p_np = x_curr[0:3].detach().cpu().numpy()
        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs["p"]) - obs["r"]
            min_obstacle_dist = min(min_obstacle_dist, d)
            if d < 0:
                collision_flag = True

    terminal_err = np.linalg.norm(x_curr[0:3].detach().cpu().numpy() - x_sp[0:3].detach().cpu().numpy())
    success = (not collision_flag) and (terminal_err < 0.5)
    total_cost = sum(step_costs)

    return {
        'p_iae': float(p_iae),
        'v_iae': float(v_iae),
        'collision_rate': 100.0 if collision_flag else 0.0,
        'success_rate': 100.0 if success else 0.0,
        'terminal_err': float(terminal_err),
        'cbf_interventions': int(cbf_intervention_count),
        'min_obstacle_dist': float(min_obstacle_dist),
        'total_cost': float(total_cost),
    }


def summarize_trials(trial_results, num_trials):
    """汇总试验结果"""
    summary = {}
    for k in trial_results[0].keys():
        vals = [r[k] for r in trial_results]
        if k in ('collision_rate', 'success_rate'):
            summary[k] = float(np.mean(vals))
        else:
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
    return summary


def random_x_init():
    """生成随机初始状态（y<0一侧出发，必须穿越障碍物通道到达y>2的目标）"""
    return torch.tensor([
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(-1.0, 0.0),   # y<0：在障碍物墙下方
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4),     # y方向正向速度
        np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


# ==============================================================
# 主实验
# ==============================================================
def main():
    device = torch.device("cpu")
    print(f"设备: {device}")
    total_start = time.time()

    # ============================================================
    # 关键参数设定
    # ============================================================
    # 目标[2,3,2]：在障碍物墙上方(y>2)，必须穿越通道到达
    x_sp = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    SIM_STEPS = 250   # 5秒仿真
    NMPC_TRIALS = 10
    MAIN_TRIALS = 100

    env = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env, horizon=10)

    print(f"目标设定: {x_sp.numpy()}")
    print(f"仿真步数: {SIM_STEPS} ({SIM_STEPS * env.dt:.1f}s)")
    print(f"障碍物: ", [(o['p'].tolist(), o['r']) for o in env.obstacles])

    # 验证目标可达性
    for obs in env.obstacles:
        d = np.linalg.norm(x_sp[0:3].numpy() - obs['p']) - obs['r']
        print(f"  目标到障碍物{obs['p'].tolist()} r={obs['r']}: d-r = {d:.3f}m (>0.5m: {'✓' if d > 0.5 else '✗'})")

    # ============================================================
    # [1/7] 训练策略网络
    # ============================================================
    print("\n" + "="*60)
    print("[1/7] 阶段一：策略网络预训练")
    print("="*60)
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params} (期望 27,935)")
    assert total_params == 27935

    # 训练数据使用x_sp=[2,3,2]（与测试目标一致）
    # 初始位置范围覆盖通道两侧
    dataset = generate_quadrotor_dataset(
        env, solver, size=500,
        x_sp=x_sp,
        pos_range=[(-0.5, 1.5), (-1.0, 2.0), (-0.5, 1.5)]
    )
    model, history = train_trm_jointly(model, dataset, env, epochs=100, patience=15, verbose=True)

    # ============================================================
    # [2/7] Q-head 多样性微调
    # ============================================================
    print("\n" + "="*60)
    print("[2/7] 阶段二：Q-head 多样性微调")
    print("="*60)
    model = finetune_qhead_with_diversity(
        model, dataset, env,
        epochs=80, lr=5e-4, K_diverse=20, D_steps=16,
        sigma_diverse=0.8, patience=15, verbose=True
    )

    # 验证Q-head排序能力
    print("\n验证Q-head排序能力...")
    x_test = torch.tensor([0.0, -0.5, 0.0, 0.3, 0.3, 0.3, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0],
                          dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        X_rep = x_test.repeat(50, 1)
        y_h = model.forward_steps(X_rep, D=16, noise_scale=0.8)
        u_cands, y_final = y_h[-1]
        scores = model.f_Q(y_final).squeeze(-1)
        costs = evaluate_batch_decoded_trajectory_cost(env, X_rep, u_cands)

        rank_by_score = torch.argsort(scores, descending=True)
        rank_by_cost = torch.argsort(costs)
        best_by_score = rank_by_score[0].item()
        best_by_cost = rank_by_cost[0].item()
        cost_of_score_best = costs[best_by_score].item()
        cost_of_cost_best = costs[best_by_cost].item()
        cost_std = costs.std().item()
        cost_mean = costs.mean().item()

        print(f"  50候选代价: mean={cost_mean:.1f}, std={cost_std:.1f}")
        print(f"  Q-head最优候选代价: {cost_of_score_best:.1f}")
        print(f"  真实最优候选代价: {cost_of_cost_best:.1f}")
        print(f"  Q-head选择代价vs最优: {cost_of_score_best - cost_of_cost_best:.1f}")
        print(f"  代价多样性(std/mean): {cost_std/cost_mean*100:.1f}%")

    # 保存模型
    model_path = os.path.join(RESULTS_DIR, 'v4_trained_model.pt')
    torch.save(model.state_dict(), model_path)

    all_results = {}

    # ============================================================
    # [3/7] 实验一：标准避障（强CBF，alpha_d=0.8）
    # ============================================================
    print("\n" + "="*60)
    print("[3/7] 实验一：标准避障（强CBF，alpha_d=0.8）")
    print("="*60)

    exp1_configs = [
        ('NMPC+CBF',         NMPCWrapper(solver, env),         NMPC_TRIALS),
        ('DetTRM+CBF(K=1)',  PTRMNMPCPredictor(model, env, K=1, D=16, sigma=0.0),   MAIN_TRIALS),
        ('PTRM+CBF(K=10)',   PTRMNMPCPredictor(model, env, K=10, D=16, sigma=0.8),  MAIN_TRIALS),
        ('PTRM+CBF(K=50)',   PTRMNMPCPredictor(model, env, K=50, D=16, sigma=0.8),  MAIN_TRIALS),
        ('PTRM+CBF(K=100)',  PTRMNMPCPredictor(model, env, K=100, D=16, sigma=0.8), MAIN_TRIALS),
    ]

    exp1_results = {}
    for label, predictor, n_trials in exp1_configs:
        print(f"  {label} ({n_trials} trials)...", end="", flush=True)
        t0 = time.time()
        trials = []
        for _ in range(n_trials):
            x_init = random_x_init()
            r = run_single_trial(env, predictor, x_init, x_sp, SIM_STEPS, enable_cbf=True)
            trials.append(r)
        s = summarize_trials(trials, n_trials)
        exp1_results[label] = s
        dt = time.time() - t0
        print(f" Succ={s['success_rate']:.1f}% | Coll={s['collision_rate']:.1f}% | "
              f"IAE={s['p_iae_mean']:.2f}±{s['p_iae_std']:.2f} | "
              f"TErr={s['terminal_err_mean']:.3f}m | Cost={s['total_cost_mean']:.0f} | "
              f"CBF={s['cbf_interventions_mean']:.1f} | MinD={s['min_obstacle_dist_mean']:.3f}m | "
              f"[{dt:.0f}s]")
    all_results['exp1_standard'] = exp1_results

    # ============================================================
    # [4/7] 实验二：弱CBF避障（alpha_d=0.3）— 核心实验！
    # ============================================================
    print("\n" + "="*60)
    print("[4/7] 实验二：弱CBF避障（alpha_d=0.3 — 展示K的安全增益）")
    print("="*60)

    env_weak_cbf = QuadrotorDynamics()
    env_weak_cbf.alpha_d = 0.3
    env_weak_cbf.gamma_d = 0.1
    solver_weak = GoldenNMPCSolver(env_weak_cbf, horizon=10)

    exp2_configs = [
        ('NMPC+弱CBF',         NMPCWrapper(solver_weak, env_weak_cbf),    NMPC_TRIALS),
        ('DetTRM+弱CBF(K=1)',  PTRMNMPCPredictor(model, env_weak_cbf, K=1, D=16, sigma=0.0),   MAIN_TRIALS),
        ('PTRM+弱CBF(K=10)',   PTRMNMPCPredictor(model, env_weak_cbf, K=10, D=16, sigma=0.8),  MAIN_TRIALS),
        ('PTRM+弱CBF(K=50)',   PTRMNMPCPredictor(model, env_weak_cbf, K=50, D=16, sigma=0.8),  MAIN_TRIALS),
        ('PTRM+弱CBF(K=100)',  PTRMNMPCPredictor(model, env_weak_cbf, K=100, D=16, sigma=0.8), MAIN_TRIALS),
    ]

    exp2_results = {}
    for label, predictor, n_trials in exp2_configs:
        print(f"  {label} ({n_trials} trials)...", end="", flush=True)
        t0 = time.time()
        trials = []
        for _ in range(n_trials):
            x_init = random_x_init()
            r = run_single_trial(env_weak_cbf, predictor, x_init, x_sp, SIM_STEPS, enable_cbf=True)
            trials.append(r)
        s = summarize_trials(trials, n_trials)
        exp2_results[label] = s
        dt = time.time() - t0
        print(f" Succ={s['success_rate']:.1f}% | Coll={s['collision_rate']:.1f}% | "
              f"IAE={s['p_iae_mean']:.2f} | Cost={s['total_cost_mean']:.0f} | "
              f"MinD={s['min_obstacle_dist_mean']:.3f}m | [{dt:.0f}s]")
    all_results['exp2_weak_cbf'] = exp2_results

    # ============================================================
    # [5/7] 实验三：参数失配鲁棒性
    # ============================================================
    print("\n" + "="*60)
    print("[5/7] 实验三：+50%质量失配鲁棒性（强CBF）")
    print("="*60)

    exp3_configs = [
        ('DetTRM+CBF(K=1)', PTRMNMPCPredictor(model, env, K=1, D=16, sigma=0.0),  MAIN_TRIALS),
        ('PTRM+CBF(K=50)',  PTRMNMPCPredictor(model, env, K=50, D=16, sigma=0.8), MAIN_TRIALS),
    ]

    exp3_results = {}
    for label, predictor, n_trials in exp3_configs:
        print(f"  {label} ({n_trials} trials, mismatch)...", end="", flush=True)
        t0 = time.time()
        trials = []
        for _ in range(n_trials):
            x_init = random_x_init()
            r = run_single_trial(env, predictor, x_init, x_sp, SIM_STEPS,
                                enable_cbf=True, use_mismatch=True, process_noise=0.015)
            trials.append(r)
        s = summarize_trials(trials, n_trials)
        exp3_results[label] = s
        dt = time.time() - t0
        print(f" Succ={s['success_rate']:.1f}% | Coll={s['collision_rate']:.1f}% | "
              f"IAE={s['p_iae_mean']:.2f} | TErr={s['terminal_err_mean']:.3f}m | "
              f"Cost={s['total_cost_mean']:.0f} | [{dt:.0f}s]")
    all_results['exp3_mismatch'] = exp3_results

    # ============================================================
    # [5b/7] 实验四：无CBF安全性比较
    # ============================================================
    print("\n" + "="*60)
    print("[5b/7] 实验四：无CBF安全性比较（展示K的内在安全增益）")
    print("="*60)

    exp4_configs = [
        ('DetTRM-CBF(K=1)',   PTRMNMPCPredictor(model, env, K=1, D=16, sigma=0.0),  MAIN_TRIALS),
        ('PTRM-CBF(K=10)',    PTRMNMPCPredictor(model, env, K=10, D=16, sigma=0.8), MAIN_TRIALS),
        ('PTRM-CBF(K=50)',    PTRMNMPCPredictor(model, env, K=50, D=16, sigma=0.8), MAIN_TRIALS),
        ('PTRM-CBF(K=100)',   PTRMNMPCPredictor(model, env, K=100, D=16, sigma=0.8), MAIN_TRIALS),
    ]

    exp4_results = {}
    for label, predictor, n_trials in exp4_configs:
        print(f"  {label} ({n_trials} trials)...", end="", flush=True)
        t0 = time.time()
        trials = []
        for _ in range(n_trials):
            x_init = random_x_init()
            r = run_single_trial(env, predictor, x_init, x_sp, SIM_STEPS, enable_cbf=False)
            trials.append(r)
        s = summarize_trials(trials, n_trials)
        exp4_results[label] = s
        dt = time.time() - t0
        print(f" Succ={s['success_rate']:.1f}% | Coll={s['collision_rate']:.1f}% | "
              f"IAE={s['p_iae_mean']:.2f} | Cost={s['total_cost_mean']:.0f} | "
              f"MinD={s['min_obstacle_dist_mean']:.3f}m | [{dt:.0f}s]")
    all_results['exp4_no_cbf'] = exp4_results

    # ============================================================
    # [6/7] 消融实验
    # ============================================================
    print("\n" + "="*60)
    print("[6/7] 消融实验")
    print("="*60)

    # K 消融（弱CBF — 更有区分度）
    print("\n--- K 消融 (弱CBF) ---")
    k_ablation_weak = {}
    for k in [1, 5, 10, 20, 50, 100]:
        pred = PTRMNMPCPredictor(model, env_weak_cbf, K=k, D=16, sigma=0.8 if k > 1 else 0.0)
        trials = [run_single_trial(env_weak_cbf, pred, random_x_init(), x_sp, SIM_STEPS, enable_cbf=True)
                  for _ in range(MAIN_TRIALS)]
        s = summarize_trials(trials, MAIN_TRIALS)
        k_ablation_weak[f'K={k}'] = s
        print(f"  K={k:3d}: Succ={s['success_rate']:.1f}% | Coll={s['collision_rate']:.1f}% | "
              f"IAE={s['p_iae_mean']:.2f} | MinD={s['min_obstacle_dist_mean']:.3f}m")
    all_results['ablation_K_weak'] = k_ablation_weak

    # K 消融（强CBF）
    print("\n--- K 消融 (强CBF) ---")
    k_ablation = {}
    for k in [1, 5, 10, 20, 50, 100]:
        pred = PTRMNMPCPredictor(model, env, K=k, D=16, sigma=0.8 if k > 1 else 0.0)
        trials = [run_single_trial(env, pred, random_x_init(), x_sp, SIM_STEPS, enable_cbf=True)
                  for _ in range(MAIN_TRIALS)]
        s = summarize_trials(trials, MAIN_TRIALS)
        k_ablation[f'K={k}'] = s
        print(f"  K={k:3d}: Succ={s['success_rate']:.1f}% | Coll={s['collision_rate']:.1f}% | "
              f"IAE={s['p_iae_mean']:.2f} | Cost={s['total_cost_mean']:.0f}")
    all_results['ablation_K'] = k_ablation

    # σ 消融
    print("\n--- σ 消融 (K=50, 强CBF) ---")
    sigma_ablation = {}
    for sig in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
        pred = PTRMNMPCPredictor(model, env, K=50, D=16, sigma=sig)
        trials = [run_single_trial(env, pred, random_x_init(), x_sp, SIM_STEPS, enable_cbf=True)
                  for _ in range(MAIN_TRIALS)]
        s = summarize_trials(trials, MAIN_TRIALS)
        sigma_ablation[f'sigma={sig:.1f}'] = s
        print(f"  σ={sig:.1f}: Succ={s['success_rate']:.1f}% | IAE={s['p_iae_mean']:.2f} | Cost={s['total_cost_mean']:.0f}")
    all_results['ablation_sigma'] = sigma_ablation

    # D 消融
    print("\n--- D 消融 (K=50, σ=0.8) ---")
    d_ablation = {}
    for d in [4, 8, 12, 16, 20, 24]:
        pred = PTRMNMPCPredictor(model, env, K=50, D=d, sigma=0.8)
        trials = [run_single_trial(env, pred, random_x_init(), x_sp, SIM_STEPS, enable_cbf=True)
                  for _ in range(MAIN_TRIALS)]
        s = summarize_trials(trials, MAIN_TRIALS)
        d_ablation[f'D={d}'] = s
        print(f"  D={d:2d}: Succ={s['success_rate']:.1f}% | IAE={s['p_iae_mean']:.2f} | Cost={s['total_cost_mean']:.0f}")
    all_results['ablation_D'] = d_ablation

    # 滞回消融
    print("\n--- 滞回消融 (K=50, σ=0.8) ---")
    hyst_ablation = {}
    for eta_val in [0.0, 0.01, 0.03, 0.05, 0.08, 0.10]:
        pred = PTRMNMPCPredictor(model, env, K=50, D=16, sigma=0.8, eta_hyst=eta_val)
        trials = [run_single_trial(env, pred, random_x_init(), x_sp, SIM_STEPS, enable_cbf=True)
                  for _ in range(MAIN_TRIALS)]
        s = summarize_trials(trials, MAIN_TRIALS)
        hyst_ablation[f'eta={eta_val:.2f}'] = s
        print(f"  η={eta_val:.2f}: IAE={s['p_iae_mean']:.2f} | Cost={s['total_cost_mean']:.0f}")
    all_results['ablation_hysteresis'] = hyst_ablation

    # ============================================================
    # [7/7] 计算效率实验
    # ============================================================
    print("\n" + "="*60)
    print("[7/7] 计算效率与宽度缩放")
    print("="*60)

    widths_K = [1, 5, 10, 20, 50, 100]
    latencies = []
    cost_changes = []
    rollout_costs = []
    x_test_state = torch.tensor([0.0, -0.5, 0.0, 0.3, 0.3, 0.3], dtype=torch.float32)

    # K=1 基准代价
    ref_pred = PTRMNMPCPredictor(model, env, K=1, D=16, sigma=0.0)
    ref_pred.reset()
    ref_cost = 0.0
    x_c = x_test_state.clone()
    for step in range(SIM_STEPS):
        u, _ = ref_pred.predict_action(x_c, x_sp, enable_cbf=True)
        x_c = env.step_discrete(x_c, u, process_noise=0.008)
        err = x_c[0:6].detach().cpu().numpy() - x_sp[0:6].detach().cpu().numpy()
        ref_cost += float(np.dot(np.array([15,15,15,1,1,1]) * err, err) + 0.02 * float(torch.sum(u.cpu()**2)))

    for k in widths_K:
        tester = PTRMNMPCPredictor(model, env, K=k, D=16, sigma=0.8 if k > 1 else 0.0)
        start_t = time.time()
        for _ in range(50):
            _, _ = tester.predict_action(x_test_state, x_sp, enable_cbf=True)
        avg_latency = (time.time() - start_t) / 50.0 * 1000.0
        latencies.append(avg_latency)

        tester.reset()
        total_cost = 0.0
        x_c = x_test_state.clone()
        for step in range(SIM_STEPS):
            u, _ = tester.predict_action(x_c, x_sp, enable_cbf=True)
            x_c = env.step_discrete(x_c, u, process_noise=0.008)
            err = x_c[0:6].detach().cpu().numpy() - x_sp[0:6].detach().cpu().numpy()
            total_cost += float(np.dot(np.array([15,15,15,1,1,1]) * err, err) + 0.02 * float(torch.sum(u.cpu()**2)))

        rollout_costs.append(total_cost)
        cost_change = ((ref_cost - total_cost) / ref_cost * 100.0) if ref_cost > 1e-6 else 0.0
        cost_changes.append(cost_change)
        print(f"  K={k:3d} | Latency: {avg_latency:.3f} ms | Cost: {total_cost:.1f} | Δ vs K=1: {cost_change:+.1f}%")

    # NMPC延迟
    solver_times = []
    for _ in range(30):
        start_t = time.time()
        _ = solver.solve(x_test_state, x_sp)
        solver_times.append((time.time() - start_t) * 1000.0)
    expert_latency = np.mean(solver_times)
    print(f"  Expert NMPC | Latency: {expert_latency:.3f} ms")

    all_results['exp_runtime'] = {
        'widths_K': widths_K,
        'latencies_ms': latencies,
        'cost_change_pct': cost_changes,
        'rollout_costs': rollout_costs,
        'expert_latency_ms': expert_latency,
        'ref_cost_K1': ref_cost,
    }

    # ============================================================
    # 保存结果
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(RESULTS_DIR, f"v4_final_{timestamp}")
    with open(f"{prefix}_results.json", 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n实验数据已保存: {prefix}_results.json")

    # ============================================================
    # 绘图
    # ============================================================
    _plot_all_results(all_results)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"全部实验完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"结果保存在: {RESULTS_DIR}")
    print(f"{'='*60}")

    _print_paper_tables(all_results)


def _plot_all_results(results):
    """绘制所有实验图表"""
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'results')

    # --- Exp I + II + IV 对比 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    exp1 = results['exp1_standard']
    exp2 = results['exp2_weak_cbf']

    # Exp I: 成功率 + 碰撞率
    methods_cbf = list(exp1.keys())
    x_pos = np.arange(len(methods_cbf))
    axes[0, 0].bar(x_pos - 0.15, [exp1[m]['success_rate'] for m in methods_cbf],
                   0.3, label='Success Rate (%)', color='green', alpha=0.7)
    axes[0, 0].bar(x_pos + 0.15, [exp1[m]['collision_rate'] for m in methods_cbf],
                   0.3, label='Collision Rate (%)', color='red', alpha=0.7)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(methods_cbf, rotation=25, ha='right', fontsize=7)
    axes[0, 0].set_title('Exp I: Standard (Strong CBF α_d=0.8)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis='y')

    # Exp I: IAE
    axes[1, 0].bar(x_pos, [exp1[m]['p_iae_mean'] for m in methods_cbf],
                   yerr=[exp1[m]['p_iae_std'] for m in methods_cbf], capsize=3, color='coral', alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(methods_cbf, rotation=25, ha='right', fontsize=7)
    axes[1, 0].set_ylabel('Position IAE (m·s)')
    axes[1, 0].set_title('Exp I: Tracking Error')
    axes[1, 0].grid(True, axis='y')

    # Exp II: 弱CBF 碰撞率 + 成功率
    methods_weak = list(exp2.keys())
    x_pos2 = np.arange(len(methods_weak))
    axes[0, 1].bar(x_pos2 - 0.15, [exp2[m]['success_rate'] for m in methods_weak],
                   0.3, label='Success %', color='green', alpha=0.7)
    axes[0, 1].bar(x_pos2 + 0.15, [exp2[m]['collision_rate'] for m in methods_weak],
                   0.3, label='Collision %', color='red', alpha=0.7)
    axes[0, 1].set_xticks(x_pos2)
    axes[0, 1].set_xticklabels(methods_weak, rotation=25, ha='right', fontsize=7)
    axes[0, 1].set_title('Exp II: Weak CBF (α_d=0.3)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, axis='y')

    # Exp II: MinD
    axes[1, 1].bar(x_pos2, [exp2[m]['min_obstacle_dist_mean'] for m in methods_weak],
                   color='steelblue', alpha=0.7)
    axes[1, 1].set_xticks(x_pos2)
    axes[1, 1].set_xticklabels(methods_weak, rotation=25, ha='right', fontsize=7)
    axes[1, 1].set_ylabel('Min Obstacle Distance (m)')
    axes[1, 1].set_title('Exp II: Safety Margin')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1, 1].grid(True, axis='y')

    # Exp IV: 无CBF
    exp4 = results['exp4_no_cbf']
    methods_nocbf = list(exp4.keys())
    x_pos3 = np.arange(len(methods_nocbf))
    axes[0, 2].bar(x_pos3, [exp4[m]['collision_rate'] for m in methods_nocbf], color='red', alpha=0.7)
    axes[0, 2].set_xticks(x_pos3)
    axes[0, 2].set_xticklabels(methods_nocbf, rotation=25, ha='right', fontsize=7)
    axes[0, 2].set_ylabel('Collision Rate (%)')
    axes[0, 2].set_title('Exp IV: No CBF - Collision Rate')
    axes[0, 2].grid(True, axis='y')

    axes[1, 2].bar(x_pos3, [exp4[m]['min_obstacle_dist_mean'] for m in methods_nocbf], color='steelblue', alpha=0.7)
    axes[1, 2].set_xticks(x_pos3)
    axes[1, 2].set_xticklabels(methods_nocbf, rotation=25, ha='right', fontsize=7)
    axes[1, 2].set_ylabel('Min Obstacle Dist (m)')
    axes[1, 2].set_title('Exp IV: No CBF - Safety Margin')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1, 2].grid(True, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'v4_exp1_exp2_exp4.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    # --- Exp III: 失配鲁棒性 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    exp3 = results['exp3_mismatch']
    methods = list(exp3.keys())
    x_pos = np.arange(len(methods))

    axes[0].bar(x_pos, [exp3[m]['success_rate'] for m in methods], color='green', alpha=0.7)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods, fontsize=9)
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Exp III: +50% Mass Mismatch - Success Rate')
    axes[0].grid(True, axis='y')

    axes[1].bar(x_pos, [exp3[m]['terminal_err_mean'] for m in methods],
                yerr=[exp3[m]['terminal_err_std'] for m in methods], capsize=3, color='coral', alpha=0.7)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(methods, fontsize=9)
    axes[1].set_ylabel('Terminal Error (m)')
    axes[1].set_title('Exp III: Terminal Accuracy under Mismatch')
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'v4_exp3_mismatch.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    # --- 消融实验 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # K消融 (弱CBF)
    abl_k = results['ablation_K_weak']
    k_labels = list(abl_k.keys())
    axes[0, 0].bar(k_labels, [abl_k[k]['collision_rate'] for k in k_labels], color='red', alpha=0.7)
    axes[0, 0].set_ylabel('Collision Rate (%)')
    axes[0, 0].set_title('(a) K Ablation (Weak CBF): Collision')
    axes[0, 0].grid(True, axis='y')

    axes[1, 0].bar(k_labels, [abl_k[k]['success_rate'] for k in k_labels], color='green', alpha=0.7)
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('(a) K Ablation (Weak CBF): Success')
    axes[1, 0].grid(True, axis='y')

    # σ消融
    abl_s = results['ablation_sigma']
    s_labels = list(abl_s.keys())
    axes[0, 1].plot(s_labels, [abl_s[k]['total_cost_mean'] for k in s_labels], 'b-o', linewidth=2)
    axes[0, 1].set_ylabel('Total Cost')
    axes[0, 1].set_title('(b) σ Ablation: Cost')
    axes[0, 1].grid(True)
    axes[0, 1].tick_params(axis='x', rotation=45)

    axes[1, 1].plot(s_labels, [abl_s[k]['p_iae_mean'] for k in s_labels], 'r-s', linewidth=2)
    axes[1, 1].set_ylabel('Position IAE (m·s)')
    axes[1, 1].set_title('(b) σ Ablation: Tracking Error')
    axes[1, 1].grid(True)
    axes[1, 1].tick_params(axis='x', rotation=45)

    # D消融
    abl_d = results['ablation_D']
    d_labels = list(abl_d.keys())
    axes[0, 2].plot(d_labels, [abl_d[k]['total_cost_mean'] for k in d_labels], 'g-^', linewidth=2)
    axes[0, 2].set_ylabel('Total Cost')
    axes[0, 2].set_title('(c) D Ablation: Cost')
    axes[0, 2].grid(True)

    axes[1, 2].plot(d_labels, [abl_d[k]['p_iae_mean'] for k in d_labels], 'm-d', linewidth=2)
    axes[1, 2].set_ylabel('Position IAE (m·s)')
    axes[1, 2].set_title('(c) D Ablation: Tracking')
    axes[1, 2].grid(True)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'v4_ablation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    # --- 运行时缩放 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    exp_r = results['exp_runtime']
    axes[0].plot(exp_r['widths_K'], exp_r['latencies_ms'], 'b-o', linewidth=2, markersize=8, label='PTRM Latency')
    axes[0].axhline(y=exp_r['expert_latency_ms'], color='red', linestyle='--', linewidth=2, label='Expert NMPC')
    axes[0].set_xlabel('Width K')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Inference Latency Scaling')
    axes[0].legend()
    axes[0].grid(True)

    colors = ['green' if c >= 0 else 'red' for c in exp_r['cost_change_pct']]
    axes[1].bar(range(len(exp_r['widths_K'])), exp_r['cost_change_pct'], color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(exp_r['widths_K'])))
    axes[1].set_xticklabels([f'K={k}' for k in exp_r['widths_K']])
    axes[1].set_ylabel('Cost Change vs K=1 (%)')
    axes[1].set_title('Cost Reduction by Width Scaling')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'v4_runtime.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"所有图表已保存至: {RESULTS_DIR}")


def _print_paper_tables(results):
    """打印论文表格所需的精确数据"""
    print("\n" + "="*60)
    print("论文表格数据（可直接填入 LaTeX）")
    print("="*60)

    # Table 1: 标准避障
    exp1 = results['exp1_standard']
    print("\n【Table 1: Experiment I — Non-Convex Obstacle Corridor (Strong CBF)】")
    print(f"{'Framework':<35} {'Succ%':>6} {'Pos IAE':>12} {'Vel IAE':>12} {'Coll':>5}")
    print("-" * 75)
    for m in exp1:
        d = exp1[m]
        print(f"{m:<35} {d['success_rate']:>5.1f}% "
              f"{d['p_iae_mean']:>6.2f}±{d['p_iae_std']:<4.2f} "
              f"{d['v_iae_mean']:>6.2f}±{d['v_iae_std']:<4.2f} "
              f"{d['collision_rate']:>4.1f}%")

    # Table 2: 弱CBF
    exp2 = results['exp2_weak_cbf']
    print("\n【Table 2: Experiment II — Weak CBF (α_d=0.3): PTRM Safety Gain】")
    print(f"{'Framework':<35} {'Succ%':>6} {'Coll%':>6} {'Pos IAE':>12} {'MinD(m)':>10}")
    print("-" * 75)
    for m in exp2:
        d = exp2[m]
        print(f"{m:<35} {d['success_rate']:>5.1f}% {d['collision_rate']:>5.1f}% "
              f"{d['p_iae_mean']:>6.2f}±{d['p_iae_std']:<4.2f} "
              f"{d['min_obstacle_dist_mean']:>8.3f}")

    # Table 3: 失配
    exp3 = results['exp3_mismatch']
    print("\n【Table 3: Experiment III — Robustness Under +50% Mass Mismatch】")
    print(f"{'Framework':<35} {'Succ%':>6} {'Coll%':>6} {'Pos IAE':>12}")
    print("-" * 65)
    for m in exp3:
        d = exp3[m]
        print(f"{m:<35} {d['success_rate']:>5.1f}% {d['collision_rate']:>5.1f}% "
              f"{d['p_iae_mean']:>6.2f}±{d['p_iae_std']:<4.2f}")

    # Table 4: 无CBF
    exp4 = results['exp4_no_cbf']
    print("\n【Table 4: No-CBF Safety — Intrinsic Safety from Test-Time Compute】")
    print(f"{'Framework':<35} {'Coll%':>6} {'Succ%':>6} {'MinD (m)':>10} {'Cost':>10}")
    print("-" * 72)
    for m in exp4:
        d = exp4[m]
        print(f"{m:<35} {d['collision_rate']:>5.1f}% {d['success_rate']:>5.1f}% "
              f"{d['min_obstacle_dist_mean']:>8.3f} {d['total_cost_mean']:>8.0f}")

    # Table 5: 运行时
    exp_r = results['exp_runtime']
    print("\n【Table 5: Computational Efficiency & Width Scaling】")
    print(f"{'Width K':>10} {'Latency (ms)':>14} {'Cost Δ vs K=1':>16} {'Rollout Cost':>14}")
    print("-" * 58)
    for i, k in enumerate(exp_r['widths_K']):
        print(f"{'K='+str(k):>10} {exp_r['latencies_ms'][i]:>12.3f} {exp_r['cost_change_pct'][i]:>+13.1f}% {exp_r['rollout_costs'][i]:>12.1f}")
    print(f"{'NMPC':>10} {exp_r['expert_latency_ms']:>12.3f} {'N/A':>16} {'N/A':>14}")

    # K消融（弱CBF）
    abl_k = results['ablation_K_weak']
    print("\n【Table 6: K Ablation under Weak CBF — Key Result】")
    print(f"{'K':>10} {'Succ%':>6} {'Coll%':>6} {'IAE':>8} {'MinD(m)':>10}")
    print("-" * 45)
    for k_label in abl_k:
        d = abl_k[k_label]
        print(f"{k_label:>10} {d['success_rate']:>5.1f}% {d['collision_rate']:>5.1f}% "
              f"{d['p_iae_mean']:>7.2f} {d['min_obstacle_dist_mean']:>9.3f}")


if __name__ == "__main__":
    main()
