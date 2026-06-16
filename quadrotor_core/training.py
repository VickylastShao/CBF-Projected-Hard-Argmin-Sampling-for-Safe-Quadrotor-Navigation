# -*- coding: utf-8 -*-
"""
TRM-NMPC 联合训练模块

策略网络深度监督训练 + Q-head 回归训练联合优化，
Q-head 标签基于当前解码候选轨迹代价动态生成。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dataset import evaluate_batch_decoded_trajectory_cost


def train_trm_jointly(model, dataset, env, epochs=35, batch_size=32,
                      lr=0.0025, gamma=0.95, lambda_Q=0.1, V_max=150.0,
                      val_ratio=0.2, patience=15, verbose=True):
    """
    启动 TRM 网络深度监督训练，并基于当前解码候选轨迹代价动态拟合 Q 头。

    参数:
        model: TRMNMPC 模型
        dataset: 专家数据集 [(X_feature, u_opt), ...]
        env: QuadrotorDynamics 环境
        epochs: 最大训练轮次
        batch_size: 批量大小
        lr: Adam 学习率
        gamma: 深度监督折扣因子
        lambda_Q: Q-head 损失权重
        V_max: Q-head 目标值上限截断
        val_ratio: 验证集比例
        patience: 早停耐心值
        verbose: 是否打印训练进度
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 拆分训练集与验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_all = torch.stack([d[0] for d in dataset]).to(device)
    Y_true_all = torch.stack([d[1] for d in dataset]).to(device)

    X_train, Y_train = X_all[train_indices], Y_true_all[train_indices]
    X_val, Y_val = X_all[val_indices], Y_true_all[val_indices]

    history = {
        'train_policy_loss': [],
        'train_q_loss': [],
        'train_total_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_size)
        epoch_policy_loss = 0.0
        epoch_q_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for i in range(0, train_size, batch_size):
            batch_indices = permutation[i:i+batch_size]
            batch_x = X_train[batch_indices]
            batch_y_true = Y_train[batch_indices]
            optimizer.zero_grad()
            y_history = model.forward_steps(batch_x, D=16)

            loss_policy = 0.0
            for t in range(16):
                u_seq_t, _ = y_history[t]
                weight = gamma ** (15 - t)
                loss_policy += weight * mse_loss(u_seq_t, batch_y_true)

            final_u_seq, final_latent_y = y_history[-1]
            q_predicted = model.f_Q(final_latent_y)
            with torch.no_grad():
                decoded_cost = evaluate_batch_decoded_trajectory_cost(env, batch_x, final_u_seq)
                q_target = torch.clamp(V_max - decoded_cost, min=0.0).unsqueeze(1)
            loss_q = mse_loss(q_predicted, q_target)

            total_loss = loss_policy + lambda_Q * loss_q
            total_loss.backward()
            optimizer.step()

            epoch_policy_loss += loss_policy.item()
            epoch_q_loss += loss_q.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_y_history = model.forward_steps(X_val, D=16)
            val_loss = 0.0
            for t in range(16):
                u_seq_t, _ = val_y_history[t]
                weight = gamma ** (15 - t)
                val_loss += weight * mse_loss(u_seq_t, Y_val).item()

        avg_train_policy = epoch_policy_loss / n_batches
        avg_train_q = epoch_q_loss / n_batches
        avg_train_total = epoch_total_loss / n_batches

        history['train_policy_loss'].append(avg_train_policy)
        history['train_q_loss'].append(avg_train_q)
        history['train_total_loss'].append(avg_train_total)
        history['val_loss'].append(val_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | 策略损失: {avg_train_policy:.4f} | Q损失: {avg_train_q:.4f} | 联合损失: {avg_train_total:.4f} | 验证损失: {val_loss:.4f} | 早停计数: {epochs_without_improve}/{patience}")

        if epochs_without_improve >= patience:
            if verbose:
                print(f"早停触发：验证损失连续 {patience} 轮未改善，恢复最佳模型。")
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def train_trm_candidate_conditioned(model, dataset, env, epochs=50, batch_size=32,
                                     lr=0.002, gamma=0.95, lambda_Q=0.3,
                                     V_max=5000.0, K_train=10,
                                     output_noise_sigma=2.0,
                                     val_ratio=0.2, patience=20, verbose=True):
    """
    候选条件化TRM训练 (Candidate-Conditioned Training)

    核心创新：训练时对每个样本生成 K_train 个输出噪声候选，
    通过 W_y 编码让 Q-head 学会区分同一状态下不同候选的质量。

    与标准训练的区别：
    - 标准训练：Q-head 只看确定性推理结果 → 学会"状态评估"(跨样本区分)
    - 候选条件化：Q-head 看 W_y 编码的多个噪声候选 → 学会"候选排序"(相同样本内区分)

    Q-head目标设计（关键修复 v2）：
    - 直接回归归一化的 rollout cost：q_target = cost / cost_scale
    - cost_scale 基于初始batch的cost统计量自动设定
    - 推理时：Q-head输出 = 预测cost，排序取argmin（而非argmax）
    - 这避免了V_max截断导致q_target全为0的bug

    参数:
        model: TRMNMPC 模型
        dataset: 专家数据集 [(X_feature, u_opt), ...]
        env: QuadrotorDynamics 环境
        epochs: 最大训练轮次
        batch_size: 批量大小
        lr: Adam 学习率
        gamma: 深度监督折扣因子
        lambda_Q: Q-head 损失权重（比标准训练增大，因为Q-head任务更难）
        V_max: 保留兼容接口（实际不再用于截断，改用 cost_scale 归一化）
        K_train: 训练时每样本生成的候选数量
        output_noise_sigma: 输出空间噪声标准差（与推理时候选噪声匹配）
        val_ratio: 验证集比例
        patience: 早停耐心值
        verbose: 是否打印训练进度
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 拆分训练集与验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_all = torch.stack([d[0] for d in dataset]).to(device)
    Y_true_all = torch.stack([d[1] for d in dataset]).to(device)

    X_train, Y_train = X_all[train_indices], Y_true_all[train_indices]
    X_val, Y_val = X_all[val_indices], Y_true_all[val_indices]

    # 预估cost_scale：基于专家动作的rollout cost
    with torch.no_grad():
        sample_cost = evaluate_batch_decoded_trajectory_cost(env, X_train[:50], Y_train[:50])
        cost_scale = sample_cost.mean().item()
    if verbose:
        print(f"  [Q-head归一化] cost_scale = {cost_scale:.1f} (基于专家动作平均rollout cost)")

    history = {
        'train_policy_loss': [],
        'train_q_loss': [],
        'train_total_loss': [],
        'val_loss': [],
        'q_rank_correlation': [],  # 跟踪Q-head排序能力
        'cost_scale': cost_scale,
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_size)
        epoch_policy_loss = 0.0
        epoch_q_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for i in range(0, train_size, batch_size):
            batch_indices = permutation[i:i+batch_size]
            batch_x = X_train[batch_indices]
            batch_y_true = Y_train[batch_indices]
            B = batch_x.shape[0]

            optimizer.zero_grad()

            # ============================================================
            # 1. 策略训练：深度监督（与标准训练完全相同）
            # ============================================================
            y_history = model.forward_steps(batch_x, D=16, noise_scale=0.0)

            loss_policy = 0.0
            for t in range(16):
                u_seq_t, _ = y_history[t]
                weight = gamma ** (15 - t)
                loss_policy += weight * mse_loss(u_seq_t, batch_y_true)

            # ============================================================
            # 2. 候选条件化 Q-head 训练（核心创新）
            # ============================================================
            # 2a. 从当前TRM输出生成 K_train 个噪声候选
            with torch.no_grad():
                u_base = y_history[-1][0].detach()  # (B, 30) - 确定性TRM输出

            # 在输出空间添加噪声生成多个候选
            u_base_expanded = u_base.unsqueeze(1).repeat(1, K_train, 1)  # (B, K, 30)

            # 第一步噪声较大（与推理时匹配），后续步骤噪声递减
            output_noise = torch.zeros(B, K_train, 30, device=device)
            output_noise[:, :, 0:3] = torch.randn(B, K_train, 3, device=device) * output_noise_sigma
            for step_i in range(1, 10):
                decay = max(0.3, 1.0 - step_i * 0.1)  # 后续步骤噪声递减
                output_noise[:, :, step_i*3:(step_i+1)*3] = (
                    torch.randn(B, K_train, 3, device=device) * output_noise_sigma * decay
                )

            u_candidates = u_base_expanded + output_noise  # (B, K, 30)

            # 2b. 展平后通过 W_y 编码传入 TRM
            u_flat = u_candidates.reshape(B * K_train, 30)
            x_flat = batch_x.unsqueeze(1).repeat(1, K_train, 1).reshape(B * K_train, 12)

            y_cand = model.forward_steps(x_flat, D=16, noise_scale=0.0,
                                          u_seq_external=u_flat)
            _, final_y = y_cand[-1]
            q_predicted = model.f_Q(final_y).reshape(B, K_train)  # (B, K)

            # 2c. 计算每个候选的 rollout cost 作为 Q-target
            # 关键修复：直接回归归一化cost，不再用V_max截断
            # Q-head学习预测 cost/cost_scale，推理时排序取argmin
            with torch.no_grad():
                q_targets_list = []
                for k in range(K_train):
                    cost_k = evaluate_batch_decoded_trajectory_cost(
                        env, batch_x, u_flat[k::K_train]
                    )
                    # 归一化到[0, ~2]范围，使MSE梯度有效
                    q_target_k = cost_k / cost_scale
                    q_targets_list.append(q_target_k)
                q_target = torch.stack(q_targets_list, dim=1)  # (B, K)

            loss_q = mse_loss(q_predicted, q_target)

            # ============================================================
            # 3. 联合优化
            # ============================================================
            total_loss = loss_policy + lambda_Q * loss_q
            total_loss.backward()
            optimizer.step()

            epoch_policy_loss += loss_policy.item()
            epoch_q_loss += loss_q.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_y_history = model.forward_steps(X_val, D=16)
            val_loss = 0.0
            for t in range(16):
                u_seq_t, _ = val_y_history[t]
                weight = gamma ** (15 - t)
                val_loss += weight * mse_loss(u_seq_t, Y_val).item()

        avg_train_policy = epoch_policy_loss / n_batches
        avg_train_q = epoch_q_loss / n_batches
        avg_train_total = epoch_total_loss / n_batches

        history['train_policy_loss'].append(avg_train_policy)
        history['train_q_loss'].append(avg_train_q)
        history['train_total_loss'].append(avg_train_total)
        history['val_loss'].append(val_loss)

        # 周期性评估Q-head排序能力
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_x = X_val[:5]  # 取5个验证样本
                rhos = []
                for si in range(5):
                    x_s = sample_x[si:si+1]
                    u_s = model.forward_steps(x_s, D=16, noise_scale=0.0)[-1][0]
                    # 生成K_train个噪声候选（与训练一致）
                    u_cands = u_s.repeat(K_train, 1)
                    u_cands[:, 0:3] = u_cands[:, 0:3] + torch.randn(K_train, 3) * output_noise_sigma
                    for step_i in range(1, 10):
                        decay = max(0.3, 1.0 - step_i * 0.1)
                        u_cands[:, step_i*3:(step_i+1)*3] = (
                            u_cands[:, step_i*3:(step_i+1)*3]
                            + torch.randn(K_train, 3) * output_noise_sigma * decay
                        )
                    x_rep = x_s.repeat(K_train, 1)
                    y_c = model.forward_steps(x_rep, D=16, noise_scale=0.0,
                                               u_seq_external=u_cands)
                    # Q-head现在预测归一化cost，排序取argmin
                    q_s = model.f_Q(y_c[-1][1]).squeeze(-1).numpy()
                    costs = []
                    for ki in range(K_train):
                        c = evaluate_batch_decoded_trajectory_cost(env, x_s, u_cands[ki:ki+1])
                        costs.append(c.item())
                    from scipy.stats import spearmanr
                    # Q-head预测cost → 与真实cost正相关 = 好的排序
                    rho, _ = spearmanr(q_s, costs)
                    if not np.isnan(rho):
                        rhos.append(rho)
                avg_rho = np.mean(rhos) if rhos else 0.0
                history['q_rank_correlation'].append(avg_rho)
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  [Q排序评估] 验证集平均Spearman ρ = {avg_rho:.4f}")

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | 策略损失: {avg_train_policy:.4f} | "
                  f"Q损失: {avg_train_q:.4f} | 联合损失: {avg_train_total:.4f} | "
                  f"验证损失: {val_loss:.4f} | 早停计数: {epochs_without_improve}/{patience}")

        if epochs_without_improve >= patience:
            if verbose:
                print(f"早停触发：验证损失连续 {patience} 轮未改善，恢复最佳模型。")
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def train_simple_encoder_qhead(model, dataset, env, epochs=35, batch_size=32,
                                lr=0.0025, V_max=150.0, val_ratio=0.2,
                                patience=15, verbose=True):
    """
    训练简单编码器Q-head模型（路径A验证实验3：TRM架构消融对照）

    仅训练Q-head回归目标（与TRM Q-head相同的训练信号），
    不涉及策略解码器f_O，因为PD模式下不使用TRM的解码输出。

    Q-head训练目标：
      q_target = clamp(V_max - rollout_cost, min=0)
      loss = MSE(q_predicted, q_target)

    Args:
        model: SimpleEncoderQHead 模型
        dataset: 专家数据集 [(X_feature, u_opt), ...]（复用TRM训练数据集）
        env: QuadrotorDynamics 环境
        epochs: 最大训练轮次
        batch_size: 批量大小
        lr: Adam 学习率
        V_max: Q-head 目标值上限截断
        val_ratio: 验证集比例
        patience: 早停耐心值
        verbose: 是否打印训练进度
    """
    from .dataset import evaluate_batch_decoded_trajectory_cost

    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 拆分训练集与验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_all = torch.stack([d[0] for d in dataset]).to(device)
    Y_true_all = torch.stack([d[1] for d in dataset]).to(device)

    X_train = X_all[train_indices]
    X_val = X_all[val_indices]
    Y_val = Y_true_all[val_indices]

    # 预估cost_scale（与候选条件化训练一致）
    with torch.no_grad():
        sample_cost = evaluate_batch_decoded_trajectory_cost(env, X_train[:50], Y_true_all[train_indices[:50]])
        cost_scale = sample_cost.mean().item()
    if verbose:
        print(f"  [SimpleEncoder] cost_scale = {cost_scale:.1f}")

    history = {
        'train_q_loss': [],
        'val_q_loss': [],
        'cost_scale': cost_scale,
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_size)
        epoch_q_loss = 0.0
        n_batches = 0

        for i in range(0, train_size, batch_size):
            batch_indices = permutation[i:i+batch_size]
            batch_x = X_train[batch_indices]
            batch_y_true = Y_true_all[train_indices][batch_indices]
            optimizer.zero_grad()

            latent_y, q_predicted = model(batch_x)
            with torch.no_grad():
                # 使用专家动作序列计算rollout代价作为Q-head标签
                # 同样用归一化cost（与候选条件化训练一致）
                decoded_cost = evaluate_batch_decoded_trajectory_cost(env, batch_x, batch_y_true)
                q_target = (decoded_cost / cost_scale).unsqueeze(1)
            loss_q = mse_loss(q_predicted, q_target)
            loss_q.backward()
            optimizer.step()

            epoch_q_loss += loss_q.item()
            n_batches += 1

        # 验证集评估
        model.eval()
        with torch.no_grad():
            _, val_q_predicted = model(X_val)
            val_decoded_cost = evaluate_batch_decoded_trajectory_cost(env, X_val, Y_val)
            val_q_target = (val_decoded_cost / cost_scale).unsqueeze(1)
            val_loss = mse_loss(val_q_predicted, val_q_target).item()

        avg_train_q = epoch_q_loss / n_batches
        history['train_q_loss'].append(avg_train_q)
        history['val_q_loss'].append(val_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Q损失: {avg_train_q:.4f} | 验证Q损失: {val_loss:.4f} | 早停计数: {epochs_without_improve}/{patience}")

        if epochs_without_improve >= patience:
            if verbose:
                print(f"早停触发：验证损失连续 {patience} 轮未改善，恢复最佳模型。")
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
