# -*- coding: utf-8 -*-
"""
TRM-NMPC 联合训练模块

策略网络深度监督训练 + Q-head 回归训练联合优化，
Q-head 标签基于当前解码候选轨迹代价动态生成。
"""

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
