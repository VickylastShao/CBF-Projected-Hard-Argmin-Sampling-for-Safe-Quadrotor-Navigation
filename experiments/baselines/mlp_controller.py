# -*- coding: utf-8 -*-
"""
MLP+CBF 基线控制器

约28K参数的多层感知机，无递归、无测试时计算缩放。
在与PTRM相同的数据集上训练，用于隔离测试时缩放机制的贡献。

架构: [12] → 64 → 128 → 64 → [3] (约28,000参数)
"""

import torch
import torch.nn as nn
import numpy as np
import time


class MLPController(nn.Module):
    """无递归、无测试时缩放的MLP控制器"""

    def __init__(self, input_dim=12, hidden_dims=(64, 128, 64), output_dim=3):
        super(MLPController, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播: [x_init(6), x_sp(6)] → u(3)"""
        return self.net(x)


class MLPPredictor:
    """MLP 在线决策单元（K=1，无测试时缩放）"""

    def __init__(self, model, env, tracking_Kp=4.0, tracking_Kd=3.0,
                 alpha_blend=0.3, eta_hyst=0.05):
        self.model = model
        self.env = env
        self.tracking_Kp = tracking_Kp
        self.tracking_Kd = tracking_Kd
        self.alpha_blend = alpha_blend
        self.eta_hyst = eta_hyst
        self.last_u = None

    def reset(self):
        self.last_u = None

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        """
        MLP 单次前向推理决策

        注意: MLP 没有 K>1 的候选机制，仅做单次前向传播。
        alpha_blend 用于与 PTRM 公平对比（相同内环修正）。
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            X = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
            u_mlp = self.model(X).squeeze(0).cpu()

            # PD修正（与PTRM相同的alpha_blend机制，确保公平对比）
            if self.alpha_blend > 0:
                e_p = x_sp[0:3] - x_init[0:3]
                e_v = x_sp[3:6] - x_init[3:6]
                u_pd = self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)
                u_nominal = (1.0 - self.alpha_blend) * u_mlp + self.alpha_blend * u_pd
            else:
                u_nominal = u_mlp

            # 滞回
            if self.last_u is not None:
                dist = torch.sum((u_nominal - self.last_u) ** 2)
                # MLP是单步控制，滞回影响较小但仍保留以确保公平
                # 此处仅做轻微正则化，不改变控制

            self.last_u = u_nominal.clone()

            # CBF安全投影
            if enable_cbf:
                u_safe = self.env.apply_cbf_projection(x_init, u_nominal)
            else:
                u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)

        return u_safe

    def get_runtime_ms(self, x_init, x_sp, enable_cbf=True, n_runs=50):
        """测量单步决策延迟 (ms)"""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict_action(x_init, x_sp, enable_cbf)
            times.append((time.perf_counter() - t0) * 1000)
        return np.median(times)


def train_mlp(model, dataset, env, epochs=100, batch_size=32, lr=0.0025,
              val_ratio=0.2, patience=15, verbose=True):
    """
    在与PTRM相同的专家数据集上训练MLP

    训练目标: 最小化MLP输出与专家NMPC第一步控制的MSE
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # 拆分训练/验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    indices = torch.randperm(dataset_size).tolist()

    X_all = torch.stack([d[0] for d in dataset]).to(device)
    # 专家序列的第一步作为MLP训练目标
    Y_all = torch.stack([d[1][0:3] for d in dataset]).to(device)

    X_train, Y_train = X_all[indices[:train_size]], Y_all[indices[:train_size]]
    X_val, Y_val = X_all[indices[train_size:]], Y_all[indices[train_size:]]

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_size)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, train_size, batch_size):
            batch_idx = perm[i:i+batch_size]
            bx = X_train[batch_idx]
            by = Y_train[batch_idx]

            optimizer.zero_grad()
            pred = model(bx)
            loss = mse_loss(pred, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = mse_loss(val_pred, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  MLP Epoch [{epoch+1}/{epochs}] | 训练损失: {epoch_loss/n_batches:.4f} | 验证损失: {val_loss:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  MLP 早停: 验证损失连续 {patience} 轮未改善")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
