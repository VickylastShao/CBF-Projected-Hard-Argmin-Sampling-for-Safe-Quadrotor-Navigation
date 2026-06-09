# -*- coding: utf-8 -*-
"""
TRM-NMPC 神经网络结构

具有权重共享递归及回归 Q 头打分器的微型递归网络，
27,935 可训练参数，严格对应论文 Section 3.B Table 4。
"""

import torch
import torch.nn as nn


class TRMNMPC(nn.Module):
    """具有权重共享递归及回归 Q 头打分器的微型递归网络"""

    def __init__(self, input_dim=12, latent_dim=64, mpc_horizon=30):
        super(TRMNMPC, self).__init__()
        self.H = mpc_horizon
        self.latent_dim = latent_dim

        self.W_x = nn.Linear(input_dim, latent_dim)
        self.W_y = nn.Linear(mpc_horizon, latent_dim)
        self.W_z = nn.Linear(latent_dim, latent_dim)

        self.M_y = nn.Linear(latent_dim, latent_dim)
        self.M_z = nn.Linear(latent_dim, latent_dim)

        self.recur_cell_z = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.recur_cell_y = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.f_O = nn.Linear(latent_dim, mpc_horizon)

        # 伴随训练的原生 Q 头打分器
        self.f_Q = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward_steps(self, X, D=16, noise_scale=0.0, noise_mode='both',
                      u_seq_external=None):
        """
        D步权重共享递归前向推理

        Args:
            X: 输入张量 (batch, 12) — [x_init(6), x_sp(6)]
            D: 递归步数
            noise_scale: 噪声强度标量
            noise_mode: 噪声注入模式
                'both'    — 潜在空间 + 输出空间双通道 (默认，完整PTRM)
                'latent'  — 仅潜在空间扰动 (消融: 隔离输出噪声贡献)
                'output'  — 仅输出空间扰动 (消融: 隔离潜在噪声贡献)
                'none'    — 无噪声 (确定性推理，K=1时使用)
            u_seq_external: 外部动作序列 (batch, 30) — 当提供时，
                替代TRM自身解码输出参与W_y编码，使Q-head评分基于外部候选。
                在PD+TRM-Eval模式下传入PD候选序列，解决Q-head/候选架构断裂问题。
                f_O解码仍正常执行，但W_y编码使用外部序列。
        """
        batch_size = X.shape[0]
        device = X.device

        z_t = torch.zeros(batch_size, self.latent_dim, device=device)
        u_seq_decoded = torch.zeros(batch_size, self.H, device=device)
        y_t = torch.zeros(batch_size, self.latent_dim, device=device)

        y_history = []

        for t in range(D):
            # 潜在空间扰动: clamped Gaussian noise on latent state z_t
            # 对应论文 Section 3.C 的公式 (3.5)
            if noise_scale > 0.0 and noise_mode in ('both', 'latent'):
                epsilon = torch.randn_like(z_t) * noise_scale
                epsilon = torch.clamp(epsilon, min=-1.0, max=1.0)
                z_t = z_t + epsilon

            # 决定W_y编码的动作序列来源：
            # - 无外部输入时：使用TRM自身解码输出（标准模式）
            # - 有外部输入时：使用外部候选序列（PD+TRM-Eval修复模式）
            #   使Q-head评分基于外部候选而非TRM幻觉
            if u_seq_external is not None:
                u_to_encode = u_seq_external
            else:
                u_to_encode = u_seq_decoded

            proj_z_input = self.W_x(X) + self.W_y(u_to_encode) + self.W_z(z_t)
            z_t = torch.tanh(self.recur_cell_z(proj_z_input))

            proj_y_input = self.M_y(y_t) + self.M_z(z_t)
            y_t = torch.tanh(self.recur_cell_y(proj_y_input))

            u_seq_decoded = self.f_O(y_t)

            # 输出空间扰动: Gaussian noise on decoded action sequences
            # 确保即使潜在动力学饱和也能产生候选多样性
            # 对应论文 Section 3.C 的公式 (3.6)
            if noise_scale > 0.0 and noise_mode in ('both', 'output'):
                u_noise = torch.randn_like(u_seq_decoded) * noise_scale * 0.5
                u_seq_decoded = u_seq_decoded + u_noise

            y_history.append((u_seq_decoded, y_t))

        return y_history


class SimpleEncoderQHead(nn.Module):
    """
    简单编码器 + Q-head 模型（路径A验证实验3：TRM架构消融对照）

    用最简单的线性编码器（12→64+ReLU）替代TRM的递归编码器，
    保留相同的Q-head结构（64→32→1），用于验证TRM递归结构
    是否对Q-head特征提取有独特贡献。

    参数量对比：
      SimpleEncoderQHead: 832 (编码器) + 2081 (Q-head) = 2913 参数
      TRMNMPC: 27935 参数（含编码器 + Q-head + f_O解码器）
    """

    def __init__(self, input_dim=12, latent_dim=64):
        super(SimpleEncoderQHead, self).__init__()
        self.latent_dim = latent_dim

        # 简单编码器：单层线性 + ReLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )

        # Q-head：与TRMNMPC相同结构
        self.f_Q = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        """
        前向推理：编码 → Q-head评分

        Args:
            X: 输入张量 (batch, 12) — [x_init(6), x_sp(6)]

        Returns:
            latent_y: 潜在表示 (batch, 64)
            q_score: Q-head评分 (batch, 1)
        """
        latent_y = self.encoder(X)
        q_score = self.f_Q(latent_y)
        return latent_y, q_score
