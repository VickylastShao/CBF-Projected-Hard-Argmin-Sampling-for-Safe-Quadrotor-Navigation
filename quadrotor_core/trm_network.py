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

    def forward_steps(self, X, D=16, noise_scale=0.0):
        batch_size = X.shape[0]
        device = X.device

        z_t = torch.zeros(batch_size, self.latent_dim, device=device)
        u_seq_decoded = torch.zeros(batch_size, self.H, device=device)
        y_t = torch.zeros(batch_size, self.latent_dim, device=device)

        y_history = []

        for t in range(D):
            if noise_scale > 0.0:
                # Latent-space perturbation with wider clamp for meaningful diversity
                epsilon = torch.randn_like(z_t) * noise_scale
                epsilon = torch.clamp(epsilon, min=-1.0, max=1.0)
                z_t = z_t + epsilon

            proj_z_input = self.W_x(X) + self.W_y(u_seq_decoded) + self.W_z(z_t)
            z_t = torch.tanh(self.recur_cell_z(proj_z_input))

            proj_y_input = self.M_y(y_t) + self.M_z(z_t)
            y_t = torch.tanh(self.recur_cell_y(proj_y_input))

            u_seq_decoded = self.f_O(y_t)

            # Output-space perturbation: directly perturb decoded action sequences
            # This ensures candidate diversity even when latent dynamics saturate
            if noise_scale > 0.0:
                u_noise = torch.randn_like(u_seq_decoded) * noise_scale * 0.5
                u_seq_decoded = u_seq_decoded + u_noise

            y_history.append((u_seq_decoded, y_t))

        return y_history
