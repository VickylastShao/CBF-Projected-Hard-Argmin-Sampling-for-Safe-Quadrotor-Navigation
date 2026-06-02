# -*- coding: utf-8 -*-
"""
离线高精度数值 NMPC 专家求解器

基于 PyTorch 自动求导与 L-BFGS 的名义 NMPC 优化求解器，
用于生成离线训练数据和 Q-head 回归标签。
"""

import torch
import torch.optim as optim


class GoldenNMPCSolver:
    """基于 PyTorch 自动求导与 L-BFGS 的名义 NMPC 优化求解器"""

    def __init__(self, env, horizon=10):
        self.env = env
        self.H = horizon
        self.Q_cost = torch.tensor([[15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 15.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 15.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self.R_cost = 0.02

    def solve(self, x_init, x_sp, use_mismatch=False):
        u_seq = torch.zeros(self.H * 3, requires_grad=True)
        optimizer = optim.LBFGS([u_seq], lr=0.08, max_iter=40, tolerance_grad=1e-5)

        def closure():
            optimizer.zero_grad()
            cost = 0.0
            x_curr = x_init.clone()

            for i in range(self.H):
                u = u_seq[i*3 : (i+1)*3]
                x_curr = self.env.step_discrete(x_curr, u, use_mismatch=use_mismatch)
                error = x_curr - x_sp
                cost += error.unsqueeze(0) @ self.Q_cost @ error.unsqueeze(1)
                cost += self.R_cost * torch.sum(u ** 2)

            cost.backward()
            return cost

        optimizer.step(closure)
        return torch.clamp(u_seq.detach(), self.env.u_min, self.env.u_max)

    def evaluate_cost(self, x_init, x_sp, u_sequence):
        """计算给定输入序列的实际 NMPC 轨迹代价值 (用于回归模型的 Q 值对齐)"""
        cost = 0.0
        x_curr = x_init.clone()
        steps = min(self.H, len(u_sequence) // 3)
        for i in range(steps):
            u = u_sequence[i*3 : (i+1)*3]
            x_curr = self.env.step_discrete(x_curr, u)
            error = x_curr - x_sp
            cost += error.unsqueeze(0) @ self.Q_cost @ error.unsqueeze(1)
            cost += self.R_cost * torch.sum(u ** 2)
        return cost.item()
