# -*- coding: utf-8 -*-
"""
离线高精度数值 NMPC 专家求解器

基于 PyTorch 自动求导与 L-BFGS 的名义 NMPC 优化求解器，
用于生成离线训练数据和 Q-head 回归标签。

包含终端代价 P_f（通过离散代数 Riccati 方程求解），
使名义 Lyapunov 递减条件 (5.16) 可从标准 NMPC 稳定性理论推导。
"""

import torch
import torch.optim as optim


def _solve_dare(A, B, Q, R, max_iter=200, tol=1e-10):
    """
    迭代法求解离散代数 Riccati 方程 (DARE):
      P = A'PA - A'PB (R + B'PB)^{-1} B'PA + Q

    用于计算终端代价矩阵 P_f，使名义 NMPC 值函数满足 Lyapunov 递减条件。
    """
    P_curr = Q.clone()
    for _ in range(max_iter):
        BP = B.T @ P_curr
        BPB = BP @ B
        S = R + BPB
        K = torch.linalg.solve(S, BP @ A)
        P_new = A.T @ P_curr @ A - A.T @ P_curr @ B @ K + Q
        if torch.norm(P_new - P_curr) < tol:
            return P_new
        P_curr = P_new
    return P_curr


class GoldenNMPCSolver:
    """基于 PyTorch 自动求导与 L-BFGS 的名义 NMPC 优化求解器（含终端代价）"""

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

        # 终端代价 P_f: 通过 DARE 求解线性化系统的最优终端权重
        # 线性化动力学 (围绕平衡点): A = [0, I; 0, -b/m * I], B = [0; I/m]
        # 使用缩放因子 kappa_f 使终端代价与阶段代价量级匹配，
        # 避免 P_f 主导优化导致忽略障碍物约束
        dt = env.dt
        b_m = env.b_drag / env.m
        A = torch.zeros(6, 6)
        A[0:3, 3:6] = torch.eye(3)
        A[3:6, 3:6] = torch.eye(3) - dt * b_m * torch.eye(3)
        A[0:3, 0:3] = torch.eye(3)
        B = torch.zeros(6, 3)
        B[3:6, :] = dt / env.m * torch.eye(3)
        P_dare = _solve_dare(A, B, self.Q_cost, self.R_cost * torch.eye(3))
        # 缩放因子: 使终端代价与 horizon 步阶段代价量级相当
        kappa_f = 0.1
        self.P_f = kappa_f * P_dare

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

            # 终端代价: (x_H - x_sp)' P_f (x_H - x_sp)
            terminal_error = x_curr - x_sp
            cost += terminal_error.unsqueeze(0) @ self.P_f @ terminal_error.unsqueeze(1)

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
        # 终端代价
        terminal_error = x_curr - x_sp
        cost += terminal_error.unsqueeze(0) @ self.P_f @ terminal_error.unsqueeze(1)
        return cost.item()
