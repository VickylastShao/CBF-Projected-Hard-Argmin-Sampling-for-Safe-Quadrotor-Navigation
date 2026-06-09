# -*- coding: utf-8 -*-
"""
MPPI (Model Predictive Path Integral) 基线控制器

实现标准 MPPI 采样控制策略，作为 PTRM-NMPC 的直接竞争基线。
MPPI 通过 K 个高斯扰动候选 + 重要性加权选择控制输入，
无需离线训练，是与 PTRM 最直接对比的采样式 MPC 方法。

参考: Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning", ICRA 2017
"""

import torch
import numpy as np
import time


class MPPIController:
    """MPPI 采样控制器 — PTRM-NMPC 的无训练基线对比"""

    def __init__(self, env, K=50, sigma=2.0, lam=0.1, rollout_steps=20,
                 Kp=4.0, Kd=3.0, obs_weight=2000.0, eta_hyst=0.05):
        """
        Args:
            env: QuadrotorDynamics 环境
            K: 候选数量
            sigma: 采样标准差 (动作空间, N)
            lam: MPPI 温度参数 (越小越贪心)
            rollout_steps: 前向 rollout 步数
            Kp, Kd: PD 基线增益
            obs_weight: 障碍物代价比重
            eta_hyst: 滞回系数
        """
        self.env = env
        self.K = K
        self.sigma = sigma
        self.lam = lam
        self.rollout_steps = rollout_steps
        self.Kp = Kp
        self.Kd = Kd
        self.obs_weight = obs_weight
        self.eta_hyst = eta_hyst
        self.last_u = None

        # 预计算常量
        self.m = env.m
        self.b_drag = env.b_drag
        self.dt = env.dt
        self.q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        self.R_U = 0.02

    def reset(self):
        self.last_u = None

    def _compute_pd_baseline(self, x_init, x_sp):
        """PD基线控制"""
        e_p = x_sp[0:3] - x_init[0:3]
        e_v = x_sp[3:6] - x_init[3:6]
        return self.m * (self.Kp * e_p + self.Kd * e_v)

    def _batch_rollout_cost(self, x_init, u_candidates, x_sp):
        """批量 rollout 代价评估（与v5实验完全一致）"""
        K_val = u_candidates.shape[0]
        x = x_init.unsqueeze(0).repeat(K_val, 1)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(K_val, 1)
        cost = torch.zeros(K_val)
        q = self.q_diag.unsqueeze(0)

        for s in range(self.rollout_steps):
            p = x[:, 0:3]
            v = x[:, 3:6]
            v_dot = u_candidates / self.m - (self.b_drag / self.m) * v
            p_next = p + self.dt * v
            v_next = v + self.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)

            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + self.R_U * torch.sum(u_candidates * u_candidates, dim=1)

            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs['p'], dtype=torch.float32).unsqueeze(0).repeat(K_val, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs['r']
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2

        return cost

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        """
        MPPI 决策：K个高斯扰动 → rollout代价评估 → 重要性加权

        与 PTRM 的关键区别：
        - MPPI 使用重要性加权平均（软选择），而非 argmin（硬选择）
        - MPPI 不需要离线训练
        - MPPI 的候选生成在动作空间，而非潜在空间
        """
        u_pd = self._compute_pd_baseline(x_init, x_sp)

        if self.K == 1:
            u_nominal = u_pd
        else:
            # 生成 K 个候选 (高斯扰动)
            noise = torch.randn(self.K, 3) * self.sigma
            u_candidates = u_pd.unsqueeze(0) + noise

            # Rollout 代价评估
            cost = self._batch_rollout_cost(x_init, u_candidates, x_sp)

            # 滞回惩罚
            if self.last_u is not None:
                dist = torch.sum((u_candidates - self.last_u.unsqueeze(0)) ** 2, dim=1)
                cost = cost + self.eta_hyst * dist

            # MPPI 重要性加权
            # w_k = exp(-1/λ * (cost_k - min_cost)) / Σ exp(-1/λ * (cost_j - min_cost))
            cost_shifted = cost - torch.min(cost)  # 数值稳定性
            weights = torch.exp(-cost_shifted / self.lam)
            weights = weights / torch.sum(weights)

            # 加权平均控制
            u_nominal = torch.sum(weights.unsqueeze(1) * u_candidates, dim=0)

        self.last_u = u_nominal.clone()

        # CBF 安全投影
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
