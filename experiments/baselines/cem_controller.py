# -*- coding: utf-8 -*-
"""
CEM (Cross-Entropy Method) 基线控制器

实现迭代精化的采样控制策略，作为 PTRM-NMPC 的真测试时计算缩放基线。
CEM 通过多轮采样→评估→精英选择→分布更新实现迭代改进，
总计算量 = n_iter × K，是与 PTRM 的 K-Scaling 最直接的对比。

参考: De Boer et al., "A Tutorial on the Cross-Entropy Method", Annals of OR 2005
"""

import torch
import numpy as np
import time


class CEMController:
    """CEM 迭代精化控制器 — PTRM-NMPC 的真测试时缩放基线"""

    def __init__(self, env, K=50, n_iter=3, sigma=2.0, elite_frac=0.2,
                 rollout_steps=20, Kp=4.0, Kd=3.0, obs_weight=2000.0, eta_hyst=0.05):
        """
        Args:
            env: QuadrotorDynamics 环境
            K: 每轮迭代采样数
            n_iter: CEM 迭代轮数 (总计算量 = n_iter × K)
            sigma: 初始采样标准差 (动作空间, N)
            elite_frac: 精英比例 (0.2 = top 20%)
            rollout_steps: 前向 rollout 步数
            Kp, Kd: PD 基线增益
            obs_weight: 障碍物代价比重
            eta_hyst: 滞回系数
        """
        self.env = env
        self.K = K
        self.n_iter = n_iter
        self.sigma = sigma
        self.elite_frac = elite_frac
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
        """批量 rollout 代价评估（与MPPI一致）"""
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
        CEM 决策：迭代采样 → 代价评估 → 精英选择 → 分布更新

        与 PTRM 的关键区别：
        - CEM 使用迭代精化（n_iter轮），而非 PTRM 的单次大量并行候选
        - CEM 的总计算量 = n_iter × K，可比性更强
        - CEM 每轮缩小采样分布，逐步聚焦到最优区域
        """
        u_pd = self._compute_pd_baseline(x_init, x_sp)

        # 初始采样分布以PD基线为中心
        mu = u_pd.clone()
        sigma_curr = self.sigma

        n_elite = max(1, int(self.K * self.elite_frac))

        for iteration in range(self.n_iter):
            # 采样 K 个候选
            noise = torch.randn(self.K, 3) * sigma_curr
            u_candidates = mu.unsqueeze(0) + noise

            # Rollout 代价评估
            cost = self._batch_rollout_cost(x_init, u_candidates, x_sp)

            # 滞回惩罚
            if self.last_u is not None:
                dist = torch.sum((u_candidates - self.last_u.unsqueeze(0)) ** 2, dim=1)
                cost = cost + self.eta_hyst * dist

            # 选择精英（代价最低的 top elite_frac）
            elite_indices = torch.argsort(cost)[:n_elite]
            elite_samples = u_candidates[elite_indices]

            # 更新分布参数
            mu = torch.mean(elite_samples, dim=0)
            sigma_curr = max(0.1, torch.std(elite_samples, dim=0).mean().item())

        u_nominal = mu
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
