# -*- coding: utf-8 -*-
"""
在线概率加噪并行推理决策单元

概率并行加噪最优筛选（PTRM-NMPC）在线决策单元，
包含轨迹空间级滞回抗路径抖动机制（手稿 Remark 7 严格代码落地）。

设计理念（与论文 Section 3.D 对齐）：
  PTRM 网络提供外环 NMPC 轨迹规划（规划层面的力序列），
  内环通过微分平坦映射 + 重力预补偿修正跟踪误差。
  6D 抽象动力学中重力已被预补偿（不在 v_dot 中重复扣除），
  网络输出直接对应总推力分量。

v6 修订核心架构变更（针对审稿意见 R1, R3, S1 对齐）：
  新增候选生成模式 candidate_mode:
    'trm' (默认) — TRM递归产生候选 + PD修正融合（原手稿架构）
                   alpha_blend 控制PD修正比例
    'pd'         — PD基线 + 高斯扰动产生候选 + TRM Q-head/rollout评估
                   TRM在评估层面发挥作用而非直接控制输出
                   这种模式下 alpha_blend 含义：1.0=纯PD(无TRM评估) 0.0=完全TRM评估
    'trm_pd'     — TRM-PD混合基线 + 高斯扰动候选 + Rollout评估
                   闭环训练TRM提供状态依赖策略先验，PD提供实时反馈稳定性
                   alpha_blend 控制PD修正比例: u_base = (1-α)*TRM + α*PD

  两阶段候选评估策略（S1, R3）：
    阶段1: Q-head 粗筛 top-M 候选（低成本代理）
    阶段2: rollout 代价精排 top-M 中最优（精确评估）
  当 use_rollout_cost=False 时回退到纯 Q-head 排序（消融用）。

  PTRM vs MPPI 架构对比：
    PTRM ('pd' 模式):
      候选生成: PD + 高斯扰动 (与MPPI相同)
      评估: Q-head粗筛top-M → rollout精排 (vs MPPI: 全量rollout + 加权平均)
      优势: Q-head预筛减少rollout计算量 (K→M, M≪K)
             潜在空间结构化信息提供更好的排序质量

    PTRM ('trm' 模式):
      候选生成: TRM递归 + 潜在空间扰动提供结构化多样性
      评估: Q-head粗筛 → rollout精排
      优势: TRM时间一致性 + 结构化候选空间
"""

import torch
import numpy as np


class PTRMNMPCPredictor:
    """概率并行加噪最优筛选（PTRM-NMPC）在线决策单元"""

    def __init__(self, model, env, K=50, D=16, sigma=0.25, eta_hyst=0.05,
                 tracking_Kp=4.0, tracking_Kd=3.0, alpha_blend=0.3,
                 noise_mode='both', candidate_mode='trm',
                 use_rollout_cost=True, rollout_top_m=10,
                 rollout_steps=20, obs_weight=2000.0,
                 pd_sigma=2.0, ranking_mode='q_head'):
        self.model = model
        self.env = env
        self.K = K
        self.D = D
        self.sigma = sigma
        self.eta_hyst = eta_hyst  # 轨迹空间级滞回抗路径抖动系数
        self.last_u_seq = None    # 保存上一步选择的最优动作序列副本

        # 内环 PD 追踪修正增益（论文 Section 2.B 微分平坦映射的离散化实现）
        self.tracking_Kp = tracking_Kp
        self.tracking_Kd = tracking_Kd

        # PD修正融合系数（论文 Section 4.D）
        # candidate_mode='trm' 时:
        #   alpha_blend=0.3 意味着 70% 来自 PTRM 规划，30% 来自 PD 修正
        # candidate_mode='pd' 时:
        #   alpha_blend 控制TRM评估的信任度（消融用）
        self.alpha_blend = alpha_blend

        # 候选生成模式（路径D: 去Q-head架构修订）
        # 'trm': TRM递归产生候选 + PD修正（原手稿架构，失败：0%成功率）
        # 'pd': PD+高斯扰动产生候选 + Rollout评估（v6修订）
        # 'trm_v2': TRM确定性输出 + 输出空间噪声候选 + W_y编码 + Q-head评估
        #           （路径C：候选条件化架构，Q-head失效——递归吸引子抹平候选差异）
        # 'trm_rollout': TRM确定性输出 + 输出空间噪声候选 + 纯Rollout评估
        #                （路径D：去Q-head，TRM提供策略基线，Rollout排序候选）
        # 'trm_pd': TRM-PD混合基线 + 高斯扰动候选 + 纯Rollout评估
        #           闭环训练TRM提供状态依赖策略先验 + PD实时反馈稳定性
        #           alpha_blend控制PD比例: u_base = (1-α)*TRM + α*PD
        self.candidate_mode = candidate_mode

        # PD候选模式下的高斯扰动强度（动作空间量纲，N）
        self.pd_sigma = pd_sigma

        # trm_v2 模式下的输出空间噪声强度（N）
        # 应与训练时 output_noise_sigma 匹配
        self.output_noise_sigma = pd_sigma  # 默认复用pd_sigma=2.0

        # 噪声注入模式（消融用）
        self.noise_mode = noise_mode

        # 两阶段评估策略（v6修订）
        self.use_rollout_cost = use_rollout_cost
        self.rollout_top_m = min(rollout_top_m, K)
        self.rollout_steps = rollout_steps
        self.obs_weight = obs_weight

        # 候选排序模式（路径A验证实验用）
        # 'q_head': Q-head评分排序（默认，当前实现）
        # 'random': 随机排序（消融对照：验证Q-head是否提供有信息量的排序信号）
        # 'rollout_all': 全量rollout排序（上界参照：跳过Q-head，直接评估所有K个候选）
        self.ranking_mode = ranking_mode

        # Rollout 代价评估常量
        self.q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        self.R_U = 0.02

    def reset(self):
        """重置动作轨迹历史，消除不同独立试验循环间的历史状态污染"""
        self.last_u_seq = None

    def _compute_tracking_correction(self, x_init, x_sp):
        """
        内环 PD 追踪修正（论文 Section 2.B 微分平坦映射）

        计算基于当前状态误差的修正力：
          u_corr = m * (Kp * (x_sp[:3] - x[:3]) + Kd * (x_sp[3:6] - x[3:6]))
        """
        e_p = x_sp[0:3] - x_init[0:3]
        e_v = x_sp[3:6] - x_init[3:6]
        u_corr = self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)
        return u_corr

    def _batch_rollout_cost(self, x_init, u_first_candidates, x_sp):
        """
        批量 rollout 代价评估（用于Q-head粗筛后的精排阶段）

        对每个候选的第一步控制 u_first 执行前向 rollout，
        后续步使用PD基线控制。

        Args:
            x_init: 当前状态 (6,)
            u_first_candidates: M个候选第一步控制 (M, 3)
            x_sp: 目标设定点 (6,)

        Returns:
            costs: (M,) 代价向量
        """
        M = u_first_candidates.shape[0]
        x = x_init.unsqueeze(0).repeat(M, 1)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(M, 1)
        cost = torch.zeros(M)
        q = self.q_diag.unsqueeze(0)

        for s in range(self.rollout_steps):
            if s == 0:
                u = u_first_candidates
            else:
                # 后续步使用PD基线控制
                e_p = x_sp6[:, 0:3] - x[:, 0:3]
                e_v = x_sp6[:, 3:6] - x[:, 3:6]
                u = self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)

            u = torch.clamp(u, self.env.u_min, self.env.u_max)
            p = x[:, 0:3]
            v = x[:, 3:6]
            # 一阶Euler积分
            v_dot = u / self.env.m - (self.env.b_drag / self.env.m) * v
            p_next = p + self.env.dt * v
            v_next = v + self.env.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)

            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + self.R_U * torch.sum(u * u, dim=1)

            # 障碍物代价
            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs['p'], dtype=torch.float32).unsqueeze(0).repeat(M, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs['r']
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2

        return cost

    def _generate_candidates_trm(self, x_init, x_sp, device):
        """
        TRM候选生成模式（原手稿架构）

        TRM递归产生K个候选动作序列 + PD修正融合
        """
        X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
        X_parallel = X_single.repeat(self.K, 1)

        # 多步递归产生 K 个并行候选动作
        y_history = self.model.forward_steps(X_parallel, D=self.D,
                                              noise_scale=self.sigma,
                                              noise_mode=self.noise_mode)
        u_candidates, final_latent_y = y_history[-1]  # u_candidates: (K, 30)

        # PD修正融合
        if self.alpha_blend > 0:
            u_tracking = self._compute_tracking_correction(x_init.cpu(), x_sp.cpu())
            u_pd = u_tracking.unsqueeze(0).to(device)
            u_candidates_corrected = u_candidates.clone()
            u_first = u_candidates[:, 0:3]
            u_candidates_corrected[:, 0:3] = (1.0 - self.alpha_blend) * u_first + self.alpha_blend * u_pd
        else:
            u_candidates_corrected = u_candidates

        return u_candidates_corrected, final_latent_y

    def _generate_candidates_pd(self, x_init, x_sp, device):
        """
        PD候选生成模式（v6修订架构）

        PD基线 + 高斯扰动产生K个候选，TRM Q-head/rollout评估排序
        TRM在评估层面发挥作用，不直接输出控制

        候选生成与MPPI相同（PD+高斯），但评估方式不同：
        - PTRM: Q-head粗筛 → rollout精排（两阶段，计算高效）
        - MPPI: 全量rollout + 重要性加权平均
        """
        u_pd = self._compute_tracking_correction(x_init.cpu(), x_sp.cpu())

        if self.K == 1:
            # K=1时直接返回PD基线（每3维一步，共10步=30维）
            u_pd_30 = u_pd.unsqueeze(0).repeat(1, 10)  # (1, 30)
            u_candidates = u_pd_30.to(device)
            # K=1时无需评估，但为保持一致性仍传入候选序列
            X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
            y_history = self.model.forward_steps(X_single, D=self.D,
                                                  noise_scale=0.0,
                                                  noise_mode='none',
                                                  u_seq_external=u_candidates)
            _, final_latent_y = y_history[-1]
            return u_candidates, final_latent_y

        # 生成K个PD+高斯候选（与MPPI相同的候选生成）
        noise = torch.randn(self.K, 3) * self.pd_sigma
        u_first_candidates = u_pd.unsqueeze(0) + noise  # (K, 3)

        # 扩展为30维动作序列（第一步=候选，后续=PD基线）
        u_pd_full = u_pd.unsqueeze(0).repeat(self.K, 1)  # (K, 3)
        u_candidates = torch.zeros(self.K, 30, device=device)
        u_candidates[:, 0:3] = u_first_candidates.to(device)
        for i in range(10):
            u_candidates[:, i*3:(i+1)*3] = u_pd_full.to(device) if i > 0 else u_first_candidates.to(device)

        # 执行TRM前向获取Q-head评分所需的潜在表示
        X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
        X_parallel = X_single.repeat(self.K, 1)

        # 关键修复：将PD候选序列通过W_y注入TRM递归过程
        # 使Q-head评分基于PD候选而非TRM自身解码输出
        # 这解决了Q-head/候选架构断裂问题：
        # 修复前：Q-head评估TRM幻觉（与PD候选解耦）
        # 修复后：Q-head评估PD候选（通过W_y编码的信息）
        y_history = self.model.forward_steps(X_parallel, D=self.D,
                                              noise_scale=self.sigma,
                                              noise_mode=self.noise_mode,
                                              u_seq_external=u_candidates)
        _, final_latent_y = y_history[-1]

        return u_candidates, final_latent_y

    def _generate_candidates_trm_v2(self, x_init, x_sp, device):
        """
        TRM-v2候选生成模式（路径C：候选条件化架构）

        核心设计：
        1. 确定性TRM推理获取策略基线 u_base
        2. 在输出空间添加高斯噪声生成K个候选
        3. 通过W_y编码将候选信息注入TRM递归
        4. Q-head基于候选-条件化隐状态评分

        与训练流程保持一致：
        - 训练时：u_base + output_noise → W_y编码 → Q-head学习排序
        - 推理时：u_base + output_noise → W_y编码 → Q-head评估排序
        → 消除训练/推理分布不匹配问题

        优势：
        - 候选多样性来自输出空间噪声（不破坏递归动力学）
        - Q-head能区分不同候选（W_y编码提供候选信息）
        - TRM策略基线质量接近专家NMPC（已验证）
        """
        X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)

        if self.K == 1:
            # K=1时确定性推理，无需噪声
            y_history = self.model.forward_steps(X_single, D=self.D,
                                                  noise_scale=0.0,
                                                  noise_mode='none')
            u_candidates, final_latent_y = y_history[-1]
            return u_candidates, final_latent_y

        # Step 1: 确定性推理获取策略基线
        with torch.no_grad():
            y_det = self.model.forward_steps(X_single, D=self.D,
                                              noise_scale=0.0,
                                              noise_mode='none')
        u_base = y_det[-1][0].squeeze()  # (30,) TRM策略基线

        # Step 2: 在输出空间添加噪声生成K个候选
        u_candidates = u_base.unsqueeze(0).repeat(self.K, 1)  # (K, 30)

        # 第一步噪声较大（主要决策步骤）
        first_noise = torch.randn(self.K, 3) * self.output_noise_sigma
        u_candidates[:, 0:3] = u_candidates[:, 0:3] + first_noise.to(device)

        # 后续步骤噪声递减（开环预测的可信度递减）
        for step_i in range(1, 10):
            decay = max(0.3, 1.0 - step_i * 0.1)
            step_noise = torch.randn(self.K, 3) * self.output_noise_sigma * decay
            u_candidates[:, step_i*3:(step_i+1)*3] = (
                u_candidates[:, step_i*3:(step_i+1)*3] + step_noise.to(device)
            )

        # Step 3: 通过W_y编码将候选信息注入TRM递归
        X_parallel = X_single.repeat(self.K, 1)
        y_history = self.model.forward_steps(X_parallel, D=self.D,
                                              noise_scale=0.0,
                                              noise_mode='none',
                                              u_seq_external=u_candidates)
        _, final_latent_y = y_history[-1]

        return u_candidates, final_latent_y

    def _generate_candidates_trm_rollout(self, x_init, x_sp, device):
        """
        TRM-Rollout候选生成模式（路径D：去Q-head架构）

        核心设计：
        1. TRM确定性推理获取策略基线 u_base（比PD基线质量更高）
        2. 在输出空间添加高斯噪声生成K个候选
        3. 纯Rollout代价评估排序（不需要Q-head）

        优势：
        - TRM递归提供高质量策略先验（vs PD的简单线性反馈）
        - 候选多样性来自输出空间噪声
        - Rollout排序直接反映控制质量
        - 无Q-head训练/推理不匹配问题
        """
        X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)

        if self.K == 1:
            y_history = self.model.forward_steps(X_single, D=self.D,
                                                  noise_scale=0.0,
                                                  noise_mode='none')
            u_candidates, final_latent_y = y_history[-1]
            return u_candidates, final_latent_y

        # Step 1: 确定性推理获取策略基线
        with torch.no_grad():
            y_det = self.model.forward_steps(X_single, D=self.D,
                                              noise_scale=0.0,
                                              noise_mode='none')
        u_base = y_det[-1][0].squeeze()  # (30,) TRM策略基线

        # Step 2: 在输出空间添加噪声生成K个候选
        u_candidates = u_base.unsqueeze(0).repeat(self.K, 1)  # (K, 30)

        # 第一步噪声（主要决策步骤，sigma较大）
        first_noise = torch.randn(self.K, 3) * self.output_noise_sigma
        u_candidates[:, 0:3] = u_candidates[:, 0:3] + first_noise.to(device)

        # 后续步骤噪声递减
        for step_i in range(1, 10):
            decay = max(0.3, 1.0 - step_i * 0.1)
            step_noise = torch.randn(self.K, 3) * self.output_noise_sigma * decay
            u_candidates[:, step_i*3:(step_i+1)*3] = (
                u_candidates[:, step_i*3:(step_i+1)*3] + step_noise.to(device)
            )

        # trm_rollout模式不需要final_latent_y（Q-head不参与排序）
        # 但保留接口一致性
        final_latent_y = None
        return u_candidates, final_latent_y

    def _generate_candidates_trm_pd(self, x_init, x_sp, device):
        """
        TRM-PD混合候选生成模式

        核心设计：
        1. 闭环训练TRM确定性推理获取状态依赖策略基线 u_trm
        2. PD反馈计算实时修正 u_pd
        3. 混合基线: u_base = (1-α)*u_trm + α*u_pd（alpha_blend控制PD比例）
        4. 在混合基线上添加高斯噪声生成K个候选
        5. 纯Rollout代价评估排序

        优势：
        - TRM提供状态依赖策略先验（学习到的障碍物回避、制动力等）
        - PD提供实时反馈稳定性（无模型、无训练依赖、保证收敛方向）
        - 混合基线比纯PD或纯TRM更具信息量
        - K候选 + Rollout评估提供test-time compute scaling
        """
        X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)

        # PD反馈修正
        u_pd = self._compute_tracking_correction(x_init.cpu(), x_sp.cpu())  # (3,)

        if self.K == 1:
            # K=1时返回混合基线（无噪声）
            with torch.no_grad():
                y_det = self.model.forward_steps(X_single, D=self.D,
                                                  noise_scale=0.0,
                                                  noise_mode='none')
            u_trm_first = y_det[-1][0].squeeze()[0:3]  # TRM第一步输出 (3,)
            # 混合基线
            u_base_first = (1.0 - self.alpha_blend) * u_trm_first.cpu() + self.alpha_blend * u_pd
            # 扩展为30维序列
            u_pd_full = u_pd.unsqueeze(0).repeat(1, 10)  # (1, 30)
            u_candidates = u_pd_full.clone()
            u_candidates[0, 0:3] = u_base_first
            u_candidates = u_candidates.to(device)
            final_latent_y = None
            return u_candidates, final_latent_y

        # Step 1: 确定性TRM推理获取策略基线
        with torch.no_grad():
            y_det = self.model.forward_steps(X_single, D=self.D,
                                              noise_scale=0.0,
                                              noise_mode='none')
        u_trm = y_det[-1][0].squeeze()  # (30,) TRM完整策略序列

        # Step 2: 构建混合基线
        # 第一步: TRM + PD混合
        u_base_first = (1.0 - self.alpha_blend) * u_trm[0:3].cpu() + self.alpha_blend * u_pd

        # 扩展为30维候选序列
        # 第一步用混合基线，后续步用PD（因为TRM后续步是开环预测，PD更稳定）
        u_candidates = torch.zeros(self.K, 30, device=device)
        u_pd_full = u_pd.unsqueeze(0).repeat(self.K, 1)  # (K, 3)
        for i in range(10):
            if i == 0:
                u_candidates[:, 0:3] = u_base_first.unsqueeze(0).repeat(self.K, 1).to(device)
            else:
                # 后续步: 混合TRM开环预测和PD（衰减TRM权重）
                trm_weight = (1.0 - self.alpha_blend) * max(0.0, 1.0 - i * 0.2)
                pd_weight = 1.0 - trm_weight
                u_step = trm_weight * u_trm[i*3:(i+1)*3].cpu() + pd_weight * u_pd
                u_candidates[:, i*3:(i+1)*3] = u_step.unsqueeze(0).repeat(self.K, 1).to(device)

        # Step 3: 在第一步控制上添加高斯噪声生成K个候选
        noise = torch.randn(self.K, 3) * self.pd_sigma
        u_candidates[:, 0:3] = u_candidates[:, 0:3] + noise.to(device)

        # 后续步也可以添加较小噪声（增强多样性）
        for step_i in range(1, 10):
            decay = max(0.2, 1.0 - step_i * 0.15)
            step_noise = torch.randn(self.K, 3) * self.pd_sigma * decay * 0.3
            u_candidates[:, step_i*3:(step_i+1)*3] = (
                u_candidates[:, step_i*3:(step_i+1)*3] + step_noise.to(device)
            )

        final_latent_y = None
        return u_candidates, final_latent_y

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            # ===== 候选生成 =====
            if self.candidate_mode == 'pd':
                u_candidates_corrected, final_latent_y = self._generate_candidates_pd(
                    x_init, x_sp, device)
            elif self.candidate_mode == 'trm_v2':
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm_v2(
                    x_init, x_sp, device)
            elif self.candidate_mode == 'trm_rollout':
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm_rollout(
                    x_init, x_sp, device)
            elif self.candidate_mode == 'trm_pd':
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm_pd(
                    x_init, x_sp, device)
            else:
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm(
                    x_init, x_sp, device)

            # ===== 候选排序 =====
            if self.candidate_mode in ('trm_rollout', 'trm_pd'):
                # 路径D：纯Rollout排序（无需Q-head）
                # 直接评估所有K个候选的rollout代价
                if self.K > 1:
                    u_first_all = u_candidates_corrected[:, 0:3].cpu()
                    rollout_costs_all = self._batch_rollout_cost(x_init.cpu(), u_first_all, x_sp.cpu())
                    scores = -rollout_costs_all  # 取负：cost越低 → score越高
                else:
                    scores = torch.tensor([1.0])
            elif self.ranking_mode == 'random':
                # 随机排序模式：用随机排列代替Q-head排序
                # 目的：验证Q-head排序是否比随机选择更有信息量
                scores = torch.rand(self.K)
            elif self.ranking_mode == 'rollout_all':
                # 全量rollout模式：跳过Q-head，直接评估所有K个候选
                # 目的：提供上界参照（理想排序）
                u_first_all = u_candidates_corrected[:, 0:3].cpu()
                rollout_costs_all = self._batch_rollout_cost(x_init.cpu(), u_first_all, x_sp.cpu())
                scores = -rollout_costs_all  # 取负使得topk选出最小rollout代价
            else:
                # Q-head评分模式（默认）
                # 注意：候选条件化训练后，Q-head预测归一化cost（越高越差）
                # 因此排序取argmin，需要取负以适配topk(argmax)逻辑
                if final_latent_y is not None:
                    q_values = self.model.f_Q(final_latent_y).squeeze(-1)  # (K,)
                    scores = -q_values  # 取负：cost越低 → score越高 → 被topk选中
                else:
                    # 无Q-head可用时回退到rollout排序
                    if self.K > 1:
                        u_first_all = u_candidates_corrected[:, 0:3].cpu()
                        rollout_costs_all = self._batch_rollout_cost(x_init.cpu(), u_first_all, x_sp.cpu())
                        scores = -rollout_costs_all
                    else:
                        scores = torch.tensor([1.0])

            # ===== 滞回正则化（Remark 7 严格代码落地） =====
            if self.K > 1 and self.last_u_seq is not None:
                u_shift = torch.cat([self.last_u_seq[3:], self.last_u_seq[-3:]]).to(device)
                u_shift_batch = u_shift.unsqueeze(0).repeat(self.K, 1)
                dist = torch.sum((u_candidates_corrected - u_shift_batch) ** 2, dim=1).to(scores.device)
                scores = scores - self.eta_hyst * dist

            # ===== 候选评估与选择 =====
            if self.candidate_mode in ('trm_rollout', 'trm_pd') and self.K > 1:
                # 路径D：纯Rollout排序已在上面完成，直接选最优
                best_idx = torch.argmax(scores).item()
            elif self.use_rollout_cost and self.K > 1:
                if self.ranking_mode == 'rollout_all':
                    # 全量rollout已在上一步完成，直接选最优
                    best_idx = torch.argmax(scores).item()
                else:
                    # 两阶段评估：Q-head粗筛 → Rollout精排
                    M = min(self.rollout_top_m, self.K)
                    _, top_indices = torch.topk(scores, min(M, self.K))
                    top_indices = top_indices.sort()[0]

                    # 阶段2: Rollout 代价精排 — 在 top-M 中选出最优
                    u_first_top = u_candidates_corrected[top_indices, 0:3].cpu()
                    rollout_costs = self._batch_rollout_cost(x_init.cpu(), u_first_top, x_sp.cpu())
                    best_in_top = torch.argmin(rollout_costs).item()
                    best_idx = top_indices[best_in_top].item()
            else:
                # 纯排序（消融模式 / K=1）
                best_idx = torch.argmax(scores).item()

            best_u_sequence = u_candidates_corrected[best_idx]

            # 动态跟踪动作序列基准
            self.last_u_seq = best_u_sequence.clone()
            u_nominal = best_u_sequence[0:3].cpu()

            if enable_cbf:
                u_safe = self.env.apply_cbf_projection(x_init.cpu(), u_nominal)
                safe_u_sequence = best_u_sequence.clone().cpu()
                safe_u_sequence[0:3] = u_safe
            else:
                u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
                safe_u_sequence = torch.clamp(best_u_sequence.cpu(), self.env.u_min, self.env.u_max)

            return u_safe, safe_u_sequence

    def predict_action_with_correlation(self, x_init, x_sp, enable_cbf=True):
        """
        带相关性数据采集的动作预测（路径A验证实验2专用）

        与 predict_action() 相同的决策流程，但额外返回：
        - q_head_scores: K个候选的Q-head评分
        - pd_rollout_costs: K个PD候选的rollout代价
        用于计算Q-head对PD候选的排序相关性（Spearman ρ, Pearson r）

        Returns:
            u_safe, safe_u_sequence, q_head_scores, pd_rollout_costs
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            # ===== 候选生成 =====
            if self.candidate_mode == 'pd':
                u_candidates_corrected, final_latent_y = self._generate_candidates_pd(
                    x_init, x_sp, device)
            elif self.candidate_mode == 'trm_v2':
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm_v2(
                    x_init, x_sp, device)
            elif self.candidate_mode == 'trm_rollout':
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm_rollout(
                    x_init, x_sp, device)
            elif self.candidate_mode == 'trm_pd':
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm_pd(
                    x_init, x_sp, device)
            else:
                u_candidates_corrected, final_latent_y = self._generate_candidates_trm(
                    x_init, x_sp, device)

            # ===== 采集Q-head评分 =====
            # Q-head预测归一化cost（越高越差），返回原始值供相关性分析
            q_head_scores = self.model.f_Q(final_latent_y).squeeze(-1)  # (K,)

            # ===== 采集PD候选的rollout代价 =====
            if self.K > 1:
                u_first_all = u_candidates_corrected[:, 0:3].cpu()
                pd_rollout_costs = self._batch_rollout_cost(x_init.cpu(), u_first_all, x_sp.cpu())
            else:
                pd_rollout_costs = torch.zeros(1)

            # ===== 正常决策流程 =====
            scores = -q_head_scores.clone()  # 取负：cost越低 → score越高

            if self.K > 1 and self.last_u_seq is not None:
                u_shift = torch.cat([self.last_u_seq[3:], self.last_u_seq[-3:]]).to(device)
                u_shift_batch = u_shift.unsqueeze(0).repeat(self.K, 1)
                dist = torch.sum((u_candidates_corrected - u_shift_batch) ** 2, dim=1)
                scores = scores - self.eta_hyst * dist

            if self.use_rollout_cost and self.K > 1:
                M = min(self.rollout_top_m, self.K)
                _, top_indices = torch.topk(scores, min(M, self.K))
                top_indices = top_indices.sort()[0]
                u_first_top = u_candidates_corrected[top_indices, 0:3].cpu()
                rollout_costs = self._batch_rollout_cost(x_init.cpu(), u_first_top, x_sp.cpu())
                best_in_top = torch.argmin(rollout_costs).item()
                best_idx = top_indices[best_in_top].item()
            else:
                best_idx = torch.argmax(scores).item()

            best_u_sequence = u_candidates_corrected[best_idx]
            self.last_u_seq = best_u_sequence.clone()
            u_nominal = best_u_sequence[0:3].cpu()

            if enable_cbf:
                u_safe = self.env.apply_cbf_projection(x_init.cpu(), u_nominal)
                safe_u_sequence = best_u_sequence.clone().cpu()
                safe_u_sequence[0:3] = u_safe
            else:
                u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
                safe_u_sequence = torch.clamp(best_u_sequence.cpu(), self.env.u_min, self.env.u_max)

            return u_safe, safe_u_sequence, q_head_scores.cpu(), pd_rollout_costs
