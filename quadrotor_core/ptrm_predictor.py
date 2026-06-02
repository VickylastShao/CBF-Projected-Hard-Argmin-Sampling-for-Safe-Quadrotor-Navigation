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
"""

import torch


class PTRMNMPCPredictor:
    """概率并行加噪最优筛选（PTRM-NMPC）在线决策单元"""

    def __init__(self, model, env, K=50, D=16, sigma=0.25, eta_hyst=0.05,
                 tracking_Kp=4.0, tracking_Kd=3.0):
        self.model = model
        self.env = env
        self.K = K
        self.D = D
        self.sigma = sigma
        self.eta_hyst = eta_hyst  # 轨迹空间级滞回抗路径抖动系数
        self.last_u_seq = None    # 保存上一步选择的最优动作序列副本

        # 内环 PD 追踪修正增益（论文 Section 2.B 微分平坦映射的离散化实现）
        # Kp 控制位置误差修正力度，Kd 控制速度阻尼
        # 对应 m * (Kp * e_p + Kd * e_v) 项，其中 m=1.5kg
        self.tracking_Kp = tracking_Kp
        self.tracking_Kd = tracking_Kd

    def reset(self):
        """重置动作轨迹历史，消除不同独立试验循环间的历史状态污染"""
        self.last_u_seq = None

    def _compute_tracking_correction(self, x_init, x_sp):
        """
        内环 PD 追踪修正（论文 Section 2.B 微分平坦映射）

        计算基于当前状态误差的修正力：
          u_corr = m * (Kp * (x_sp[:3] - x[:3]) + Kd * (x_sp[3:6] - x[3:6]))

        这在控制理论中是标准的内环PD + 外环规划架构：
        - 外环(PTRM): 提供参考轨迹/标称推力序列
        - 内环(PD): 修正跟踪误差，确保渐进稳定性

        在6D抽象动力学中，重力已被预补偿（Section 2.C），
        因此修正力直接作用于抽象速度空间。
        """
        e_p = x_sp[0:3] - x_init[0:3]
        e_v = x_sp[3:6] - x_init[3:6]
        u_corr = self.env.m * (self.tracking_Kp * e_p + self.tracking_Kd * e_v)
        return u_corr

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            # 自动设备搬运逻辑，完美支持 CPU/GPU 异构高频执行
            X_single = torch.cat([x_init.to(device), x_sp.to(device)]).unsqueeze(0)
            X_parallel = X_single.repeat(self.K, 1)

            # 多步递归产生 K 个并行候选动作
            y_history = self.model.forward_steps(X_parallel, D=self.D, noise_scale=self.sigma)
            u_candidates, final_latent_y = y_history[-1]  # u_candidates: (K, 30)

            # 内环 PD 修正：为每个候选的第一步控制添加基于当前误差的修正
            # 这确保即使 PTRM 网络输出因递归吸引子效应而趋同，
            # 闭环系统仍具备渐进跟踪稳定性
            u_tracking = self._compute_tracking_correction(x_init.cpu(), x_sp.cpu())

            # 对每个候选的第一步(3维)添加追踪修正
            # 修正力度通过 alpha_blend 系数与 PTRM 规划输出融合
            # alpha_blend=0.3 意味着 70% 来自 PTRM 规划，30% 来自 PD 修正
            alpha_blend = 0.3
            u_pd = u_tracking.unsqueeze(0).to(device)  # (1, 3)
            u_candidates_corrected = u_candidates.clone()
            u_first = u_candidates[:, 0:3]
            u_candidates_corrected[:, 0:3] = (1.0 - alpha_blend) * u_first + alpha_blend * u_pd

            # 使用回归 Q 头评估得分
            scores = self.model.f_Q(final_latent_y).squeeze(-1)  # Scores: (K,)

            # 【重要控制理论对齐】动作序列级滞回（Remark 7 严格代码落地）：
            # 直接在控制轨迹空间中惩罚偏离 receding-horizon 移位前驱最优动作的项，从而实现 ADT 约束证明
            if self.K > 1 and self.last_u_seq is not None:
                # 动作移位：向右平移一拍，尾部执行外推占位
                u_shift = torch.cat([self.last_u_seq[3:], self.last_u_seq[-3:]]).to(device)
                u_shift_batch = u_shift.unsqueeze(0).repeat(self.K, 1)

                # 计算与基准移位轨迹的 L2 范数物理偏差
                dist = torch.sum((u_candidates_corrected - u_shift_batch) ** 2, dim=1)
                scores = scores - self.eta_hyst * dist

            best_idx = torch.argmax(scores).item()
            best_u_sequence = u_candidates_corrected[best_idx]

            # 动态跟踪动作序列基准
            self.last_u_seq = best_u_sequence.clone()
            u_nominal = best_u_sequence[0:3].cpu()  # 移回CPU提供物理积分器步进

            if enable_cbf:
                u_safe = self.env.apply_cbf_projection(x_init.cpu(), u_nominal)
                safe_u_sequence = best_u_sequence.clone().cpu()
                safe_u_sequence[0:3] = u_safe
            else:
                u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)
                safe_u_sequence = torch.clamp(best_u_sequence.cpu(), self.env.u_min, self.env.u_max)

            return u_safe, safe_u_sequence
