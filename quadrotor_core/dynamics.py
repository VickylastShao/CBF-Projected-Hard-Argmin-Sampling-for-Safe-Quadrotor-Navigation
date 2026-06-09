# -*- coding: utf-8 -*-
"""
6维四轴飞行器平移动力学模拟器（集成 DT-CCBF 安全投影）

精确相对度对齐（DT-CCBF）：基于相对度2的位置动力学雅可比，
计算解析控制雅可比，与手稿公式 (4.14) 的修正取得 100% 数学一致性对齐。
"""

import numpy as np
import torch


class QuadrotorDynamics:
    """6维四轴飞行器外环平移动力学模拟器，集成多障碍物环境与安全缓冲区保护机制"""

    def __init__(self, m=1.5, b_drag=0.1, dt=0.02, obstacles=None):
        self.g = 9.81        # 重力加速度 (m/s^2)
        self.m = m           # 标称质量 (kg) (支持在实验中引入物理参数失配)
        self.b_drag = b_drag # 空气摩擦阻尼因数 (N*s/m)
        self.dt = dt         # 仿真离散步长 (s)

        # 物理限制
        self.u_max = 15.0    # 最大虚拟控制力 (N)
        self.u_min = -15.0

        # 离散 CCBF 参数（严格契合论文 Section 4.C 设定）
        self.alpha_d = 0.8   # 离散 CCBF 收缩调节参数 (alpha_d)
        self.gamma_d = 0.2   # 离散 CCBF 耗散调节参数 (gamma_d)

        # 保守防撞缓冲区，用以在致动器饱和进入 Fallback 阶段时提供物理缓冲，确保系统"自愈"过程中绝对安全
        self.delta_buffer = 0.15

        # CBF 触发/回退统计计数器（审稿修订：P3 实证支持）
        self.cbf_active_count = 0    # CBF 投影触发次数（u_box 违反约束需修正）
        self.cbf_fallback_count = 0  # CBF 回退触发次数（二分无解，致动器饱和）
        self.cbf_call_count = 0      # CBF 总调用次数

        # 定义由交错球体障碍物形成的非凸避障通道环境
        # 支持自定义障碍物配置（审稿修订R5: 多障碍物环境实验）
        if obstacles is not None:
            self.obstacles = obstacles
        else:
            # 默认配置: 三个交错球体障碍物
            self.obstacles = [
                {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
                {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
                {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
            ]

    def step_discrete(self, x, u, use_mismatch=False, process_noise=0.0):
        """
        利用 4 阶龙格-库塔法 (RK4) 执行离散动力学状态步进。
        x = [px, py, pz, vx, vy, vz]^T
        """
        u = torch.clamp(u, self.u_min, self.u_max)

        m_val = self.m * 1.5 if use_mismatch else self.m
        b_val = self.b_drag * 2.0 if use_mismatch else self.b_drag

        def f(state, ctrl):
            p = state[0:3]
            v = state[3:6]
            # u 为重力预补偿后的外环虚拟力；重力项在差分平坦映射中并入实际总推力，
            # 因此 6D 平移动力学中不再重复扣除 g e3。
            v_dot = ctrl / m_val - (b_val / m_val) * v
            return torch.cat([v, v_dot])

        k1 = f(x, u)
        k2 = f(x + 0.5 * self.dt * k1, u)
        k3 = f(x + 0.5 * self.dt * k2, u)
        k4 = f(x + self.dt * k3, u)

        x_next = x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 模拟物理环境中的持续高斯过程噪声
        if process_noise > 0.0:
            noise = torch.randn_like(x_next) * process_noise
            noise = torch.clamp(noise, -3 * process_noise, 3 * process_noise)
            x_next = x_next + noise

        return x_next

    def reset_cbf_stats(self):
        """重置 CBF 统计计数器"""
        self.cbf_active_count = 0
        self.cbf_fallback_count = 0
        self.cbf_call_count = 0

    def get_cbf_stats(self):
        """返回当前 CBF 统计摘要"""
        return {
            "cbf_calls": self.cbf_call_count,
            "cbf_active": self.cbf_active_count,
            "cbf_fallback": self.cbf_fallback_count,
            "cbf_active_rate": self.cbf_active_count / max(self.cbf_call_count, 1),
            "cbf_fallback_rate": self.cbf_fallback_count / max(self.cbf_call_count, 1),
        }

    def apply_cbf_projection(self, x, u_nominal, kappa=5.0):
        """
        采用 Log-Sum-Exponential 光滑离散时间复合 Control Barrier Function（DT-CCBF）。
        与手稿公式 (4.14) 对齐，基于相对度为2的位置动力学雅可比构造一阶线性安全代理约束。
        """
        p_val = x[0:3].detach().cpu().numpy()
        v_val = x[3:6].detach().cpu().numpy()
        p_k1 = p_val + self.dt * v_val

        # 1. 计算当前的 B_smooth(p_k, v_k) 用于耗散约束
        B_curr_list = []
        for obs in self.obstacles:
            r_safe = obs["r"] + self.delta_buffer
            h_k = np.dot(p_val - obs["p"], p_val - obs["p"]) - r_safe**2
            h_k1 = np.dot(p_k1 - obs["p"], p_k1 - obs["p"]) - r_safe**2
            B_j = h_k1 - (1.0 - self.alpha_d) * h_k
            B_curr_list.append(B_j)
        B_curr_arr = np.array(B_curr_list)
        max_neg_kB = np.max(-kappa * B_curr_arr)
        sum_exp = np.sum(np.exp(-kappa * B_curr_arr - max_neg_kB))
        B_smooth = -1.0 / kappa * (np.log(sum_exp) + max_neg_kB)

        # 2. 推导不含控制作用 u_k 的 drift 状态量：p_{k+2, drift}
        # 外环输入采用重力预补偿定义，因此 drift 项只包含阻尼，不包含 -g e3。
        v_k1_drift = v_val + self.dt * (-(self.b_drag / self.m) * v_val)
        p_k2_drift = p_k1 + self.dt * v_k1_drift

        # 3. 计算 drift 状态下的 B_j_drift 和对应的 B_smooth_drift
        B_drift_list = []
        for obs in self.obstacles:
            r_safe = obs["r"] + self.delta_buffer
            h_k1 = np.dot(p_k1 - obs["p"], p_k1 - obs["p"]) - r_safe**2
            h_k2_drift = np.dot(p_k2_drift - obs["p"], p_k2_drift - obs["p"]) - r_safe**2
            B_j_drift = h_k2_drift - (1.0 - self.alpha_d) * h_k1
            B_drift_list.append(B_j_drift)
        B_drift_arr = np.array(B_drift_list)
        max_neg_kB_drift = np.max(-kappa * B_drift_arr)
        sum_exp_drift = np.sum(np.exp(-kappa * B_drift_arr - max_neg_kB_drift))
        B_smooth_drift = -1.0 / kappa * (np.log(sum_exp_drift) + max_neg_kB_drift)
        weights_drift = np.exp(-kappa * B_drift_arr - max_neg_kB_drift) / sum_exp_drift

        # 4. 严格解析拼装 A_smooth * u_k <= b_smooth
        # 对应二阶相对度在未来两步位置预测中的控制导数，dp_du = dt**2 / m
        dp_du = (self.dt**2) / self.m
        A_smooth = np.zeros(3)
        for j, obs in enumerate(self.obstacles):
            w_j = weights_drift[j]
            A_smooth += w_j * (-2.0 * dp_du * (p_k2_drift - obs["p"]))

        b_smooth = B_smooth_drift - (1.0 - self.gamma_d) * B_smooth

        self.cbf_call_count += 1
        u_safe, was_active, was_fallback = self._project_control(
            u_nominal.detach().cpu().numpy(), A_smooth, b_smooth)
        if was_active:
            self.cbf_active_count += 1
        if was_fallback:
            self.cbf_fallback_count += 1
        return torch.tensor(u_safe, dtype=torch.float32)

    def _project_control(self, u_nominal, A, b):
        """解析法高精度二分求解 Lagrange 乘子以满足 active 避障约束

        返回 (u_safe, was_active, was_fallback):
          - was_active: True 表示 u_box 违反约束，CBF 投影被触发
          - was_fallback: True 表示二分法无解，进入致动器饱和紧急回退
        """
        u_box = np.clip(u_nominal, self.u_min, self.u_max)
        if np.dot(A, u_box) <= b:
            return u_box, False, False

        # 寻找 lambda 上界
        low = 0.0
        high = 50.0
        for _ in range(10):
            u_test = np.clip(u_nominal - high * A, self.u_min, self.u_max)
            if np.dot(A, u_test) <= b:
                break
            high *= 2.0

        best_u = u_box
        has_solution = False
        for _ in range(25):
            mid = (low + high) / 2.0
            u_test = np.clip(u_nominal - mid * A, self.u_min, self.u_max)
            val = np.dot(A, u_test)
            if val <= b:
                best_u = u_test
                high = mid
                has_solution = True
            else:
                low = mid

        # 当致动器饱和导致控制屏障函数 (CBF) 约束无解时，执行盒约束下的紧急最大减小 A^T u 安全回退。
        if not has_solution:
            if np.linalg.norm(A) < 1e-9:
                best_u = u_box
            else:
                best_u = np.where(A >= 0.0, self.u_min, self.u_max)
            return best_u, True, True

        return best_u, True, False
