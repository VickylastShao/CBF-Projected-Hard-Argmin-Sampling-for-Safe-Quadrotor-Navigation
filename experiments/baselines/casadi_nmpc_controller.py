# -*- coding: utf-8 -*-
"""
CasADi + IPOPT 直接 NMPC 求解器基线

基于 CasADi 符号计算框架和 IPOPT 内点法实现的直接 NMPC 求解器，
用于与 PTRM-NMPC 的采样式 test-time scaling 方法进行对比。

采用 SQP/RTI 风格的单次 NLP 求解（warm-start），每次 MPC 步仅求解一个 NLP。

参考: CasADi — A software framework for nonlinear optimization and optimal control
      https://web.casadi.org/
"""

import time
import numpy as np
import torch
import casadi as ca


class CasADiNMPCController:
    """CasADi + IPOPT 直接 NMPC 求解器"""

    def __init__(self, env, horizon=10, obs_weight=2000.0, obs_buffer=0.3):
        self.env = env
        self.H = horizon
        self.dt = env.dt
        self.m = env.m
        self.b_drag = env.b_drag
        self.u_min_val = env.u_min
        self.u_max_val = env.u_max
        self.obs_weight = obs_weight  # 软障碍物外部惩罚权重 (与 MPPI/CEM 一致)
        self.obs_buffer = obs_buffer  # 障碍物安全裕度 (m)

        # 代价矩阵
        self.Q_diag = np.array([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        self.R_val = 0.02

        # 障碍物快照 (在 _build_nlp 之前固定，便于 NLP 编译)
        self._obstacles = [
            (float(o["p"][0]), float(o["p"][1]), float(o["p"][2]),
             float(o["r"])) for o in env.obstacles
        ]

        # 终端代价 P_f (via DARE)
        self.P_f = self._compute_terminal_cost()

        # 构建 NLP 问题（符号化，仅构建一次）
        self._build_nlp()

        # IPOPT 选项
        self.opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-8,
            'ipopt.warm_start_mult_bound_push': 1e-8,
            'print_time': 0,
            'verbose': False,
        }

        # 创建求解器
        self.solver = ca.nlpsol('nmpc_solver', 'ipopt', self.nlp, self.opts)

        # Warm-start 上一次解
        self.last_u_opt = None

    def _compute_terminal_cost(self):
        """通过 DARE 计算终端代价矩阵 P_f"""
        dt = self.dt
        b_m = self.b_drag / self.m
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 3:6] = np.eye(3) - dt * b_m * np.eye(3)
        A[0:3, 0:3] = np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, :] = dt / self.m * np.eye(3)
        Q = np.diag(self.Q_diag)
        R = self.R_val * np.eye(3)

        # 迭代求解 DARE
        P = Q.copy()
        for _ in range(200):
            BP = B.T @ P
            BPB = BP @ B
            S = R + BPB
            K = np.linalg.solve(S, BP @ A)
            P_new = A.T @ P @ A - A.T @ P @ B @ K + Q
            if np.linalg.norm(P_new - P) < 1e-10:
                return P_new
            P = P_new
        return P

    def _build_nlp(self):
        """构建 CasADi 符号 NLP 问题"""
        H = self.H
        dt = self.dt
        m = self.m
        b_drag = self.b_drag

        # 决策变量: u_0, ..., u_{H-1}，每个 3D
        u_sym = ca.SX.sym('u', 3 * H)
        # 参数: x_init (6D) + x_sp (6D)
        p_sym = ca.SX.sym('p', 12)
        x_init = p_sym[0:6]
        x_sp = p_sym[6:12]

        # 代价函数
        cost = 0.0
        x_curr = x_init

        for i in range(H):
            u = u_sym[i * 3:(i + 1) * 3]

            # RK4 积分
            def f(state, ctrl):
                pos = state[0:3]
                vel = state[3:6]
                v_dot = ctrl / m - (b_drag / m) * vel
                return ca.vertcat(vel, v_dot)

            k1 = f(x_curr, u)
            k2 = f(x_curr + 0.5 * dt * k1, u)
            k3 = f(x_curr + 0.5 * dt * k2, u)
            k4 = f(x_curr + dt * k3, u)
            x_curr = x_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 阶段代价
            error = x_curr - x_sp
            Q_mat = ca.diag(ca.DM(self.Q_diag))
            cost += ca.mtimes(error.T, ca.mtimes(Q_mat, error))
            cost += self.R_val * ca.mtimes(u.T, u)

            # 软障碍物外部惩罚 (与 MPPI/CEM 完全一致：W * max(buffer - d, 0)^2)
            pos_next = x_curr[0:3]
            for (ox, oy, oz, rr) in self._obstacles:
                d = ca.sqrt((pos_next[0]-ox)**2 + (pos_next[1]-oy)**2 +
                            (pos_next[2]-oz)**2 + 1e-6) - rr
                viol = ca.fmax(self.obs_buffer - d, 0)
                cost += self.obs_weight * viol * viol

        # 终端代价
        terminal_error = x_curr - x_sp
        cost += ca.mtimes(terminal_error.T, ca.mtimes(ca.DM(self.P_f), terminal_error))

        # 约束: 控制量边界 (通过 NLP 的 lbx/ubx 实现)
        # 无额外非线性约束
        g = ca.SX([])  # 空约束

        self.nlp = {
            'x': u_sym,
            'f': cost,
            'g': g,
            'p': p_sym,
        }

    def solve(self, x_init_np, x_sp_np):
        """
        求解 NMPC 问题

        Args:
            x_init_np: numpy array (6,) 初始状态
            x_sp_np: numpy array (6,) 目标设定点

        Returns:
            u_opt: numpy array (3*H,) 最优控制序列
        """
        H = self.H
        p_val = np.concatenate([x_init_np, x_sp_np])

        # 边界
        lbx = np.full(3 * H, self.u_min_val)
        ubx = np.full(3 * H, self.u_max_val)

        # Warm-start
        x0 = self.last_u_opt if self.last_u_opt is not None else np.zeros(3 * H)

        try:
            sol = self.solver(
                x0=x0,
                lbx=lbx,
                ubx=ubx,
                p=p_val,
            )
            u_opt = np.array(sol['x']).flatten()
            self.last_u_opt = u_opt.copy()
            return u_opt
        except RuntimeError:
            # IPOPT 失败时回退到 PD 控制
            self.last_u_opt = None
            return np.zeros(3 * H)

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        """
        与其他基线一致的接口

        Args:
            x_init: torch tensor (6,)
            x_sp: torch tensor (6,)
            enable_cbf: 是否应用 CBF 安全投影

        Returns:
            u_safe: torch tensor (3,)
        """
        x_init_np = x_init.detach().cpu().numpy()
        x_sp_np = x_sp.detach().cpu().numpy()

        u_opt = self.solve(x_init_np, x_sp_np)
        u_first = u_opt[0:3]

        u_nominal = torch.tensor(u_first, dtype=torch.float32)

        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x_init, u_nominal)
        else:
            u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)

        return u_safe

    def get_runtime_ms(self, x_init, x_sp, enable_cbf=True, n_runs=10):
        """测量单步决策延迟 (ms)"""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict_action(x_init, x_sp, enable_cbf)
            times.append((time.perf_counter() - t0) * 1000)
        return np.median(times)

    def reset(self):
        """重置 warm-start 状态"""
        self.last_u_opt = None
