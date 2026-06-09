# -*- coding: utf-8 -*-
"""
专家数据集生成及批量轨迹代价评估

生成离线 L-BFGS 专家 NMPC 动作数据集，
并提供批量解码轨迹代价计算用于 Q-head 训练标签动态生成。
"""

import time
import numpy as np
import torch


def generate_quadrotor_dataset(env, solver, size=150, x_sp=None, pos_range=None):
    """
    离线产生 6D 四轴飞行器专家动作数据集。
    Q 头标签将在训练阶段基于当前解码候选轨迹动态计算。

    Args:
        x_sp: 目标设定点 (默认 [3,3,3,0,0,0])
        pos_range: 初始位置范围 [[p_min, p_max], ...] 每个轴 (默认 [-0.5, 2.0])
    """
    if x_sp is None:
        x_sp = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    if pos_range is None:
        pos_range = [(-0.5, 2.0), (-0.5, 2.0), (-0.5, 2.0)]

    print(f"正在离线产生 6D 四轴飞行器专家动作数据集 (size={size})；Q 头标签将在训练阶段基于当前解码候选轨迹动态计算...")
    print(f"  目标设定点: {x_sp.numpy()}")
    start_time = time.time()
    dataset = []

    for i in range(size):
        px = np.random.uniform(*pos_range[0])
        py = np.random.uniform(*pos_range[1])
        pz = np.random.uniform(*pos_range[2])
        vx = np.random.uniform(-0.4, 0.4)
        vy = np.random.uniform(-0.4, 0.4)
        vz = np.random.uniform(-0.4, 0.4)

        x_init = torch.tensor([px, py, pz, vx, vy, vz], dtype=torch.float32)

        u_opt = solver.solve(x_init, x_sp)

        # 策略监督仍使用专家最优序列；Q 头训练在 train_trm_jointly() 中基于当前解码候选轨迹动态生成标签。
        X_feature = torch.cat([x_init, x_sp])
        dataset.append((X_feature, u_opt))

        if (i+1) % 50 == 0:
            print(f"数据生成进度: {i+1}/{size} | 累计耗时: {time.time() - start_time:.2f}s")

    return dataset


def generate_cl_trm_dataset(env, x_sp=None, size=500, steps_per_traj=10,
                             Kp=4.0, Kd=3.0, pos_range=None):
    """
    生成 PD+CBF 闭环训练数据集（CL-TRM 训练用）

    对每个样本：随机初始状态 → PD 控制器闭环执行 10 步 →
    记录 3D 动作序列展开为 30D 标签。格式与 generate_quadrotor_dataset() 一致。

    CBF 投影作用于实际执行，训练标签使用 CBF 投影后的安全动作
    （与推理时 predict_action 返回 u_safe 的逻辑对齐）。

    Args:
        env: QuadrotorDynamics 环境（可含障碍物）
        x_sp: 目标设定点 (6,)，默认 [2, 3, 2, 0, 0, 0]
        size: 生成样本数（轨迹条数）
        steps_per_traj: 每条轨迹收集的步数（= MPC horizon 步数 = 10）
        Kp, Kd: PD 控制增益
        pos_range: 初始位置范围，默认 [(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)]

    Returns:
        dataset: [(X_feature(12), u_opt(30)), ...]
    """
    if x_sp is None:
        x_sp = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    if pos_range is None:
        pos_range = [(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)]

    print(f"正在生成 PD+CBF 闭环训练数据集 (size={size}, steps={steps_per_traj})...")
    start_time = time.time()
    dataset = []

    for i in range(size):
        px = np.random.uniform(*pos_range[0])
        py = np.random.uniform(*pos_range[1])
        pz = np.random.uniform(*pos_range[2])
        vx = np.random.uniform(-0.4, 0.4)
        vy = np.random.uniform(-0.4, 0.4)
        vz = np.random.uniform(-0.4, 0.4)

        x = torch.tensor([px, py, pz, vx, vy, vz], dtype=torch.float32)
        x_init = x.clone()

        u_seq = torch.zeros(steps_per_traj * 3)
        for t in range(steps_per_traj):
            e_p = x_sp[0:3] - x[0:3]
            e_v = x_sp[3:6] - x[3:6]
            u_pd = env.m * (Kp * e_p + Kd * e_v)
            u_safe = env.apply_cbf_projection(x, u_pd)
            u_seq[t*3:(t+1)*3] = u_safe
            x = env.step_discrete(x, u_safe)

        X_feature = torch.cat([x_init, x_sp])
        dataset.append((X_feature, u_seq))

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{size} | 耗时: {time.time() - start_time:.1f}s")

    print(f"  数据集生成完成: {size} 样本, 耗时 {time.time() - start_time:.1f}s")
    return dataset


def evaluate_batch_decoded_trajectory_cost(env, X_batch, u_sequences):
    """
    批量计算当前解码控制序列的名义轨迹代价（含终端代价 P_f），
    用于 Q 头在线候选序列价值对齐。
    与 step_discrete 保持同一重力预补偿抽象；采用批量一阶离散化以避免训练期 Python 循环开销过高。
    """
    from .nmpc_solver import _solve_dare

    x_curr = X_batch[:, 0:6]
    x_sp = X_batch[:, 6:12]
    device = X_batch.device
    dtype = X_batch.dtype
    q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)
    cost = torch.zeros(X_batch.shape[0], device=device, dtype=dtype)
    steps = min(u_sequences.shape[1] // 3, 10)

    for i in range(steps):
        u = torch.clamp(u_sequences[:, i*3:(i+1)*3], env.u_min, env.u_max)
        p = x_curr[:, 0:3]
        v = x_curr[:, 3:6]
        v_dot = u / env.m - (env.b_drag / env.m) * v
        p_next = p + env.dt * v
        v_next = v + env.dt * v_dot
        x_curr = torch.cat([p_next, v_next], dim=1)

        error = x_curr - x_sp
        cost = cost + torch.sum(q_diag * error * error, dim=1) + 0.02 * torch.sum(u * u, dim=1)

    # 终端代价: (x_final - x_sp)' P_f (x_final - x_sp)
    dt = env.dt
    b_m = env.b_drag / env.m
    A = torch.zeros(6, 6, device=device, dtype=dtype)
    A[0:3, 3:6] = torch.eye(3, device=device, dtype=dtype)
    A[3:6, 3:6] = torch.eye(3, device=device, dtype=dtype) - dt * b_m * torch.eye(3, device=device, dtype=dtype)
    A[0:3, 0:3] = torch.eye(3, device=device, dtype=dtype)
    B = torch.zeros(6, 3, device=device, dtype=dtype)
    B[3:6, :] = dt / env.m * torch.eye(3, device=device, dtype=dtype)
    P_f = _solve_dare(A, B, torch.diag(q_diag).to(device=device, dtype=dtype),
                       0.02 * torch.eye(3, device=device, dtype=dtype))
    terminal_error = x_curr - x_sp
    cost = cost + torch.sum(terminal_error @ P_f * terminal_error, dim=1)
    return cost
