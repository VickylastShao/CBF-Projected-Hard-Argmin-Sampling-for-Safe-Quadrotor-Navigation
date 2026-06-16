# -*- coding: utf-8 -*-
"""
MCIS (Maximal Controlled Invariant Set) 内近似计算

通过后向可达性采样，验证论文 Assumption 2：
安全集 C 是否为 MCIS 的子集。

方法：
1. 在障碍物周围区域均匀采样大量初始状态
2. 使用 PD+CBF 控制器执行闭环仿真
3. 检查哪些初始状态能安全到达 x_sp
4. 报告成功区域的几何范围（MCIS 内近似）
"""

import sys
import os
import time
import json
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quadrotor_core import QuadrotorDynamics, GoldenNMPCSolver

SEED = 2026
N_SAMPLES = 2000
N_STEPS = 500
DT = 0.02
X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
SUCCESS_TERR = 0.3  # 终端误差阈值
SUCCESS_IAE_MAX = 5.0  # IAE 阈值


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pd_controller(x, x_sp, Kp=5.0, Kd=3.0):
    """PD 控制器作为名义控制器"""
    pos_err = x_sp[0:3] - x[0:3]
    vel_err = x_sp[3:6] - x[3:6]
    u = Kp * pos_err + Kd * vel_err
    return torch.clamp(u, -15.0, 15.0)


def check_reachable(env, x_init, x_sp, n_steps=N_STEPS):
    """检查从 x_init 出发，PD+CBF 能否安全到达 x_sp"""
    x = x_init.clone()
    collision = False
    iae = 0.0

    for step in range(n_steps):
        u_nominal = pd_controller(x, x_sp)
        u_safe = env.apply_cbf_projection(x, u_nominal)
        x = env.step_discrete(x, u_safe)
        p_np = x[0:3].detach().numpy()

        for obs in env.obstacles:
            d = np.linalg.norm(p_np - obs['p']) - obs['r']
            if d < 0:
                collision = True
                return False, float('inf'), True

        iae += torch.norm(x[0:3] - x_sp[0:3]).item()

    iae = iae / n_steps
    terr = torch.norm(x[0:3] - x_sp[0:3]).item()
    success = (not collision) and (terr < SUCCESS_TERR) and (iae < SUCCESS_IAE_MAX)
    return success, terr, collision


def main():
    set_seed(SEED)
    t_start = time.time()

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    os.makedirs(save_dir, exist_ok=True)

    env = QuadrotorDynamics()

    # 定义采样区域：围绕障碍物通道的扩展区域
    # 障碍物中心: [1,1,1], [2,1.5,2], [1.5,2.2,1.5]
    # 采样范围: 位置 [-1, 3.5], 速度 [-2, 2]
    pos_ranges = [(-1.0, 3.5), (-1.0, 3.5), (-1.0, 3.5)]
    vel_ranges = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]

    print("=" * 80)
    print("MCIS 内近似计算 (后向可达性采样)")
    print("=" * 80)
    print(f"采样数量: {N_SAMPLES}")
    print(f"仿真步数: {N_STEPS} (={N_STEPS * DT:.1f}s)")
    print(f"目标点: {X_SP.tolist()}")
    print(f"障碍物: {len(env.obstacles)} 个球体")
    print()

    # 生成随机初始状态
    samples = []
    for i in range(N_SAMPLES):
        pos = np.array([np.random.uniform(lo, hi) for lo, hi in pos_ranges])
        vel = np.array([np.random.uniform(lo, hi) for lo, hi in vel_ranges])
        x_init = torch.tensor(np.concatenate([pos, vel]), dtype=torch.float32)
        samples.append(x_init)

    # 检查每个样本的可达性
    results = []
    n_success = 0
    n_collision = 0
    n_timeout = 0

    for i, x_init in enumerate(samples):
        success, terr, collision = check_reachable(env, x_init, X_SP)
        results.append({
            'x_init': x_init.tolist(),
            'success': success,
            'terminal_error': terr,
            'collision': collision,
        })
        if success:
            n_success += 1
        elif collision:
            n_collision += 1
        else:
            n_timeout += 1

        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{N_SAMPLES}, "
                  f"成功={n_success}, 碰撞={n_collision}, 超时={n_timeout}")

    # 统计结果
    success_rate = n_success / N_SAMPLES * 100
    collision_rate = n_collision / N_SAMPLES * 100
    timeout_rate = n_timeout / N_SAMPLES * 100

    print(f"\n结果:")
    print(f"  成功到达 x_sp: {n_success}/{N_SAMPLES} ({success_rate:.1f}%)")
    print(f"  碰撞失败: {n_collision}/{N_SAMPLES} ({collision_rate:.1f}%)")
    print(f"  超时/未收敛: {n_timeout}/{N_SAMPLES} ({timeout_rate:.1f}%)")

    # 分析成功样本的状态范围（MCIS 内近似几何）
    success_samples = [r for r in results if r['success']]
    if success_samples:
        success_states = np.array([r['x_init'] for r in success_samples])
        print(f"\nMCIS 内近似几何范围 (成功样本):")
        labels = ['px', 'py', 'pz', 'vx', 'vy', 'vz']
        for i, label in enumerate(labels):
            lo = success_states[:, i].min()
            hi = success_states[:, i].max()
            print(f"  {label}: [{lo:.3f}, {hi:.3f}]")

    # 分析失败样本（碰撞）
    collision_samples = [r for r in results if r['collision']]
    if collision_samples:
        collision_states = np.array([r['x_init'] for r in collision_samples])
        print(f"\n碰撞样本状态范围:")
        for i, label in enumerate(labels):
            lo = collision_states[:, i].min()
            hi = collision_states[:, i].max()
            print(f"  {label}: [{lo:.3f}, {hi:.3f}]")

    # 验证 x_sp 本身是否在 MCIS 内（应该成功）
    print(f"\n验证 x_sp 自身可达性:")
    x_sp_test = X_SP.clone()
    success_sp, terr_sp, coll_sp = check_reachable(env, x_sp_test, X_SP)
    print(f"  x_sp -> x_sp: 成功={success_sp}, TErr={terr_sp:.6f}m, 碰撞={coll_sp}")

    # 验证附近点的可达性
    print(f"\n验证 x_sp 附近点:")
    for delta in [0.1, 0.3, 0.5, 1.0]:
        n_near_success = 0
        n_near_total = 50
        for _ in range(n_near_total):
            x_near = X_SP.clone()
            x_near[0:3] += torch.randn(3) * delta
            x_near[3:6] += torch.randn(3) * delta * 0.5
            s, _, _ = check_reachable(env, x_near, X_SP)
            if s:
                n_near_success += 1
        print(f"  δ={delta:.1f}: {n_near_success}/{n_near_total} 成功 "
              f"({n_near_success/n_near_total*100:.0f}%)")

    t_total = time.time() - t_start
    print(f"\n总计算时间: {t_total:.1f}s ({t_total/60:.1f}min)")

    # 保存结果
    save_data = {
        'n_samples': N_SAMPLES,
        'n_steps': N_STEPS,
        'success_count': n_success,
        'collision_count': n_collision,
        'timeout_count': n_timeout,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'mcis_position_range': {},
        'mcis_velocity_range': {},
    }

    if success_samples:
        success_states = np.array([r['x_init'] for r in success_samples])
        for i, label in enumerate(labels[:3]):
            save_data['mcis_position_range'][label] = {
                'min': float(success_states[:, i].min()),
                'max': float(success_states[:, i].max()),
            }
        for i, label in enumerate(labels[3:]):
            save_data['mcis_velocity_range'][label] = {
                'min': float(success_states[:, i+3].min()),
                'max': float(success_states[:, i+3].max()),
            }

    results_path = os.path.join(save_dir, 'mcis_computation_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n结果已保存至 {results_path}")


if __name__ == '__main__':
    main()
