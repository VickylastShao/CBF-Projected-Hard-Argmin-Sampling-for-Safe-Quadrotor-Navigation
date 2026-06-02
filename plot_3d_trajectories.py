# -*- coding: utf-8 -*-
"""生成3D轨迹可视化（使用已训练的模型）"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

from quadrotor_core import (
    QuadrotorDynamics,
    GoldenNMPCSolver,
    TRMNMPC,
    PTRMNMPCPredictor,
    generate_quadrotor_dataset,
    train_trm_jointly,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'results')

device = torch.device("cpu")
print(f"设备: {device}")

env = QuadrotorDynamics()
solver = GoldenNMPCSolver(env, horizon=10)
model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)

# 重新训练（与主脚本一致的配置）
dataset = generate_quadrotor_dataset(env, solver, size=500)
model, history = train_trm_jointly(model, dataset, env, epochs=100, patience=15, verbose=False)

x_sp = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32)
sim_steps = 100
num_display = 3

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制障碍物球体
for obs in env.obstacles:
    u_sphere = np.linspace(0, 2 * np.pi, 30)
    v_sphere = np.linspace(0, np.pi, 20)
    x_s = obs["p"][0] + obs["r"] * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_s = obs["p"][1] + obs["r"] * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_s = obs["p"][2] + obs["r"] * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
    ax.plot_surface(x_s, y_s, z_s, alpha=0.3, color='red')

# 运行轨迹
for method_name, predictor, cbf_flag, color in [
    ('NMPC', None, True, 'black'),
    ('Det TRM (K=1)', PTRMNMPCPredictor(model, env, K=1, D=16, sigma=0.0), False, 'red'),
    ('PTRM-NMPC (K=50)', PTRMNMPCPredictor(model, env, K=50, D=16, sigma=0.25), True, 'green'),
]:
    for i in range(num_display):
        if predictor is not None:
            predictor.reset()
        init_px = 0.0 + np.random.normal(0, 0.02)
        init_py = 0.0 + np.random.normal(0, 0.02)
        init_pz = 0.0 + np.random.normal(0, 0.02)
        init_vx = 0.5 + np.random.normal(0, 0.01)
        init_vy = 0.5 + np.random.normal(0, 0.01)
        init_vz = 0.5 + np.random.normal(0, 0.01)
        x_init = torch.tensor([init_px, init_py, init_pz, init_vx, init_vy, init_vz], dtype=torch.float32)

        x_curr = x_init.clone()
        trajectory = [x_curr[0:3].detach().cpu().numpy()]

        for step in range(sim_steps):
            if predictor is None:
                u_nominal = solver.solve(x_curr, x_sp)[0:3]
                u = env.apply_cbf_projection(x_curr, u_nominal)
            else:
                u, _ = predictor.predict_action(x_curr, x_sp, enable_cbf=cbf_flag)
            x_curr = env.step_discrete(x_curr, u, process_noise=0.008)
            trajectory.append(x_curr[0:3].detach().cpu().numpy())

        trajectory = np.array(trajectory)
        style = '--' if method_name == 'NMPC' else (':' if 'Det' in method_name else '-')
        alpha = 0.4 if i > 0 else 1.0
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                style, color=color, alpha=alpha, linewidth=1.5 if i == 0 else 0.8,
                label=method_name if i == 0 else None)
        print(f"  {method_name} 轨迹 {i+1} 完成")

ax.scatter([0], [0], [0], c='blue', s=100, marker='o', label='Start', zorder=5)
ax.scatter([3], [3], [3], c='gold', s=100, marker='*', label='Target', zorder=5)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Trajectory Comparison: NMPC vs Deterministic TRM vs PTRM-NMPC')
ax.legend(loc='upper left')

fig_path = os.path.join(RESULTS_DIR, 'ptrm_nmpc_3d_trajectories.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
fig_path_pdf = os.path.join(RESULTS_DIR, 'ptrm_nmpc_3d_trajectories.pdf')
plt.savefig(fig_path_pdf, bbox_inches='tight')
print(f"3D 轨迹图已保存至: {fig_path}")
plt.close(fig)
