#!/usr/bin/env python3
"""
Generate IEEE-quality 3D trajectory comparison figure.
Uses the trained CL-TRM model from v6 experiments.
Compares: NMPC expert, PD+CBF K=1, PTRM-NMPC K=50, MPPI K=50.
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.linewidth': 0.6,
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures_ieee')
RESULTS_V6 = os.path.join(SCRIPT_DIR, 'results_v6')

from quadrotor_core import (
    QuadrotorDynamics,
    GoldenNMPCSolver,
    TRMNMPC,
    PTRMNMPCPredictor,
)
from experiments.baselines.mppi_controller import MPPIController

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")

# Initialize environment and expert
env = QuadrotorDynamics()
solver = GoldenNMPCSolver(env, horizon=10)

# Load trained CL-TRM model
model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
model_path = os.path.join(RESULTS_V6, 'cl_trm_model.pt')
if os.path.exists(model_path):
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from {model_path}")
else:
    print(f"WARNING: {model_path} not found, using untrained model")

model.eval()

# Setpoint
x_sp = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32)
sim_steps = 100

# Run trajectory simulation for each method
def run_trajectory(predictor_fn, x_init, steps=sim_steps, process_noise=0.0):
    x_curr = x_init.clone()
    traj = [x_curr[0:3].detach().cpu().numpy()]
    for step in range(steps):
        u = predictor_fn(x_curr)
        x_curr = env.step_discrete(x_curr, u, use_mismatch=False, process_noise=process_noise)
        traj.append(x_curr[0:3].detach().cpu().numpy())
    return np.array(traj)


# Initialize predictors
ptrm = PTRMNMPCPredictor(
    model, env, K=50, D=16, sigma=0.25,
    candidate_mode='trm_pd', ranking_mode='rollout_all',
    alpha_blend=0.3, pd_sigma=2.0,
)
mppi = MPPIController(env, K=50, sigma=2.0, lam=0.1)

def pd_cbf_fn(x):
    u_nom = ptrm._compute_tracking_correction(x, x_sp)
    return env.apply_cbf_projection(x, u_nom)

def ptrm_fn(x):
    ptrm.reset()
    u, _ = ptrm.predict_action(x, x_sp, enable_cbf=True)
    return u

def mppi_fn(x):
    return mppi.predict_action(x, x_sp, enable_cbf=True)

def nmpc_fn(x):
    u_nom = solver.solve(x, x_sp)[0:3]
    return env.apply_cbf_projection(x, u_nom)


# Generate initial conditions (slight perturbations)
np.random.seed(SEED)
x_inits = []
for _ in range(3):
    x0 = torch.tensor([
        0.0 + np.random.normal(0, 0.02),
        0.0 + np.random.normal(0, 0.02),
        0.0 + np.random.normal(0, 0.02),
        0.5 + np.random.normal(0, 0.01),
        0.5 + np.random.normal(0, 0.01),
        0.5 + np.random.normal(0, 0.01),
    ], dtype=torch.float32)
    x_inits.append(x0)

methods = [
    ('NMPC', nmpc_fn, 'black', '--'),
    ('PD+CBF', pd_cbf_fn, '#2CA02C', ':'),
    ('PTRM $K\\!=\\!50$', ptrm_fn, '#1F77B4', '-'),
    ('MPPI $K\\!=\\!50$', mppi_fn, '#FF7F0E', '-.'),
]

# Create figure
fig = plt.figure(figsize=(7.16, 5.0))
ax = fig.add_subplot(111, projection='3d')

# Draw obstacles
for obs in env.obstacles:
    u_sphere = np.linspace(0, 2 * np.pi, 20)
    v_sphere = np.linspace(0, np.pi, 15)
    x_s = obs["p"][0] + obs["r"] * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_s = obs["p"][1] + obs["r"] * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_s = obs["p"][2] + obs["r"] * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
    ax.plot_surface(x_s, y_s, z_s, alpha=0.15, color='red', linewidth=0)

# Run and plot trajectories
for method_name, method_fn, color, ls in methods:
    for i, x_init in enumerate(x_inits):
        try:
            traj = run_trajectory(method_fn, x_init)
            alpha = 1.0 if i == 0 else 0.4
            lw = 1.5 if i == 0 else 0.8
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    ls, color=color, alpha=alpha, linewidth=lw,
                    label=method_name if i == 0 else None)
            print(f"  {method_name} trajectory {i+1} done")
        except Exception as e:
            print(f"  {method_name} trajectory {i+1} FAILED: {e}")

# Start and target markers
ax.scatter([0], [0], [0], c='blue', s=80, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=0.5)
ax.scatter([3], [3], [3], c='gold', s=80, marker='*', label='Target', zorder=5, edgecolors='black', linewidth=0.5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG_DIR, 'fig5_trajectories_3d.pdf'), bbox_inches='tight', pad_inches=0.03)
fig.savefig(os.path.join(FIG_DIR, 'fig5_trajectories_3d.png'), bbox_inches='tight', pad_inches=0.03)
plt.close(fig)
print(f"\nSaved: fig5_trajectories_3d.pdf/.png")
