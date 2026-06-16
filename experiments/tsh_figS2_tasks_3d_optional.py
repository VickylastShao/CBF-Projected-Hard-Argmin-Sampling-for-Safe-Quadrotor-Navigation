#!/usr/bin/env python3
"""
Optional 3D benchmark overview for supplementary material.
Uses the same task source:
    from tsh_ptrm_advantage_quick import TASK_FACTORIES, sample_initial_states
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tsh_ptrm_advantage_quick import TASK_FACTORIES, sample_initial_states
from tsh_plot_style_optimized import setup_rcparams, PALETTE

OUT = HERE / 'experiments/results_v6/figS2_tasks_3d.pdf'


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def plot_sphere(ax, center, radius, color='#9A9A9A', alpha=0.32, edge='#444444'):
    u = np.linspace(0, 2*np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color=edge, linewidth=0.2, alpha=0.25)


def draw_task_3d(ax, task_name: str, seed: int = 7777):
    task = TASK_FACTORIES[task_name](seed)
    inits = sample_initial_states(task, 40, seed)
    x_sp = _to_np(task['x_sp'])
    obstacles = task['obstacles']

    for obs in obstacles:
        p = _to_np(obs['p'])
        r = float(obs['r'])
        center = p[:3] if p.shape[0] >= 3 else np.array([p[0], p[1], 1.0])
        plot_sphere(ax, center, r)

    init_xyz = np.array([_to_np(x)[:3] for x in inits])
    ax.scatter(init_xyz[:, 0], init_xyz[:, 1], init_xyz[:, 2], s=10, color='#7FAED6', alpha=0.45)
    ax.scatter([init_xyz[:,0].mean()], [init_xyz[:,1].mean()], [init_xyz[:,2].mean()],
               s=46, facecolors='white', edgecolors=PALETTE['tsh'], linewidths=1.3)
    ax.scatter([x_sp[0]], [x_sp[1]], [x_sp[2]], s=58, marker='x', color=PALETTE['pd'], linewidths=1.8)

    ax.set_title(task_name.upper(), fontsize=9.4, fontweight='bold', pad=6)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.5, 3.5)
    zmin = min(init_xyz[:, 2].min(), x_sp[2]) - 0.5
    zmax = max(init_xyz[:, 2].max(), x_sp[2]) + 0.8
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=20, azim=-58)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.grid(True, alpha=0.25)


def main():
    setup_rcparams()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.0, 3.45), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
    for ax, name in zip(axes, ['narrow', 'two_gate', 'u_shape']):
        draw_task_3d(ax, name, seed=7777)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#7FAED6', markeredgecolor='none',
               markersize=5.5, alpha=0.7, label='initial states'),
        Line2D([0], [0], marker='o', color=PALETTE['tsh'], markerfacecolor='white', markersize=5.5,
               lw=1.2, label='init mean'),
        Line2D([0], [0], marker='x', color=PALETTE['pd'], markersize=7.0, lw=0, label='setpoint'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(OUT, bbox_inches='tight', dpi=300)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
