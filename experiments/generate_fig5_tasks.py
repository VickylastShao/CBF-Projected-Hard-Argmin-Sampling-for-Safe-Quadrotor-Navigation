#!/usr/bin/env python3
"""
生成 fig5_tasks.pdf — narrow / two_gate / u_shape 三个 benchmark task 的
top-down obstacle-and-setpoint 可视化。每个 task 是一个 sub-panel：
  - 灰色圆 = 障碍 (XY 投影)
  - 红色 X = setpoint
  - 蓝色 ○ = 初始状态分布中心 (x0_mean)
  - 浅蓝色阴影 = 初始状态分布范围 (sample bounding box)
输出: experiments/results_v6/fig5_tasks.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from experiments.ptrm_advantage_quick import TASK_FACTORIES, sample_initial_states


def draw_task(ax, task_name: str, seed: int = 7777):
    task = TASK_FACTORIES[task_name](seed)
    inits = sample_initial_states(task, 40, seed)
    x_sp = task["x_sp"].numpy()
    obstacles = task["obstacles"]

    # XY projection
    for obs in obstacles:
        circ = Circle(
            (obs["p"][0], obs["p"][1]),
            obs["r"],
            color="dimgray", alpha=0.55, zorder=2,
        )
        ax.add_patch(circ)

    init_xy = np.array([x[:2].numpy() for x in inits])
    ax.scatter(init_xy[:, 0], init_xy[:, 1],
               s=14, color="steelblue", alpha=0.55, zorder=3,
               label="initial states ($N_{\\rm MC}=40$)")
    ax.scatter([x_sp[0]], [x_sp[1]], marker="x", color="red", s=120,
               linewidths=3, zorder=4, label="setpoint")
    init_mean = init_xy.mean(axis=0)
    ax.scatter([init_mean[0]], [init_mean[1]], marker="o",
               facecolors="none", edgecolors="navy", s=180, linewidth=2,
               zorder=4, label="init mean")

    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"\\texttt{{{task_name}}}", fontsize=11)


def main():
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif", size=10)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, name in zip(axes, ["narrow", "two_gate", "u_shape"]):
        draw_task(ax, name, seed=7777)

    # 共享 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, fontsize=9, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = ROOT / "experiments/results_v6/fig5_tasks.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
