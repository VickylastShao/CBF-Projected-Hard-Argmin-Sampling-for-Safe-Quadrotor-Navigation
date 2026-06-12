#!/usr/bin/env python3
"""
Generate figS1_tasks.pdf — benchmark geometry figure for supplementary material.

This polished version keeps the original data-reading方式:
    from experiments.tsh_ptrm_advantage_quick import TASK_FACTORIES, sample_initial_states

Output path is kept the same as the original script:
    experiments/results_v6/figS1_tasks.pdf
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
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from experiments.tsh_ptrm_advantage_quick import TASK_FACTORIES, sample_initial_states

try:
    from plot_style_tsh import setup_rcparams, soften_axes, panel_title, PALETTE, ellipse_from_cov
except Exception:
    def setup_rcparams():
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif", size=9)
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    def soften_axes(ax, grid="both"):
        if grid:
            ax.grid(True, alpha=0.35)
    def panel_title(ax, title):
        ax.set_title(title, fontsize=10, fontweight="bold")
    PALETTE = {
        "tsh": "#2A6FBB", "pd": "#C23B22",
        "obstacle_face": "#7A7A7A", "obstacle_edge": "#2E2E2E",
    }
    def ellipse_from_cov(mean, cov, n_std=2.0, num=200):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.linspace(0, 2*np.pi, num)
        circle = np.stack([np.cos(theta), np.sin(theta)])
        ell = vecs @ np.diag(np.sqrt(vals) * n_std) @ circle
        return ell.T + mean


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def draw_task(ax, task_name: str, seed: int = 7777):
    task = TASK_FACTORIES[task_name](seed)
    inits = sample_initial_states(task, 40, seed)
    x_sp = _to_np(task["x_sp"])
    obstacles = task["obstacles"]

    for obs in obstacles:
        p = _to_np(obs["p"])
        r = float(obs["r"])
        ax.add_patch(Circle((p[0], p[1]), r,
                            fc=PALETTE["obstacle_face"],
                            ec=PALETTE["obstacle_edge"],
                            lw=0.9, alpha=0.95, zorder=2))

    init_xy = np.array([_to_np(x)[:2] for x in inits])
    init_mean = init_xy.mean(axis=0)
    init_cov = np.cov(init_xy.T) if len(init_xy) > 2 else np.eye(2) * 0.01

    ax.scatter(init_xy[:, 0], init_xy[:, 1],
               s=12, color="#7FAED6", alpha=0.55, zorder=3,
               label="initial states ($N_{\\rm MC}=40$)")
    ell = ellipse_from_cov(init_mean, init_cov, n_std=2.0)
    ax.plot(ell[:, 0], ell[:, 1], color=PALETTE["tsh"], lw=1.2, ls=(0, (4, 2)),
            zorder=4, label="2$\\sigma$ start ellipse")
    ax.scatter([init_mean[0]], [init_mean[1]], marker="o",
               facecolors="white", edgecolors=PALETTE["tsh"], s=60, linewidth=1.5,
               zorder=5, label="init mean")
    ax.scatter([x_sp[0]], [x_sp[1]], marker="x", color=PALETTE["pd"], s=78,
               linewidths=2.0, zorder=5, label="setpoint")

    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    soften_axes(ax, "both")
    panel_title(ax, task_name)


def main():
    setup_rcparams()
    out_path = ROOT / "experiments/results_v6/figS1_tasks.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), constrained_layout=True)
    for ax, name in zip(axes, ["narrow", "two_gate", "u_shape"]):
        draw_task(ax, name, seed=7777)

    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7FAED6",
               markeredgecolor="none", markersize=5.5, alpha=0.7,
               label="initial states ($N_{\\rm MC}=40$)"),
        Line2D([0], [0], color=PALETTE["tsh"], ls=(0, (4, 2)), lw=1.2,
               label="2$\\sigma$ start ellipse"),
        Line2D([0], [0], marker="o", color=PALETTE["tsh"], markerfacecolor="white",
               markersize=5.5, lw=1.2, label="init mean"),
        Line2D([0], [0], marker="x", color=PALETTE["pd"], markersize=7.0,
               lw=0, label="setpoint"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.08))

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
