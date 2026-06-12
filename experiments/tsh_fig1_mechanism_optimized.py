#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from tsh_plot_style_optimized import setup_rcparams, PALETTE

setup_rcparams()
OUT = Path('experiments/results_v6/fig1_mechanism.pdf')


def draw_obstacles(ax, conservative=False):
    obs = [(-1.35, 0.55, 0.92), (1.35, -0.40, 0.92)]
    for cx, cy, r in obs:
        if conservative:
            ax.add_patch(Circle((cx, cy), r + 0.32, fill=False, ls=(0, (4, 3)),
                                lw=1.1, ec='#B7B7B7', zorder=1))
        ax.add_patch(Circle((cx, cy), r, fc=PALETTE['obstacle_face'],
                            ec=PALETTE['obstacle_edge'], lw=1.0, zorder=2))


def style_panel(ax, title: str):
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FCFCFC')
    for s in ax.spines.values():
        s.set_color('#4E4E4E')
        s.set_linewidth(0.8)
    ax.set_title(title, fontweight='bold', fontsize=9.6, pad=5)


fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.9), constrained_layout=True)
(ax1, ax2), (ax3, ax4) = axes

style_panel(ax1, '(a) Conservative barrier')
draw_obstacles(ax1, conservative=True)
ax1.annotate('', xy=(-1.35, 0.82), xytext=(-0.12, -1.96),
             arrowprops=dict(arrowstyle='-|>', lw=1.9, color=PALETTE['pd']))
ax1.text(-0.40, -1.67, 'PD nominal', color=PALETTE['pd'], fontsize=7.8)
ax1.annotate('', xy=(0.02, 1.90), xytext=(0.02, 0.45),
             arrowprops=dict(arrowstyle='-|>', lw=1.7, color=PALETTE['tsh'], linestyle='--'))
ax1.text(0.14, 1.50, 'escape\ndirection', color=PALETTE['tsh'], fontsize=7.8)
ax1.text(0.0, 0.05, 'geometric gap\n≈ 0.6 m', ha='center', va='center', fontsize=7.6,
         bbox=dict(boxstyle='round,pad=0.20', fc='#F5F1D7', ec='#A8A084', lw=0.8))
ax1.text(0.02, 0.82, 'LSE-conservative\nchannel', ha='center', va='center', fontsize=7.0,
         color='#666666')

pd0 = np.array([0.0, -1.85])

style_panel(ax2, '(b) Narrow sampling')
draw_obstacles(ax2, conservative=False)
rng = np.random.default_rng(4)
pts = rng.normal(size=(34, 2)) * np.array([0.48, 0.44]) + pd0
ax2.scatter(pts[:, 0], pts[:, 1], s=12, fc='#F0B64A', ec='none', alpha=0.85, zorder=3)
ax2.add_patch(Ellipse(pd0, 2.0, 1.7, angle=0, fill=False, lw=1.2, ls=(0, (3, 2)), ec='#D99A21'))
ax2.scatter([pd0[0]], [pd0[1]], s=30, c=PALETTE['pd'], zorder=4)
ax2.text(pd0[0] + 0.18, pd0[1] - 0.06, 'PD nominal', color=PALETTE['pd'], fontsize=7.8)
ax2.text(0.02, 2.10, 'no candidate through gap', ha='center', va='center', fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.16', fc='#F8E7E4', ec='#BA908B', lw=0.8))

style_panel(ax3, '(c) Wide sampling')
draw_obstacles(ax3, conservative=False)
rng = np.random.default_rng(7)
wide = rng.normal(size=(34, 2)) * np.array([1.35, 1.40]) + pd0
ax3.scatter(wide[:, 0], wide[:, 1], s=12, fc='#8FB6D8', ec='none', alpha=0.92, zorder=2)
ax3.add_patch(Ellipse(pd0, 5.2, 5.0, angle=0, fill=False, lw=1.2, ls=(0, (4, 2)), ec=PALETTE['tsh']))
escape_mask = (wide[:, 1] > -0.35) & (np.abs(wide[:, 0]) < 2.2)
escape = wide[escape_mask]
ax3.scatter(escape[:, 0], escape[:, 1], s=28, fc='#4CAF50', ec='#235A24', zorder=4)
selected = escape[np.argmin((escape[:, 0]-0.7)**2 + (escape[:, 1]-0.8)**2)]
ax3.annotate('escape\ncandidates', xy=(escape[:, 0].mean(), escape[:, 1].mean()),
             xytext=(-2.40, -2.10), fontsize=7.5, color='#2E7D32',
             arrowprops=dict(arrowstyle='->', lw=1.0, color='#2E7D32'))
ax3.annotate('selected', xy=selected, xytext=(1.42, 0.20), fontsize=7.5, color='#235A24',
             arrowprops=dict(arrowstyle='->', lw=1.1, color='#235A24'))
ax3.scatter([pd0[0]], [pd0[1]], s=30, c=PALETTE['pd'], zorder=4)

style_panel(ax4, '(d) Hard argmin + CBF')
draw_obstacles(ax4, conservative=False)
ax4.scatter(wide[:, 0], wide[:, 1], s=10, fc='#D7D7D7', ec='none', alpha=0.62, zorder=1)
pre = selected
post = np.array([pre[0]*0.76, pre[1]*0.66 + 0.28])
ax4.scatter([pre[0]], [pre[1]], s=34, c='#4CAF50', ec='#235A24', zorder=4)
ax4.scatter([post[0]], [post[1]], s=40, c=PALETTE['cbf'], marker='s', zorder=5)
ax4.add_patch(FancyArrowPatch(pre, post, arrowstyle='-|>', mutation_scale=11, lw=1.8, color=PALETTE['cbf']))
ax4.text(pre[0] + 0.08, pre[1] + 0.10, 'selected', fontsize=7.6, color='#235A24')
ax4.text(post[0] + 0.10, post[1] + 0.12, 'projected', fontsize=7.6, color=PALETTE['cbf'])
ax4.text(-1.95, -0.48, 'soft avg.\npulls back', fontsize=7.2, color='#6B6B6B', ha='left', va='center',
         bbox=dict(boxstyle='round,pad=0.16', fc='#F0F0F0', ec='#BEBEBE', lw=0.7))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Wrote {OUT}')
