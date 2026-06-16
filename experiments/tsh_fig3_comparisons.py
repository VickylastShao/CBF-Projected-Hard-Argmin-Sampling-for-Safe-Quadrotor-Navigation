#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tsh_plot_style import setup_rcparams, soften_axes, panel_title, PALETTE

setup_rcparams()
OUT = Path('experiments/results_v6/fig3_comparisons.pdf')

fig = plt.figure(figsize=(9.6, 3.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

# (a) learned vs random, horizontal for readability
methods = ['A3 learned', 'R0, σ = 5 random', 'R1, σ = 5 random']
succ = np.array([88.7, 98.7, 97.7])
counts = ['266/300', '296/300', '293/300']
colors = [PALETTE['learned'], PALETTE['tsh'], PALETTE['tsh']]
y = np.arange(len(methods))
ax1.barh(y, succ, color=colors, height=0.55)
for yi, s, cnt in zip(y, succ, counts):
    ax1.text(s + 0.4, yi, f'{cnt}   {s:.1f}%', va='center', ha='left', fontsize=7.6)
ax1.set_yticks(y)
ax1.set_yticklabels(methods)
ax1.set_xlim(80, 102)
ax1.set_xlabel('Success rate (%)')
panel_title(ax1, '(a) Random vs learned source')
soften_axes(ax1, 'x')
ax1.text(82.0, 0.38, 'coverage >\nprediction', color=PALETTE['learned'], fontsize=7.4,
         bbox=dict(boxstyle='round,pad=0.18', fc='#FFF4DD', ec='#E0C06E', lw=0.8))

# (b) hard-argmin vs MPPI/CEM, horizontal bars
methods_b = ['Hard argmin', 'CEM (σ = 5, iter = 3)', 'MPPI (σ = 5)']
terr = np.array([0.036, 0.110, 0.160])
cols_b = [PALETTE['tsh'], PALETTE['cem'], PALETTE['mppi']]
yb = np.arange(len(methods_b))
ax2.barh(yb, terr, color=cols_b, height=0.55)
for yi, t in zip(yb, terr):
    ax2.text(t + 0.004, yi, f'{t:.3f}', va='center', ha='left', fontsize=7.6)
ax2.set_yticks(yb)
ax2.set_yticklabels(methods_b)
ax2.set_xlabel('Terminal error TErr [m]')
ax2.set_xlim(0, 0.19)
panel_title(ax2, '(b) MPPI/CEM vs hard argmin')
soften_axes(ax2, 'x')
ax2.text(0.146, 1.9, '4.4× worse', color=PALETTE['mppi'], fontsize=7.4)
ax2.text(0.106, 1.1, '3.3× worse\n(3× budget)', color=PALETTE['cem'], fontsize=7.2)

# (c) CasADi vs TSH trade-off
lat = np.array([5.3, 6.0, 10.0, 20.0])
terr_sc = np.array([0.003, 0.032, 0.015, 0.044])
labels = ['CasADi H = 20', 'TSH K = 20', 'TSH K = 50', 'TSH K = 150']
colors_sc = [PALETTE['casadi'], PALETTE['tsh'], PALETTE['tsh'], PALETTE['tsh']]
markers = ['o', 'o', 'o', 'o']
sizes = [48, 44, 44, 44]
ax3.plot(lat[1:], terr_sc[1:], color=PALETTE['tsh'], lw=1.4, ls='--', alpha=0.7)
for x, yv, lab, c, m, s in zip(lat, terr_sc, labels, colors_sc, markers, sizes):
    ax3.scatter([x], [yv], s=s, color=c, edgecolor='white', linewidth=1.0, zorder=3)
    dx, dy = (5, 5)
    if 'CasADi' in lab:
        dx, dy = (4, 2)
    elif '150' in lab:
        dx, dy = (-14, 4)
    ax3.annotate(lab, (x, yv), textcoords='offset points', xytext=(dx, dy), fontsize=7.1)
ax3.set_xlabel('Mean latency [ms]')
ax3.set_ylabel('Terminal error TErr [m]')
ax3.set_xlim(4.5, 21.5)
ax3.set_ylim(0, 0.048)
panel_title(ax3, '(c) CasADi–TSH trade-off')
soften_axes(ax3, 'both')
ax3.text(8.1, 0.026, 'lower TErr\n& lower latency', fontsize=7.2, color='#666666',
         bbox=dict(boxstyle='round,pad=0.18', fc='#F5F5F5', ec='#C8C8C8', lw=0.7))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Wrote {OUT}')
