#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tsh_plot_style_optimized import setup_rcparams, soften_axes, panel_title, PALETTE

setup_rcparams()
OUT = Path('experiments/results_v6/fig2_quant.pdf')

K = np.array([5, 10, 20])
pd_succ = np.array([57.5, 66.3, 73.8])
tsh_succ = np.array([87.5, 98.8, 96.3])
pd_terr = np.array([0.644, 0.380, 0.273])
tsh_terr = np.array([0.171, 0.039, 0.040])

sigma = np.array([1, 2, 3, 5, 8])
sigma_succ = np.array([75.0, 90.0, 85.0, 100.0, 100.0])

cfg = ['Nominal', '+50% mass', '+50% drag']
pd_rob = np.array([67.5, 0.0, 65.0])
tsh_rob = np.array([100.0, 77.5, 95.0])

fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.55), constrained_layout=True)
(ax1, ax2), (ax3, ax4) = axs

for y, c, lab in [(pd_succ, PALETTE['pd'], 'PD (σ = 2 N)'), (tsh_succ, PALETTE['tsh'], 'TSH-NMPC (σ = 5 N)')]:
    ax1.plot(K, y, marker='o', ms=5.3, lw=2.0, color=c, label=lab)
    ax1.scatter(K, y, s=34, color='white', edgecolor=c, linewidth=1.4, zorder=3)
ax1.set_xticks(K)
ax1.set_xlabel('Rollout budget K')
ax1.set_ylabel('Success rate (%)')
ax1.set_ylim(50, 102)
panel_title(ax1, '(a) Success rate vs K')
soften_axes(ax1, 'both')
ax1.legend(loc='lower right', frameon=False)
ax1.annotate('TSH at K = 5\noutperforms PD at K = 20', xy=(20, 73.8), xytext=(9.0, 82.2),
             fontsize=7.6, color=PALETTE['tsh'], arrowprops=dict(arrowstyle='->', lw=1.0, color=PALETTE['tsh']))

for y, c in [(pd_terr, PALETTE['pd']), (tsh_terr, PALETTE['tsh'])]:
    ax2.plot(K, y, marker='o', ms=5.3, lw=2.0, color=c)
    ax2.scatter(K, y, s=34, color='white', edgecolor=c, linewidth=1.4, zorder=3)
ax2.set_xticks(K)
ax2.set_xlabel('Rollout budget K')
ax2.set_ylabel('Terminal error TErr [m]')
panel_title(ax2, '(b) Terminal error vs K')
soften_axes(ax2, 'both')
ax2.annotate('9.8× lower\nat K = 10', xy=(10, 0.039), xytext=(12.8, 0.18), fontsize=7.6,
             color=PALETTE['tsh'], arrowprops=dict(arrowstyle='->', lw=1.0, color=PALETTE['tsh']))

ax3.axvspan(4.4, 8.2, color='#DDEAF7', alpha=0.55, zorder=0)
ax3.plot(sigma, sigma_succ, color=PALETTE['tsh'], marker='o', ms=5.3, lw=2.0)
ax3.scatter(sigma[sigma < 5], sigma_succ[sigma < 5], s=38, color=PALETTE['pd'], edgecolor='white', linewidth=1.2, zorder=3)
ax3.scatter(sigma[sigma >= 5], sigma_succ[sigma >= 5], s=38, color=PALETTE['tsh'], edgecolor='white', linewidth=1.2, zorder=3)
ax3.set_xticks(sigma)
ax3.set_xlabel('Sampling scale σ_r [N]')
ax3.set_ylabel('Success rate (%)')
ax3.set_ylim(70, 103)
panel_title(ax3, '(c) σ_r sweep (K = 10)')
soften_axes(ax3, 'both')
ax3.text(6.2, 101.2, 'operating plateau', ha='center', va='top', fontsize=7.4, color=PALETTE['tsh'],
         bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#AAC6E8', lw=0.8))

x = np.arange(len(cfg)); w = 0.32
ax4.bar(x - w/2, pd_rob, width=w, color=PALETTE['pd'], label='PD', alpha=0.95)
ax4.bar(x + w/2, tsh_rob, width=w, color=PALETTE['tsh'], label='TSH-NMPC', alpha=0.95)
for xi, y in zip(x - w/2, pd_rob):
    ax4.text(xi, max(y, 1.5) + 2.0, f'{y:.0f}', ha='center', va='bottom', fontsize=7.3, color=PALETTE['pd'])
for xi, y in zip(x + w/2, tsh_rob):
    label = f'{y:.1f}' if abs(y - round(y)) > 1e-9 else f'{y:.0f}'
    ax4.text(xi, y + 2.0, label, ha='center', va='bottom', fontsize=7.3, color=PALETTE['tsh'])
ax4.set_xticks(x)
ax4.set_xticklabels(cfg)
ax4.set_ylabel('Success rate (%)')
ax4.set_ylim(0, 110)
panel_title(ax4, '(d) Robustness (K = 10, N_MC = 40)')
soften_axes(ax4, 'y')
ax4.annotate('PD collapse', xy=(1 - w/2, 0), xytext=(0.58, 19), fontsize=7.4,
             color=PALETTE['pd'], arrowprops=dict(arrowstyle='->', lw=1.0, color=PALETTE['pd']))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Wrote {OUT}')
