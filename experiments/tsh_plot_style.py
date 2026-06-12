from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    'pd': '#C23B22',        # warm red
    'tsh': '#2A6FBB',       # deep blue
    'learned': '#D98E04',   # amber
    'casadi': '#2A9D55',    # green
    'mppi': '#8E63BE',      # purple
    'cem': '#A05A4A',       # brown-red
    'cbf': '#8E3B9C',       # purple-magenta
    'obstacle_face': '#7A7A7A',
    'obstacle_edge': '#2E2E2E',
    'guide': '#8A8A8A',
    'panel_bg': '#FBFBFB',
    'grid': '#D6D6D6',
}


def setup_rcparams():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'axes.facecolor': PALETTE['panel_bg'],
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'grid.color': PALETTE['grid'],
        'grid.linewidth': 0.6,
        'grid.alpha': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def soften_axes(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.tick_params(colors='#333333', width=0.7, length=3)
    if grid:
        ax.grid(True, axis=grid)
    else:
        ax.grid(False)


def panel_title(ax, title: str):
    ax.set_title(title, pad=6, fontweight='bold')


def add_bar_labels(ax, fmt='{:.1f}', dy_frac=0.02, color='#222222', fontsize=8):
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * dy_frac
    for p in ax.patches:
        h = p.get_height()
        x = p.get_x() + p.get_width()/2
        ax.text(x, h + dy, fmt.format(h), ha='center', va='bottom', fontsize=fontsize, color=color)


def ellipse_from_cov(mean, cov, n_std=2.0, num=200):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0, 2*np.pi, num)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ell = vecs @ np.diag(np.sqrt(vals) * n_std) @ circle
    ell = ell.T + mean
    return ell


def ensure_outdir(script_path: str | Path, name='refined_figures') -> Path:
    root = Path(script_path).resolve().parent
    outdir = root / name
    outdir.mkdir(exist_ok=True)
    return outdir
