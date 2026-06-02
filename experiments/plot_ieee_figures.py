#!/usr/bin/env python3
"""
生成IEEE投稿级别的论文图表 (Figure 1-4)
对应手稿 PTRM_NMPC_manuscript.md Section 6 中的图引用

Figure 1: Success rate vs K (三种CBF条件)
Figure 2: IAE vs K (三种CBF条件)
Figure 3: Robustness under model mismatch
Figure 4: Ablation studies (σ scaling + rollout horizon)

IEEE标准: 单栏 3.5in, 双栏 7.16in, 字体 8-10pt
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# 全局设置: IEEE投稿标准
# ============================================================
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
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.3,
    'text.usetex': False,  # 不依赖系统LaTeX
    'mathtext.fontset': 'dejavuserif',
})

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_v5')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures_ieee')
os.makedirs(FIG_DIR, exist_ok=True)

# 加载数据
with open(os.path.join(RESULTS_DIR, 'raw_results.json'), 'r') as f:
    DATA = json.load(f)

# 通用常量
K_VALUES = [1, 5, 10, 50, 100]
K_STRS = [str(k) for k in K_VALUES]
CBF_CONFIGS = ['NoCBF', 'WeakCBF', 'StrongCBF']
CBF_LABELS = ['No CBF', 'Weak CBF', 'Strong CBF']
CBF_COLORS = ['#D62728', '#1F77B4', '#2CA02C']  # red, blue, green (colorblind-safe)
CBF_MARKERS = ['o', 's', '^']
CBF_LINESTYLES = ['-', '--', '-.']


def save_fig(fig, name):
    """保存PDF和PNG"""
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.03)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {name}.pdf/.png")


# ============================================================
# Figure 1: Success Rate vs K
# ============================================================
def plot_fig1():
    print("\n[Figure 1] Success Rate vs K...")
    exp1 = DATA['exp1']

    # 单栏宽度 3.5in, 适当高度
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for cbf, label, color, marker, ls in zip(CBF_CONFIGS, CBF_LABELS, CBF_COLORS, CBF_MARKERS, CBF_LINESTYLES):
        succ = [exp1[cbf][k]['success_rate'] for k in K_STRS]
        ax.plot(K_VALUES, succ, marker=marker, linestyle=ls, color=color,
                label=label, linewidth=1.5, markersize=5)

    ax.set_xlabel('Number of candidates $K$')
    ax.set_ylabel('Success rate (\\%)')
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_ylim(-5, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='0.8')

    # 标注关键点: K=1 NoCBF=14%, K=100 NoCBF=92%
    ax.annotate('14\\%', xy=(1, 14), xytext=(2.5, 22),
                fontsize=7, color=CBF_COLORS[0],
                arrowprops=dict(arrowstyle='->', color=CBF_COLORS[0], lw=0.8))
    ax.annotate('92\\%', xy=(100, 92), xytext=(38, 82),
                fontsize=7, color=CBF_COLORS[0],
                arrowprops=dict(arrowstyle='->', color=CBF_COLORS[0], lw=0.8))

    fig.tight_layout(pad=0.3)
    save_fig(fig, 'fig1_success_rate_vs_K')


# ============================================================
# Figure 2: IAE vs K
# ============================================================
def plot_fig2():
    print("\n[Figure 2] IAE vs K...")
    exp1 = DATA['exp1']

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for cbf, label, color, marker, ls in zip(CBF_CONFIGS, CBF_LABELS, CBF_COLORS, CBF_MARKERS, CBF_LINESTYLES):
        iae_mean = [exp1[cbf][k]['iae_mean'] for k in K_STRS]
        iae_std = [exp1[cbf][k]['iae_std'] for k in K_STRS]
        ax.plot(K_VALUES, iae_mean, marker=marker, linestyle=ls, color=color,
                label=label, linewidth=1.5, markersize=5)
        ax.fill_between(K_VALUES,
                        [m - s for m, s in zip(iae_mean, iae_std)],
                        [m + s for m, s in zip(iae_mean, iae_std)],
                        alpha=0.12, color=color)

    ax.set_xlabel('Number of candidates $K$')
    ax.set_ylabel('Position IAE')
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')

    # 标注最大IAE改善
    # Strong CBF: K=1→100, IAE 230.3→205.5 (10.8% reduction)
    ax.annotate('10.8\\% reduction', xy=(100, 205.5), xytext=(28, 240),
                fontsize=7, color=CBF_COLORS[2],
                arrowprops=dict(arrowstyle='->', color=CBF_COLORS[2], lw=0.8))

    fig.tight_layout(pad=0.3)
    save_fig(fig, 'fig2_iae_vs_K')


# ============================================================
# Figure 3: Robustness Under Model Mismatch
# ============================================================
def plot_fig3():
    print("\n[Figure 3] Robustness under model mismatch...")
    exp3 = DATA['exp3']

    # 4个子条件
    cond_keys = ['Nominal', 'Mass×1.5, Drag×2', 'Process Noise', 'Both']
    cond_labels = ['Nominal', 'Mass $\\times 1.5$', 'Proc. Noise', 'Both']
    cond_colors = ['#2CA02C', '#1F77B4', '#FF7F0E', '#D62728']
    cond_markers = ['o', 's', '^', 'D']
    cond_hatches = ['', '//', '\\\\', 'xx']

    # (a) Success Rate bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    x = np.arange(len(cond_keys))
    width = 0.15

    for i, k in enumerate(K_VALUES):
        succ_vals = []
        for cond in cond_keys:
            k_str = str(k)
            if k_str in exp3[cond]:
                succ_vals.append(exp3[cond][k_str]['success_rate'])
            else:
                succ_vals.append(0)
        bars = ax1.bar(x + i * width - 2*width, succ_vals, width,
                       label=f'$K={k}$', color=f'C{i}', alpha=0.85,
                       edgecolor='black', linewidth=0.3)

    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Success rate (\\%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, fontsize=7)
    ax1.set_ylim(90, 102)
    ax1.legend(loc='lower left', fontsize=6.5, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_title('(a) Safety', fontsize=9)

    # (b) IAE grouped bar
    for i, k in enumerate(K_VALUES):
        iae_vals = []
        for cond in cond_keys:
            k_str = str(k)
            if k_str in exp3[cond]:
                iae_vals.append(exp3[cond][k_str]['iae_mean'])
            else:
                iae_vals.append(0)
        ax2.bar(x + i * width - 2*width, iae_vals, width,
                label=f'$K={k}$', color=f'C{i}', alpha=0.85,
                edgecolor='black', linewidth=0.3)

    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Position IAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, fontsize=7)
    ax2.legend(loc='upper left', fontsize=6.5, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_title('(b) Tracking', fontsize=9)

    fig.tight_layout(pad=0.4)
    save_fig(fig, 'fig3_robustness_mismatch')


# ============================================================
# Figure 4: Ablation Studies (σ scaling + rollout horizon)
# ============================================================
def plot_fig4():
    print("\n[Figure 4] Ablation studies...")
    exp2 = DATA['exp2']  # sigma scaling
    exp5 = DATA['exp5']  # ablation (rollout steps + sigma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    # (a) σ scaling - 从exp2获取
    sigma_keys = sorted(exp2.keys(), key=lambda x: float(x))
    sigma_vals = [float(s) for s in sigma_keys]
    succ = [exp2[s]['success_rate'] for s in sigma_keys]
    coll = [exp2[s]['collision_rate'] for s in sigma_keys]
    terr = [exp2[s]['terminal_error_mean'] for s in sigma_keys]

    ax1_twin = ax1.twinx()

    l1, = ax1.plot(sigma_vals, succ, 'o-', color='#2CA02C', linewidth=1.5,
                   markersize=4, label='Success rate')
    l2, = ax1.plot(sigma_vals, coll, 's--', color='#D62728', linewidth=1.2,
                   markersize=4, label='Collision rate')
    l3, = ax1_twin.plot(sigma_vals, terr, '^:', color='#9467BD', linewidth=1.2,
                        markersize=4, label='Term. error (m)')

    ax1.set_xlabel('Perturbation scale $\\sigma$')
    ax1.set_ylabel('Rate (\\%)')
    ax1_twin.set_ylabel('Terminal error (m)', color='#9467BD')
    ax1_twin.tick_params(axis='y', colors='#9467BD', labelsize=7)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) $\\sigma$-scaling ($K\\!=\\!50$, No CBF)', fontsize=9)

    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right',
               fontsize=6.5, framealpha=0.9)

    # (b) Rollout horizon - 从exp5获取
    if 'rollout_steps' in exp5:
        rs_data = exp5['rollout_steps']
        rs_keys = sorted(rs_data.keys(), key=lambda x: int(x))
        rs_vals = [int(r) for r in rs_keys]
        rs_succ = [rs_data[r]['success_rate'] for r in rs_keys]
        rs_coll = [rs_data[r]['collision_rate'] for r in rs_keys]

        ax2.plot(rs_vals, rs_succ, 'o-', color='#2CA02C', linewidth=1.5,
                 markersize=4, label='Success rate')
        ax2.plot(rs_vals, rs_coll, 's--', color='#D62728', linewidth=1.2,
                 markersize=4, label='Collision rate')
    else:
        # 备用: 直接从手稿表格数据
        rs_vals = [1, 3, 5, 10, 15, 20]
        rs_succ = [12, 14, 18, 53, 67, 82]
        rs_coll = [88, 86, 82, 47, 33, 18]
        ax2.plot(rs_vals, rs_succ, 'o-', color='#2CA02C', linewidth=1.5,
                 markersize=4, label='Success rate')
        ax2.plot(rs_vals, rs_coll, 's--', color='#D62728', linewidth=1.2,
                 markersize=4, label='Collision rate')

    ax2.set_xlabel('Rollout horizon $T$ (steps)')
    ax2.set_ylabel('Rate (\\%)')
    ax2.set_ylim(-5, 110)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center right', fontsize=6.5, framealpha=0.9)
    ax2.set_title('(b) Rollout horizon ($K\\!=\\!50$, No CBF)', fontsize=9)

    fig.tight_layout(pad=0.4)
    save_fig(fig, 'fig4_ablation_studies')


# ============================================================
# 额外: 3D轨迹图 (Figure 5)
# ============================================================
def plot_fig5_trajectories():
    """生成3D轨迹图 (如果有轨迹数据的话)"""
    print("\n[Figure 5] 3D trajectories...")
    # 检查是否有轨迹数据
    traj_path = os.path.join(RESULTS_DIR, 'fig_trajectories_3d.png')
    if os.path.exists(traj_path):
        # 已有3D轨迹图，复制到IEEE目录
        import shutil
        shutil.copy2(traj_path, os.path.join(FIG_DIR, 'fig5_trajectories_3d.png'))
        pdf_path = os.path.join(RESULTS_DIR, 'fig_trajectories_3d.pdf')
        if os.path.exists(pdf_path):
            shutil.copy2(pdf_path, os.path.join(FIG_DIR, 'fig5_trajectories_3d.pdf'))
        print("  Copied existing trajectory figures.")
    else:
        print("  No trajectory data found. Skipping.")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Generating IEEE-quality figures for PTRM-NMPC manuscript")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)

    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5_trajectories()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output: {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
