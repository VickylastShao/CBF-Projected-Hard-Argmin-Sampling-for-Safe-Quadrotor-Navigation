#!/usr/bin/env python3
"""
Generate IEEE-quality figures for PTRM-NMPC manuscript.
Uses v6 experimental data (N_MC=100, 3 seeds).

Figures per manuscript Section 6.J:
  Fig. 1: TErr vs K (K-scaling)
  Fig. 2: Baseline comparison (TErr vs method)
  Fig. 3: Robustness under process noise + mismatch
  Fig. 4: Ablation studies (selection mechanism + failure analysis)

IEEE standard: single-column 3.5in, double-column 7.16in, font 8-10pt.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# Global settings: IEEE submission standard
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
    'text.usetex': False,
    'mathtext.fontset': 'dejavuserif',
})

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_V6 = os.path.join(SCRIPT_DIR, 'results_v6')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures_ieee')
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
with open(os.path.join(RESULTS_V6, 'p02_expanded_results.json'), 'r') as f:
    P02 = json.load(f)
with open(os.path.join(RESULTS_V6, 'p03_ablation_results.json'), 'r') as f:
    P03 = json.load(f)
with open(os.path.join(RESULTS_V6, 'retrain_e1_ablation.json'), 'r') as f:
    E1 = json.load(f)


def save_fig(fig, name):
    """Save PDF and PNG."""
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.03)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {name}.pdf/.png")


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    p_hat = p / 100.0
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denom
    return max(0, (center - margin) * 100), min(100, (center + margin) * 100)


# ============================================================
# Figure 1: TErr vs K (K-scaling)
# ============================================================
def plot_fig1():
    print("\n[Figure 1] TErr vs K (K-scaling)...")
    exp_a = P02['exp_a_k_scaling']

    k_vals = [1, 10, 50]
    k_labels = ['1', '10', '50']
    terr_mean = [exp_a[f'K={k}']['terr_mean'] for k in k_vals]
    terr_std = [exp_a[f'K={k}'].get('terr_std_within', 0) for k in k_vals]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.errorbar(k_vals, terr_mean, yerr=terr_std, fmt='o-', color='#1F77B4',
                linewidth=1.5, markersize=6, capsize=3, capthick=1,
                label='PTRM-NMPC (Strong CBF)')

    ax.set_xlabel('Number of candidates $K$')
    ax.set_ylabel('Terminal tracking error $T_{\\mathrm{err}}$ (m)')
    ax.set_xscale('log')
    ax.set_xticks(k_vals)
    ax.set_xticklabels(k_labels)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')

    # Annotate key values
    for k, t in zip(k_vals, terr_mean):
        ax.annotate(f'{t:.4f}m', xy=(k, t), xytext=(0, 10),
                    textcoords='offset points', fontsize=7, ha='center',
                    color='#1F77B4')

    fig.tight_layout(pad=0.3)
    save_fig(fig, 'fig1_terr_vs_K')


# ============================================================
# Figure 2: Baseline Comparison
# ============================================================
def plot_fig2():
    print("\n[Figure 2] Baseline comparison...")
    exp_b = P02['exp_b_baselines']

    methods = ['PD_K1', 'PTRM_K50', 'MPPI_K50', 'CEM_K50']
    labels = ['PD $K\\!=\\!1$', 'PTRM $K\\!=\\!50$', 'MPPI $K\\!=\\!50$', 'CEM $K\\!=\\!50$']
    colors = ['#2CA02C', '#1F77B4', '#FF7F0E', '#9467BD']

    terr_mean = [exp_b[m]['terr_mean'] for m in methods]
    terr_std = [exp_b[m].get('terr_std_within', 0) for m in methods]
    iae_mean = [exp_b[m].get('iae_mean', 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    x = np.arange(len(methods))
    bars = ax1.bar(x, terr_mean, 0.6, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=0.3, yerr=terr_std,
                   capsize=3, error_kw={'linewidth': 0.8})
    ax1.set_ylabel('Terminal tracking error $T_{\\mathrm{err}}$ (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_title('(a) Terminal tracking error', fontsize=9)

    # Add value labels
    for bar, val in zip(bars, terr_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{val:.4f}', ha='center', va='bottom', fontsize=6.5)

    bars2 = ax2.bar(x, iae_mean, 0.6, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=0.3)
    ax2.set_ylabel('Position IAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_title('(b) Integral absolute error', fontsize=9)

    for bar, val in zip(bars2, iae_mean):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5)

    fig.tight_layout(pad=0.4)
    save_fig(fig, 'fig2_baseline_comparison')


# ============================================================
# Figure 3: Robustness Under Process Noise
# ============================================================
def plot_fig3():
    print("\n[Figure 3] Robustness under process noise...")
    exp_c = P02['exp_c_noise']

    conditions = ['nominal', 'mismatch_only', 'noise_0.01', 'noise_0.05']
    labels = ['Nominal', 'Mismatch\nonly', '$\\sigma_w$=\n0.01', '$\\sigma_w$=\n0.05']
    colors = ['#2CA02C', '#1F77B4', '#FF7F0E', '#D62728']

    # PTRM K=50 data
    succ_ptrm = [exp_c[c]['success_mean'] for c in conditions]
    terr_ptrm = [exp_c[c]['terr_mean'] for c in conditions]

    # PD K=1 data from P0-3 (b3_pd_noise)
    pd_data = P03.get('b3_pd_noise', {})
    pd_conditions = ['nominal', 'mismatch_only', 'noise_0.01', 'noise_0.05']
    succ_pd = []
    terr_pd = []
    for c in pd_conditions:
        if c in pd_data:
            succ_pd.append(pd_data[c].get('success_mean', 0))
            terr_pd.append(pd_data[c].get('terr_mean', 0))
        else:
            succ_pd.append(0)
            terr_pd.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    x = np.arange(len(conditions))
    width = 0.3

    # (a) Success rate
    bars1 = ax1.bar(x - width/2, succ_ptrm, width, label='PTRM $K\\!=\\!50$',
                    color='#1F77B4', alpha=0.85, edgecolor='black', linewidth=0.3)
    bars2 = ax1.bar(x + width/2, succ_pd, width, label='PD $K\\!=\\!1$',
                    color='#2CA02C', alpha=0.85, edgecolor='black', linewidth=0.3)

    ax1.set_ylabel('Success rate (\\%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylim(0, 110)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_title('(a) Success rate', fontsize=9)

    # (b) Terminal error
    ax2.bar(x - width/2, terr_ptrm, width, label='PTRM $K\\!=\\!50$',
            color='#1F77B4', alpha=0.85, edgecolor='black', linewidth=0.3)
    ax2.bar(x + width/2, terr_pd, width, label='PD $K\\!=\\!1$',
            color='#2CA02C', alpha=0.85, edgecolor='black', linewidth=0.3)

    ax2.set_ylabel('Terminal tracking error $T_{\\mathrm{err}}$ (m)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_title('(b) Terminal tracking error', fontsize=9)

    fig.tight_layout(pad=0.4)
    save_fig(fig, 'fig3_robustness_noise')


# ============================================================
# Figure 4: Ablation Studies
# ============================================================
def plot_fig4():
    print("\n[Figure 4] Ablation studies (two-panel)...")
    b1 = P03['b1_selection_ablation']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    # ---- Left panel: Selection mechanism ablation ----
    configs = ['PTRM_argmin', 'MPPI_weighted', 'Argmin_MPPI', 'Random_K50', 'PD_K1']
    labels_l = ['PTRM\nargmin', 'MPPI\nweighted', 'Argmin\nMPPI', 'Random', 'PD\n$K\\!=\\!1$']
    colors_l = ['#1F77B4', '#FF7F0E', '#9467BD', '#D62728', '#2CA02C']

    terr_mean = [b1[c]['terr_mean'] for c in configs]
    terr_std = [b1[c].get('terr_std_within', 0) for c in configs]

    x = np.arange(len(configs))
    bars = ax1.bar(x, terr_mean, 0.6, color=colors_l, alpha=0.85,
                   edgecolor='black', linewidth=0.3, yerr=terr_std,
                   capsize=3, error_kw={'linewidth': 0.8})

    ax1.set_ylabel('Terminal tracking error $T_{\\mathrm{err}}$ (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_l, fontsize=7)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_title('(a) Selection strategy', fontsize=9)

    for bar, val in zip(bars, terr_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=6.5)

    # ---- Right panel: Candidate generation / alpha_blend ablation ----
    # alpha=0.0 → pure TRM (fails), alpha=0.5 → TRM+PD blend, alpha=0.8, alpha=1.0 → pure PD
    alpha_keys = ['trm_rollout_k50', 'trm_pd_a05', 'trm_pd_a08', 'pd_rollout']
    alpha_vals = [0.0, 0.5, 0.8, 1.0]
    labels_r = ['TRM only\n$\\alpha\\!=\\!0$', '$\\alpha\\!=\\!0.5$', '$\\alpha\\!=\\!0.8$', 'PD only\n$\\alpha\\!=\\!1$']
    colors_r = ['#D62728', '#FF7F0E', '#1F77B4', '#2CA02C']

    terr_alpha = [E1[k]['terminal_error_mean'] for k in alpha_keys]
    succ_alpha = [E1[k]['success_rate'] for k in alpha_keys]

    x2 = np.arange(len(alpha_keys))
    bars2 = ax2.bar(x2, terr_alpha, 0.6, color=colors_r, alpha=0.85,
                    edgecolor='black', linewidth=0.3)

    ax2.set_ylabel('Terminal tracking error $T_{\\mathrm{err}}$ (m)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_r, fontsize=7)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_title('(b) Candidate generation ($\\alpha$ ablation)', fontsize=9)

    for bar, val, s in zip(bars2, terr_alpha, succ_alpha):
        label = f'{val:.2f}m' if val < 1 else f'{val:.0f}m'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                 f'{label}\n({s:.0f}%)', ha='center', va='bottom', fontsize=6)

    fig.tight_layout(pad=0.4)
    save_fig(fig, 'fig4_ablation_studies')


# ============================================================
# Main
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

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output: {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
