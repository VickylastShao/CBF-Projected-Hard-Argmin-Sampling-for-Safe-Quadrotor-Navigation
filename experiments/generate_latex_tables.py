#!/usr/bin/env python3
"""
Generate LaTeX table code from v6 experimental data.
Outputs booktabs-style tables for direct inclusion in the manuscript.
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_V6 = os.path.join(SCRIPT_DIR, 'results_v6')
TABLE_DIR = os.path.join(SCRIPT_DIR, 'tables_latex')
os.makedirs(TABLE_DIR, exist_ok=True)

with open(os.path.join(RESULTS_V6, 'p02_expanded_results.json'), 'r') as f:
    P02 = json.load(f)
with open(os.path.join(RESULTS_V6, 'p03_ablation_results.json'), 'r') as f:
    P03 = json.load(f)
with open(os.path.join(RESULTS_V6, 'latency_decomposition_results.json'), 'r') as f:
    LATENCY = json.load(f)


def save_tex(name, content):
    path = os.path.join(TABLE_DIR, f'{name}.tex')
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Saved: {name}.tex")


def fmt(val, decimals=4):
    """Format float with given decimal places."""
    return f"{val:.{decimals}f}"


def fmt_pm(mean, std, decimals=4):
    """Format mean ± std."""
    return f"${fmt(mean, decimals)} \\pm {fmt(std, decimals)}$"


def fmt_ci(mean, ci_lo, ci_hi, decimals=1):
    """Format mean with Wilson CI."""
    return f"{fmt(mean, decimals)}\\% [{fmt(ci_lo, decimals)}, {fmt(ci_hi, decimals)}]"


# ============================================================
# Table 1: K-Scaling (from exp_a)
# ============================================================
def table_k_scaling():
    print("\n[Table] K-scaling...")
    exp_a = P02['exp_a_k_scaling']
    rows = []
    for k in ['K=1', 'K=10', 'K=50']:
        r = exp_a[k]
        rows.append(
            f"  ${k.replace('K=', '')}$ & "
            f"{fmt(r['success_mean'], 1)}\\% & "
            f"{fmt_pm(r['terr_mean'], r.get('terr_std_within', 0))} & "
            f"{fmt(r.get('iae_mean', 0), 2)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{K-scaling: effect of candidate count on PTRM-NMPC performance ($N_{\mathrm{MC}}=300$, 3 seeds).}
\label{tab:k_scaling}
\begin{tabular}{cccc}
\toprule
$K$ & Success Rate & $T_{\mathrm{err}}$ (m) & IAE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_k_scaling', tex)


# ============================================================
# Table 2: Baseline Comparison (from exp_b)
# ============================================================
def table_baselines():
    print("\n[Table] Baseline comparison...")
    exp_b = P02['exp_b_baselines']
    method_map = [
        ('PD_K1', 'PD $K\\!=\\!1$'),
        ('PTRM_K50', 'PTRM $K\\!=\\!50$'),
        ('MPPI_K50', 'MPPI $K\\!=\\!50$'),
        ('CEM_K50', 'CEM $K\\!=\\!50$'),
        ('MLP_CBF', 'MLP+CBF'),
    ]
    rows = []
    for key, label in method_map:
        r = exp_b[key]
        rows.append(
            f"  {label} & "
            f"{fmt(r['success_mean'], 1)}\\% & "
            f"{fmt_pm(r['terr_mean'], r.get('terr_std_within', 0))} & "
            f"{fmt(r.get('iae_mean', 0), 2)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Baseline comparison under nominal conditions ($N_{\mathrm{MC}}=300$, 3 seeds). Best nominal $T_{\mathrm{err}}$ in \textbf{bold}; MLP+CBF fails all trials.}
\label{tab:baselines}
\begin{tabular}{lccc}
\toprule
Method & Success Rate & $T_{\mathrm{err}}$ (m) & IAE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_baselines', tex)


# ============================================================
# Table 3: Robustness under noise (exp_c + b3)
# ============================================================
def table_robustness():
    print("\n[Table] Robustness...")
    exp_c = P02['exp_c_noise']
    b3 = P03['b3_pd_noise']

    cond_map = [
        ('nominal', 'Nominal', 'PD_nominal'),
        ('mismatch_only', 'Mismatch only', 'PD_mismatch'),
        ('noise_0.01', '$\\sigma_w=0.01$', 'PD_noise_0.01'),
        ('noise_0.05', '$\\sigma_w=0.05$', 'PD_noise_0.05'),
    ]
    rows = []
    for c_key, c_label, pd_key in cond_map:
        ptrm = exp_c[c_key]
        pd = b3[pd_key]
        rows.append(
            f"  {c_label} & "
            f"{fmt_ci(ptrm['success_mean'], ptrm['success_ci_lo'], ptrm['success_ci_hi'])} & "
            f"{fmt(ptrm['terr_mean'], 4)} & "
            f"{fmt_ci(pd['success_mean'], pd.get('success_ci_lo', 0), pd.get('success_ci_hi', 100))} & "
            f"{fmt(pd['terr_mean'], 4)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Robustness comparison: PTRM $K\!=\!50$ vs.\ PD $K\!=\\!1$ under model mismatch and process noise ($N_{\mathrm{MC}}=300$, 3 seeds). Wilson 95\% CI shown for success rates.}
\label{tab:robustness}
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{PTRM $K\!=\!50$} & \multicolumn{2}{c}{PD $K\!=\!1$} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Condition & Success & $T_{\mathrm{err}}$ & Success & $T_{\mathrm{err}}$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_robustness', tex)


# ============================================================
# Table 4: Selection Mechanism Ablation (b1)
# ============================================================
def table_selection_ablation():
    print("\n[Table] Selection ablation...")
    b1 = P03['b1_selection_ablation']
    config_map = [
        ('PTRM_argmin', 'PTRM argmin'),
        ('MPPI_weighted', 'MPPI weighted'),
        ('Argmin_MPPI', 'Argmin + MPPI'),
        ('Random_K50', 'Random $K\\!=\\!50$'),
        ('PD_K1', 'PD $K\\!=\\!1$'),
    ]
    rows = []
    for key, label in config_map:
        r = b1[key]
        rows.append(
            f"  {label} & "
            f"{fmt(r.get('success_mean', 0), 1)}\\% & "
            f"{fmt_pm(r['terr_mean'], r.get('terr_std_within', 0))} & "
            f"{fmt(r.get('iae_mean', 0), 2)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Selection mechanism ablation ($K\!=\!50$, $N_{\mathrm{MC}}=300$, 3 seeds). All methods use identical candidate generation (PD+Gaussian+rollout); only the selection rule differs.}
\label{tab:selection_ablation}
\begin{tabular}{lccc}
\toprule
Selection & Success Rate & $T_{\mathrm{err}}$ (m) & IAE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_selection_ablation', tex)


# ============================================================
# Table 5: MPPI Selection Comparison (b4)
# ============================================================
def table_mppi_selection():
    print("\n[Table] MPPI selection comparison...")
    b4 = P03['b4_selection_comparison']
    rows = []
    for k in ['K10', 'K50']:
        for sel in ['weighted', 'argmin']:
            key = f'{sel}_{k}'
            r = b4[key]
            rows.append(
                f"  ${k.replace('K', '')}$ & "
                f"{'Weighted' if sel == 'weighted' else 'Argmin'} & "
                f"{fmt_pm(r['terr_mean'], r.get('terr_std_within', 0))} & "
                f"{fmt(r.get('success_mean', 0), 1)}\\% \\\\"
            )

    tex = r"""\begin{table}[t]
\centering
\caption{MPPI selection comparison: importance-weighted vs.\ argmin ($N_{\mathrm{MC}}=300$, 3 seeds). Differences are within one standard deviation at both $K$ values.}
\label{tab:mppi_selection}
\begin{tabular}{cccl}
\toprule
$K$ & Selection & $T_{\mathrm{err}}$ (m) & Success \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_mppi_selection', tex)


# ============================================================
# Table 6: Multi-Obstacle / Random Obstacles (b7)
# ============================================================
def table_random_obstacles():
    print("\n[Table] Random obstacles...")
    b7 = P03['b7_random_obstacles']
    obs_map = [
        ('3_obs', '3 obstacles'),
        ('5_obs', '5 obstacles'),
        ('7_obs', '7 obstacles'),
    ]
    rows = []
    for obs_key, obs_label in obs_map:
        data = b7[obs_key]
        ptrm = data['PTRM']
        mppi = data['MPPI']
        cem = data['CEM']
        rows.append(
            f"  {obs_label} & "
            f"{fmt(ptrm['success_mean'], 1)}\\% & {fmt(ptrm['terr_mean'], 4)} & "
            f"{fmt(mppi['success_mean'], 1)}\\% & {fmt(mppi['terr_mean'], 4)} & "
            f"{fmt(cem['success_mean'], 1)}\\% & {fmt(cem['terr_mean'], 4)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Random obstacle placement: PTRM, MPPI, and CEM with 3, 5, and 7 randomly placed obstacles ($N_{\mathrm{MC}}=300$, 3 seeds).}
\label{tab:random_obstacles}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{PTRM} & \multicolumn{2}{c}{MPPI} & \multicolumn{2}{c}{CEM} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Scenario & Success & $T_{\mathrm{err}}$ & Success & $T_{\mathrm{err}}$ & Success & $T_{\mathrm{err}}$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_random_obstacles', tex)


# ============================================================
# Table 8: Setpoint Evaluation (exp_e)
# ============================================================
def table_setpoint():
    print("\n[Table] Setpoint evaluation...")
    exp_e = P02['exp_e_setpoints']
    sp_map = [
        ('SP_A', 'Setpoint A'),
        ('SP_B', 'Setpoint B'),
    ]
    method_map = [
        ('PTRM', 'PTRM $K\\!=\\!50$'),
        ('MPPI', 'MPPI $K\\!=\\!50$'),
        ('PD_K1', 'PD $K\\!=\\!1$'),
    ]
    rows = []
    for sp_key, sp_label in sp_map:
        sp_data = exp_e[sp_key]
        for i, (m_key, m_label) in enumerate(method_map):
            r = sp_data[m_key]
            prefix = sp_label if i == 0 else ''
            rows.append(
                f"  {prefix} & {m_label} & "
                f"{fmt(r.get('success_mean', 0), 1)}\\% & "
                f"{fmt_pm(r['terr_mean'], r.get('terr_std_within', 0))} & "
                f"{fmt(r.get('iae_mean', 0), 2)} \\\\"
            )
        rows.append(r"  \midrule")

    # Remove last midrule
    rows = rows[:-1]

    tex = r"""\begin{table}[t]
\centering
\caption{Additional setpoint evaluation ($K\!=\!50$, Strong CBF, $N_{\mathrm{MC}}=300$, 3 seeds).}
\label{tab:setpoint}
\begin{tabular}{llccc}
\toprule
Setpoint & Method & Success Rate & $T_{\mathrm{err}}$ (m) & IAE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_setpoint', tex)


# ============================================================
# Table 9: Dynamic Obstacle (exp_f)
# ============================================================
def table_dynamic_obstacle():
    print("\n[Table] Dynamic obstacle...")
    exp_f = P02['exp_f_dynamic_obstacle']
    method_map = [
        ('PTRM_K50', 'PTRM $K\\!=\\!50$'),
        ('MPPI_K50', 'MPPI $K\\!=\\!50$'),
        ('PD_K1', 'PD $K\\!=\\!1$'),
    ]
    rows = []
    for m_key, m_label in method_map:
        r = exp_f[m_key]
        rows.append(
            f"  {m_label} & "
            f"{fmt(r.get('success_mean', 0), 1)}\\% & "
            f"{fmt_pm(r['terr_mean'], r.get('terr_std_within', 0))} & "
            f"{fmt(r.get('iae_mean', 0), 2)} \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Dynamic obstacle scenario ($K\!=\!50$, Strong CBF, $N_{\mathrm{MC}}=300$, 3 seeds). A moving obstacle traverses the corridor during the trajectory.}
\label{tab:dynamic_obstacle}
\begin{tabular}{lccc}
\toprule
Method & Success Rate & $T_{\mathrm{err}}$ (m) & IAE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_dynamic_obstacle', tex)


# ============================================================
# Table 5: Latency Decomposition
# ============================================================
def table_latency_decomposition():
    print("\n[Table] Latency decomposition...")
    rows = []
    for key in ['K1', 'K10', 'K50', 'K100']:
        r = LATENCY[key]
        headroom = 20.0 / r['total_ms']
        rows.append(
            f"  ${r['K']}$ & "
            f"{fmt(r['gen_ms'], 2)} & "
            f"{fmt(r['rollout_all_ms'], 2)} & "
            f"{fmt(r['hysteresis_select_ms'], 3)} & "
            f"{fmt(r['cbf_ms'], 3)} & "
            f"{fmt(r['total_ms'], 2)} & "
            f"{fmt(r['total_std'], 2)} & "
            f"{fmt(headroom, 1)}$\\times$ \\\\"
        )

    tex = r"""\begin{table}[t]
\centering
\caption{Online single-step inference latency decomposition for the proposed TRM-PD hybrid candidate generation with pure rollout-all evaluation. Timing uses CPU execution, 10 warmup calls, and 50 repeated calls.}
\label{tab:latency}
\begin{tabular}{cccccccc}
\toprule
$K$ & Gen & Rollout & Hyst+Sel & CBF & Total & Total std & vs. $dt$ \\
& (ms) & (ms) & (ms) & (ms) & (ms) & (ms) & budget \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_tex('table_latency_decomposition', tex)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Generating LaTeX tables from v6 experimental data")
    print(f"Output directory: {TABLE_DIR}")
    print("=" * 60)

    table_k_scaling()
    table_baselines()
    table_robustness()
    table_selection_ablation()
    table_mppi_selection()
    table_random_obstacles()
    table_setpoint()
    table_dynamic_obstacle()
    table_latency_decomposition()

    print("\n" + "=" * 60)
    print("All LaTeX tables generated!")
    print(f"Output: {TABLE_DIR}/")
    for f in sorted(os.listdir(TABLE_DIR)):
        print(f"  {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
