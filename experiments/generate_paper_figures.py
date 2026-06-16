#!/usr/bin/env python3
"""Generate IEEE-grade figures for TSH-NMPC paper.

Figure 2: σ_r sweep on narrow (K=10, n=40), success / TErr / IAE vs σ_r.
Figure 3: negative ablation, A3 vs R1_s5 vs R2_s5 at K∈{1,5,10,20}, n=20.
Figure 4: cross-configuration robustness bar chart (nominal/+mass/+drag).

Output: experiments/results_v6/fig{2,3,4}.{pdf,png} at 300 DPI.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results_v6"

# IEEE single-column figure size and font
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig2_sigma_sweep():
    """σ_r ∈ {1,2,3,5,8} sweep on narrow K=10 n=40."""
    d = json.load(open(RESULTS / "sigma_sweep_n40.json"))
    sigmas = [1, 2, 3, 5, 8]
    succ, terr, iae = [], [], []
    for s in sigmas:
        rec = d["by_method"][f"R1_s{s}_K10"]
        succ.append(100.0 * rec["success_count"] / rec["n"])
        terr.append(rec["TErr_mean"])
        iae.append(rec["IAE_mean"])
    pd = d["by_method"]["PD_K10"]
    pd_succ = 100.0 * pd["success_count"] / pd["n"]
    pd_terr, pd_iae = pd["TErr_mean"], pd["IAE_mean"]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))
    sigmas_arr = np.array(sigmas)

    ax = axes[0]
    ax.plot(sigmas_arr, succ, "o-", color="C0", label="TSH-NMPC (R1_s$\\sigma_r$)")
    ax.axhline(pd_succ, ls="--", color="C3", label=f"PD baseline ({pd_succ:.1f}%)")
    ax.set_xlabel("$\\sigma_r$ [N]")
    ax.set_ylabel("Success rate [%]")
    ax.set_title("(a) Success rate")
    ax.set_ylim(50, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    ax.plot(sigmas_arr, terr, "o-", color="C0")
    ax.axhline(pd_terr, ls="--", color="C3", label=f"PD ({pd_terr:.3f})")
    ax.set_xlabel("$\\sigma_r$ [N]")
    ax.set_ylabel("TErr [m] (succ.)")
    ax.set_title("(b) Terminal error")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    ax.plot(sigmas_arr, iae, "o-", color="C0")
    ax.axhline(pd_iae, ls="--", color="C3", label=f"PD ({pd_iae:.2f})")
    ax.set_xlabel("$\\sigma_r$ [N]")
    ax.set_ylabel("IAE (succ.)")
    ax.set_title("(c) Integral abs.\\ error")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)

    # operating range shading
    for ax in axes:
        ax.axvspan(5, 8, alpha=0.10, color="green", lw=0)

    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_sigma_sweep.pdf")
    fig.savefig(RESULTS / "fig2_sigma_sweep.png")
    plt.close(fig)
    print(f"wrote {RESULTS / 'fig2_sigma_sweep.pdf'}")


def fig3_negative_ablation():
    """Negative ablation: A3 vs R1_s5 vs R2_s5 at K ∈ {1,5,10,20}, n=20."""
    d = json.load(open(RESULTS / "random_ablation_n20.json"))
    Ks = [1, 5, 10, 20]
    methods = ["A3", "R1_s5", "R2_s5", "PD"]
    labels = {"A3": "Learned (TRM) Source B",
              "R1_s5": "Random-around-PD ($\\sigma_r{=}5$, proposed)",
              "R2_s5": "Random-around-zero ($\\sigma_r{=}5$)",
              "PD": "Single-source PD"}
    colors = {"A3": "C2", "R1_s5": "C0", "R2_s5": "C1", "PD": "C3"}
    markers = {"A3": "s", "R1_s5": "o", "R2_s5": "^", "PD": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))

    # (a) success
    ax = axes[0]
    for m in methods:
        ys = []
        for K in Ks:
            rec = d["by_method"][f"{m}_K{K}"]
            ys.append(100.0 * rec["success_count"] / rec["n"])
        ax.plot(Ks, ys, markers[m] + "-", color=colors[m], label=labels[m])
    ax.set_xlabel("Candidate budget $K$")
    ax.set_ylabel("Success rate [%]")
    ax.set_title("(a) Success rate")
    ax.set_xscale("log")
    ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=7)

    # (b) TErr
    ax = axes[1]
    for m in methods:
        ys = []
        for K in Ks:
            rec = d["by_method"][f"{m}_K{K}"]
            ys.append(rec["TErr_mean"])
        ax.plot(Ks, ys, markers[m] + "-", color=colors[m], label=labels[m])
    ax.set_xlabel("Candidate budget $K$")
    ax.set_ylabel("TErr [m] (succ.)")
    ax.set_title("(b) Terminal error")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_negative_ablation.pdf")
    fig.savefig(RESULTS / "fig3_negative_ablation.png")
    plt.close(fig)
    print(f"wrote {RESULTS / 'fig3_negative_ablation.pdf'}")


def fig4_cross_config():
    """Cross-configuration bar chart: nominal vs +mass vs +drag, PD vs R1_s5."""
    files = {
        "Nominal": "sigma_sweep_n40.json",   # PD_K10 in sigma_sweep is nominal
        "+50% mass": "cross_config_mass_n40.json",
        "+50% drag": "cross_config_drag_n40.json",
    }
    labels = list(files.keys())
    pd_succ, r1_succ = [], []
    for cfg, fname in files.items():
        d = json.load(open(RESULTS / fname))
        pd = d["by_method"]["PD_K10"]
        r1 = d["by_method"]["R1_s5_K10"]
        pd_succ.append(100.0 * pd["success_count"] / pd["n"])
        r1_succ.append(100.0 * r1["success_count"] / r1["n"])

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(labels)); w = 0.36
    b1 = ax.bar(x - w/2, pd_succ, w, color="C3", label="Single-source PD")
    b2 = ax.bar(x + w/2, r1_succ, w, color="C0", label="TSH-NMPC ($\\sigma_r{=}5$)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate [%]")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    for b, v in zip(b1, pd_succ):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5)
    for b, v in zip(b2, r1_succ):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_cross_config.pdf")
    fig.savefig(RESULTS / "fig4_cross_config.png")
    plt.close(fig)
    print(f"wrote {RESULTS / 'fig4_cross_config.pdf'}")


def fig1_method_diagram():
    """Schematic of two-source candidate pooling. Pure matplotlib diagram."""
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis("off")

    boxes = [
        (0.4, 1.5, 1.6, 1.4, "PD nominal\n$u_{\\mathrm{pd}}$"),
        (2.5, 2.4, 1.7, 0.9, "Source A\n$\\mathcal{N}(u_{\\mathrm{pd}}, \\sigma_{\\mathrm{pd}}^2 I)$, $K/2$"),
        (2.5, 0.7, 1.7, 0.9, "Source B\n$\\mathcal{N}(u_{\\mathrm{pd}}, \\sigma_r^2 I)$, $K/2$"),
        (4.8, 1.5, 1.5, 1.4, "Rollout\nsimulator\n(3.5)"),
        (6.7, 1.5, 0.9, 1.4, "$\\arg\\min$"),
        (8.0, 1.5, 1.7, 1.4, "DT-CCBF\nproject (2.4)"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=1.0))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=8)
    arrows = [(2.0, 2.85, 2.5, 2.85),
              (2.0, 1.15, 2.5, 1.15),
              (4.2, 2.85, 4.8, 2.5),
              (4.2, 1.15, 4.8, 2.0),
              (6.3, 2.2, 6.7, 2.2),
              (7.6, 2.2, 8.0, 2.2)]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.text(9.7, 2.2, "$u_{\\mathrm{safe}}$", fontsize=10, va="center")
    ax.annotate("", xy=(10.0, 2.2), xytext=(9.7, 2.2),
                arrowprops=dict(arrowstyle="->", lw=0.9))
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_method.pdf")
    fig.savefig(RESULTS / "fig1_method.png")
    plt.close(fig)
    print(f"wrote {RESULTS / 'fig1_method.pdf'}")


if __name__ == "__main__":
    fig1_method_diagram()
    fig2_sigma_sweep()
    fig3_negative_ablation()
    fig4_cross_config()
    print("done")
