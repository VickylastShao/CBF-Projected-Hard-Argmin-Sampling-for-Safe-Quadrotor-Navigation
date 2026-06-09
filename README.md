# CBF-Projected Hard-Argmin Sampling for Safe Quadrotor Navigation

**Companion code, data, and figures for the manuscript:**

> *Wide-Scale Sampling with Hard-Argmin Selection and CBF Projection for Safe Quadrotor Navigation in Narrow Passages*
> Zhuang Shao, Lijun Lei, Peng Wang, Liang Zheng, Jie Zhou. 2026.

This repository contains everything needed to **reproduce every numerical table and figure** in the manuscript: the controller implementation, all baselines, the experiment scripts, and the raw JSON result files cited in each table.

The manuscript PDF (line-numbered, single-column, 23 pages) is included at the repository root: [`PTRM_NMPC_manuscript.pdf`](PTRM_NMPC_manuscript.pdf). The cover letter for journal submission is at [`COVER_LETTER.md`](COVER_LETTER.md).

---

## Method in One Paragraph

The controller is called **TSH-NMPC** (Test-time Scaling with Hard-argmin). At each control step, it draws $K=10$ candidate first-step controls from a wide Gaussian ($\sigma = 5\,$N per axis) centered at the PD nominal, scores each candidate by a short closed-loop rollout (20 steps × 0.02 s), selects the hard-argmin (no soft-weighting), and projects the selected control onto a DT-CCBF safe set via a single QP. The three components — **(P1)** wide-scale sampling, **(P2)** hard-argmin selection, **(P3)** DT-CCBF projection — are jointly sufficient for the narrow-passage regime studied here. Removing any one collapses performance.

## Key Results at a Glance

| Comparison | Setting | TSH-NMPC | Baseline | Significance | JSON |
|---|---|---|---|---|---|
| **Main** | narrow, $K{=}10$, $N{=}80$ | 98.8% succ. | 66.3% (PD) | McNemar $p<0.001$ | `r0_vs_r1_paired_narrow_n80_s2026.json` |
| **+50% mass** | $N{=}40$ paired | 31/40 | 0/40 (PD) | $b=31,c=0,p<0.001$ | `r0_r1_pd_mass_n40_s2026.json` |
| **MPPI sweep** | $\lambda \in [0.01, 5]$, $N{=}80$ | hard-argmin wins | best $\lambda{=}5.0$: 43/80 | $p\!<\!0.001$ | `mppi_lambda_sweep_paired_stats.json` |
| **CasADi+IPOPT** | $H{=}20$, $N{=}80$ | parity | NLP solver | McNemar $p{=}1.0$ | `casadi_nmpc_narrow_n80_s7777.json` |
| **Negative ablation** | random vs. learned TRM, $N{=}300$ | random wins | learned 27,935-param | $p{=}7.4\times 10^{-6}$ | `negabl_unified_n80_s7777.json` |

Full bootstrap CIs, Wilson intervals, and Holm-corrected p-values appear in the JSON files. **No table cell in the manuscript is hand-edited**; every number is regenerated from these JSONs by `experiments/audit_manuscript_vs_json.py`.

---

## Repository Layout

```
.
├── PTRM_NMPC_manuscript.pdf      # The submission PDF (line-numbered)
├── PTRM_NMPC_manuscript.tex      # LaTeX source (single-column, generic article class)
├── COVER_LETTER.md               # Cover letter for journal submission
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── quadrotor_core/               # Controller implementation
│   ├── dynamics.py               #   6D quadrotor + DT-CCBF safety filter
│   ├── nmpc_solver.py            #   L-BFGS golden NMPC expert
│   ├── ptrm_predictor.py         #   TSH-NMPC online inference
│   ├── trm_network.py            #   TRM network (used only in negative ablation)
│   └── training.py               #   Joint training (TRM + Q-head)
│
└── experiments/
    ├── README.md                 # Detailed experiment-by-experiment guide
    ├── baselines/                #   MPPI, CEM, iCEM, CasADi+IPOPT, MLP+CBF
    ├── results_v6/               #   JSON results + figure PDFs cited in the paper
    │
    ├── audit_manuscript_vs_json.py        # Verifies every table number ↔ JSON
    ├── compute_r0_vs_r1_stats.py          # Reproduces Table II (main narrow)
    ├── compute_mass_three_seed.py         # Reproduces Table IV (+50% mass)
    ├── compute_mppi_lambda_paired.py      # Reproduces Table V (MPPI sweep)
    ├── compute_casadi_vs_tsh_kscale.py    # Reproduces Table VII (CasADi parity)
    ├── compute_holm_m13.py                # Holm-Bonferroni across m=13 primary tests
    └── plot_ieee_figures.py               # Regenerates fig1-fig5 PDFs
```

---

## Quick Reproducibility Checks (5 minutes, no GPU needed)

These commands re-derive the central manuscript numbers **from the included JSON files** — no simulation re-run required:

```bash
# 1. Verify all 11 manuscript tables match the JSON ground truth
python experiments/audit_manuscript_vs_json.py

# 2. Recompute the main paired McNemar for Table II (narrow K=10 N=80)
python experiments/compute_r0_vs_r1_stats.py

# 3. Recompute the +50% mass mismatch paired McNemar (Table IV)
python experiments/compute_mass_three_seed.py

# 4. Recompute the Holm-Bonferroni correction across m=13 primary tests
python experiments/compute_holm_m13.py
```

Each of the four scripts above prints its computed statistics to stdout and should agree with the manuscript to the last decimal place reported.

## Full Re-Run (CPU: hours; GPU: minutes)

To re-execute the underlying Monte Carlo trials (not just the JSON re-derivation):

```bash
# Install dependencies
pip install -r requirements.txt

# Main narrow benchmark, paired N=80 seeds (Table II)
python experiments/compute_matched_paired.py --task narrow --K 10 --n_mc 80 --seed 2026

# +50% mass mismatch (Table IV)
python experiments/compute_mass_three_seed.py --n_mc 40 --seeds 2026,7777,42

# MPPI temperature sweep (Table V)
python experiments/compute_mppi_lambda_paired.py --lambdas 0.01,0.05,0.1,0.5,1.0,2.0,5.0

# CasADi+IPOPT H=20 vs TSH K-scaling (Table VII)
python experiments/compute_casadi_vs_tsh_kscale.py --H 20 --K 5,10,20,50,100

# Regenerate paper figures from JSON
python experiments/plot_ieee_figures.py
```

JSON outputs land in `experiments/results_v6/`; running these scripts will **overwrite** the included files with bit-identical (modulo CUDA non-determinism) results.

---

## Installation

```bash
# Recommended: Python 3.10+ with conda
conda create -n tsh-nmpc python=3.11
conda activate tsh-nmpc
pip install -r requirements.txt
```

CUDA is **not required** — every experiment runs on CPU. CUDA speeds up the TRM training step (~10×) but does not affect the test-time controller itself, which is implemented in plain NumPy + PyTorch tensor ops.

`casadi` is only needed for the CasADi+IPOPT baseline in `experiments/baselines/casadi_nmpc_controller.py`; if you only want to reproduce TSH-NMPC and the MPPI/CEM baselines, you can skip its installation.

---

## License & Citation

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

If you use the code, data, or figures in your own work, please cite:

```bibtex
@article{Shao2026TSHNMPC,
  title  = {Wide-Scale Sampling with Hard-Argmin Selection and {CBF} Projection
            for Safe Quadrotor Navigation in Narrow Passages},
  author = {Shao, Zhuang and Lei, Lijun and Wang, Peng and
            Zheng, Liang and Zhou, Jie},
  year   = {2026},
  note   = {Manuscript under review.}
}
```

---

## Contact

- **Corresponding Author:** Zhuang Shao — shaozhuang@crpower.com.cn  ·  ORCID: [0000-0003-2496-0797](https://orcid.org/0000-0003-2496-0797)
- **Affiliation:** China Resources Power Technology Research Institute Co., Ltd., Shenzhen 518000, China

For reproducibility issues, technical questions, or peer-review correspondence, please open a GitHub issue or email the corresponding author directly.
