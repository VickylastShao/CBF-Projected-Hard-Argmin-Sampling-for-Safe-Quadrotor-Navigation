# CBF-Projected Hard-Argmin Sampling for Safe Quadrotor Navigation

[![Release](https://img.shields.io/github/v/release/VickylastShao/CBF-Projected-Hard-Argmin-Sampling-for-Safe-Quadrotor-Navigation?label=release&color=blue)](https://github.com/VickylastShao/CBF-Projected-Hard-Argmin-Sampling-for-Safe-Quadrotor-Navigation/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-full_data,_code,_figures-informational)](https://github.com/VickylastShao/CBF-Projected-Hard-Argmin-Sampling-for-Safe-Quadrotor-Navigation#quick-start)

**Code, data, and figures for the manuscript:**

> *Wide-Scale Sampling with Hard-Argmin Selection and CBF Projection for Safe Quadrotor Navigation in Narrow Passages*
> Zhuang Shao, Lijun Lei, Peng Wang, Liang Zheng, Jie Zhou. 2026.

This repository contains everything needed to **reproduce every result, table, and figure** in the manuscript: the controller implementation, all baselines, experiment scripts, raw JSON result files, trained models, and the LaTeX source.

---

## Manuscript Files

| File | Description |
|---|---|
| `manuscript.tex` | LaTeX source (16 pages, 3 figures) |
| `manuscript.pdf` | Compiled PDF |
| `manuscript.docx` | Word version (IJRA submission format) |
| `manuscript_supplementary.tex` | Supplementary material (S1–S5) |
| `manuscript_supplementary.docx` | Supplementary Word version |
| `cover_page_ijra.docx` | Cover page with author info and abstract |
| `COVER_LETTER.md` | Cover letter for journal submission |

## Method Summary

**TSH-NMPC** (Test-time Scaling with Hard-argmin): at each control step, draws $K=10$ candidate first-step controls from a wide PD-centred Gaussian ($\sigma = 5$ N per axis), scores them by short closed-loop rollout, selects the hard-argmin (no soft-weighting), and projects the result through a DT-CBF QP. The three components — wide-scale sampling, hard-argmin, DT-CBF projection — each address a different part of the narrow-passage failure mode.

## Key Results

| Comparison | TSH-NMPC | Baseline | Significance |
|---|---|---|---|
| Narrow benchmark ($K{=}10$, $N{=}80$) | 97–99% success | 64–70% (PD) | McNemar $p<0.001$ |
| +50% mass mismatch ($N{=}40$) | 77.5–82.5% | 0% (PD) | $p<0.001$ |
| CasADi+IPOPT ($H{=}20$) | binary success parity | lower TErr (0.003 m) | $p=1.0$ (binary) |
| Learned vs. random ($N{=}300$) | random: 97.7% | learned: 88.7% | $p=7.4\times10^{-6}$ |

Full CIs, Wilson intervals, and McNemar counts are in the supplementary material.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main experiment suite
python experiments/v6_quick_test.py

# Regenerate all figures
python experiments/tsh_fig1_mechanism.py
python experiments/tsh_fig2_quant.py
python experiments/tsh_fig3_comparisons.py
python experiments/tsh_figS1_tasks.py

# Convert LaTeX to DOCX
python tools/tex_to_docx_ijra.py -i manuscript.tex -o manuscript.docx
```

## Repository Layout

```
.
├── manuscript.tex                 # Main manuscript LaTeX source
├── manuscript.docx                # Main manuscript Word file
├── manuscript.pdf                 # Compiled PDF
├── manuscript_supplementary.tex   # Supplementary material
├── manuscript_supplementary.docx  # Supplementary Word file
├── cover_page_ijra.docx           # Cover page (author info, abstract)
├── COVER_LETTER.md                # Cover letter
├── make_cover_page.py             # Cover page generator
├── README.md
├── requirements.txt
│
├── quadrotor_core/                # Core simulation modules
│   ├── dynamics.py                #   6D quadrotor + DT-CCBF safety filter
│   ├── nmpc_solver.py             #   L-BFGS golden NMPC solver
│   ├── ptrm_predictor.py          #   TSH-NMPC online inference
│   ├── trm_network.py             #   TRM network (negative ablation)
│   └── training.py                #   Joint training (TRM + Q-head)
│
├── tools/
│   └── tex_to_docx_ijra.py        # LaTeX → DOCX converter
│
└── experiments/
    ├── baselines/                 #   MPPI, CEM, iCEM, MLP+CBF controllers
    ├── results_v6/                #   Raw JSON results + figure PDFs
    │   ├── *.json                 #     57 experiment result files
    │   ├── *.pt                   #     Trained models
    │   └── fig*.pdf               #     Rendered figures
    ├── tsh_fig1_mechanism.py      #   Figure 1: mechanism schematic
    ├── tsh_fig2_quant.py          #   Figure 2: main quantitative results
    ├── tsh_fig3_comparisons.py    #   Figure 3: controlled comparisons
    ├── tsh_figS1_tasks.py         #   Figure S1: benchmark geometries
    ├── tsh_figS2_tasks_3d_optional.py  # Figure S2: 3D overview (optional)
    ├── tsh_plot_style.py          #   Shared plotting style
    ├── tsh_ptrm_advantage_quick.py     # Task definitions
    ├── v6_quick_test.py           #   Main experiment runner
    ├── v6_supplement.py           #   Supplement experiment runner
    └── audit_manuscript_vs_json.py     #   Table ↔ JSON verification
```

## Reproducibility

All numerical values in the manuscript are generated from the JSON files in `experiments/results_v6/`. Run `experiments/audit_manuscript_vs_json.py` to verify every table number against its source data. Figures are regenerated by the `tsh_fig*.py` scripts. The core simulation is deterministic given a fixed random seed.
