#!/usr/bin/env python3
"""
Marginal-estimator paired stats (P1-2) + Holm-Bonferroni (P1-3).

替换 §5.3 Table II 的 "both-success" Δ 为 marginal estimator：

  对所有 N_MC paired trial：
    TErr_i  = TErr_i if success else 10.0   (failure penalty)
    IAE_i   = IAE_i  if success else 20.0
  Δ_i = R1_i - PD_i
  bootstrap CI on mean(Δ_i) with B=10000.

对 K∈{5,10,20} 三组 McNemar p-value 做 Holm-Bonferroni 校正
(family size = 25 一并展示给用户用做规划)。

Usage:
  python experiments/compute_marginal_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "experiments/results_v6"

FAIL_TERR = 10.0   # m, per CLAUDE 指令
FAIL_IAE = 20.0    # per CLAUDE 指令
B = 10000
ALPHA = 0.05


def marginal_paired(pd_rows: list[dict], r1_rows: list[dict]) -> dict[str, Any]:
    """Marginal-estimator paired ΔTErr / ΔIAE."""
    assert len(pd_rows) == len(r1_rows), "paired length mismatch"
    n = len(pd_rows)
    terr_pd = np.array([r["TErr"] if r["success"] else FAIL_TERR for r in pd_rows])
    iae_pd = np.array([r["IAE"] if r["success"] else FAIL_IAE for r in pd_rows])
    terr_r1 = np.array([r["TErr"] if r["success"] else FAIL_TERR for r in r1_rows])
    iae_r1 = np.array([r["IAE"] if r["success"] else FAIL_IAE for r in r1_rows])
    dterr = terr_r1 - terr_pd
    diae = iae_r1 - iae_pd
    rng = np.random.default_rng(7777)
    boot_dterr = np.empty(B)
    boot_diae = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot_dterr[b] = dterr[idx].mean()
        boot_diae[b] = diae[idx].mean()
    return {
        "n": n,
        "dterr_mean": float(dterr.mean()),
        "dterr_ci": [float(np.percentile(boot_dterr, 2.5)),
                     float(np.percentile(boot_dterr, 97.5))],
        "diae_mean": float(diae.mean()),
        "diae_ci": [float(np.percentile(boot_diae, 2.5)),
                    float(np.percentile(boot_diae, 97.5))],
    }


def mcnemar_p(pd_rows: list[dict], r1_rows: list[dict]) -> tuple[int, int, float]:
    """Exact mid-p McNemar (b = PD-only, c = R1-only)."""
    from math import comb
    b = sum(1 for p, r in zip(pd_rows, r1_rows) if p["success"] and not r["success"])
    c = sum(1 for p, r in zip(pd_rows, r1_rows) if not p["success"] and r["success"])
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial
    k = min(b, c)
    p_tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_two = min(1.0, 2.0 * p_tail)
    return b, c, p_two


def holm_bonferroni(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Step-down Holm-Bonferroni: returns reject[] per test."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        thresh = alpha / (m - rank)
        if pvals[i] <= thresh:
            reject[i] = True
        else:
            break  # step-down stops at first non-reject
    return reject


def load_paired(jsonpath: Path, key_pd: str, key_r1: str) -> tuple[list, list]:
    raw = json.loads(jsonpath.read_text())
    pd_rows = raw["by_method"][key_pd]["individual"]
    r1_rows = raw["by_method"][key_r1]["individual"]
    return pd_rows, r1_rows


def main():
    print("=" * 78)
    print("Marginal-estimator paired stats — Main result (narrow, N_MC=80)")
    print("=" * 78)

    jsonpath = RES_DIR / "r1_s5_narrow_n80.json"
    if not jsonpath.exists():
        print(f"missing: {jsonpath}")
        return

    main_pvals = []
    main_keys = []
    out_rows = []

    for K in [5, 10, 20]:
        pd_rows, r1_rows = load_paired(jsonpath, f"PD_K{K}", f"R1_s5_K{K}")
        m = marginal_paired(pd_rows, r1_rows)
        b, c, p = mcnemar_p(pd_rows, r1_rows)
        out_rows.append({
            "K": K,
            "n": m["n"],
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
            "dterr_marginal_mean": m["dterr_mean"],
            "dterr_marginal_ci": m["dterr_ci"],
            "diae_marginal_mean": m["diae_mean"],
            "diae_marginal_ci": m["diae_ci"],
        })
        main_pvals.append(p)
        main_keys.append(f"K={K} narrow")
        print(f"\n[K={K}]")
        print(f"  McNemar (b={b}, c={c}) p={p:.3e}")
        print(f"  ΔTErr (marginal) = {m['dterr_mean']:+.3f}  CI=[{m['dterr_ci'][0]:+.3f}, {m['dterr_ci'][1]:+.3f}]")
        print(f"  ΔIAE  (marginal) = {m['diae_mean']:+.3f}  CI=[{m['diae_ci'][0]:+.3f}, {m['diae_ci'][1]:+.3f}]")

    # Try cross-config paired files (if present) for full Holm correction family
    extra_pvals: list[float] = []
    extra_keys: list[str] = []
    for tag, fn, methods in [
        ("mass narrow K=10", "cross_config_mass_n40.json", ("PD_K10", "R1_s5_K10")),
        ("drag narrow K=10", "cross_config_drag_n40.json", ("PD_K10", "R1_s5_K10")),
        ("two_gate K=10", "r1_s5_two_gate_n40.json", ("PD_K10", "R1_s5_K10")),
        ("u_shape K=10", "r1_s5_u_shape_n40.json", ("PD_K10", "R1_s5_K10")),
    ]:
        p_path = RES_DIR / fn
        if p_path.exists():
            try:
                pd_rows, r1_rows = load_paired(p_path, *methods)
                _, _, p = mcnemar_p(pd_rows, r1_rows)
                extra_pvals.append(p)
                extra_keys.append(tag)
            except Exception as e:
                print(f"  skip {fn}: {e}")

    all_pvals = main_pvals + extra_pvals
    all_keys = main_keys + extra_keys
    print("\n" + "=" * 78)
    print(f"Holm-Bonferroni step-down (family size = {len(all_pvals)}, α={ALPHA})")
    print("=" * 78)
    if not all_pvals:
        print("  no p-values collected")
    else:
        rejects = holm_bonferroni(all_pvals, ALPHA)
        order = sorted(range(len(all_pvals)), key=lambda i: all_pvals[i])
        for rank, i in enumerate(order):
            thr = ALPHA / (len(all_pvals) - rank)
            mark = "REJECT H0" if rejects[i] else "keep H0"
            print(f"  rank {rank+1:2d}  p={all_pvals[i]:.3e}  thr={thr:.3e}  [{mark}]  {all_keys[i]}")

    # Write JSON
    out_path = RES_DIR / "marginal_stats_main.json"
    out_path.write_text(json.dumps({
        "main_K": out_rows,
        "holm": {
            "family_size": len(all_pvals),
            "alpha": ALPHA,
            "tests": [{"key": k, "p": p, "reject": r}
                      for k, p, r in zip(all_keys, all_pvals,
                                          holm_bonferroni(all_pvals, ALPHA))],
        },
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
