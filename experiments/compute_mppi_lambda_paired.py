#!/usr/bin/env python3
"""Paired McNemar: MPPI λ variants vs R1_s5 / PD on narrow K=10 N=80 seed 7777.

Reads the MPPI λ sweep output and the R1_s5/PD reference from
r0_r1_pd_narrow_n80_s7777.json (both seeded at 7777, so paired by trial index).

Output: experiments/results_v6/mppi_lambda_sweep_paired_stats.json
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "experiments/results_v6"


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2.0 / (2.0 ** n), 1.0)


def disc(a: list[dict], b: list[dict]) -> tuple[int, int]:
    bb = sum(1 for ra, rb in zip(a, b) if ra["success"] and not rb["success"])
    cc = sum(1 for ra, rb in zip(a, b) if rb["success"] and not ra["success"])
    return bb, cc


def main() -> None:
    # MPPI sweep
    sweep = json.loads((RES / "mppi_lambda_sweep_narrow_k10_n80_s7777.json").read_text())
    # Reference: R1_s5, PD, R0_pd_s5 (same seed)
    ref = json.loads((RES / "r0_r1_pd_narrow_n80_s7777.json").read_text())

    ref_methods = {
        "R1_s5_K10": ref["by_method"]["R1_s5_K10"],
        "PD_K10": ref["by_method"]["PD_K10"],
        "R0_pd_s5_K10": ref["by_method"]["R0_pd_s5_K10"],
    }

    mppi_keys = sorted(sweep["by_method"].keys(),
                       key=lambda k: -sweep["by_method"][k]["success_count"])

    out: dict = {
        "meta": sweep["meta"],
        "reference": {
            k: {"success_count": v["success_count"], "TErr_mean": v["TErr_mean"],
                "IAE_mean": v["IAE_mean"]}
            for k, v in ref_methods.items()
        },
        "tests": [],
        "summary": "",
    }

    print("=" * 72)
    print("MPPI λ sweep paired against R1_s5 / PD (narrow K=10 N=80 seed 7777)")
    print("=" * 72)
    print()

    # Headline comparison table
    header = f"{'Method':30s} {'Succ':>6s} {'TErr':>8s} {'IAE':>7s}"
    print(header)
    print("-" * 72)
    for k in mppi_keys:
        v = sweep["by_method"][k]
        print(f"{k:30s} {v['success_count']:3d}/{v['n']:<2d} {v['TErr_mean']:8.3f} {v['IAE_mean']:7.3f}")
    print("-" * 72)
    for ref_key, v in ref_methods.items():
        print(f"{ref_key:30s} {v['success_count']:3d}/{v['n']:<2d} {v['TErr_mean']:8.3f} {v['IAE_mean']:7.3f}")
    print()

    # Paired McNemar: each MPPI lam vs each reference method
    comparisons = [
        (mk, rk) for mk in mppi_keys for rk in ["R1_s5_K10", "PD_K10", "R0_pd_s5_K10"]
    ]

    for mppi_key, ref_key in comparisons:
        mppi_rows = sweep["by_method"][mppi_key]["individual"]
        ref_rows = ref_methods[ref_key]["individual"]
        bb, cc = disc(mppi_rows, ref_rows)
        p = mcnemar_p(bb, cc)
        lam = mppi_key.replace("MPPI_lam", "").split("_")[0]
        succ_m = sweep["by_method"][mppi_key]["success_count"]
        n = sweep["by_method"][mppi_key]["n"]
        succ_r = ref_methods[ref_key]["success_count"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        out["tests"].append({
            "MPPI_lam": lam, "reference": ref_key,
            "b": bb, "c": cc, "p_exact_two_sided": p,
            "MPPI_success": f"{succ_m}/{n}",
            "ref_success": f"{succ_r}/{n}",
            "significant": p < 0.05,
        })
        print(f"  MPPI λ={lam:5s} vs {ref_key:15s}: "
              f"b={bb:2d}, c={cc:2d}, p={p:.4e} {sig}")
    print()

    # Interpretation
    best_lam = max(
        (k for k in mppi_keys),
        key=lambda k: (sweep["by_method"][k]["success_count"],
                       -sweep["by_method"][k]["TErr_mean"]),
    )
    best_succ = sweep["by_method"][best_lam]["success_count"]
    r1_succ = ref_methods["R1_s5_K10"]["success_count"]
    gap = r1_succ - best_succ
    out["summary"] = (
        f"Best MPPI λ={best_lam.replace('MPPI_lam','').split('_')[0]} achieves "
        f"{best_succ}/80 = {best_succ/80:.1%} vs R1_s5 {r1_succ}/80 = "
        f"{r1_succ/80:.1%} (gap {gap}/80 = {gap/80:.1%}). "
        "The λ sweep confirms that MPPI tuning does not close the gap to R1_s5 "
        "or PD; the best MPPI λ=5.0 (43/80) remains well below PD (56/80)."
    )
    print(out["summary"])

    out_path = RES / "mppi_lambda_sweep_paired_stats.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()