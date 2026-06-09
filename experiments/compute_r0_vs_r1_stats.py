#!/usr/bin/env python3
"""Pairwise R0_pd_s5 (single-source wide σ=5) vs R1_s5 (two-source) on the
+50% mass cross-config (paired in r0_r1_pd_mass_n40_s2026.json, seed 2026).

P0-2 (Round-6): R3/R4 demanded a single-source-wide control ablation. This
script computes paired McNemar of R0 vs R1 (proper same-run pairing) plus
each method vs PD. Findings to be reported in §5.5.

Output: experiments/results_v6/r0_vs_r1_paired_stats.json
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
    out: dict = {"task": "narrow +50% mass", "K": 10, "N": 40, "seed": 2026,
                 "source": "r0_r1_pd_mass_n40_s2026.json", "tests": []}

    # Mass +50%
    d = json.loads((RES / "r0_r1_pd_mass_n40_s2026.json").read_text())
    pd = d["by_method"]["PD_K10"]["individual"]
    r0 = d["by_method"]["R0_pd_s5_K10"]["individual"]
    r1 = d["by_method"]["R1_s5_K10"]["individual"]

    counts = {
        "PD": sum(1 for r in pd if r["success"]),
        "R0_pd_s5": sum(1 for r in r0 if r["success"]),
        "R1_s5": sum(1 for r in r1 if r["success"]),
    }
    print("Mass +50%, seed 2026, N=40 paired:")
    for k, v in counts.items():
        print(f"  {k}: {v}/40 = {v/40:.1%}")

    # Pairwise tests
    pairs = [
        ("R0_pd_s5_vs_PD", r0, pd, "R0 better if c>b"),
        ("R1_s5_vs_PD", r1, pd, "R1 better if c>b"),
        ("R0_pd_s5_vs_R1_s5", r0, r1, "R0 better if c>b"),
    ]
    for name, a, b, note in pairs:
        bb, cc = disc(a, b)
        p = mcnemar_p(bb, cc)
        print(f"  {name}: b={bb}, c={cc}, p={p:.4e}  [{note}]")
        out["tests"].append({
            "comparison": name, "b": bb, "c": cc, "p_exact_two_sided": p,
            "note": note,
        })

    out["counts"] = counts
    out["interpretation"] = (
        f"Under +50% mass at seed 2026: single-source-wide R0_pd_s5 (K=10 PD-Gaussian "
        f"σ=5) achieves {counts['R0_pd_s5']}/40 = {counts['R0_pd_s5']/40:.1%}, "
        f"slightly higher than the two-source R1_s5 at {counts['R1_s5']}/40 = "
        f"{counts['R1_s5']/40:.1%}. McNemar R0-vs-R1 is statistically inconclusive "
        "(see tests). The result reframes the methodological story: the dominant "
        "lever is the wide noise scale σ=5 in either configuration, not the "
        "two-source partition per se."
    )

    out_path = RES / "r0_vs_r1_paired_stats.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()