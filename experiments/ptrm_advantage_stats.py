#!/usr/bin/env python3
"""
PTRM advantage 统计后处理：在 narrow n=100 与 D-sweep n=50 上做配对显著性检验。

不修改 raw JSON，输出 markdown 表格到 stdout 和 results_v6/ptrm_advantage_stats.md。

统计协议：
- success rate：Wilson 95% CI；配对差异用 McNemar 精确检验 (b, c)。
- TErr / IAE：仅在 (i,j 同时 success) 的 paired trials 上做 bootstrap 95% CI of mean diff。
- 报告 effective rollouts 以便公平 Pareto 比较。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_MD = ROOT / "results_v6" / "ptrm_advantage_stats.md"


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact_p(b: int, c: int) -> float:
    """精确双侧 McNemar p-value：H0: b=c 在 Binomial(b+c, 0.5) 下的双侧检验。"""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    obs = min(b, c)
    total = 2 ** n
    p = 0.0
    for k in range(0, obs + 1):
        p += comb(n, k)
    p *= 2 / total
    return min(1.0, p)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """paired bootstrap on mean(a - b); returns (mean_diff, lo, hi) 95% CI."""
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def compare_cell(cell: dict[str, Any], baseline: str, alt: str) -> dict[str, Any]:
    base = cell[baseline]["individual"]
    alts = cell[alt]["individual"]
    assert len(base) == len(alts), "trial counts must match for paired analysis"
    n = len(base)

    base_s = np.array([1 if t["success"] else 0 for t in base])
    alt_s = np.array([1 if t["success"] else 0 for t in alts])

    k_b, k_a = int(base_s.sum()), int(alt_s.sum())
    p_b, lo_b, hi_b = wilson_ci(k_b, n)
    p_a, lo_a, hi_a = wilson_ci(k_a, n)

    # McNemar: b = alt success & base fail; c = base success & alt fail
    b = int(((alt_s == 1) & (base_s == 0)).sum())
    c = int(((alt_s == 0) & (base_s == 1)).sum())
    p_val = mcnemar_exact_p(b, c)

    # paired TErr/IAE among trials where BOTH succeed
    both = (base_s == 1) & (alt_s == 1)
    n_both = int(both.sum())
    if n_both >= 5:
        terr_a = np.array([t["TErr"] for t in alts])[both]
        terr_b = np.array([t["TErr"] for t in base])[both]
        iae_a = np.array([t["IAE"] for t in alts])[both]
        iae_b = np.array([t["IAE"] for t in base])[both]
        terr_d, terr_lo, terr_hi = bootstrap_mean_diff(terr_a, terr_b, seed=11)
        iae_d, iae_lo, iae_hi = bootstrap_mean_diff(iae_a, iae_b, seed=13)
    else:
        terr_d = terr_lo = terr_hi = iae_d = iae_lo = iae_hi = float("nan")

    return {
        "baseline": baseline,
        "alt": alt,
        "n": n,
        "succ_base": (p_b, lo_b, hi_b, k_b),
        "succ_alt": (p_a, lo_a, hi_a, k_a),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": p_val,
        "n_both_succ": n_both,
        "terr_diff": (terr_d, terr_lo, terr_hi),  # alt - base
        "iae_diff": (iae_d, iae_lo, iae_hi),
    }


def fmt_pct(p: float, lo: float, hi: float) -> str:
    return f"{p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def fmt_diff(d: float, lo: float, hi: float, unit: str = "") -> str:
    if not math.isfinite(d):
        return "n/a"
    sig = "" if (lo <= 0 <= hi) else " *"
    return f"{d:+.3f}{unit} [{lo:+.3f}, {hi:+.3f}]{sig}"


def render_block(title: str, cells: list[tuple[str, dict[str, Any]]]) -> str:
    lines = [f"### {title}", ""]
    lines.append("| K / D | n | Baseline succ | Alt succ | McNemar (b, c, p) | ΔTErr (alt−base) | ΔIAE (alt−base) | n_both |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for tag, r in cells:
        lines.append(
            "| {tag} | {n} | {sb} | {sa} | ({b}, {c}, p={p:.3f}) | {dt} | {di} | {nb} |".format(
                tag=tag,
                n=r["n"],
                sb=fmt_pct(*r["succ_base"][:3]),
                sa=fmt_pct(*r["succ_alt"][:3]),
                b=r["mcnemar_b"],
                c=r["mcnemar_c"],
                p=r["mcnemar_p"],
                dt=fmt_diff(*r["terr_diff"], unit="m"),
                di=fmt_diff(*r["iae_diff"]),
                nb=r["n_both_succ"],
            )
        )
    lines.append("")
    lines.append("`*` 表示 95% CI 不含 0（i.e. statistically significant at α=0.05）。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    md_parts: list[str] = ["# PTRM advantage paired statistics", ""]

    # --- narrow n=100, D=16, alpha=0.95 ---
    d100 = json.load(open(ROOT / "results_v6" / "ptrm_advantage_narrow_alpha095_n100.json"))
    cells = []
    for K in (5, 10):
        cell = d100["tasks"]["narrow"]["results"]["D16"][f"K{K}"]
        r = compare_cell(cell, baseline="PD+Rollout", alt="TRM+PD+Rollout")
        cells.append((f"K={K}, D=16", r))
    md_parts.append("## narrow n=100, alpha_blend=0.95, alt=TRM+PD+Rollout vs base=PD+Rollout")
    md_parts.append("")
    md_parts.append(render_block("Paired comparison (narrow n=100)", cells))

    # CEM vs PD on same n=100 for context
    cem_cells = []
    for K in (5, 10):
        cell = d100["tasks"]["narrow"]["results"]["D16"][f"K{K}"]
        r = compare_cell(cell, baseline="PD+Rollout", alt="CEM")
        cem_cells.append((f"K={K} (CEM eff={K*3})", r))
    md_parts.append(render_block("CEM vs PD+Rollout (same trials, n=100)", cem_cells))

    # --- D-sweep n=50 ---
    d50 = json.load(open(ROOT / "results_v6" / "ptrm_advantage_narrow_d_sweep_n50.json"))
    dsweep_cells = []
    for D in (8, 16, 24):
        cell = d50["tasks"]["narrow"]["results"][f"D{D}"]["K10"]
        r = compare_cell(cell, baseline="PD+Rollout", alt="TRM+PD+Rollout")
        dsweep_cells.append((f"D={D}, K=10", r))
    md_parts.append("## narrow D-sweep n=50, K=10, alpha_blend=0.95, alt=TRM+PD+Rollout vs base=PD+Rollout")
    md_parts.append("")
    md_parts.append(render_block("Paired D-sweep", dsweep_cells))

    text = "\n".join(md_parts)
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[stats] written to {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
