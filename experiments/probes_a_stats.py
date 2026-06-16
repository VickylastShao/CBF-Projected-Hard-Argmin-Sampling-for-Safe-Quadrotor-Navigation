#!/usr/bin/env python3
"""
对 probes_a_narrow_n*.json 做配对统计：
  - McNemar 精确检验：success rate paired diff
  - Wilson 95% CI：each method's success rate
  - Bootstrap 95% CI：TErr/IAE paired diff (paired success trials only)

输入: experiments/results_v6/probes_a_narrow_n*.json
输出: stdout markdown + experiments/results_v6/probes_a_stats.md

CLI:
  python3 experiments/probes_a_stats.py \
      --input experiments/results_v6/probes_a_narrow_n20.json \
      --output experiments/results_v6/probes_a_stats.md \
      --pairs A3,PD A3_mag,PD A3,TRM_PD_a095 A3_mag,TRM_PD_a095
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
Z975 = 1.959963984540054


def wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + Z975 * Z975 / n
    centre = (p + Z975 * Z975 / (2 * n)) / denom
    half = Z975 * math.sqrt(p * (1 - p) / n + Z975 * Z975 / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact_p(b: int, c: int) -> float:
    """精确双侧 McNemar: H0: b=c, binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 双侧 = 2 * P(X <= k) clipped at 1
    p_one = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_one)


def bootstrap_paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                          seed: int = 0) -> tuple[float, float, float]:
    """paired (a[i] - b[i]) 的均值 bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = float(diff[idx].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_pair(rows_a: list[dict], rows_b: list[dict],
                 label_a: str, label_b: str) -> dict:
    """rows_a, rows_b 必须 paired (相同 initial states 顺序)。"""
    n = len(rows_a)
    assert len(rows_b) == n, "paired requires equal length"
    succ_a = np.array([r["success"] for r in rows_a], dtype=bool)
    succ_b = np.array([r["success"] for r in rows_b], dtype=bool)
    sa = int(succ_a.sum()); sb = int(succ_b.sum())
    pa, la, ua = wilson_ci(sa, n)
    pb, lb, ub = wilson_ci(sb, n)
    # McNemar: b = (a succ, b fail), c = (a fail, b succ)
    b = int(np.sum(succ_a & ~succ_b))
    c = int(np.sum(~succ_a & succ_b))
    pval = mcnemar_exact_p(b, c)

    # 只在 both-success trials 上比较 TErr/IAE
    both = succ_a & succ_b
    terr_a = np.array([r["TErr"] for r in rows_a])[both]
    terr_b = np.array([r["TErr"] for r in rows_b])[both]
    iae_a = np.array([r["IAE"] for r in rows_a])[both]
    iae_b = np.array([r["IAE"] for r in rows_b])[both]
    if both.sum() >= 3:
        d_terr, lo_terr, hi_terr = bootstrap_paired_diff(terr_a, terr_b)
        d_iae, lo_iae, hi_iae = bootstrap_paired_diff(iae_a, iae_b)
    else:
        d_terr = lo_terr = hi_terr = float("nan")
        d_iae = lo_iae = hi_iae = float("nan")
    return {
        "label_a": label_a, "label_b": label_b, "n": n,
        "succ_a": sa, "succ_b": sb,
        "rate_a": pa, "rate_a_ci": (la, ua),
        "rate_b": pb, "rate_b_ci": (lb, ub),
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": pval,
        "n_both_success": int(both.sum()),
        "terr_diff_mean": d_terr, "terr_diff_ci": (lo_terr, hi_terr),
        "iae_diff_mean": d_iae, "iae_diff_ci": (lo_iae, hi_iae),
    }


def fmt_pct(x):
    return f"{100*x:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pairs", type=str, nargs="+", required=True,
                        help="形如 A3,PD A3_mag,PD")
    parser.add_argument("--k-values", type=str, default="1,5,10,20")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    by_method = data["by_method"]
    k_values = [int(k) for k in args.k_values.split(",")]
    pairs = [tuple(p.split(",")) for p in args.pairs]

    md = ["# Probes A statistical comparison", "",
          f"input: `{args.input}`  n_mc per cell = {data['meta']['n_mc']}", ""]

    for label_a, label_b in pairs:
        md.append(f"## {label_a} vs {label_b}")
        md.append("")
        md.append("| K | succ_A | succ_B | McNemar (b,c) | p | ΔTErr [95% CI] | ΔIAE [95% CI] | n_both |")
        md.append("|---|---|---|---|---|---|---|---|")
        for K in k_values:
            key_a = f"{label_a}_K{K}"
            key_b = f"{label_b}_K{K}"
            if key_a not in by_method or key_b not in by_method:
                continue
            rows_a = by_method[key_a]["individual"]
            rows_b = by_method[key_b]["individual"]
            r = compare_pair(rows_a, rows_b, label_a, label_b)
            sig = "**" if r["mcnemar_p"] < 0.05 else ""
            md.append(
                f"| {K} | {r['succ_a']}/{r['n']} ({fmt_pct(r['rate_a'])}) | "
                f"{r['succ_b']}/{r['n']} ({fmt_pct(r['rate_b'])}) | "
                f"({r['mcnemar_b']},{r['mcnemar_c']}) | {sig}{r['mcnemar_p']:.3f}{sig} | "
                f"{r['terr_diff_mean']:+.3f} [{r['terr_diff_ci'][0]:+.3f},{r['terr_diff_ci'][1]:+.3f}] | "
                f"{r['iae_diff_mean']:+.3f} [{r['iae_diff_ci'][0]:+.3f},{r['iae_diff_ci'][1]:+.3f}] | "
                f"{r['n_both_success']} |"
            )
        md.append("")

    md.append("## Notes")
    md.append("- ΔX = mean(A) - mean(B) on paired both-success trials. Negative ΔTErr/ΔIAE = A 更好.")
    md.append("- McNemar 双侧精确检验; p<0.05 显著（粗体）。")
    md.append("- Wilson 95% CI for individual rates.")

    out_text = "\n".join(md)
    Path(args.output).write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
