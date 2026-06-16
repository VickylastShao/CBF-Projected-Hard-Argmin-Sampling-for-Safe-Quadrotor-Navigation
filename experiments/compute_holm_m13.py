#!/usr/bin/env python3
"""Compute Holm m=13 corrected p-values for the primary paired-test family.

P0-3 (Round-6): marginal_stats_main.json only ships m=10 (the 7 main+cross tests
+ 3 A3-vs-R1_s5 negative-ablation tests). Manuscript §V claims m=13: m=10 + 3
'A3-vs-PD' cells from probes_a3_narrow_n80.json (properly paired — same run).

McNemar paired tests MUST use individual records from the SAME experiment
(shared initial-state RNG and shared controller-noise RNG). Cross-file pairing
is invalid due to RNG drift (P0-4).

We take the verified m=10 values from marginal_stats_main.json and augment
with the 3 A3-vs-PD cells computed fresh from probes_a3_narrow_n80.json.
The 3 cross-task cells (two_gate, u_shape) have no discordant pairs so p=1.
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
    p = sum(comb(n, i) for i in range(k + 1)) * 2.0 / (2.0 ** n)
    return min(p, 1.0)


def paired_disc(rows_a: list[dict], rows_b: list[dict]) -> tuple[int, int]:
    b = sum(1 for ra, rb in zip(rows_a, rows_b) if ra["success"] and not rb["success"])
    c = sum(1 for ra, rb in zip(rows_a, rows_b) if rb["success"] and not ra["success"])
    return b, c


def holm_reject(ps: list[tuple[str, float, dict]], m: int, alpha: float = 0.05) -> list[dict]:
    sorted_ps = sorted(ps, key=lambda kv: kv[1])
    out = []
    still_reject = True
    for rank, (key, p, meta) in enumerate(sorted_ps, start=1):
        thresh = alpha / max(m - rank + 1, 1)
        reject = still_reject and (p <= thresh)
        if not reject:
            still_reject = False
        out.append({
            "key": key, "p": p, "rank": rank,
            "holm_m13_threshold": thresh, "reject_holm_m13": reject,
            **meta,
        })
    return out


def main() -> None:
    # ── Load existing m=10 values from marginal_stats_main.json ──
    msm = json.loads((RES / "marginal_stats_main.json").read_text())
    existing_tests = msm["holm_realized_m10"]["tests"]
    # Map existing tests by key
    exist_by_key: dict[str, dict] = {t["key"]: t for t in existing_tests}

    # ── Compute the 3 new A3-vs-PD cells from probes_a3_narrow_n80.json ──
    da = json.loads((RES / "probes_a3_narrow_n80.json").read_text())
    a3_vs_pd_tests: list[tuple[str, float, dict]] = []
    for K in (5, 10, 20):
        pd = da["by_method"][f"PD_K{K}"]["individual"]
        a3 = da["by_method"][f"A3_K{K}"]["individual"]
        b, c = paired_disc(pd, a3)
        p = mcnemar_p(b, c)
        # Direction check: is A3 better (c > b)?
        dir_text = "A3 better" if c > b else ("PD better" if b > c else "tie")
        a3_vs_pd_tests.append((f"negabl A3-vs-PD K={K}", p, {
            "n_pair": len(pd), "b": b, "c": c,
            "pair_type": "A3 vs PD (same run, properly paired)",
            "direction": dir_text,
        }))
        print(f"A3-vs-PD K={K}: b={b}, c={c}, p={p:.4e} ({dir_text})")

    # ── Assemble full m=13 test list ──
    # The 7 main+cross tests from existing (keys starting with 'main' or 'crossconfig')
    # The 2 cross-task tests from existing (keys starting with 'crosstask')
    # The 3 negabl A3-vs-R1 tests from existing (keys starting with 'negabl A3-vs-R1')
    # Plus 3 new A3-vs-PD tests

    def get_existing(key_pattern: str) -> list[tuple[str, float, dict]]:
        matches = [(k, v) for k, v in exist_by_key.items() if k.startswith(key_pattern)]
        matches.sort(key=lambda kv: kv[0])
        return [(k, v["p"], {
            "n_pair": v.get("n_pair", v.get("n", 80)),
            "b": v.get("b", 0),
            "c": v.get("c", 0),
            "pair_type": "existing (marginal_stats_main)",
        }) for k, v in matches]

    tests: list[tuple[str, float, dict]] = []
    tests += get_existing("main narrow")
    tests += get_existing("crossconfig")
    tests += get_existing("crosstask")
    tests += get_existing("negabl A3vsR1")
    tests += a3_vs_pd_tests

    assert len(tests) == 13, f"expected 13 tests, got {len(tests)}"

    print(f"\n{'Test':30s} {'n':>4} {'b':>4} {'c':>4} {'p':>12} {'Holm m=13':>10}")
    results = holm_reject(tests, m=13)
    # Also compute m=28 sensitivity
    results28 = holm_reject(tests, m=28)
    by28 = {r["key"]: r for r in results28}

    for r in sorted(results, key=lambda x: x["rank"]):
        key = r["key"][:30]
        meta = {k: tests[[t[0] for t in tests].index(r["key"])][2][k]
                for k in ("n_pair", "b", "c")}
        print(f"{key:30s} {meta['n_pair']:>4d} {meta['b']:>4d} {meta['c']:>4d} "
              f"{r['p']:>12.4e} {'reject' if r['reject_holm_m13'] else 'fail':>10s}")

    n_rej = sum(1 for r in results if r["reject_holm_m13"])
    n_rej28 = sum(1 for r in results if r.get("reject_holm_m13", False) and
                  by28[r["key"]]["reject_holm_m13"])
    n28 = len(results28)
    print(f"\nHolm m=13: {n_rej}/{len(results)} reject")
    print(f"Holm m=28 sensitivity: {n_rej}/{n28} would survive")

    # ── Write output ──
    out_tests = []
    for r in sorted(results, key=lambda x: x["rank"]):
        key = r["key"]
        idx = [t[0] for t in tests].index(key)
        meta_src = tests[idx][2]
        out_tests.append({
            "key": key,
            "n_pair": meta_src.get("n_pair"),
            "b": meta_src.get("b"),
            "c": meta_src.get("c"),
            "p": r["p"],
            "rank": r["rank"],
            "holm_m13_threshold": r["holm_m13_threshold"],
            "reject_holm_m13": r["reject_holm_m13"],
            "holm_m28_threshold": by28[key]["holm_m13_threshold"],
            "reject_holm_m28_sensitivity": by28[key]["reject_holm_m13"],
            "pair_type": meta_src.get("pair_type"),
            "direction": meta_src.get("direction", "R1_s5 better"),
        })

    out_path = RES / "holm_m13_primary_family.json"
    out_path.write_text(json.dumps({
        "family_size_primary": 13,
        "family_size_sensitivity": 28,
        "alpha": 0.05,
        "source": (
            "7 main+cross tests: marginal_stats_main.json (holm_realized_m10). "
            "2 cross-task tests: marginal_stats_main.json (same). "
            "3 A3-vs-R1_s5 tests: marginal_stats_main.json (same). "
            "3 A3-vs-PD tests: recomputed from probes_a3_narrow_n80.json (same-run paired)."
        ),
        "tests": out_tests,
    }, indent=2))
    print(f"\nwrote {out_path}")

    # Patch into marginal_stats_main.json
    msm["holm_m13_primary"] = {
        "family_size": 13,
        "alpha": 0.05,
        "tests": out_tests,
        "note": "primary family from §5.1 multiple-comparison correction; "
                "augmented from m=10 (exist) + 3 A3-vs-PD cells from probes_a3.",
    }
    (RES / "marginal_stats_main.json").write_text(json.dumps(msm, indent=2))
    print(f"patched marginal_stats_main.json with holm_m13_primary")


if __name__ == "__main__":
    main()