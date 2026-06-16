#!/usr/bin/env python3
"""Failure-penalty sensitivity for the marginal ΔTErr / ΔIAE estimator.

P1-a: 审稿人质疑 failure penalty (TErr=10 m, IAE=20) 的 defensibility。
本脚本扫四组 (p_TErr, p_IAE) 并验证 sign of Δ 不变。

Output: experiments/results_v6/penalty_sensitivity.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "experiments/results_v6"


def main() -> None:
    d = json.loads((RES_DIR / "r1_s5_narrow_n80.json").read_text())
    rows: list[dict] = []
    print(f"{'K':>3} {'p_TErr':>7} {'p_IAE':>7} {'ΔTErr':>9} {'ΔIAE':>9}")
    for K in [5, 10, 20]:
        pd = d["by_method"][f"PD_K{K}"]["individual"]
        r1 = d["by_method"][f"R1_s5_K{K}"]["individual"]
        for pT, pI in [(5, 10), (10, 20), (20, 40), (50, 100)]:
            terr_pd = np.array([r["TErr"] if r["success"] else pT for r in pd])
            iae_pd = np.array([r["IAE"] if r["success"] else pI for r in pd])
            terr_r1 = np.array([r["TErr"] if r["success"] else pT for r in r1])
            iae_r1 = np.array([r["IAE"] if r["success"] else pI for r in r1])
            dT = float((terr_r1 - terr_pd).mean())
            dI = float((iae_r1 - iae_pd).mean())
            rows.append({
                "K": K, "penalty_TErr": pT, "penalty_IAE": pI,
                "delta_TErr": dT, "delta_IAE": dI,
            })
            print(f"{K:>3} {pT:>7} {pI:>7} {dT:>+9.3f} {dI:>+9.3f}")

    # Verify sign invariance
    signs_T = {np.sign(r["delta_TErr"]) for r in rows}
    signs_I = {np.sign(r["delta_IAE"]) for r in rows}
    assert signs_T == {-1.0}, f"ΔTErr sign not invariant: {signs_T}"
    assert signs_I == {-1.0}, f"ΔIAE sign not invariant: {signs_I}"
    print("\nSign invariance: ΔTErr<0 and ΔIAE<0 for ALL (K, penalty) combos. ✓")

    out = RES_DIR / "penalty_sensitivity.json"
    out.write_text(json.dumps({
        "source": "r1_s5_narrow_n80.json",
        "comparison": "PD_K{} vs R1_s5_K{}",
        "rows": rows,
        "sign_invariant": True,
        "note": "All ΔTErr and ΔIAE remain negative across penalty sweep "
                "(5,10) → (50,100); R1_s5 advantage robust to penalty choice.",
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
