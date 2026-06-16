#!/usr/bin/env python3
"""
Manuscript-vs-JSON audit (P-submission gating).

把 PTRM_NMPC_manuscript.tex 的所有 success-count / TErr / IAE / latency / fallback
数字与 experiments/results_v6/*.json 的源数据交叉比对。

输出: pass/fail 报告 + 任何不一致即拒绝投稿。
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "experiments/results_v6"
TEX_PATH = ROOT / "PTRM_NMPC_manuscript.tex"

# --- Helpers ---
def load(name: str) -> dict:
    return json.loads((RES_DIR / name).read_text())

def succ_terr_iae(d: dict, key: str) -> tuple[int, int, float, float]:
    """Return (succ_count, n_total, TErr_all_mean, IAE_all_mean).

    NOTE: TErr/IAE are *all-rows* means (matches what Table II reports),
    not success-only.  failure rows use the simulator's terminal value
    (no penalty applied — penalty is only for the marginal Δ column).
    """
    cell = d["by_method"][key]
    rows = cell["individual"]
    succ = [r for r in rows if r["success"]]
    sc = len(succ)
    n = len(rows)
    terr = float(np.mean([r["TErr"] for r in rows])) if rows else float("nan")
    iae = float(np.mean([r["IAE"] for r in rows])) if rows else float("nan")
    return sc, n, terr, iae

# --- Audits ---
issues: list[str] = []
ok_count = 0

# ===== Table II: main narrow N=80 =====
print("=" * 78)
print("Table II — Main result, narrow, N_MC=80 (seed=7777, paired)")
print("=" * 78)
r1 = load("r1_s5_narrow_n80.json")
EXP_TAB_II = {
    # K: (PD_succ, PD_TErr_allmean, PD_IAE_allmean, R1_succ, R1_TErr_allmean, R1_IAE_allmean)
    5:  (46, 0.644, 9.31, 70, 0.171, 6.94),
    10: (53, 0.380, 8.53, 79, 0.039, 6.22),
    20: (59, 0.273, 8.03, 77, 0.040, 5.82),
}
for K, (pd_s, pd_t, pd_i, r1_s, r1_t, r1_i) in EXP_TAB_II.items():
    sc_pd, n_pd, t_pd, i_pd = succ_terr_iae(r1, f"PD_K{K}")
    sc_r1, n_r1, t_r1, i_r1 = succ_terr_iae(r1, f"R1_s5_K{K}")
    line = f"K={K}: PD {sc_pd}/{n_pd} TErr={t_pd:.3f} IAE={i_pd:.3f}  R1 {sc_r1}/{n_r1} TErr={t_r1:.3f} IAE={i_r1:.3f}"
    print(line)
    # Claim mismatches
    if sc_pd != pd_s:
        issues.append(f"Tab II K={K} PD succ: tex={pd_s} vs json={sc_pd}")
    if sc_r1 != r1_s:
        issues.append(f"Tab II K={K} R1 succ: tex={r1_s} vs json={sc_r1}")
    if abs(t_pd - pd_t) > 0.005:
        issues.append(f"Tab II K={K} PD TErr: tex={pd_t} vs json={t_pd:.3f}")
    if abs(t_r1 - r1_t) > 0.005:
        issues.append(f"Tab II K={K} R1 TErr: tex={r1_t} vs json={t_r1:.3f}")
    if abs(i_pd - pd_i) > 0.01:
        issues.append(f"Tab II K={K} PD IAE: tex={pd_i} vs json={i_pd:.3f}")
    if abs(i_r1 - r1_i) > 0.01:
        issues.append(f"Tab II K={K} R1 IAE: tex={r1_i} vs json={i_r1:.3f}")
    ok_count += 6

# ===== Table VII: negative ablation N=80 =====
print()
print("=" * 78)
print("Table VII — Negative ablation, narrow, N_MC=80 (seed=7777)")
print("=" * 78)
a3 = load("probes_a3_narrow_n80.json")
r1_paired_a3 = load("probes_r1_s5_narrow_n80.json")  # paired-with-A3 R1_s5
EXP_TAB_VII = {
    # K: (A3_succ, A3_TErr_succmean, A3_IAE_succmean, R1_succ, R1_TErr_succmean, R1_IAE_succmean)
    # NOTE: Tab VII columns explicitly labelled "(succ.)" — uses success-only means
    5:  (70, 0.075, 6.45, 76, 0.052, 6.62),
    10: (69, 0.040, 6.08, 75, 0.022, 6.14),
    20: (77, 0.029, 5.91, 80, 0.018, 5.80),
}
def succ_only(d, key):
    cell = d["by_method"][key]
    rows = cell["individual"]
    succ = [r for r in rows if r["success"]]
    sc = len(succ)
    n = len(rows)
    terr = float(np.mean([r["TErr"] for r in succ])) if succ else float("nan")
    iae = float(np.mean([r["IAE"] for r in succ])) if succ else float("nan")
    return sc, n, terr, iae

for K, (a3_s, a3_t, a3_i, r1_s, r1_t, r1_i) in EXP_TAB_VII.items():
    a3_sc, _, a3_terr, a3_iae = succ_only(a3, f"A3_K{K}")
    r1_sc, _, r1_terr, r1_iae = succ_only(r1_paired_a3, f"R1_s5_K{K}")
    print(f"K={K}: A3 {a3_sc}/80 TErr={a3_terr:.3f} IAE={a3_iae:.3f}  R1 {r1_sc}/80 TErr={r1_terr:.3f} IAE={r1_iae:.3f}")
    if a3_sc != a3_s:
        issues.append(f"Tab VII K={K} A3 succ: tex={a3_s} vs json={a3_sc}")
    if r1_sc != r1_s:
        issues.append(f"Tab VII K={K} R1 succ: tex={r1_s} vs json={r1_sc}")
    if abs(a3_terr - a3_t) > 0.005:
        issues.append(f"Tab VII K={K} A3 TErr: tex={a3_t} vs json={a3_terr:.3f}")
    if abs(r1_terr - r1_t) > 0.005:
        issues.append(f"Tab VII K={K} R1 TErr: tex={r1_t} vs json={r1_terr:.3f}")
    if abs(a3_iae - a3_i) > 0.02:
        issues.append(f"Tab VII K={K} A3 IAE: tex={a3_i} vs json={a3_iae:.3f}")
    if abs(r1_iae - r1_i) > 0.02:
        issues.append(f"Tab VII K={K} R1 IAE: tex={r1_i} vs json={r1_iae:.3f}")
    ok_count += 6

# ===== §5.5.2 MPPI/CEM/iCEM nominal narrow K=10 =====
print()
print("=" * 78)
print("§5.5.2 MPPI/CEM/iCEM matched-budget nominal narrow K=10")
print("=" * 78)
mce = load("mce_narrow_n40.json")
EXP_MCE = {
    # (key in tex paragraph): (claim_succ_count, claim_lat_ms)
    "MPPI_K10": (26, 7.0),    # text: "MPPI 7.0ms ... 65%"
    "CEM_K10":  (37, 33.2),
    "iCEM_K10": (40, 58.8),
}
for k, (claim_s, claim_lat) in EXP_MCE.items():
    cell = mce["by_method"][k]
    sc = cell["success_count"]
    lat_med = float(cell.get("latency_ms_median", -1))
    print(f"{k}: succ={sc}/40 lat_median={lat_med:.2f}ms (claim={claim_s}, {claim_lat}ms)")
    if sc != claim_s:
        issues.append(f"§5.5.2 {k} succ: tex={claim_s} vs json={sc}")
    if abs(lat_med - claim_lat) > 0.2:
        issues.append(f"§5.5.2 {k} latency: tex={claim_lat} vs json={lat_med:.2f}")
    ok_count += 2

# ===== §5.5.2 +50% mass / drag MPPI K=10 =====
print()
print("MPPI K=10 cross-config:")
mce_mass = load("mce_mass_n40.json")
mce_drag = load("mce_drag_n40.json")
m_sc = mce_mass["by_method"]["MPPI_K10"]["success_count"]
d_sc = mce_drag["by_method"]["MPPI_K10"]["success_count"]
print(f"  +50% mass MPPI_K10: {m_sc}/40   (tex claim: 1/40)")
print(f"  +50% drag MPPI_K10: {d_sc}/40   (tex claim: 29/40)")
if m_sc != 1:
    issues.append(f"§5.5.2 +50% mass MPPI succ: tex=1 vs json={m_sc}")
if d_sc != 29:
    issues.append(f"§5.5.2 +50% drag MPPI succ: tex=29 vs json={d_sc}")
ok_count += 2

# ===== Cross-config (Table V) =====
print()
print("=" * 78)
print("Table V — Cross-Configuration, N_MC=40 (paired)")
print("=" * 78)
for cfg_file, cfg_name, claim_pd, claim_r1 in [
    ("cross_config_mass_n40.json", "+50% mass", 0, 31),
    ("cross_config_drag_n40.json", "+50% drag", None, 38),  # PD drag from json
]:
    d = load(cfg_file)
    pd_sc = d["by_method"]["PD_K10"]["success_count"]
    r1_sc = d["by_method"]["R1_s5_K10"]["success_count"]
    print(f"{cfg_name}: PD K=10 {pd_sc}/40   R1 K=10 {r1_sc}/40   (tex PD={claim_pd}, R1={claim_r1})")
    if claim_pd is not None and pd_sc != claim_pd:
        issues.append(f"Tab V {cfg_name} PD: tex={claim_pd} vs json={pd_sc}")
    if r1_sc != claim_r1:
        issues.append(f"Tab V {cfg_name} R1: tex={claim_r1} vs json={r1_sc}")
    ok_count += 1

# ===== §5.5.1 PI / Adaptive PD =====
print()
print("=" * 78)
print("§5.5.1 PI / Adaptive PD (+50% mass)")
print("=" * 78)
pi = load("pi_baseline_mass_n40.json")
EXP_PI = {
    "PI_K10":       (0, 2.76, 9.49),
    "Adaptive_K10": (28, 0.48, 8.97),
}
for k, (s, t, i) in EXP_PI.items():
    cell = pi["by_method"][k]
    sc = cell["success_count"]
    tm = float(cell.get("TErr_mean", -1))
    im = float(cell.get("IAE_mean", -1))
    print(f"{k}: {sc}/40 TErr_mean={tm:.3f} IAE_mean={im:.3f}")
    if sc != s:
        issues.append(f"§5.5.1 {k} succ: tex={s} vs json={sc}")
    if abs(tm - t) > 0.02:
        issues.append(f"§5.5.1 {k} TErr_mean: tex={t} vs json={tm:.3f}")
    if abs(im - i) > 0.05:
        issues.append(f"§5.5.1 {k} IAE_mean: tex={i} vs json={im:.3f}")
    ok_count += 3

# ===== §5.7 Fallback audit table =====
print()
print("=" * 78)
print("§5.7 Fallback audit — full table (PD vs R1_s5)")
print("=" * 78)
fb = load("fallback_audit_narrow.json")
EXP_FB = {
    # (config/method/K): claimed rate %
    "nominal/PD/K5":     20.4, "nominal/R1_s5/K5":  9.4,
    "nominal/PD/K10":    17.2, "nominal/R1_s5/K10": 6.9,
    "nominal/PD/K20":    13.7, "nominal/R1_s5/K20": 5.0,
    "mass/PD/K5":        48.7, "mass/R1_s5/K5":     17.0,
    "mass/PD/K10":       48.3, "mass/R1_s5/K10":    13.4,
    "mass/PD/K20":       47.5, "mass/R1_s5/K20":    11.3,
    "drag/PD/K5":        19.3, "drag/R1_s5/K5":     8.7,
    "drag/PD/K10":       15.9, "drag/R1_s5/K10":    7.1,
    "drag/PD/K20":       14.1, "drag/R1_s5/K20":    4.6,
}
for key, claim in EXP_FB.items():
    actual = float(fb["cells"][key]["fallback_rate_pct"])
    mark = "OK" if abs(actual - claim) < 0.15 else "MISMATCH"
    print(f"{key:<22} json={actual:6.3f}%  tex={claim:5.1f}%  [{mark}]")
    if abs(actual - claim) > 0.15:
        issues.append(f"§5.7 fallback {key}: tex={claim} vs json={actual:.2f}")
    ok_count += 1

# ===== §5.5 cross-task two_gate / u_shape =====
print()
print("=" * 78)
print("§5.4 cross-task two_gate / u_shape (N_MC=40)")
print("=" * 78)
for fn, task in [("r1_s5_two_gate_n40.json", "two_gate"),
                 ("r1_s5_u_shape_n40.json", "u_shape")]:
    d = load(fn)
    for k in ["PD_K10", "R1_s5_K10"]:
        cell = d["by_method"][k]
        sc = cell["success_count"]
        print(f"  {task} {k}: {sc}/40")
    ok_count += 2

# ===== Latency Table VI =====
print()
print("=" * 78)
print("Table VI — Latency (R1_s5 K=10 should be ~6.2 ms median)")
print("=" * 78)
for K in [10, 20]:
    cell = r1["by_method"][f"R1_s5_K{K}"]
    lat = [r["latency_ms_mean"] for r in cell["individual"]]
    print(f"R1_s5 K={K}: mean={np.mean(lat):.2f} median={np.median(lat):.2f} p95={np.percentile(lat,95):.2f}")
for K in [10, 20]:
    cell = r1["by_method"][f"PD_K{K}"]
    lat = [r["latency_ms_mean"] for r in cell["individual"]]
    print(f"PD    K={K}: mean={np.mean(lat):.2f} median={np.median(lat):.2f} p95={np.percentile(lat,95):.2f}")

# ===== Final report =====
print()
print("=" * 78)
print(f"AUDIT SUMMARY — checks performed: {ok_count}, mismatches: {len(issues)}")
print("=" * 78)
if issues:
    print("MISMATCHES:")
    for i in issues:
        print(f"  ✘ {i}")
else:
    print("  ✓ All audited cells match the claims in PTRM_NMPC_manuscript.tex")
print("=" * 78)
exit(1 if issues else 0)
