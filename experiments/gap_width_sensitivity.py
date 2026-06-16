#!/usr/bin/env python3
"""
Gap-width sensitivity experiment.

Purpose: Verify that the three-pillar mechanism (P1: wide σ, P2: hard-argmin,
P3: CBF) is robust to changes in obstacle gap width. The current manuscript
only tests at one fixed gap geometry; this experiment sweeps the gap width
by scaling all obstacle radii with a multiplier r_scale.

Design:
- Base task: task_narrow (5 obstacles, radii [0.55, 0.50, 0.50, 0.45, 0.45])
- r_scale ∈ {0.80, 0.90, 1.00, 1.10, 1.15}
  - r_scale < 1.0: wider gaps (easier)
  - r_scale = 1.0: default narrow task
  - r_scale > 1.0: narrower gaps (harder)
- Methods: PD_K10, R0_pd_s5_K10, R1_s5_K10
- N_MC = 40, seed = 7777, K = 10
- Paired McNemar test between each method and PD at each r_scale

Key hypothesis:
- At wider gaps (r_scale ≤ 0.9), even narrow-σ PD should succeed →
  three-pillar advantage diminishes
- At default/narrower gaps (r_scale ≥ 1.0), three-pillar advantage is clear
- The transition point shows where wide-σ becomes necessary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core import QuadrotorDynamics, PTRMNMPCPredictor, TRMNMPC
from experiments.ptrm_advantage_quick import (
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
    load_trm_model, to_jsonable,
)


# ──────────────────────────────────────────────
# Task factory with parametric gap width
# ──────────────────────────────────────────────
def task_narrow_scaled(seed: int, r_scale: float) -> dict[str, Any]:
    """task_narrow with all obstacle radii scaled by r_scale."""
    base = TASK_FACTORIES["narrow"](seed)
    scaled_obs = []
    for o in base["obstacles"]:
        scaled_obs.append({
            "p": o["p"].copy(),
            "r": o["r"] * r_scale,
        })
    return {
        "name": f"narrow_r{r_scale:.2f}",
        "obstacles": scaled_obs,
        "x_sp": base["x_sp"],
        "init_sampler": base["init_sampler"],
        "config": {"family": "narrow", "seed": seed, "r_scale": r_scale},
    }


# ──────────────────────────────────────────────
# Controller adapters (same as other experiment scripts)
# ──────────────────────────────────────────────
def pd_first_step(env, x, x_sp, Kp=4.0, Kd=3.0):
    e_p = x_sp[0:3] - x[0:3]
    e_v = x_sp[3:6] - x[3:6]
    u = env.m * (Kp * e_p + Kd * e_v)
    return torch.clamp(u, env.u_min, env.u_max)


class ProbeAdapter:
    """Wraps PTRMNMPCPredictor to produce (u_safe, _) returns."""
    def __init__(self, predictor):
        self.predictor = predictor

    def reset(self):
        self.predictor.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe, _ = self.predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


# ──────────────────────────────────────────────
# Trial runner
# ──────────────────────────────────────────────
def run_trial(controller, env, x0, x_sp, n_steps, enable_cbf=True):
    controller.reset()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    lat_list = []
    iae = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        lat_list.append((time.perf_counter() - t0) * 1000.0)
        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3], dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    return {
        "TErr": terr, "IAE": iae,
        "success": (terr < 0.30) and (not collided),
        "collided": collided,
        "latency_ms_mean": float(np.mean(lat_list)),
    }


def aggregate(rows):
    n = len(rows)
    succ = sum(1 for r in rows if r["success"])
    terrs = np.array([r["TErr"] for r in rows])
    iaes = np.array([r["IAE"] for r in rows])
    return {
        "n": n, "success_count": succ, "success_rate": succ / n,
        "TErr_mean": float(terrs.mean()), "TErr_median": float(np.median(terrs)),
        "IAE_mean": float(iaes.mean()),
        "individual": rows,
    }


# ──────────────────────────────────────────────
# McNemar test
# ──────────────────────────────────────────────
def mcnemar_discordant(rows_a, rows_b):
    """Count discordant pairs (b, c) for McNemar test."""
    b = sum(1 for a, bb in zip(rows_a, rows_b) if a["success"] and not bb["success"])
    c = sum(1 for a, bb in zip(rows_a, rows_b) if not a["success"] and bb["success"])
    return b, c


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value using binomial CDF."""
    if b + c == 0:
        return 1.0
    n = b + c
    # P(X <= min(b,c)) where X ~ Binom(n, 0.5)
    k = min(b, c)
    # Two-sided: 2 * P(X <= k) but capped at 1.0
    p_val = 0.0
    for i in range(k + 1):
        # Binomial coefficient * 0.5^n
        from math import comb
        p_val += comb(n, i) * (0.5 ** n)
    p_val = min(2.0 * p_val, 1.0)
    return p_val


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-scales", type=str, default="0.80,0.90,1.00,1.10,1.15")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"))
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    r_scales = [float(x) for x in args.r_scales.split(",") if x.strip()]

    # Load TRM model
    model = load_trm_model(Path(args.model), "cpu")
    model.eval()

    out = {
        "meta": {
            "experiment": "gap_width_sensitivity",
            "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k,
            "r_scales": r_scales,
            "mass": args.mass, "drag": args.drag,
            "rng_isolation": "per_method_per_trial",
        },
        "by_r_scale": {},
        "paired_tests": [],
    }

    method_names = ["PD", "R0_pd_s5", "R1_s5"]

    for r_scale in r_scales:
        task = task_narrow_scaled(args.seed, r_scale)
        x_sp = task["x_sp"]
        obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                     for o in task["obstacles"]]
        env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
        inits = sample_initial_states(task, args.n_mc, args.seed)

        print(f"\n{'='*60}")
        print(f"r_scale = {r_scale:.2f}  (radii × {r_scale:.2f})")
        print(f"  obstacle radii: {[o['r'] for o in obstacles]}")
        print(f"{'='*60}")

        cell = {}
        for method_idx, method_name in enumerate(method_names):
            print(f"  [{method_name}]", end=" ", flush=True)
            rows = []
            for i, x0 in enumerate(inits):
                set_method_seed(args.seed, f"{method_name}_r{r_scale:.2f}", method_idx, i)
                if method_name == "PD":
                    ctrl = ProbeAdapter(PTRMNMPCPredictor(
                        model, env, K=args.k, D=16, sigma=0.25,
                        candidate_mode="pd", ranking_mode="rollout_all",
                        alpha_blend=1.0, pd_sigma=2.0,
                    ))
                elif method_name == "R0_pd_s5":
                    ctrl = ProbeAdapter(PTRMNMPCPredictor(
                        model, env, K=args.k, D=16, sigma=0.25,
                        candidate_mode="pd", ranking_mode="rollout_all",
                        alpha_blend=1.0, pd_sigma=5.0,
                    ))
                elif method_name == "R1_s5":
                    ctrl = ProbeAdapter(PTRMNMPCPredictor(
                        model, env, K=args.k, D=16, sigma=0.25,
                        candidate_mode="trm_pd", ranking_mode="rollout_all",
                        alpha_blend=0.95, pd_sigma=5.0,
                    ))
                r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
                rows.append(r)
            agg = aggregate(rows)
            cell[method_name] = agg
            print(f"succ={agg['success_count']}/{agg['n']} "
                  f"TErr={agg['TErr_mean']:.3f} IAE={agg['IAE_mean']:.3f}")

        # Paired McNemar tests
        for a_name, b_name in [("R0_pd_s5", "PD"), ("R1_s5", "PD"),
                                ("R0_pd_s5", "R1_s5")]:
            bb, cc = mcnemar_discordant(cell[a_name]["individual"],
                                        cell[b_name]["individual"])
            p = mcnemar_p(bb, cc)
            sig = "***" if p < 0.001 else "**" if p < 0.01 \
                else "*" if p < 0.05 else "n.s."
            out["paired_tests"].append({
                "r_scale": r_scale, "method_a": a_name, "method_b": b_name,
                "succ_a": cell[a_name]["success_count"],
                "succ_b": cell[b_name]["success_count"],
                "b": bb, "c": cc, "p_exact_two_sided": p, "sig": sig,
            })
            print(f"  McNemar {a_name} vs {b_name}: "
                  f"b={bb} c={cc} p={p:.4f} {sig}")

        out["by_r_scale"][f"r{r_scale:.2f}"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "individual"}
            for k, v in cell.items()
        }

    # Summary table
    print(f"\n{'='*70}")
    print("GAP-WIDTH SENSITIVITY SUMMARY")
    print(f"{'r_scale':>8s}  {'PD':>10s}  {'R0_pd_s5':>10s}  {'R1_s5':>10s}  "
          f"{'R0-PD p':>10s}  {'R1-PD p':>10s}")
    print("-" * 70)
    for r_scale in r_scales:
        key = f"r{r_scale:.2f}"
        cell = out["by_r_scale"][key]
        pd_s = f"{cell['PD']['success_count']}/{cell['PD']['n']}"
        r0_s = f"{cell['R0_pd_s5']['success_count']}/{cell['R0_pd_s5']['n']}"
        r1_s = f"{cell['R1_s5']['success_count']}/{cell['R1_s5']['n']}"
        r0_p = next((t for t in out["paired_tests"]
                      if t["r_scale"] == r_scale and t["method_a"] == "R0_pd_s5"
                      and t["method_b"] == "PD"), None)
        r1_p = next((t for t in out["paired_tests"]
                      if t["r_scale"] == r_scale and t["method_a"] == "R1_s5"
                      and t["method_b"] == "PD"), None)
        r0_pv = f"{r0_p['p_exact_two_sided']:.4f}" if r0_p else "—"
        r1_pv = f"{r1_p['p_exact_two_sided']:.4f}" if r1_p else "—"
        print(f"{r_scale:>8.2f}  {pd_s:>10s}  {r0_s:>10s}  {r1_s:>10s}  "
              f"{r0_pv:>10s}  {r1_pv:>10s}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")
    print(f"\n[gap-width] wrote {out_path}")


if __name__ == "__main__":
    main()
