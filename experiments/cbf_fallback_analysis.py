#!/usr/bin/env python3
"""
CBF fallback streak analysis.

Purpose: Extract actual CBF projection / fallback frequency data to
substantiate P3 (CBF as safety net) in the three-pillar framework.

Design:
- Run PD_K10, R0_pd_s5_K10, R1_s5_K10 on narrow task with CBF stats tracking
- Per-trial CBF stats: active_rate, fallback_rate, max_consecutive_fallback_streak
- Aggregate across N_MC=40 trials
- Compare CBF activation across methods (wide-σ methods should trigger CBF less)

Key hypothesis:
- PD_σ2 (narrow) should have higher CBF activation (candidates near obstacles)
- R0_pd_s5 (wide) and R1_s5 should have lower CBF activation (wide σ finds
  better candidates that don't need CBF correction)
- Fallback streaks should be rare (CBF projection typically succeeds)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


class ProbeAdapter:
    def __init__(self, predictor):
        self.predictor = predictor

    def reset(self):
        self.predictor.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe, _ = self.predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


def run_trial_with_cbf_stats(controller, env, x0, x_sp, n_steps, enable_cbf=True):
    """Run one trial and collect per-step CBF statistics."""
    controller.reset()
    env.reset_cbf_stats()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    lat_list = []
    iae = 0.0

    # Per-step CBF tracking
    cbf_active_steps = []  # True/False per step
    cbf_fallback_steps = []
    min_obstacle_dists = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        lat_list.append((time.perf_counter() - t0) * 1000.0)

        # Record CBF state after prediction
        stats = env.get_cbf_stats()
        # We track deltas: the predictor may call apply_cbf_projection multiple
        # times per step (e.g., in rollout). We track cumulative counts.
        cbf_active_steps.append(stats["cbf_active"])
        cbf_fallback_steps.append(stats["cbf_fallback"])

        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3], dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt

        # Track minimum obstacle distance
        pos = x[:3].detach().cpu().numpy()
        d_min = float(min(
            np.linalg.norm(pos - o["p"]) - o["r"]
            for o in env.obstacles
        ))
        min_obstacle_dists.append(d_min)

        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True

    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))

    # Compute CBF streaks from per-step deltas
    active_deltas = np.diff([0] + cbf_active_steps)
    fallback_deltas = np.diff([0] + cbf_fallback_steps)

    # Steps where CBF was newly activated
    active_steps = [i for i in range(n_steps) if active_deltas[i] > 0]
    fallback_steps = [i for i in range(n_steps) if fallback_deltas[i] > 0]

    # Max consecutive fallback streak
    max_fallback_streak = 0
    current_streak = 0
    for i in range(n_steps):
        if fallback_deltas[i] > 0:
            current_streak += 1
            max_fallback_streak = max(max_fallback_streak, current_streak)
        else:
            current_streak = 0

    final_stats = env.get_cbf_stats()

    return {
        "TErr": terr, "IAE": iae,
        "success": (terr < 0.30) and (not collided),
        "collided": collided,
        "latency_ms_mean": float(np.mean(lat_list)),
        "cbf_calls": final_stats["cbf_calls"],
        "cbf_active": final_stats["cbf_active"],
        "cbf_fallback": final_stats["cbf_fallback"],
        "cbf_active_rate": final_stats["cbf_active_rate"],
        "cbf_fallback_rate": final_stats["cbf_fallback_rate"],
        "cbf_active_steps": len(active_steps),
        "cbf_fallback_steps": len(fallback_steps),
        "max_fallback_streak": max_fallback_streak,
        "min_obstacle_dist": float(min(min_obstacle_dists)) if min_obstacle_dists else 0.0,
    }


def aggregate(rows):
    n = len(rows)
    succ = sum(1 for r in rows if r["success"])
    terrs = np.array([r["TErr"] for r in rows])
    iaes = np.array([r["IAE"] for r in rows])
    active_rates = np.array([r["cbf_active_rate"] for r in rows])
    fallback_rates = np.array([r["cbf_fallback_rate"] for r in rows])
    max_streaks = np.array([r["max_fallback_streak"] for r in rows])
    active_steps_list = np.array([r["cbf_active_steps"] for r in rows])
    fallback_steps_list = np.array([r["cbf_fallback_steps"] for r in rows])
    min_dists = np.array([r["min_obstacle_dist"] for r in rows])

    return {
        "n": n, "success_count": succ, "success_rate": succ / n,
        "TErr_mean": float(terrs.mean()), "TErr_median": float(np.median(terrs)),
        "IAE_mean": float(iaes.mean()),
        "cbf_active_rate_mean": float(active_rates.mean()),
        "cbf_active_rate_std": float(active_rates.std()),
        "cbf_fallback_rate_mean": float(fallback_rates.mean()),
        "cbf_fallback_rate_std": float(fallback_rates.std()),
        "cbf_active_steps_mean": float(active_steps_list.mean()),
        "cbf_fallback_steps_mean": float(fallback_steps_list.mean()),
        "max_fallback_streak_mean": float(max_streaks.mean()),
        "max_fallback_streak_max": int(max_streaks.max()),
        "min_obstacle_dist_mean": float(min_dists.mean()),
        "individual": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"))
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)

    model = load_trm_model(Path(args.model), "cpu")
    model.eval()

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    out = {
        "meta": {
            "experiment": "cbf_fallback_analysis",
            "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k,
            "mass": args.mass, "drag": args.drag,
            "rng_isolation": "per_method_per_trial",
        },
        "by_method": {},
    }

    methods = [
        ("PD_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=args.k, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=2.0,
        ))),
        ("R0_pd_s5_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=args.k, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=5.0,
        ))),
        ("R1_s5_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=args.k, D=16, sigma=0.25,
            candidate_mode="trm_pd", ranking_mode="rollout_all",
            alpha_blend=0.95, pd_sigma=5.0,
        ))),
    ]

    for method_idx, (name, factory) in enumerate(methods):
        print(f"\n[{name}]", end=" ", flush=True)
        rows = []
        for i, x0 in enumerate(inits):
            set_method_seed(args.seed, name, method_idx, i)
            ctrl = factory()
            r = run_trial_with_cbf_stats(ctrl, env, x0, x_sp, args.n_steps)
            rows.append(r)
        agg = aggregate(rows)
        out["by_method"][name] = agg
        print(f"succ={agg['success_count']}/{agg['n']} "
              f"TErr={agg['TErr_mean']:.3f} "
              f"CBF_active={agg['cbf_active_rate_mean']:.3f}±{agg['cbf_active_rate_std']:.3f} "
              f"CBF_fallback={agg['cbf_fallback_rate_mean']:.4f}±{agg['cbf_fallback_rate_std']:.4f} "
              f"max_streak={agg['max_fallback_streak_max']}")

    # Summary table
    print(f"\n{'='*90}")
    print("CBF FALLBACK ANALYSIS SUMMARY")
    print(f"{'Method':<18s} {'Succ':>8s} {'Active%':>10s} "
          f"{'Fallback%':>12s} {'MaxStreak':>10s} {'MinDist':>10s}")
    print("-" * 90)
    for name, _ in methods:
        v = out["by_method"][name]
        print(f"{name:<18s} {v['success_count']:>3d}/{v['n']:<3d} "
              f"{100*v['cbf_active_rate_mean']:>8.1f}%±{100*v['cbf_active_rate_std']:<4.1f}% "
              f"{100*v['cbf_fallback_rate_mean']:>9.2f}%±{100*v['cbf_fallback_rate_std']:<4.2f}% "
              f"{v['max_fallback_streak_max']:>6d}     "
              f"{v['min_obstacle_dist_mean']:>8.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove individual rows for cleaner JSON
    for k in out["by_method"]:
        out["by_method"][k] = {kk: vv for kk, vv in out["by_method"][k].items()
                               if kk != "individual"}
    out_path.write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")
    print(f"\n[cbf-analysis] wrote {out_path}")


if __name__ == "__main__":
    main()
