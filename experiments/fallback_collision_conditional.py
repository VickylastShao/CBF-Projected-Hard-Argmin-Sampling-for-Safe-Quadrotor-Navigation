#!/usr/bin/env python3
"""
Fallback→Collision conditional probability analysis.

Purpose: Compute P(collision | CBF fallback triggered) to address
reviewer concern about Proposition 1's QP feasibility assumption.

From the CBF fallback analysis experiment, we know:
- PD_K10: 28/40 success, fallback rate ~16.5%
- R0_pd_s5_K10: 40/40 success, fallback rate ~6.0%
- R1_s5_K10: 40/40 success, fallback rate ~6.0%

But we need per-trial individual data to compute P(collision | fallback).
This script re-runs the CBF analysis with individual data preserved,
and also computes the conditional probability.
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

from quadrotor_core import QuadrotorDynamics, PTRMNMPCPredictor
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


def run_trial_fallback_collision(controller, env, x0, x_sp, n_steps,
                                  enable_cbf=True):
    """Run one trial tracking per-step fallback and collision events."""
    controller.reset()
    env.reset_cbf_stats()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    iae = 0.0

    # Per-step tracking
    fallback_steps = []   # step indices where fallback was triggered
    collision_step = None  # first collision step (if any)
    prev_fallback = 0

    for step in range(n_steps):
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)

        # Check CBF stats delta
        stats = env.get_cbf_stats()
        curr_fallback = stats["cbf_fallback"]
        if curr_fallback > prev_fallback:
            fallback_steps.append(step)
        prev_fallback = curr_fallback

        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3],
                                dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt

        # Check collision after step
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                if not collided:
                    collision_step = step
                collided = True

    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    success = (terr < 0.30) and (not collided)

    final_stats = env.get_cbf_stats()

    return {
        "TErr": terr,
        "IAE": iae,
        "success": success,
        "collided": collided,
        "collision_step": collision_step,
        "cbf_active": final_stats["cbf_active"],
        "cbf_fallback": final_stats["cbf_fallback"],
        "cbf_active_rate": final_stats["cbf_active_rate"],
        "cbf_fallback_rate": final_stats["cbf_fallback_rate"],
        "n_fallback_steps": len(fallback_steps),
        "fallback_step_indices": fallback_steps,
    }


def main():
    parser = argparse.ArgumentParser()
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
    model = load_trm_model(Path(args.model), "cpu")
    model.eval()

    task = TASK_FACTORIES["narrow"](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    methods = [
        ("PD_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=10, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=2.0,
        ))),
        ("R0_pd_s5_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=10, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=5.0,
        ))),
        ("R1_s5_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=10, D=16, sigma=0.25,
            candidate_mode="trm_pd", ranking_mode="rollout_all",
            alpha_blend=0.95, pd_sigma=5.0,
        ))),
    ]

    out: dict[str, Any] = {
        "meta": {
            "experiment": "fallback_collision_conditional",
            "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "mass": args.mass, "drag": args.drag,
        },
        "by_method": {},
    }

    for method_idx, (name, factory) in enumerate(methods):
        print(f"\n[{name}]", end=" ", flush=True)
        rows = []
        for i, x0 in enumerate(inits):
            set_method_seed(args.seed, name, method_idx, i)
            ctrl = factory()
            r = run_trial_fallback_collision(ctrl, env, x0, x_sp, args.n_steps)
            rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)

        n = len(rows)
        succ = sum(1 for r in rows if r["success"])
        collided = sum(1 for r in rows if r["collided"])
        had_fallback = sum(1 for r in rows if r["n_fallback_steps"] > 0)
        fallback_and_collided = sum(1 for r in rows
                                     if r["n_fallback_steps"] > 0 and r["collided"])
        fallback_and_success = sum(1 for r in rows
                                    if r["n_fallback_steps"] > 0 and r["success"])
        no_fallback_and_collided = sum(1 for r in rows
                                        if r["n_fallback_steps"] == 0 and r["collided"])

        # Conditional probabilities
        p_collision_given_fallback = (fallback_and_collided / had_fallback
                                       if had_fallback > 0 else 0.0)
        p_collision_given_no_fallback = (no_fallback_and_collided / (n - had_fallback)
                                          if (n - had_fallback) > 0 else 0.0)

        agg = {
            "n": n,
            "success_count": succ,
            "collided_count": collided,
            "had_fallback_count": had_fallback,
            "fallback_and_collided": fallback_and_collided,
            "fallback_and_success": fallback_and_success,
            "no_fallback_and_collided": no_fallback_and_collided,
            "P_collision_given_fallback": p_collision_given_fallback,
            "P_collision_given_no_fallback": p_collision_given_no_fallback,
            "TErr_mean": float(np.mean([r["TErr"] for r in rows])),
            "fallback_rate_mean": float(np.mean([r["cbf_fallback_rate"] for r in rows])),
            "individual": rows,
        }
        out["by_method"][name] = agg

        print(f"\n  succ={succ}/{n}  collided={collided}  had_fallback={had_fallback}")
        print(f"  P(collision|fallback) = {fallback_and_collided}/{had_fallback} "
              f"= {p_collision_given_fallback:.3f}")
        print(f"  P(collision|no_fallback) = {no_fallback_and_collided}/{n - had_fallback} "
              f"= {p_collision_given_no_fallback:.3f}")

    # Summary
    print(f"\n{'='*80}")
    print("FALLBACK→COLLISION CONDITIONAL PROBABILITY SUMMARY")
    print(f"{'Method':<18s} {'Succ':>8s} {'Coll':>6s} {'FB':>6s} "
          f"{'FB∧Coll':>8s} {'P(C|FB)':>8s} {'P(C|¬FB)':>9s}")
    print("-" * 80)
    for name, _ in methods:
        v = out["by_method"][name]
        print(f"{name:<18s} {v['success_count']:>3d}/{v['n']:<3d} "
              f"{v['collided_count']:>4d}   {v['had_fallback_count']:>4d}   "
              f"{v['fallback_and_collided']:>4d}     "
              f"{v['P_collision_given_fallback']:>6.3f}   "
              f"{v['P_collision_given_no_fallback']:>6.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")
    print(f"\n[fallback-collision] wrote {out_path}")


if __name__ == "__main__":
    main()
