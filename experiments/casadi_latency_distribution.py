#!/usr/bin/env python3
"""
CasADi NMPC per-step latency distribution experiment.

Purpose: Collect per-step solve latency for CasADi NMPC and TSH-NMPC
to compare their latency distributions (mean, median, p95, p99, max).

Key question: Does CasADi's per-step latency have a long tail exceeding
the 20ms control deadline?
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
from experiments.baselines.casadi_nmpc_controller import CasADiNMPCController
from experiments.ptrm_advantage_quick import (
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
    load_trm_model, to_jsonable,
)


class CasADiAdapter:
    def __init__(self, env: QuadrotorDynamics, horizon: int = 10):
        self.inner = CasADiNMPCController(env, horizon=horizon)

    def reset(self) -> None:
        self.inner.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe = self.inner.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


class ProbeAdapter:
    def __init__(self, predictor):
        self.predictor = predictor

    def reset(self):
        self.predictor.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe, _ = self.predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


def run_trial_per_step_latency(controller, env, x0, x_sp, n_steps,
                                enable_cbf=True):
    """Run one trial, record per-step latency."""
    controller.reset()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    step_lats = []  # per-step latency in ms
    iae = 0.0

    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        lat = (time.perf_counter() - t0) * 1000.0
        step_lats.append(lat)

        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3],
                                dtype=torch.float32)
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
        "step_lats_ms": step_lats,  # raw per-step data
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="narrow")
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

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    methods = [
        ("CasADi_H10", lambda: CasADiAdapter(env, horizon=10)),
        ("CasADi_H20", lambda: CasADiAdapter(env, horizon=20)),
        ("R1_s5_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=10, D=16, sigma=0.25,
            candidate_mode="trm_pd", ranking_mode="rollout_all",
            alpha_blend=0.95, pd_sigma=5.0,
        ))),
        ("PD_K10", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=10, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=2.0,
        ))),
    ]

    out: dict[str, Any] = {
        "meta": {
            "experiment": "casadi_latency_distribution",
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "mass": args.mass, "drag": args.drag,
            "deadline_ms": 20.0,
        },
        "by_method": {},
    }

    for method_idx, (name, factory) in enumerate(methods):
        print(f"\n[{name}]", end=" ", flush=True)
        all_step_lats = []  # all steps across all trials
        trial_stats = []

        for i, x0 in enumerate(inits):
            set_method_seed(args.seed, name, method_idx, i)
            ctrl = factory()
            r = run_trial_per_step_latency(ctrl, env, x0, x_sp, args.n_steps)
            lats = np.array(r["step_lats_ms"])
            all_step_lats.extend(lats.tolist())
            trial_stats.append({
                "success": r["success"],
                "TErr": r["TErr"],
                "lat_mean": float(lats.mean()),
                "lat_median": float(np.median(lats)),
                "lat_p95": float(np.percentile(lats, 95)),
                "lat_p99": float(np.percentile(lats, 99)),
                "lat_max": float(lats.max()),
            })
            if (i + 1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)

        all_lats = np.array(all_step_lats)
        n_total = len(all_lats)
        n_over_20 = int(np.sum(all_lats > 20.0))
        n_over_15 = int(np.sum(all_lats > 15.0))
        n_over_10 = int(np.sum(all_lats > 10.0))

        agg = {
            "n_trials": args.n_mc,
            "n_steps_per_trial": args.n_steps,
            "n_total_steps": n_total,
            "success_count": sum(1 for t in trial_stats if t["success"]),
            "TErr_mean": float(np.mean([t["TErr"] for t in trial_stats])),
            "lat_mean_ms": float(all_lats.mean()),
            "lat_median_ms": float(np.median(all_lats)),
            "lat_p95_ms": float(np.percentile(all_lats, 95)),
            "lat_p99_ms": float(np.percentile(all_lats, 99)),
            "lat_max_ms": float(all_lats.max()),
            "lat_std_ms": float(all_lats.std()),
            "n_over_10ms": n_over_10,
            "n_over_15ms": n_over_15,
            "n_over_20ms": n_over_20,
            "pct_over_10ms": float(100.0 * n_over_10 / n_total),
            "pct_over_15ms": float(100.0 * n_over_15 / n_total),
            "pct_over_20ms": float(100.0 * n_over_20 / n_total),
            # Histogram bins (ms)
            "hist_0_5": int(np.sum(all_lats <= 5)),
            "hist_5_10": int(np.sum((all_lats > 5) & (all_lats <= 10))),
            "hist_10_15": int(np.sum((all_lats > 10) & (all_lats <= 15))),
            "hist_15_20": int(np.sum((all_lats > 15) & (all_lats <= 20))),
            "hist_20_30": int(np.sum((all_lats > 20) & (all_lats <= 30))),
            "hist_30_50": int(np.sum((all_lats > 30) & (all_lats <= 50))),
            "hist_50_plus": int(np.sum(all_lats > 50)),
        }
        out["by_method"][name] = agg
        print(f"\n  succ={agg['success_count']}/{args.n_mc} "
              f"TErr={agg['TErr_mean']:.4f}")
        print(f"  latency: mean={agg['lat_mean_ms']:.2f} "
              f"median={agg['lat_median_ms']:.2f} "
              f"p95={agg['lat_p95_ms']:.2f} "
              f"p99={agg['lat_p99_ms']:.2f} "
              f"max={agg['lat_max_ms']:.2f}")
        print(f"  >10ms: {agg['pct_over_10ms']:.2f}%  "
              f">15ms: {agg['pct_over_15ms']:.2f}%  "
              f">20ms: {agg['pct_over_20ms']:.2f}%")

    # Summary comparison
    print(f"\n{'='*80}")
    print("LATENCY DISTRIBUTION SUMMARY")
    print(f"{'Method':<15s} {'Mean':>7s} {'Med':>7s} {'P95':>7s} "
          f"{'P99':>7s} {'Max':>7s} {'%>20ms':>7s} {'Succ':>8s}")
    print("-" * 80)
    for name, _ in methods:
        v = out["by_method"][name]
        print(f"{name:<15s} {v['lat_mean_ms']:>6.2f}  {v['lat_median_ms']:>6.2f}  "
              f"{v['lat_p95_ms']:>6.2f}  {v['lat_p99_ms']:>6.2f}  "
              f"{v['lat_max_ms']:>6.2f}  {v['pct_over_20ms']:>6.2f}% "
              f"{v['success_count']:>3d}/{v['n_trials']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")
    print(f"\n[latency-dist] wrote {out_path}")


if __name__ == "__main__":
    main()
