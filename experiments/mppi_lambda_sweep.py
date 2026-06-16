#!/usr/bin/env python3
"""MPPI temperature (λ) sweep — review P0 follow-up.

Reviewers R1/R3/R4 flagged that the manuscript's MPPI baseline runs at the
default λ=0.1 without tuning, weakening the "TSH-NMPC beats MPPI" headline.
This script sweeps λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0} on the narrow
benchmark at K=10 N=80 seed 7777 (the same cell used by probes_a) so the
result is paired with the existing R1_s5 / PD numbers.

Output: experiments/results_v6/mppi_lambda_sweep.json — includes per-trial
success/TErr/IAE so paired McNemar against R1_s5 / PD can be computed by
experiments/compute_*.py downstream.

Usage:
    python experiments/mppi_lambda_sweep.py \
        --task narrow --k 10 --n-mc 80 --seed 7777 \
        --lambdas 0.01,0.05,0.1,0.5,1.0,5.0 \
        --output experiments/results_v6/mppi_lambda_sweep.json
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

from quadrotor_core import QuadrotorDynamics                                # noqa: E402
from experiments.baselines.mppi_controller import MPPIController            # noqa: E402
from experiments.ptrm_advantage_quick import (                              # noqa: E402
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
)


class MPPIProbeAdapter:
    """Wraps MPPIController so it matches probes_a's (u_safe, _) return."""

    def __init__(self, env: QuadrotorDynamics, K: int, lam: float,
                 sigma: float = 2.0, rollout_steps: int = 20):
        self.inner = MPPIController(env, K=K, sigma=sigma, lam=lam,
                                    rollout_steps=rollout_steps)

    def reset(self) -> None:
        self.inner.reset()

    def predict_action(self, x: torch.Tensor, x_sp: torch.Tensor,
                       enable_cbf: bool = True):
        u_safe = self.inner.predict_action(x, x_sp, enable_cbf=enable_cbf)
        # second slot kept for parity with probes_a.run_trial; unused
        return u_safe, None


def run_trial(controller, env: QuadrotorDynamics, x0: torch.Tensor,
              x_sp: torch.Tensor, n_steps: int, enable_cbf: bool = True
              ) -> dict[str, Any]:
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
        "latency_ms_mean": float(np.mean(lat_list)),
        "latency_ms_median": float(np.median(lat_list)),
        "latency_ms_p95": float(np.percentile(lat_list, 95)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    succ = sum(1 for r in rows if r["success"])
    terrs = np.array([r["TErr"] for r in rows])
    iaes = np.array([r["IAE"] for r in rows])
    lats = np.array([r["latency_ms_mean"] for r in rows])
    return {
        "n": n, "success_count": succ, "success_rate": succ / n,
        "TErr_mean": float(terrs.mean()), "TErr_median": float(np.median(terrs)),
        "IAE_mean": float(iaes.mean()),
        "latency_ms_mean": float(lats.mean()),
        "individual": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lambdas", type=str,
                        default="0.01,0.05,0.1,0.5,1.0,5.0")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="MPPI action-space noise σ (default 2.0; matches v6)")
    parser.add_argument("--n-mc", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-cbf", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    out: dict[str, Any] = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k, "sigma": args.sigma,
            "lambdas": lambdas, "mass": args.mass, "drag": args.drag,
            "enable_cbf": (not args.no_cbf),
            "rng_isolation": "per_method_per_trial",
        },
        "by_method": {},
    }

    for method_idx, lam in enumerate(lambdas):
        key = f"MPPI_lam{lam}_K{args.k}"
        print(f"[mppi-sweep] {key}")
        rows = []
        for i, x0 in enumerate(inits):
            set_method_seed(args.seed, key, method_idx, i)
            ctrl = MPPIProbeAdapter(env, K=args.k, lam=lam, sigma=args.sigma)
            r = run_trial(ctrl, env, x0, x_sp, args.n_steps,
                          enable_cbf=(not args.no_cbf))
            rows.append(r)
        agg = aggregate(rows)
        print(f"  succ={agg['success_count']}/{agg['n']}  "
              f"TErr_mean={agg['TErr_mean']:.3f}  "
              f"TErr_median={agg['TErr_median']:.3f}  "
              f"IAE={agg['IAE_mean']:.3f}  "
              f"lat_med={agg['latency_ms_mean']:.2f}ms")
        out["by_method"][key] = agg

    # Quick ranking summary
    ranking = sorted(
        out["by_method"].items(),
        key=lambda kv: (-kv[1]["success_count"], kv[1]["TErr_mean"]),
    )
    out["ranking"] = [
        {"method": k, "success_count": v["success_count"],
         "TErr_mean": v["TErr_mean"], "IAE_mean": v["IAE_mean"]}
        for k, v in ranking
    ]
    print("\n[mppi-sweep] ranking (by success then TErr):")
    for r in out["ranking"]:
        print(f"  {r['method']:30s} {r['success_count']:3d}/{args.n_mc} "
              f"TErr={r['TErr_mean']:.3f} IAE={r['IAE_mean']:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[mppi-sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
