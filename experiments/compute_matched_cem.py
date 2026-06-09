#!/usr/bin/env python3
"""Compute-matched CEM vs probes_a methods — review P0 follow-up.

Reviewers asked: "PTRM K=50 single-iteration vs CEM K=50 × 3 iter (=150
effective rollouts) — that's a 3× compute mismatch favoring CEM. What
happens at matched compute?"

Two compute-matched cells on narrow N=80 seed 7777:
  (a) Effective-rollouts = 150
        CEM K=50 n_iter=3           (current baseline)
        CEM K=150 n_iter=1          (single-iter variant)
        R1_s5 K=150                 (proposed; two-source σ=5 + argmin)
        R0_pd_s5 K=150              (single-source σ=5 + argmin)
        PD K=150                    (single-source σ=2 + argmin)
  (b) Effective-rollouts = 30
        CEM K=10 n_iter=3
        R1_s5 K=30 / PD K=30 / R0_pd_s5 K=30

Output: experiments/results_v6/compute_matched_n80_s7777.json
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
from experiments.baselines.cem_controller import CEMController              # noqa: E402
from experiments.ptrm_advantage_quick import (                              # noqa: E402
    TASK_FACTORIES, load_trm_model, sample_initial_states, set_seed,
)
from experiments.probes_a_zero_train import (                               # noqa: E402
    PROBE_FACTORIES, make_baseline,
)

DEFAULT_MODEL_PATH = ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"
DEVICE = "cpu"


class CEMAdapter:
    """Wrap CEMController to (u_safe, _) interface used by run_trial."""

    def __init__(self, env: QuadrotorDynamics, K: int, n_iter: int,
                 sigma: float = 2.0):
        self.inner = CEMController(env, K=K, n_iter=n_iter, sigma=sigma)

    def reset(self) -> None:
        self.inner.reset()

    def predict_action(self, x: torch.Tensor, x_sp: torch.Tensor,
                       enable_cbf: bool = True):
        u_safe = self.inner.predict_action(x, x_sp, enable_cbf=enable_cbf)
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
    parser.add_argument("--n-mc", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-cbf", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    model_path = Path(args.model)
    print(f"[compute-matched] loading TRM from {model_path}")
    model = load_trm_model(model_path, DEVICE)
    model.eval()

    # Define the compute-matched cells.
    # Each tuple: (method_key, controller_factory, effective_rollouts)
    methods: list[tuple[str, Any, int]] = [
        # 150-rollout budget
        ("CEM_K50_iter3", lambda: CEMAdapter(env, K=50, n_iter=3, sigma=2.0), 150),
        ("CEM_K150_iter1", lambda: CEMAdapter(env, K=150, n_iter=1, sigma=2.0), 150),
        ("R1_s5_K150", lambda: PROBE_FACTORIES["R1_s5"](model, env, 150), 150),
        ("R0_pd_s5_K150", lambda: PROBE_FACTORIES["R0_pd_s5"](model, env, 150), 150),
        ("PD_K150", lambda: make_baseline("PD", model, env, 150), 150),
        # 30-rollout budget
        ("CEM_K10_iter3", lambda: CEMAdapter(env, K=10, n_iter=3, sigma=2.0), 30),
        ("R1_s5_K30", lambda: PROBE_FACTORIES["R1_s5"](model, env, 30), 30),
        ("R0_pd_s5_K30", lambda: PROBE_FACTORIES["R0_pd_s5"](model, env, 30), 30),
        ("PD_K30", lambda: make_baseline("PD", model, env, 30), 30),
    ]

    out: dict[str, Any] = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "mass": args.mass, "drag": args.drag,
            "enable_cbf": (not args.no_cbf),
            "model_path": str(model_path),
            "note": "effective_rollouts = K * n_iter for CEM; K for single-iter methods",
        },
        "by_method": {},
    }

    for key, factory, eff_rollouts in methods:
        print(f"[compute-matched] {key} (effective rollouts = {eff_rollouts})")
        rows = []
        for x0 in inits:
            ctrl = factory()
            r = run_trial(ctrl, env, x0, x_sp, args.n_steps,
                          enable_cbf=(not args.no_cbf))
            rows.append(r)
        agg = aggregate(rows)
        agg["effective_rollouts"] = eff_rollouts
        print(f"  succ={agg['success_count']}/{agg['n']}  "
              f"TErr_mean={agg['TErr_mean']:.3f}  "
              f"IAE={agg['IAE_mean']:.3f}  "
              f"lat={agg['latency_ms_mean']:.2f}ms")
        out["by_method"][key] = agg

    # Summary table
    print("\n[compute-matched] summary:")
    header = f"{'Method':25s} {'Eff-rollouts':>12s} {'Succ':>8s} {'TErr':>7s} {'IAE':>6s} {'lat(ms)':>9s}"
    print(header)
    print("-" * 78)
    for key, agg in out["by_method"].items():
        print(f"{key:25s} {agg['effective_rollouts']:>12d} "
              f"{agg['success_count']:>5d}/{agg['n']:<2d} "
              f"{agg['TErr_mean']:7.3f} {agg['IAE_mean']:6.2f} "
              f"{agg['latency_ms_mean']:9.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[compute-matched] wrote {out_path}")


if __name__ == "__main__":
    main()
