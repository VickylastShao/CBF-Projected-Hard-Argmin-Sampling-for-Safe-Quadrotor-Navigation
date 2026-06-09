#!/usr/bin/env python3
"""Dynamic obstacle benchmark — true constant-velocity moving obstacle.

Reviewer P1: original `narrow` cells use static obstacles; reviewers asked
for a true-dynamic environment with constant-velocity drift (no teleport).
We update the obstacle position by `p += v_obs * dt` BEFORE each control
step. The controller observes the obstacle at its CURRENT position (no
predictive knowledge of its trajectory) — this is the standard real-time
moving-obstacle setting where the controller must react online.

Cells:
  - narrow geometry with the central obstacle drifting at v = (0, vy, 0) m/s
    for vy in {0.0, 0.2, 0.5, 1.0}  (vy=0 reproduces the static baseline)
  - PD K=10, R1_s5 K=10, R0_pd_s5 K=10, CasADi-IPOPT H=20
  - N_MC = 40, seed 7777, n_steps = 150

Output: experiments/results_v6/dynamic_obstacle_narrow_n40_s7777.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core import QuadrotorDynamics                                # noqa: E402
from experiments.probes_a_zero_train import (                                # noqa: E402
    PROBE_FACTORIES, make_baseline,
)
from experiments.ptrm_advantage_quick import (                              # noqa: E402
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
)
from experiments.baselines.casadi_nmpc_controller import (                  # noqa: E402
    CasADiNMPCController,
)


class CasADiAdapter:
    def __init__(self, env: QuadrotorDynamics, horizon: int = 20):
        self.inner = CasADiNMPCController(env, horizon=horizon)

    def reset(self) -> None:
        self.inner.reset()

    def predict_action(self, x, x_sp, enable_cbf: bool = True):
        u_safe = self.inner.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2.0 / (2.0 ** n), 1.0)


def disc(a, b):
    bb = sum(1 for ra, rb in zip(a, b) if ra["success"] and not rb["success"])
    cc = sum(1 for ra, rb in zip(a, b) if rb["success"] and not ra["success"])
    return bb, cc


def run_dynamic_trial(controller, env: QuadrotorDynamics,
                      obstacle_velocities: list[np.ndarray],
                      original_obstacle_positions: list[np.ndarray],
                      x0: torch.Tensor, x_sp: torch.Tensor,
                      n_steps: int, enable_cbf: bool = True) -> dict[str, Any]:
    """Run one trial. Obstacle positions update by p += v_obs * dt each step.

    The controller's CBF and rollout use env.obstacles at the CURRENT
    (pre-step) position; no predictive obstacle motion is given.
    """
    controller.reset()
    # Reset obstacles to their nominal positions (each trial starts fresh).
    for j, p_orig in enumerate(original_obstacle_positions):
        env.obstacles[j]["p"] = p_orig.copy()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    lat_list = []
    iae = 0.0
    min_dist = np.inf
    for _ in range(n_steps):
        # Controller observes obstacles at their current position.
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        lat_list.append((time.perf_counter() - t0) * 1000.0)
        # Apply control; the obstacle pose at this control instant is the
        # one the controller saw.
        u_first = u_safe[:3] if u_safe.numel() == 3 else torch.tensor(
            u_safe.detach().cpu().numpy()[:3], dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        # Update obstacles for next step (constant-velocity drift).
        for j, v_obs in enumerate(obstacle_velocities):
            env.obstacles[j]["p"] = env.obstacles[j]["p"] + v_obs * env.dt
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt
        # Collision check vs CURRENT (post-update) obstacle position.
        for o in env.obstacles:
            d = float(np.linalg.norm(x.numpy()[:3] - o["p"]))
            min_dist = min(min_dist, d - o["r"])
            if d < o["r"]:
                collided = True
    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    return {
        "TErr": terr if not collided else 10.0,
        "IAE": iae if not collided else 20.0,
        "success": (terr < 0.30) and (not collided),
        "collided": collided,
        "min_dist": float(min_dist),
        "latency_ms_mean": float(np.mean(lat_list)),
        "latency_ms_p95": float(np.percentile(lat_list, 95)),
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    succ = sum(1 for r in rows if r["success"])
    coll = sum(1 for r in rows if r["collided"])
    terrs = np.array([r["TErr"] for r in rows])
    iaes = np.array([r["IAE"] for r in rows])
    return {
        "n": n, "success_count": succ, "collisions": coll,
        "success_rate": succ / n,
        "TErr_mean": float(terrs.mean()),
        "IAE_mean": float(iaes.mean()),
        "min_dist_mean": float(np.mean([r["min_dist"] for r in rows])),
        "latency_ms_mean": float(np.mean([r["latency_ms_mean"] for r in rows])),
        "individual": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--velocities", type=str, default="0.0,0.2,0.5,1.0",
                        help="obstacle y-velocity values to sweep (m/s)")
    parser.add_argument("--moving-obstacle-idx", type=int, default=1,
                        help="index of the obstacle to set in motion")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--model", type=str,
                        default="experiments/results_v6/cl_trm_model.pt")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-cbf", action="store_true")
    parser.add_argument("--casadi-horizon", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    velocities = [float(v) for v in args.velocities.split(",") if v.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles_blueprint = [
        {"p": o["p"].astype(np.float32), "r": float(o["r"])}
        for o in task["obstacles"]
    ]
    inits_all = sample_initial_states(task, args.n_mc, args.seed)

    # Load CL-TRM model for R1_s5
    from quadrotor_core import TRMNMPC
    model = TRMNMPC()
    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    else:
        model.load_state_dict(sd)
    model.eval()

    out: dict = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k,
            "moving_obstacle_idx": args.moving_obstacle_idx,
            "velocities_y_mps": velocities,
            "mass": args.mass, "drag": args.drag,
            "enable_cbf": (not args.no_cbf),
            "casadi_horizon": args.casadi_horizon,
            "note": ("constant-velocity obstacle drift; controller observes "
                     "current position with no predictive knowledge"),
            "rng_isolation": "per_method_per_trial",
        },
        "by_velocity": {},
        "paired_tests": [],
    }

    for vy in velocities:
        v_obs_list = [np.zeros(3, dtype=np.float32)
                      for _ in range(len(obstacles_blueprint))]
        v_obs_list[args.moving_obstacle_idx] = np.array(
            [0.0, vy, 0.0], dtype=np.float32)
        orig_positions = [o["p"].copy() for o in obstacles_blueprint]

        print(f"\n=== v_y = {vy:.2f} m/s on obstacle {args.moving_obstacle_idx} ===")
        cell_results = {}
        for method_idx, method_name in enumerate(["PD", "R0_pd_s5", "R1_s5", "CasADi_H20"]):
            print(f"  [{method_name}]", end=" ", flush=True)
            rows = []
            for i, x0 in enumerate(inits_all):
                set_method_seed(args.seed, f"{method_name}_dyn_{vy}", method_idx, i)
                # Fresh env per trial so obstacle state resets.
                env = QuadrotorDynamics(
                    m=args.mass, b_drag=args.drag,
                    obstacles=deepcopy(obstacles_blueprint))
                if method_name == "PD":
                    ctrl = make_baseline("PD", model, env, args.k)
                elif method_name in PROBE_FACTORIES:
                    ctrl = PROBE_FACTORIES[method_name](model, env, args.k)
                elif method_name == "CasADi_H20":
                    ctrl = CasADiAdapter(env, horizon=args.casadi_horizon)
                else:
                    raise ValueError(method_name)
                r = run_dynamic_trial(
                    ctrl, env, v_obs_list, orig_positions, x0, x_sp,
                    args.n_steps, enable_cbf=(not args.no_cbf))
                rows.append(r)
            agg = aggregate(rows)
            cell_results[method_name] = agg
            print(f"succ={agg['success_count']}/{agg['n']} coll={agg['collisions']} "
                  f"TErr={agg['TErr_mean']:.3f} min_d={agg['min_dist_mean']:.3f}")

        # Paired McNemar within this velocity cell.
        pairs = [
            ("R1_s5", "PD"), ("R1_s5", "CasADi_H20"),
            ("R0_pd_s5", "PD"), ("R0_pd_s5", "CasADi_H20"),
            ("CasADi_H20", "PD"),
        ]
        cell_tests = []
        for a, b in pairs:
            ra = cell_results[a]["individual"]
            rb = cell_results[b]["individual"]
            bb, cc = disc(ra, rb)
            p = mcnemar_p(bb, cc)
            cell_tests.append({
                "v_y": vy, "method_a": a, "method_b": b,
                "succ_a": cell_results[a]["success_count"],
                "succ_b": cell_results[b]["success_count"],
                "b": bb, "c": cc, "p_exact_two_sided": p,
            })
            sig = "***" if p < 0.001 else "**" if p < 0.01 \
                else "*" if p < 0.05 else "n.s."
            print(f"    {a:12s} vs {b:12s}: b={bb}, c={cc}, p={p:.4e} {sig}")
        out["by_velocity"][f"vy_{vy:.2f}"] = cell_results
        out["paired_tests"].extend(cell_tests)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
