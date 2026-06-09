#!/usr/bin/env python3
"""
Adaptive-PD vs R1_s5 paired experiment.

Purpose: Address reviewer concern that PD with σ=2 is a straw-man baseline.
An adaptive PD that uses σ=5 near obstacles and σ=2 far from obstacles
should theoretically capture the "best of both worlds". Does it match R1_s5?

Design:
- Adaptive-σ PD: σ = σ_narrow (2.0) when d_min > d_thresh, else σ_wide (5.0)
  where d_min is the minimum distance from the current position to any obstacle.
- d_thresh ∈ {1.0, 1.5, 2.0} — sensitivity sweep
- Baselines: PD_σ2 (narrow), R0_pd_s5 (wide), R1_s5 (two-source)
- N_MC = 40, seed = 7777, K = 10

Key hypothesis:
- Adaptive PD should perform between PD_σ2 and R0_pd_s5
- If it matches R0_pd_s5, that confirms the benefit is purely from wider σ
  near obstacles — no "straw man" issue
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


def pd_first_step(env, x, x_sp, Kp=4.0, Kd=3.0):
    e_p = x_sp[0:3] - x[0:3]
    e_v = x_sp[3:6] - x[3:6]
    u = env.m * (Kp * e_p + Kd * e_v)
    return torch.clamp(u, env.u_min, env.u_max)


class AdaptivePDPredictor:
    """PD baseline with adaptive σ: wide σ near obstacles, narrow σ far away."""
    def __init__(self, env, K, sigma_near=5.0, sigma_far=2.0, d_thresh=1.5):
        self.env = env
        self.K = K
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.d_thresh = d_thresh

    def reset(self):
        pass

    def _min_obstacle_dist(self, x):
        pos = x[:3].detach().cpu().numpy()
        return float(min(
            np.linalg.norm(pos - o["p"]) - o["r"]
            for o in self.env.obstacles
        ))

    def _current_sigma(self, x):
        d_min = self._min_obstacle_dist(x)
        if d_min < self.d_thresh:
            return self.sigma_near
        return self.sigma_far

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_pd = pd_first_step(self.env, x, x_sp)
        sigma = self._current_sigma(x)
        if self.K == 1:
            cand = u_pd.unsqueeze(0)
        else:
            noise = torch.randn(self.K, 3) * sigma
            cand = u_pd.unsqueeze(0) + noise
        cand = torch.clamp(cand, self.env.u_min, self.env.u_max)

        # Rollout-based hard-argmin selection
        costs = self._rollout_costs(x, cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = cand[best]

        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x, u_nom)
        else:
            u_safe = torch.clamp(u_nom, self.env.u_min, self.env.u_max)
        return u_safe, None

    def _rollout_costs(self, x_init, u_first, x_sp, steps=20):
        M = u_first.shape[0]
        x = x_init.unsqueeze(0).repeat(M, 1)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(M, 1)
        cost = torch.zeros(M)
        q = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0]).unsqueeze(0)
        for s in range(steps):
            if s == 0:
                u = u_first
            else:
                e_p = x_sp6[:, 0:3] - x[:, 0:3]
                e_v = x_sp6[:, 3:6] - x[:, 3:6]
                u = self.env.m * (4.0 * e_p + 3.0 * e_v)
            u = torch.clamp(u, self.env.u_min, self.env.u_max)
            v = x[:, 3:6]
            v_dot = u / self.env.m - (self.env.b_drag / self.env.m) * v
            p_next = x[:, 0:3] + self.env.dt * v
            v_next = v + self.env.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)
            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + 0.02 * torch.sum(u * u, dim=1)
            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs["p"], dtype=torch.float32).unsqueeze(0).repeat(M, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs["r"]
                cost = cost + 2000.0 * torch.clamp(0.3 - d, min=0.0) ** 2
        return cost


class ProbeAdapter:
    def __init__(self, predictor):
        self.predictor = predictor

    def reset(self):
        self.predictor.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe, _ = self.predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


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


def mcnemar_discordant(rows_a, rows_b):
    b = sum(1 for a, bb in zip(rows_a, rows_b) if a["success"] and not bb["success"])
    c = sum(1 for a, bb in zip(rows_a, rows_b) if not a["success"] and bb["success"])
    return b, c


def mcnemar_p(b, c):
    if b + c == 0:
        return 1.0
    from math import comb
    n = b + c
    k = min(b, c)
    p_val = sum(comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(2.0 * p_val, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-threshes", type=str, default="1.0,1.5,2.0")
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
    d_threshes = [float(x) for x in args.d_threshes.split(",") if x.strip()]

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
            "experiment": "adaptive_pd_vs_r1s5",
            "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k,
            "d_threshes": d_threshes,
            "mass": args.mass, "drag": args.drag,
            "rng_isolation": "per_method_per_trial",
        },
        "by_method": {},
        "paired_tests": [],
    }

    # Reference methods
    ref_methods = [
        ("PD_s2", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=args.k, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=2.0,
        ))),
        ("R0_pd_s5", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=args.k, D=16, sigma=0.25,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0, pd_sigma=5.0,
        ))),
        ("R1_s5", lambda: ProbeAdapter(PTRMNMPCPredictor(
            model, env, K=args.k, D=16, sigma=0.25,
            candidate_mode="trm_pd", ranking_mode="rollout_all",
            alpha_blend=0.95, pd_sigma=5.0,
        ))),
    ]

    # Adaptive methods
    adaptive_methods = [
        (f"AdaptPD_d{dt:.1f}", lambda dt=dt: ProbeAdapter(
            AdaptivePDPredictor(env, K=args.k, sigma_near=5.0, sigma_far=2.0, d_thresh=dt)
        ))
        for dt in d_threshes
    ]

    all_methods = ref_methods + adaptive_methods

    for method_idx, (name, factory) in enumerate(all_methods):
        print(f"\n[{name}]", end=" ", flush=True)
        rows = []
        for i, x0 in enumerate(inits):
            set_method_seed(args.seed, name, method_idx, i)
            ctrl = factory()
            r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
            rows.append(r)
        agg = aggregate(rows)
        out["by_method"][name] = agg
        print(f"succ={agg['success_count']}/{agg['n']} "
              f"TErr={agg['TErr_mean']:.3f} IAE={agg['IAE_mean']:.3f}")

    # Paired McNemar: each adaptive method vs PD_s2 and vs R0_pd_s5
    for a_name in [n for n, _ in adaptive_methods]:
        for b_name in ["PD_s2", "R0_pd_s5", "R1_s5"]:
            bb, cc = mcnemar_discordant(
                out["by_method"][a_name]["individual"],
                out["by_method"][b_name]["individual"],
            )
            p = mcnemar_p(bb, cc)
            sig = "***" if p < 0.001 else "**" if p < 0.01 \
                else "*" if p < 0.05 else "n.s."
            out["paired_tests"].append({
                "method_a": a_name, "method_b": b_name,
                "b": bb, "c": cc, "p_exact_two_sided": p, "sig": sig,
            })
            print(f"  McNemar {a_name} vs {b_name}: b={bb} c={cc} p={p:.4f} {sig}")

    # Summary table
    print(f"\n{'='*70}")
    print("ADAPTIVE-PD vs R1_s5 SUMMARY")
    print(f"{'Method':<20s} {'Succ':>8s} {'Rate':>8s} {'TErr':>8s} {'IAE':>8s}")
    print("-" * 70)
    for name, _ in all_methods:
        v = out["by_method"][name]
        print(f"{name:<20s} {v['success_count']:>3d}/{v['n']:<3d} "
              f"{100*v['success_rate']:>6.1f}%  {v['TErr_mean']:>7.4f}  {v['IAE_mean']:>7.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove individual rows for cleaner JSON
    for k in out["by_method"]:
        out["by_method"][k] = {kk: vv for kk, vv in out["by_method"][k].items()
                               if kk != "individual"}
    out_path.write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")
    print(f"\n[adaptive-pd] wrote {out_path}")


if __name__ == "__main__":
    main()
