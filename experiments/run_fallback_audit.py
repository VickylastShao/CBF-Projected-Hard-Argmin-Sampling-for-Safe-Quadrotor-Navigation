#!/usr/bin/env python3
"""
CBF fallback-trigger rate audit.

Wraps QuadrotorDynamics._project_control to count fallback invocations per
(method, K, config) cell, replays paired trials, and produces a per-cell
fallback-rate table (count / total CBF calls).

Output: experiments/results_v6/fallback_audit.json + .md
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

from quadrotor_core.dynamics import QuadrotorDynamics
from experiments.ptrm_advantage_quick import (
    TASK_FACTORIES, sample_initial_states, set_seed, load_trm_model,
)
from experiments.probes_a_zero_train import PROBE_FACTORIES, make_baseline


# Monkey-patch fallback counter
_FALLBACK_COUNTER = {"calls": 0, "fallback": 0}
_orig_project = QuadrotorDynamics._project_control


def _instrumented_project(self, u_nominal, A, b):
    _FALLBACK_COUNTER["calls"] += 1
    u_box = np.clip(u_nominal, self.u_min, self.u_max)
    if np.dot(A, u_box) <= b:
        return u_box
    low, high = 0.0, 50.0
    for _ in range(10):
        u_test = np.clip(u_nominal - high * A, self.u_min, self.u_max)
        if np.dot(A, u_test) <= b:
            break
        high *= 2.0
    best_u = u_box
    has_solution = False
    for _ in range(25):
        mid = (low + high) / 2.0
        u_test = np.clip(u_nominal - mid * A, self.u_min, self.u_max)
        val = np.dot(A, u_test)
        if val <= b:
            best_u = u_test
            high = mid
            has_solution = True
        else:
            low = mid
    if not has_solution:
        _FALLBACK_COUNTER["fallback"] += 1
        if np.linalg.norm(A) < 1e-9:
            best_u = u_box
        else:
            best_u = np.where(A >= 0.0, self.u_min, self.u_max)
    return best_u


QuadrotorDynamics._project_control = _instrumented_project


def reset_counter():
    _FALLBACK_COUNTER["calls"] = 0
    _FALLBACK_COUNTER["fallback"] = 0


def get_counter():
    return _FALLBACK_COUNTER["calls"], _FALLBACK_COUNTER["fallback"]


def run_trial(controller, env, x0, x_sp, n_steps):
    if hasattr(controller, "reset"):
        controller.reset()
    x = x0.clone()
    collided = False
    for _ in range(n_steps):
        result = controller.predict_action(x, x_sp, enable_cbf=True)
        if isinstance(result, tuple):
            u_safe = result[0]
        else:
            u_safe = result
        u_first = u_safe.detach().cpu().numpy()[:3] if hasattr(u_safe, 'detach') else u_safe[:3]
        u_first_t = torch.tensor(u_first, dtype=torch.float32)
        x = env.step_discrete(x, u_first_t)
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    terr = float(np.linalg.norm(x[:3].numpy() - x_sp[:3].numpy()))
    return {"TErr": terr, "success": (terr < 0.5) and (not collided), "collided": collided}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--methods", type=str, default="PD,R1_s5",
                       help="probe methods (R1_s5) or baselines (PD)")
    parser.add_argument("--k-values", type=str, default="5,10,20")
    parser.add_argument("--configs", type=str, default="nominal,mass,drag",
                       help="comma-list of nominal,mass,drag")
    parser.add_argument("--model", type=str,
                       default=str(ROOT / "experiments/results_v6/cl_trm_model.pt"))
    parser.add_argument("--output", type=str,
                       default=str(ROOT / "experiments/results_v6/fallback_audit.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    methods = [m.strip() for m in args.methods.split(",")]
    k_values = [int(k) for k in args.k_values.split(",")]
    configs = [c.strip() for c in args.configs.split(",")]

    CONFIG_PARAMS = {
        "nominal": {"mass": 1.5, "drag": 0.1},
        "mass": {"mass": 2.25, "drag": 0.1},
        "drag": {"mass": 1.5, "drag": 0.15},
    }

    model = load_trm_model(Path(args.model), "cpu")
    model.eval()

    task_dict = TASK_FACTORIES[args.task](args.seed)
    x_sp = task_dict["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task_dict["obstacles"]]

    results: dict[str, Any] = {
        "meta": {"task": args.task, "seed": args.seed, "n_mc": args.n_mc,
                 "n_steps": args.n_steps, "methods": methods, "k_values": k_values,
                 "configs": configs},
        "cells": {},
    }

    for config in configs:
        cp = CONFIG_PARAMS[config]
        env = QuadrotorDynamics(m=cp["mass"], b_drag=cp["drag"], obstacles=obstacles)
        inits = sample_initial_states(task_dict, args.n_mc, args.seed)

        for K_val in k_values:
            for name in methods:
                cell_key = f"{config}/{name}/K{K_val}"
                print(f"[audit] {cell_key}")
                reset_counter()
                t0 = time.perf_counter()
                rows = []
                for x0 in inits:
                    if name in PROBE_FACTORIES:
                        ctrl = PROBE_FACTORIES[name](model, env, K_val)
                    else:
                        ctrl = make_baseline(name, model, env, K_val)
                    r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
                    rows.append(r)
                calls, fb = get_counter()
                succ = sum(1 for r in rows if r["success"])
                elapsed = time.perf_counter() - t0
                print(f"  succ={succ}/{len(rows)} CBF_calls={calls} "
                      f"fallback={fb} rate={fb/max(calls,1)*100:.3f}% ({elapsed:.1f}s)")
                results["cells"][cell_key] = {
                    "config": config, "method": name, "K": K_val,
                    "n_trials": len(rows), "success_count": succ,
                    "cbf_calls_total": calls,
                    "fallback_invocations": fb,
                    "fallback_rate_pct": fb / max(calls, 1) * 100.0,
                    "elapsed_s": elapsed,
                }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out_path}")


if __name__ == "__main__":
    main()