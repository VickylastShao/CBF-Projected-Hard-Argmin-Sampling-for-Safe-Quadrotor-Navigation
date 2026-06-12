#!/usr/bin/env python3
"""
MPPI / iCEM / CEM baseline 跑分，用于补 Tables II/III/VI 中缺失的采样基线对照。

接口兼容 probes_a_zero_train.py 的 trial 结构，输出 JSON 存入 results_v6/。

Usage:
  # MPPI baseline — narrow, K∈{5,10,20}, N_MC=40
  python experiments/run_mce_baselines.py \
      --methods MPPI,CEM \
      --k 5,10,20 --n-mc 40 --task narrow \
      --output experiments/results_v6/mce_narrow_n40.json

  # Cross-config with MPPI
  python experiments/run_mce_baselines.py \
      --methods MPPI \
      --k 10 --n-mc 40 --task narrow --mass 2.25 \
      --output experiments/results_v6/mce_mass_n40.json

  # Cross-config drag
  python experiments/run_mce_baselines.py \
      --methods MPPI \
      --k 10 --n-mc 40 --task narrow --drag 0.15 \
      --output experiments/results_v6/mce_drag_n40.json

  # Tasks (two_gate, u_shape)
  python experiments/run_mce_baselines.py \
      --methods MPPI,CEM \
      --k 10 --n-mc 40 --task two_gate \
      --output experiments/results_v6/mce_two_gate_n40.json
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
from experiments.baselines.mppi_controller import MPPIController
from experiments.baselines.cem_controller import CEMController
from experiments.ptrm_advantage_quick import (
    TASK_FACTORIES, sample_initial_states, set_seed,
)

DEVICE = "cpu"
TErr_THRESH = 0.5

# =============================================================================
# iCEM: CEM with PD-mean warm-start every iteration (Pinneri et al., 2020)
# =============================================================================
class iCEMController(CEMController):
    """iCEM 引入 $u_{\text{pd}}$ warm-start 在每个 iteration 作为额外精英。

    参考: Pinneri et al., 2020 "The Importance Sampling CEM".
    """
    def __init__(self, env, K=50, n_iter=3, sigma=2.0, elite_frac=0.2,
                 rollout_steps=20, Kp=4.0, Kd=3.0, obs_weight=2000.0, eta_hyst=0.05):
        super().__init__(env, K, n_iter, sigma, elite_frac,
                         rollout_steps, Kp, Kd, obs_weight, eta_hyst)

    def predict_action(self, x_init, x_sp, enable_cbf=True):
        u_pd = self._compute_pd_baseline(x_init, x_sp)
        mu = u_pd.clone()
        sigma_curr = self.sigma
        n_elite = max(1, int(self.K * self.elite_frac))

        for iteration in range(self.n_iter):
            noise = torch.randn(self.K, 3) * sigma_curr
            u_candidates = mu.unsqueeze(0) + noise

            cost = self._batch_rollout_cost(x_init, u_candidates, x_sp)
            if self.last_u is not None:
                dist = torch.sum((u_candidates - self.last_u.unsqueeze(0)) ** 2, dim=1)
                cost = cost + self.eta_hyst * dist

            # iCEM: 注入 PD baseline 作为额外精英候选
            pd_cost = self._batch_rollout_cost(x_init, u_pd.unsqueeze(0), x_sp)
            elite_costs = torch.cat([cost, pd_cost])
            elite_candidates = torch.cat([u_candidates, u_pd.unsqueeze(0)], dim=0)

            elite_indices = torch.argsort(elite_costs)[:n_elite]
            elite_samples = elite_candidates[elite_indices]

            mu = torch.mean(elite_samples, dim=0)
            sigma_curr = max(0.1, torch.std(elite_samples, dim=0).mean().item())

        u_nominal = mu
        self.last_u = u_nominal.clone()

        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x_init, u_nominal)
        else:
            u_safe = torch.clamp(u_nominal, self.env.u_min, self.env.u_max)

        return u_safe


# =============================================================================
# Controllers factory
# =============================================================================
METHOD_FACTORIES = {
    "MPPI": lambda env, K, kwargs: MPPIController(env, K=K, sigma=2.0, lam=0.1, **kwargs),
    "CEM": lambda env, K, kwargs: CEMController(env, K=K, n_iter=3, sigma=2.0, elite_frac=0.2, **kwargs),
    "iCEM": lambda env, K, kwargs: iCEMController(env, K=K, n_iter=3, sigma=2.0, elite_frac=0.2, **kwargs),
}


# =============================================================================
# Trial runner (adapted from probes_a_zero_train.py run_trial)
# =============================================================================
def run_trial(controller, env: QuadrotorDynamics, x0: torch.Tensor,
              x_sp: torch.Tensor, n_steps: int) -> dict[str, Any]:
    if hasattr(controller, "reset"):
        controller.reset()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    lat_list = []
    iae = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_safe = controller.predict_action(x, x_sp, enable_cbf=True)
        lat_list.append((time.perf_counter() - t0) * 1000.0)
        if isinstance(u_safe, tuple):
            u_safe = u_safe[0]  # rare: MPPI returning (u_safe, u_seq)
        u_first = u_safe.detach().cpu().numpy()[:3] if hasattr(u_safe, 'detach') else u_safe[:3]
        u_first_t = torch.tensor(u_first, dtype=torch.float32)
        x = env.step_discrete(x, u_first_t)
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    return {
        "TErr": terr,
        "IAE": iae,
        "success": (terr < TErr_THRESH) and (not collided),
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
        "TErr_success_mean": float(terrs[list(r["success"] for r in rows)].mean()) if any(r["success"] for r in rows) else -1.0,
        "IAE_mean": float(iaes.mean()),
        "latency_ms_mean": float(lats.mean()),
        "latency_ms_median": float(np.median(lats)),
        "individual": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="MPPI,CEM")
    parser.add_argument("--k", type=str, default="5,10,20")
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    k_values = [int(k) for k in args.k.split(",") if k.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])} for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    results: dict[str, Any] = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "k_values": k_values,
            "methods": methods, "mass": args.mass, "drag": args.drag,
        },
        "by_method": {},
    }

    for K_val in k_values:
        for name in methods:
            key = f"{name}_K{K_val}"
            print(f"[mce] {key}")
            ctrl = METHOD_FACTORIES[name](env, K_val, {})
            rows = []
            for i, x0 in enumerate(inits):
                r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
                rows.append(r)
            agg = aggregate(rows)
            succ_str = f"{agg['success_count']}/{agg['n']}"
            print(f"  succ={succ_str} TErr={agg.get('TErr_success_mean', agg['TErr_mean']):.3f} "
                  f"IAE={agg['IAE_mean']:.3f} lat={agg['latency_ms_median']:.2f}ms")
            results["by_method"][key] = agg

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[mce] wrote {out_path}")


if __name__ == "__main__":
    main()