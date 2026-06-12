#!/usr/bin/env python3
"""
PI / mass-rescaled-PD baselines under +50% mass mismatch.

Devil's-advocate claim：+50% mass collapse 是 tautological，因为 PD 把
nominal mass 写死。本脚本测试两个修正后的 single-source baseline：

1. **PI-PD**: PD + integral term, 在线消除恒定 mass error。
2. **Adaptive-PD**: mass-rescaled PD = `(true_m/nominal_m) * u_pd`，模拟一个
   理想化的 "已知 mass scale" baseline。这是 PI baseline 的上界。

若这两个 single-source baseline 在 +50% mass narrow 上还是 collapse，论文
的 "two-source = passive parameter adaptation" 叙事就更立得住。

Usage:
  python experiments/run_pi_baseline.py \
      --methods PI,Adaptive \
      --k 10 --n-mc 40 \
      --output experiments/results_v6/pi_baseline_mass_n40.json
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
    TASK_FACTORIES, sample_initial_states, set_seed,
)

DEVICE = "cpu"


# =============================================================================
# PI-PD: PD + integral term to remove constant mass-induced bias
# =============================================================================
class PI_PD_RolloutController:
    """PI-PD + 单源 Gaussian + rollout，主要测试 integral 项是否消除 mass collapse。"""
    def __init__(self, env, K=10, sigma_pd=2.0, Kp=4.0, Kd=3.0, Ki=2.0,
                 rollout_steps=20, obs_weight=2000.0):
        self.env = env
        self.K = K
        self.sigma_pd = sigma_pd
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.rollout_steps = rollout_steps
        self.obs_weight = obs_weight
        self.q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        self.R_U = 0.02
        self.integral_e_p = torch.zeros(3)
        self.i_clamp = 5.0

    def reset(self):
        self.integral_e_p = torch.zeros(3)

    def _pi_pd_first_step(self, x, x_sp):
        e_p = x_sp[0:3] - x[0:3]
        e_v = x_sp[3:6] - x[3:6]
        # integral update (per step, dt=0.02)
        self.integral_e_p = torch.clamp(
            self.integral_e_p + self.env.dt * e_p,
            -self.i_clamp, self.i_clamp,
        )
        u = self.env.m * (self.Kp * e_p + self.Kd * e_v + self.Ki * self.integral_e_p)
        return torch.clamp(u, self.env.u_min, self.env.u_max)

    def _rollout_cost(self, x_init, u_cand, x_sp):
        K_val = u_cand.shape[0]
        x = x_init.unsqueeze(0).repeat(K_val, 1)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(K_val, 1)
        cost = torch.zeros(K_val)
        q = self.q_diag.unsqueeze(0)
        # local copy of integral to avoid mutating state during rollout
        i_local = self.integral_e_p.clone()
        for s in range(self.rollout_steps):
            if s == 0:
                u = u_cand
            else:
                e_p = x_sp6[:, 0:3] - x[:, 0:3]
                e_v = x_sp6[:, 3:6] - x[:, 3:6]
                i_local = torch.clamp(i_local + self.env.dt * e_p.mean(dim=0),
                                       -self.i_clamp, self.i_clamp)
                u = self.env.m * (self.Kp * e_p + self.Kd * e_v + self.Ki * i_local.unsqueeze(0))
            u = torch.clamp(u, self.env.u_min, self.env.u_max)
            v = x[:, 3:6]
            v_dot = u / self.env.m - (self.env.b_drag / self.env.m) * v
            p_next = x[:, 0:3] + self.env.dt * v
            v_next = v + self.env.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)
            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + self.R_U * torch.sum(u * u, dim=1)
            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs["p"], dtype=torch.float32).unsqueeze(0).repeat(K_val, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs["r"]
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2
        return cost

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_pi = self._pi_pd_first_step(x, x_sp)
        if self.K == 1:
            u_cand = u_pi.unsqueeze(0)
        else:
            noise = torch.randn(self.K, 3) * self.sigma_pd
            u_cand = u_pi.unsqueeze(0) + noise
        u_cand = torch.clamp(u_cand, self.env.u_min, self.env.u_max)
        costs = self._rollout_cost(x, u_cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = u_cand[best]
        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x, u_nom)
        else:
            u_safe = torch.clamp(u_nom, self.env.u_min, self.env.u_max)
        return u_safe


# =============================================================================
# Adaptive-PD: oracle mass-rescaled PD (best-case single-source PI)
# =============================================================================
class Adaptive_PD_RolloutController:
    """oracle mass-rescaled PD: u = true_m * (Kp e_p + Kd e_v); 单源 + rollout.

    这是 PI baseline 的 oracle upper bound: 假设 controller 已知真实 mass。
    若这个 baseline 在 narrow + mass mismatch 还是 collapse，问题就不是
    'PD 用了 nominal mass', 而是 narrow 本身需要 candidate diversity.
    """
    def __init__(self, env, K=10, sigma_pd=2.0, Kp=4.0, Kd=3.0,
                 rollout_steps=20, obs_weight=2000.0, true_mass=None):
        self.env = env
        self.K = K
        self.sigma_pd = sigma_pd
        self.Kp = Kp
        self.Kd = Kd
        self.rollout_steps = rollout_steps
        self.obs_weight = obs_weight
        self.true_mass = true_mass if true_mass is not None else env.m
        self.q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
        self.R_U = 0.02

    def reset(self):
        pass

    def _adaptive_pd_first_step(self, x, x_sp):
        e_p = x_sp[0:3] - x[0:3]
        e_v = x_sp[3:6] - x[3:6]
        u = self.true_mass * (self.Kp * e_p + self.Kd * e_v)
        return torch.clamp(u, self.env.u_min, self.env.u_max)

    def _rollout_cost(self, x_init, u_cand, x_sp):
        K_val = u_cand.shape[0]
        x = x_init.unsqueeze(0).repeat(K_val, 1)
        x_sp6 = x_sp[:6].unsqueeze(0).repeat(K_val, 1)
        cost = torch.zeros(K_val)
        q = self.q_diag.unsqueeze(0)
        for s in range(self.rollout_steps):
            if s == 0:
                u = u_cand
            else:
                e_p = x_sp6[:, 0:3] - x[:, 0:3]
                e_v = x_sp6[:, 3:6] - x[:, 3:6]
                u = self.true_mass * (self.Kp * e_p + self.Kd * e_v)
            u = torch.clamp(u, self.env.u_min, self.env.u_max)
            v = x[:, 3:6]
            v_dot = u / self.env.m - (self.env.b_drag / self.env.m) * v
            p_next = x[:, 0:3] + self.env.dt * v
            v_next = v + self.env.dt * v_dot
            x = torch.cat([p_next, v_next], dim=1)
            err = x - x_sp6
            cost = cost + torch.sum(q * err * err, dim=1) + self.R_U * torch.sum(u * u, dim=1)
            for obs in self.env.obstacles:
                obs_p = torch.tensor(obs["p"], dtype=torch.float32).unsqueeze(0).repeat(K_val, 1)
                d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs["r"]
                cost = cost + self.obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2
        return cost

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_adp = self._adaptive_pd_first_step(x, x_sp)
        if self.K == 1:
            u_cand = u_adp.unsqueeze(0)
        else:
            noise = torch.randn(self.K, 3) * self.sigma_pd
            u_cand = u_adp.unsqueeze(0) + noise
        u_cand = torch.clamp(u_cand, self.env.u_min, self.env.u_max)
        costs = self._rollout_cost(x, u_cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = u_cand[best]
        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x, u_nom)
        else:
            u_safe = torch.clamp(u_nom, self.env.u_min, self.env.u_max)
        return u_safe


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
        "TErr": terr, "IAE": iae,
        "success": (terr < 0.5) and (not collided),
        "collided": collided,
        "latency_ms_mean": float(np.mean(lat_list)),
        "latency_ms_median": float(np.median(lat_list)),
        "latency_ms_p95": float(np.percentile(lat_list, 95)),
    }


def aggregate(rows):
    n = len(rows)
    succ = sum(1 for r in rows if r["success"])
    terrs = np.array([r["TErr"] for r in rows])
    iaes = np.array([r["IAE"] for r in rows])
    lats = np.array([r["latency_ms_mean"] for r in rows])
    return {
        "n": n, "success_count": succ, "success_rate": succ / n,
        "TErr_mean": float(terrs.mean()),
        "IAE_mean": float(iaes.mean()),
        "latency_ms_mean": float(lats.mean()),
        "latency_ms_median": float(np.median(lats)),
        "individual": rows,
    }


METHOD_FACTORIES = {
    "PI": lambda env, K, true_mass: PI_PD_RolloutController(env, K=K),
    "Adaptive": lambda env, K, true_mass: Adaptive_PD_RolloutController(env, K=K, true_mass=true_mass),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="PI,Adaptive")
    parser.add_argument("--k", type=str, default="10")
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=2.25, help="env true mass; nominal=1.5")
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
            print(f"[pi] {key}")
            ctrl = METHOD_FACTORIES[name](env, K_val, args.mass)
            rows = []
            for i, x0 in enumerate(inits):
                r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
                rows.append(r)
            agg = aggregate(rows)
            print(f"  succ={agg['success_count']}/{agg['n']} TErr={agg['TErr_mean']:.3f} "
                  f"IAE={agg['IAE_mean']:.3f} lat={agg['latency_ms_median']:.2f}ms")
            results["by_method"][key] = agg

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[pi] wrote {out_path}")


if __name__ == "__main__":
    main()