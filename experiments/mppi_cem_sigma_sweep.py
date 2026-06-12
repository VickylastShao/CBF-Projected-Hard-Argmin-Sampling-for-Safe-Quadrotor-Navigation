#!/usr/bin/env python3
"""
MPPI / CEM σ-sweep: 关键缺口实验

目的：证明 hard-argmin 选择机制在宽 σ 下的必要性。
当 MPPI 使用 σ=5 (与 TSH 相同的候选分布宽度) 时，
importance-weighted mean 稀释最优候选，无法受益于宽探索。

实验设计：
- Task: narrow (5 个障碍物, same as mppi_lambda_sweep)
- K=10, N_MC=80, seed=7777 (与主实验配对)
- MPPI σ ∈ {2, 3, 5, 8}, λ ∈ {0.1, 5.0}
- CEM σ ∈ {2, 5}, n_iter ∈ {1, 3}
- TSH R0_pd_s5 σ=5 K=10 (argmin 参照)
- PD baseline K=1 (CBF-only)

关键区别 vs mppi_lambda_sweep.py:
- 本脚本 sweep σ 而非 λ
- 加入了 CEM 和 TSH 参照
- 使用完全相同的 TASK_FACTORIES / sample_initial_states
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
from experiments.baselines.mppi_controller import MPPIController
from experiments.baselines.cem_controller import CEMController
from experiments.ptrm_advantage_quick import (
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
)


class ProbeAdapter:
    """Wraps any controller so it matches probes_a's (u_safe, _) return."""
    def __init__(self, inner):
        self.inner = inner

    def reset(self):
        self.inner.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe = self.inner.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


class PTRMAdapter:
    """Wraps PTRMNMPCPredictor to match probes_a's run_trial signature."""
    def __init__(self, predictor):
        self.predictor = predictor

    def reset(self):
        self.predictor.reset()

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_safe, _ = self.predictor.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


def run_trial(controller, env, x0, x_sp, n_steps, enable_cbf=True):
    """与 mppi_lambda_sweep.py 完全一致的 trial 逻辑"""
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
        "TErr_mean": float(terrs.mean()), "TErr_median": float(np.median(terrs)),
        "IAE_mean": float(iaes.mean()),
        "latency_ms_mean": float(lats.mean()),
        "individual": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sigmas", type=str, default="2,3,5,8")
    parser.add_argument("--lambdas", type=str, default="0.1,5.0")
    parser.add_argument("--n-mc", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    out = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k,
            "sigmas": sigmas, "lambdas": lambdas,
            "mass": args.mass, "drag": args.drag,
            "enable_cbf": True,
            "purpose": "Show hard-argmin is necessary for wide-σ benefit",
            "rng_isolation": "per_method_per_trial",
        },
        "by_method": {},
    }

    # =====================================================
    # 1. MPPI σ-sweep (each λ × each σ)
    # =====================================================
    method_idx = 0
    for lam in lambdas:
        for sigma in sigmas:
            key = f"MPPI_lam{lam}_s{sigma:.0f}_K{args.k}"
            print(f"[sigma-sweep] {key}")
            rows = []
            for i, x0 in enumerate(inits):
                set_method_seed(args.seed, key, method_idx, i)
                ctrl = ProbeAdapter(MPPIController(env, K=args.k, sigma=sigma, lam=lam))
                r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
                rows.append(r)
            agg = aggregate(rows)
            print(f"  succ={agg['success_count']}/{agg['n']}  "
                  f"TErr={agg['TErr_mean']:.3f}  IAE={agg['IAE_mean']:.3f}")
            out["by_method"][key] = agg
            method_idx += 1

    # =====================================================
    # 2. CEM σ-sweep
    # =====================================================
    for sigma in [2.0, 5.0]:
        for n_iter in [1, 3]:
            key = f"CEM_s{sigma:.0f}_iter{n_iter}_K{args.k}"
            print(f"[sigma-sweep] {key}")
            rows = []
            for i, x0 in enumerate(inits):
                set_method_seed(args.seed, key, method_idx, i)
                ctrl = ProbeAdapter(CEMController(env, K=args.k, n_iter=n_iter, sigma=sigma))
                r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
                rows.append(r)
            agg = aggregate(rows)
            print(f"  succ={agg['success_count']}/{agg['n']}  "
                  f"TErr={agg['TErr_mean']:.3f}  IAE={agg['IAE_mean']:.3f}")
            out["by_method"][key] = agg
            method_idx += 1

    # =====================================================
    # 3. TSH R0_pd_s5 (hard-argmin, σ=5, K=10)
    # =====================================================
    key_r0 = "R0_pd_s5_K10"
    print(f"[sigma-sweep] {key_r0} (hard-argmin, σ=5)")
    # 加载 TRM 模型
    save_dir = ROOT / "experiments" / "results_v6"
    trm_path = save_dir / "cl_trm_model.pt"
    if not trm_path.exists():
        trm_path = save_dir / "trm_model.pt"
    if not trm_path.exists():
        print("ERROR: No TRM model found!")
        sys.exit(1)

    device = torch.device('cpu')
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    trm_model.eval()

    rows = []
    for i, x0 in enumerate(inits):
        set_method_seed(args.seed, key_r0, method_idx, i)
        predictor = PTRMNMPCPredictor(
            trm_model, env, K=args.k, D=16, sigma=0.25,
            alpha_blend=0.0, candidate_mode='pd',
            pd_sigma=5.0, use_rollout_cost=True,
        )
        ctrl = PTRMAdapter(predictor)
        r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
        rows.append(r)
    agg = aggregate(rows)
    print(f"  succ={agg['success_count']}/{agg['n']}  "
          f"TErr={agg['TErr_mean']:.3f}  IAE={agg['IAE_mean']:.3f}")
    out["by_method"][key_r0] = agg
    method_idx += 1

    # =====================================================
    # 4. PD baseline (K=1, σ=0)
    # =====================================================
    key_pd = "PD_K1"
    print(f"[sigma-sweep] {key_pd} (CBF-only baseline)")
    rows = []
    for i, x0 in enumerate(inits):
        set_method_seed(args.seed, key_pd, method_idx, i)
        ctrl = ProbeAdapter(MPPIController(env, K=1, sigma=0.0))
        r = run_trial(ctrl, env, x0, x_sp, args.n_steps)
        rows.append(r)
    agg = aggregate(rows)
    print(f"  succ={agg['success_count']}/{agg['n']}  "
          f"TErr={agg['TErr_mean']:.3f}  IAE={agg['IAE_mean']:.3f}")
    out["by_method"][key_pd] = agg

    # =====================================================
    # 摘要
    # =====================================================
    print("\n" + "=" * 70)
    print("SUMMARY: MPPI/CEM σ-sweep vs TSH (hard-argmin) σ=5")
    print("=" * 70)
    print(f"{'Method':<36s} {'Succ':>5s} {'Rate':>7s} {'TErr':>8s} {'IAE':>7s}")
    print("-" * 70)

    order = ['PD_K1', 'R0_pd_s5_K10']
    for lam in lambdas:
        for sigma in sigmas:
            order.append(f"MPPI_lam{lam}_s{sigma:.0f}_K{args.k}")
    for sigma in [2.0, 5.0]:
        for n_iter in [1, 3]:
            order.append(f"CEM_s{sigma:.0f}_iter{n_iter}_K{args.k}")

    for k in order:
        if k in out["by_method"]:
            v = out["by_method"][k]
            print(f"  {k:<34s} {v['success_count']:>3d}/{v['n']} {100*v['success_rate']:>6.1f}%  "
                  f"{v['TErr_mean']:>7.4f}  {v['IAE_mean']:>6.3f}")

    # 排名
    ranking = sorted(out["by_method"].items(),
                     key=lambda kv: (-kv[1]["success_count"], kv[1]["TErr_mean"]))
    out["ranking"] = [
        {"method": k, "success_count": v["success_count"],
         "TErr_mean": v["TErr_mean"], "IAE_mean": v["IAE_mean"]}
        for k, v in ranking
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[sigma-sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
