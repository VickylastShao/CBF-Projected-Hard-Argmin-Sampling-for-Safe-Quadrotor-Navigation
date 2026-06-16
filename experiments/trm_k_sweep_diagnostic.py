#!/usr/bin/env python3
"""
TRM-only / PD-only / Expert 在 narrow 上的 K-sweep 闭环对比。

补 trm_failure_diagnostic.py：
  - 之前 K=1 闭环 expert/TRM/PD 都 0% → 不能区分 TRM 的问题。
  - 这里跑 K∈{1,5,10,20} × {TRM_rollout, PD+Rollout}，加 expert K=1 参考。
  - 同样 10 条 narrow initial states，paired。
输出: experiments/results_v6/trm_k_sweep_diagnostic.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core import QuadrotorDynamics                            # noqa: E402
from quadrotor_core.nmpc_solver import GoldenNMPCSolver                 # noqa: E402
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor              # noqa: E402

from experiments.ptrm_advantage_quick import (                          # noqa: E402
    TASK_FACTORIES, load_trm_model, sample_initial_states, set_seed,
)

OUT_JSON = ROOT / "experiments" / "results_v6" / "trm_k_sweep_diagnostic.json"
OUT_MD = ROOT / "experiments" / "results_v6" / "trm_k_sweep_diagnostic.md"
MODEL_PATH = ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"
DEVICE = "cpu"
SEED = 2026
N_TRIALS = 10
N_STEPS = 150
TASK_NAME = "narrow"
K_VALUES = [1, 5, 10, 20]


def make_trm_rollout(model, env, K):
    return PTRMNMPCPredictor(model=model, env=env, K=K, D=16, sigma=0.0,
                              candidate_mode="trm_rollout", ranking_mode="rollout_all",
                              alpha_blend=0.0, pd_sigma=2.0, noise_mode="none")


def make_pd_rollout(model, env, K):
    return PTRMNMPCPredictor(model=model, env=env, K=K, D=16,
                              candidate_mode="pd", ranking_mode="rollout_all",
                              alpha_blend=1.0, pd_sigma=2.0)


def make_trm_pd_rollout(model, env, K, alpha=0.95):
    return PTRMNMPCPredictor(model=model, env=env, K=K, D=16,
                              candidate_mode="trm_pd", ranking_mode="rollout_all",
                              alpha_blend=alpha, pd_sigma=2.0)


def run_closed_loop(predictor, env, x0, x_sp, n_steps=N_STEPS):
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    for _ in range(n_steps):
        u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=True)
        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3], dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        traj.append(x.numpy().copy())
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    return {"terminal_error": terr, "collided": collided, "success": terr < 0.30 and not collided}


def run_expert_loop(solver, env, x0, x_sp, n_steps=N_STEPS):
    """Expert K=1 with CBF: 每步用 GoldenNMPC 重解，取首步过 CBF。"""
    x = x0.clone()
    collided = False
    for _ in range(n_steps):
        u_seq = solver.solve(x, x_sp).numpy()
        u_first = torch.tensor(u_seq[:3], dtype=torch.float32)
        u_safe = env.apply_cbf_projection(x, u_first)
        x = env.step_discrete(x, u_safe)
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    terr = float(np.linalg.norm(x.numpy()[:3] - x_sp[:3].numpy()))
    return {"terminal_error": terr, "collided": collided, "success": terr < 0.30 and not collided}


def main():
    set_seed(SEED)
    task = TASK_FACTORIES[TASK_NAME](SEED)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])} for o in task["obstacles"]]
    env = QuadrotorDynamics(obstacles=obstacles)
    inits = sample_initial_states(task, N_TRIALS, SEED)

    model = load_trm_model(MODEL_PATH, DEVICE)
    model.eval()
    solver = GoldenNMPCSolver(env, horizon=10)

    results = {"task": TASK_NAME, "n_trials": N_TRIALS, "n_steps": N_STEPS, "k_values": K_VALUES,
               "by_method": {}}

    # Expert K=1 reference
    print(f"[k-sweep] Expert NMPC K=1 (reference)")
    rows = []
    for i, x0 in enumerate(inits):
        r = run_expert_loop(solver, env, x0, x_sp)
        rows.append(r)
        print(f"  trial {i}: terr={r['terminal_error']:.2f} succ={r['success']} coll={r['collided']}")
    results["by_method"]["Expert_K1"] = rows

    for K in K_VALUES:
        for label, make_fn in (("TRM_Rollout", make_trm_rollout),
                               ("PD_Rollout", make_pd_rollout),
                               ("TRM_PD_Rollout_a095", make_trm_pd_rollout)):
            key = f"{label}_K{K}"
            print(f"[k-sweep] {key}")
            rows = []
            for i, x0 in enumerate(inits):
                ctrl = make_fn(model, env, K)
                r = run_closed_loop(ctrl, env, x0, x_sp)
                rows.append(r)
            succ = sum(r["success"] for r in rows)
            mean_terr = float(np.mean([r["terminal_error"] for r in rows]))
            print(f"  succ={succ}/{len(rows)} mean_terr={mean_terr:.2f}")
            results["by_method"][key] = rows

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # markdown summary
    md = ["# TRM-only / PD / TRM+PD K-sweep diagnostic", "",
          f"task=**{TASK_NAME}**, n_trials={N_TRIALS}, n_steps={N_STEPS}, model={MODEL_PATH.name}",
          "", "## Summary table",
          "| method | K | success | TErr mean | TErr median |", "|---|---|---|---|---|"]
    methods_order = [("Expert_K1", None)]
    for K in K_VALUES:
        for label in ("TRM_Rollout", "PD_Rollout", "TRM_PD_Rollout_a095"):
            methods_order.append((f"{label}_K{K}", K))
    for key, K in methods_order:
        rows = results["by_method"][key]
        terrs = [r["terminal_error"] for r in rows]
        succ = sum(r["success"] for r in rows)
        md.append(f"| {key} | {K if K else 1} | {succ}/{len(rows)} | {np.mean(terrs):.2f} | {np.median(terrs):.2f} |")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[k-sweep] wrote {OUT_JSON}, {OUT_MD}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
