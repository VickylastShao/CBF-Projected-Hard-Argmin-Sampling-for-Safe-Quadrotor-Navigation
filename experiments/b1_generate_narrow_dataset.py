#!/usr/bin/env python3
"""
B1: Generate narrow-matched expert dataset for CL-TRM retrain.

策略：
  1. 用最强已知 controller (TRM+PD+Rollout, K=20, α=0.95) 在 narrow 上跑 N 条闭环。
  2. 只保留 success 轨迹（terminal_error < 0.30 且 no collision）。
  3. 在每条 success 轨迹的每个 state x_t，用 GoldenNMPCSolver 回填 expert u_seq (30,)。
  4. 输出 (x_t, x_sp, u_seq_expert) 三元组，与 train_cl_trm 兼容的格式。
  5. 同时输出 narrow 任务的 obstacle 配置。

只跑数据生成，不训练。

CLI:
  python3 experiments/b1_generate_narrow_dataset.py --n-traj 200 --output experiments/results_v6/narrow_dataset.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core.dynamics import QuadrotorDynamics                  # noqa: E402
from quadrotor_core.nmpc_solver import GoldenNMPCSolver                # noqa: E402
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor             # noqa: E402

from experiments.ptrm_advantage_quick import (                          # noqa: E402
    TASK_FACTORIES, load_trm_model, sample_initial_states, set_seed,
)

DEFAULT_MODEL = ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"


def run_collect(env, x0, x_sp, ctrl, solver, n_steps=150):
    """跑一条闭环，记录 (x_t, x_sp, expert_u_seq) 序列。"""
    x = x0.clone()
    traj_states = [x.clone()]
    expert_seqs = []
    collided = False
    for _ in range(n_steps):
        u_safe, _ = ctrl.predict_action(x, x_sp, enable_cbf=True)
        # expert u_seq for current state x (in parallel)
        u_exp = solver.solve(x, x_sp)  # (30,)
        expert_seqs.append(u_exp.clone())
        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3], dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        traj_states.append(x.clone())
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
                break
        if collided:
            break
    terr = float(torch.norm(x[:3] - x_sp[:3]).item())
    success = (terr < 0.30) and (not collided)
    # 跟 expert_seqs 长度对齐 states（去掉最后一个 state，没有对应 u）
    states_in = traj_states[:len(expert_seqs)]
    return success, terr, collided, states_in, expert_seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=200, help="目标 success 轨迹数")
    parser.add_argument("--max-attempts", type=int, default=600, help="最大尝试次数")
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])} for o in task["obstacles"]]
    env = QuadrotorDynamics(obstacles=obstacles)
    model = load_trm_model(Path(args.model), "cpu")
    model.eval()
    solver = GoldenNMPCSolver(env, horizon=10)

    print(f"[b1] task={args.task} target n_traj={args.n_traj} max={args.max_attempts}")

    all_states = []
    all_useqs = []
    n_success = 0
    n_attempt = 0
    t0 = time.time()

    inits = sample_initial_states(task, args.max_attempts, args.seed)

    for i, x0 in enumerate(inits):
        if n_success >= args.n_traj:
            break
        n_attempt += 1
        ctrl = PTRMNMPCPredictor(model=model, env=env, K=args.k, D=16,
                                  candidate_mode="trm_pd", ranking_mode="rollout_all",
                                  alpha_blend=args.alpha, pd_sigma=2.0)
        success, terr, collided, states, useqs = run_collect(
            env, x0, x_sp, ctrl, solver, args.n_steps,
        )
        elapsed = time.time() - t0
        flag = "OK" if success else ("COLL" if collided else "FAIL")
        print(f"[b1] attempt {n_attempt}/{i+1} succ_so_far={n_success} {flag} terr={terr:.2f} "
              f"steps={len(states)} elapsed={elapsed:.0f}s")
        if success:
            n_success += 1
            all_states.extend(states)
            all_useqs.extend(useqs)

    if n_success == 0:
        print("[b1] ERROR: no success trajectories collected")
        return 1

    states_t = torch.stack(all_states).to(torch.float32)  # (N, 6)
    useqs_t = torch.stack(all_useqs).to(torch.float32)    # (N, 30)
    x_sp_t = x_sp.to(torch.float32).expand(states_t.shape[0], -1).clone()  # (N, 6)
    X = torch.cat([states_t, x_sp_t], dim=1)  # (N, 12)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": X,         # (N, 12) = [x_init(6), x_sp(6)]
        "Y": useqs_t,   # (N, 30) expert u_seq
        "meta": {
            "task": args.task, "n_success": n_success, "n_attempt": n_attempt,
            "n_samples": X.shape[0], "k": args.k, "alpha": args.alpha,
            "n_steps": args.n_steps,
            "obstacles": [{"p": o["p"].tolist(), "r": float(o["r"])} for o in obstacles],
            "x_sp": x_sp.tolist(),
        },
    }
    torch.save(payload, out_path)
    print(f"[b1] success={n_success}/{n_attempt} samples={X.shape[0]} wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
