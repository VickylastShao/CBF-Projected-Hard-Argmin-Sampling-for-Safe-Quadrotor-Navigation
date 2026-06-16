#!/usr/bin/env python3
"""
B1: Retrain CL-TRM on narrow-matched expert dataset.

输入: experiments/results_v6/narrow_dataset_v1.pt (120 success traj, 18000 samples)
  payload: {"X": (N,12) tensor, "Y": (N,30) expert u_seq tensor, "meta": ...}

训练 27,935-param TRMNMPC (与原 CL-TRM 完全相同的架构) on narrow-matched data。
环境 = narrow obstacles (来自 dataset.meta.obstacles)，保证训练-评估分布对齐。

输出: experiments/results_v6/cl_trm_narrow_v1.pt
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
from quadrotor_core.trm_network import TRMNMPC                         # noqa: E402
from quadrotor_core.training import train_trm_jointly                  # noqa: E402

SEED = 4242


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default=str(ROOT / "experiments" / "results_v6" / "narrow_dataset_v1.pt"))
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "experiments" / "results_v6" / "cl_trm_narrow_v1.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print(f"[b1-train] loading dataset from {args.dataset}")
    payload = torch.load(args.dataset, map_location=device, weights_only=False)
    X = payload["X"]  # (N, 12)
    Y = payload["Y"]  # (N, 30)
    meta = payload["meta"]
    print(f"[b1-train] dataset: N={X.shape[0]} from {meta['n_success']}/{meta['n_attempt']} success traj "
          f"on task={meta['task']}")

    # 重要：env 必须用 narrow obstacles，与 dataset 同分布
    obstacles = [{"p": np.array(o["p"], dtype=np.float32), "r": float(o["r"])}
                 for o in meta["obstacles"]]
    env = QuadrotorDynamics(obstacles=obstacles)
    print(f"[b1-train] env: {len(obstacles)} obstacles (narrow)")

    # 转 list[(X_i, Y_i)] 格式
    dataset = [(X[i], Y[i]) for i in range(X.shape[0])]

    # 与原 cl_trm_model 完全相同的架构
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[b1-train] model params: {n_params}")
    assert n_params == 27935, f"参数量必须 27935，实际 {n_params}"

    print(f"[b1-train] starting joint training: epochs={args.epochs} batch={args.batch_size} "
          f"lr={args.lr} patience={args.patience}")
    t0 = time.time()
    model, history = train_trm_jointly(
        model, dataset, env,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        gamma=0.95, lambda_Q=0.1, V_max=150.0,
        val_ratio=0.2, patience=args.patience,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[b1-train] training done in {elapsed:.1f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "model_state_dict": model.state_dict(),
        "meta": {
            "source_dataset": str(args.dataset),
            "n_samples": X.shape[0],
            "task": meta["task"],
            "epochs_run": len(history["train_total_loss"]),
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
            "elapsed_sec": elapsed,
            "obstacles": meta["obstacles"],
        },
        "history": history,
    }
    torch.save(save_payload, out_path)
    sz = out_path.stat().st_size / 1024
    print(f"[b1-train] saved {out_path} ({sz:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
