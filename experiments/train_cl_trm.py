# -*- coding: utf-8 -*-
"""
CL-TRM 闭环训练脚本

用 PD+CBF 闭环轨迹数据训练 TRM，生成 candidate_mode='trm_pd' 所需的模型。
输出: experiments/results_v6/cl_trm_model.pt
"""

import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quadrotor_core import (
    QuadrotorDynamics, TRMNMPC,
    generate_cl_trm_dataset, train_trm_jointly,
)

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'results_v6')
MODEL_PATH = os.path.join(SAVE_DIR, 'cl_trm_model.pt')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cpu')

    # 1. 生成 PD+CBF 闭环训练数据
    print("=" * 60)
    print("Step 1: 生成 PD+CBF 闭环训练数据")
    print("=" * 60)

    x_sp = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    env = QuadrotorDynamics()

    dataset = generate_cl_trm_dataset(
        env, x_sp=x_sp, size=500, steps_per_traj=10,
        Kp=4.0, Kd=3.0,
        pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)],
    )

    # 2. 训练 TRM
    print("\n" + "=" * 60)
    print("Step 2: 训练 CL-TRM")
    print("=" * 60)

    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params}")

    t0 = time.time()
    model, history = train_trm_jointly(
        model, dataset, env,
        epochs=100, batch_size=32, lr=0.001, gamma=0.95,
        lambda_Q=0.1, V_max=150.0, val_ratio=0.2, patience=20,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n训练完成，耗时 {elapsed:.1f}s")

    # 3. 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至: {MODEL_PATH}")
    print(f"文件大小: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
