# -*- coding: utf-8 -*-
"""
P0-1.3: 重新训练 TRM 模型（含终端代价的 NMPC 专家数据）

在 NMPC 求解器添加终端代价 P_f 后，重新生成专家数据集并训练 TRM 模型。
"""

import sys
import os
import time
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quadrotor_core import (
    QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
    generate_quadrotor_dataset, train_trm_jointly
)

SEED = 2026


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    os.makedirs(save_dir, exist_ok=True)
    trm_path = os.path.join(save_dir, 'trm_model.pt')

    # 备份旧模型
    if os.path.exists(trm_path):
        backup_path = trm_path + '.backup'
        os.rename(trm_path, backup_path)
        print(f"旧模型已备份至 {backup_path}")

    # 1. 生成数据集（含终端代价的 NMPC 专家）
    print("=" * 60)
    print("步骤 1: 生成专家数据集 (含终端代价 P_f)")
    print("=" * 60)
    env_train = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env_train, horizon=10)
    print(f"P_f 对角元素: {torch.diag(solver.P_f).tolist()}")

    t0 = time.time()
    dataset = generate_quadrotor_dataset(env_train, solver, size=500)
    print(f"数据集生成耗时: {time.time() - t0:.1f}s")

    # 2. 训练 TRM 模型
    print("\n" + "=" * 60)
    print("步骤 2: 训练 TRM 模型")
    print("=" * 60)
    trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)

    t0 = time.time()
    trm_model, history = train_trm_jointly(
        trm_model, dataset, env_train,
        epochs=100, lr=0.001, patience=20, verbose=True
    )
    print(f"训练耗时: {time.time() - t0:.1f}s")

    # 3. 保存模型
    torch.save(trm_model.state_dict(), trm_path)
    print(f"\n模型已保存至 {trm_path}")
    print(f"最佳验证损失: {min(history['val_loss']):.6f}")

    # 4. 快速验证
    print("\n" + "=" * 60)
    print("步骤 3: 快速验证")
    print("=" * 60)
    trm_model.eval()
    from quadrotor_core import PTRMNMPCPredictor

    env = QuadrotorDynamics()
    x_sp = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    predictor = PTRMNMPCPredictor(trm_model, env, K=50, D=16, sigma=0.25,
                                   alpha_blend=0.3, candidate_mode='pd',
                                   ranking_mode='rollout_all', pd_sigma=2.0)

    n_test = 10
    n_success = 0
    for i in range(n_test):
        x_init = torch.tensor([
            np.random.uniform(-0.5, 1.5), np.random.uniform(-1.0, 0.0),
            np.random.uniform(-0.5, 1.5), 0.0, 0.0, 0.0,
        ], dtype=torch.float32)
        predictor.reset()
        x = x_init.clone()
        collision = False

        for step in range(300):
            u_safe, _ = predictor.predict_action(x, x_sp, enable_cbf=True)
            x = env.step_discrete(x, u_safe)
            p_np = x[0:3].detach().numpy()
            for obs in env.obstacles:
                d = np.linalg.norm(p_np - obs['p']) - obs['r']
                if d < 0:
                    collision = True
                    break
            if collision:
                break

        terr = torch.norm(x[0:3] - x_sp[0:3]).item()
        success = (not collision) and (terr < 0.5)
        n_success += int(success)
        print(f"  Trial {i+1}: TErr={terr:.3f}m, Collision={collision}, Success={success}")

    print(f"\n验证成功率: {n_success}/{n_test} ({n_success/n_test*100:.0f}%)")
    print("重训完成!")


if __name__ == '__main__':
    main()
