# -*- coding: utf-8 -*-
"""
一键运行全部 PTRM-NMPC 实验

用法:
    python run_all_experiments.py [--skip-training] [--trials N] [--steps N]

选项:
    --skip-training   跳过训练阶段（需已有训练好的模型）
    --trials N        蒙特卡洛试验次数 (默认 200)
    --steps N         每次试验仿真步数 (默认 100)
"""

import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser(description='Run all PTRM-NMPC experiments')
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase')
    parser.add_argument('--trials', type=int, default=200, help='Number of Monte Carlo trials')
    parser.add_argument('--steps', type=int, default=100, help='Simulation steps per trial')
    args = parser.parse_args()

    print("=" * 60)
    print("PTRM-NMPC 全流程实验脚本")
    print("=" * 60)

    total_start = time.time()

    # Step 1: Training
    if not args.skip_training:
        print("\n[1/4] 训练 PTRM-NMPC 模型...")
        print("-" * 40)
        from quadrotor_core import (
            QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
            generate_quadrotor_dataset, train_trm_jointly,
        )
        import torch
        import numpy as np

        SEED = 2026
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        env = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env, horizon=10)
        model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
        dataset = generate_quadrotor_dataset(env, solver, size=500)
        model, history = train_trm_jointly(model, dataset, env, epochs=100, patience=15, verbose=True)

        # 保存模型
        import os
        os.makedirs('experiments/results', exist_ok=True)
        torch.save(model.state_dict(), 'experiments/results/trm_model_best.pt')
        print("模型已保存至 experiments/results/trm_model_best.pt")
    else:
        print("\n[1/4] 跳过训练阶段")
        print("-" * 40)

    # Step 2: Core experiments (original)
    print("\n[2/4] 运行核心实验 (原始宽通道)...")
    print("-" * 40)
    import quadrotor_core_simulation
    # The module's main() is inside if __name__ == "__main__", so we call the function directly
    # Re-import to ensure clean state
    import importlib
    importlib.reload(quadrotor_core_simulation)
    # We need to set up env and model first
    from quadrotor_core import QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC
    import torch, numpy as np
    SEED = 2026
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = QuadrotorDynamics()
    solver = GoldenNMPCSolver(env, horizon=10)
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    from quadrotor_core import generate_quadrotor_dataset, train_trm_jointly
    dataset = generate_quadrotor_dataset(env, solver, size=500)
    model, _ = train_trm_jointly(model, dataset, env, epochs=100, patience=15, verbose=False)
    metrics, latencies, cost_lowerings, expert_latency = quadrotor_core_simulation.run_monte_carlo_experiments(
        env, solver, model, num_trials=args.trials, sim_steps=args.steps
    )

    # Step 3: Revised experiments (narrow corridor)
    print("\n[3/4] 运行修正实验 (窄通道, 公平对比)...")
    print("-" * 40)
    import quadrotor_core_simulation_v2
    importlib.reload(quadrotor_core_simulation_v2)
    from quadrotor_core_simulation_v2 import NarrowCorridorDynamics, run_revised_experiments, plot_revised_results
    env_narrow = NarrowCorridorDynamics()
    env_wide = QuadrotorDynamics()
    solver_narrow = GoldenNMPCSolver(env_narrow, horizon=10)
    solver_wide = GoldenNMPCSolver(env_wide, horizon=10)
    # Retrain with wide corridor data for the revised experiments
    model_v2 = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    dataset_v2 = generate_quadrotor_dataset(env_wide, solver_wide, size=500)
    model_v2, _ = train_trm_jointly(model_v2, dataset_v2, env_wide, epochs=100, patience=15, verbose=False)
    results_v2 = run_revised_experiments(
        env_narrow, env_wide, solver_narrow, solver_wide,
        model_v2, num_trials=args.trials, sim_steps=args.steps
    )
    plot_revised_results(results_v2, env_narrow, env_wide)

    # Step 4: 3D trajectory visualization
    print("\n[4/4] 生成3D轨迹可视化...")
    print("-" * 40)
    import plot_3d_trajectories
    # This script runs standalone; we'll note it can be run separately
    print("3D轨迹图请单独运行: python plot_3d_trajectories.py")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"全部实验完成！总耗时: {total_time/60:.1f} 分钟")
    print("结果保存在: experiments/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
