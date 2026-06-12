# -*- coding: utf-8 -*-
"""
S14: 延迟分解实验 — 将单步推理延迟分解为各组件
Latency decomposition: candidate generation, full rollout evaluation, CBF, overhead
"""

import sys
import os
import time
import json
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from quadrotor_core import (
    QuadrotorDynamics, GoldenNMPCSolver, TRMNMPC,
    PTRMNMPCPredictor, generate_quadrotor_dataset, train_trm_jointly
)

SEED = 2026
N_WARMUP = 10
N_REPEAT = 50

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    set_seed(SEED)
    save_dir = os.path.join(os.path.dirname(__file__), 'results_v6')
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cpu')

    # 加载模型
    trm_path = os.path.join(os.path.dirname(__file__), 'results_v6', 'trm_model.pt')
    if not os.path.exists(trm_path):
        trm_path = os.path.join(os.path.dirname(__file__), 'results', 'trm_model.pt')

    if os.path.exists(trm_path):
        print("加载TRM模型...")
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model.load_state_dict(torch.load(trm_path, map_location=device, weights_only=True))
    else:
        print("训练TRM模型...")
        env_train = QuadrotorDynamics()
        solver = GoldenNMPCSolver(env_train, horizon=10)
        X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        dataset = generate_quadrotor_dataset(env_train, solver, size=500,
                                              x_sp=X_SP,
                                              pos_range=[(-0.5, 1.5), (-1.0, 0.0), (-0.5, 1.5)])
        trm_model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30).to(device)
        trm_model, _ = train_trm_jointly(trm_model, dataset, env_train,
                                           epochs=100, lr=0.001, patience=20, verbose=True)

    trm_model.eval()
    env = QuadrotorDynamics()
    X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    x = torch.tensor([0.5, 0.0, 0.5, 0.3, 0.1, 0.2], dtype=torch.float32)
    x_sp = X_SP

    K_VALUES = [1, 10, 50, 100]
    all_results = {}

    for K in K_VALUES:
        print(f"\n{'=' * 60}")
        print(f"K={K} 延迟分解...")
        print(f"{'=' * 60}")

        predictor = PTRMNMPCPredictor(
            trm_model,
            env,
            K=K,
            candidate_mode='trm_pd',
            ranking_mode='rollout_all',
            use_rollout_cost=True,
            pd_sigma=2.0,
            alpha_blend=0.8,
        )

        # 预热
        for _ in range(N_WARMUP):
            predictor.predict_action(x, x_sp, enable_cbf=True)

        times_gen = []
        times_rollout = []
        times_hysteresis = []
        times_cbf = []
        times_total = []

        for rep in range(N_REPEAT):
            predictor.reset()

            # === 总时间 ===
            t_total_start = time.perf_counter()

            # === 阶段1: 候选生成（proposed: TRM-PD hybrid + action-space Gaussian） ===
            t0 = time.perf_counter()
            with torch.no_grad():
                u_candidates_corrected, _ = predictor._generate_candidates_trm_pd(x, x_sp, device)
            t1 = time.perf_counter()
            times_gen.append((t1 - t0) * 1000)

            # === 阶段2: 全量Rollout评估（proposed: no Q-head screening） ===
            t2_start = time.perf_counter()
            with torch.no_grad():
                if K > 1:
                    u_first_all = u_candidates_corrected[:, 0:3].cpu()
                    rollout_costs = predictor._batch_rollout_cost(x.cpu(), u_first_all, x_sp.cpu())
                    scores = -rollout_costs
                else:
                    scores = torch.tensor([1.0])
            t2_end = time.perf_counter()
            times_rollout.append((t2_end - t2_start) * 1000)

            # === 阶段3: 滞回正则化与最优候选选择 ===
            t_hyst_start = time.perf_counter()
            with torch.no_grad():
                if K > 1 and predictor.last_u_seq is not None:
                    u_shift = torch.cat([predictor.last_u_seq[3:], predictor.last_u_seq[-3:]]).to(device)
                    u_shift_batch = u_shift.unsqueeze(0).repeat(K, 1)
                    dist = torch.sum((u_candidates_corrected - u_shift_batch) ** 2, dim=1).to(scores.device)
                    scores = scores - predictor.eta_hyst * dist

                best_idx = torch.argmax(scores).item()
                best_u_sequence = u_candidates_corrected[best_idx]
                predictor.last_u_seq = best_u_sequence.clone()
                u_nominal = best_u_sequence[0:3].cpu()
            t_hyst_end = time.perf_counter()
            times_hysteresis.append((t_hyst_end - t_hyst_start) * 1000)

            # === 阶段3: CBF投影 ===
            t3_start = time.perf_counter()
            with torch.no_grad():
                u_safe = env.apply_cbf_projection(x.cpu(), u_nominal)
            t3_end = time.perf_counter()
            times_cbf.append((t3_end - t3_start) * 1000)

            t_total_end = time.perf_counter()
            times_total.append((t_total_end - t_total_start) * 1000)

        overhead = np.mean(times_total) - np.mean(times_gen) - np.mean(times_rollout) - np.mean(times_hysteresis) - np.mean(times_cbf)
        result = {
            'K': K,
            'candidate_mode': 'trm_pd',
            'ranking_mode': 'rollout_all',
            'n_warmup': N_WARMUP,
            'n_repeat': N_REPEAT,
            'gen_ms': round(np.mean(times_gen), 2),
            'gen_std': round(np.std(times_gen), 2),
            'rollout_all_ms': round(np.mean(times_rollout), 2),
            'rollout_all_std': round(np.std(times_rollout), 2),
            'hysteresis_select_ms': round(np.mean(times_hysteresis), 3),
            'hysteresis_select_std': round(np.std(times_hysteresis), 3),
            'cbf_ms': round(np.mean(times_cbf), 3),
            'cbf_std': round(np.std(times_cbf), 3),
            'total_ms': round(np.mean(times_total), 2),
            'total_std': round(np.std(times_total), 2),
            'overhead_ms': round(overhead, 2),
        }
        all_results[f"K{K}"] = result

        print(f"  Gen:        {result['gen_ms']:.2f} ± {result['gen_std']:.2f} ms")
        print(f"  Rollout:    {result['rollout_all_ms']:.2f} ± {result['rollout_all_std']:.2f} ms")
        print(f"  Hyst+Sel:   {result['hysteresis_select_ms']:.3f} ± {result['hysteresis_select_std']:.3f} ms")
        print(f"  CBF:        {result['cbf_ms']:.3f} ± {result['cbf_std']:.3f} ms")
        print(f"  Overhead:   {result['overhead_ms']:.2f} ms")
        print(f"  Total:      {result['total_ms']:.2f} ± {result['total_std']:.2f} ms")

    # 保存结果
    output_path = os.path.join(save_dir, 'latency_decomposition_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"延迟分解实验完成! 结果保存到: {output_path}")
    print(f"{'=' * 60}")

    print("\n| K | Gen (ms) | Rollout-all (ms) | Hyst+Select (ms) | CBF (ms) | Overhead (ms) | Total (ms) |")
    print("|---|----------|------------------|------------------|----------|--------------|------------|")
    for K in K_VALUES:
        r = all_results[f"K{K}"]
        print(f"| {K} | {r['gen_ms']:.2f} | {r['rollout_all_ms']:.2f} | "
              f"{r['hysteresis_select_ms']:.3f} | {r['cbf_ms']:.3f} | "
              f"{r['overhead_ms']:.2f} | {r['total_ms']:.2f} |")


if __name__ == '__main__':
    main()
