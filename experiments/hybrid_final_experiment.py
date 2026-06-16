# -*- coding: utf-8 -*-
"""
最终版实验：去Q-head架构 (CL-TRM + Pure Rollout)
用于手稿Section 6数据更新

核心架构变更：
1. 去Q-head：纯Rollout评估代替Q-head粗筛+Rollout精排
2. CL-TRM+PD混合基线：闭环训练TRM提供策略先验 + PD反馈稳定性
3. Test-time compute scaling = 鲁棒性提升（非标称性能提升）

对比基线：MPPI, CEM, MLP+CBF, PD+CBF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from quadrotor_core.dynamics import QuadrotorDynamics
from quadrotor_core.trm_network import TRMNMPC
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor
from experiments.baselines.mppi_controller import MPPIController
from experiments.baselines.cem_controller import CEMController
from experiments.baselines.mlp_controller import MLPController, MLPPredictor

# ============================================================
# 配置（与v6对齐）
# ============================================================
CL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_v6', 'cl_trm_model.pt')
MLP_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'experiments', 'results_v6', 'mlp_model.pt')

N_MC = 20
N_STEPS = 300
DT = 0.02
TErr_THRESH = 0.5
SEED = 2026

X_SP = torch.tensor([2.0, 3.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32)

OBSTACLES = [
    {"p": np.array([1.0, 1.0, 1.0]), "r": 0.5},
    {"p": np.array([2.0, 1.5, 2.0]), "r": 0.5},
    {"p": np.array([1.5, 2.2, 1.5]), "r": 0.4}
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_x_init():
    return torch.tensor([
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(-1.0, 0.0),
        np.random.uniform(-0.5, 1.5),
        np.random.uniform(0.0, 0.6),
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.6),
    ], dtype=torch.float32)


def load_cl_trm_model(model_path, device='cpu'):
    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def run_mc_trials(env, controller, x_sp, n_mc=N_MC,
                   use_mismatch=False, process_noise=0.0):
    """
    通用MC试验
    controller 必须有 predict_action(x, x_sp, enable_cbf=True) 方法
    返回值可以是 tuple (u_safe, ...) 或 单个 u_safe
    """
    results = []
    for mc in range(n_mc):
        set_seed(SEED + mc)
        x = random_x_init()
        controller.reset()

        collision = False
        min_dist = float('inf')
        iae = 0.0

        for step in range(N_STEPS):
            result = controller.predict_action(x, x_sp, enable_cbf=True)
            # 兼容 tuple 和 单值返回
            if isinstance(result, tuple):
                u_safe = result[0]
            else:
                u_safe = result

            x = env.step_discrete(x, u_safe,
                                    use_mismatch=use_mismatch,
                                    process_noise=process_noise)

            pos = x[0:3].numpy()
            for obs in env.obstacles:
                d = np.linalg.norm(pos - obs['p']) - obs['r']
                min_dist = min(min_dist, d)
                if d < -0.05:
                    collision = True

            err = (x_sp[0:3] - x[0:3])
            iae += torch.norm(err).item() * DT

        # IAE归一化: 除以总仿真时间 T
        T_actual = N_STEPS * DT
        iae = iae / T_actual if T_actual > 0 else 0.0

        terr = torch.norm(x[0:3] - x_sp[0:3]).item()
        results.append({
            'TErr': terr, 'IAE': iae,
            'success': (not collision) and (terr < TErr_THRESH),
            'collision': collision
        })

    t_errs = [r['TErr'] for r in results]
    iaes = [r['IAE'] for r in results]
    n_success = sum(1 for r in results if r['success'])
    n_collision = sum(1 for r in results if r['collision'])

    return {
        'TErr_mean': np.mean(t_errs), 'TErr_std': np.std(t_errs),
        'IAE_mean': np.mean(iaes), 'IAE_std': np.std(iaes),
        'success_rate': n_success / n_mc * 100,
        'collision_rate': n_collision / n_mc * 100,
    }


def main():
    print("=" * 80)
    print("最终版实验：去Q-head架构 (CL-TRM + Pure Rollout)")
    print(f"设定点: {X_SP.numpy()[:3]}, N_MC={N_MC}, N_STEPS={N_STEPS}")
    print("=" * 80)

    device = 'cpu'
    cl_model = load_cl_trm_model(CL_MODEL_PATH, device)
    print(f"CL-TRM模型: {sum(p.numel() for p in cl_model.parameters() if p.requires_grad)} 参数")

    env = QuadrotorDynamics(obstacles=OBSTACLES)

    # 加载MLP模型
    mlp_model = MLPController(input_dim=12, hidden_dims=(64, 128, 64), output_dim=3)
    if os.path.exists(MLP_MODEL_PATH):
        ckpt = torch.load(MLP_MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            mlp_model.load_state_dict(ckpt['model_state_dict'])
        else:
            mlp_model.load_state_dict(ckpt)
        mlp_model.eval()
        has_mlp = True
        print(f"MLP模型: {sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)} 参数")
    else:
        has_mlp = False
        print("MLP模型未找到，跳过MLP+CBF基线")

    # ============================================================
    # Table 1: PTRM K-Scaling (去Q-head, 纯Rollout)
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 1: PTRM-NMPC Test-Time Compute Scaling (Pure Rollout, No Q-head)")
    print("=" * 80)

    k_values = [1, 10, 50, 100]
    table1 = {}

    for K in k_values:
        predictor = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25 if K > 1 else 0.0, pd_sigma=2.0,
            candidate_mode='trm_pd', alpha_blend=0.5,
            use_rollout_cost=K > 1, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )
        t0 = time.time()
        r = run_mc_trials(env, predictor, X_SP, N_MC)
        elapsed = time.time() - t0
        table1[K] = r
        print(f"  K={K:>3d} | Succ={r['success_rate']:5.0f}% | TErr={r['TErr_mean']:.4f}±{r['TErr_std']:.4f} "
              f"| IAE={r['IAE_mean']:.1f} | 耗时={elapsed:.1f}s")

    # ============================================================
    # Table 2: Baseline Comparison (K=50, Strong CBF)
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 2: Baseline Comparison (K=50, Strong CBF)")
    print("=" * 80)

    # PTRM-NMPC (our method)
    ptrm_pred = PTRMNMPCPredictor(
        model=cl_model, env=env, K=50, D=16,
        sigma=0.25, pd_sigma=2.0,
        candidate_mode='trm_pd', alpha_blend=0.5,
        use_rollout_cost=True, rollout_top_m=10,
        rollout_steps=20, obs_weight=2000.0
    )
    r_ptrm = run_mc_trials(env, ptrm_pred, X_SP, N_MC)
    print(f"  PTRM-NMPC (Ours)  | Succ={r_ptrm['success_rate']:5.0f}% | TErr={r_ptrm['TErr_mean']:.4f}±{r_ptrm['TErr_std']:.4f} | IAE={r_ptrm['IAE_mean']:.1f}")

    # MPPI K=50
    mppi = MPPIController(env, K=50, sigma=2.0, lam=0.1, rollout_steps=20,
                           Kp=4.0, Kd=3.0, obs_weight=2000.0)
    r_mppi = run_mc_trials(env, mppi, X_SP, N_MC)
    print(f"  MPPI K=50         | Succ={r_mppi['success_rate']:5.0f}% | TErr={r_mppi['TErr_mean']:.4f}±{r_mppi['TErr_std']:.4f} | IAE={r_mppi['IAE_mean']:.1f}")

    # CEM K=50 (3 iter)
    cem = CEMController(env, K=50, n_iter=3, sigma=2.0, elite_frac=0.2,
                         rollout_steps=20, Kp=4.0, Kd=3.0, obs_weight=2000.0)
    r_cem = run_mc_trials(env, cem, X_SP, N_MC)
    print(f"  CEM K=50 (3iter)  | Succ={r_cem['success_rate']:5.0f}% | TErr={r_cem['TErr_mean']:.4f}±{r_cem['TErr_std']:.4f} | IAE={r_cem['IAE_mean']:.1f}")

    # MLP+CBF
    r_mlp = None
    if has_mlp:
        mlp_pred = MLPPredictor(mlp_model, env, tracking_Kp=4.0, tracking_Kd=3.0, alpha_blend=0.3)
        r_mlp = run_mc_trials(env, mlp_pred, X_SP, N_MC)
        print(f"  MLP+CBF           | Succ={r_mlp['success_rate']:5.0f}% | TErr={r_mlp['TErr_mean']:.4f}±{r_mlp['TErr_std']:.4f} | IAE={r_mlp['IAE_mean']:.1f}")

    # PD+CBF K=1
    pd_pred = PTRMNMPCPredictor(
        model=cl_model, env=env, K=1, D=16,
        sigma=0.0, pd_sigma=2.0,
        candidate_mode='pd', alpha_blend=0.3,
        use_rollout_cost=False
    )
    r_pd = run_mc_trials(env, pd_pred, X_SP, N_MC)
    print(f"  PD+CBF (K=1)      | Succ={r_pd['success_rate']:5.0f}% | TErr={r_pd['TErr_mean']:.4f}±{r_pd['TErr_std']:.4f} | IAE={r_pd['IAE_mean']:.1f}")

    # ============================================================
    # Table 3: MPPI K-Scaling vs PTRM K-Scaling
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 3: MPPI vs PTRM K-Scaling")
    print("=" * 80)

    for K in [1, 10, 50, 100]:
        # PTRM
        ptrm_k = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25 if K > 1 else 0.0, pd_sigma=2.0,
            candidate_mode='trm_pd', alpha_blend=0.5,
            use_rollout_cost=K > 1, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )
        r_p = run_mc_trials(env, ptrm_k, X_SP, N_MC)

        # MPPI
        mppi_k = MPPIController(env, K=K, sigma=2.0, lam=0.1, rollout_steps=20,
                                 Kp=4.0, Kd=3.0, obs_weight=2000.0)
        r_m = run_mc_trials(env, mppi_k, X_SP, N_MC)

        print(f"  K={K:>3d} | PTRM: {r_p['TErr_mean']:.4f}/{r_p['IAE_mean']:.1f} | "
              f"MPPI: {r_m['TErr_mean']:.4f}/{r_m['IAE_mean']:.1f} | "
              f"PTRM/MPPI TErr比: {r_p['TErr_mean']/max(r_m['TErr_mean'],1e-6):.2f}")

    # ============================================================
    # Table 4: Robustness (失配 + 噪声)
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 4: Robustness Under Parameter Mismatch and Process Noise")
    print("=" * 80)

    robustness_configs = [
        ("Nominal", False, 0.0),
        ("Mass×1.5+Drag×2", True, 0.0),
        ("Noise σ=0.005", False, 0.005),
        ("Noise σ=0.01", False, 0.01),
        ("Noise σ=0.02", False, 0.02),
        ("Noise σ=0.05", False, 0.05),
    ]

    table4 = {}
    for label, mis, noise in robustness_configs:
        # 需要新的predictor实例避免last_u_seq污染
        ptrm_robust = PTRMNMPCPredictor(
            model=cl_model, env=env, K=50, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='trm_pd', alpha_blend=0.5,
            use_rollout_cost=True, rollout_top_m=10,
            rollout_steps=20, obs_weight=2000.0
        )
        r = run_mc_trials(env, ptrm_robust, X_SP, N_MC,
                           use_mismatch=mis, process_noise=noise)
        table4[label] = r
        print(f"  {label:20s} | Succ={r['success_rate']:5.0f}% | TErr={r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | IAE={r['IAE_mean']:.1f}")

    # ============================================================
    # Table 5: Evaluation Ablation (Q+Rollout vs Pure Rollout vs Random)
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 5: Evaluation Strategy Ablation (K=50)")
    print("=" * 80)

    eval_configs = [
        ("Q+Rollout (v6)", 'pd', 0.3, 'q_head', 50),
        ("Pure Rollout", 'pd', 0.3, 'rollout_all', 50),
        ("Random Selection", 'pd', 0.3, 'random', 50),
        ("PD K=1 baseline", 'pd', 0.3, 'q_head', 1),
    ]

    table5 = {}
    for label, mode, alpha, ranking, K in eval_configs:
        pred = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25 if K > 1 else 0.0, pd_sigma=2.0,
            candidate_mode=mode, alpha_blend=alpha,
            use_rollout_cost=K > 1, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0,
            ranking_mode=ranking
        )
        r = run_mc_trials(env, pred, X_SP, N_MC)
        table5[label] = r
        print(f"  {label:25s} | Succ={r['success_rate']:5.0f}% | TErr={r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | IAE={r['IAE_mean']:.1f}")

    # ============================================================
    # Table 6: Candidate Mode Ablation (TRM-PD vs PD vs TRM)
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 6: Candidate Generation Mode Ablation (K=50)")
    print("=" * 80)

    mode_configs = [
        ("CL-TRM+PD(α=0.5)+Rollout", 'trm_pd', 0.5, 50),
        ("CL-TRM+PD(α=0.8)+Rollout", 'trm_pd', 0.8, 50),
        ("PD+PureRollout", 'pd', 0.3, 50),
        ("CL-TRM+Rollout", 'trm_rollout', 0.0, 50),
        ("CL-TRM+Rollout K=1", 'trm_rollout', 0.0, 1),
    ]

    table6 = {}
    for label, mode, alpha, K in mode_configs:
        pred = PTRMNMPCPredictor(
            model=cl_model, env=env, K=K, D=16,
            sigma=0.25 if K > 1 else 0.0, pd_sigma=2.0,
            candidate_mode=mode, alpha_blend=alpha,
            use_rollout_cost=K > 1, rollout_top_m=min(10, K),
            rollout_steps=20, obs_weight=2000.0
        )
        r = run_mc_trials(env, pred, X_SP, N_MC)
        table6[label] = r
        print(f"  {label:30s} | Succ={r['success_rate']:5.0f}% | TErr={r['TErr_mean']:.4f}±{r['TErr_std']:.4f} | IAE={r['IAE_mean']:.1f}")

    # ============================================================
    # Table 7: High Noise Robustness (PD vs TRM+PD, 50% mismatch)
    # ============================================================
    print("\n" + "=" * 80)
    print("Table 7: High Noise Robustness — PD vs TRM+PD (50% mismatch)")
    print("=" * 80)

    table7 = {}
    for noise in [0.0, 0.01, 0.05, 0.1]:
        # PD K=1
        pd1 = PTRMNMPCPredictor(
            model=cl_model, env=env, K=1, D=16,
            sigma=0.0, pd_sigma=2.0,
            candidate_mode='pd', alpha_blend=0.3,
            use_rollout_cost=False
        )
        r_pd1 = run_mc_trials(env, pd1, X_SP, N_MC,
                               use_mismatch=True, process_noise=noise)

        # TRM+PD K=50
        hybrid = PTRMNMPCPredictor(
            model=cl_model, env=env, K=50, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='trm_pd', alpha_blend=0.5,
            use_rollout_cost=True, rollout_top_m=10,
            rollout_steps=20, obs_weight=2000.0
        )
        r_hybrid = run_mc_trials(env, hybrid, X_SP, N_MC,
                                  use_mismatch=True, process_noise=noise)

        # PD K=50 + Rollout
        pd50 = PTRMNMPCPredictor(
            model=cl_model, env=env, K=50, D=16,
            sigma=0.25, pd_sigma=2.0,
            candidate_mode='pd', alpha_blend=0.3,
            use_rollout_cost=True, rollout_top_m=10,
            rollout_steps=20, obs_weight=2000.0,
            ranking_mode='rollout_all'
        )
        r_pd50 = run_mc_trials(env, pd50, X_SP, N_MC,
                                use_mismatch=True, process_noise=noise)

        table7[f'sigma={noise}'] = {
            'PD_K1': r_pd1, 'TRM_PD_K50': r_hybrid, 'PD_K50_Rollout': r_pd50
        }
        print(f"  σ={noise:.2f}: PD K=1={r_pd1['TErr_mean']:.4f} | "
              f"TRM+PD K=50={r_hybrid['TErr_mean']:.4f} | "
              f"PD K=50 Rollout={r_pd50['TErr_mean']:.4f} | "
              f"成功率: {r_pd1['success_rate']:.0f}/{r_hybrid['success_rate']:.0f}/{r_pd50['success_rate']:.0f}%")

    # ============================================================
    # 保存结果
    # ============================================================
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_hybrid')
    os.makedirs(save_dir, exist_ok=True)

    all_results = {
        'table1_k_scaling': {str(k): v for k, v in table1.items()},
        'table2_baselines': {
            'PTRM': r_ptrm, 'MPPI': r_mppi, 'CEM': r_cem, 'PD_K1': r_pd,
        },
        'table4_robustness': table4,
        'table5_eval_ablation': table5,
        'table6_candidate_ablation': table6,
        'table7_noise_robustness': table7,
    }
    if r_mlp is not None:
        all_results['table2_baselines']['MLP'] = r_mlp

    torch.save(all_results, os.path.join(save_dir, 'final_experiment_results.pt'))
    print(f"\n结果已保存到: {save_dir}/final_experiment_results.pt")


if __name__ == '__main__':
    main()
