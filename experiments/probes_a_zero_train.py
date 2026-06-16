#!/usr/bin/env python3
"""
A1/A2/A3/A4 zero-training probes for PTRM advantage.

每个探针都是 PTRMNMPCPredictor 的一个非侵入式 wrapper，不修改主仓库源码：

A1 (mag rescale)    : TRM 输出 first step 乘以 1/mag_scale (默认 1/1.4)。
A2 (full plan rollout): rollout 评估 TRM 完整 30 维 plan (10 步)，而非只 first step。
A3 (hybrid pool)    : K/2 PD+Gauss 候选 + K/2 TRM+Gauss 候选，统一 rollout。
A4 (direction prior): TRM 第一步取方向, 幅度用 PD baseline。

每个 wrapper 实现 predict_action(x, x_sp, enable_cbf=True) -> (u_safe (3,), u_seq (30,))
与原 PTRMNMPCPredictor 接口兼容，可以直接传给 run_trial。

CLI:
  python3 experiments/probes_a_zero_train.py \
      --probes A1,A2,A3,A4 \
      --baselines PD,TRM_only,TRM_PD_a095 \
      --k 1,5,10,20 \
      --n-mc 20 \
      --task narrow \
      --output experiments/results_v6/probes_a_narrow_n20.json
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

from quadrotor_core import QuadrotorDynamics, TRMNMPC                  # noqa: E402
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor             # noqa: E402

from experiments.ptrm_advantage_quick import (                          # noqa: E402
    TASK_FACTORIES, load_trm_model, sample_initial_states,
    set_seed, set_method_seed,
)

DEFAULT_MODEL_PATH = ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"
DEVICE = "cpu"

# 默认 mag scale 来自 trm_failure_diagnostic.md: TRM mag ratio mean ≈ 1.40
DEFAULT_MAG_SCALE = 1.40


# =============================================================================
# Base utilities (借鉴 ptrm_predictor 内部逻辑，不复用以保持探针独立)
# =============================================================================
def pd_first_step(env: QuadrotorDynamics, x: torch.Tensor, x_sp: torch.Tensor,
                  Kp: float = 4.0, Kd: float = 3.0) -> torch.Tensor:
    e_p = x_sp[0:3] - x[0:3]
    e_v = x_sp[3:6] - x[3:6]
    u = env.m * (Kp * e_p + Kd * e_v)
    return torch.clamp(u, env.u_min, env.u_max)


def trm_full_seq(model: TRMNMPC, x: torch.Tensor, x_sp: torch.Tensor, D: int = 16) -> torch.Tensor:
    X = torch.cat([x, x_sp]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y_history = model.forward_steps(X, D=D, noise_scale=0.0, noise_mode="none")
    return y_history[-1][0].squeeze()  # (30,)


def rollout_cost_first_step(env: QuadrotorDynamics, x_init: torch.Tensor,
                            u_first: torch.Tensor, x_sp: torch.Tensor,
                            steps: int = 20, q_diag: torch.Tensor | None = None,
                            R_U: float = 0.02, obs_weight: float = 2000.0,
                            Kp: float = 4.0, Kd: float = 3.0) -> torch.Tensor:
    """Rollout-cost：第1步 = u_first(M,3)，后续 = PD baseline。"""
    if q_diag is None:
        q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
    M = u_first.shape[0]
    x = x_init.unsqueeze(0).repeat(M, 1)
    x_sp6 = x_sp[:6].unsqueeze(0).repeat(M, 1)
    cost = torch.zeros(M)
    q = q_diag.unsqueeze(0)
    for s in range(steps):
        if s == 0:
            u = u_first
        else:
            e_p = x_sp6[:, 0:3] - x[:, 0:3]
            e_v = x_sp6[:, 3:6] - x[:, 3:6]
            u = env.m * (Kp * e_p + Kd * e_v)
        u = torch.clamp(u, env.u_min, env.u_max)
        v = x[:, 3:6]
        v_dot = u / env.m - (env.b_drag / env.m) * v
        p_next = x[:, 0:3] + env.dt * v
        v_next = v + env.dt * v_dot
        x = torch.cat([p_next, v_next], dim=1)
        err = x - x_sp6
        cost = cost + torch.sum(q * err * err, dim=1) + R_U * torch.sum(u * u, dim=1)
        for obs in env.obstacles:
            obs_p = torch.tensor(obs["p"], dtype=torch.float32).unsqueeze(0).repeat(M, 1)
            d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs["r"]
            cost = cost + obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2
    return cost


def rollout_cost_full_seq(env: QuadrotorDynamics, x_init: torch.Tensor,
                          u_seq: torch.Tensor, x_sp: torch.Tensor,
                          extra_steps: int = 10, q_diag: torch.Tensor | None = None,
                          R_U: float = 0.02, obs_weight: float = 2000.0,
                          Kp: float = 4.0, Kd: float = 3.0) -> torch.Tensor:
    """评估 K 个完整 30 维 plan：前 10 步用 u_seq[k]，后续 extra_steps 用 PD."""
    if q_diag is None:
        q_diag = torch.tensor([15.0, 15.0, 15.0, 1.0, 1.0, 1.0])
    K = u_seq.shape[0]
    x = x_init.unsqueeze(0).repeat(K, 1)
    x_sp6 = x_sp[:6].unsqueeze(0).repeat(K, 1)
    cost = torch.zeros(K)
    q = q_diag.unsqueeze(0)
    H = 10
    for s in range(H + extra_steps):
        if s < H:
            u = u_seq[:, s*3:(s+1)*3]
        else:
            e_p = x_sp6[:, 0:3] - x[:, 0:3]
            e_v = x_sp6[:, 3:6] - x[:, 3:6]
            u = env.m * (Kp * e_p + Kd * e_v)
        u = torch.clamp(u, env.u_min, env.u_max)
        v = x[:, 3:6]
        v_dot = u / env.m - (env.b_drag / env.m) * v
        p_next = x[:, 0:3] + env.dt * v
        v_next = v + env.dt * v_dot
        x = torch.cat([p_next, v_next], dim=1)
        err = x - x_sp6
        cost = cost + torch.sum(q * err * err, dim=1) + R_U * torch.sum(u * u, dim=1)
        for obs in env.obstacles:
            obs_p = torch.tensor(obs["p"], dtype=torch.float32).unsqueeze(0).repeat(K, 1)
            d = torch.norm(x[:, 0:3] - obs_p, dim=1) - obs["r"]
            cost = cost + obs_weight * torch.clamp(0.3 - d, min=0.0) ** 2
    return cost


# =============================================================================
# Probe controllers (统一接口：predict_action(x, x_sp, enable_cbf=True))
# =============================================================================
class _BaseProbe:
    def __init__(self, model, env, K, D=16):
        self.model = model
        self.env = env
        self.K = K
        self.D = D

    def reset(self):
        pass

    def _cbf_safe(self, x, u_nom, enable_cbf):
        if enable_cbf:
            u_safe = self.env.apply_cbf_projection(x, u_nom)
        else:
            u_safe = torch.clamp(u_nom, self.env.u_min, self.env.u_max)
        return u_safe


class A1MagRescale(_BaseProbe):
    """TRM 输出 first step 乘以 1/mag_scale，K 个候选 = rescaled TRM + Gaussian。
    rollout: first step only（同 baseline trm_rollout）。"""
    def __init__(self, model, env, K, D=16, mag_scale=DEFAULT_MAG_SCALE, sigma=2.0):
        super().__init__(model, env, K, D)
        self.mag_scale = mag_scale
        self.sigma = sigma

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_trm = trm_full_seq(self.model, x, x_sp, D=self.D)  # (30,)
        u_first = u_trm[0:3] / self.mag_scale  # 校准幅值
        if self.K == 1:
            u_cand = u_first.unsqueeze(0)
        else:
            noise = torch.randn(self.K, 3) * self.sigma
            u_cand = u_first.unsqueeze(0) + noise
        u_cand = torch.clamp(u_cand, self.env.u_min, self.env.u_max)
        costs = rollout_cost_first_step(self.env, x, u_cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = u_cand[best]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_trm.clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


class A2FullPlanRollout(_BaseProbe):
    """K 个完整 30 维 TRM plan + Gaussian on first step，rollout 评估完整 10 步。"""
    def __init__(self, model, env, K, D=16, sigma=2.0, mag_scale=1.0):
        super().__init__(model, env, K, D)
        self.sigma = sigma
        self.mag_scale = mag_scale

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_trm = trm_full_seq(self.model, x, x_sp, D=self.D)  # (30,)
        if self.mag_scale != 1.0:
            u_trm = u_trm.clone()
            u_trm[0:3] = u_trm[0:3] / self.mag_scale
        if self.K == 1:
            u_seq = u_trm.unsqueeze(0)
        else:
            u_seq = u_trm.unsqueeze(0).repeat(self.K, 1)
            # 主要 noise 注入 first step（与 trm_rollout 一致）
            noise = torch.randn(self.K, 3) * self.sigma
            u_seq = u_seq.clone()
            u_seq[:, 0:3] = u_seq[:, 0:3] + noise
            # 后续 9 步加小衰减 noise，避免 K 个完整相同
            for i in range(1, 10):
                decay = max(0.3, 1.0 - i * 0.1)
                step_noise = torch.randn(self.K, 3) * self.sigma * decay
                u_seq[:, i*3:(i+1)*3] = u_seq[:, i*3:(i+1)*3] + step_noise
        u_seq = torch.clamp(u_seq, self.env.u_min, self.env.u_max)
        costs = rollout_cost_full_seq(self.env, x, u_seq, x_sp, extra_steps=10)
        best = int(torch.argmin(costs).item())
        u_nom = u_seq[best, 0:3]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_seq[best].clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


class A3HybridPool(_BaseProbe):
    """K/2 PD+Gauss 候选 + K/2 TRM+Gauss 候选，统一 rollout(first step)。"""
    def __init__(self, model, env, K, D=16, sigma=2.0, mag_scale=1.0):
        super().__init__(model, env, K, D)
        self.sigma = sigma
        self.mag_scale = mag_scale

    def predict_action(self, x, x_sp, enable_cbf=True):
        K_pd = max(1, self.K // 2)
        K_trm = max(1, self.K - K_pd)

        u_pd = pd_first_step(self.env, x, x_sp)  # (3,)
        u_trm_full = trm_full_seq(self.model, x, x_sp, D=self.D)
        u_trm = u_trm_full[0:3] / self.mag_scale

        if K_pd == 1:
            cand_pd = u_pd.unsqueeze(0)
        else:
            cand_pd = u_pd.unsqueeze(0) + torch.randn(K_pd, 3) * self.sigma
        if K_trm == 1:
            cand_trm = u_trm.unsqueeze(0)
        else:
            cand_trm = u_trm.unsqueeze(0) + torch.randn(K_trm, 3) * self.sigma

        cand = torch.cat([cand_pd, cand_trm], dim=0)  # (K, 3)
        cand = torch.clamp(cand, self.env.u_min, self.env.u_max)
        costs = rollout_cost_first_step(self.env, x, cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = cand[best]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_pd.repeat(10).clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


class R1RandomAroundPD(_BaseProbe):
    """Ablation: K/2 PD+Gauss(σ_pd) + K/2 random-around-PD(σ_r)，统一 rollout。
    与 A3 严格对称，只替换 'TRM around mean' 为 'PD around mean'，sigma 可调。
    用以测试 A3 advantage 是否仅来自 candidate diversity（多源 + 大 σ）。
    """
    def __init__(self, model, env, K, D=16, sigma_pd=2.0, sigma_r=2.0):
        super().__init__(model, env, K, D)
        self.sigma_pd = sigma_pd
        self.sigma_r = sigma_r

    def predict_action(self, x, x_sp, enable_cbf=True):
        K_pd = max(1, self.K // 2)
        K_r = max(1, self.K - K_pd)
        u_pd = pd_first_step(self.env, x, x_sp)  # (3,)

        if K_pd == 1:
            cand_pd = u_pd.unsqueeze(0)
        else:
            cand_pd = u_pd.unsqueeze(0) + torch.randn(K_pd, 3) * self.sigma_pd
        # R1: 第二组也是 around-PD，但用大 σ_r 提供 extra diversity
        cand_r = u_pd.unsqueeze(0) + torch.randn(K_r, 3) * self.sigma_r

        cand = torch.cat([cand_pd, cand_r], dim=0)
        cand = torch.clamp(cand, self.env.u_min, self.env.u_max)
        costs = rollout_cost_first_step(self.env, x, cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = cand[best]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_pd.repeat(10).clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


class R0SingleSourceWidePD(_BaseProbe):
    """Single-source-wide control: ALL K candidates = PD + Gaussian(σ=σ_pd).
    一源宽采样对照：检验 R1_s5 的收益是否仅来自 σ 增大、而非两源混合。
    与 PD baseline 区别是 σ_pd 改大（默认 5）。
    与 R1_s5 区别是没有窄 σ 子源（不是 K/2 σ=2 + K/2 σ=5，而是 K 个全部 σ=5）。
    """
    def __init__(self, model, env, K, D=16, sigma_pd=5.0):
        super().__init__(model, env, K, D)
        self.sigma_pd = sigma_pd

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_pd = pd_first_step(self.env, x, x_sp)
        if self.K == 1:
            cand = u_pd.unsqueeze(0)
        else:
            cand = u_pd.unsqueeze(0) + torch.randn(self.K, 3) * self.sigma_pd
        cand = torch.clamp(cand, self.env.u_min, self.env.u_max)
        costs = rollout_cost_first_step(self.env, x, cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = cand[best]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_pd.repeat(10).clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


class R2RandomAroundZero(_BaseProbe):
    """Ablation: K/2 PD+Gauss(σ_pd) + K/2 random-around-zero(σ_r)，统一 rollout。
    worst-case 对照：第二组完全 uninformed，仅作为 'TRM 候选并非必要' 的 null hypothesis。
    """
    def __init__(self, model, env, K, D=16, sigma_pd=2.0, sigma_r=5.0):
        super().__init__(model, env, K, D)
        self.sigma_pd = sigma_pd
        self.sigma_r = sigma_r

    def predict_action(self, x, x_sp, enable_cbf=True):
        K_pd = max(1, self.K // 2)
        K_r = max(1, self.K - K_pd)
        u_pd = pd_first_step(self.env, x, x_sp)

        if K_pd == 1:
            cand_pd = u_pd.unsqueeze(0)
        else:
            cand_pd = u_pd.unsqueeze(0) + torch.randn(K_pd, 3) * self.sigma_pd
        # R2: 第二组 centered at 0（uninformed）
        cand_r = torch.randn(K_r, 3) * self.sigma_r

        cand = torch.cat([cand_pd, cand_r], dim=0)
        cand = torch.clamp(cand, self.env.u_min, self.env.u_max)
        costs = rollout_cost_first_step(self.env, x, cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = cand[best]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_pd.repeat(10).clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


class A4DirectionPrior(_BaseProbe):
    """TRM 第一步方向 + PD 幅度构造主候选，外加 K-1 个 Gaussian 扰动。"""
    def __init__(self, model, env, K, D=16, sigma=2.0):
        super().__init__(model, env, K, D)
        self.sigma = sigma

    def predict_action(self, x, x_sp, enable_cbf=True):
        u_pd = pd_first_step(self.env, x, x_sp)  # (3,)
        u_trm = trm_full_seq(self.model, x, x_sp, D=self.D)[0:3]
        n_trm = torch.norm(u_trm)
        n_pd = torch.norm(u_pd)
        if n_trm < 1e-6:
            u_main = u_pd
        else:
            u_main = (u_trm / n_trm) * n_pd

        if self.K == 1:
            cand = u_main.unsqueeze(0)
        else:
            cand = u_main.unsqueeze(0).repeat(self.K, 1)
            cand[1:] = cand[1:] + torch.randn(self.K - 1, 3) * self.sigma
        cand = torch.clamp(cand, self.env.u_min, self.env.u_max)
        costs = rollout_cost_first_step(self.env, x, cand, x_sp)
        best = int(torch.argmin(costs).item())
        u_nom = cand[best]
        u_safe = self._cbf_safe(x, u_nom, enable_cbf)
        u_seq_30 = u_pd.repeat(10).clone()
        u_seq_30[0:3] = u_safe
        return u_safe, u_seq_30


# =============================================================================
# Baselines (reuse PTRMNMPCPredictor for parity with v6 results)
# =============================================================================
def make_baseline(name: str, model, env, K, D=16, alpha_blend=0.95):
    if name == "PD":
        return PTRMNMPCPredictor(model=model, env=env, K=K, D=D,
                                  candidate_mode="pd", ranking_mode="rollout_all",
                                  alpha_blend=1.0, pd_sigma=2.0)
    if name == "TRM_only":
        return PTRMNMPCPredictor(model=model, env=env, K=K, D=D, sigma=0.0,
                                  candidate_mode="trm_rollout", ranking_mode="rollout_all",
                                  alpha_blend=0.0, pd_sigma=2.0, noise_mode="none")
    if name == "TRM_PD_a095":
        return PTRMNMPCPredictor(model=model, env=env, K=K, D=D,
                                  candidate_mode="trm_pd", ranking_mode="rollout_all",
                                  alpha_blend=alpha_blend, pd_sigma=2.0)
    raise ValueError(f"unknown baseline: {name}")


PROBE_FACTORIES = {
    "A1": lambda model, env, K: A1MagRescale(model, env, K),
    "A2": lambda model, env, K: A2FullPlanRollout(model, env, K),
    "A2_mag": lambda model, env, K: A2FullPlanRollout(model, env, K, mag_scale=DEFAULT_MAG_SCALE),
    "A3": lambda model, env, K: A3HybridPool(model, env, K),
    "A3_mag": lambda model, env, K: A3HybridPool(model, env, K, mag_scale=DEFAULT_MAG_SCALE),
    "A4": lambda model, env, K: A4DirectionPrior(model, env, K),
    # Random ablations: 与 A3 严格对称, 替换 TRM 候选源为 random
    "R1_s1": lambda model, env, K: R1RandomAroundPD(model, env, K, sigma_r=1.0),
    "R1_s2": lambda model, env, K: R1RandomAroundPD(model, env, K, sigma_r=2.0),
    "R1_s3": lambda model, env, K: R1RandomAroundPD(model, env, K, sigma_r=3.0),
    "R1_s5": lambda model, env, K: R1RandomAroundPD(model, env, K, sigma_r=5.0),
    "R1_s8": lambda model, env, K: R1RandomAroundPD(model, env, K, sigma_r=8.0),
    "R2_s2": lambda model, env, K: R2RandomAroundZero(model, env, K, sigma_r=2.0),
    "R2_s5": lambda model, env, K: R2RandomAroundZero(model, env, K, sigma_r=5.0),
    # P0-2 (Round-6): single-source-wide PD ablation.
    "R0_pd_s3": lambda model, env, K: R0SingleSourceWidePD(model, env, K, sigma_pd=3.0),
    "R0_pd_s5": lambda model, env, K: R0SingleSourceWidePD(model, env, K, sigma_pd=5.0),
    "R0_pd_s8": lambda model, env, K: R0SingleSourceWidePD(model, env, K, sigma_pd=8.0),
}


# =============================================================================
# Trial runner
# =============================================================================
def run_trial(controller, env: QuadrotorDynamics, x0: torch.Tensor, x_sp: torch.Tensor,
              n_steps: int, enable_cbf: bool = True) -> dict[str, Any]:
    controller.reset() if hasattr(controller, "reset") else None
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
        "TErr": terr,
        "IAE": iae,
        "success": (terr < 0.30) and (not collided),
        "collided": collided,
        "latency_ms_mean": float(np.mean(lat_list)),
        "latency_ms_median": float(np.median(lat_list)),
        "latency_ms_p95": float(np.percentile(lat_list, 95)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    parser.add_argument("--probes", type=str, default="A1,A2,A3,A4")
    parser.add_argument("--baselines", type=str, default="PD,TRM_only,TRM_PD_a095")
    parser.add_argument("--k", type=str, default="1,5,10,20")
    parser.add_argument("--n-mc", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--mass", type=float, default=1.5,
                        help="quadrotor mass (kg), default 1.5 (nominal)")
    parser.add_argument("--drag", type=float, default=0.1,
                        help="drag coefficient, default 0.1 (nominal)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-cbf", action="store_true",
                        help="disable DT-CCBF safety projection (ablation)")
    args = parser.parse_args()

    set_seed(args.seed)
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    k_values = [int(k) for k in args.k.split(",") if k.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])} for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    model_path = Path(args.model)
    print(f"[probes] loading TRM from {model_path}")
    model = load_trm_model(model_path, DEVICE)
    model.eval()

    results: dict[str, Any] = {
        "meta": {"task": args.task, "seed": args.seed, "n_mc": args.n_mc, "n_steps": args.n_steps,
                 "k_values": k_values, "probes": probes, "baselines": baselines,
                 "model_path": str(model_path),
                 "mass": args.mass, "drag": args.drag,
                 "enable_cbf": (not args.no_cbf),
                 "mag_scale": DEFAULT_MAG_SCALE,
                 "rng_isolation": "per_method_per_trial"},
        "by_method": {},
    }

    methods: list[tuple[str, int, Any]] = []
    for K in k_values:
        for b in baselines:
            methods.append((b, K, "baseline"))
        for p in probes:
            methods.append((p, K, "probe"))

    for method_idx, (name, K_val, kind) in enumerate(methods):
        key = f"{name}_K{K_val}"
        print(f"[probes] {key}")
        rows = []
        for i, x0 in enumerate(inits):
            # 每个方法-试次组合使用独立 RNG 流，消除跨方法泄漏
            set_method_seed(args.seed, name, method_idx, i)
            if kind == "baseline":
                ctrl = make_baseline(name, model, env, K_val)
            else:
                ctrl = PROBE_FACTORIES[name](model, env, K_val)
            r = run_trial(ctrl, env, x0, x_sp, args.n_steps,
                          enable_cbf=(not args.no_cbf))
            rows.append(r)
        agg = aggregate(rows)
        print(f"  succ={agg['success_count']}/{agg['n']} TErr_mean={agg['TErr_mean']:.3f} "
              f"TErr_median={agg['TErr_median']:.3f} IAE={agg['IAE_mean']:.3f}")
        results["by_method"][key] = agg

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[probes] wrote {out_path}")


if __name__ == "__main__":
    main()
