#!/usr/bin/env python3
"""
TRM-only failure root cause diagnostic.

抽 narrow 10 条 trial，对每条 initial state 同时取得：
  - GoldenNMPC expert u_seq (30,)
  - TRM-only deterministic u_seq
  - PD-only u_seq
然后做三组对比：
  (1) u_seq overlay：逐 step 比较 magnitude / direction / cosine similarity
  (2) open-loop rollout（无 CBF）：用 step_discrete 跑 10 步看 TRM 自己能不能跟踪
  (3) closed-loop with CBF：跑 150 步，记录 CBF intervention 次数、轨迹偏离

输出：
  experiments/results_v6/trm_failure_diagnostic.json    数值
  experiments/results_v6/trm_failure_diagnostic.md      摘要表
  experiments/figures_diag/trm_trial_{i}_u.png          每条 trial 的 u 对比
  experiments/figures_diag/trm_trial_{i}_traj.png       3D 轨迹对比
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core import QuadrotorDynamics, TRMNMPC                  # noqa: E402
from quadrotor_core.nmpc_solver import GoldenNMPCSolver                # noqa: E402
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor             # noqa: E402

from experiments.ptrm_advantage_quick import (                          # noqa: E402
    TASK_FACTORIES, load_trm_model, sample_initial_states, set_seed,
)

OUT_JSON = ROOT / "experiments" / "results_v6" / "trm_failure_diagnostic.json"
OUT_MD = ROOT / "experiments" / "results_v6" / "trm_failure_diagnostic.md"
FIG_DIR = ROOT / "experiments" / "figures_diag"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"
DEVICE = "cpu"
N_TRIALS = 10
TASK_NAME = "narrow"
SEED = 2026
N_CLOSED_STEPS = 150


def pd_u_seq(env: QuadrotorDynamics, x_init: torch.Tensor, x_sp: torch.Tensor, horizon: int = 10) -> np.ndarray:
    """PD baseline expanded to a 30-dim sequence (same convention as PTRMNMPCPredictor._compute_tracking_correction)."""
    Kp, Kd = 4.0, 3.0
    err_p = x_sp[0:3] - x_init[0:3]
    err_v = x_sp[3:6] - x_init[3:6]
    u_pd = (Kp * err_p + Kd * err_v).numpy()
    u_pd = np.clip(u_pd, -env.u_max, env.u_max)
    return np.tile(u_pd, horizon)


def trm_only_u_seq(model: TRMNMPC, x_init: torch.Tensor, x_sp: torch.Tensor, D: int = 16) -> np.ndarray:
    """Deterministic TRM forward (no PD blending, no noise)."""
    X = torch.cat([x_init, x_sp]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y_history = model.forward_steps(X, D=D, noise_scale=0.0, noise_mode="none")
    u_seq = y_history[-1][0].squeeze().cpu().numpy()
    return u_seq


def expert_u_seq(solver: GoldenNMPCSolver, x_init: torch.Tensor, x_sp: torch.Tensor) -> np.ndarray:
    return solver.solve(x_init, x_sp).numpy()


def sequence_metrics(u_ref: np.ndarray, u_test: np.ndarray) -> dict:
    """Per-step magnitude / direction comparison; u shape (30,) = (10 steps, 3 dims)."""
    u_ref_s = u_ref.reshape(10, 3)
    u_test_s = u_test.reshape(10, 3)
    cos = []
    for r, t in zip(u_ref_s, u_test_s):
        nr, nt = np.linalg.norm(r), np.linalg.norm(t)
        if nr < 1e-6 or nt < 1e-6:
            cos.append(float("nan"))
        else:
            cos.append(float(np.dot(r, t) / (nr * nt)))
    return {
        "mag_ref_mean": float(np.linalg.norm(u_ref_s, axis=1).mean()),
        "mag_test_mean": float(np.linalg.norm(u_test_s, axis=1).mean()),
        "mag_ratio_mean": float(np.linalg.norm(u_test_s, axis=1).mean() /
                                max(1e-6, np.linalg.norm(u_ref_s, axis=1).mean())),
        "cos_first_step": cos[0],
        "cos_mean": float(np.nanmean(cos)),
        "cos_min": float(np.nanmin(cos)),
        "l2_step1": float(np.linalg.norm(u_test_s[0] - u_ref_s[0])),
        "l2_full": float(np.linalg.norm(u_test - u_ref)),
    }


def open_loop_rollout(env: QuadrotorDynamics, x0: torch.Tensor, u_seq: np.ndarray) -> dict:
    """Apply 10-step u_seq deterministically (no CBF, no recompute)."""
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    for i in range(10):
        u = torch.tensor(u_seq[i*3:(i+1)*3], dtype=torch.float32)
        x = env.step_discrete(x, u)
        traj.append(x.numpy().copy())
        # check obstacles
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    traj_np = np.array(traj)
    return {"trajectory": traj_np, "collided": collided}


def closed_loop_test(env: QuadrotorDynamics, x0: torch.Tensor, x_sp: torch.Tensor,
                     get_u_first, n_steps: int = N_CLOSED_STEPS) -> dict:
    """通用闭环：每步调 get_u_first(x) -> u_nominal(3,)，过 CBF，记录 intervention 次数。"""
    x = x0.clone()
    traj = [x.numpy().copy()]
    n_cbf_intervene = 0
    collided = False
    for _ in range(n_steps):
        u_nom = get_u_first(x)
        u_nom_t = torch.tensor(u_nom, dtype=torch.float32)
        u_safe = env.apply_cbf_projection(x, u_nom_t)
        diff = float(torch.norm(u_safe - u_nom_t).item())
        if diff > 0.05:
            n_cbf_intervene += 1
        x = env.step_discrete(x, u_safe)
        traj.append(x.numpy().copy())
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    traj_np = np.array(traj)
    terminal_err = float(np.linalg.norm(traj_np[-1, :3] - x_sp[:3].numpy()))
    success = (terminal_err < 0.30) and (not collided)
    return {
        "trajectory": traj_np,
        "terminal_error": terminal_err,
        "collided": collided,
        "success": success,
        "n_cbf_intervene": n_cbf_intervene,
    }


def plot_u_compare(idx: int, u_exp: np.ndarray, u_trm: np.ndarray, u_pd: np.ndarray, out_path: Path):
    u_exp_s = u_exp.reshape(10, 3)
    u_trm_s = u_trm.reshape(10, 3)
    u_pd_s = u_pd.reshape(10, 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    labels = ["u_x", "u_y", "u_z"]
    steps = np.arange(10)
    for dim in range(3):
        ax = axes[dim]
        ax.plot(steps, u_exp_s[:, dim], "k-o", label="Expert (L-BFGS)", lw=2, ms=4)
        ax.plot(steps, u_trm_s[:, dim], "r--s", label="TRM-only", lw=1.5, ms=4)
        ax.plot(steps, u_pd_s[:, dim], "b:^", label="PD", lw=1.2, ms=4)
        ax.set_title(labels[dim])
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("control (N)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"trial {idx}: control sequence comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_traj_compare(idx: int, x_sp: torch.Tensor, env: QuadrotorDynamics,
                      traj_exp: np.ndarray, traj_trm: np.ndarray, traj_pd: np.ndarray, out_path: Path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for o in env.obstacles:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        xs = o["p"][0] + o["r"] * np.outer(np.cos(u), np.sin(v))
        ys = o["p"][1] + o["r"] * np.outer(np.sin(u), np.sin(v))
        zs = o["p"][2] + o["r"] * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="gray", alpha=0.2, linewidth=0)
    ax.plot(traj_exp[:, 0], traj_exp[:, 1], traj_exp[:, 2], "k-", lw=2, label="Expert closed-loop")
    ax.plot(traj_trm[:, 0], traj_trm[:, 1], traj_trm[:, 2], "r--", lw=2, label="TRM-only closed-loop")
    ax.plot(traj_pd[:, 0], traj_pd[:, 1], traj_pd[:, 2], "b:", lw=1.5, label="PD closed-loop")
    ax.scatter(*traj_exp[0, :3], c="green", s=60, label="start")
    ax.scatter(*x_sp[:3].numpy(), c="purple", s=80, marker="*", label="goal")
    ax.set_title(f"trial {idx}: closed-loop trajectories (narrow)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    set_seed(SEED)
    print(f"[diag] task={TASK_NAME} n_trials={N_TRIALS} steps={N_CLOSED_STEPS}")

    task = TASK_FACTORIES[TASK_NAME](SEED)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])} for o in task["obstacles"]]
    env = QuadrotorDynamics(obstacles=obstacles)

    inits = sample_initial_states(task, N_TRIALS, SEED)

    print(f"[diag] loading TRM from {MODEL_PATH}")
    model = load_trm_model(MODEL_PATH, DEVICE)
    model.eval()
    solver = GoldenNMPCSolver(env, horizon=10)

    # 用于闭环测试的 controller 工厂
    def trm_only_predictor():
        # K=1 + trm_rollout + sigma=0 → 确定性 TRM 输出，无任何噪声 / PD blending
        return PTRMNMPCPredictor(
            model=model, env=env, K=1, D=16, sigma=0.0,
            candidate_mode="trm_rollout", ranking_mode="rollout_all",
            alpha_blend=0.0, pd_sigma=0.0, noise_mode="none",
        )

    def pd_predictor():
        return PTRMNMPCPredictor(
            model=model, env=env, K=1, D=16,
            candidate_mode="pd", ranking_mode="rollout_all",
            alpha_blend=1.0,
        )

    records = []
    for i, x0 in enumerate(inits):
        print(f"[diag] trial {i}/{N_TRIALS-1} x0={x0.numpy().round(3).tolist()}")

        u_exp = expert_u_seq(solver, x0, x_sp)
        u_trm = trm_only_u_seq(model, x0, x_sp)
        u_pd = pd_u_seq(env, x0, x_sp)

        seq_trm = sequence_metrics(u_exp, u_trm)
        seq_pd = sequence_metrics(u_exp, u_pd)

        # 开环 10 步
        ol_exp = open_loop_rollout(env, x0, u_exp)
        ol_trm = open_loop_rollout(env, x0, u_trm)
        ol_pd = open_loop_rollout(env, x0, u_pd)

        # 闭环 150 步（带 CBF）
        ctrl_trm = trm_only_predictor()
        ctrl_pd = pd_predictor()

        def get_u_trm(x):
            u_safe, _ = ctrl_trm.predict_action(x, x_sp, enable_cbf=True)
            return u_safe.detach().cpu().numpy().reshape(-1)[:3]

        def get_u_pd(x):
            u_safe, _ = ctrl_pd.predict_action(x, x_sp, enable_cbf=True)
            return u_safe.detach().cpu().numpy().reshape(-1)[:3]

        def get_u_expert(x):
            u = solver.solve(x, x_sp).numpy()[:3]
            u_t = torch.tensor(u, dtype=torch.float32)
            u_safe = env.apply_cbf_projection(x, u_t)
            return u_safe.detach().cpu().numpy().reshape(-1)[:3]

        cl_exp = closed_loop_test(env, x0, x_sp, get_u_expert)
        cl_trm = closed_loop_test(env, x0, x_sp, get_u_trm)
        cl_pd = closed_loop_test(env, x0, x_sp, get_u_pd)

        # 画图
        plot_u_compare(i, u_exp, u_trm, u_pd, FIG_DIR / f"trm_trial_{i}_u.png")
        plot_traj_compare(i, x_sp, env, cl_exp["trajectory"], cl_trm["trajectory"], cl_pd["trajectory"],
                          FIG_DIR / f"trm_trial_{i}_traj.png")

        # failure mode 判定（仅对 TRM）
        mode = []
        if seq_trm["mag_ratio_mean"] > 1.5 or seq_trm["mag_ratio_mean"] < 0.5:
            mode.append("magnitude")
        if seq_trm["cos_first_step"] < 0.3:
            mode.append("direction(first)")
        if seq_trm["cos_mean"] < 0.3:
            mode.append("direction(avg)")
        if cl_trm["n_cbf_intervene"] > N_CLOSED_STEPS * 0.5:
            mode.append("infeasible(CBF>50%)")
        if cl_trm["collided"]:
            mode.append("collided")
        if not mode:
            mode.append("other")

        records.append({
            "trial": i,
            "x0": x0.numpy().tolist(),
            "seq_trm_vs_expert": seq_trm,
            "seq_pd_vs_expert": seq_pd,
            "open_loop_collided_trm": ol_trm["collided"],
            "open_loop_collided_pd": ol_pd["collided"],
            "open_loop_collided_exp": ol_exp["collided"],
            "closed_loop_trm": {k: cl_trm[k] for k in ["terminal_error", "collided", "success",
                                                       "n_cbf_intervene"]},
            "closed_loop_pd": {k: cl_pd[k] for k in ["terminal_error", "collided", "success",
                                                     "n_cbf_intervene"]},
            "closed_loop_exp": {k: cl_exp[k] for k in ["terminal_error", "collided", "success",
                                                       "n_cbf_intervene"]},
            "trm_failure_mode": mode,
        })

    # 写 JSON
    OUT_JSON.write_text(json.dumps({"task": TASK_NAME, "n_trials": N_TRIALS,
                                    "n_closed_steps": N_CLOSED_STEPS,
                                    "model_path": str(MODEL_PATH), "records": records}, indent=2),
                        encoding="utf-8")

    # 写 markdown 摘要
    md = ["# TRM-only failure diagnostic", ""]
    md.append(f"task=**{TASK_NAME}**, n_trials={N_TRIALS}, closed-loop steps={N_CLOSED_STEPS}, model={MODEL_PATH.name}")
    md.append("")
    md.append("## Per-trial summary")
    md.append("| trial | mag(TRM)/mag(exp) | cos_first(TRM,exp) | cos_mean | CL TRM succ | CL TRM TErr | CBF interv. | failure mode |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        s = r["seq_trm_vs_expert"]
        c = r["closed_loop_trm"]
        md.append(f"| {r['trial']} | {s['mag_ratio_mean']:.2f} | {s['cos_first_step']:.2f} | "
                  f"{s['cos_mean']:.2f} | {c['success']} | {c['terminal_error']:.3f} | "
                  f"{c['n_cbf_intervene']}/{N_CLOSED_STEPS} | {', '.join(r['trm_failure_mode'])} |")
    md.append("")

    # PD/Expert 闭环对照
    md.append("## Reference closed-loop (same trials, same CBF)")
    md.append("| trial | PD succ | PD TErr | Expert succ | Expert TErr |")
    md.append("|---|---|---|---|---|")
    for r in records:
        cp = r["closed_loop_pd"]; ce = r["closed_loop_exp"]
        md.append(f"| {r['trial']} | {cp['success']} | {cp['terminal_error']:.3f} | "
                  f"{ce['success']} | {ce['terminal_error']:.3f} |")
    md.append("")

    # aggregate
    succ_trm = sum(1 for r in records if r["closed_loop_trm"]["success"])
    succ_pd = sum(1 for r in records if r["closed_loop_pd"]["success"])
    succ_exp = sum(1 for r in records if r["closed_loop_exp"]["success"])
    mag_ratios = [r["seq_trm_vs_expert"]["mag_ratio_mean"] for r in records]
    cos_firsts = [r["seq_trm_vs_expert"]["cos_first_step"] for r in records]
    md.append("## Aggregates")
    md.append(f"- TRM closed-loop success: {succ_trm}/{len(records)}")
    md.append(f"- PD  closed-loop success: {succ_pd}/{len(records)}")
    md.append(f"- Expert closed-loop success: {succ_exp}/{len(records)}")
    md.append(f"- TRM mag ratio mean={np.mean(mag_ratios):.2f} std={np.std(mag_ratios):.2f}")
    md.append(f"- TRM cos_first mean={np.mean(cos_firsts):.2f} std={np.std(cos_firsts):.2f}")
    md.append("")
    md.append("Figures: `experiments/figures_diag/trm_trial_*_{u,traj}.png`")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[diag] wrote {OUT_JSON}")
    print(f"[diag] wrote {OUT_MD}")
    print("\n" + "\n".join(md[-7:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
