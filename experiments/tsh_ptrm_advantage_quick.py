# -*- coding: utf-8 -*-
"""
PTRM advantage 快速验证实验。

目标不是挑选单个有利样例，而是在固定 procedural task family 上快速筛查：
PTRM/TRM 候选先验是否在低 K、非凸/多同伦/局部贪心陷阱任务中表现出
相对 PD+Gaussian+Rollout+CBF 的候选效率优势。

默认实验较小，适合作为 smoke/validation；正式结果应提高 --n-mc 和 --n-steps。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrotor_core.dynamics import QuadrotorDynamics
from quadrotor_core.ptrm_predictor import PTRMNMPCPredictor
from quadrotor_core.trm_network import TRMNMPC
from experiments.baselines.cem_controller import CEMController

DEFAULT_MODEL_PATH = ROOT / "experiments" / "results_v6" / "cl_trm_model.pt"
DEFAULT_OUTPUT_PATH = ROOT / "experiments" / "results_v6" / "ptrm_advantage_quick.json"
TErr_THRESH = 0.3
DT = 0.02


METHODS = [
    "PD+Rollout",
    "TRM+Rollout",
    "TRM+PD+Rollout",
    "CEM",
]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    _ = torch.manual_seed(seed)


def derive_method_seed(base_seed: int, method_name: str, method_idx: int,
                       trial_idx: int) -> int:
    """为每个 (method, trial) 组合派生独立种子，消除跨方法 RNG 泄漏。

    使用 SHA-256 确保派生种子之间统计独立，即使 method_name 或
    method_idx 相近也不会产生可检测的相关性。
    """
    import hashlib
    tag = f"{base_seed}:{method_name}:{method_idx}:{trial_idx}"
    h = hashlib.sha256(tag.encode()).digest()
    return int.from_bytes(h[:4], "big")


def set_method_seed(base_seed: int, method_name: str, method_idx: int,
                    trial_idx: int) -> int:
    """派生并设置 per-method per-trial 的全局 RNG 种子。"""
    seed = derive_method_seed(base_seed, method_name, method_idx, trial_idx)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    return obj


def load_trm_model(model_path: Path, device: str) -> TRMNMPC:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = TRMNMPC(input_dim=12, latent_dim=64, mpc_horizon=30)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(f"不支持的 checkpoint 类型: {type(checkpoint)!r}")

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        keys = list(state_dict.keys())[:20] if isinstance(state_dict, dict) else []
        raise RuntimeError(f"模型加载失败，前20个key={keys}") from exc

    model.to(device)
    model.eval()
    return model


def clearance(x: torch.Tensor, obstacles: list[dict[str, Any]]) -> float:
    pos = x[0:3].detach().cpu().numpy()
    return float(min(np.linalg.norm(pos - obs["p"]) - obs["r"] for obs in obstacles))


def is_state_clear(x: torch.Tensor, obstacles: list[dict[str, Any]], margin: float = 0.1) -> bool:
    return clearance(x, obstacles) > margin


def sample_initial_states(task: dict[str, Any], n_mc: int, seed: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    states: list[torch.Tensor] = []
    attempts = 0
    while len(states) < n_mc and attempts < 1000:
        attempts += 1
        x = task["init_sampler"](rng)
        if is_state_clear(x, task["obstacles"], margin=0.05):
            states.append(x)
    if len(states) < n_mc:
        raise RuntimeError(f"任务 {task['name']} 无法采样足够初始状态: {len(states)}/{n_mc}")
    return states


def task_two_gate(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z0 = 1.35 + rng.uniform(-0.05, 0.05)
    # 用两组球体形成中部屏障，保留 y 低/高两个通道。
    obstacles = [
        {"p": np.array([1.25, 1.15, z0]), "r": 0.42},
        {"p": np.array([1.75, 1.50, z0 + 0.05]), "r": 0.45},
        {"p": np.array([2.20, 1.85, z0 - 0.05]), "r": 0.42},
        {"p": np.array([1.45, 2.35, z0 + 0.10]), "r": 0.36},
        {"p": np.array([2.05, 0.65, z0 - 0.10]), "r": 0.36},
    ]
    x_sp = torch.tensor([3.0, 3.0, z0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def init_sampler(local_rng: np.random.Generator) -> torch.Tensor:
        return torch.tensor([
            local_rng.uniform(-0.15, 0.25),
            local_rng.uniform(-0.20, 0.20),
            z0 + local_rng.uniform(-0.12, 0.12),
            local_rng.uniform(0.0, 0.25),
            local_rng.uniform(0.0, 0.25),
            local_rng.uniform(-0.05, 0.05),
        ], dtype=torch.float32)

    return {
        "name": "two_gate",
        "obstacles": obstacles,
        "x_sp": x_sp,
        "init_sampler": init_sampler,
        "config": {"family": "two_gate", "seed": seed, "z0": z0},
    }


def task_narrow(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    jitter = lambda scale: rng.uniform(-scale, scale, size=3)
    obstacles = [
        {"p": np.array([0.8, 0.8, 0.8]) + jitter(0.03), "r": 0.55},
        {"p": np.array([1.5, 1.8, 1.5]) + jitter(0.03), "r": 0.50},
        {"p": np.array([2.2, 1.2, 2.2]) + jitter(0.03), "r": 0.50},
        {"p": np.array([1.2, 2.5, 1.0]) + jitter(0.03), "r": 0.45},
        {"p": np.array([2.0, 2.0, 2.5]) + jitter(0.03), "r": 0.45},
    ]
    x_sp = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def init_sampler(local_rng: np.random.Generator) -> torch.Tensor:
        return torch.tensor([
            local_rng.uniform(-0.20, 0.20),
            local_rng.uniform(-0.20, 0.20),
            local_rng.uniform(-0.20, 0.20),
            local_rng.uniform(0.0, 0.25),
            local_rng.uniform(0.0, 0.25),
            local_rng.uniform(0.0, 0.25),
        ], dtype=torch.float32)

    return {
        "name": "narrow",
        "obstacles": obstacles,
        "x_sp": x_sp,
        "init_sampler": init_sampler,
        "config": {"family": "narrow", "seed": seed},
    }


def task_u_shape(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z0 = 1.25 + rng.uniform(-0.05, 0.05)
    # 球形障碍近似 U 形墙，开口朝起点侧，要求控制器先横向绕行。
    obstacles = [
        {"p": np.array([1.15, 1.00, z0]), "r": 0.38},
        {"p": np.array([1.55, 1.00, z0]), "r": 0.38},
        {"p": np.array([1.95, 1.00, z0]), "r": 0.38},
        {"p": np.array([1.15, 1.45, z0]), "r": 0.36},
        {"p": np.array([1.15, 1.90, z0]), "r": 0.36},
        {"p": np.array([1.95, 1.45, z0]), "r": 0.36},
        {"p": np.array([1.95, 1.90, z0]), "r": 0.36},
    ]
    x_sp = torch.tensor([2.6, 2.55, z0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def init_sampler(local_rng: np.random.Generator) -> torch.Tensor:
        return torch.tensor([
            local_rng.uniform(0.0, 0.25),
            local_rng.uniform(0.75, 1.15),
            z0 + local_rng.uniform(-0.12, 0.12),
            local_rng.uniform(0.0, 0.20),
            local_rng.uniform(0.0, 0.20),
            local_rng.uniform(-0.05, 0.05),
        ], dtype=torch.float32)

    return {
        "name": "u_shape",
        "obstacles": obstacles,
        "x_sp": x_sp,
        "init_sampler": init_sampler,
        "config": {"family": "u_shape", "seed": seed, "z0": z0},
    }


TASK_FACTORIES: dict[str, Callable[[int], dict[str, Any]]] = {
    "two_gate": task_two_gate,
    "narrow": task_narrow,
    "u_shape": task_u_shape,
}


def make_controller(method: str, model: TRMNMPC, env: QuadrotorDynamics, K: int, D: int,
                    args: argparse.Namespace):
    sigma = args.sigma if K > 1 else 0.0
    pd_sigma = args.pd_sigma if K > 1 else 0.0

    if method == "PD+Rollout":
        return PTRMNMPCPredictor(
            model=model, env=env, K=K, D=D, sigma=sigma,
            candidate_mode="pd", pd_sigma=pd_sigma, alpha_blend=args.alpha_blend,
            use_rollout_cost=True, ranking_mode="rollout_all",
            rollout_steps=args.rollout_steps, obs_weight=args.obs_weight,
        )
    if method == "TRM+Rollout":
        return PTRMNMPCPredictor(
            model=model, env=env, K=K, D=D, sigma=sigma,
            candidate_mode="trm_rollout", pd_sigma=pd_sigma, alpha_blend=0.0,
            use_rollout_cost=True, ranking_mode="rollout_all",
            rollout_steps=args.rollout_steps, obs_weight=args.obs_weight,
        )
    if method == "TRM+PD+Rollout":
        return PTRMNMPCPredictor(
            model=model, env=env, K=K, D=D, sigma=sigma,
            candidate_mode="trm_pd", pd_sigma=pd_sigma, alpha_blend=args.alpha_blend,
            use_rollout_cost=True, ranking_mode="rollout_all",
            rollout_steps=args.rollout_steps, obs_weight=args.obs_weight,
        )
    if method == "CEM":
        return CEMController(
            env=env, K=K, n_iter=args.cem_n_iter, sigma=pd_sigma,
            rollout_steps=args.rollout_steps, obs_weight=args.obs_weight,
        )
    raise ValueError(f"未知方法: {method}")


def compute_candidate_quality(controller, x_init: torch.Tensor, x_sp: torch.Tensor):
    if not isinstance(controller, PTRMNMPCPredictor):
        return None

    model = controller.model
    device = next(model.parameters()).device
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    try:
        with torch.no_grad():
            mode = controller.candidate_mode
            if mode == "pd":
                candidates, _ = controller._generate_candidates_pd(x_init, x_sp, device)
            elif mode == "trm_rollout":
                candidates, _ = controller._generate_candidates_trm_rollout(x_init, x_sp, device)
            elif mode == "trm_pd":
                candidates, _ = controller._generate_candidates_trm_pd(x_init, x_sp, device)
            elif mode == "trm_v2":
                candidates, _ = controller._generate_candidates_trm_v2(x_init, x_sp, device)
            else:
                candidates, _ = controller._generate_candidates_trm(x_init, x_sp, device)
            costs = controller._batch_rollout_cost(x_init.cpu(), candidates[:, 0:3].cpu(), x_sp.cpu())
        return {
            "best_rollout_cost": float(torch.min(costs).item()),
            "mean_rollout_cost": float(torch.mean(costs).item()),
            "std_rollout_cost": float(torch.std(costs).item()) if costs.numel() > 1 else 0.0,
        }
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)


def run_trial(controller, env: QuadrotorDynamics, x_init: torch.Tensor, x_sp: torch.Tensor,
              n_steps: int, enable_cbf: bool, method: str) -> dict[str, Any]:
    if hasattr(controller, "reset"):
        controller.reset()

    x = x_init.clone()
    collision = False
    min_dist = float("inf")
    iae = 0.0
    latencies = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        if isinstance(controller, PTRMNMPCPredictor):
            u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        else:
            u_safe = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        x = env.step_discrete(x, u_safe)
        d = clearance(x, env.obstacles)
        min_dist = min(min_dist, d)
        if d < 0.0:
            collision = True
        iae += torch.norm(x_sp[0:3] - x[0:3]).item() * DT

    terr = torch.norm(x[0:3] - x_sp[0:3]).item()
    lat = np.array(latencies, dtype=float)
    return {
        "method": method,
        "TErr": float(terr),
        "IAE": float(iae / max(n_steps * DT, 1e-9)),
        "success": bool((not collision) and terr < TErr_THRESH),
        "collision": bool(collision),
        "min_dist": float(min_dist),
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
    }


def aggregate_trials(individual: list[dict[str, Any]], candidate_quality: list[dict[str, Any] | None]) -> dict[str, Any]:
    def arr(key: str) -> np.ndarray[Any, Any]:
        return np.array([r[key] for r in individual], dtype=float)

    t_err = arr("TErr")
    iae = arr("IAE")
    min_dist = arr("min_dist")
    lat_mean = arr("latency_ms_mean")
    lat_median = arr("latency_ms_median")
    lat_p95 = arr("latency_ms_p95")
    success = np.array([r["success"] for r in individual], dtype=bool)
    collision = np.array([r["collision"] for r in individual], dtype=bool)

    quality_valid = [q for q in candidate_quality if q is not None]
    if quality_valid:
        best_cost = np.array([q["best_rollout_cost"] for q in quality_valid], dtype=float)
        mean_cost = np.array([q["mean_rollout_cost"] for q in quality_valid], dtype=float)
        std_cost = np.array([q["std_rollout_cost"] for q in quality_valid], dtype=float)
        quality_summary = {
            "candidate_best_rollout_cost_mean": float(np.mean(best_cost)),
            "candidate_best_rollout_cost_std": float(np.std(best_cost)),
            "candidate_mean_rollout_cost_mean": float(np.mean(mean_cost)),
            "candidate_std_rollout_cost_mean": float(np.mean(std_cost)),
        }
    else:
        quality_summary = {
            "candidate_best_rollout_cost_mean": None,
            "candidate_best_rollout_cost_std": None,
            "candidate_mean_rollout_cost_mean": None,
            "candidate_std_rollout_cost_mean": None,
        }

    return {
        "n_mc": len(individual),
        "success_rate": float(np.mean(success) * 100.0),
        "collision_rate": float(np.mean(collision) * 100.0),
        "terminal_error_mean": float(np.mean(t_err)),
        "terminal_error_std": float(np.std(t_err)),
        "iae_mean": float(np.mean(iae)),
        "iae_std": float(np.std(iae)),
        "min_distance_mean": float(np.mean(min_dist)),
        "min_distance_min": float(np.min(min_dist)),
        "latency_ms_mean": float(np.mean(lat_mean)),
        "latency_ms_median": float(np.mean(lat_median)),
        "latency_ms_p95": float(np.mean(lat_p95)),
        **quality_summary,
        "individual": individual,
        "candidate_quality_individual": candidate_quality,
    }


def run_combo(task: dict[str, Any], method: str, model: TRMNMPC, K: int, D: int,
              init_states: list[torch.Tensor], args: argparse.Namespace) -> dict[str, Any]:
    env = QuadrotorDynamics(obstacles=task["obstacles"])
    controller = make_controller(method, model, env, K, D, args)
    individual = []
    quality = []

    for idx, x_init in enumerate(init_states):
        set_seed(args.seed + 100000 * idx + 1000 * K + 10 * D + METHODS.index(method))
        if hasattr(controller, "reset"):
            controller.reset()
        q = compute_candidate_quality(controller, x_init, task["x_sp"])
        quality.append(q)
        trial = run_trial(
            controller=controller,
            env=env,
            x_init=x_init,
            x_sp=task["x_sp"],
            n_steps=args.n_steps,
            enable_cbf=bool(args.enable_cbf),
            method=method,
        )
        individual.append(trial)

    out = aggregate_trials(individual, quality)
    out["effective_rollouts"] = int(K * args.cem_n_iter) if method == "CEM" else int(K)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="PTRM advantage 快速验证实验")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tasks", type=str, default="two_gate,narrow,u_shape")
    parser.add_argument("--k-values", type=str, default="1,5,10")
    parser.add_argument("--d-values", type=str, default="16")
    parser.add_argument("--n-mc", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--enable-cbf", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--pd-sigma", type=float, default=2.0)
    parser.add_argument("--alpha-blend", type=float, default=0.3)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--obs-weight", type=float, default=2000.0)
    parser.add_argument("--cem-n-iter", type=int, default=3)
    args = parser.parse_args()

    tasks = parse_csv_strings(args.tasks)
    k_values = parse_csv_ints(args.k_values)
    d_values = parse_csv_ints(args.d_values)

    unknown = [name for name in tasks if name not in TASK_FACTORIES]
    if unknown:
        raise ValueError(f"未知任务: {unknown}; 可用任务={list(TASK_FACTORIES)}")

    set_seed(args.seed)
    model = load_trm_model(args.model_path, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {
        "meta": {
            "seed": args.seed,
            "model_path": str(args.model_path),
            "n_mc": args.n_mc,
            "n_steps": args.n_steps,
            "k_values": k_values,
            "d_values": d_values,
            "methods": METHODS,
            "enable_cbf": bool(args.enable_cbf),
            "sigma": args.sigma,
            "pd_sigma": args.pd_sigma,
            "alpha_blend": args.alpha_blend,
            "rollout_steps": args.rollout_steps,
            "obs_weight": args.obs_weight,
            "cem_n_iter": args.cem_n_iter,
            "note": "candidate quality is diagnostic and excluded from control latency",
        },
        "tasks": {},
    }

    for task_idx, task_name in enumerate(tasks):
        print(f"\n=== Task: {task_name} ===")
        task_seed = args.seed + task_idx * 10000
        task = TASK_FACTORIES[task_name](task_seed)
        init_states = sample_initial_states(task, args.n_mc, task_seed + 123)
        output["tasks"][task_name] = {
            "config": {
                **task["config"],
                "obstacles": task["obstacles"],
                "x_sp": task["x_sp"],
                "initial_states": init_states,
            },
            "results": {},
        }

        for D in d_values:
            d_key = f"D{D}"
            output["tasks"][task_name]["results"][d_key] = {}
            for K in k_values:
                k_key = f"K{K}"
                output["tasks"][task_name]["results"][d_key][k_key] = {}
                print(f"  D={D}, K={K}")
                for method in METHODS:
                    print(f"    - {method}", flush=True)
                    result = run_combo(task, method, model, K, D, init_states, args)
                    output["tasks"][task_name]["results"][d_key][k_key][method] = result
                    print(
                        f"      succ={result['success_rate']:.1f}% "
                        f"TErr={result['terminal_error_mean']:.3f} "
                        f"IAE={result['iae_mean']:.3f} "
                        f"lat={result['latency_ms_median']:.2f}ms"
                    )
                    args.output.write_text(json.dumps(to_jsonable(output), indent=2), encoding="utf-8")

    args.output.write_text(json.dumps(to_jsonable(output), indent=2), encoding="utf-8")
    print(f"\n结果已保存: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
