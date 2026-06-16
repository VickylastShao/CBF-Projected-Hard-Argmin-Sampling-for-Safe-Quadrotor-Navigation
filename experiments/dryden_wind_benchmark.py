#!/usr/bin/env python3
"""Dryden wind-disturbance benchmark.

Replaces the dimensionless `process_noise` with a Dryden-model wind
gust added as an external force on the 6D translational dynamics.

We use the MIL-F-8785C low-altitude Dryden longitudinal/lateral
spectra with first-order shaping filters driven by unit-variance
white Gaussian noise. The body-frame turbulent velocity components
u_g, v_g, w_g (m/s) are sampled at the simulation step dt = 0.02 s
and converted to a disturbance force F_d = -b_drag * v_g (N) by the
linear-drag model that already governs the nominal dynamics. This
keeps the disturbance physically interpretable.

Parameters (low-altitude, light turbulence):
  - V (airspeed used to scale spectra) = 5 m/s
  - L_u = L_v = 200 m, L_w = 50 m  (turbulence scale lengths)
  - sigma_u = sigma_v = 0.5 m/s, sigma_w = 0.3 m/s (RMS gust velocity)

Cells:
  - narrow N=40 seed 7777 with Dryden intensities {nominal, 2x, 5x}
  - Methods: PD K=10, R0_pd_s5 K=10, R1_s5 K=10

Output: experiments/results_v6/dryden_wind_narrow_n40_s7777.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from math import comb
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core import QuadrotorDynamics                                # noqa: E402
from experiments.probes_a_zero_train import (                                # noqa: E402
    PROBE_FACTORIES, make_baseline,
)
from experiments.ptrm_advantage_quick import (                              # noqa: E402
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
)


class DrydenWind:
    """Three-axis Dryden gust filter (first-order shaping)."""

    def __init__(self, dt: float, V: float = 5.0,
                 sigma_uvw: tuple[float, float, float] = (0.5, 0.5, 0.3),
                 L_uvw: tuple[float, float, float] = (200.0, 200.0, 50.0),
                 rng: np.random.Generator = None):
        self.dt = dt
        self.V = max(V, 0.5)  # guard against division by zero
        self.sigma = np.array(sigma_uvw)
        self.L = np.array(L_uvw)
        self.rng = rng or np.random.default_rng()
        # First-order Dryden time constants tau_i = L_i / V
        self.tau = self.L / self.V
        # Discrete-time autoregressive coefficient a = exp(-dt/tau)
        self.a = np.exp(-self.dt / self.tau)
        # Innovation std so steady-state variance = sigma^2
        self.innov = self.sigma * np.sqrt(1.0 - self.a ** 2)
        # State (m/s)
        self.v_gust = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.v_gust = np.zeros(3, dtype=np.float32)

    def step(self) -> np.ndarray:
        # v_{k+1} = a * v_k + innov * w_k,  w_k ~ N(0, 1)
        w = self.rng.standard_normal(3)
        self.v_gust = (self.a * self.v_gust + self.innov * w).astype(np.float32)
        return self.v_gust.copy()


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2.0 / (2.0 ** n), 1.0)


def disc(a, b):
    bb = sum(1 for ra, rb in zip(a, b) if ra["success"] and not rb["success"])
    cc = sum(1 for ra, rb in zip(a, b) if rb["success"] and not ra["success"])
    return bb, cc


def run_dryden_trial(controller, env: QuadrotorDynamics, wind: DrydenWind,
                     x0: torch.Tensor, x_sp: torch.Tensor,
                     n_steps: int, enable_cbf: bool = True) -> dict:
    """Run one trial with Dryden gust adding F_d = -b_drag * v_gust force."""
    controller.reset()
    wind.reset()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    lat_list = []
    iae = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        lat_list.append((time.perf_counter() - t0) * 1000.0)
        u_first = u_safe[:3] if u_safe.numel() == 3 else torch.tensor(
            u_safe.detach().cpu().numpy()[:3], dtype=torch.float32)
        # Dryden gust as external force: F_d = -b_drag * v_gust.
        v_gust = wind.step()
        F_d = -env.b_drag * v_gust
        u_total = u_first + torch.tensor(F_d, dtype=torch.float32)
        x = env.step_discrete(x, u_total)
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt
        for o in env.obstacles:
            d = float(np.linalg.norm(x.numpy()[:3] - o["p"]))
            if d < o["r"]:
                collided = True
    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    return {
        "TErr": terr if not collided else 10.0,
        "IAE": iae if not collided else 20.0,
        "success": (terr < 0.30) and (not collided),
        "collided": collided,
        "latency_ms_mean": float(np.mean(lat_list)),
    }


def aggregate(rows):
    n = len(rows)
    succ = sum(1 for r in rows if r["success"])
    coll = sum(1 for r in rows if r["collided"])
    terrs = np.array([r["TErr"] for r in rows])
    iaes = np.array([r["IAE"] for r in rows])
    return {
        "n": n, "success_count": succ, "collisions": coll,
        "success_rate": succ / n,
        "TErr_mean": float(terrs.mean()),
        "IAE_mean": float(iaes.mean()),
        "latency_ms_mean": float(np.mean([r["latency_ms_mean"] for r in rows])),
        "individual": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensities", type=str, default="1.0,2.0,5.0",
                        help="Dryden sigma multiplier (1.0 = MIL-F-8785C light)")
    parser.add_argument("--V", type=float, default=5.0,
                        help="airspeed (m/s) for Dryden spectra")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-mc", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--model", type=str,
                        default="experiments/results_v6/cl_trm_model.pt")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-cbf", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    intensities = [float(s) for s in args.intensities.split(",") if s.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    inits_all = sample_initial_states(task, args.n_mc, args.seed)

    from quadrotor_core import TRMNMPC
    model = TRMNMPC()
    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    else:
        model.load_state_dict(sd)
    model.eval()

    out = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "K": args.k,
            "V_airspeed_mps": args.V,
            "intensity_multipliers": intensities,
            "dryden_base_sigma_mps": [0.5, 0.5, 0.3],
            "dryden_L_m": [200.0, 200.0, 50.0],
            "mass": args.mass, "drag": args.drag,
            "enable_cbf": (not args.no_cbf),
            "note": ("Dryden first-order longitudinal/lateral gust filters "
                     "applied as F_d = -b_drag * v_gust external force"),
            "rng_isolation": "per_method_per_trial",
        },
        "by_intensity": {},
        "paired_tests": [],
    }

    for scale in intensities:
        sigma_uvw = (0.5 * scale, 0.5 * scale, 0.3 * scale)
        rng = np.random.default_rng(args.seed + int(1000 * scale))
        print(f"\n=== Dryden sigma scale = {scale:.2f}x "
              f"(sigma = {sigma_uvw}) ===")
        cell = {}
        for method_idx, method in enumerate(["PD", "R0_pd_s5", "R1_s5"]):
            print(f"  [{method}]", end=" ", flush=True)
            rows = []
            for i, x0 in enumerate(inits_all):
                set_method_seed(args.seed, f"{method}_dryden_{scale}", method_idx, i)
                env = QuadrotorDynamics(
                    m=args.mass, b_drag=args.drag,
                    obstacles=deepcopy(obstacles))
                if method == "PD":
                    ctrl = make_baseline("PD", model, env, args.k)
                else:
                    ctrl = PROBE_FACTORIES[method](model, env, args.k)
                wind = DrydenWind(env.dt, V=args.V, sigma_uvw=sigma_uvw,
                                  rng=rng)
                r = run_dryden_trial(ctrl, env, wind, x0, x_sp,
                                     args.n_steps,
                                     enable_cbf=(not args.no_cbf))
                rows.append(r)
            agg = aggregate(rows)
            cell[method] = agg
            print(f"succ={agg['success_count']}/{agg['n']} "
                  f"coll={agg['collisions']} TErr={agg['TErr_mean']:.3f} "
                  f"IAE={agg['IAE_mean']:.3f}")

        cell_tests = []
        for a, b in [("R1_s5", "PD"), ("R0_pd_s5", "PD"),
                     ("R0_pd_s5", "R1_s5")]:
            bb, cc = disc(cell[a]["individual"], cell[b]["individual"])
            p = mcnemar_p(bb, cc)
            cell_tests.append({
                "scale": scale, "method_a": a, "method_b": b,
                "succ_a": cell[a]["success_count"],
                "succ_b": cell[b]["success_count"],
                "b": bb, "c": cc, "p_exact_two_sided": p,
            })
            sig = "***" if p < 0.001 else "**" if p < 0.01 \
                else "*" if p < 0.05 else "n.s."
            print(f"    {a:10s} vs {b:10s}: b={bb}, c={cc}, p={p:.4e} {sig}")
        out["by_intensity"][f"x{scale:.2f}"] = cell
        out["paired_tests"].extend(cell_tests)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
