#!/usr/bin/env python3
"""CasADi+IPOPT NMPC baseline vs probes_a methods — review P0 follow-up.

Reviewers (R2/R3) asked for a modern NLP-based NMPC baseline (acados/HPIPM
or IPOPT) to contextualize the comparison. acados is not installed; we use
CasADi 3.7 + IPOPT (same NLP, slower solver — acados is typically 5-20×
faster on the same problem, but the OPTIMAL value is identical).

Cell: narrow N=80 seed 7777 (matches r0_r1_pd_narrow_n80_s7777.json).

Two CasADi variants:
  - CasADi_H10           horizon 10 (= GoldenNMPCSolver, paper standard)
  - CasADi_H20           horizon 20 (double horizon — fairer to CEM rollout)

Output: experiments/results_v6/casadi_nmpc_narrow_n80_s7777.json + paired
McNemar against R1_s5_K10 / R0_pd_s5_K10 / PD_K10.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quadrotor_core import QuadrotorDynamics                                # noqa: E402
from experiments.baselines.casadi_nmpc_controller import (                  # noqa: E402
    CasADiNMPCController,
)
from experiments.ptrm_advantage_quick import (                              # noqa: E402
    TASK_FACTORIES, sample_initial_states, set_seed, set_method_seed,
)


class CasADiAdapter:
    def __init__(self, env: QuadrotorDynamics, horizon: int = 10):
        self.inner = CasADiNMPCController(env, horizon=horizon)

    def reset(self) -> None:
        self.inner.reset()

    def predict_action(self, x: torch.Tensor, x_sp: torch.Tensor,
                       enable_cbf: bool = True):
        u_safe = self.inner.predict_action(x, x_sp, enable_cbf=enable_cbf)
        return u_safe, None


def run_trial(controller, env: QuadrotorDynamics, x0: torch.Tensor,
              x_sp: torch.Tensor, n_steps: int, enable_cbf: bool = True
              ) -> dict[str, Any]:
    controller.reset()
    x = x0.clone()
    traj = [x.numpy().copy()]
    collided = False
    lat_list = []
    iae = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u_safe, _ = controller.predict_action(x, x_sp, enable_cbf=enable_cbf)
        lat_list.append((time.perf_counter() - t0) * 1000.0)
        u_first = torch.tensor(u_safe.detach().cpu().numpy()[:3],
                               dtype=torch.float32)
        x = env.step_discrete(x, u_first)
        traj.append(x.numpy().copy())
        iae += float(torch.norm(x[:3] - x_sp[:3]).item()) * env.dt
        for o in env.obstacles:
            if float(np.linalg.norm(x.numpy()[:3] - o["p"])) < o["r"]:
                collided = True
    arr = np.array(traj)
    terr = float(np.linalg.norm(arr[-1, :3] - x_sp[:3].numpy()))
    return {
        "TErr": terr, "IAE": iae,
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
        "latency_ms_p95": float(np.percentile(lats, 95)),
        "individual": rows,
    }


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2.0 / (2.0 ** n), 1.0)


def disc(a: list[dict], b: list[dict]) -> tuple[int, int]:
    bb = sum(1 for ra, rb in zip(a, b) if ra["success"] and not rb["success"])
    cc = sum(1 for ra, rb in zip(a, b) if rb["success"] and not ra["success"])
    return bb, cc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="narrow")
    parser.add_argument("--horizons", type=str, default="10,20",
                        help="comma list of horizons to evaluate")
    parser.add_argument("--n-mc", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--mass", type=float, default=1.5)
    parser.add_argument("--drag", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-cbf", action="store_true")
    parser.add_argument("--reference", type=str,
                        default="experiments/results_v6/r0_r1_pd_narrow_n80_s7777.json",
                        help="paired-comparison reference JSON")
    args = parser.parse_args()

    set_seed(args.seed)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    task = TASK_FACTORIES[args.task](args.seed)
    x_sp = task["x_sp"]
    obstacles = [{"p": o["p"].astype(np.float32), "r": float(o["r"])}
                 for o in task["obstacles"]]
    env = QuadrotorDynamics(m=args.mass, b_drag=args.drag, obstacles=obstacles)
    inits = sample_initial_states(task, args.n_mc, args.seed)

    out: dict[str, Any] = {
        "meta": {
            "task": args.task, "seed": args.seed, "n_mc": args.n_mc,
            "n_steps": args.n_steps, "horizons": horizons,
            "mass": args.mass, "drag": args.drag,
            "enable_cbf": (not args.no_cbf),
            "note": "CasADi+IPOPT NMPC (paper-grade NLP solver) — modern reference",
            "rng_isolation": "per_method_per_trial",
        },
        "by_method": {},
        "paired_tests": [],
    }

    for method_idx, h in enumerate(horizons):
        key = f"CasADi_IPOPT_H{h}"
        print(f"[casadi-nmpc] {key}")
        rows = []
        for i, x0 in enumerate(inits):
            set_method_seed(args.seed, key, method_idx, i)
            ctrl = CasADiAdapter(env, horizon=h)
            r = run_trial(ctrl, env, x0, x_sp, args.n_steps,
                          enable_cbf=(not args.no_cbf))
            rows.append(r)
        agg = aggregate(rows)
        print(f"  succ={agg['success_count']}/{agg['n']}  "
              f"TErr_mean={agg['TErr_mean']:.3f}  "
              f"IAE={agg['IAE_mean']:.3f}  "
              f"lat={agg['latency_ms_mean']:.2f}ms (p95={agg['latency_ms_p95']:.2f})")
        out["by_method"][key] = agg

    # Paired McNemar against reference cells
    ref_path = ROOT / args.reference
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        refs = {
            k: ref["by_method"][k]["individual"]
            for k in ("R1_s5_K10", "R0_pd_s5_K10", "PD_K10")
            if k in ref["by_method"]
        }
        print("\n[casadi-nmpc] paired McNemar vs reference (narrow N=80 seed 7777):")
        for h in horizons:
            cas_rows = out["by_method"][f"CasADi_IPOPT_H{h}"]["individual"]
            for ref_key, ref_rows in refs.items():
                bb, cc = disc(cas_rows, ref_rows)
                p = mcnemar_p(bb, cc)
                sig = "***" if p < 0.001 else "**" if p < 0.01 \
                    else "*" if p < 0.05 else "n.s."
                out["paired_tests"].append({
                    "method": f"CasADi_IPOPT_H{h}", "vs": ref_key,
                    "b": bb, "c": cc, "p_exact_two_sided": p,
                })
                print(f"  CasADi_IPOPT_H{h:2d} vs {ref_key:15s}: "
                      f"b={bb}, c={cc}, p={p:.4e} {sig}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[casadi-nmpc] wrote {out_path}")


if __name__ == "__main__":
    main()
