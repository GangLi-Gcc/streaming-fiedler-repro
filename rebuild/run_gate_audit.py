#!/usr/bin/env python3
"""rebuild/run_gate_audit.py — crossing-level audit of the restart gate.

For every MTR crossing (first delta with att>=0.5) on the 4-block hierarchical
SBM rig, report the quantities the gate actually uses and decompose the
warm-start residual:

  - branch: 2 = direct gate restart (res > gate), 1 = refine then
    verification-triggered restart, 0 = refine accepted (no restart)
  - res:      the deployed warm-start residual  ||L'v - lam*v||,  lam = v^T L' v
              (the Rayleigh quotient, i.e. the minimiser over the line; the code
              does NOT use the previous eigenvalue here)
  - res_prev: ||L'v - lam_old*v|| = ||d bb^T v|| = sqrt(2)|d| |v_i - v_j|  at the
              previous exact eigenvalue lam_old = lam2(L0)  (reviewer's check)
  - res_exact: ||L'v - lam2'*v|| = sin(theta)*sigma_w  (rotation-only residual at
              the exact updated eigenvalue, eq:resfactor)
  - lam:      v^T L' v  (the RQ the code gates on)
  - lam2_old, lam2_new: exact pre/post Fiedler eigenvalue
  - Delta:    lam2_new - lam2_old  (eigenvalue-mismatch term)
  - gap_est:  max(lam3_old - lam, 1e-9)  (the tracker's gap estimate)
  - gate:     0.03 * max(gap_est, gate_floor)
  - res3:     verification residual after refinement (NaN for branch 2)
  - k_used:   refinement steps actually taken (0 for branch 2)
  - theta, bracket (theta_lo, theta_hi), att*, capture*, d_end*

Output: rebuild/results/gate_audit.json (all deltas) + gate_audit_crossings.json
(crossing rows only) + a printed partition summary.
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import fiedler_exact, WarmStartTracker, align_cos
from run_mdr import build_hier, laplacian, BLK, N_BLK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

P_NEAR_GRID = [0.005, 0.02, 0.08, 0.30]
W_HUB_GRID = [0.0, 10.0, 30.0]
DELTA_GRID = np.logspace(-4, 1, 11)
SEEDS = [0, 1, 2, 3, 4]


def main():
    all_rows = []
    t0 = __import__("time").time()
    for p_near in P_NEAR_GRID:
        for w_hub in W_HUB_GRID:
            for seed in SEEDS:
                rng = np.random.default_rng(3000 + seed)
                edges0, n = build_hier(rng, p_near, w_hub)
                a = int(rng.choice(list(BLK(0))))
                b = int(rng.choice(list(BLK(2))))
                L0, _ = laplacian(edges0, n)
                vals0, vecs0 = fiedler_exact(L0)
                lam2_old = float(vals0[1])
                lam3_old = float(vals0[2]) if len(vals0) > 2 else lam2_old * 2 + 1e-3
                v_old = vecs0[:, 1].copy()
                for delta in DELTA_GRID:
                    e1 = dict(edges0)
                    e1[(min(a, b), max(a, b))] = \
                        e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                    L1, _ = laplacian(e1, n)
                    vals1, vecs1 = fiedler_exact(L1)
                    lam2_new = float(vals1[1])
                    v_new = vecs1[:, 1].copy()
                    theta = float(np.arccos(np.clip(align_cos(v_new, v_old), -1, 1)))
                    d_ex = 1.0 - align_cos(v_new, v_old)
                    # -- exact rotation-only residual at the updated eigenvalue --
                    r_exact_vec = L1 @ v_old - lam2_new * v_old
                    res_exact = float(np.linalg.norm(r_exact_vec))
                    sin_th = abs(float(np.sin(theta))) if theta < np.pi else 0.0
                    sigma_w = res_exact / sin_th if sin_th > 1e-15 else 0.0
                    # -- residual at the previous eigenvalue (reviewer's check) --
                    r_prev_vec = L1 @ v_old - lam2_old * v_old
                    res_prev = float(np.linalg.norm(r_prev_vec))
                    # -- the Rayleigh quotient the code actually gates on --
                    lam = float(v_old @ (L1 @ v_old))
                    res = float(np.linalg.norm(L1 @ v_old - lam * v_old))
                    gap_est = max(lam3_old - lam, 1e-9)
                    gate = 0.03 * max(gap_est, 0.0)
                    Delta = lam2_new - lam2_old
                    # -- run the tracker, capture its branch diagnostics --
                    trk = WarmStartTracker(restart_tol_ratio=0.03,
                                           refine_abs_tol=1e-6)
                    trk.update(L0)
                    vh_old = trk.v2.copy()
                    trk.update(L1)
                    vh_new = trk.v2.copy()
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    d_end = 1.0 - align_cos(vh_new, v_new)
                    att = d_tr / max(d_ex, 1e-15)
                    capture = 1.0 - d_end / max(d_ex, 1e-15)
                    (branch, res_d, gap_d, k_used, res3, lam2_after, dmax) = trk.diag[-1]
                    all_rows.append({
                        "p_near": p_near, "w_hub": w_hub, "seed": seed,
                        "delta": float(delta), "theta": theta,
                        "theta_deg": float(np.degrees(theta)),
                        "branch": int(branch), "res": res, "res_prev": res_prev,
                        "res_exact": res_exact, "sigma_w": sigma_w,
                        "lam": lam, "lam2_old": lam2_old, "lam2_new": lam2_new,
                        "Delta": Delta, "gap_est": gap_est, "gate": gate,
                        "res3": res3, "k_used": int(k_used),
                        "att": att, "capture": capture, "d_end": d_end,
                    })
            print(f"p_near={p_near} w_hub={w_hub} seed={seed} done "
                  f"({len(all_rows)} rows)", flush=True)

    with open(os.path.join(RESULT_DIR, "gate_audit.json"), "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=1)

    # ---- identify crossings (first delta with att >= 0.5) and emit bracket rows ----
    crossings = []
    for p_near in P_NEAR_GRID:
        for w_hub in W_HUB_GRID:
            for seed in SEEDS:
                rs = sorted([r for r in all_rows if r["p_near"] == p_near
                             and r["w_hub"] == w_hub and r["seed"] == seed],
                            key=lambda r: r["delta"])
                # left-censored if att >= 0.5 already at smallest delta
                if rs[0]["att"] >= 0.5 and rs[0]["theta"] > 1e-12:
                    continue
                for k in range(len(rs)):
                    if rs[k]["att"] >= 0.5 and rs[k]["theta"] > 1e-12:
                        r = dict(rs[k])
                        r["theta_lo"] = rs[k - 1]["theta"] if k > 0 else 0.0
                        r["theta_hi"] = r["theta"]
                        crossings.append(r)
                        break
    with open(os.path.join(RESULT_DIR, "gate_audit_crossings.json"),
              "w", encoding="utf-8") as f:
        json.dump(crossings, f, indent=1)

    # ---- partition summary ----
    n2 = sum(1 for r in crossings if r["branch"] == 2)
    n1 = sum(1 for r in crossings if r["branch"] == 1)
    n0 = sum(1 for r in crossings if r["branch"] == 0)
    print(f"\n=== gate audit: {len(crossings)} bracketed crossings ===")
    print(f"  direct-gate restart (branch 2): {n2}")
    print(f"  verification-triggered restart (branch 1): {n1}")
    print(f"  refine-only crossing (branch 0): {n0}")
    for r in crossings:
        if r["branch"] == 0:
            print(f"    refine-only: p={r['p_near']} w={r['w_hub']} s={r['seed']} "
                  f"att={r['att']:.4f}")
    # mismatch vs rotation at crossings
    print("\n  eigenvalue-mismatch vs rotation at crossings:")
    mis = []
    for r in crossings:
        rot = r["res_exact"]          # sin(theta) sigma_w
        dmis = abs(r["Delta"]) * abs(np.cos(r["theta"]))  # Delta cos(theta)
        ratio = dmis / rot if rot > 1e-15 else float("inf")
        mis.append(ratio)
        print(f"    p={r['p_near']} w={r['w_hub']} s={r['seed']} br={r['branch']} "
              f"th={r['theta']:.2e} rot={rot:.3e} Delta*cos={dmis:.3e} "
              f"ratio={ratio:.2e} res={r['res']:.3e} gate={r['gate']:.3e}")
    mis = np.array(mis)
    print(f"    mismatch/rotation ratio: median {np.median(mis):.3e}, "
          f"max {mis.max():.3e}, min {mis.min():.3e}")
    print(f"    # crossings with ratio > 0.1: {int(np.sum(mis > 0.1))}")
    print(f"    # crossings with ratio > 0.5: {int(np.sum(mis > 0.5))}")


if __name__ == "__main__":
    main()
