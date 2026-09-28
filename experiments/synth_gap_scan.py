"""P0k: gap STRONG scan -- verify the MDR law is stiffness-controlled, not
gap-controlled, by sweeping the spectral gap (lam3-lam2) over a wide range
while holding stiffness (c-lam2) ~fixed, and re-checking the sqrt stiffness
scaling across hub dials within each gap family.

Why a new graph family (P0h honest boundary #1): in the two-block SBM of
synth_grid/synth_mdr, gap co-moves with lam2 and only spans ~20%. Here:
  4-block HIERARCHICAL SBM (75 x 4): far pairs {1,2}|{3,4} coupled weakly
  (p_far=0.01, w_far=0.3) -> lam2 ~ far-coupling scale (the {12}|{34} split);
  near pairs {1,2} and {3,4} coupled by COUNT (p_near grid, w=1)
  -> gap = lam3-lam2 ~ near-coupling scale, sweepable ~60x while lam2 stays
  put and d_max grows only mildly (clean at w_hub=10/30 where the hub
  dominates c = 2*d_max+1).
Rotation knob (as synth_mdr): ONE cross-far edge of weight delta between
block 1 and block 3 -> continuous theta from 1e-4 rad upward; per-config
exact solve + fresh WarmStartTracker single-step protocol (comparable to
P0h). theta* = first theta with att >= 0.5 (unfreeze threshold, P0h def).

Checks:
  C1 gap invariance: at fixed w_hub, theta* constant across p_near levels
                    (log-log slope of theta* vs gap ~ 0).
  C2 stiffness scaling: within each p_near family, theta* ~ sqrt(c-lam2)
                    with the P0h constant 1.2e-4.

Output: runs/synth_gap_scan.json + printed theta* table.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from trackers import fiedler_exact, WarmStartTracker  # noqa: E402
from synth_grid import laplacian, align_cos  # noqa: E402

N_BLK = 75                     # 4 blocks of 75 -> n=300
P_IN = 0.15
W_IN = 1.0
P_FAR = 0.01
W_FAR = 0.3
M_HUB = 40
P_NEAR_GRID = [0.005, 0.02, 0.08, 0.30]   # gap dial (count, w=1)
W_HUB_GRID = [0.0, 10.0, 30.0]            # stiffness dial
DELTA_GRID = np.logspace(-4, 1, 11)
SEEDS = [0, 1, 2, 3, 4]
Q99_FLOOR_REAL = 8.0e-7
BLK = lambda b: range(b * N_BLK, (b + 1) * N_BLK)   # noqa: E731


def build_hier(rng, p_near, w_hub):
    edges = {}
    for b in range(4):
        for i in BLK(b):
            for j in range(i + 1, (b + 1) * N_BLK):
                if rng.random() < P_IN:
                    edges[(i, j)] = W_IN
    # far coupling: all cross-pair block pairs (1,3),(1,4),(2,3),(2,4)
    for a in (0, 1):
        for b in (2, 3):
            for i in BLK(a):
                for j in BLK(b):
                    if rng.random() < P_FAR:
                        edges[(i, j)] = W_FAR
    # near coupling: within pairs {1,2} and {3,4}, weight 1, count dial
    for a, b in ((0, 1), (2, 3)):
        for i in BLK(a):
            for j in BLK(b):
                if rng.random() < p_near:
                    edges[(i, j)] = W_IN
    n = 4 * N_BLK
    if w_hub > 0:
        hub = n
        for t in rng.choice(n, M_HUB, replace=False):
            edges[(min(hub, int(t)), max(hub, int(t)))] = w_hub
        n += 1
    return edges, n


def main():
    recs = []
    for p_near in P_NEAR_GRID:
        for w_hub in W_HUB_GRID:
            for seed in SEEDS:
                rng = np.random.default_rng(3000 + seed)
                edges0, n = build_hier(rng, p_near, w_hub)
                a = int(rng.choice(list(BLK(0))))
                b = int(rng.choice(list(BLK(2))))
                L0, _ = laplacian(edges0, n)
                v0, vecs0 = fiedler_exact(L0)
                for delta in DELTA_GRID:
                    e1 = dict(edges0)
                    e1[(min(a, b), max(a, b))] = \
                        e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                    L1, deg1 = laplacian(e1, n)
                    v1, vecs1 = fiedler_exact(L1)
                    v2_0, v2_1 = vecs0[:, 1], vecs1[:, 1]
                    cos_rot = align_cos(v2_1, v2_0)
                    d_ex = 1.0 - cos_rot
                    trk = WarmStartTracker(restart_tol_ratio=0.03,
                                           refine_abs_tol=1e-6)
                    trk.update(-1, -1, 0, L0)
                    vh_old = np.array(trk.v2)
                    trk.update(-1, -1, 0, L1)
                    vh_new = np.array(trk.v2)
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    c = 2.0 * float(deg1.max()) + 1.0
                    recs.append({
                        "p_near": p_near, "w_hub": w_hub, "seed": seed,
                        "delta": float(delta),
                        "lam2": float(v1[1]),
                        "lam3": float(v1[2]) if len(v1) > 2 else np.nan,
                        "gap": float(v1[2] - v1[1]) if len(v1) > 2 else np.nan,
                        "c_minus_lam2": c - float(v1[1]),
                        "dmax": float(deg1.max()),
                        "theta": float(np.arccos(np.clip(cos_rot, -1, 1))),
                        "d_ex": d_ex, "d_tr": d_tr,
                        "att": d_tr / max(d_ex, 1e-15),
                        "snr_q99": d_tr / Q99_FLOOR_REAL,
                    })
            print(f"p_near={p_near} w_hub={w_hub} done ({len(recs)} recs)",
                  flush=True)
    with open("../runs/synth_gap_scan.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    # ---- theta* table: first delta (seed-averaged att curve) with att>=0.5
    print("\n== theta* by p_near x w_hub (seed-averaged att curve) ==")
    hdr = f"{'p_near':>7s} {'w_hub':>5s} | " + " ".join(
        f"{d:>8.0e}" for d in DELTA_GRID) + " | gap_med  c-lam2_med  theta*"
    print(hdr)
    summary = []
    for p_near in P_NEAR_GRID:
        for w_hub in W_HUB_GRID:
            gaps, cms, thetas = [], [], []
            att_curve = []
            for delta in DELTA_GRID:
                g = [r for r in recs if r["p_near"] == p_near
                     and r["w_hub"] == w_hub and r["delta"] == float(delta)]
                att_curve.append(float(np.mean([r["att"] for r in g])))
            g0 = [r for r in recs if r["p_near"] == p_near
                  and r["w_hub"] == w_hub and r["delta"] == float(DELTA_GRID[0])]
            gap_med = float(np.median([r["gap"] for r in g0]))
            cm_med = float(np.median([r["c_minus_lam2"] for r in g0]))
            theta_star = next(
                (float(np.mean([r["theta"] for r in recs
                                if r["p_near"] == p_near
                                and r["w_hub"] == w_hub
                                and r["delta"] == float(d)]))
                 for d, at in zip(DELTA_GRID, att_curve) if at >= 0.5),
                np.nan)
            summary.append({"p_near": p_near, "w_hub": w_hub,
                            "gap_med": gap_med, "c_minus_lam2": cm_med,
                            "theta_star": theta_star})
            print(f"{p_near:7.3f} {w_hub:5.0f} | " +
                  " ".join(f"{a:8.2f}" for a in att_curve) +
                  f" | {gap_med:8.2f} {cm_med:11.1f}  {theta_star:.2e}")
    with open("../runs/synth_gap_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
