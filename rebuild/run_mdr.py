#!/usr/bin/env python3
"""rebuild/run_mdr.py — re-run the 43-configuration MDR rig (paper Sec. 4).

Protocol (identical to synth_gap_scan.py, single-step fresh tracker per delta):
  4-block hierarchical SBM (75 x 4 = 300 nodes), P_IN=0.15 W_IN=1, P_FAR=0.01
  W_FAR=0.3; near-pair coupling p_near in [0.005,0.02,0.08,0.30] is the gap dial,
  hub weight w_hub in [0,10,30] is the stiffness dial, one cross-far edge of
  weight delta in logspace(-4,1,11) is the rotation dial.
  For each (p_near, w_hub, seed, delta): exact (lam2, v2) before/after, then a
  FRESH WarmStartTracker seeded on L0 does one update on L1.  att = d_tr/d_ex.

Output rebuild/results/mdr_configs.json (raw records) and a k-band summary.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import fiedler_exact, WarmStartTracker, align_cos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

N_BLK = 75
P_IN = 0.15
W_IN = 1.0
P_FAR = 0.01
W_FAR = 0.3
M_HUB = 40
P_NEAR_GRID = [0.005, 0.02, 0.08, 0.30]
W_HUB_GRID = [0.0, 10.0, 30.0]
DELTA_GRID = np.logspace(-4, 1, 11)
SEEDS = [0, 1, 2, 3, 4]
BLK = lambda b: range(b * N_BLK, (b + 1) * N_BLK)


def build_hier(rng, p_near, w_hub):
    edges = {}
    for b in range(4):
        for i in BLK(b):
            for j in range(i + 1, (b + 1) * N_BLK):
                if rng.random() < P_IN:
                    edges[(i, j)] = W_IN
    for a in (0, 1):
        for b in (2, 3):
            for i in BLK(a):
                for j in BLK(b):
                    if rng.random() < P_FAR:
                        edges[(i, j)] = W_FAR
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


def laplacian(edges, n):
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for (a, b), w in edges.items():
        deg[a] += w; deg[b] += w
        rows += [a, b]; cols += [b, a]; vals += [-w, -w]
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(deg[i])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)), deg


def main():
    recs = []
    t0 = time.time()
    for p_near in P_NEAR_GRID:
        for w_hub in W_HUB_GRID:
            for seed in SEEDS:
                rng = np.random.default_rng(3000 + seed)
                edges0, n = build_hier(rng, p_near, w_hub)
                a = int(rng.choice(list(BLK(0))))
                b = int(rng.choice(list(BLK(2))))
                L0, _ = laplacian(edges0, n)
                vals0, vecs0 = fiedler_exact(L0)
                for delta in DELTA_GRID:
                    e1 = dict(edges0)
                    e1[(min(a, b), max(a, b))] = \
                        e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                    L1, deg1 = laplacian(e1, n)
                    vals1, vecs1 = fiedler_exact(L1)
                    d_ex = 1.0 - align_cos(vecs1[:, 1], vecs0[:, 1])
                    trk = WarmStartTracker(restart_tol_ratio=0.03,
                                           refine_abs_tol=1e-6)
                    trk.update(L0)
                    vh_old = trk.v2.copy()
                    trk.update(L1)
                    vh_new = trk.v2.copy()
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    # direction fidelity: how far the tracked vector ends from the
                    # EXACT post-update Fiedler vector v1 (not from where it started).
                    d_end = 1.0 - align_cos(vh_new, vecs1[:, 1])
                    # directional capture ratio: fraction of the true rotation
                    # actually closed. 1 = perfect tracking, 0 = no movement
                    # toward v1, <0 = movement away from v1.
                    capture_dir = 1.0 - d_end / max(d_ex, 1e-15)
                    c = 2.0 * float(deg1.max()) + 1.0
                    recs.append({
                        "p_near": p_near, "w_hub": w_hub, "seed": seed,
                        "delta": float(delta),
                        "lam2": float(vals1[1]),
                        "gap": float(vals1[2] - vals1[1]),
                        "stiff": c - float(vals1[1]),
                        "theta": float(np.arccos(np.clip(align_cos(vecs1[:, 1], vecs0[:, 1]), -1, 1))),
                        "d_ex": d_ex, "d_tr": d_tr, "d_end": d_end,
                        "att": d_tr / max(d_ex, 1e-15),
                        "capture_dir": capture_dir,
                    })
        print(f"p_near={p_near} w_hub={w_hub} done "
              f"({len(recs)} recs, {time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(RESULT_DIR, "mdr_configs.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    # ---- k-band summary (per-config theta_star / sqrt_stiff) ----
    # Two definitions side by side:
    #   att-based      : first theta with d_tr/d_ex >= 0.5 (movement magnitude)
    #   capture-based  : first theta with capture_dir >= 0.5 (directional fidelity)
    kb = []
    for p_near in P_NEAR_GRID:
        for w_hub in W_HUB_GRID:
            for seed in SEEDS:
                rs = [r for r in recs if r["p_near"] == p_near
                      and r["w_hub"] == w_hub and r["seed"] == seed]
                rs = sorted(rs, key=lambda r: r["theta"])
                th_star_att = th_star_cap = None
                for r in rs:
                    if th_star_att is None and r["att"] >= 0.5 and r["theta"] > 0:
                        th_star_att = r["theta"]
                    if th_star_cap is None and r["capture_dir"] >= 0.5 and r["theta"] > 0:
                        th_star_cap = r["theta"]
                stiff = rs[0]["stiff"]
                kb.append({"p_near": p_near, "w_hub": w_hub, "seed": seed,
                           "theta_star": th_star_cap, "theta_star_att": th_star_att,
                           "sqrt_stiff": np.sqrt(stiff),
                           "k": th_star_cap / np.sqrt(stiff) if th_star_cap else None,
                           "k_att": th_star_att / np.sqrt(stiff) if th_star_att else None})
    # config medians
    cfgs = sorted({(r["p_near"], r["w_hub"]) for r in kb})
    print(f"\n=== k-band over {len(cfgs)} config medians (5 seeds each) ===")
    cfg_med = []
    for c in cfgs:
        rs = [r for r in kb if (r["p_near"], r["w_hub"]) == c]
        ks = [r["k"] for r in rs if r["k"] is not None]
        if ks:
            cfg_med.append(float(np.median(ks)))
    cfg_med = np.array(cfg_med)
    if len(cfg_med):
        print(f"  n configs with valid k: {len(cfg_med)}")
        print(f"  k band over config medians: [{cfg_med.min():.3e}, {cfg_med.max():.3e}] "
              f"= {cfg_med.max()/cfg_med.min():.1f}x, median {np.median(cfg_med):.3e}")
    with open(os.path.join(RESULT_DIR, "mdr_kband.json"), "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=1)


if __name__ == "__main__":
    main()
