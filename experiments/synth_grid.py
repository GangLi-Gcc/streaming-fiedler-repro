"""P0h: controlled synthetic experiment -- arbitrate gap vs stiffness (c-lam2)
in the tracking attenuation law, and calibrate the minimum detectable rotation.

Design (decouples the two factors that co-moved along the email-Eu ramp):
  base: two-block SBM (n=300, 150+150), intra p=0.15 w=1, inter p_out x w_out
        -> lam2 (small) and gap=lam3-lam2 both grow as inter coupling grows
  stiffness dial: an optional heavy HUB (m=40 edges of weight W) attached to
        random nodes -> d_max up, c=2*d_max+1 up, lam2 barely moves
  rotation dial: relocate q nodes from block B to block A (delete B intra
        edges, add A intra edges) -> controlled v2 rotation theta(q)
  null controls: q=0 replicates measure the drift floor per hub group

For each config: exact (lam2, lam3, v2) before/after + WarmStartTracker
(restart_tol=0.03, refine_abs_tol=1e-6) update. att = d_tr/d_ex.

Outputs runs/synth_grid.json with per-config records for regression
log10 att ~ log10 gap + log10(c-lam2) and MDR calibration.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from trackers import fiedler_exact, WarmStartTracker  # noqa: E402

N_BLOCK = 150
P_IN = 0.15
W_IN = 1.0
M_HUB = 40
P_OUT_GRID = [0.01, 0.02, 0.04]
W_HUB_GRID = [0.0, 3.0, 10.0, 30.0]      # 0 = no hub
Q_GRID = [0, 1, 2, 4, 8, 16, 32]         # relocated nodes (q=0 = null control)
SEEDS = [0, 1, 2]


def build_base(rng, p_out, w_out, w_hub):
    """Two-block SBM + optional hub. Returns (L, n_total)."""
    edges = {}
    # intra blocks
    for blk in range(2):
        lo, hi = blk * N_BLOCK, (blk + 1) * N_BLOCK
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                if rng.random() < P_IN:
                    edges[(i, j)] = W_IN
    # inter block
    for i in range(N_BLOCK):
        for j in range(N_BLOCK, 2 * N_BLOCK):
            if rng.random() < p_out:
                edges[(i, j)] = w_out
    n = 2 * N_BLOCK
    if w_hub > 0:
        hub = n
        targets = rng.choice(n, M_HUB, replace=False)
        for t in targets:
            edges[(min(hub, int(t)), max(hub, int(t)))] = w_hub
        n += 1
    return edges, n


def relocate(rng, edges, q):
    """Move q nodes from block B to block A: drop their B-intra edges, add
    A-intra edges at P_IN. Returns new edges dict."""
    e = dict(edges)
    movers = rng.choice(range(N_BLOCK, 2 * N_BLOCK), q, replace=False)
    for v in movers:
        for k in list(e.keys()):
            if v in k:
                a, b = k
                other = b if a == v else a
                if other >= N_BLOCK:          # intra-B edge: drop
                    del e[k]
        for u in range(N_BLOCK):              # intra-A edges: add
            if u == v or rng.random() >= P_IN:
                continue
            e[(min(u, v), max(u, v))] = W_IN
    return e


def laplacian(edges, n):
    rows, cols, vals = [], [], []
    deg = np.zeros(n)
    for (a, b), w in edges.items():
        deg[a] += w; deg[b] += w
        rows += [a, b]; cols += [b, a]; vals += [-w, -w]
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(deg[i])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)), deg


def align_cos(v_new, v_old):
    c = float(np.dot(v_new, v_old))
    return abs(c)


def main():
    recs = []
    for w_hub in W_HUB_GRID:
        for p_out in P_OUT_GRID:
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                edges0, n = build_base(rng, p_out, W_IN * 0.5, w_hub)
                for q in Q_GRID:
                    e1 = relocate(rng, edges0, q)
                    L0, _ = laplacian(edges0, n)
                    L1, deg1 = laplacian(e1, n)
                    v0, vecs0 = fiedler_exact(L0)
                    v1, vecs1 = fiedler_exact(L1)
                    lam2_0, lam2_1 = float(v0[1]), float(v1[1])
                    lam3_1 = float(v1[2]) if len(v1) > 2 else np.nan
                    v2_0, v2_1 = vecs0[:, 1], vecs1[:, 1]
                    cos_rot = align_cos(v2_1, v2_0)
                    d_ex = 1.0 - cos_rot
                    # tracked update (fresh tracker: init on L0, one batch update)
                    trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
                    trk.update(-1, -1, 0, L0)
                    vh_old = np.array(trk.v2)
                    lam_trk = trk.update(-1, -1, 0, L1)
                    vh_new = np.array(trk.v2)
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    eps = min(np.linalg.norm(vh_new - v2_1),
                              np.linalg.norm(vh_new + v2_1))
                    c = 2.0 * float(deg1.max()) + 1.0
                    recs.append({
                        "w_hub": w_hub, "p_out": p_out, "seed": seed, "q": q,
                        "lam2": lam2_1, "gap": lam3_1 - lam2_1,
                        "c_minus_lam2": c - lam2_1, "dmax": float(deg1.max()),
                        "d_ex": d_ex, "d_tr": d_tr, "att": d_tr / max(d_ex, 1e-15),
                        "eps": eps, "lam_rel_err": abs(lam_trk - lam2_1) / max(lam2_1, 1e-12),
                        "theta": float(np.arccos(np.clip(cos_rot, -1, 1))),
                    })
        print(f"w_hub={w_hub} p_out={p_out} done ({len(recs)} recs)", flush=True)
    with open("../runs/synth_grid.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)
    print("saved runs/synth_grid.json")


if __name__ == "__main__":
    main()
