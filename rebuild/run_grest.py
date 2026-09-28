#!/usr/bin/env python3
"""rebuild/run_grest.py — G-REST vs WarmStartTracker head-to-head (paper Sec. 5).

Streaming protocol (identical to grest_compare.py): one growing-delta stream per
(w_hub, seed), tracker state persists across steps, L_prev advances at each step,
d_ex between consecutive exact Fiedler vectors, d_tr between the tracker's own
consecutive outputs.  The only addition is the residual-triggered G-REST variant.

Two metrics, both reported because they DISAGREE (see FINDINGS.md):
  att       = d_tr / d_ex   (per-step capture ratio; magnitude only)
  drift     = final |angle| between tracked v2 and exact v2

Output rebuild/results/grest_compare.json with per-seed rows.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import fiedler_exact, WarmStartTracker, GRESTTracker, align_cos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

N_BLOCK = 150
P_IN = 0.15
W_IN = 1.0
M_HUB = 40
P_OUT = 0.02
W_HUB_GRID = [0.0, 10.0, 30.0]
DELTA_GRID = np.logspace(-4, 1, 11)
SEEDS = [0, 1, 2, 3, 4]
NOISE_FLOOR = 1e-9


def build_base(rng, p_out, w_out, w_hub):
    edges = {}
    for blk in range(2):
        lo, hi = blk * N_BLOCK, (blk + 1) * N_BLOCK
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                if rng.random() < P_IN:
                    edges[(i, j)] = W_IN
    for i in range(N_BLOCK):
        for j in range(N_BLOCK, 2 * N_BLOCK):
            if rng.random() < p_out:
                edges[(i, j)] = w_out
    n = 2 * N_BLOCK
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
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def main():
    records = []
    for w_hub in W_HUB_GRID:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            edges0, n = build_base(rng, P_OUT, 0.5, w_hub)
            a = int(rng.integers(0, N_BLOCK))
            b = int(N_BLOCK + rng.integers(0, N_BLOCK))
            L_prev = laplacian(edges0, n)
            _, vecs_prev = fiedler_exact(L_prev)

            trackers = {
                "warmstart": WarmStartTracker(restart_tol_ratio=0.03,
                                              refine_abs_tol=1e-6),
                "grest_K5": GRESTTracker(K=5, restart_policy="none"),
                "grest_K5_p4": GRESTTracker(K=5, restart_policy="periodic",
                                            restart_every=4),
                "grest_K5_rtrig": GRESTTracker(K=5, restart_policy="residual",
                                               restart_tol=0.03),
            }
            steps = {k: [] for k in trackers}

            for delta in DELTA_GRID:
                e1 = dict(edges0)
                key = (min(a, b), max(a, b))
                e1[key] = e1.get(key, 0.0) + float(delta)
                L1 = laplacian(e1, n)
                _, vecs1 = fiedler_exact(L1)
                d_ex = 1.0 - align_cos(vecs1[:, 1], vecs_prev[:, 1])

                for name, trk in trackers.items():
                    if trk.v2 is None:
                        trk.update(L_prev)
                    nres_before = trk.n_restart
                    vh_old = np.array(trk.v2)
                    trk.update(L1)
                    vh_new = np.array(trk.v2)
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    restarted = trk.n_restart > nres_before
                    steps[name].append((float(delta), float(d_ex), float(d_tr),
                                        float(d_tr / max(d_ex, 1e-15)),
                                        bool(restarted)))
                L_prev, vecs_prev = L1, vecs1

            row = {"w_hub": w_hub, "seed": seed, "n": n}
            for name, trk in trackers.items():
                solid = [s for s in steps[name] if s[1] > NOISE_FLOOR]
                att_all = [s[3] for s in solid]
                att_nr = [s[3] for s in solid if not s[4]]
                _, vex = fiedler_exact(L_prev, k=3)
                drift = 1.0 - align_cos(trk.v2, vex[:, 1])
                row[name] = {
                    "att_med_all": float(np.median(att_all)) if att_all else float("nan"),
                    "att_med_nonrestart": float(np.median(att_nr)) if att_nr else float("nan"),
                    "att_per_seed": [round(s[3], 4) for s in solid],
                    "n_restart": int(trk.n_restart),
                    "final_drift": float(drift),
                }
            records.append(row)
            print(f"  w={w_hub:<5} seed={seed} ws_att={row['warmstart']['att_med_all']:.3f} "
                  f"gr={row['grest_K5']['att_med_all']:.3f} "
                  f"rtrig={row['grest_K5_rtrig']['att_med_nonrestart']:.3f} "
                  f"ws_drift={row['warmstart']['final_drift']:.1e} "
                  f"gr_drift={row['grest_K5']['final_drift']:.1e}", flush=True)

    with open(os.path.join(RESULT_DIR, "grest_compare.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)

    # summary
    print("\n=== median over 5 seeds ===")
    print(f"  {'w':>4} {'WS att':>7} {'WS drift':>10} {'GR att':>7} {'GR drift':>10} "
          f"{'rtrig att':>10}")
    for w in W_HUB_GRID:
        rs = [r for r in records if r["w_hub"] == w]
        med = lambda k, f: float(np.median([r[k][f] for r in rs]))
        print(f"  {w:>4.0f} {med('warmstart','att_med_all'):>7.3f} "
              f"{med('warmstart','final_drift'):>10.1e} {med('grest_K5','att_med_all'):>7.3f} "
              f"{med('grest_K5','final_drift'):>10.1e} "
              f"{med('grest_K5_rtrig','att_med_nonrestart'):>10.3f}")

    # drift separation check
    ws_drift = [r["warmstart"]["final_drift"] for r in records]
    gr_drift = [r["grest_K5"]["final_drift"] for r in records]
    print(f"\n  drift separation: every G-REST > every WarmStart? "
          f"{min(gr_drift) > max(ws_drift)} "
          f"(GR min {min(gr_drift):.2e} vs WS max {max(ws_drift):.2e})")


if __name__ == "__main__":
    main()
