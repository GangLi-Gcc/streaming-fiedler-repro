"""P0m: MDR constant vs tracker calibration (tol dependence curve).

Question (report honest boundary, MDR-related #1): is the constant
k = theta*/sqrt(c-lam2) ~ 1.2e-4 stable across the tracker's own
calibration knobs -- refine_abs_tol x restart_tol_ratio?

Design: reuse the P0k 4-block hierarchical SBM family at the 8
well-resolved configs (p_near x w_hub in {10,30}; hub0 excluded:
non-freeze zone at gap>=1.9 and near-degenerate mechanism at gap=0.13,
both flagged as different mechanisms in P0k). Rotation knob identical:
one cross-far edge of weight delta.

Grid: delta logspace(-4, 1, 33) fixed (1.41x spacing, finer than
P0k's 11-point grid). Exact (theta, d_ex) and L1 cached ONCE per
(config, seed) and shared across all tol combos. Tracker protocol per
combo: ONE exact init on L0 (cached state reused for every delta --
v2/lam3 are the only state update() reads), then per delta a fresh
single-step update L0->L1 (same protocol as P0h/P0k).

Combos: abs_tol {1e-4, 1e-6, 1e-8, 1e-10} x restart_tol_ratio
{0.01, 0.03, 0.1} + legacy gap-relative mode (abs_tol=None, rtr=0.03,
the P0f k=1 lock configuration). Reference = (1e-6, 0.03).

Per-record: att, branch, k_used (refine steps used this update;
cost proxy averaged downstream).

Output: runs/synth_tol_dep.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from trackers import fiedler_exact, WarmStartTracker  # noqa: E402
from synth_grid import laplacian, align_cos  # noqa: E402
from synth_gap_scan import build_hier, N_BLK  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [(0.005, 10.0), (0.005, 30.0),
           (0.02, 10.0), (0.02, 30.0),
           (0.08, 10.0), (0.08, 30.0),
           (0.30, 10.0), (0.30, 30.0)]
DELTA_GRID = np.logspace(-4, 1, 33)
ABS_TOLS = [1e-4, 1e-6, 1e-8, 1e-10]
RTRS = [0.01, 0.03, 0.1]
COMBOS = [(a, r) for a in ABS_TOLS for r in RTRS] + [(None, 0.03)]


def tracker_from_state(v2, lam3, abs_tol, rtr):
    """Fresh tracker seeded with cached (v2, lam3) -- avoids re-running
    the exact init for every delta (v2/lam3 are the only state)."""
    trk = WarmStartTracker(restart_tol_ratio=rtr, refine_abs_tol=abs_tol)
    trk.v2 = np.array(v2)
    trk.lam3 = float(lam3)
    return trk


def main():
    t_start = time.perf_counter()
    cache = {}
    for ci, (p_near, w_hub) in enumerate(CONFIGS):
        for seed in SEEDS:
            rng = np.random.default_rng(3000 + seed)
            edges0, n = build_hier(rng, p_near, w_hub)
            a = int(rng.choice(list(range(0 * N_BLK, 1 * N_BLK))))
            b = int(rng.choice(list(range(2 * N_BLK, 3 * N_BLK))))
            L0, _ = laplacian(edges0, n)
            v0, vecs0 = fiedler_exact(L0)
            lam3_0 = float(v0[2]) if len(v0) > 2 else float(v0[1]) * 2 + 1e-3
            rows, L1s = [], []
            for delta in DELTA_GRID:
                e1 = dict(edges0)
                e1[(min(a, b), max(a, b))] = \
                    e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                L1, deg1 = laplacian(e1, n)
                v1, vecs1 = fiedler_exact(L1)
                cos_rot = align_cos(vecs1[:, 1], vecs0[:, 1])
                c = 2.0 * float(deg1.max()) + 1.0
                rows.append({"delta": float(delta),
                             "lam2": float(v1[1]),
                             "gap": float(v1[2] - v1[1]) if len(v1) > 2 else np.nan,
                             "c_minus_lam2": c - float(v1[1]),
                             "theta": float(np.arccos(np.clip(cos_rot, -1, 1))),
                             "d_ex": 1.0 - cos_rot})
                L1s.append(L1)
            cache[(ci, seed)] = {"L0": L0, "v2_0": vecs0[:, 1],
                                 "lam3_0": lam3_0, "rows": rows, "L1s": L1s}
        print(f"cache cfg{ci} ({p_near}, hub{w_hub:.0f}) "
              f"({time.perf_counter()-t_start:.0f}s)", flush=True)

    recs = []
    for ci, (p_near, w_hub) in enumerate(CONFIGS):
        for seed in SEEDS:
            pack = cache[(ci, seed)]
            L0, v2_0, lam3_0 = pack["L0"], pack["v2_0"], pack["lam3_0"]
            rows, L1s = pack["rows"], pack["L1s"]
            for abs_tol, rtr in COMBOS:
                # exact init once per (config, seed, combo)
                trk_init = tracker_from_state(v2_0, lam3_0, abs_tol, rtr)
                base_v2 = np.array(trk_init.v2)
                for row, L1 in zip(rows, L1s):
                    trk = tracker_from_state(v2_0, lam3_0, abs_tol, rtr)
                    old = np.array(trk.v2)
                    trk.update(-1, -1, 0, L1)
                    new = np.array(trk.v2)
                    if np.dot(new, old) < 0:
                        new = -new
                    d_tr = 1.0 - align_cos(new, old)
                    branch, k_used = np.nan, np.nan
                    if trk.diag:
                        branch = int(trk.diag[-1][0])
                        k_used = int(trk.diag[-1][3])
                    recs.append({"p_near": p_near, "w_hub": w_hub,
                                 "seed": seed,
                                 "abs_tol": abs_tol, "rtr": rtr,
                                 "delta": row["delta"],
                                 "theta": row["theta"],
                                 "c_minus_lam2": row["c_minus_lam2"],
                                 "gap": row["gap"],
                                 "d_ex": row["d_ex"], "d_tr": d_tr,
                                 "att": d_tr / max(row["d_ex"], 1e-15),
                                 "branch": branch, "k_used": k_used})
        print(f"combo sweep cfg{ci} done ({time.perf_counter()-t_start:.0f}s)",
              flush=True)
    with open("../runs/synth_tol_dep.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)
    print(f"saved {len(recs)} recs ({time.perf_counter()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
