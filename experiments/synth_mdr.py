"""P0h-part2: MDR calibration -- fine rotation sweep via small inter-block
edge weight delta (continuous theta), crossed with the hub stiffness dial.

Base: two-block SBM (as synth_grid) with light inter coupling; the batch
change = add ONE inter-block edge of weight delta. theta ~ delta, continuous
from ~1e-4 rad upward. att(delta, stiffness) locates the tracker's rotation
threshold; SNR vs the real-stream q99 floor (8.0e-7, email-eu-tA) gives MDR.
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
from synth_grid import build_base, laplacian, align_cos, N_BLOCK  # noqa: E402

P_OUT = 0.02
DELTA_GRID = np.logspace(-4, 1, 11)      # 1e-4 .. 10
W_HUB_GRID = [0.0, 10.0, 30.0]
SEEDS = [0, 1, 2, 3, 4]
Q99_FLOOR_REAL = 8.0e-7                   # email-eu-tA tracked consecutive-d q99


def main():
    recs = []
    for w_hub in W_HUB_GRID:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            edges0, n = build_base(rng, P_OUT, 0.5, w_hub)
            # fixed pair of endpoints for the delta edge (cross-block)
            a = int(rng.integers(0, N_BLOCK))
            b = int(N_BLOCK + rng.integers(0, N_BLOCK))
            L0, _ = laplacian(edges0, n)
            v0, vecs0 = fiedler_exact(L0)
            trk0 = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
            trk0.update(-1, -1, 0, L0)
            for delta in DELTA_GRID:
                e1 = dict(edges0)
                e1[(min(a, b), max(a, b))] = e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                L1, deg1 = laplacian(e1, n)
                v1, vecs1 = fiedler_exact(L1)
                v2_0, v2_1 = vecs0[:, 1], vecs1[:, 1]
                cos_rot = align_cos(v2_1, v2_0)
                d_ex = 1.0 - cos_rot
                trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
                trk.update(-1, -1, 0, L0)
                vh_old = np.array(trk.v2)
                lam_trk = trk.update(-1, -1, 0, L1)
                vh_new = np.array(trk.v2)
                if np.dot(vh_new, vh_old) < 0:
                    vh_new = -vh_new
                d_tr = 1.0 - align_cos(vh_new, vh_old)
                eps = min(np.linalg.norm(vh_new - v2_1), np.linalg.norm(vh_new + v2_1))
                c = 2.0 * float(deg1.max()) + 1.0
                recs.append({"w_hub": w_hub, "seed": seed, "delta": float(delta),
                             "lam2": float(v1[1]), "gap": float(v1[2] - v1[1]) if len(v1) > 2 else np.nan,
                             "c_minus_lam2": c - float(v1[1]),
                             "theta": float(np.arccos(np.clip(cos_rot, -1, 1))),
                             "d_ex": d_ex, "d_tr": d_tr,
                             "att": d_tr / max(d_ex, 1e-15), "eps": eps,
                             "snr_q99": d_tr / Q99_FLOOR_REAL})
        print(f"w_hub={w_hub} done ({len(recs)} recs)", flush=True)
    with open("../runs/synth_mdr.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)
    # summary table
    print("\ntheta / att / SNR(q99) by delta x w_hub (seed-averaged):")
    print(f"{'delta':>8s} | " + " | ".join(f"hub{wh:<4.0f} theta/att/SNR" for wh in W_HUB_GRID))
    for delta in DELTA_GRID:
        cells = []
        for wh in W_HUB_GRID:
            g = [r for r in recs if r["w_hub"] == wh and r["delta"] == float(delta)]
            cells.append(f"{np.mean([r['theta'] for r in g]):.1e}/"
                         f"{np.mean([r['att'] for r in g]):.2f}/"
                         f"{np.mean([r['snr_q99'] for r in g]):.1f}")
        print(f"{delta:8.0e} | " + " | ".join(f"{c:>18s}" for c in cells))
    print("saved runs/synth_mdr.json")


if __name__ == "__main__":
    main()
