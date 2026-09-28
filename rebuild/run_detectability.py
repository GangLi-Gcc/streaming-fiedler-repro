#!/usr/bin/env python3
"""rebuild/run_detectability.py — validate MDR against measured noise and
actual alarm thresholds (Round-7 CRITICAL #1 experiment).

The paper's MDR is the smallest rotation theta at which att = d_tr/d_ex >= 0.5,
i.e. a *relative* tracking response. The reviewer's objection: a relative
response of 0.5 can sit below tracker noise, so "minimum detectable rotation"
is an unsupported label. This script supplies the two missing ingredients:

  (1) measured noise: the per-step tracking response when the graph changes by
      same-distribution resampling only (no structural rotation);
  (2) actual alarm thresholds: the vec-D window-z detector (the same detector
      used in Sec. 5) run across the rotation grid, calibrated on the noise
      segment.

Two read-outs are produced per (p_near, w_hub, seed, delta):

  SNR(delta)   = d_tr(delta) / MAD(per-step d_tr over the noise segment),
                 the single-step signal-to-noise of the tracked response.
                 A rotation is "single-step detectable" iff SNR crosses a
                 threshold (3 and 5 are both reported).

  detect(delta) = whether the vec-D window-z detector fires in the T1 samples
                  after the rotation, under control-segment calibration
                  (max |z| on the noise segment + ulp margin) — the same
                  strict protocol as Sec. 5.

Both are compared against att(delta) from run_mdr.py, so the claim
"att>=0.5 <=> detectable" can be checked directly: does theta_att (MDR) sit
above, at, or below theta_snr / theta_det?

Protocol per (p_near, w_hub, seed, delta):
  - build G0 (same 4-block hierarchical SBM as run_mdr.py)
  - exact v2_0 = fiedler_exact(G0)
  - noise segment: T0 same-distribution resampled graphs G'_t (no rotation),
    a tracker seeded at v2_0 takes one update each, record per-step d_tr.
  - signal: tracker seeded at v2_0 takes one update on G_inj = G0 + delta,
    record d_tr(delta); also exact d_ex for att.
  - detector segment: a *continuous* tracker run — T0 resampled steps, then
    the delta rotation, then T1 fixed-G_inj steps — into drift_series ->
    window_z -> alarm vs control-segment max.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import fiedler_exact, WarmStartTracker, align_cos, drift_series, window_z

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# --- rig parameters (identical to run_mdr.py) ---
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

# --- detector / noise-segment lengths ---
T0 = 80          # noise-segment length (>= win_base+win_recent for window_z)
T1 = 60          # post-rotation detector length (>= match window 40)
L_CUM = 10       # drift_series lag (matches Sec. 5)


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


def main(p_near_subset=None, w_hub_subset=None, seed_subset=None):
    pn_grid = p_near_subset or P_NEAR_GRID
    wh_grid = w_hub_subset or W_HUB_GRID
    sd_grid = seed_subset or SEEDS

    recs = []
    t0 = time.time()
    for p_near in pn_grid:
        for w_hub in wh_grid:
            for seed in sd_grid:
                rng0 = np.random.default_rng(3000 + seed)
                edges0, n = build_hier(rng0, p_near, w_hub)
                L0, _ = laplacian(edges0, n)
                vals0, vecs0 = fiedler_exact(L0)
                v2_0 = vecs0[:, 1]
                lam2_0, lam3_0 = vals0[1], vals0[2]
                a = int(rng0.choice(list(BLK(0))))
                b = int(rng0.choice(list(BLK(2))))

                # ---- noise segment: same-distribution resampled graphs ----
                # Precompute resampled Laplacians once per (config, seed).
                rng_n = np.random.default_rng(10000 + seed)
                L_noise = []
                for _ in range(T0):
                    e_t, n_t = build_hier(rng_n, p_near, w_hub)
                    L_noise.append(laplacian(e_t, n_t)[0])

                trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
                trk.seed(v2_0, lam2_0, lam3_0)
                dtr_noise = []
                v_traj = [v2_0.copy()]
                v_prev = v2_0
                for L_t in L_noise:
                    trk.update(L_t)
                    v_cur = trk.v2.copy()
                    if np.dot(v_cur, v_prev) < 0:
                        v_cur = -v_cur
                    dtr_noise.append(1.0 - align_cos(v_cur, v_prev))
                    v_prev = trk.v2.copy()
                    v_traj.append(trk.v2.copy())

                mad_noise = 1.4826 * np.median(np.abs(np.array(dtr_noise)
                                                    - np.median(dtr_noise)))

                # sigma_local: single random cross-far edge of weight 1 (noise
                # floor, the *smallest* plausible per-step perturbation). If the
                # MDR signal is far below even this floor, the detectable label
                # is indefensible under any noise model.
                dtr_local = []
                rng_l = np.random.default_rng(50000 + seed)
                for _ in range(20):
                    ai = int(rng_l.choice(list(BLK(0))))
                    bi = int(rng_l.choice(list(BLK(2))))
                    e_local = dict(edges0)
                    e_local[(min(ai, bi), max(ai, bi))] = \
                        e_local.get((min(ai, bi), max(ai, bi)), 0.0) + 1.0
                    L_loc, _ = laplacian(e_local, n)
                    trk_l = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
                    trk_l.seed(v2_0, lam2_0, lam3_0)
                    trk_l.update(L_loc)
                    v_l = trk_l.v2.copy()
                    if np.dot(v_l, v2_0) < 0:
                        v_l = -v_l
                    dtr_local.append(1.0 - align_cos(v_l, v2_0))
                mad_local = 1.4826 * np.median(np.abs(np.array(dtr_local)
                                                      - np.median(dtr_local)))

                for di, delta in enumerate(DELTA_GRID):
                    e_inj = dict(edges0)
                    e_inj[(min(a, b), max(a, b))] = \
                        e_inj.get((min(a, b), max(a, b)), 0.0) + float(delta)
                    L_inj, deg_inj = laplacian(e_inj, n)
                    vals_inj, vecs_inj = fiedler_exact(L_inj)
                    d_ex = 1.0 - align_cos(vecs_inj[:, 1], v2_0)

                    # (1) single-step signal from the exact v2_0
                    trk_s = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
                    trk_s.seed(v2_0, lam2_0, lam3_0)
                    trk_s.update(L_inj)
                    v_sig = trk_s.v2.copy()
                    if np.dot(v_sig, v2_0) < 0:
                        v_sig = -v_sig
                    d_tr = 1.0 - align_cos(v_sig, v2_0)
                    att = d_tr / max(d_ex, 1e-15)
                    snr = d_tr / max(mad_noise, 1e-15)

                    # (2) continuous detector: T0 resampled steps (noise), then
                    # T1 resampled steps with the delta edge superimposed. The
                    # ONLY difference across the change point is the delta edge,
                    # so the graph dynamics is matched either side.
                    trk_d = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
                    trk_d.seed(v2_0, lam2_0, lam3_0)
                    cont = [v2_0.copy()]
                    for L_t in L_noise:
                        trk_d.update(L_t)
                        cont.append(trk_d.v2.copy())
                    rng_i = np.random.default_rng(60000 + seed * 100 + di)
                    for _ in range(T1):
                        e_t, n_t = build_hier(rng_i, p_near, w_hub)
                        e_t[(min(a, b), max(a, b))] = \
                            e_t.get((min(a, b), max(a, b)), 0.0) + float(delta)
                        L_it, _ = laplacian(e_t, n_t)
                        trk_d.update(L_it)
                        cont.append(trk_d.v2.copy())

                    D = drift_series(np.asarray(cont), L_CUM)
                    # sig_floor: noise-segment drift-increment MAD, so the fixed
                    # tail does not collapse the dispersion estimate to 0.
                    inc = np.diff(D[L_CUM:T0 + 1])
                    sig_floor = 1.4826 * np.median(np.abs(inc - np.median(inc)))
                    z, _ = window_z(D, sig_floor=sig_floor)
                    z_noise = z[1:T0 + 1]
                    cal = float(np.nanmax(np.abs(z_noise))) * (1 + 1e-6)
                    z_after = z[T0 + 1:T0 + 1 + T1 + 1]
                    fired = [t for t, zz in enumerate(z_after)
                             if np.isfinite(zz) and abs(zz) > cal]
                    detect = bool(fired)

                    recs.append({
                        "p_near": p_near, "w_hub": w_hub, "seed": seed,
                        "delta": float(delta),
                        "d_ex": d_ex, "d_tr": d_tr, "att": att,
                        "mad_noise": float(mad_noise), "snr": float(snr),
                        "mad_local": float(mad_local),
                        "snr_local": float(d_tr / max(mad_local, 1e-15)),
                        "detect": detect, "n_alarm": len(fired),
                        "cal": cal,
                        "zmax_after": float(np.nanmax(np.abs(z_after)))
                        if np.any(np.isfinite(z_after)) else None,
                    })
        print(f"p_near={p_near} done ({len(recs)} recs, {time.time()-t0:.0f}s)",
              flush=True)

    with open(os.path.join(RESULT_DIR, "detectability.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    # ---- per-config summary: theta_att vs theta_snr vs theta_det ----
    print("\n=== per-config detectability summary ===")
    cfgs = sorted({(r["p_near"], r["w_hub"]) for r in recs})
    for c in cfgs:
        rs = [r for r in recs if (r["p_near"], r["w_hub"]) == c]
        seeds = sorted({r["seed"] for r in rs})
        row = {"p_near": c[0], "w_hub": c[1]}
        for key, thr, name in [("att", 0.5, "theta_att"),
                               ("snr", 3.0, "theta_snr3"),
                               ("snr", 5.0, "theta_snr5"),
                               ("snr_local", 3.0, "theta_snr3_local"),
                               ("snr_local", 5.0, "theta_snr5_local")]:
            thetas = []
            for s in seeds:
                r_s = sorted([r for r in rs if r["seed"] == s],
                             key=lambda r: r["delta"])
                for r in r_s:
                    if r[key] >= thr:
                        thetas.append(np.arccos(np.clip(1.0 - r["d_ex"], -1, 1)))
                        break
            row[name] = float(np.median(thetas)) if thetas else None
        # theta_det = smallest delta with majority (>=3/5) seeds detecting
        deltas = sorted({r["delta"] for r in rs})
        th_det = None
        for d in deltas:
            r_d = [r for r in rs if r["delta"] == d]
            if sum(1 for r in r_d if r["detect"]) >= 3:
                d_ex = np.median([r["d_ex"] for r in r_d])
                th_det = float(np.arccos(np.clip(1.0 - d_ex, -1, 1)))
                break
        row["theta_det"] = th_det
        # median mad_noise and a representative d_ex at att~0.5
        row["mad_noise"] = float(np.median([r["mad_noise"] for r in rs]))
        row["mad_local"] = float(np.median([r["mad_local"] for r in rs]))
        def f_(x):
            return f"{x:.2e}" if x is not None else "None"
        print(f"  p_near={c[0]:.3f} w_hub={c[1]:>2}: "
              f"theta_att={f_(row['theta_att'])} "
              f"theta_snr3={f_(row['theta_snr3'])} "
              f"theta_snr5={f_(row['theta_snr5'])} "
              f"theta_snr3loc={f_(row['theta_snr3_local'])} "
              f"theta_det={f_(row['theta_det'])} "
              f"mad_noise={row['mad_noise']:.2e} mad_local={row['mad_local']:.2e}")

    return recs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pn", type=float, nargs="+", default=None)
    ap.add_argument("--wh", type=float, nargs="+", default=None)
    ap.add_argument("--seed", type=int, nargs="+", default=None)
    args = ap.parse_args()
    main(p_near_subset=args.pn, w_hub_subset=args.wh, seed_subset=args.seed)
