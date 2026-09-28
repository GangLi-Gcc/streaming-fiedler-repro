"""P0d: Q1<->Q2 coupling calibration -- WarmStartTracker vs exact eigsh on real streams.

Replays the email-Eu core-300 stream EXACTLY like real_data_noise.py (same decay,
batching, injection), but at each sample computes BOTH:
  - exact reference  (lam_ex, v_ex)  via fiedler_exact
  - tracked estimate (lam_trk, v_trk) via WarmStartTracker
Quantifies how tracking error inflates the vector-drift noise floor:
  - lambda_2 relative error (median / p99 / max) on the stationary segment
  - eigenvector error eps_t = min(||v_trk - v_ex||, ||v_trk + v_ex||)
  - drift baselines: median / MAD / max of consecutive drift d, exact vs tracked
  - zero-FP threshold ratio thr_trk / thr_ex
  - vec-D detection retention on CP-injected runs (P0c's lambda2-blind cases)

Usage: python track_quality.py --tag email-eu [--inject-sample 400 --inject-frac 0.3
       --inject-seed 1 | --inject-scale 0.6]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers import fiedler_exact, WarmStartTracker           # noqa: E402
from real_data_noise import load_edges, DATA                   # noqa: E402
from vec_drift import drift_series, window_z, evaluate, STATIONARY0, INJECT_SEG_IDX  # noqa: E402

L_CUM = 10


def replay(tag, args):
    cfg = DATA["email-eu"]
    edges = load_edges(cfg["path"])
    act = Counter()
    for u, v, _ in edges:
        act[u] += 1; act[v] += 1
    core = {x for x, _ in act.most_common(cfg["core_k"])}
    edges = [(u, v, ts) for (u, v, ts) in edges if u in core and v in core]
    nodes = sorted(core)
    remap = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    tau_edges = cfg["tau_steps"] * cfg["macro"]
    sample_every = cfg["macro"] * 50
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = cfg["prune_w"]

    trk = WarmStartTracker(restart_tol_ratio=args.restart_tol,
                           refine_abs_tol=args.refine_abs_tol)
    lam_ex, lam3_ex, lam_trk, vecs_ex, vecs_trk, ts_l = [], [], [], [], [], []
    steps, batch = 0, []
    pair_w = {}
    start = time.time()
    for idx, (u, v, ts) in enumerate(edges):
        a, b = (remap[u], remap[v]) if remap[u] < remap[v] else (remap[v], remap[u])
        batch.append((a, b))
        if (idx + 1) % sample_every != 0:
            continue
        for k in list(pair_w.keys()):
            w = pair_w[k] * gamma
            if w < prune_w:
                del pair_w[k]
            else:
                pair_w[k] = w
        for (a, b) in batch:
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + 1.0
        batch = []
        if args.inject_sample is not None and steps == args.inject_sample:
            if args.inject_scale is not None:
                for kk in list(pair_w.keys()):
                    pair_w[kk] *= args.inject_scale
            else:
                rng = np.random.default_rng(args.inject_seed)
                keys = list(pair_w.keys())
                k = int(round(args.inject_frac * len(keys)))
                for kk in rng.choice(len(keys), k, replace=False):
                    del pair_w[keys[kk]]
        rows, cols, vals = [], [], []
        deg = {}
        for (a, b), w in pair_w.items():
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
            rows += [a, b]; cols += [b, a]; vals += [-w, -w]
        for a, d_ in deg.items():
            rows.append(a); cols.append(a); vals.append(d_)
        Lcsr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        v_, _vec = fiedler_exact(Lcsr)
        lam_ex.append(float(v_[1]))
        lam3_ex.append(float(v_[2]) if len(v_) > 2 else np.nan)
        vecs_ex.append(np.asarray(_vec[:, 1], dtype=np.float32))
        lam_trk.append(float(trk.update(-1, -1, 0, Lcsr)))
        vt = np.asarray(trk.v2, dtype=np.float32)
        # align tracked vector to exact for error measurement
        if np.dot(vt, vecs_ex[-1]) < 0:
            vt = -vt
        vecs_trk.append(vt)
        ts_l.append(ts)
        steps += 1
        if steps % 100 == 0:
            print(f"  sample {steps} ({time.time()-start:.0f}s)", flush=True)

    out = dict(lam_ex=np.asarray(lam_ex), lam3_ex=np.asarray(lam3_ex),
               lam_trk=np.asarray(lam_trk),
               v_ex=np.asarray(vecs_ex).T, v_trk=np.asarray(vecs_trk).T,
               ts=np.asarray(ts_l),
               diag=np.asarray(trk.diag, dtype=float),   # (samples-1, 7)
               n_restart=trk.n_restart, n_refine=trk.n_refine,
               avg_ms=np.mean(trk.time_per_event) * 1e3)
    os.makedirs("../runs", exist_ok=True)
    np.savez_compressed(f"../runs/trackq_{tag}.npz", **out)
    print(f"{tag}: {steps} samples, restarts={trk.n_restart} refines={trk.n_refine} "
          f"avg {out['avg_ms']:.1f} ms/sample")
    return out


def analyze(tag, out):
    sl = slice(STATIONARY0, None)
    lex, ltr = out["lam_ex"][sl], out["lam_trk"][sl]
    vex, vtr = out["v_ex"][:, sl].T, out["v_trk"][:, sl].T

    rel_err = np.abs(ltr - lex) / np.maximum(np.abs(lex), 1e-12)
    eps = np.linalg.norm(vtr - vex, axis=1)

    d_ex, D_ex = drift_series(vex)
    d_tr, D_tr = drift_series(vtr)

    def base(d):
        return {"median": float(np.median(d[1:])), "mad": float(1.4826 * np.median(
                    np.abs(np.diff(d[1:]) - np.median(np.diff(d[1:])))) + 1e-15),
                "max": float(d[1:].max())}

    b_ex, b_tr = base(d_ex), base(d_tr)
    thr_ex = b_ex["max"] * (1 + 1e-6)
    thr_tr = b_tr["max"] * (1 + 1e-6)

    # vec-D z detection on both series
    aD_ex, _ = window_z(D_ex, 1e9)
    aD_tr, _ = window_z(D_tr, 1e9)
    _, zmaxD_ex = window_z(D_ex, 1e9)
    _, zmaxD_tr = window_z(D_tr, 1e9)
    res_ex = evaluate(window_z(D_ex, zmaxD_ex * (1 + 1e-6))[0], INJECT_SEG_IDX)
    res_tr = evaluate(window_z(D_tr, zmaxD_tr * (1 + 1e-6))[0], INJECT_SEG_IDX)

    summary = {
        "tag": tag, "samples": int(len(lex)),
        "lam_rel_err": {"median": float(np.median(rel_err)), "p99": float(np.percentile(rel_err, 99)),
                        "max": float(rel_err.max())},
        "v_err": {"median": float(np.median(eps)), "p99": float(np.percentile(eps, 99)),
                  "max": float(eps.max())},
        "drift_baseline_exact": b_ex, "drift_baseline_tracked": b_tr,
        "threshold_ratio_consecutive": thr_tr / thr_ex,
        "vecD_zmax": {"exact": zmaxD_ex, "tracked": zmaxD_tr},
        "vecD_detection": {"exact": res_ex, "tracked": res_tr},
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--inject-sample", type=int, default=None)
    ap.add_argument("--inject-frac", type=float, default=0.3)
    ap.add_argument("--inject-seed", type=int, default=1)
    ap.add_argument("--inject-scale", type=float, default=None)
    ap.add_argument("--restart-tol", type=float, default=0.1,
                    help="WarmStartTracker restart_tol_ratio (tracker aggressiveness)")
    ap.add_argument("--refine-abs-tol", type=float, default=None,
                    help="absolute refinement residual tol (P0f fix; None = gap-relative)")
    args = ap.parse_args()
    out = replay(args.tag, args)
    summary = analyze(args.tag, out)
    path = "../runs/track_quality.json"
    try:
        with open(path, encoding="utf-8") as f:
            all_s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_s = {}
    all_s[args.tag] = summary
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_s, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
