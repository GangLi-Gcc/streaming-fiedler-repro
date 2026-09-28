"""Research task #1: quantify the drift noise-floor of lambda_2(t).

Part A (noise floor): run the event stream with NO changepoints over several
seeds; characterize the natural fluctuation of lambda_2(t):
  - increment statistics (std, mean, lag-1 autocorrelation)
  - rolling-window mean z-score extremes  -> zero-FP threshold for any detector
  - power spectrum peak of the detrended trajectory

Part B (precursor test): replay the SAME seed with a single planted CP
(pin_drop at t=1500). Since pin_drop does not consume rng, the event streams
are identical up to t=1500 -> a controlled comparison of pre-CP drift against
the no-CP control tells us whether the early alarms seen in sim_v1 are a
reproducible artifact of the event process or genuine precursors.

Usage: python drift_baseline.py [--seeds 4] [--events 3000] [--n 1200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dyngraph import DynamicSBM
from trackers import BaselineFull, WarmStartTracker


def laplacian(A):
    return sp.lil_matrix(sp.diags(A.sum(1)) - A)


def run_stream(n, events, seed, cp_at=None, cp_kind="pin_drop", cp_factor=0.4,
               exact_every=25):
    g = DynamicSBM(n=n, seed=seed)
    L = laplacian(g.A)
    base = BaselineFull()
    trk = WarmStartTracker()
    lam_trk, lam_exact_t, lam_exact_v = [], [], []
    for t in range(events):
        if cp_at is not None and t == cp_at:
            g.trigger(t, cp_kind, factor=cp_factor)
            L = laplacian(g.A)
        else:
            i, j, s = g.next_event()
            if s > 0:
                if L[i, j] == 0:
                    L[i, i] += 1; L[j, j] += 1; L[i, j] = L[j, i] = -1
            else:
                if L[i, j] < 0:
                    L[i, i] -= 1; L[j, j] -= 1; L[i, j] = L[j, i] = 0
        Lcsr = L.tocsr()
        lam_trk.append(trk.update(-1, -1, 0, Lcsr))
        if t % exact_every == 0:
            lam_exact_t.append(t)
            lam_exact_v.append(base.update(-1, -1, 0, Lcsr))
    return np.asarray(lam_trk), np.asarray(lam_exact_t), np.asarray(lam_exact_v)


def noise_floor_stats(lam, win=200):
    d = np.diff(lam)
    ac1 = np.corrcoef(d[:-1], d[1:])[0, 1] if len(d) > 10 else np.nan
    # rolling window mean, z-scored by global increment std
    W = np.lib.stride_tricks.sliding_window_view(lam, win)
    wm = W.mean(axis=1)
    z = (wm - lam.mean()) / (d.std() * np.sqrt(win / 3.0) + 1e-12)
    # periodogram peak of detrended signal
    x = lam - np.convolve(lam, np.ones(200) / 200, mode="same")
    fx = np.abs(np.fft.rfft(x)) ** 2
    fx[0] = 0
    peak_period = len(x) / np.argmax(fx) if fx.argmax() > 0 else np.inf
    return {
        "lam_mean": float(lam.mean()), "lam_std": float(lam.std()),
        "inc_std": float(d.std()), "inc_mean": float(d.mean()),
        "inc_ac1": float(ac1),
        "max_abs_window_z": float(np.abs(z).max()),
        "p999_abs_window_z": float(np.percentile(np.abs(z), 99.9)),
        "spec_peak_period": float(peak_period),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--cp-at", type=int, default=1500)
    args = ap.parse_args()

    control_stats, precursor_z = [], []
    trajs = {}
    for seed in range(args.seeds):
        lam_c, te, ve = run_stream(args.n, args.events, seed, cp_at=None)
        control_stats.append(noise_floor_stats(lam_c))
        trajs[f"control_seed{seed}"] = lam_c
        lam_cp, _, _ = run_stream(args.n, args.events, seed, cp_at=args.cp_at)
        trajs[f"cp_seed{seed}"] = lam_cp
        # pre-CP window drift relative to control distribution
        w0, w1 = args.cp_at - 200, args.cp_at
        pooled = np.concatenate([trajs[f"control_seed{s}"][w0:w1] for s in range(seed + 1)])
        z = (lam_cp[w0:w1].mean() - pooled.mean()) / (pooled.std() + 1e-12)
        precursor_z.append(float(z))

    keys = control_stats[0].keys()
    agg = {k: {"mean": float(np.mean([s[k] for s in control_stats])),
               "std": float(np.std([s[k] for s in control_stats]))}
           for k in keys}

    out = {
        "config": vars(args),
        "control_noise_floor": agg,
        "per_seed": control_stats,
        "precursor_test": {
            "z_of_preCP_window_vs_control": precursor_z,
            "interpretation": "|z|>2 on most seeds => pre-CP drift is a "
                              "reproducible artifact of the event process; "
                              "|z|<2 broadly => sim_v1 early alarms were noise",
        },
    }
    os.makedirs("../runs", exist_ok=True)
    np.savez_compressed("../runs/drift_trajs.npz", **trajs)
    with open("../runs/drift_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
