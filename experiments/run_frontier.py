"""Task #3: detection-frontier scaling experiment.

Generates pin_drop amplitude sweeps (factor in {0.2,0.3,0.5,0.7} -> CP size),
measures per-run signal (Delta lambda), local noise scale (sigma_inc of the
same run's pre-CP increments), runs detectors v3/cusum, and tests whether
the detection boundary collapses onto a single SNR curve
    SNR = |Delta lambda| / (sigma_inc * sqrt(win_recent)).

Usage: python run_frontier.py [--n 1200] [--events 2500] [--seeds 3] [--cp-at 1250]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_baseline import run_stream
from detectors import CUSUMDrift, evaluate_detection
from detectors_v2 import PersistenceDrift, alarms_per_1000

TOL = 300


def measure_shift(lam, cp_at, pre=(900, 1200), post=(1500, 1800)):
    return float(lam[post[0]:post[1]].mean() - lam[pre[0]:pre[1]].mean())


def measure_sigma_inc(lam, cp_at, start=400):
    """Increment std from pre-CP segment of the SAME run (local scale)."""
    seg = lam[start:cp_at]
    return float(np.diff(seg).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--events", type=int, default=2500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cp-at", type=int, default=1250)
    args = ap.parse_args()

    factors = [0.2, 0.3, 0.5, 0.7]
    cache_path = "../runs/frontier_traj_cache.npz"
    cache = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}

    rows = []
    for factor in factors:
        for seed in range(args.seeds):
            key = f"pin_drop_f{factor}_s{seed}"
            if key not in cache:
                lam, _, _ = run_stream(args.n, args.events, seed,
                                       cp_at=args.cp_at, cp_kind="pin_drop_batch",
                                       cp_factor=factor, exact_every=50)
                cache[key] = lam
                np.savez_compressed(cache_path, **cache)
            lam = cache[key]
            delta = measure_shift(lam, args.cp_at)
            sig = measure_sigma_inc(lam, args.cp_at)
            snr = abs(delta) / (sig * np.sqrt(100) + 1e-12)
            dets = {
                "cusum": CUSUMDrift(window=240, min_slack=4.0, cooldown=120),
                "v3_persist": PersistenceDrift(win_base=300, win_recent=100,
                                               threshold=3.0, persist_k=25,
                                               cooldown=120),
            }
            for dname, det in dets.items():
                for t, v in enumerate(lam):
                    det.update(t, v)
                ev = evaluate_detection(det.alarms, [args.cp_at], tol=TOL)
                rows.append({
                    "factor": factor, "seed": seed, "detector": dname,
                    "delta_lambda": round(delta, 4),
                    "sigma_inc": round(sig, 5),
                    "snr": round(float(snr), 3),
                    "detected": ev["tp"] > 0,
                    "delay": ev["mean_delay"],
                    "fp": ev["fp"],
                    "alarms_per_1000": round(alarms_per_1000(det.alarms, args.events), 2),
                })
                print(f"[f={factor} seed={seed} {dname}] "
                      f"Delta={delta:.3f} sig={sig:.4f} SNR={snr:.2f} "
                      f"det={ev['tp']>0} fp={ev['fp']}", flush=True)

    out = {"config": vars(args), "rows": rows}
    with open("../runs/frontier_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # compact frontier view
    print("\nSNR-sorted detection outcomes:")
    for r in sorted(rows, key=lambda r: r["snr"]):
        print(f"  f={r['factor']} s={r['seed']} {r['detector']:11s} "
              f"SNR={r['snr']:6.2f} det={r['detected']} fp={r['fp']}")


if __name__ == "__main__":
    main()
