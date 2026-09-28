"""Task #3 (calibrated frontier): per-seed threshold calibration, then detection.

Why: at background alarm rates of ~1/150-400 events and tol=300, ANY detector
'detects' by chance. Calibration: for each seed, pick the lowest detector
threshold that yields ZERO alarms on that seed's CONTROL (no-CP) trajectory.
Apply the calibrated detector to the CP trajectories of the same seed.

This isolates the frontier question: at what SNR does the CP signal exceed
the maximum wander excursion of the same graph?

Usage: python run_frontier_calibrated.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import CUSUMDrift, evaluate_detection
from detectors_v2 import PersistenceDrift

TOL = 300
CP_AT = 1250
FACTORS = [0.2, 0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]


def make(det_name, thr):
    if det_name == "cusum":
        return CUSUMDrift(window=240, min_slack=thr, cooldown=120)
    return PersistenceDrift(win_base=300, win_recent=100, threshold=thr,
                            persist_k=25, cooldown=120)


def run_det(det, lam):
    for t, v in enumerate(lam):
        det.update(t, v)
    return det


def calibrate_threshold(det_name, control_lam, grid):
    """Lowest threshold with zero alarms on the control trajectory."""
    best = None
    for thr in sorted(grid):
        det = run_det(make(det_name, thr), control_lam)
        if not det.alarms:
            best = thr
            break
    if best is None:  # all grid points alarm -> use max grid + 1
        best = max(grid) * 1.5
    return best


def main():
    ctrl = dict(np.load("../runs/traj_cache.npz"))
    cp = dict(np.load("../runs/frontier_traj_cache.npz"))

    grids = {
        "cusum": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0],
        "v3": [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0],
    }
    det_of = {"cusum": "cusum", "v3": "v3"}

    rows = []
    for seed in SEEDS:
        control = ctrl[f"control_{seed}"]
        for dname in ["cusum", "v3"]:
            thr = calibrate_threshold(det_of[dname], control, grids[dname])
            for f in FACTORS:
                lam = cp[f"pin_drop_f{f}_s{seed}"]
                det = run_det(make(det_of[dname], thr), lam)
                ev = evaluate_detection(det.alarms, [CP_AT], tol=TOL)
                delta = float(lam[1500:1800].mean() - lam[900:1200].mean())
                sig = float(np.diff(lam[400:CP_AT]).std())
                rows.append({
                    "detector": dname, "seed": seed, "factor": f,
                    "calibrated_thr": thr,
                    "delta_lambda": round(delta, 4),
                    "sigma_inc": round(sig, 5),
                    "snr": round(abs(delta) / (sig * 10.0 + 1e-12), 3),
                    "detected": ev["tp"] > 0, "fp": ev["fp"],
                    "delay": ev["mean_delay"],
                    "n_alarms": len(det.alarms),
                })
                print(f"[s{seed} {dname} thr={thr} f={f}] SNR={rows[-1]['snr']:.2f} "
                      f"det={ev['tp']>0} alarms={len(det.alarms)}", flush=True)

    with open("../runs/frontier_calibrated.json", "w", encoding="utf-8") as fobj:
        json.dump({"rows": rows}, fobj, indent=2, ensure_ascii=False)
    print("\n=== frontier (calibrated, zero-FP-on-control thresholds) ===")
    for r in sorted(rows, key=lambda r: r["snr"]):
        print(f"  s{r['seed']} {r['detector']:6s} f={r['factor']} thr={r['calibrated_thr']:<5} "
              f"SNR={r['snr']:6.2f} det={r['detected']} alarms={r['n_alarms']}")


if __name__ == "__main__":
    main()
