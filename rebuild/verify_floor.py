#!/usr/bin/env python3
"""verify_floor.py — test whether a sig floor makes vec-D detection robust to
the eigensolver's float32/float64 representation of v2.

For each candidate floor, compute the email-Eu vec-D detection count from the
old float32-cached v2 and from a fresh float64 rebuild, and report whether they
agree.  A robust channel is one whose detection count does not change with the
floating-point representation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (DATASETS, build_stream, fiedler_exact, drift_series,
                  window_z, evaluate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 40
CP = 100  # email-eu: inject_at 400 - stationary0 300


def vecD_detections(v2, seg, cal_floor, cal_thr, floor):
    """Return list of (run, detected) for the vec-D channel with a sig floor."""
    D_ctrl = drift_series(v2["control"][seg], 10)
    _, zmax_ctrl = window_z(D_ctrl, sig_floor=floor)
    thr = zmax_ctrl * (1 + 1e-6)
    out = []
    for tag in ["inj015a", "inj015b", "inj030a", "inj030b",
                "inj050a", "inj050b", "sc06a", "sc06b"]:
        D = drift_series(v2[tag][seg], 10)
        z, _ = window_z(D, sig_floor=floor)
        alarms = [t for t in range(len(z)) if np.isfinite(z[t]) and abs(z[t]) > thr]
        det, fp, delay = evaluate(alarms, CP, TOL)
        out.append((tag, det, fp))
    return out


def main():
    # build float64 v2 for all runs
    print("building float64 v2 (all runs)...", flush=True)
    v2_64 = {}
    for tag in ["control"] + list(DATASETS["email-eu"]["runs"].keys()):
        inject = None if tag == "control" else DATASETS["email-eu"]["runs"][tag]
        snaps, n = build_stream("email-eu", inject=inject)
        v2 = np.zeros((len(snaps), n))
        for i, (L, ts) in enumerate(snaps):
            v2[i] = fiedler_exact(L, k=6)[1][:, 1]
        v2_64[tag] = v2

    # load float32 cached v2
    v2_32 = {}
    for tag in ["control", "inj015a", "inj015b", "inj030a", "inj030b",
                "inj050a", "inj050b", "sc06a", "sc06b"]:
        key = "email-eu" if tag == "control" else tag
        d = np.load(os.path.join(ROOT, "runs", f"real_{key}_spec.npz"))
        v2_32[tag] = np.asarray(d["v2"], dtype=float).T

    seg = slice(300, None)

    print(f"\n{'floor':>10s} {'float32':>10s} {'float64':>10s} {'agree':>6s}")
    for floor in [0.0, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5]:
        r32 = vecD_detections(v2_32, seg, None, None, floor)
        r64 = vecD_detections(v2_64, seg, None, None, floor)
        n32 = sum(1 for _, d, _ in r32 if d)
        n64 = sum(1 for _, d, _ in r64 if d)
        agree = all(r32[i][1] == r64[i][1] for i in range(len(r32)))
        print(f"{floor:10.1e} {n32}/8{'':>4} {n64}/8{'':>4} {agree}")

    # detail at a promising floor
    print(f"\n=== detail at floor=1e-6 (float32 vs float64) ===")
    for floor in [1e-6]:
        r32 = vecD_detections(v2_32, seg, None, None, floor)
        r64 = vecD_detections(v2_64, seg, None, None, floor)
        for (t32, d32, f32), (t64, d64, f64) in zip(r32, r64):
            mark = "  <-- DIFF" if d32 != d64 else ""
            print(f"  {t32}: f32={d32}(fp{f32})  f64={d64}(fp{f64}){mark}")


if __name__ == "__main__":
    main()
