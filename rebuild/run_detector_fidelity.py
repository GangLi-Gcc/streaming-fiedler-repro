#!/usr/bin/env python3
"""rebuild/run_detector_fidelity.py — tracked-vs-exact lag-10 detector-input
fidelity on the connected email-Eu stream.

Round 11 (user-chosen direction: "检测输入保真 replay"). The Round-10 reviewer's
single weakest link: the seed-level audit proves *single-update* directionality
(capture >= 0.92 at the att-crossing), but the vec-D channel consumes a *lag-10*
cosine deficit of the tracked vector. Does the tracked lag-10 displacement
faithfully reproduce the exact lag-10 displacement, and do the resulting alarms
agree?

This script reuses core.py to build the exact and tracked trajectories, then
compares the two lag-10 drift series D^ex_t = 1-|<v^ex_t, v^ex_{t-10}>| and
D^tr_t = 1-|<v^tr_t, v^tr_{t-10}>| pointwise (threshold-independent) and at the
alarm level under a common threshold.

Outputs rebuild/results/detector_fidelity_email-eu.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (DATASETS, build_stream, fiedler_exact, WarmStartTracker,
                  drift_series, window_z, alarms_from_z, evaluate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

DATASET = "email-eu"
L_CUM = 10
TOL = 40


def main():
    cfg = DATASETS[DATASET]
    stationary0 = cfg["stationary0"]
    cp = cfg["inject_at"] - stationary0

    print(f"{DATASET}: building exact + tracked trajectories", flush=True)
    tags = ["control"] + list(cfg["runs"].keys())
    Dex, Dtr = {}, {}
    v2ex, v2tr = {}, {}
    for tag in tags:
        inject = None if tag == "control" else cfg["runs"][tag]
        t0 = time.time()
        snapshots, n = build_stream(DATASET, inject=inject)
        ex, tr = [], []
        trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
        for L, ts in snapshots:
            vals, vecs = fiedler_exact(L, k=3)
            ex.append(vecs[:, 1])
            trk.update(L)
            tr.append(trk.v2.copy())
        ex = np.asarray(ex); tr = np.asarray(tr)
        seg = slice(stationary0, None)
        v2ex[tag] = ex[seg]
        v2tr[tag] = tr[seg]
        Dex[tag] = drift_series(ex[seg], L_CUM)
        Dtr[tag] = drift_series(tr[seg], L_CUM)
        print(f"  {tag}: {len(snapshots)} snapshots ({time.time()-t0:.0f}s)", flush=True)

    # ---- pointwise fidelity (threshold-independent), control + injected ----
    # compare D^ex vs D^tr on all samples where both are finite
    all_dex, all_dtr, by_tag = [], [], {}
    for tag in tags:
        de, dt = Dex[tag], Dtr[tag]
        m = np.isfinite(de) & np.isfinite(dt) & (de > 0) & (dt > 0)
        de, dt = de[m], dt[m]
        by_tag[tag] = (de, dt)
        all_dex.append(de); all_dtr.append(dt)
    all_dex = np.concatenate(all_dex); all_dtr = np.concatenate(all_dtr)

    # Spearman + log10-Pearson correlation
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    sp = spearman(all_dex, all_dtr)
    lp = float(np.corrcoef(np.log10(all_dex), np.log10(all_dtr))[0, 1])
    abs_err = np.abs(all_dex - all_dtr)
    rel_err = np.abs(all_dex - all_dtr) / all_dex
    log_ratio = np.log10(all_dtr / all_dex)

    print(f"\n=== pointwise D^ex vs D^tr (email-Eu segment, n={len(all_dex)}) ===")
    print(f"  Spearman r            = {sp:+.3f}")
    print(f"  log10-Pearson r       = {lp:+.3f}")
    print(f"  median |D^ex - D^tr|  = {np.median(abs_err):.3e}")
    print(f"  median |rel err|      = {np.median(rel_err):.3f}")
    print(f"  log10(D^tr/D^ex): median {np.median(log_ratio):+.3f} "
          f"IQR [{np.percentile(log_ratio,25):+.3f}, {np.percentile(log_ratio,75):+.3f}]")

    # ---- alarm-level agreement under a COMMON threshold (exact control) ----
    zex_c, zmax_ex = window_z(Dex["control"])
    thr_common = float(zmax_ex) * (1 + 1e-6)
    # separate thresholds (as the paper reports)
    ztr_c, zmax_tr = window_z(Dtr["control"])
    thr_tr = float(zmax_tr) * (1 + 1e-6)

    print(f"\n=== alarm-level agreement (cp={cp}, tol={TOL}) ===")
    print(f"  common threshold (exact control max) = {thr_common:.2f}")
    print(f"  separate tracked threshold           = {thr_tr:.2f}")
    print(f"  {'run':10s} {'ex/common':>9s} {'tr/common':>9s} {'tr/own':>7s}")
    agree_common = {"ex": 0, "tr": 0, "agree_det": 0, "agree_miss": 0, "total": 0}
    for tag in cfg["runs"]:
        ze, _ = window_z(Dex[tag])
        zt, _ = window_z(Dtr[tag])
        # alarms under common threshold
        ae = alarms_from_z(ze, thr_common)
        at = alarms_from_z(zt, thr_common)
        # own-threshold tracked alarms
        at_own = alarms_from_z(zt, thr_tr)
        det_e, fp_e, _ = evaluate(ae, cp, tol=TOL)
        det_t, fp_t, _ = evaluate(at, cp, tol=TOL)
        det_to, fp_to, _ = evaluate(at_own, cp, tol=TOL)
        agree_common["total"] += 1
        agree_common["ex"] += int(det_e); agree_common["tr"] += int(det_t)
        agree_common["agree_det"] += int(det_e and det_t)
        agree_common["agree_miss"] += int((not det_e) and (not det_t))
        print(f"  {tag:10s} {str(det_e):>9s} {str(det_t):>9s} {str(det_to):>7s} "
              f"(ex_fp={fp_e}, tr_common_fp={fp_t}, tr_own_fp={fp_to})")

    print(f"\n  under COMMON threshold: ex {agree_common['ex']}/8, "
          f"tr {agree_common['tr']}/8, agree on detection {agree_common['agree_det']}, "
          f"agree on miss {agree_common['agree_miss']}")

    out = {
        "dataset": DATASET, "stationary0": stationary0, "cp": cp, "tol": TOL,
        "n_points": int(len(all_dex)),
        "spearman_r": sp, "log10_pearson_r": lp,
        "median_abs_err": float(np.median(abs_err)),
        "median_rel_err": float(np.median(rel_err)),
        "log10_ratio_median": float(np.median(log_ratio)),
        "log10_ratio_iqr": [float(np.percentile(log_ratio, 25)),
                            float(np.percentile(log_ratio, 75))],
        "threshold_common": thr_common, "threshold_tracked_own": thr_tr,
        "alarm_agreement_common": agree_common,
    }
    with open(os.path.join(RESULT_DIR, "detector_fidelity_email-eu.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\n--> detector_fidelity_email-eu.json")


if __name__ == "__main__":
    main()
