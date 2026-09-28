#!/usr/bin/env python3
"""rebuild/run_detection_union.py — email-Eu exact vs tracked detection with
alarm-level UNION pooling (Round 11 supplement).

Reports, for each channel (exact/tracked vec-D, exact/tracked lambda2-z) and for
the exact and tracked unions, the pooled recall/FP/precision/FAR/delay. Alarm
locations are pooled so overlapping alarms in a union are not double-counted.

Outputs rebuild/results/detection_union_email-eu.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (DATASETS, build_stream, fiedler_exact, WarmStartTracker,
                  drift_series, window_z, evaluate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

DATASET = "email-eu"
L_CUM = 10
TOL = 40


def z_alarms(z, thr):
    return sorted({t for t in range(len(z)) if np.isfinite(z[t]) and abs(z[t]) > thr})


def main():
    cfg = DATASETS[DATASET]
    stationary0 = cfg["stationary0"]
    cp = cfg["inject_at"] - stationary0

    tags = ["control"] + list(cfg["runs"].keys())
    series = {}
    for tag in tags:
        inject = None if tag == "control" else cfg["runs"][tag]
        snapshots, n = build_stream(DATASET, inject=inject)
        lam, v2 = [], []
        trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
        lam_trk, v2_trk = [], []
        for L, ts in snapshots:
            vals, vecs = fiedler_exact(L, k=3)
            lam.append(vals[1]); v2.append(vecs[:, 1])
            lam_trk.append(trk.update(L)); v2_trk.append(trk.v2.copy())
        seg = slice(stationary0, None)
        series[tag] = {
            "lam": np.asarray(lam)[seg],
            "vecD": drift_series(np.asarray(v2)[seg], L_CUM),
            "lam_trk": np.asarray(lam_trk)[seg],
            "vecD_trk": drift_series(np.asarray(v2_trk)[seg], L_CUM),
        }
        print(f"  {tag} done", flush=True)

    # calibration on control (per-channel)
    c = series["control"]
    cal = {}
    _, zmax = window_z(c["vecD"]);     cal["vecD"] = float(zmax) * (1 + 1e-6)
    _, zmax = window_z(c["lam"]);      cal["lam"] = float(zmax) * (1 + 1e-6)
    _, zmax = window_z(c["vecD_trk"]); cal["vecD_trk"] = float(zmax) * (1 + 1e-6)
    _, zmax = window_z(c["lam_trk"]);  cal["lam_trk"] = float(zmax) * (1 + 1e-6)

    n_seg = series["control"]["lam"].shape[0]
    eligible = (n_seg - (TOL + 1)) * len(cfg["runs"])

    # collect per-run alarm locations + detections for each channel and union
    channels = ["vecD", "lam", "vecD_trk", "lam_trk"]
    per_run = {k: [] for k in channels}
    for tag in cfg["runs"]:
        s = series[tag]
        for k in channels:
            z, _ = window_z(s[k])
            alarms = z_alarms(z, cal[k])
            det, fp, delay = evaluate(alarms, cp, tol=TOL)
            per_run[k].append((det, fp, delay))
        # unions
        z, _ = window_z(s["vecD"]);     zl, _ = window_z(s["lam"])
        au_ex = set(z_alarms(z, cal["vecD"])) | set(z_alarms(zl, cal["lam"]))
        z, _ = window_z(s["vecD_trk"]); zl, _ = window_z(s["lam_trk"])
        au_tr = set(z_alarms(z, cal["vecD_trk"])) | set(z_alarms(zl, cal["lam_trk"]))
        per_run.setdefault("union_ex", []).append(evaluate(sorted(au_ex), cp, tol=TOL))
        per_run.setdefault("union_tr", []).append(evaluate(sorted(au_tr), cp, tol=TOL))

    def agg(key):
        rows = per_run[key]
        n_cp = len(rows)
        tp = sum(1 for d, _, _ in rows if d)
        fp = sum(f for _, f, _ in rows)
        delay = [dd for _, _, dd in rows if dd is not None]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        return {"tp": tp, "n_cp": n_cp, "recall": tp / n_cp, "fp": fp,
                "precision": prec, "far_per_1000": 1000 * fp / eligible,
                "delay": (min(delay), max(delay)) if delay else None}

    out = {"dataset": DATASET, "cp": cp, "tol": TOL, "n_seg": n_seg,
           "eligible_per_stream": n_seg - (TOL + 1), "eligible_pooled": eligible,
           "calibration": cal, "summary": {}}
    print(f"\n  {'channel':12s} {'rec':>5s} {'FP':>4s} {'prec':>6s} {'FAR/1k':>7s} {'delay':>9s}")
    for k in channels + ["union_ex", "union_tr"]:
        s = agg(k)
        out["summary"][k] = s
        d = f"{s['delay'][0]}-{s['delay'][1]}" if s["delay"] else "n/a"
        print(f"  {k:12s} {s['tp']}/{s['n_cp']:<4d} {s['fp']:>4d} "
              f"{s['precision']:>6.2f} {s['far_per_1000']:>7.2f} {d:>9s}")

    with open(os.path.join(RESULT_DIR, "detection_union_email-eu.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n--> detection_union_email-eu.json")


if __name__ == "__main__":
    main()
