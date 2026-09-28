#!/usr/bin/env python3
"""rebuild/run_provenance.py — traceable per-run record for the email-Eu
detection tables (reproducibility manifest data).

Reuses core.py to rebuild the streams and channels, then records, for every
channel and every run: the calibrated threshold, the full alarm timestamp list
(segment index and cp-relative), and the event peak inside the match window.
Also records the population arithmetic (total snapshots -> burn-in -> segment
-> finite lag-10 samples -> eligible non-event samples).

Outputs rebuild/results/provenance_email-eu.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (DATASETS, build_stream, fiedler_exact, WarmStartTracker,
                  drift_series, window_z, lad_zstar, evaluate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

TOL = 40
L_CUM = 10


def main():
    dataset = "email-eu"
    cfg = DATASETS[dataset]
    stationary0 = cfg["stationary0"]
    cp = cfg["inject_at"] - stationary0

    tags = ["control"] + list(cfg["runs"].keys())

    exact, tracked = {}, {}
    for tag in tags:
        inject = None if tag == "control" else cfg["runs"][tag]
        snapshots, n = build_stream(dataset, inject=inject)
        lam, spec, v2 = [], [], []
        for L, ts in snapshots:
            vals, vecs = fiedler_exact(L, k=6)
            lam.append(vals[1]); spec.append(vals[:6]); v2.append(vecs[:, 1])
        exact[tag] = {"lam": np.asarray(lam), "spec": np.asarray(spec),
                      "v2": np.asarray(v2)}

        trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
        lam_trk, v2_trk = [], []
        for L, ts in snapshots:
            lam_trk.append(trk.update(L)); v2_trk.append(trk.v2.copy())
        tracked[tag] = {"lam_trk": np.asarray(lam_trk),
                        "v2_trk": np.asarray(v2_trk), "n_restart": trk.n_restart}

    n_total = len(exact["control"]["lam"])
    seg = slice(stationary0, None)
    n_seg = exact["control"]["lam"][seg].shape[0]

    # channel series on the segment
    def exact_ch(tag):
        e = exact[tag]
        return {"lam": e["lam"][seg], "spec": e["spec"][seg],
                "vecD": drift_series(e["v2"][seg], L_CUM),
                "LAD": lad_zstar(e["spec"][seg])}

    def tracked_ch(tag):
        t = tracked[tag]
        return {"lam_trk": t["lam_trk"][seg],
                "vecD": drift_series(t["v2_trk"][seg], L_CUM)}

    # calibration on control
    ec = exact_ch("control"); tc = tracked_ch("control")
    cal = {
        "LAD": float(np.nanmax(ec["LAD"])) * (1 + 1e-6),
        "lam": float(window_z(ec["lam"])[1]) * (1 + 1e-6),
        "vecD": float(window_z(ec["vecD"])[1]) * (1 + 1e-6),
        "lam_trk": float(window_z(tc["lam_trk"])[1]) * (1 + 1e-6),
        "vecD_trk": float(window_z(tc["vecD"])[1]) * (1 + 1e-6),
    }

    def alarm_lists(z, thr, one_sided=False):
        """Return alarm segment indices (cp-relative) and the event peak."""
        if one_sided:
            idx = [t for t in range(len(z)) if np.isfinite(z[t]) and z[t] > thr]
        else:
            idx = [t for t in range(len(z)) if np.isfinite(z[t]) and abs(z[t]) > thr]
        # event peak inside the match window [cp, cp+TOL]
        win = z[cp:cp + TOL + 1]
        win = win[np.isfinite(win)]
        if one_sided:
            peak = float(np.max(win)) if win.size else None
        else:
            peak = float(np.max(np.abs(win))) if win.size else None
        return {"alarms_seg": idx, "alarms_cp": [t - cp for t in idx],
                "event_peak": peak}

    runs = {}
    for tag in cfg["runs"]:
        e = exact_ch(tag); t = tracked_ch(tag)
        zlam, _ = window_z(e["lam"]); zvec, _ = window_z(e["vecD"])
        zlamt, _ = window_z(t["lam_trk"]); zvect, _ = window_z(t["vecD"])
        entry = {"run": tag}
        entry["LAD"] = alarm_lists(e["LAD"], cal["LAD"], one_sided=True)
        entry["lam"] = alarm_lists(zlam, cal["lam"])
        entry["vecD"] = alarm_lists(zvec, cal["vecD"])
        entry["lam_trk"] = alarm_lists(zlamt, cal["lam_trk"])
        entry["vecD_trk"] = alarm_lists(zvect, cal["vecD_trk"])
        runs[tag] = entry

    # population arithmetic
    finite_lag10 = n_seg - L_CUM          # lag-10 drift values on the segment
    eligible_per_stream = n_seg - (TOL + 1)
    eligible_pooled = eligible_per_stream * len(cfg["runs"])

    out = {
        "dataset": dataset,
        "population": {
            "n_total_snapshots": n_total,
            "stationary0_burnin": stationary0,
            "n_seg": n_seg,
            "cp_segment_index": cp,
            "tol": TOL,
            "L_CUM": L_CUM,
            "finite_lag10_per_traj": finite_lag10,
            "finite_lag10_pooled_9traj": finite_lag10 * (len(cfg["runs"]) + 1),
            "eligible_per_stream": eligible_per_stream,
            "eligible_pooled": eligible_pooled,
            "n_runs": len(cfg["runs"]),
        },
        "calibration": cal,
        "runs": runs,
    }
    with open(os.path.join(RESULT_DIR, "provenance_email-eu.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"n_total={n_total} stationary0={stationary0} n_seg={n_seg} "
          f"finite_lag10={finite_lag10} eligible/stream={eligible_per_stream} "
          f"eligible_pooled={eligible_pooled}")
    print("calibration:", {k: round(v, 4) for k, v in cal.items()})
    for tag, entry in runs.items():
        print(f"\n{tag}:")
        for ch in ["LAD", "lam", "vecD", "lam_trk", "vecD_trk"]:
            a = entry[ch]
            print(f"  {ch:9s} thr={cal[ch]:.4g} peak={a['event_peak']} "
                  f"alarms_cp={a['alarms_cp']}")


if __name__ == "__main__":
    main()
