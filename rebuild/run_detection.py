#!/usr/bin/env python3
"""rebuild/run_detection.py — real-stream change-point detection, clean pipeline.

Two lines, both reported:

  exact   : per-snapshot fiedler_exact -> lam, v2, spec (6-smallest eigenvalues),
            plus spec_large (6-largest) for the published LAD signature
            channels: LAD (largest Z*), LAD_small (smallest Z*, audit),
                      lambda2-z, vec-D (window-z)
  tracked : WarmStartTracker streaming trajectory -> lam_trk, v2_trk
            channels: lambda2-z, vec-D on the tracked trajectory

Channels run on the segment [stationary0:].  CP segment index = inject_at -
stationary0 = 100 for both datasets.  Calibration: per-channel control-stream
max + ulp margin (LAD one-sided on Z*; z channels two-sided on |z|).
Evaluation: STRICT (every alarm outside [cp, cp+tol] is a false alarm).

The LAD channel uses the PUBLISHED top-singular-value signature (largest
eigenvalues); the smallest-eigenvalue signature is kept as ``LAD_small`` for the
audit of Appendix (j) and for the CollegeMsg non-reproducibility table.

Outputs rebuild/results/detection_{dataset}.json, including per-channel
valid non-event decision denominators (finite samples outside the match window,
pooled over the injected streams).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (DATASETS, build_stream, fiedler_exact, largest_eigs,
                  WarmStartTracker, drift_series, consecutive_drift, window_z,
                  lad_zstar, evaluate, aggregate_detection)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

TOL = 40
L_CUM = 10


def lad_alarms(zstar, thr):
    """One-sided: Z* > thr."""
    return [t for t in range(len(zstar)) if np.isfinite(zstar[t]) and zstar[t] > thr]


def z_alarms(z, thr):
    """Two-sided: |z| > thr."""
    return [t for t in range(len(z)) if np.isfinite(z[t]) and abs(z[t]) > thr]


def valid_nonevent(z, cp, tol):
    """Per-stream count of finite samples OUTSIDE [cp, cp+tol]: the number of
    non-event samples on which this channel's statistic is actually defined
    (could have alarmed).  A channel whose statistic is NaN on its
    initialization samples (window-z needs 40, lag-10 drift needs 10, LAD Z*
    needs l+1=11) cannot alarm there, so this is strictly less than the raw
    non-event sample count (125 per stream)."""
    return int(sum(1 for t in range(len(z))
                   if not (cp <= t <= cp + tol) and np.isfinite(z[t])))


def run_dataset(dataset):
    cfg = DATASETS[dataset]
    stationary0 = cfg["stationary0"]
    cp = cfg["inject_at"] - stationary0

    print(f"\n{'='*70}\n{dataset}: building streams (exact + tracked)\n{'='*70}",
          flush=True)

    exact, tracked = {}, {}
    tags = ["control"] + list(cfg["runs"].keys())
    for tag in tags:
        inject = None if tag == "control" else cfg["runs"][tag]
        t0 = time.time()
        snapshots, n = build_stream(dataset, inject=inject)
        lam, spec, spec_large, v2 = [], [], [], []
        for L, ts in snapshots:
            vals, vecs = fiedler_exact(L, k=6)
            lam.append(vals[1]); spec.append(vals[:6]); v2.append(vecs[:, 1])
            spec_large.append(largest_eigs(L, k=6))
        exact[tag] = {"lam": np.asarray(lam), "spec": np.asarray(spec),
                      "spec_large": np.asarray(spec_large),
                      "v2": np.asarray(v2)}

        trk = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6)
        lam_trk, v2_trk = [], []
        for L, ts in snapshots:
            lam_trk.append(trk.update(L)); v2_trk.append(trk.v2.copy())
        tracked[tag] = {"lam_trk": np.asarray(lam_trk),
                        "v2_trk": np.asarray(v2_trk), "n_restart": trk.n_restart}
        print(f"  {tag}: {len(snapshots)} snapshots n={n} "
              f"(exact+tracked {time.time()-t0:.0f}s, restarts={trk.n_restart})",
              flush=True)

    seg = slice(stationary0, None)
    n_seg = exact["control"]["lam"][seg].shape[0]

    # ---- channel series on the segment ----
    def exact_ch(tag):
        e = exact[tag]
        return {"lam": e["lam"][seg], "spec": e["spec"][seg],
                "spec_large": e["spec_large"][seg],
                "vecD": drift_series(e["v2"][seg], L_CUM),
                "LAD": lad_zstar(e["spec_large"][seg]),
                "LAD_small": lad_zstar(e["spec"][seg])}

    def tracked_ch(tag):
        t = tracked[tag]
        return {"lam_trk": t["lam_trk"][seg],
                "vecD": drift_series(t["v2_trk"][seg], L_CUM)}

    # ---- calibration on control ----
    ec = exact_ch("control"); tc = tracked_ch("control")
    cal = {}
    cal["LAD"] = float(np.nanmax(ec["LAD"])) * (1 + 1e-6)
    cal["LAD_small"] = float(np.nanmax(ec["LAD_small"])) * (1 + 1e-6)
    _, zmax = window_z(ec["lam"]);    cal["lam"] = float(zmax) * (1 + 1e-6)
    _, zmax = window_z(ec["vecD"]);   cal["vecD"] = float(zmax) * (1 + 1e-6)
    _, zmax = window_z(tc["lam_trk"]); cal["lam_trk"] = float(zmax) * (1 + 1e-6)
    _, zmax = window_z(tc["vecD"]);   cal["vecD_trk"] = float(zmax) * (1 + 1e-6)
    print(f"  calibration: LAD={cal['LAD']:.4f} LAD_small={cal['LAD_small']:.4f} "
          f"lam={cal['lam']:.2f} vecD={cal['vecD']:.2f} "
          f"lam_trk={cal['lam_trk']:.2f} vecD_trk={cal['vecD_trk']:.2f}", flush=True)

    # ---- per-run evaluation + valid-decision denominators ----
    rows = []
    valid = {k: 0 for k in ["LAD", "LAD_small", "lam", "vecD", "lam_trk", "vecD_trk"]}
    for tag in cfg["runs"]:
        e = exact_ch(tag); t = tracked_ch(tag)
        zlam, _ = window_z(e["lam"]); zvec, _ = window_z(e["vecD"])
        zlamt, _ = window_z(t["lam_trk"]); zvect, _ = window_z(t["vecD"])
        row = {"run": tag}
        row["LAD"] = evaluate(lad_alarms(e["LAD"], cal["LAD"]), cp)
        row["LAD_small"] = evaluate(lad_alarms(e["LAD_small"], cal["LAD_small"]), cp)
        row["lam"] = evaluate(z_alarms(zlam, cal["lam"]), cp)
        row["vecD"] = evaluate(z_alarms(zvec, cal["vecD"]), cp)
        row["lam_trk"] = evaluate(z_alarms(zlamt, cal["lam_trk"]), cp)
        row["vecD_trk"] = evaluate(z_alarms(zvect, cal["vecD_trk"]), cp)
        for k in ["LAD", "LAD_small", "lam", "vecD", "lam_trk", "vecD_trk"]:
            det, fp, _ = row[k]
            row[f"{k}_n"] = fp + (1 if det else 0)
        valid["LAD"] += valid_nonevent(e["LAD"], cp, TOL)
        valid["LAD_small"] += valid_nonevent(e["LAD_small"], cp, TOL)
        valid["lam"] += valid_nonevent(zlam, cp, TOL)
        valid["vecD"] += valid_nonevent(zvec, cp, TOL)
        valid["lam_trk"] += valid_nonevent(zlamt, cp, TOL)
        valid["vecD_trk"] += valid_nonevent(zvect, cp, TOL)
        rows.append(row)

    # ---- pooled metrics ----
    summary = {}
    for k in ["vecD", "lam", "LAD", "LAD_small", "vecD_trk", "lam_trk"]:
        summary[k] = aggregate_detection([(r[k][0], r[k][1], r[k][2]) for r in rows],
                                         n_seg, tol=TOL, cp=cp)

    out = {"dataset": dataset, "stationary0": stationary0, "cp": cp,
           "tol": TOL, "n_seg": n_seg, "calibration": cal,
           "valid_nonevent_pooled": valid, "rows": rows, "summary": summary}
    with open(os.path.join(RESULT_DIR, f"detection_{dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"\n  pooled (segment n={n_seg}, cp={cp}, tol={TOL}):")
    print(f"  {'channel':10s} {'rec':>5s} {'CI95':>14s} {'FP':>4s} "
          f"{'prec':>6s} {'FAR/1k':>7s} {'valid':>6s} {'delay':>9s}")
    for k in ["vecD", "lam", "LAD", "LAD_small", "vecD_trk", "lam_trk"]:
        s = summary[k]
        ci = f"[{s['recall_ci95'][0]:.2f},{s['recall_ci95'][1]:.2f}]"
        d = f"{s['delay_min']}-{s['delay_max']}" if s['delay_min'] is not None else "n/a"
        print(f"  {k:10s} {s['tp']}/{s['n_cp']:<4d} {ci:14s} {s['fp']:>4d} "
              f"{s['precision']:>6.2f} {s['far_per_1000']:>7.2f} "
              f"{valid[k]:>6d} {d:>9s}")
    return out


if __name__ == "__main__":
    for ds in ["email-eu", "college-msg"]:
        run_dataset(ds)
