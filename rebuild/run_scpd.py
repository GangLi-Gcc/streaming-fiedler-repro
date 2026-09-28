"""rebuild/run_scpd.py — SCPD structural-channel baseline in the clean pipeline.

Reproduces the SCPD row of tab:email / tab:union from core.build_stream +
core.dos_signature + core.lad_zstar + core.evaluate, closing the provenance
loop so every baseline in the paper regenerates from the single rebuild/ source
tree (the earlier SCPD figure came from experiments/scpd_baseline.py, which
shares the identical stream/injection/scoring config but a different code path).

Protocol (identical to run_detection.py's LAD):
  - stream: core.build_stream(dataset, inject)  (same DATASETS config)
  - signature: core.dos_signature per snapshot (50-bin normalized-Laplacian DOS)
  - scorer: core.lad_zstar (dual windows s=5/l=10) on the DOS embedding
  - calibration: control-stream max of Z* x (1+1e-6), one-sided (Z* > thr)
  - evaluation: core.evaluate strict [cp, cp+40]; fp = every alarm outside
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATASETS, build_stream, dos_signature, lad_zstar, evaluate, wilson_ci


def replay_dos(dataset, inject):
    snapshots, n = build_stream(dataset, inject=inject)
    T = len(snapshots)
    dos = np.zeros((T, 50))
    for t, (L, _ts) in enumerate(snapshots):
        dos[t] = dos_signature(L)
    return dos


def main(dataset="email-eu"):
    cfg = DATASETS[dataset]
    runs = cfg["runs"]
    stationary0 = cfg["stationary0"]
    cp = cfg["inject_at"] - stationary0  # 100 for email-eu, 100 for college-msg
    tol = 40

    # per-run Z* series on the post-burn-in segment
    seg = slice(stationary0, None)
    zs = {}
    for tag, inject in runs.items():
        dos = replay_dos(dataset, inject)
        zs[tag] = lad_zstar(dos[seg])

    # control-stream calibration (one-sided max of Z*)
    dos_ctrl = replay_dos(dataset, None)
    z_ctrl = lad_zstar(dos_ctrl[seg])
    zmax = float(np.nanmax(np.where(np.isfinite(z_ctrl), z_ctrl, 0.0)))
    thr = zmax * (1 + 1e-6)
    print(f"{dataset}: SCPD control max Z* = {zmax:.6e} -> thr = {thr:.6e}")

    # evaluate
    recs = []
    for tag, inject in runs.items():
        z = zs[tag]
        alarms = [t for t in range(len(z)) if np.isfinite(z[t]) and z[t] > thr]
        det, fp, delay = evaluate(alarms, cp=cp, tol=tol)
        recs.append((det, fp, delay))
        print(f"  {tag}: det={det} fp={fp} delay={delay} nalarms={len(alarms)}")

    tp = sum(1 for d, _, _ in recs if d)
    fp = sum(f for _, f, _ in recs)
    n_cp = len(recs)
    lo, hi = wilson_ci(tp, n_cp)
    print(f"\nSCPD {dataset}: {tp}/{n_cp} detected, {fp} FP, "
          f"precision {tp/(tp+fp):.3f}, Wilson [{lo:.2f},{hi:.2f}]")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", f"scpd_{dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"dataset": dataset, "thr": thr, "cp": cp, "tol": tol,
               "tp": tp, "n_cp": n_cp, "fp": fp,
               "precision": tp / (tp + fp) if (tp + fp) else 0.0,
               "recall_ci95": [lo, hi],
               "per_run": [{"tag": runs[t], "det": bool(recs[i][0]),
                            "fp": recs[i][1], "delay": recs[i][2]}
                           for i, t in enumerate(runs)]},
              open(out, "w", encoding="utf-8"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="email-eu")
    main(ap.parse_args().dataset)
