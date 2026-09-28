"""P0o-part2: evaluate vec-D and lam-z on the tracker trajectory.

Channels on segment [STATIONARY0:], same protocol as P0i/P0j/P0n:
  vecD-trk : window-z on cumulative drift D=1-|<v_t,v_{t-10}>| of the
             WarmStartTracker trajectory
  lam-trk  : window-z on tracker lambda2 (Rayleigh quotient along the
             continuation -- smooth in the disconnected regime)
Calibration: zero-alarm budget on control + 1e-6 margin.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lad_baseline import (window_z_alarms, evaluate, drift_series,  # noqa: E402
                          set_dataset)

RUNS = ["college-msg", "cmsg-inj015a", "cmsg-inj015b", "cmsg-inj030a",
        "cmsg-inj030b", "cmsg-inj050a", "cmsg-inj050b", "cmsg-sc06a",
        "cmsg-sc06b"]


def main():
    set_dataset("college-msg")
    seg0, cp = 20, 100
    ch = {}
    for t in RUNS:
        d = np.load(f"../runs/tv_{t}.npz")
        s = slice(seg0, None)
        ch[t] = {"vecD": drift_series(np.asarray(d["v2_trk"])[s]),
                 "lam": np.asarray(d["lam_trk"], dtype=float)[s]}

    ctrl = ch["college-msg"]
    cal = {}
    for name in ("vecD", "lam"):
        _, zmax = window_z_alarms(ctrl[name], 1e9)
        cal[name] = zmax * (1 + 1e-6)
    print("control calibration:", {k: f"{v:.3e}" for k, v in cal.items()})
    xf = ctrl["vecD"][np.isfinite(ctrl["vecD"])]
    print(f"control vecD-trk range: [{xf.min():.3e}, {xf.max():.3e}] "
          f"median={np.median(xf):.3e}")
    xl = ctrl["lam"][np.isfinite(ctrl["lam"])]
    print(f"control lam-trk range: [{xl.min():.3e}, {xl.max():.3e}] "
          f"median={np.median(xl):.3e}")

    rows = []
    for t in RUNS[1:]:
        row = {"run": t}
        for name in ("vecD", "lam"):
            alarms, _ = window_z_alarms(ch[t][name], cal[name])
            d, f, dl = evaluate(alarms, cp=cp)
            row[name] = [bool(d), int(f), (int(dl) if dl is not None else None)]
            row[f"{name}_nalarms"] = len(alarms)
        rows.append(row)
        print(f"[{t}] vecD-trk:{row['vecD'][0]}/{row['vecD'][1]}"
              f"(d{row['vecD'][2]}, n={row['vecD_nalarms']}) "
              f"lam-trk:{row['lam'][0]}/{row['lam'][1]}"
              f"(d{row['lam'][2]}, n={row['lam_nalarms']})")

    totals = {n: sum(r[n][0] for r in rows) for n in ("vecD", "lam")}
    fps = {n: sum(r[n][1] for r in rows) for n in ("vecD", "lam")}
    print(f"\ntotals: {totals}/8, total fp: {fps}")
    print("P0j reference (fresh eigsh per snapshot): vecD 5/8 fp{inj050b:10}, "
          "lam-z 0/8, LAD 2/8")

    with open("../runs/trackvec_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"calibration": cal, "rows": rows, "totals": totals,
                   "fps": fps, "inject_seg_idx": cp, "tol": 40}, f, indent=2)


if __name__ == "__main__":
    main()
