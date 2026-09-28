"""P0n-part2: detection channels on per-component features (CollegeMsg).

Channels (all on the segment [STATIONARY0:], same coordinates/protocol as
P0j / lad_baseline):
  lamDom-z : window-z on dominant-component lambda2
  vecD-dom : window-z on intersection-restricted cumulative drift of the
             dominant component's Fiedler vector (sign-aligned, cosine
             renormalized on the node intersection with lag L_CUM)
  ncomp-z  : window-z on component count (exact integer, solver-noise-free)
  ndom-z   : diagnostic only (dominant component size)

Calibration: strict zero-alarm budget on control + 1e-6 margin (identical
to P0i/P0j). Evaluation: det if any alarm in [cp, cp+40], fp = alarms
before cp, delay = first alarm - cp.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lad_baseline import window_z_alarms, evaluate, set_dataset  # noqa: E402

RUNS = ["college-msg", "cmsg-inj015a", "cmsg-inj015b", "cmsg-inj030a",
        "cmsg-inj030b", "cmsg-inj050a", "cmsg-inj050b", "cmsg-sc06a",
        "cmsg-sc06b"]


def load(tag):
    d = np.load(f"../runs/pc_{tag}.npz")
    return {k: d[k] for k in d.files}


def drift_series_dom(v2, mask, L=10):
    T, n = v2.shape
    va = np.array(v2, dtype=float, copy=True)
    for t in range(1, T):
        inter = mask[t] & mask[t - 1]
        if inter.sum() >= 2 and np.dot(va[t][inter], va[t - 1][inter]) < 0:
            va[t] = -va[t]
    D = np.full(T, np.nan)
    for t in range(L, T):
        inter = mask[t] & mask[t - L]
        if inter.sum() < 2:
            continue
        a = va[t][inter]
        b = va[t - L][inter]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        D[t] = 1.0 - min(1.0, abs(float(np.dot(a, b)) / (na * nb)))
    return D


def main():
    set_dataset("college-msg")
    seg0 = lb_seg0 = 20            # STATIONARY0 for college-msg
    cp = 120 - seg0                # INJECT_SEG_IDX in segment coords
    data = {t: load(t) for t in RUNS}

    ch = {}
    for t in RUNS:
        d = data[t]
        s = slice(seg0, None)
        ch[t] = {"lamDom": np.asarray(d["lam_dom"], dtype=float)[s],
                 "ncomp": np.asarray(d["ncomp"], dtype=float)[s],
                 "ndom": np.asarray(d["ndom"], dtype=float)[s],
                 "vecDdom": drift_series_dom(
                     np.asarray(d["v2_dom"], dtype=float),
                     np.asarray(d["mask_dom"], dtype=bool))[s]}

    ctrl = ch["college-msg"]
    cal = {}
    for name in ("lamDom", "vecDdom", "ncomp", "ndom"):
        _, zmax = window_z_alarms(ctrl[name], 1e9)
        cal[name] = zmax * (1 + 1e-6)
    print("control calibration:", {k: f"{v:.3e}" for k, v in cal.items()})

    # ---- control-flow sanity: value distributions ----
    print("\ncontrol value ranges:")
    for name in ("lamDom", "ncomp", "ndom", "vecDdom"):
        x = ctrl[name]
        xf = x[np.isfinite(x)]
        print(f"  {name:8s}: [{np.min(xf):.3e}, {np.max(xf):.3e}] "
              f"median={np.median(xf):.3e}")
    print(f"  control ncomp==1 fraction: {np.mean(ctrl['ncomp'] == 1):.3f}")

    rows = []
    for t in RUNS[1:]:
        row = {"run": t}
        for name in ("lamDom", "vecDdom", "ncomp"):
            alarms, _ = window_z_alarms(ch[t][name], cal[name])
            d, f, dl = evaluate(alarms, cp=cp)
            row[name] = [bool(d), int(f), (int(dl) if dl is not None else None)]
            row[f"{name}_nalarms"] = len(alarms)
        rows.append(row)
        print(f"[{t}] lamDom:{row['lamDom'][0]}/{row['lamDom'][1]}"
              f"(d{row['lamDom'][2]}) vecDdom:{row['vecDdom'][0]}/{row['vecDdom'][1]}"
              f"(d{row['vecDdom'][2]}) ncomp:{row['ncomp'][0]}/{row['ncomp'][1]}"
              f"(d{row['ncomp'][2]})")

    totals = {name: sum(r[name][0] for r in rows) for name in ("lamDom", "vecDdom", "ncomp")}
    fps = {name: sum(r[name][1] for r in rows) for name in ("lamDom", "vecDdom", "ncomp")}
    print(f"\ntotals: {totals}/8, total fp: {fps}")
    print("P0j reference (global channels, raw): vecD 5/8 (inj050b fp=10), "
          "lam-z 0/8 (degenerate), LAD 2/8")

    with open("../runs/percomp_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"calibration": cal, "rows": rows, "totals": totals,
                   "fps": fps, "inject_seg_idx": cp, "tol": 40}, f, indent=2)


if __name__ == "__main__":
    main()
