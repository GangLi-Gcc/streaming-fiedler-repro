"""P0c: vector-drift statistic vs lambda_2 shift on the real email-Eu stream.

Statistic:  d_t = 1 - <v_t, v_{t-1}>   (consecutive, sign-aligned Fiedler vectors)
           D_t = 1 - <v_t, v_{t-L}>   (cumulative over L=10 samples)
Both are fed to the same window-z detector as task#2 / ou_detection, calibrated
on the CONTROL segment under a zero-alarm budget (+ulp margin), then applied to
CP-injected trajectories. Compares detection with the lambda_2-based z from P0b,
and checks the real changepoint at day~388 (segment idx ~37).
"""
from __future__ import annotations

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")
STATIONARY0 = 300
INJECT_SEG_IDX = 400 - STATIONARY0
TOL = 40
L_CUM = 10
RUNS = {
    "control": None,
    "frac0.15a": ("inj015a", 0.15), "frac0.15b": ("inj015b", 0.15),
    "frac0.30a": ("inj030a", 0.30), "frac0.30b": ("inj030b", 0.30),
    "frac0.50a": ("inj050a", 0.50), "frac0.50b": ("inj050b", 0.50),
    "scale0.6a": ("sc06a", "scale0.6"), "scale0.6b": ("sc06b", "scale0.6"),
}


def load(tag):
    d = np.load(f"../runs/real_{tag}_lam.npz")
    lam = np.asarray(d["lam"], dtype=float)
    v2 = np.asarray(d["v2"], dtype=float).T      # (samples, n)
    ts = np.asarray(d["ts"], dtype=float) / 86400.0
    return lam, v2, ts


def align(v):
    out = np.array(v, dtype=float, copy=True)
    for t in range(1, len(out)):
        if np.dot(out[t], out[t - 1]) < 0:
            out[t] = -out[t]
    return out


def drift_series(v):
    va = align(v)
    d = np.zeros(len(v))
    for t in range(1, len(v)):
        d[t] = 1.0 - min(1.0, max(-1.0, float(np.dot(va[t], va[t - 1]))))
    D = np.full(len(v), np.nan)
    for t in range(L_CUM, len(v)):
        D[t] = 1.0 - min(1.0, abs(float(np.dot(va[t], va[t - L_CUM]))))
    return d, D


def window_z(x, thr, win_base=30, win_recent=10, mad_window=40):
    alarms, zmax = [], 0.0
    for t in range(win_base + win_recent, len(x)):
        base = np.median(x[t - win_base - win_recent: t - win_recent])
        recent = np.median(x[t - win_recent: t])
        inc = np.diff(x[max(0, t - mad_window):t])
        sig = 1.4826 * np.median(np.abs(inc - np.median(inc))) + 1e-12
        z = (recent - base) / (sig * np.sqrt(win_recent))
        zmax = max(zmax, abs(z))
        if abs(z) > thr:
            alarms.append(t)
    return alarms, zmax


def evaluate(alarms, cp, tol=TOL):
    det = any(0 <= a - cp <= tol for a in alarms)
    fp = len([a for a in alarms if a < cp])
    delay = min((a - cp for a in alarms if 0 <= a - cp <= tol), default=None)
    return det, fp, delay


def main():
    lam_c, v_c, ts_c = load("email-eu")
    seg = slice(STATIONARY0, None)
    lam_seg, v_seg, ts_seg = lam_c[seg], v_c[seg], ts_c[seg]

    d_seg, D_seg = drift_series(v_seg)
    # ---- calibrate both drift statistics on control (zero-FP + ulp margin) ----
    _, zmax_d = window_z(d_seg, 1e9)
    _, zmax_D = window_z(D_seg[np.isfinite(D_seg)], 1e9)
    thr_d, thr_D = zmax_d * (1 + 1e-6), zmax_D * (1 + 1e-6)
    print(f"control drift baselines: max consecutive d={d_seg[1:].max():.2e} "
          f"median={np.median(d_seg[1:]):.2e};  zmax_d={zmax_d:.1f} zmax_D={zmax_D:.1f}")

    # ---- real CP day~388 (seg idx 37): does vector drift spike there? ----
    i38 = int(np.argmin(np.abs(ts_seg - 388)))
    loc = d_seg[max(1, i38 - 30):i38 + 30]
    mad = 1.4826 * np.median(np.abs(loc - np.median(loc))) + 1e-12
    print(f"day~388 -> seg idx {i38}: d={d_seg[i38]:.2e} vs local median "
          f"{np.median(loc):.2e} (local z={(d_seg[i38]-np.median(loc))/mad:.1f}); "
          f"D={D_seg[min(len(D_seg)-1, i38+L_CUM)]:.2e}")
    imax = int(np.argmax(d_seg[1:])) + 1
    print(f"global max consecutive drift at seg idx {imax} (day ~{ts_seg[imax]:.0f}), "
          f"d={d_seg[imax]:.2e}")

    # ---- moderate-threshold scan on control: where does drift-z fire? ----
    al_d, _ = window_z(d_seg, thr=max(10.0, zmax_d * 0.05))
    al_D, _ = window_z(D_seg, thr=max(10.0, zmax_D * 0.05))
    print(f"control @5% zmax: consecutive-d alarms={al_d[:12]}"
          f"{'...' if len(al_d) > 12 else ''}  cumulative-D alarms={al_D[:12]}"
          f"{'...' if len(al_D) > 12 else ''}")

    # ---- injected runs: vector-z detection vs lambda2-z detection ----
    rows = []
    for name, spec in RUNS.items():
        if spec is None:
            continue
        tag, frac = spec
        lam, v, ts = load(tag)
        d, D = drift_series(v[seg])
        a_d, _ = window_z(d, thr_d)
        a_D, _ = window_z(D, thr_D)
        a_l, _ = window_z(lam[seg], thr=2674.39 * (1 + 1e-6))
        vd, fd, dd = evaluate(a_d, INJECT_SEG_IDX)
        vD, fD, dD = evaluate(a_D, INJECT_SEG_IDX)
        vl, fl, dl = evaluate(a_l, INJECT_SEG_IDX)
        rows.append({"run": name, "frac": frac,
                     "vec_d": [vd, fd, dd], "vec_D": [vD, fD, dD],
                     "lam": [vl, fl, dl]})
        print(f"[{name}] vec-d z: det={vd} fp={fd} delay={dd} | "
              f"vec-D z: det={vD} fp={fD} delay={dD} | lam z: det={vl} fp={fl} delay={dl}")

    with open("../runs/vec_drift_real.json", "w", encoding="utf-8") as f:
        json.dump({"calibration": {"thr_d": thr_d, "thr_D": thr_D},
                   "day388_seg_idx": int(i38), "rows": rows}, f, indent=2)
    for label, key in [("vec-d", "vec_d"), ("vec-D", "vec_D"), ("lam", "lam")]:
        tot = sum(r[key][0] for r in rows)
        print(f"{label}: {tot}/{len(rows)} detected")


if __name__ == "__main__":
    main()
