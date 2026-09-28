"""Detection on the real email-Eu stream: OU-null residual CUSUM vs window-z.

Null model: AR(1)/OU on the stationary lambda_2 segment
    x_t = mu + phi * (x_{t-1} - mu) + eps_t        (fit on control segment)
Detector A (residual CUSUM): innovations e_t, one-sided CUSUM with allowance k.
Detector B (window-z): task-#2 median/MAD statistic.
Both are calibrated on the CONTROL segment under a zero-alarm budget, then
applied to CP-injected trajectories (injection at global sample 400 ->
segment index 100). Also checks whether a 1-alarm-budget calibration fires
near the REAL candidate changepoint at segment sample ~145 (day ~510).
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
STATIONARY0 = 300
INJECT_SEG_IDX = 400 - STATIONARY0
TOL = 40
RUNS = {
    "control": None,
    "frac0.15a": ("inj015a", 0.15), "frac0.15b": ("inj015b", 0.15),
    "frac0.30a": ("inj030a", 0.30), "frac0.30b": ("inj030b", 0.30),
    "frac0.50a": ("inj050a", 0.50), "frac0.50b": ("inj050b", 0.50),
    "scale0.6a": ("sc06a", "scale0.6"), "scale0.6b": ("sc06b", "scale0.6"),
}


def load_seg(tag):
    d = np.load(f"../runs/real_{tag}_lam.npz")
    return np.asarray(d["lam"][STATIONARY0:], dtype=float)


def fit_ar1(x):
    mu = x.mean()
    y = x - mu
    phi = float(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]))
    e = y[1:] - phi * y[:-1]
    return mu, phi, float(e.std())


def innovations(x, mu, phi, sig):
    y = x - mu
    e = (y[1:] - phi * y[:-1]) / (sig + 1e-12)
    return np.concatenate([[0.0], e])


def cusum_run(e, h, k=0.5):
    S, alarms = 0.0, []
    for t, v in enumerate(e):
        S = max(0.0, S + v - k)
        if S > h:
            alarms.append(t)
            S = 0.0
    return alarms


def zscore_run(x, thr, win_base=30, win_recent=10, mad_window=40):
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


def evaluate(alarms, cps, tol=TOL):
    det = any(0 <= a - cps[0] <= tol for a in alarms)
    fp = len([a for a in alarms if a < cps[0]])
    delay = min((a - cps[0] for a in alarms if 0 <= a - cps[0] <= tol), default=None)
    return det, fp, delay


def main():
    control = load_seg("email-eu")
    mu, phi, sig = fit_ar1(control)
    print(f"AR(1) fit on control stationary: mu={mu:.3f} phi={phi:.3f} sigma_eps={sig:.4f}")

    e_ctrl = innovations(control, mu, phi, sig)
    # --- calibrate under zero-alarm budget on control ---
    S = 0.0
    s_trace = []
    for v in e_ctrl:
        S = max(0.0, S + v - 0.5)
        s_trace.append(S)
        if S > 0:  # track only pre-reset maxima; resets never happen on control by construction of h below
            pass
    # calibration under a strict zero-alarm budget on the control segment;
    # add a tiny relative margin so bit-level eigsh nondeterminism across
    # independently-generated runs cannot flip a strict > at the peak point
    h_cal = max(s_trace) * (1 + 1e-6)
    z_alarms_ctrl, zmax_ctrl = zscore_run(control, thr=1e9)
    thr_z_cal = zmax_ctrl * (1 + 1e-6)
    print(f"calibration: h_cusum={h_cal:.2f}  thr_z={thr_z_cal:.2f}")

    # --- 1-alarm-budget: where does each detector fire on control? (day-510 check) ---
    S = 0.0
    alarm1 = None
    for t, v in enumerate(e_ctrl):
        S = max(0.0, S + v - 0.5)
        if S > 0.85 * h_cal and alarm1 is None:   # slightly relaxed budget
            alarm1 = t
            break
    if alarm1 is not None:
        ts_seg = np.load("../runs/real_email-eu_lam.npz")["ts"][STATIONARY0:]
        print(f"control, relaxed budget: residual-CUSUM first alarm at segment "
              f"sample {alarm1} (day ~{ts_seg[alarm1] / 86400:.0f})")
    else:
        print("control, relaxed budget: no alarm")

    rows = []
    for name, spec in RUNS.items():
        if spec is None:
            continue
        tag, frac = spec
        x = load_seg(tag)
        e = innovations(x, mu, phi, sig)
        a_cus = cusum_run(e, h_cal)
        a_z, _ = zscore_run(x, thr_z_cal)
        d1, fp1, del1 = evaluate(a_cus, [INJECT_SEG_IDX])
        d2, fp2, del2 = evaluate(a_z, [INJECT_SEG_IDX])
        print(f"    raw alarms: cusum={a_cus[:8]}{'...' if len(a_cus) > 8 else ''}  z={a_z[:8]}{'...' if len(a_z) > 8 else ''}")
        shift = float(x[INJECT_SEG_IDX + 15: INJECT_SEG_IDX + 40].mean()
                      - x[INJECT_SEG_IDX - 40: INJECT_SEG_IDX - 10].mean())
        rows.append({"run": name, "frac": frac,
                     "shift_lambda": round(shift, 3),
                     "cusum_det": d1, "cusum_fp": fp1, "cusum_delay": del1,
                     "z_det": d2, "z_fp": fp2, "z_delay": del2})
        print(f"[{name}] shift={shift:+.3f}  cusum: det={d1} fp={fp1} delay={del1}   "
              f"z: det={d2} fp={fp2} delay={del2}")

    with open("../runs/ou_detection_real.json", "w", encoding="utf-8") as f:
        json.dump({"ar1": {"mu": mu, "phi": phi, "sigma_eps": sig},
                   "calibration": {"h_cusum": h_cal, "thr_z": thr_z_cal},
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    det = {}
    for r in rows:
        det.setdefault(r["frac"], []).append(r)
    print("\ndetection rate by injected fraction (2 seeds each):")
    for frac, rs in sorted(det.items(), key=lambda kv: str(kv[0])):
        c = sum(r["cusum_det"] for r in rs); z = sum(r["z_det"] for r in rs)
        print(f"  frac={frac}: cusum {c}/{len(rs)}  window-z {z}/{len(rs)}  "
              f"shifts={[r['shift_lambda'] for r in rs]}")


if __name__ == "__main__":
    main()
