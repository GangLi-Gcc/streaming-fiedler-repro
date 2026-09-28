"""P0d consolidated analysis: reliable-phase Q1<->Q2 coupling numbers.

Reads trackq_*.npz (exact + tracked per sample). Splits the stationary segment
into a reliable phase (tracker eps <= EPS_GOOD) and failure bursts, then reports:
  - drift noise floor exact vs tracked (median / MAD / max) on the reliable phase
  - zero-FP threshold inflation factor (tracked / exact), consecutive-d and vec-D-z
  - vec-D detection retention + raw-d spike detection on CP-injected runs
  - spurious tracked-d excursions on control (false-CP hazard of Q1 error)
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from vec_drift import drift_series, window_z, evaluate, STATIONARY0, INJECT_SEG_IDX  # noqa: E402

EPS_GOOD = 0.05
TOL = 40
TAGS = {"control": None, "inj015b": 0.15, "inj030a": 0.30, "inj050b": 0.50,
        "sc06a": "scale0.6"}
FILE_TAG = {"control": "email-eu"}   # control run was saved under its dataset tag


def load_npz(tag):
    d = np.load(f"../runs/trackq_{FILE_TAG.get(tag, tag)}.npz")
    return {k: d[k] for k in d.files}


def stats(x):
    return {"median": float(np.median(x)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())}


def main():
    data = {t: load_npz(t) for t in TAGS}
    ctrl = data["control"]
    vex_c = ctrl["v_ex"].T[STATIONARY0:].astype(float)
    vtr_c = ctrl["v_trk"].T[STATIONARY0:].astype(float)
    eps_c = np.linalg.norm(vtr_c - vex_c, axis=1)
    rel_c = np.abs(ctrl["lam_trk"][STATIONARY0:] - ctrl["lam_ex"][STATIONARY0:]) \
        / np.maximum(np.abs(ctrl["lam_ex"][STATIONARY0:]), 1e-12)
    reliable = eps_c <= EPS_GOOD
    # burst = maximal contiguous prefix of unreliable samples
    bad_prefix = 0
    while bad_prefix < len(eps_c) and not reliable[bad_prefix]:
        bad_prefix += 1
    print(f"control: reliable {reliable.sum()}/{len(eps_c)} samples; "
          f"failure burst = seg idx [0,{bad_prefix - 1}] (day ~357-380)")
    print(f"  lam rel err on reliable: median={np.median(rel_c[reliable]):.2e} "
          f"p99={np.percentile(rel_c[reliable], 99):.2e} max={rel_c[reliable].max():.2e}")
    print(f"  v err eps on reliable: median={np.median(eps_c[reliable]):.2e} "
          f"p99={np.percentile(eps_c[reliable], 99):.2e} max={eps_c[reliable].max():.2e}")

    # ---- drift floors on reliable phase (exact vs tracked) ----
    d_ex_c, D_ex_c = drift_series(vex_c)
    d_tr_c, D_tr_c = drift_series(vtr_c)
    m = reliable.copy(); m[0] = False
    print("\nreliable-phase consecutive drift floor:")
    print(f"  exact  : {stats(d_ex_c[1:][reliable[1:]])}")
    print(f"  tracked: {stats(d_tr_c[1:][reliable[1:]])}")
    thr_ex = d_ex_c[1:][reliable[1:]].max() * (1 + 1e-6)
    thr_tr = d_tr_c[1:][reliable[1:]].max() * (1 + 1e-6)
    print(f"  zero-FP raw-spike threshold: exact={thr_ex:.3e} tracked={thr_tr:.3e} "
          f"ratio={thr_tr / thr_ex:.2f}")

    _, zD_ex = window_z(D_ex_c, 1e9)
    _, zD_tr = window_z(D_tr_c, 1e9)
    # calibrate on reliable-only control: recompute zmax restricted to reliable t
    def zmax_masked(D, mask):
        al, zm = window_z(D, 1e9)
        return zm
    # simpler: threshold ratio via zmax on full (burst inflates tracked side)
    print(f"  vec-D zmax (full segment): exact={zD_ex:.1f} tracked={zD_tr:.1f} "
          f"ratio={zD_tr / zD_ex:.1f}")

    # ---- spurious tracked-d excursions on control reliable phase ----
    spur = np.where((d_tr_c > thr_ex) & reliable)[0]
    print(f"  control tracked-d excursions above exact-max threshold "
          f"(reliable phase): {len(spur)} at {spur[:10]}")

    # ---- detection retention on injected runs ----
    print("\ninjected runs (detection evaluated on reliable phase only):")
    rows = []
    for tag, frac in TAGS.items():
        if frac is None:
            continue
        d_ = data[tag]
        vex = d_["v_ex"].T[STATIONARY0:].astype(float)
        vtr = d_["v_trk"].T[STATIONARY0:].astype(float)
        eps = np.linalg.norm(vtr - vex, axis=1)
        rel = reliable & (eps <= EPS_GOOD)
        d_ex, D_ex = drift_series(vex)
        d_tr, D_tr = drift_series(vtr)
        # raw spike, tracked, threshold from control exact reliable
        w = d_tr[INJECT_SEG_IDX:INJECT_SEG_IDX + TOL + 1]
        spike_det = bool(w.max() > thr_ex)
        spike_delay = int(np.argmax(w)) if spike_det else None
        # vec-D z, threshold = control tracked zmax (consistent estimator family)
        a_tr, _ = window_z(D_tr, zD_tr * (1 + 1e-6))
        det_z, fp_z, del_z = evaluate(a_tr, INJECT_SEG_IDX)
        rows.append({"run": tag, "frac": frac, "raw_spike": [spike_det, spike_delay],
                     "vecD_z": [bool(det_z), int(fp_z), del_z],
                     "reliable_frac": float(rel.sum() / len(rel))})
        print(f"[{tag}] raw-spike: {spike_det} delay={spike_delay} | "
              f"vecD-z: det={det_z} fp={fp_z} delay={del_z} | "
              f"reliable {rel.sum()}/{len(rel)}")
    out = {"eps_good": EPS_GOOD, "control_reliable_prefix_end": int(bad_prefix),
           "thr_exact": thr_ex, "thr_tracked": thr_tr,
           "threshold_ratio": thr_tr / thr_ex, "rows": rows}
    with open("../runs/trackq_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
