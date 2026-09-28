"""P10 theory check: gate-boundary prediction of theta* (the MDR angle).

Theory (from the P10 mechanism probe):
  1. Tracker branch is decided by the restart gate: res > rtr*gap -> exact
     restart (faithful, att=1); else kmax-step power refinement (pinned).
  2. Warm-start residual of the old Fiedler vector against the perturbed
     Laplacian: res(theta) = sin(theta) * sigma_w, where sigma_w is the
     Rayleigh-quotient spread of the rotation direction w over L's spectrum.
  3. Gate boundary: sin(theta*) = rtr*gap/sigma_w
     -> theta* = rtr*gap/sigma_w  ->  k = theta*/sqrt(s) = rtr*gap/(sigma_w*sqrt(s)).

This script rebuilds each of the 43 k-band configs (same seeds and builder
as synth_gap_scan), recomputes sigma_w(theta) on the delta grid, interpolates
sigma_w at the measured theta*, and reports:
  - theta_pred = rtr*gap/sigma_w(theta*) vs measured theta_star (ratio)
  - the combination C = sigma_w*sqrt(s)/gap, predicted constant if the gate
    theory explains the k-band collapse.

Output: runs/mdr_gate_theory.json
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import scipy.linalg as sla

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from synth_grid import laplacian, align_cos  # noqa: E402
from synth_gap_scan import build_hier, DELTA_GRID  # noqa: E402

RTR = 0.03
Q99_FLOOR = 8.0e-7   # unused here, kept for parity with the scan protocol


def sigma_w_curve(edges0, n, a, b):
    """sigma_w(delta) = res(v0)/sin(theta) with v0 the base Fiedler vector."""
    L0, _ = laplacian(edges0, n)
    vals0, vecs0 = sla.eigh(L0.toarray())
    v0 = vecs0[:, 1]
    out = []
    for delta in DELTA_GRID:
        e1 = dict(edges0)
        e1[(min(a, b), max(a, b))] = e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
        L1, deg1 = laplacian(e1, n)
        vals1, vecs1 = sla.eigh(L1.toarray())
        v2_1 = vecs1[:, 1]
        cos_rot = align_cos(v2_1, v0)
        theta = float(np.arccos(np.clip(cos_rot, -1, 1)))
        lam = float(v0 @ (L1 @ v0))
        res = float(np.linalg.norm(L1 @ v0 - lam * v0))
        s = float(2.0 * deg1.max() + 1.0 - vals1[1])
        gap = float(vals1[2] - vals1[1])
        sig = res / max(np.sin(theta), 1e-12)
        out.append({"delta": float(delta), "theta": theta, "sigma_w": sig,
                    "stiff": s, "gap": gap})
    return out


def interp_sigma(curve, theta_star):
    th = np.array([c["theta"] for c in curve])
    sg = np.array([c["sigma_w"] for c in curve])
    good = (th > 1e-8) & np.isfinite(sg) & (sg > 0)
    th, sg = th[good], sg[good]
    if len(th) < 2 or theta_star < th.min() or theta_star > th.max():
        return np.nan
    return float(np.exp(np.interp(np.log(theta_star), np.log(th), np.log(sg))))


def main():
    kband = json.load(open("../runs/p0k_kband.json"))
    key2ts = {(r["p_near"], r["w_hub"], r["seed"]): r for r in kband}
    from synth_gap_scan import N_BLK, BLK  # noqa

    recs = []
    for (p_near, w_hub, seed), kr in sorted(key2ts.items()):
        rng = np.random.default_rng(3000 + seed)
        edges0, n = build_hier(rng, p_near, w_hub)
        a = int(rng.choice(list(BLK(0))))
        b = int(rng.choice(list(BLK(2))))
        curve = sigma_w_curve(edges0, n, a, b)
        theta_star = kr["theta_star"]
        sig = interp_sigma(curve, theta_star)
        s = kr["sqrt_stiff"] ** 2
        gap = np.nanmedian([c["gap"] for c in curve])
        theta_pred = RTR * gap / sig if np.isfinite(sig) and sig > 0 else np.nan
        rec = {
            "p_near": p_near, "w_hub": w_hub, "seed": seed,
            "theta_star": theta_star, "theta_pred": theta_pred,
            "ratio": theta_star / theta_pred if np.isfinite(theta_pred) else np.nan,
            "sigma_w": sig, "gap": gap, "stiff": s,
            "C": sig * np.sqrt(s) / gap if np.isfinite(sig) else np.nan,
        }
        recs.append(rec)
        print(f"p={p_near} w={w_hub} s={seed} theta*={theta_star:.2e} "
              f"pred={theta_pred:.2e} ratio={rec['ratio']:.2f} C={rec['C']:.3f}",
              flush=True)

    with open("../runs/mdr_gate_theory.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    rats = [r["ratio"] for r in recs if np.isfinite(r["ratio"])]
    Cs = [r["C"] for r in recs if np.isfinite(r["C"])]
    print(f"\nratio theta*/pred: med={np.median(rats):.2f} "
          f"range=[{np.min(rats):.2f},{np.max(rats):.2f}]")
    print(f"C=sigma_w*sqrt(s)/gap: med={np.median(Cs):.3f} "
          f"range=[{np.min(Cs):.3f},{np.max(Cs):.3f}] "
          f"spread(max/min)={np.max(Cs)/np.min(Cs):.1f}x")
    implied_k = RTR / np.array(Cs)
    print(f"implied k = rtr/C: med={np.median(implied_k):.2e} "
          f"range=[{np.min(implied_k):.2e},{np.max(implied_k):.2e}]")
    print("\nsaved runs/mdr_gate_theory.json")


if __name__ == "__main__":
    main()
