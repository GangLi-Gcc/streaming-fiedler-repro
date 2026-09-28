"""P10 O2: reconcile the measured rtr^0.75 exponent with gate theory (rtr^1).

Gate boundary: theta* solves  sin(theta*)*sigma_w(theta*) = rtr*gap, i.e.
theta* = rtr*gap/sigma_w(theta*).  sigma_w(theta) is NOT constant: the rotation
direction changes with theta (probe: kappa varies across delta).  If locally
sigma_w ~ theta^p, the implicit equation gives

    theta* ~ rtr^(1/(1+p))

so the measured 0.75 exponent implies p = 1/3  (elasticity of sigma_w in theta).

Competing hypothesis H_mix: theta* = min(gate boundary ~rtr^1, rtr-independent
refine-pinning floor); slopes would fall toward 0 for the floored configs.

Test: rebuild each config, compute the sigma_w(theta) curve, measure the local
elasticity p = dln sigma_w / dln theta at theta*(rtr=0.03), predict
slope_pred = 1/(1+p), and compare with the per-config measured local slope of
log theta* vs log rtr (from runs/synth_tol_dep.json, abs_tol=1e-6).

Output: runs/mdr_exponent_check.json
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from synth_grid import laplacian, align_cos  # noqa: E402
from synth_gap_scan import build_hier, DELTA_GRID, BLK  # noqa: E402
from synth_grid import fiedler_exact  # noqa: E402

RTRS = [0.01, 0.03, 0.1]


def sigma_w_curve(edges0, n, a, b):
    L0, _ = laplacian(edges0, n)
    _, vecs0 = fiedler_exact(L0, k=3)
    v0 = np.asarray(vecs0[:, 1])
    out = []
    for delta in DELTA_GRID:
        e1 = dict(edges0)
        e1[(min(a, b), max(a, b))] = e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
        L1, _ = laplacian(e1, n)
        _, vecs1 = fiedler_exact(L1, k=3)
        v2_1 = np.asarray(vecs1[:, 1])
        cos_rot = align_cos(v2_1, v0)
        theta = float(np.arccos(np.clip(cos_rot, -1, 1)))
        lam = float(v0 @ (L1 @ v0))
        res = float(np.linalg.norm(L1 @ v0 - lam * v0))
        out.append({"theta": theta, "sigma_w": res / max(np.sin(theta), 1e-12)})
    return out


def local_elasticity(curve, theta0):
    th = np.array([c["theta"] for c in curve])
    sg = np.array([c["sigma_w"] for c in curve])
    good = (th > 1e-8) & np.isfinite(sg) & (sg > 0)
    th, sg = th[good], sg[good]
    if len(th) < 4 or theta0 < th.min() or theta0 > th.max():
        return np.nan
    lx, ly = np.log(th), np.log(sg)
    i = int(np.argmin(np.abs(lx - np.log(theta0))))
    lo, hi = max(0, i - 2), min(len(lx), i + 3)
    if hi - lo < 3:
        return np.nan
    return float(np.polyfit(lx[lo:hi], ly[lo:hi], 1)[0])


def main():
    tol_dep = json.load(open("../runs/synth_tol_dep.json"))
    from collections import defaultdict
    g = defaultdict(list)
    for r in tol_dep:
        if r["rtr"] is None or r["abs_tol"] is not None and abs(r["abs_tol"] - 1e-6) > 0:
            continue
        g[(r["p_near"], r["w_hub"], r["seed"], r["rtr"])].append(r)
    ts = {}
    for k, rows in g.items():
        rows = sorted(rows, key=lambda r: r["theta"])
        hit = [r for r in rows if r["att"] >= 0.5 and r["theta"] > 0]
        ts[k] = hit[0]["theta"] if hit else np.nan

    cfgs = sorted(set((p, w, s) for (p, w, s, _) in ts))
    recs = []
    for cfg in cfgs:
        p_near, w_hub, seed = cfg
        thetas = {rtr: ts.get((p_near, w_hub, seed, rtr), np.nan) for rtr in RTRS}
        xs = [r for r in RTRS if np.isfinite(thetas[r]) and thetas[r] > 0]
        ys = [thetas[r] for r in xs]
        meas_slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if len(xs) >= 3 else np.nan
        rng = np.random.default_rng(3000 + seed)
        edges0, n = build_hier(rng, p_near, w_hub)
        a = int(rng.choice(list(BLK(0))))
        b = int(rng.choice(list(BLK(2))))
        curve = sigma_w_curve(edges0, n, a, b)
        t03 = thetas.get(0.03, np.nan)
        p_el = local_elasticity(curve, t03) if np.isfinite(t03) else np.nan
        pred_slope = 1.0 / (1.0 + p_el) if np.isfinite(p_el) else np.nan
        recs.append({"cfg": list(cfg), "theta_star_by_rtr": {str(k): v for k, v in thetas.items()},
                     "meas_slope": meas_slope, "elasticity_p": p_el,
                     "pred_slope": pred_slope})
        print(f"{cfg} meas={meas_slope:.3f} p={p_el:.3f} pred={pred_slope:.3f}",
              flush=True)

    with open("../runs/mdr_exponent_check.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    both = [r for r in recs if np.isfinite(r["meas_slope"]) and np.isfinite(r["pred_slope"])]
    m = np.array([r["meas_slope"] for r in both])
    pr = np.array([r["pred_slope"] for r in both])
    print(f"\nn={len(both)} corr(meas, pred) = {np.corrcoef(m, pr)[0,1]:.3f}")
    print(f"meas: med={np.median(m):.3f} range=[{np.min(m):.3f},{np.max(m):.3f}]")
    print(f"pred: med={np.median(pr):.3f} range=[{np.min(pr):.3f},{np.max(pr):.3f}]")
    print(f"|meas-pred|: med={np.median(np.abs(m-pr)):.3f} max={np.max(np.abs(m-pr)):.3f}")
    print("\nsaved runs/mdr_exponent_check.json")


if __name__ == "__main__":
    main()
