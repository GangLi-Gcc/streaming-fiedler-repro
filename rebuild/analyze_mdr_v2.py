#!/usr/bin/env python3
"""rebuild/analyze_mdr_v2.py — directional-fidelity validation of the MDR.

The reviewer's CRITICAL objection: MDR is defined by att = d_tr/d_ex, a
*movement-magnitude* ratio, while the G-REST section calls that ratio misleading
(it measures step size, not whether the step points at the true vector).

This script answers it directly with the rebuilt data, which now records
d_end = 1 - |<v_tr, v1_exact>| (endpoint error) and
capture_dir = 1 - d_end/d_ex (fraction of the true rotation actually closed).

Two things are established here:
  1. In the small-rotation regime both att and capture_dir are PINNED at
     constants (power iteration takes a fixed number of steps before the gate
     fires, so the tracked step is a fixed function of the true rotation).  The
     pinned constants differ — att measures step size, capture_dir measures
     directional closure — which is exactly the reviewer's point.
  2. Nevertheless, wherever att crosses its 0.5 threshold, capture_dir has
     ALREADY crossed 0.5.  So the att-based MDR is a conservative (upper-bound)
     statement of directional fidelity, not a wrong one.

Outputs a table the paper can cite verbatim.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = json.load(open(os.path.join(ROOT, "rebuild", "results", "mdr_configs.json"),
                      encoding="utf-8"))
DELTAS = sorted({r["delta"] for r in recs})


def seed_avg(p_near, w_hub, delta, key):
    g = [r for r in recs if r["p_near"] == p_near and r["w_hub"] == w_hub
         and r["delta"] == delta]
    return float(np.mean([r[key] for r in g]))


def med(p_near, w_hub, delta, key):
    g = [r for r in recs if r["p_near"] == p_near and r["w_hub"] == w_hub
         and r["delta"] == delta]
    return float(np.median([r[key] for r in g]))


print("=== Part 1: att-based MDR, and directional fidelity at the crossing ===")
print(f"{'p_near':>6} {'w':>4} {'theta*':>9} {'att@*':>7} {'cap_dir@*':>10} "
      f"{'d_end/d_ex@*':>12} {'cap0':>6} {'att0':>6}")
rows = []
for pn in sorted({r["p_near"] for r in recs}):
    for wh in sorted({r["w_hub"] for r in recs}):
        # crossing of att (seed-averaged) >= 0.5
        below = [(d, seed_avg(pn, wh, d, "att")) for d in DELTAS
                 if seed_avg(pn, wh, d, "att") < 0.5]
        above = [(d, seed_avg(pn, wh, d, "att")) for d in DELTAS
                 if seed_avg(pn, wh, d, "att") >= 0.5]
        if not above:
            continue
        if not below:
            # unfrozen under att: att >= 0.5 already at smallest delta
            continue
        lo = max(below, key=lambda x: x[0])[0]
        hi = min(above, key=lambda x: x[0])[0]
        # geometric midpoint of the bracket in theta
        th_lo = med(pn, wh, lo, "theta")
        th_hi = med(pn, wh, hi, "theta")
        theta_star = float(np.sqrt(th_lo * th_hi))
        att_at = seed_avg(pn, wh, hi, "att")
        cap_at = seed_avg(pn, wh, hi, "capture_dir")
        dedx_at = seed_avg(pn, wh, hi, "d_end") / max(seed_avg(pn, wh, hi, "d_ex"), 1e-15)
        cap0 = seed_avg(pn, wh, DELTAS[0], "capture_dir")
        att0 = seed_avg(pn, wh, DELTAS[0], "att")
        gap = med(pn, wh, DELTAS[0], "gap")
        stiff = med(pn, wh, DELTAS[0], "stiff")
        rows.append(dict(p_near=pn, w_hub=wh, theta_star=theta_star, att_at=att_at,
                         cap_at=cap_at, dedx_at=dedx_at, cap0=cap0, att0=att0,
                         gap=gap, stiff=stiff))
        print(f"{pn:>6.3f} {wh:>4.1f} {theta_star:>9.2e} {att_at:>7.3f} "
              f"{cap_at:>10.3f} {dedx_at:>12.3f} {cap0:>6.3f} {att0:>6.3f}")

print(f"\n  n bracketed configs (att): {len(rows)}")
print(f"  all crossings have capture_dir >= 0.5: "
      f"{all(r['cap_at'] >= 0.5 for r in rows)}")
print(f"  min capture_dir at crossing: {min(r['cap_at'] for r in rows):.3f}")

# --- Part 2: pooled log-log regression (att-based MDR), reproduces 0.365 ---
X = np.vstack([np.log10([r["stiff"] for r in rows]),
               np.log10([r["gap"] for r in rows]),
               np.ones(len(rows))]).T
y = np.log10([r["theta_star"] for r in rows])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
r2 = 1 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
print(f"\n=== Part 2: pooled regression (n={len(rows)}) ===")
print(f"  a(stiffness)={beta[0]:+.3f}  b(gap)={beta[1]:+.3f}  R^2={r2:.3f}")
# single-factor free exponent + bootstrap CI
rng = np.random.default_rng(0)
X1 = np.vstack([np.log10([r["stiff"] for r in rows]), np.ones(len(rows))]).T
betas = []
for _ in range(2000):
    idx = rng.integers(0, len(rows), len(rows))
    b1, *_ = np.linalg.lstsq(X1[idx], y[idx], rcond=None)
    betas.append(b1[0])
print(f"  single-factor free exponent: {np.median(betas):.3f} "
      f"95% CI [{np.percentile(betas, 2.5):.3f}, {np.percentile(betas, 97.5):.3f}]")

# --- Part 3: the formula (1-kappa^k)^2 vs 1-kappa^{2k} ---
# In the pinned regime att = d_tr/d_ex and capture_dir = 1-d_end/d_ex are both
# constant.  Under a geodesic small-angle model att=(1-kk)^2, cap=1-kk^2, so
# att+cap = 2(1-kk).  Report how far the data are from that identity.
print("\n=== Part 3: geodesic-model check ===")
for r in rows:
    s = r["att0"] + r["cap0"]
    kk_geodesic = 1.0 - s / 2.0  # from att+cap = 2(1-kk)
    att_pred = (1.0 - kk_geodesic) ** 2
    cap_pred = 1.0 - kk_geodesic ** 2
    print(f"  p={r['p_near']:.3f} w={r['w_hub']:4.1f}: att0={r['att0']:.3f} "
          f"cap0={r['cap0']:.3f} | geodesic pred att={att_pred:.3f} cap={cap_pred:.3f}")
