#!/usr/bin/env python3
"""rebuild/analyze_mdr.py — analyze rebuild/results/mdr_configs.json with the
SAME bracket protocol as the original gap_scan_analysis.py, to get a like-for-like
comparison of the k-band.

theta* per config = geometric midpoint of the transition bracket
[last delta with seed-avg att<0.5, first delta with att>=0.5];
configs unfrozen from the start (att>=0.5 at the smallest delta) are EXCLUDED,
because their theta* is not resolvable on the grid.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = json.load(open(os.path.join(ROOT, "rebuild", "results", "mdr_configs.json"),
                      encoding="utf-8"))
DELTAS = sorted({r["delta"] for r in recs})


def config_curve(p_near, w_hub):
    out = []
    for d in DELTAS:
        g = [r for r in recs if r["p_near"] == p_near
             and r["w_hub"] == w_hub and r["delta"] == d]
        out.append((d, float(np.mean([r["att"] for r in g])),
                    float(np.median([r["theta"] for r in g]))))
    return out


rows = []
for p_near in sorted({r["p_near"] for r in recs}):
    for w_hub in sorted({r["w_hub"] for r in recs}):
        curve = config_curve(p_near, w_hub)
        below = [c for c in curve if c[1] < 0.5]
        above = [c for c in curve if c[1] >= 0.5]
        if not above:
            continue
        if not below:
            print(f"  [excluded] p={p_near:.3f} w={w_hub:.0f}: unfrozen "
                  f"(att={curve[0][1]:.3f} at smallest delta)")
            continue
        lo, hi = below[-1], above[0]
        mid = float(np.sqrt(lo[2] * hi[2]))
        g0 = [r for r in recs if r["p_near"] == p_near
              and r["w_hub"] == w_hub and r["delta"] == DELTAS[0]]
        gap = float(np.median([r["gap"] for r in g0]))
        stiff = float(np.median([r["stiff"] for r in g0]))
        rows.append({"p_near": p_near, "w_hub": w_hub, "gap": gap,
                     "stiff": stiff, "theta_star": mid,
                     "k": mid / np.sqrt(stiff)})
        print(f"  p={p_near:.3f} w={w_hub:.0f}: gap={gap:.2f} stiff={stiff:.1f} "
              f"theta*={mid:.2e} k={mid/np.sqrt(stiff):.2e} "
              f"bracket=[{lo[2]:.1e},{hi[2]:.1e}]")

ks = np.array([r["k"] for r in rows])
print(f"\n=== k-band over {len(rows)} config medians ===")
print(f"  k: min={ks.min():.3e} max={ks.max():.3e} "
      f"band={ks.max()/ks.min():.1f}x median={np.median(ks):.3e}")
print(f"  (original paper: [6.9e-5, 3.1e-4], 4.6x)")

# T2 pooled two-factor regression (stiffness exponent)
sub = [r for r in rows if r["theta_star"] > 0]
X = np.vstack([np.log10([r["stiff"] for r in sub]),
               np.log10([r["gap"] for r in sub]),
               np.ones(len(sub))]).T
y = np.log10([r["theta_star"] for r in sub])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
r2 = 1 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
print(f"\n  pooled log-log regression (n={len(sub)}):")
print(f"    a(stiffness)={beta[0]:+.3f}  b(gap)={beta[1]:+.3f}  R^2={r2:.3f}")
print(f"    (original paper reported a~0.5 for sqrt-stiffness, b~0)")
