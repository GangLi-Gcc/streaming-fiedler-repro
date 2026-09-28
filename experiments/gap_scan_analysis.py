"""P0k analysis: theta* brackets, gap-invariance and stiffness-scaling tests.

theta* per config = geometric midpoint of the transition bracket
[last delta with seed-avg att<0.5, first delta with att>=0.5]; bracket width
is the honest grid-resolution uncertainty (~3x).

Tests:
  T1 (gap invariance): within each w_hub level, regress log10 theta* on
      log10 gap over the 4 p_near families -> slope ~ 0?
  T2 (stiffness scaling): pooled over families, regress
      log10 theta* ~ a*log10(c-lam2) + b*log10(gap) -> expect a~0.5, b~0.
  T3 (law constant): theta* / sqrt(c-lam2) vs the P0h constant 1.2e-4.
Also repeats T2 on the original P0h synth_mdr.json for comparison
(gap nearly constant there: a is the free stiffness exponent).
"""
from __future__ import annotations

import json
import sys

import numpy as np

recs = json.load(open("../runs/synth_gap_scan.json", encoding="utf-8"))
DELTAS = sorted({r["delta"] for r in recs})


def config_curve(p_near, w_hub):
    """seed-averaged (delta -> att, theta_med)."""
    out = []
    for d in DELTAS:
        g = [r for r in recs if r["p_near"] == p_near
             and r["w_hub"] == w_hub and r["delta"] == d]
        out.append((d, float(np.mean([r["att"] for r in g])),
                    float(np.median([r["theta"] for r in g]))))
    return out


def theta_star_bracket(p_near, w_hub):
    curve = config_curve(p_near, w_hub)
    below = [c for c in curve if c[1] < 0.5]
    above = [c for c in curve if c[1] >= 0.5]
    if not above:
        return None
    if not below:                       # unfrozen from the start
        return (0.0, above[0][2], above[0][2])
    lo, hi = below[-1], above[0]
    return (lo[2], hi[2], float(np.sqrt(lo[2] * hi[2])))


rows = []
print(f"{'p_near':>7s} {'w_hub':>5s} {'gap':>7s} {'c-lam2':>8s} "
      f"{'theta* mid':>11s}  bracket")
for p_near in sorted({r["p_near"] for r in recs}):
    for w_hub in sorted({r["w_hub"] for r in recs}):
        g0 = [r for r in recs if r["p_near"] == p_near
              and r["w_hub"] == w_hub and r["delta"] == DELTAS[0]]
        gap = float(np.median([r["gap"] for r in g0]))
        cm = float(np.median([r["c_minus_lam2"] for r in g0]))
        br = theta_star_bracket(p_near, w_hub)
        if br is None:
            continue
        lo, hi, mid = br
        rows.append({"p_near": p_near, "w_hub": w_hub, "gap": gap,
                     "cm": cm, "lo": lo, "hi": hi, "mid": mid})
        print(f"{p_near:7.3f} {w_hub:5.0f} {gap:7.2f} {cm:8.1f} "
              f"{mid:11.2e}  [{lo:.1e}, {hi:.1e}]")

# ---- T1: gap invariance within hub level
print("\n== T1: log10 theta* ~ slope * log10(gap) within each w_hub ==")
for w_hub in sorted({r["w_hub"] for r in recs}):
    sub = [r for r in rows if r["w_hub"] == w_hub and r["lo"] > 0]
    if len(sub) < 3:
        print(f"w_hub={w_hub}: too few bracketed configs ({len(sub)}), skip")
        continue
    x = np.log10([r["gap"] for r in sub])
    y = np.log10([r["mid"] for r in sub])
    A = np.vstack([x, np.ones(len(x))]).T
    (slope, icept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = slope * x + icept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-300)
    print(f"w_hub={w_hub:5.0f}: slope={slope:+.3f} R^2={r2:.3f} "
          f"(n={len(sub)}, gaps {min(np.power(10,x)):.2f}..{max(np.power(10,x)):.2f}, "
          f"theta* ratio {min(np.power(10,y))/max(np.power(10,y)):.2f})")

# ---- T2: pooled two-factor regression
print("\n== T2: log10 theta* ~ a*log10(c-lam2) + b*log10(gap) (pooled) ==")
sub = [r for r in rows if r["lo"] > 0]
X = np.vstack([np.log10([r["cm"] for r in sub]),
               np.log10([r["gap"] for r in sub]),
               np.ones(len(sub))]).T
y = np.log10([r["mid"] for r in sub])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ beta
r2 = 1 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
print(f"a(stiffness)={beta[0]:+.3f}  b(gap)={beta[1]:+.3f}  "
      f"const=10^{beta[2]:.3f}  R^2={r2:.3f}  (n={len(sub)})")

# ---- T3: law constant check
print("\n== T3: theta*/sqrt(c-lam2) vs P0h constant 1.2e-4 ==")
for r in rows:
    if r["lo"] > 0:
        k = r["mid"] / np.sqrt(r["cm"])
        print(f"p_near={r['p_near']:.3f} hub={r['w_hub']:5.0f} "
              f"gap={r['gap']:6.2f} cm={r['cm']:7.1f} -> k={k:.2e} "
              f"({'OK' if 3e-5 < k < 5e-4 else 'OFF'})")

# ---- P0h comparison: same two-factor regression on synth_mdr.json
print("\n== P0h synth_mdr.json same regression (gap nearly constant) ==")
m = json.load(open("../runs/synth_mdr.json", encoding="utf-8"))
md = sorted({r["delta"] for r in m})
mrows = []
for wh in sorted({r["w_hub"] for r in m}):
    curve = []
    for d in md:
        g = [r for r in m if r["w_hub"] == wh and r["delta"] == d]
        curve.append((d, float(np.mean([r["att"] for r in g])),
                      float(np.median([r["theta"] for r in g]))))
    below = [c for c in curve if c[1] < 0.5]
    above = [c for c in curve if c[1] >= 0.5]
    if not above or not below:
        continue
    g0 = [r for r in m if r["w_hub"] == wh and r["delta"] == md[0]]
    mrows.append({"cm": float(np.median([r["c_minus_lam2"] for r in g0])),
                  "gap": float(np.median([r["gap"] for r in g0])),
                  "mid": float(np.sqrt(below[-1][2] * above[0][2]))})
X = np.vstack([np.log10([r["cm"] for r in mrows]),
               np.log10([r["gap"] for r in mrows]),
               np.ones(len(mrows))]).T
y = np.log10([r["mid"] for r in mrows])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"P0h: a(stiffness)={beta[0]:+.3f}  b(gap)={beta[1]:+.3f}  "
      f"(n={len(mrows)}; gaps {min(r['gap'] for r in mrows):.2f}.."
      f"{max(r['gap'] for r in mrows):.2f})")
print("P0h rows:", [(round(r['cm']), round(r['gap'], 2),
                     f"{r['mid']:.1e}") for r in mrows])
