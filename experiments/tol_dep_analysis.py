"""P0m analysis: is k = theta*/sqrt(c-lam2) stable across the
abs_tol x restart_tol_ratio calibration grid?

theta* definition identical to P0k: seed-averaged att curve per
(config, combo), first delta with mean att >= 0.5; theta* = mean theta
at that delta; k = theta*/sqrt(median c-lam2) over seeds.
"""
from __future__ import annotations

import json
import numpy as np

recs = json.load(open("../runs/synth_tol_dep.json", encoding="utf-8"))

CONFIGS = sorted({(r["p_near"], r["w_hub"]) for r in recs})
COMBOS = sorted({(r["abs_tol"], r["rtr"]) for r in recs},
                key=lambda c: (str(c[0]), c[1]))
REF = (1e-6, 0.03)


def theta_star(cfg, combo):
    sub = [r for r in recs if (r["p_near"], r["w_hub"]) == cfg
           and (r["abs_tol"], r["rtr"]) == combo]
    deltas = sorted({r["delta"] for r in sub})
    curve = []
    for d in deltas:
        g = [r["att"] for r in sub if r["delta"] == d]
        curve.append((d, float(np.mean(g)), len(g)))
    cross = next((d for d, a, _ in curve if a >= 0.5), None)
    if cross is None:
        return np.nan, np.nan, curve
    th = float(np.mean([r["theta"] for r in sub if r["delta"] == cross]))
    cm = float(np.median([r["c_minus_lam2"] for r in sub
                          if r["delta"] == cross]))
    return th, th / np.sqrt(cm), curve


def cost_stats(combo):
    sub = [r for r in recs if (r["abs_tol"], r["rtr"]) == combo]
    ks = [r["k_used"] for r in sub if r["k_used"] == r["k_used"]]
    br = [r["branch"] for r in sub if r["branch"] == r["branch"]]
    frac_restart = float(np.mean([b == 2 for b in br])) if br else np.nan
    return float(np.mean(ks)) if ks else np.nan, frac_restart


rows = {}
curves = {}
for combo in COMBOS:
    ks = []
    for cfg in CONFIGS:
        th, k, curve = theta_star(cfg, combo)
        ks.append(k)
        curves[(combo, cfg)] = curve
    valid = [k for k in ks if k == k]
    ks_mean = np.mean(valid) if valid else np.nan
    rows[combo] = ks

ref_ks = rows[REF]

print("== k per config x combo (k = theta*/sqrt(c-lam2)) ==\n")
hdr = f"{'combo':>16s} | " + " ".join(f"{c[0]},{c[1]:.0f}"[:9].rjust(9)
                                     for c in CONFIGS) + " |  k_med  k/min/max  x_ref"
print(hdr)
for combo in COMBOS:
    ks = rows[combo]
    valid = [k for k in ks if k == k]
    if not valid:
        print(f"{str(combo):>16s} | no crossing")
        continue
    kmed = float(np.median(valid))
    cells = " ".join((f"{k:.1e}"[:8].rjust(9) if k == k else "   n/a   ")
                     for k in ks)
    print(f"{str(combo):>16s} | {cells} | {kmed:.2e} "
          f"[{min(valid):.1e},{max(valid):.1e}] "
          f"{kmed/np.median([k for k in ref_ks if k==k]):.2f}x")

print("\n== reference combo per config ==")
for cfg, k in zip(CONFIGS, ref_ks):
    print(f"  {cfg}: k={k:.2e}" if k == k else f"  {cfg}: n/a")

print("\n== cost per combo (mean refine steps/update, frac branch-2 restarts) ==")
for combo in COMBOS:
    mk, fr = cost_stats(combo)
    print(f"  {str(combo):>16s}: k_used={mk:5.1f}  restart_frac={fr:.3f}")

# att curves at reference config for the legacy mode vs abs_tol mode
print("\n== legacy gap-relative (None, 0.03) att curve, cfg (0.02, 10) ==")
for combo in [(None, 0.03), REF, (1e-10, 0.03)]:
    curve = curves.get((combo, (0.02, 10.0)))
    if curve:
        s = " ".join(f"{d:.0e}:{a:.2f}" for d, a, _ in curve[::4])
        print(f"  {str(combo):>14s}: {s}")
