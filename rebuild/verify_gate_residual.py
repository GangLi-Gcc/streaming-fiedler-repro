#!/usr/bin/env python3
"""rebuild/verify_gate_residual.py — verify the Rayleigh-quotient residual
closed form and measure the direct numerical discrepancies the reviewer asked
for, from the already-collected gate_audit.json.

For each crossing (and each delta, for the endpoint inequalities) we check:

  (1) rho_RQ = ||(L'-qI)v||, q = v^T L' v,  equals the closed form
      |delta| |a| sqrt(2 - a^2),  a = v_i - v_j  (from res_prev = sqrt2|delta||a|).
  (2) the identity  rho_RQ^2 = sin^2(theta) sigma_w^2 - (q - lambda2')^2,
      i.e. res^2 = res_exact^2 - (lam - lam2_new)^2.
  (3) the relative discrepancy between the deployed residual (res) and the
      exact-post-eigenvalue residual (res_exact = sin theta sigma_w), the
      quantity that supports the "four significant figures" claim.
  (4) the deployed-gate inequalities at the two bracket endpoints for each
      direct-gate crossing (res < gate at lower, res > gate at upper), using the
      actual estimated gap (gap_est) and gate floor (0.0).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "rebuild", "results")
rows = json.load(open(os.path.join(RESULT_DIR, "gate_audit.json"), encoding="utf-8"))

# group by (p_near, w_hub, seed), sorted by delta
from collections import defaultdict
groups = defaultdict(list)
for r in rows:
    groups[(r["p_near"], r["w_hub"], r["seed"])].append(r)
for k in groups:
    groups[k].sort(key=lambda r: r["delta"])

# ---- closed-form + identity check across ALL deltas ----
max_closed_err = 0.0
max_identity_err = 0.0
for r in rows:
    # a = res_prev / (sqrt2 * delta)
    a = r["res_prev"] / (np.sqrt(2) * abs(r["delta"]) + 1e-300)
    a = min(a, np.sqrt(2.0))  # numerical clip
    closed = abs(r["delta"]) * a * np.sqrt(max(2.0 - a * a, 0.0))
    max_closed_err = max(max_closed_err, abs(closed - r["res"]))
    # identity res^2 = res_exact^2 - (lam - lam2_new)^2
    rhs = max(r["res_exact"] ** 2 - (r["lam"] - r["lam2_new"]) ** 2, 0.0)
    max_identity_err = max(max_identity_err, abs(np.sqrt(rhs) - r["res"]))
print(f"closed-form check: max |rho_RQ - |d||a|sqrt(2-a^2)| over all deltas = {max_closed_err:.3e}")
print(f"identity check:     max |rho_RQ - sqrt(res_exact^2-(q-lam2')^2)| = {max_identity_err:.3e}")

# ---- discrepancy res vs res_exact across all deltas (the 4-s.f. claim) ----
rel = []
for r in rows:
    if r["res_exact"] > 1e-15:
        rel.append(abs(r["res"] - r["res_exact"]) / r["res_exact"])
rel = np.array(rel)
print(f"\nrelative discrepancy |res - res_exact|/res_exact (all deltas):")
print(f"  max = {rel.max():.3e}, median = {np.median(rel):.3e}, "
      f"p99 = {np.percentile(rel, 99):.3e}")
# squared ratio: res^2 / res_exact^2 = 1 - (q-lam2')^2/res_exact^2
sq = []
for r in rows:
    if r["res_exact"] > 1e-15:
        sq.append((r["lam"] - r["lam2_new"]) ** 2 / r["res_exact"] ** 2)
sq = np.array(sq)
print(f"  (q-lam2')^2 / res_exact^2 (all deltas): max = {sq.max():.3e}, "
      f"median = {np.median(sq):.3e}")
# restricted to crossings
crossings = json.load(open(os.path.join(RESULT_DIR, "gate_audit_crossings.json"),
                           encoding="utf-8"))
relc = np.array([abs(r["res"] - r["res_exact"]) / r["res_exact"] for r in crossings
                 if r["res_exact"] > 1e-15])
sqc = np.array([(r["lam"] - r["lam2_new"]) ** 2 / r["res_exact"] ** 2 for r in crossings
                if r["res_exact"] > 1e-15])
print(f"\nAT CROSSINGS ({len(crossings)}):")
print(f"  |res - res_exact|/res_exact: max = {relc.max():.3e}, "
      f"median = {np.median(relc):.3e}")
print(f"  (q-lam2')^2 / res_exact^2 : max = {sqc.max():.3e}, "
      f"median = {np.median(sqc):.3e}")
print(f"  -> deployed res matches res_exact to ~{int(np.ceil(-np.log10(relc.max())))} "
      f"significant digits at the worst crossing")

# ---- bracket endpoint inequalities for direct-gate crossings ----
print(f"\nBracket endpoint inequalities (direct-gate crossings):")
n_direct = 0
n_lower_ok = 0
n_upper_ok = 0
bracket_ratios = []
pred_inside = 0
for r in crossings:
    if r["branch"] != 2:
        continue
    n_direct += 1
    rs = groups[(r["p_near"], r["w_hub"], r["seed"])]
    idx = [i for i, x in enumerate(rs) if abs(x["delta"] - r["delta"]) < 1e-15][0]
    if idx == 0:
        # no lower bracket (crossing at smallest delta) -- treat lower as trivially ok
        lower_ok = True
        lo = None
    else:
        lo = rs[idx - 1]
        lower_ok = lo["res"] < lo["gate"]
    upper_ok = r["res"] > r["gate"]
    n_lower_ok += lower_ok
    n_upper_ok += upper_ok
    if lo is not None:
        bracket_ratios.append(r["theta"] / lo["theta"] if lo["theta"] > 1e-15 else 0.0)
    # predicted transition inside the actual angular bracket:
    # gate uses 0.03 * gap_est; theta_pred = rtr*gap_est/sigma_w; check lo.theta < pred < hi.theta
    if lo is not None and r["sigma_w"] > 1e-15:
        pred = 0.03 * r["gap_est"] / r["sigma_w"]
        if lo["theta"] < pred < r["theta"]:
            pred_inside += 1
print(f"  direct-gate crossings: {n_direct}")
print(f"  lower endpoint res < gate: {n_lower_ok}/{n_direct}")
print(f"  upper endpoint res > gate: {n_upper_ok}/{n_direct}")
br = np.array(bracket_ratios)
print(f"  angular bracket ratios theta_hi/theta_lo: median {np.median(br):.2f}, "
      f"min {br.min():.2f}, max {br.max():.2f}")
print(f"  predicted theta (0.03*gap_est/sigma_w) inside actual angular bracket: "
      f"{pred_inside}/{len(bracket_ratios)}")
