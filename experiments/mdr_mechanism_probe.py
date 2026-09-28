"""P10 probe: mechanism instrumentation for the MDR law derivation.

For each (w_hub, seed, delta) on the synth_mdr rig, run ONE warm-start
tracker update from the previous exact Fiedler vector and record:
  branch (0=refine accepted, 1=refine-fallback-exact, 2=exact restart),
  k_used (power-refinement steps), res (pre-update residual),
  gap_est, c-lam2 (stiffness), theta_in/out, att,
  spectral decomposition of the rotation direction w = (v2_new - v2_old)/..
  against the exact eigenbasis of L_new: mass near lambda2, mid, and top
  of L's spectrum (theta_j = c - lam_j small).

Hypotheses to discriminate:
  H1 gate boundary: att -> 1 exactly when res > rtr*gap_est
  H2 step budget:  att ~= kappa^(2*k_used), kappa = theta_bar_w/theta2
  H3 early-stop clamp: theta_out ~ tol_res / sigma_out

Output: runs/mdr_mechanism_probe.json
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import scipy.linalg as sla

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from trackers import fiedler_exact, WarmStartTracker  # noqa: E402
from synth_grid import build_base, laplacian, align_cos, N_BLOCK  # noqa: E402

P_OUT = 0.02
DELTA_GRID = np.logspace(-4, 1, 11)
W_HUB_GRID = [0.0, 10.0, 30.0]
SEEDS = [0, 1]
RTR = 0.03


def spectral_mass_w(L, v_old, v_new, lam2_new, gap):
    """Decompose rotation direction w against exact eigenbasis of L."""
    vals, vecs = sla.eigh(L.toarray())
    # rotation direction, orthogonalized to v_new
    dvec = v_new - (v_new @ v_old) * v_old
    nd = np.linalg.norm(dvec)
    if nd < 1e-15:
        return None
    w = dvec / nd
    beta = (vecs.T @ w) ** 2          # mass on each eigenvector
    c = 2.0 * float(L.diagonal().max()) + 1.0
    th = c - vals                      # eigenvalues of shifted operator
    s = c - lam2_new                   # stiffness
    out = {
        "mass_near_lam2": float(beta[vals < lam2_new + 0.5 * gap].sum()),
        "mass_top_quarter": float(beta[th < 0.25 * s].sum()),
        "mass_mid": float(beta[(th >= 0.25 * s) & (vals >= lam2_new + 0.5 * gap)].sum()),
        "theta_bar_w_over_theta2": float(np.sqrt((beta * th ** 2).sum()) / (c - lam2_new)),
        "lam_max": float(vals[-1]),
        "lam_gap_above": float(vals[2] - vals[1]) if len(vals) > 2 else np.nan,
    }
    return out


def main():
    recs = []
    for w_hub in W_HUB_GRID:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            edges0, n = build_base(rng, P_OUT, 0.5, w_hub)
            a = int(rng.integers(0, N_BLOCK))
            b = int(N_BLOCK + rng.integers(0, N_BLOCK))
            L_prev, _ = laplacian(edges0, n)
            vals_prev, vecs_prev = fiedler_exact(L_prev)
            trk = WarmStartTracker(restart_tol_ratio=RTR, refine_abs_tol=1e-6)
            trk.update(-1, -1, 0, L_prev)          # seed
            for delta in DELTA_GRID:
                e1 = dict(edges0)
                e1[(min(a, b), max(a, b))] = e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                L1, _ = laplacian(e1, n)
                vals1, vecs1 = fiedler_exact(L1, k=6)
                v_old = np.asarray(vecs_prev[:, 1]).copy()
                lam2_new = float(vals1[1])
                v_new_ex = np.asarray(vecs1[:, 1])
                d_ex = 1.0 - align_cos(v_new_ex, v_old)
                gap = float(vals1[2] - vals1[1])
                dmax = float(L1.diagonal().max())
                c = 2.0 * dmax + 1.0
                # pre-update residual & gate (reproduce tracker internals)
                lam_rop = float(v_old @ (L1 @ v_old))
                res = float(np.linalg.norm(L1 @ v_old - lam_rop * v_old))
                gate = RTR * max(trk.lam3 - lam_rop, 1e-9)
                ndiag = len(trk.diag)
                lam_trk = trk.update(-1, -1, 0, L1)
                branch, res0, gap_est, k_used = trk.diag[ndiag][:4]
                v_out = np.asarray(trk.v2)
                d_tr = 1.0 - align_cos(v_new_ex, v_out)
                rec = {
                    "w_hub": w_hub, "seed": seed, "delta": float(delta),
                    "branch": int(branch), "k_used": int(k_used),
                    "res": res, "gate": gate, "gap": gap,
                    "stiff": c - lam2_new, "lam2": lam2_new,
                    "d_ex": float(d_ex), "d_tr": float(d_tr),
                    "att": float(d_tr / max(d_ex, 1e-15)),
                }
                sm = spectral_mass_w(L1, v_old, v_new_ex, lam2_new, gap)
                if sm:
                    rec.update({f"sm_{k}": v for k, v in sm.items()})
                recs.append(rec)
                L_prev, vecs_prev = L1, vecs1
            print(f"w_hub={w_hub} seed={seed} done", flush=True)

    with open("../runs/mdr_mechanism_probe.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    print("\n=== per (w_hub): pinned-regime rows (att<0.9) ===")
    for w_hub in W_HUB_GRID:
        g = [r for r in recs if r["w_hub"] == w_hub]
        for r in g:
            if r["att"] < 0.9 and r["d_ex"] > 1e-9:
                print(f"w={w_hub:4.0f} d={r['delta']:.1e} att={r['att']:.3f} "
                      f"br={r['branch']} k={r['k_used']} res={r['res']:.2e} "
                      f"gate={r['gate']:.2e} stiff={r['stiff']:.1f} "
                      f"kap={r.get('sm_theta_bar_w_over_theta2', float('nan')):.4f} "
                      f"mTop={r.get('sm_mass_top_quarter', float('nan')):.3f} "
                      f"mNear={r.get('sm_mass_near_lam2', float('nan')):.3f}")
    print("\nsaved runs/mdr_mechanism_probe.json")


if __name__ == "__main__":
    main()
