"""P0e mechanism arbitration: does branch selection explain the negative
gap-attenuation law?

Joins the 84 usable rotation pairs (gap_scaling selection) with the
instrumented tracker diagnostics (branch, k_used, res_in, gap_est, dmax).

Hypotheses:
  H_gate  small gap -> gate res<=0.03*gap_est fails more often -> exact
          restart (att~1) rescues small-gap pairs; large-gap pairs rely on
          refinement which lags -> att~0.  => branch-2 fraction decreases
          with gap; att | branch0 ~ 0 everywhere, att | branch2 ~ 1.
  H_pow   att ~ rho^k (power-iteration leftover), rho=(c-lam3)/(c-lam2).
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from vec_drift import drift_series, STATIONARY0  # noqa: E402
from gap_scaling import collect, TAGS, EPS_GOOD, D_EX_MIN  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# collect() returns pair arrays but we need the kept INDICES to join diag;
# re-implement the selection here with index tracking.
def pairs_with_diag(tag):
    d = np.load(f"../runs/trackq_{tag}.npz")
    vex = d["v_ex"].T[STATIONARY0:].astype(float)
    vtr = d["v_trk"].T[STATIONARY0:].astype(float)
    eps = np.linalg.norm(vtr - vex, axis=1)
    d_ex, _ = drift_series(vex)
    d_tr, _ = drift_series(vtr)
    gap = (d["lam3_ex"] - d["lam_ex"])[STATIONARY0:]
    bad = 0
    while bad < len(eps) and eps[bad] > EPS_GOOD:
        bad += 1
    rel = np.ones(len(eps), bool)
    rel[: min(len(eps), bad + 1)] = False
    sel = rel[1:] & (d_ex[1:] > D_EX_MIN) & np.isfinite(gap[1:])
    idx = np.where(sel)[0] + 1                     # segment index of pair (t)
    diag = d["diag"]                               # row j -> global sample j+1
    dg = diag[STATIONARY0 + idx - 1]               # diag that produced v[idx]
    return {"d_ex": d_ex[idx], "d_tr": d_tr[idx], "gap": gap[idx],
            "branch": dg[:, 0], "k": dg[:, 3], "res_in": dg[:, 1],
            "gap_est": dg[:, 2], "lam2": dg[:, 5], "dmax": dg[:, 6]}


def main():
    P = {k: [] for k in ["d_ex", "d_tr", "gap", "branch", "k", "res_in",
                         "gap_est", "lam2", "dmax"]}
    for tag in TAGS:
        p = pairs_with_diag(tag)
        for k in P:
            P[k].append(p[k])
    P = {k: np.concatenate(v) for k, v in P.items()}
    att = P["d_tr"] / np.maximum(P["d_ex"], 1e-15)
    print(f"pooled pairs: {len(att)}")

    qs = np.percentile(P["gap"], [0, 25, 50, 75, 100])
    print("\nper gap quartile: branch fractions and att by branch")
    rows = []
    for i in range(4):
        m = (P["gap"] >= qs[i]) & (P["gap"] <= qs[i + 1] + (1e-9 if i == 3 else 0))
        br = P["branch"][m]
        f2 = float((br == 2).mean()); f0 = float((br == 0).mean()); f1 = float((br == 1).mean())
        a0 = att[m][br == 0]; a2 = att[m][br == 2]; a1 = att[m][br == 1]
        row = {"q": i + 1, "gap_lo": float(qs[i]), "gap_hi": float(qs[i + 1]),
               "f_branch0_refine": f0, "f_branch1_fallback": f1, "f_branch2_exact": f2,
               "att_refine_med": float(np.median(a0)) if len(a0) else None,
               "att_exact_med": float(np.median(a2)) if len(a2) else None,
               "att_fallback_med": float(np.median(a1)) if len(a1) else None,
               "k_med": float(np.median(P["k"][m])), "k_max": float(P["k"][m].max())}
        rows.append(row)
        def fm(v):
            return f"{v:.1e}" if v is not None else "  -  "
        print(f"  Q{i+1} gap[{qs[i]:5.1f},{qs[i+1]:5.1f}] n={m.sum():3d} | "
              f"br0={f0:.2f} br1={f1:.2f} br2={f2:.2f} | "
              f"att|br0={fm(row['att_refine_med'])} att|br2={fm(row['att_exact_med'])} "
              f"att|br1={fm(row['att_fallback_med'])} | k med={row['k_med']:.0f} max={row['k_max']:.0f}")

    # gate margin: res_in / gap_est (must be <= 0.03 to enter refine)
    margin = P["res_in"] / np.maximum(P["gap_est"], 1e-12)
    print(f"\ngate margin res_in/gap_est: median={np.median(margin):.4f} "
          f"(gate=0.03)  by quartile: "
          + " ".join(f"{np.median(margin[(P['gap'] >= qs[i]) & (P['gap'] <= qs[i+1] + (1e-9 if i==3 else 0))]):.4f}"
                     for i in range(4)))

    # H_pow within branch 0: att vs rho^k
    m0 = P["branch"] == 0
    c = 2.0 * P["dmax"] + 1.0
    rho = (c - P["lam2"] - P["gap_est"]) / np.maximum(c - P["lam2"], 1e-12)
    x = P["k"] * np.log10(np.maximum(rho, 1e-12))
    sp = spearmanr(x[m0], np.log10(att[m0] + 1e-12))
    print(f"\nH_pow within refine-accept: Spearman(k*log10(rho), log10 att) = "
          f"{sp.statistic:.3f} p={sp.pvalue:.3g} (n={m0.sum()})")
    print(f"  rho: median={np.median(rho):.4f} range=[{rho.min():.4f},{rho.max():.4f}]")

    with open("../runs/gap_regression.json", "w", encoding="utf-8") as f:
        json.dump({"quartiles": rows,
                   "gate_margin_by_quartile": [float(np.median(margin[(P["gap"] >= qs[i]) & (P["gap"] <= qs[i+1] + (1e-9 if i==3 else 0))])) for i in range(4)],
                   "spearman_pow": [float(sp.statistic), float(sp.pvalue)]},
                  f, indent=2, ensure_ascii=False)
    print("saved runs/gap_regression.json")


if __name__ == "__main__":
    main()
