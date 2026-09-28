"""P0e: spectral-gap bucketing of the tracking attenuation law (Q1<->Q2 coupling).

Hypothesis (power-iteration mechanism): WarmStartTracker refines from the OLD
eigenvector, so a true rotation theta of v2 is only partially followed; the
unfollowed fraction after k power steps scales like rho^k with
rho = (c - lam3)/(c - lam2), c = 2*d_max + 1. Since c - lam2 >> gap here,
rho ~ 1 - gap/(c - lam2): LARGER gap -> tracker follows rotations better.
This script tests the empirical trend: attenuation = d_tr/d_ex vs gap = lam3-lam2,
over every consecutive sample pair in the reliable phase of the t03 replays.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from vec_drift import drift_series, STATIONARY0  # noqa: E402

EPS_GOOD = 0.05
D_EX_MIN = 1e-7          # above the exact solver noise floor
TAGS = ["email-eu-t03", "inj015b-t03", "inj030a-t03", "inj050b-t03", "sc06a-t03"]


def collect(tag):
    d = np.load(f"../runs/trackq_{tag}.npz")
    vex = d["v_ex"].T[STATIONARY0:].astype(float)
    vtr = d["v_trk"].T[STATIONARY0:].astype(float)
    eps = np.linalg.norm(vtr - vex, axis=1)
    d_ex, _ = drift_series(vex)
    d_tr, _ = drift_series(vtr)
    gap = (d["lam3_ex"] - d["lam_ex"])[STATIONARY0:]
    # reliable: drop contiguous bad prefix (and the exit boundary sample)
    bad_prefix = 0
    while bad_prefix < len(eps) and eps[bad_prefix] > EPS_GOOD:
        bad_prefix += 1
    rel = np.ones(len(eps), bool)
    rel[: min(len(eps), bad_prefix + 1)] = False
    sel = rel[1:] & (d_ex[1:] > D_EX_MIN) & np.isfinite(gap[1:])
    return {"d_ex": d_ex[1:][sel], "d_tr": d_tr[1:][sel], "gap": gap[1:][sel],
            "n_pairs": int(sel.sum())}


def main():
    pooled = {"d_ex": [], "d_tr": [], "gap": []}
    per_run = {}
    for tag in TAGS:
        c = collect(tag)
        per_run[tag] = c["n_pairs"]
        for k in pooled:
            pooled[k].append(c[k])
        print(f"{tag}: {c['n_pairs']} usable pairs")
    d_ex = np.concatenate(pooled["d_ex"])
    d_tr = np.concatenate(pooled["d_tr"])
    gap = np.concatenate(pooled["gap"])
    att = d_tr / np.maximum(d_ex, 1e-15)
    print(f"\npooled: {len(att)} pairs, attenuation d_tr/d_ex: "
          f"median={np.median(att):.2e} p10={np.percentile(att, 10):.2e} "
          f"p90={np.percentile(att, 90):.2e}")

    # quartile buckets by gap
    qs = np.percentile(gap, [0, 25, 50, 75, 100])
    print("\ngap quartile buckets (pooled):")
    rows = []
    for i in range(4):
        m = (gap >= qs[i]) & (gap <= qs[i + 1] + (1e-9 if i == 3 else 0))
        la = np.log10(att[m])
        rows.append({"q": i + 1, "gap_lo": float(qs[i]), "gap_hi": float(qs[i + 1]),
                     "n": int(m.sum()), "att_median": float(np.median(att[m])),
                     "att_p25": float(np.percentile(att[m], 25)),
                     "att_p75": float(np.percentile(att[m], 75))})
        print(f"  Q{i+1} gap [{qs[i]:6.1f},{qs[i+1]:6.1f}]: n={m.sum():4d} "
              f"att median={np.median(att[m]):.2e} "
              f"IQR=[{np.percentile(att[m],25):.2e},{np.percentile(att[m],75):.2e}]")

    # correlation on log scale
    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(gap, np.log10(att))
    pe = pearsonr(gap, np.log10(att))
    print(f"\nSpearman(gap, log10 att) rho={sp.statistic:.3f} p={sp.pvalue:.3g}")
    print(f"Pearson (gap, log10 att)  r={pe.statistic:.3f} p={pe.pvalue:.3g}")

    # also test normalized predictor gap/lam2
    with np.load("../runs/trackq_email-eu-t03.npz") as d0:
        lam2_all = np.concatenate([(np.load(f"../runs/trackq_{t}.npz")["lam_ex"]
                                    [STATIONARY0 + 1:]) for t in TAGS])
    sp2 = spearmanr(gap / lam2_all[: len(gap)], np.log10(att))
    print(f"Spearman(gap/lam2, log10 att) rho={sp2.statistic:.3f} p={sp2.pvalue:.3g}")

    with open("../runs/gap_scaling.json", "w", encoding="utf-8") as f:
        json.dump({"per_run_pairs": per_run, "quartiles": rows,
                   "spearman_gap": [float(sp.statistic), float(sp.pvalue)],
                   "pearson_gap": [float(pe.statistic), float(pe.pvalue)],
                   "spearman_gap_over_lam2": [float(sp2.statistic), float(sp2.pvalue)]},
                  f, indent=2, ensure_ascii=False)
    print("saved runs/gap_scaling.json")


if __name__ == "__main__":
    main()
