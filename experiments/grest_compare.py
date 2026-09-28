"""P0p: head-to-head WarmStartTracker vs G-REST on the P0h synth_mdr rig.

Same rig as synth_mdr.py (two-block SBM + one cross-block edge of weight
delta; DELTA_GRID logspace(-4,1,11); w_hub in {0,10,30}; seeds 0..4), but the
trackers run in STREAMING mode: state persists across the growing-delta
sequence, and att is computed per-step against the PREVIOUS snapshot's exact
Fiedler vector (the P0h fresh-jump convention references d_ex to L0, which
breaks per-step att in streaming mode -- found during bring-up).

Metrics per (w_hub, seed, tracker):
  - att_k = d_tr/k / d_ex/k   (per-step capture ratio; meaningless below the
    reference-noise floor d_ex ~ 1e-12, filtered in the summary)
  - drift = 1 - |<v2_trk, v2_exact>| at the final step
  - lam2_err at the final step
  - median per-step wall time
Warmstart restarts are counted for context.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from trackers import fiedler_exact, WarmStartTracker  # noqa: E402
from grest_tracker import GRESTTracker  # noqa: E402
from synth_grid import build_base, laplacian, align_cos, N_BLOCK  # noqa: E402

P_OUT = 0.02
DELTA_GRID = np.logspace(-4, 1, 11)
W_HUB_GRID = [0.0, 10.0, 30.0]
SEEDS = [0, 1, 2, 3, 4]


def main():
    recs = []
    for w_hub in W_HUB_GRID:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            edges0, n = build_base(rng, P_OUT, 0.5, w_hub)
            a = int(rng.integers(0, N_BLOCK))
            b = int(N_BLOCK + rng.integers(0, N_BLOCK))
            L_prev, _ = laplacian(edges0, n)
            _, vecs_prev = fiedler_exact(L_prev)
            trackers = {
                "warmstart": WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6),
                "grest_K3": GRESTTracker(K=3),
                "grest_K5": GRESTTracker(K=5),
            }
            row_lam = {}
            per_step = {k: [] for k in trackers}   # (delta, d_ex, d_tr, att)
            times = {k: [] for k in trackers}
            for delta in DELTA_GRID:
                e1 = dict(edges0)
                e1[(min(a, b), max(a, b))] = e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                L1, _ = laplacian(e1, n)
                _, vecs1 = fiedler_exact(L1)
                d_ex = 1.0 - align_cos(vecs1[:, 1], vecs_prev[:, 1])
                for name, trk in trackers.items():
                    if trk.v2 is None:
                        trk.update(-1, -1, 0, L_prev)
                    vh_old = np.array(trk.v2)
                    lam_trk = trk.update(-1, -1, 0, L1)
                    vh_new = np.array(trk.v2)
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    per_step[name].append((float(delta), float(d_ex), float(d_tr),
                                           float(d_tr / max(d_ex, 1e-15))))
                    times[name].append(trk.time_per_event[-1])
                    row_lam[name] = float(lam_trk)
                L_prev, vecs_prev = L1, vecs1
            lam2_fin = float(fiedler_exact(L1, k=3)[0][1])
            row = {"w_hub": w_hub, "seed": seed, "lam2_fin": lam2_fin}
            for name, trk in trackers.items():
                steps = per_step[name]
                solid = [s for s in steps if s[1] > 1e-9]   # above reference-noise floor
                att_solid = [s[3] for s in solid]
                vh = np.asarray(trk.v2)
                _, vex = fiedler_exact(L1, k=3)
                row[f"att_med_{name}"] = float(np.median(att_solid)) if att_solid else np.nan
                row[f"att_min_{name}"] = float(np.min(att_solid)) if att_solid else np.nan
                row[f"drift_{name}"] = float(1.0 - align_cos(vh, vex[:, 1]))
                row[f"lam2err_{name}"] = float(abs(row_lam[name] - lam2_fin))
                row[f"tstep_{name}"] = float(np.median(times[name]))
                row[f"steps_{name}"] = [s[:4] for s in steps]
                if hasattr(trk, "n_restart"):
                    row[f"n_restart_{name}"] = int(trk.n_restart)
            recs.append(row)
            print(f"w_hub={w_hub} seed={seed} done", flush=True)
    with open("../runs/grest_compare.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    print("\n=== per-step capture (att, d_ex>1e-9) / final drift / lam2 / t-step ===")
    names = list(trackers)
    for w_hub in W_HUB_GRID:
        g = [r for r in recs if r["w_hub"] == w_hub]
        print(f"-- w_hub={w_hub}")
        for name in names:
            att = [r[f"att_med_{name}"] for r in g]
            d = [r[f"drift_{name}"] for r in g]
            t = [r[f"tstep_{name}"] for r in g]
            le = [r[f"lam2err_{name}"] for r in g]
            nr = [r.get(f"n_restart_{name}", -1) for r in g]
            print(f"  {name:11s} att_med={np.nanmedian(att):6.3f}  att_min={np.nanmin(att):6.3f}  "
                  f"drift_med={np.nanmedian(d):.2e}  lam2err_med={np.nanmedian(le):.2e}  "
                  f"tstep={1e3*np.nanmedian(t):5.1f}ms  restarts={nr}")
    print("\nsaved runs/grest_compare.json")


if __name__ == "__main__":
    main()
