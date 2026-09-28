#!/usr/bin/env python3
"""A1 (corrected): G-REST with a residual-triggered restart gate vs WarmStart.

CONVENTION NOTE.  An earlier version of this script tried to reset each tracker
to the exact Fiedler pair before every delta, but (a) it reset only `v2` and not
`lam3`, so WarmStartTracker's `gap_est` -- and hence its restart gate -- carried
stale state from the previous step, (b) it guarded the eigenvalue reset with
`hasattr(trk,'lam2')`, an attribute WarmStartTracker does not have, so that
branch never ran, and (c) it never advanced the base Laplacian.  The resulting
WarmStart capture ratios (0.005 at w=10) were artefacts and contradicted the
streaming numbers obtained by grest_compare.py (0.354 at w=10) on the same rig.

This version follows grest_compare.py exactly: a single growing-delta stream per
(w_hub, seed), tracker state persisting across steps, L_prev advancing at the end
of each step, d_ex measured between consecutive exact Fiedler vectors and d_tr
between the tracker's own consecutive outputs.  The only addition is the
residual-triggered G-REST variant.

Reports att on non-restart steps (comparable to Table tab:grest-restart) AND on
all steps (requested by review), plus per-seed values so variability is visible.

Outputs runs/grest_residual_trigger.json
"""
import json, os, sys, pathlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers import fiedler_exact, WarmStartTracker           # noqa: E402
from grest_tracker import GRESTTracker                          # noqa: E402
from synth_grid import build_base, laplacian, align_cos, N_BLOCK  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
P_OUT = 0.02
DELTA_GRID = np.logspace(-4, 1, 11)
W_HUB_GRID = [0.0, 10.0, 30.0]
SEEDS = [0, 1, 2, 3, 4]
RTR = 0.03
NOISE_FLOOR = 1e-9


def make_trackers():
    return {
        "warmstart": WarmStartTracker(restart_tol_ratio=RTR, refine_abs_tol=1e-6),
        "grest_K5": GRESTTracker(K=5, restart_policy="none"),
        "grest_K5_p4": GRESTTracker(K=5, restart_policy="periodic",
                                    restart_every=4),
        "grest_K5_rtrig": GRESTTracker(K=5, restart_policy="residual",
                                       restart_tol=RTR),
    }


def main():
    records = []
    for w_hub in W_HUB_GRID:
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            edges0, n = build_base(rng, P_OUT, 0.5, w_hub)
            a = int(rng.integers(0, N_BLOCK))
            b = int(N_BLOCK + rng.integers(0, N_BLOCK))
            L_prev, _ = laplacian(edges0, n)
            _, vecs_prev = fiedler_exact(L_prev)

            trackers = make_trackers()
            steps = {k: [] for k in trackers}      # (delta, d_ex, d_tr, att, restarted)

            for delta in DELTA_GRID:
                e1 = dict(edges0)
                key = (min(a, b), max(a, b))
                e1[key] = e1.get(key, 0.0) + float(delta)
                L1, _ = laplacian(e1, n)
                _, vecs1 = fiedler_exact(L1)
                d_ex = 1.0 - align_cos(vecs1[:, 1], vecs_prev[:, 1])

                for name, trk in trackers.items():
                    if trk.v2 is None:
                        trk.update(-1, -1, 0, L_prev)
                    nres_before = getattr(trk, "n_restart", 0)
                    vh_old = np.array(trk.v2)
                    trk.update(-1, -1, 0, L1)
                    vh_new = np.array(trk.v2)
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    restarted = getattr(trk, "n_restart", 0) > nres_before
                    steps[name].append((float(delta), float(d_ex), float(d_tr),
                                        float(d_tr / max(d_ex, 1e-15)),
                                        bool(restarted)))
                L_prev, vecs_prev = L1, vecs1

            row = {"w_hub": w_hub, "seed": seed, "n": n}
            for name, trk in trackers.items():
                solid = [s for s in steps[name] if s[1] > NOISE_FLOOR]
                att_all = [s[3] for s in solid]
                att_nr = [s[3] for s in solid if not s[4]]
                vh = np.asarray(trk.v2)
                _, vex = fiedler_exact(L_prev, k=3)
                drift = 1.0 - align_cos(vh, vex[:, 1])
                row[name] = {
                    "att_med_all": float(np.median(att_all)) if att_all else float("nan"),
                    "att_med_nonrestart": float(np.median(att_nr)) if att_nr else float("nan"),
                    "n_steps_scored": len(solid),
                    "n_restart": int(getattr(trk, "n_restart", 0)),
                    "final_drift": float(drift),
                    "ms_per_step": float(1000 * np.mean(trk.time_per_event)),
                }
            records.append(row)
            print(f"  w={w_hub:<5} seed={seed}  "
                  f"ws={row['warmstart']['att_med_all']:.3f}  "
                  f"gr={row['grest_K5']['att_med_all']:.3f}  "
                  f"p4={row['grest_K5_p4']['att_med_nonrestart']:.3f}  "
                  f"rtrig={row['grest_K5_rtrig']['att_med_nonrestart']:.3f}",
                  flush=True)

    (ROOT / "runs" / "grest_residual_trigger.json").write_text(
        json.dumps(records, indent=1))

    print("\n=== median over seeds (att on non-restart steps) ===")
    print(f"{'w_hub':>6} {'WarmStart':>10} {'G-REST':>9} {'+p4':>9} {'+rtrig':>9}")
    for w in W_HUB_GRID:
        rs = [r for r in records if r["w_hub"] == w]
        def med(k, f="att_med_nonrestart"):
            v = [r[k][f] for r in rs if not np.isnan(r[k][f])]
            return np.median(v) if v else float("nan")
        print(f"{w:>6.0f} {med('warmstart'):>10.3f} {med('grest_K5'):>9.3f} "
              f"{med('grest_K5_p4'):>9.3f} {med('grest_K5_rtrig'):>9.3f}")

    print("\n=== per-seed spread of WarmStart att (all steps) ===")
    for w in W_HUB_GRID:
        rs = [r for r in records if r["w_hub"] == w]
        v = [r["warmstart"]["att_med_all"] for r in rs]
        print(f"  w={w:<5} {[round(x,3) for x in v]}")

    print("\n--> runs/grest_residual_trigger.json")


if __name__ == "__main__":
    main()
