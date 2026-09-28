"""P0y ablation: G-REST + periodic exact restart, on the P0p streaming rig.

Motivation (internal review Q3): a likely referee objection to the G-REST
head-to-head is "G-REST only lacks a restart mechanism; give it one and it
matches WarmStart." This ablation tests exactly that, in the strongest and
simplest form: every m steps the G-REST basis is replaced by the exact
eigendecomposition of the current Laplacian (an oracle the WarmStart gate
never gets).

Trackers per (w_hub, seed):
  warmstart            reference (reactive, calibrated gate)
  grest_K5             no restart (P0p baseline)
  grest_K5_periodic2   exact restart every 2nd step
  grest_K5_periodic4   exact restart every 4th step
  grest_K5_oracle      exact restart every step (fidelity upper bound / cost)

Key metric split:
  - att on NON-restart steps: what a detector actually consumes between
    restarts. Prediction: unchanged from no-restart G-REST (per-step RR
    capture under high stiffness is a property of the update rule, not of
    the restart policy).
  - drift at final step: collapses with restarts (restart fixes accumulated
    offset, not the per-step low-pass).
  - cost: mean ms/step including exact reseeds (~75 ms each).

Output: runs/grest_restart_ablation.json
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


class GRESTPeriodic:
    """GRESTTracker wrapped with a periodic exact restart."""

    def __init__(self, K: int, period: int):
        self.name = f"grest_K{K}_periodic{period}" if period > 1 else f"grest_K{K}_oracle"
        self.K = K
        self.period = period
        self.n_steps = 0
        self.n_restart = 0
        self.core = GRESTTracker(K=K)
        self.v2 = None
        self.time_per_event = []

    def update(self, i, j, sign, L):
        import time
        t0 = time.perf_counter()
        self.n_steps += 1
        if self.core.X is None:
            self.core.update(i, j, sign, L)   # seed
        elif self.period == 1 or self.n_steps % self.period == 0:
            self.core._seed(L)                # exact oracle restart
            self.n_restart += 1
        else:
            self.core.update(i, j, sign, L)
        self.v2 = np.asarray(self.core.v2)
        self.time_per_event.append(time.perf_counter() - t0)
        return float(self.core.c - self.core.Lam[1])


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
                "grest_K5": GRESTTracker(K=5),
                "grest_K5_periodic2": GRESTPeriodic(5, 2),
                "grest_K5_periodic4": GRESTPeriodic(5, 4),
                "grest_K5_oracle": GRESTPeriodic(5, 1),
            }
            per_step = {k: [] for k in trackers}   # (delta, d_ex, d_tr, att, is_restart)
            times = {k: [] for k in trackers}
            for delta in DELTA_GRID:
                e1 = dict(edges0)
                e1[(min(a, b), max(a, b))] = e1.get((min(a, b), max(a, b)), 0.0) + float(delta)
                L1, _ = laplacian(e1, n)
                _, vecs1 = fiedler_exact(L1)
                d_ex = 1.0 - align_cos(vecs1[:, 1], vecs_prev[:, 1])
                for name, trk in trackers.items():
                    restarted = bool(getattr(trk, "n_steps", 1) and
                                     hasattr(trk, "period") and
                                     trk.n_steps % trk.period == 0 and trk.period > 0 and
                                     trk.n_steps > 0)
                    if trk.v2 is None:
                        trk.update(-1, -1, 0, L_prev)
                    vh_old = np.array(trk.v2)
                    lam_trk = trk.update(-1, -1, 0, L1)
                    vh_new = np.array(trk.v2)
                    if np.dot(vh_new, vh_old) < 0:
                        vh_new = -vh_new
                    d_tr = 1.0 - align_cos(vh_new, vh_old)
                    is_restart = bool(getattr(trk, "n_steps", 0) > 0 and
                                      hasattr(trk, "period") and
                                      (trk.period == 1 or trk.n_steps % trk.period == 0))
                    per_step[name].append((float(delta), float(d_ex), float(d_tr),
                                           float(d_tr / max(d_ex, 1e-15)), is_restart))
                    times[name].append(trk.time_per_event[-1])
                L_prev, vecs_prev = L1, vecs1
            _, vex = fiedler_exact(L1, k=3)
            row = {"w_hub": w_hub, "seed": seed}
            for name, trk in trackers.items():
                steps = per_step[name]
                solid = [s for s in steps if s[1] > 1e-9]
                nr_solid = [s for s in solid if not s[4]]      # non-restart steps
                vh = np.asarray(trk.v2)
                row[f"att_med_{name}"] = float(np.median([s[3] for s in solid])) if solid else np.nan
                row[f"att_nr_med_{name}"] = float(np.median([s[3] for s in nr_solid])) if nr_solid else np.nan
                row[f"drift_{name}"] = float(1.0 - align_cos(vh, vex[:, 1]))
                row[f"tstep_{name}"] = float(np.median(times[name]))
                row[f"tstep_mean_{name}"] = float(np.mean(times[name]))
                row[f"n_restart_{name}"] = int(getattr(trk, "n_restart", 0))
            recs.append(row)
            print(f"w_hub={w_hub} seed={seed} done", flush=True)

    with open("../runs/grest_restart_ablation.json", "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=1)

    print("\n=== non-restart-step att med / final drift / ms-per-step / n_restarts ===")
    for w_hub in W_HUB_GRID:
        g = [r for r in recs if r["w_hub"] == w_hub]
        print(f"-- w_hub={w_hub}")
        for name in trackers:
            att = [r[f"att_nr_med_{name}"] for r in g]
            d = [r[f"drift_{name}"] for r in g]
            t = [r[f"tstep_{name}"] for r in g]
            nr = [r[f"n_restart_{name}"] for r in g]
            print(f"  {name:18s} att_nr={np.nanmedian(att):6.3f}  "
                  f"drift_med={np.nanmedian(d):.2e}  "
                  f"tstep={1e3*np.nanmedian(t):6.1f}ms  restarts={nr}")
    print("\nsaved runs/grest_restart_ablation.json")


if __name__ == "__main__":
    main()
