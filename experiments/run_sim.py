"""End-to-end simulation: dynamic SBM edge stream -> trackers -> detector -> metrics.

Usage: python run_sim.py [--n 2000] [--events 6000] [--cp-every 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dyngraph import DynamicSBM
from trackers import BaselineFull, WarmStartTracker
from detectors import CUSUMDrift, evaluate_detection


def laplacian(A):
    d = A.sum(1)
    return sp.lil_matrix(sp.diags(d) - A)


def run(n=2000, events=6000, cp_every=2000, seed=0, exact_every=10):
    g = DynamicSBM(n=n, seed=seed)
    base = BaselineFull()
    trk = WarmStartTracker()
    det = CUSUMDrift(window=max(200, n // 5), min_slack=4.0, cooldown=n // 10)

    lam_true, lam_trk = [], []
    L = laplacian(g.A)
    for t in range(events):
        if cp_every and t > 0 and t % cp_every == 0:
            kind = ["pin_drop", "split", "bridge_cut", "pout_jump"][(t // cp_every - 1) % 4]
            g.trigger(t, kind)
            L = laplacian(g.A)  # batched CPs applied to structure
        else:
            i, j, s = g.next_event()
            if s > 0:
                if L[i, j] == 0:
                    L[i, i] += 1; L[j, j] += 1; L[i, j] = L[j, i] = -1
            else:
                if L[i, j] < 0:
                    L[i, i] -= 1; L[j, j] -= 1; L[i, j] = L[j, i] = 0
        Lcsr = L.tocsr()
        lr = trk.update(-1, -1, 0, Lcsr)
        if t % exact_every == 0:
            lt = base.update(-1, -1, 0, Lcsr)
        else:
            lt = lam_true[-1] if lam_true else base.update(-1, -1, 0, Lcsr)
        lam_true.append(lt)
        lam_trk.append(lr)
        det.update(t, lr)

    lam_true = np.array(lam_true)
    lam_trk = np.array(lam_trk)
    rel_err = np.abs(lam_trk - lam_true) / np.maximum(lam_true, 1e-9)
    cps = [c["t"] for c in g.changepoints]
    ev = evaluate_detection(det.alarms, cps)

    out = {
        "config": {"n": n, "events": events, "cp_every": cp_every, "seed": seed},
        "changepoints": g.changepoints,
        "tracker": {
            "restarts": trk.n_restart, "refines": trk.n_refine,
            "avg_time_ms": 1e3 * float(np.mean(trk.time_per_event)),
            "p95_time_ms": 1e3 * float(np.percentile(trk.time_per_event, 95)),
        },
        "baseline": {"avg_time_ms": 1e3 * float(np.mean(base.time_per_event))},
        "accuracy": {
            "lambda2_rel_err_mean": float(rel_err.mean()),
            "lambda2_rel_err_p95": float(np.percentile(rel_err, 95)),
        },
        "detection": ev,
        "alarms": det.alarms,
    }
    return out, lam_true, lam_trk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--cp-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact-every", type=int, default=10)
    args = ap.parse_args()
    t0 = time.time()
    res, lt, lr = run(args.n, args.events, args.cp_every, args.seed, args.exact_every)
    wall = time.time() - t0
    res["wall_time_s"] = wall
    os.makedirs("../runs", exist_ok=True)
    path = f"../runs/sim_v1_n{args.n}_e{args.events}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    print("saved ->", path)
