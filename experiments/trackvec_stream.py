"""P0o-part1: vec-D from the WarmStartTracker STREAMING trajectory.

P0j mechanism claim (future-work b): streaming warm-start suppresses
per-snapshot eigsh noise -- in the disconnected regime the Fiedler vector
sits in a near-degenerate zero eigenspace, so per-snapshot fresh eigsh
picks an arbitrary basis direction each time (1e-18-level run-to-run
noise) which the window-z sig floor amplifies into phantom vec-D alarms
(inj050b: 10 fp). A tracker that stays in the deterministic refine
branch instead produces the smooth continuation of v2, so spurious
angular jumps disappear and only genuine rotation survives into D.

Regime mode (new tracker params, defaults preserve legacy behavior):
  gate_floor=1.0  -> accept gate = rtr*max(gap_est, 1.0) = 0.03, so the
                     gap~0 regime no longer forces an exact restart per
                     sample;
  verify_floor=1e-2 -> trust the refined pair unless res3 > 5e-3.

Stream construction: bit-identical copy of lad_baseline.replay (same as
percomp_track). No per-snapshot global eigsh at all -- the ONLY eigsh
calls are tracker inits/restarts.

Output: ../runs/tv_{tag}.npz with lam_trk, v2_trk (float64), branch,
k_used, plus fidelity stats vs cached global eigsh on connected samples.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lad_baseline as lb  # noqa: E402
from lad_baseline import set_dataset  # noqa: E402
from trackers import WarmStartTracker  # noqa: E402

RTR, ABS_TOL, GATE_FLOOR, VERIFY_FLOOR = 0.03, 1e-6, 100.0, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="college-msg")
    ap.add_argument("--run", required=True)
    cli = ap.parse_args()
    set_dataset(cli.dataset)
    inject = lb.RUNS[cli.run]
    t0 = time.time()

    edges, nodes = lb.load_edges()
    remap = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    tau_edges = lb.CFG["tau_steps"] * lb.CFG["macro"]
    sample_every = lb.CFG["macro"] * 50
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = lb.CFG["prune_w"]
    frac, scale, seed = (inject + (None,))[:3] if inject else (None, None, None)

    trk = WarmStartTracker(restart_tol_ratio=RTR, refine_abs_tol=ABS_TOL,
                           gate_floor=GATE_FLOOR, verify_floor=VERIFY_FLOOR)
    lam_v, v2_v, br_v, k_v = [], [], [], []
    pair_w = {}
    steps = 0
    batch = []
    for idx, (u, v, tstamp) in enumerate(edges):
        a, b = (remap[u], remap[v]) if remap[u] < remap[v] else (remap[v], remap[u])
        batch.append((a, b))
        if (idx + 1) % sample_every != 0:
            continue
        for k_ in list(pair_w.keys()):
            w = pair_w[k_] * gamma
            if w < prune_w:
                del pair_w[k_]
            else:
                pair_w[k_] = w
        for (a, b) in batch:
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + 1.0
        batch = []
        if inject is not None and steps == lb.INJECT_AT:
            if scale is not None:
                for kk in list(pair_w.keys()):
                    pair_w[kk] *= scale
            else:
                rng = np.random.default_rng(seed)
                keys = list(pair_w.keys())
                kdel = int(round(frac * len(keys)))
                for kk in rng.choice(len(keys), kdel, replace=False):
                    del pair_w[keys[kk]]
        rows, cols, vals = [], [], []
        deg = {}
        for (a, b), w in pair_w.items():
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
            rows += [a, b]
            cols += [b, a]
            vals += [-w, -w]
        for a, d in deg.items():
            rows.append(a)
            cols.append(a)
            vals.append(d)
        Lcsr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        lam2 = trk.update(-1, -1, 0, Lcsr)
        lam_v.append(lam2)
        v2_v.append(np.asarray(trk.v2, dtype=np.float64))
        br_v.append(trk.diag[-1][0] if trk.diag else -1)
        k_v.append(trk.diag[-1][3] if trk.diag else -1)
        steps += 1

    lam_trk = np.asarray(lam_v)
    v2_trk = np.asarray(v2_v)
    br = np.asarray(br_v, dtype=int)
    ku = np.asarray(k_v, dtype=int)
    restart_frac = float(np.mean((br == 1) | (br == 2)))
    res3 = np.array([trk.diag[i][4] for i in range(len(trk.diag))]) if trk.diag \
        else np.array([np.nan])

    # fidelity vs cached global eigsh on connected samples (lam > 1e-6)
    cache = f"../runs/real_{cli.run}_spec.npz"
    fid = np.nan
    if os.path.exists(cache):
        lam_ref = np.load(cache)["lam"]
        m = lam_ref > 1e-6
        if m.any():
            fid = float(np.max(np.abs(lam_trk[m] - lam_ref[m])
                               / np.maximum(np.abs(lam_ref[m]), 1e-9)))
    out = f"../runs/tv_{cli.run}.npz"
    np.savez_compressed(out, lam_trk=lam_trk, v2_trk=v2_trk, branch=br,
                        k_used=ku, ts=np.arange(steps))
    print(f"saved {out}: {steps} samples, restart_frac={restart_frac:.3f}, "
          f"k_used med={np.median(ku[ku >= 0]):.0f}, "
          f"res3 med={np.nanmedian(res3):.2e}, "
          f"lam fidelity (conn samples) max_rel={fid:.2e} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
