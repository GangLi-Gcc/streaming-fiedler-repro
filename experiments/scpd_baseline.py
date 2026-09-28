"""P11: SCPD (Huang, Danovitch, Rabusseau, Rabbany; PAKDD 2023,
arXiv:2305.08750) structural-channel replay on the P0i streams.

SCPD's contribution is the SIGNATURE, not the scorer: a per-snapshot
density-of-states (DOS) embedding -- a 50-bin histogram of the normalized
Laplacian spectrum (paper: KPM-approximated; here computed exactly, which is
the same quantity at n=300/500) -- fed to a dual-sliding-window subspace
scorer.  Its attribute channel (LDOS, |x^T q_i|^2 bucketed correlations) is
NOT applicable: email-Eu / CollegeMsg carry no node attributes.

Protocol (mirrors lad_baseline.py exactly for threshold-fairness):
  - IDENTICAL stream builder and injection tags as the LAD replay
  - scorer: same dual windows (s=5, l=10), normal direction = 1st PC of the
    window signatures, Z = cosine distance, Z* = positive increment
    (hyperparameters identical to the LAD replay -> isolates the signature)
  - calibration: zero-alarm budget on the control segment + 1e-6 relative
    margin (SCPD cosine score, like LAD, has no sig floor -> ungated)
  - evaluation: det if any alarm in [cp, cp+40]; fp = every alarm outside that
    window (see lad_baseline.evaluate)

Outputs: runs/scpd_dos_{tag}.npz (dos series cache), runs/scpd_baseline.json
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lad_baseline import (DATASETS, TOL, evaluate, set_dataset,  # noqa: E402
                          lad_zstar, _zmax_series, _lad_alarms)

NBINS = 50


def dos_signature(Lcsr):
    """50-bin histogram of the normalized Laplacian spectrum.

    L_sym = I - D^{-1/2} A D^{-1/2} (zero-degree rows left as identity);
    eigenvalues in [0, 2]; histogram L1-normalized.  This is SCPD's
    structural signature computed exactly instead of KPM-approximated.
    """
    n = Lcsr.shape[0]
    d = np.asarray(Lcsr.diagonal()).copy()
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    # A = D - L (off-diagonal -w); normalize
    Dm = sp.diags(dinv)
    A = (sp.diags(d) - Lcsr).tocsr()
    Lsym = sp.eye(n, format="csr") - Dm @ A @ Dm
    vals = np.linalg.eigvalsh(Lsym.toarray())
    hist, _ = np.histogram(vals, bins=NBINS, range=(0.0, 2.0))
    h = hist.astype(float)
    return h / (h.sum() + 1e-300)


def replay_dos(tag, inject):
    """Stream loop identical to lad_baseline.replay(), DOS signature only."""
    from lad_baseline import load_edges, CFG
    from collections import Counter

    edges, nodes = load_edges()
    remap = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    tau_edges = CFG["tau_steps"] * CFG["macro"]
    sample_every = CFG["macro"] * 50
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = CFG["prune_w"]
    inject_at = CFG["inject_at"]
    frac, scale, seed = (inject + (None,))[:3] if inject else (None, None, None)

    dos_v, ts_v = [], []
    pair_w = {}
    steps = 0
    t_start = time.time()
    batch = []
    for idx, (u, v, ts) in enumerate(edges):
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
        if inject is not None and steps == inject_at:
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
        for a, dd in deg.items():
            rows.append(a)
            cols.append(a)
            vals.append(dd)
        Lcsr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        dos_v.append(dos_signature(Lcsr))
        ts_v.append(ts)
        steps += 1
    print(f"  replay[{tag}] {steps} samples in {time.time()-t_start:.0f}s", flush=True)
    return np.asarray(dos_v), np.asarray(ts_v)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="email-eu")
    ap.add_argument("--only", default=None, help="run a single tag")
    cli = ap.parse_args()
    set_dataset(cli.dataset)
    control_tag = next(t for t, s in DATASETS[cli.dataset]["runs"].items() if s is None)
    stationary0 = DATASETS[cli.dataset]["stationary0"]
    runs = dict(DATASETS[cli.dataset]["runs"])
    if cli.only:
        runs = {cli.only: runs[cli.only]}

    data = {}
    for tag, inject in runs.items():
        cache = f"../runs/scpd_dos_{tag}.npz"
        if os.path.exists(cache):
            d = np.load(cache)
            data[tag] = (np.asarray(d["dos"], dtype=float),
                         np.asarray(d["ts"], dtype=float))
        else:
            dos, ts = replay_dos(tag, inject)
            data[tag] = (dos, ts)
            np.savez_compressed(cache, dos=dos, ts=ts)

    # ---- SCPD score: same scorer as LAD replay, DOS signature swapped in ----
    ch = {}
    for tag, (dos, ts) in data.items():
        seg = slice(stationary0, None)
        ch[tag] = {"SCPD": lad_zstar(dos[seg]), "ts": ts[seg] / 86400.0}

    # ---- zero-alarm-budget calibration on control segment ----
    _, zmax = _zmax_series(ch[control_tag]["SCPD"])
    cal = zmax * (1 + 1e-6)
    print(f"control calibration (zero-alarm + ulp): {cal:.4e}", flush=True)

    # ---- evaluate per run ----
    recs = []
    for tag in runs:
        alarms, _ = _lad_alarms(ch[tag]["SCPD"], cal)
        det, fp, delay = evaluate(alarms)
        recs.append({"tag": tag, "dataset": cli.dataset, "detected": bool(det),
                     "fp": int(fp), "delay": delay, "n_alarms": len(alarms)})
        print(f"  {tag}: det={det} fp={fp} delay={delay} alarms={len(alarms)}",
              flush=True)

    # ---- merge into runs/scpd_baseline.json ----
    out_path = "../runs/scpd_baseline.json"
    if os.path.exists(out_path):
        out = json.load(open(out_path, encoding="utf-8"))
    else:
        out = []
    out = [r for r in out if r["dataset"] != cli.dataset] + recs
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    det_tags = [r["tag"] for r in recs if r["detected"]]
    print(f"\n{cli.dataset}: SCPD detected {len(det_tags)}/{len(recs)} "
          f"injected CPs: {det_tags}")
    print("saved runs/scpd_baseline.json")


if __name__ == "__main__":
    main()
