"""P0: real-data lambda_2 noise-floor calibration.

Loads a SNAP temporal edge stream (SRC DST UNIXTS), symmetrizes it, and slides
an edge-count window over the stream. At each macro-step (batch of edges) the
graph Laplacian is updated incrementally and the Fiedler value is tracked with
WarmStartTracker (exact eigsh reference every K steps).

Answers: is the ~23% relative lambda_2 wander of the synthetic DCSBM event
model an artifact, or do real communication streams show the same spectral
noise? Also surfaces large excursions as CANDIDATE REAL CHANGEPOINTS (e.g.
academic breaks in email-Eu).

Usage: python real_data_noise.py --dataset email-eu|college-msg
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings
from collections import Counter, deque

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers import BaselineFull, WarmStartTracker

DATA = {
    "email-eu": {
        "path": "../data/email-Eu-core-temporal.txt.gz",
        "tau_steps": 2000,        # exp-decay memory (~20k edges effective)
        "macro": 10,              # edges applied per step
        "exact_every": 100,
        "prune_w": 0.02,
        "core_k": 300,            # restrict to top-K most active nodes
    },
    "college-msg": {
        "path": "../data/CollegeMsg.txt.gz",
        "tau_steps": 1500,        # ~7.5k edges effective
        "macro": 5,
        "exact_every": 100,
        "prune_w": 0.02,
        "core_k": 500,
    },
}


def load_edges(path):
    edges = []
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            u, v, ts = int(p[0]), int(p[1]), int(p[2])
            if u == v:
                continue
            edges.append((u, v, ts))
    edges.sort(key=lambda e: e[2])
    return edges


def noise_stats(lam, win=200):
    lam = np.asarray(lam, dtype=float)
    d = np.diff(lam)
    W = np.lib.stride_tricks.sliding_window_view(lam, win)
    wm = W.mean(axis=1)
    z = (wm - lam.mean()) / (d.std() * np.sqrt(win / 3.0) + 1e-12)
    return {
        "lam_mean": float(lam.mean()), "lam_std": float(lam.std()),
        "wander_ratio": float(lam.std() / max(lam.mean(), 1e-9)),
        "inc_std": float(d.std()), "inc_mean": float(d.mean()),
        "max_abs_window_z": float(np.abs(z).max()),
        "p99_abs_window_z": float(np.percentile(np.abs(z), 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATA), required=True)
    ap.add_argument("--inject-sample", type=int, default=None,
                    help="sample index at which to batch-remove inject-frac of remembered pairs")
    ap.add_argument("--inject-frac", type=float, default=0.3)
    ap.add_argument("--inject-seed", type=int, default=0)
    ap.add_argument("--inject-scale", type=float, default=None,
                    help="if set, multiply ALL remembered pair weights by this factor at inject-sample (monotone CP)")
    ap.add_argument("--tag", default=None, help="output file tag")
    args = ap.parse_args()
    cfg = DATA[args.dataset]

    print("loading edges...", flush=True)
    edges = load_edges(cfg["path"])
    # ---- restrict to the top-K most active nodes (persistent active core) ----
    # Without this, any finite window leaves most users inactive -> lambda_2 == 0
    # and the trajectory measures "fraction of active users", not structure.
    act = Counter()
    for u, v, _ in edges:
        act[u] += 1; act[v] += 1
    core = {x for x, _ in act.most_common(cfg["core_k"])}
    edges = [(u, v, ts) for (u, v, ts) in edges if u in core and v in core]
    nodes = sorted(core)
    remap = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    t0, t1 = edges[0][2], edges[-1][2]
    print(f"{args.dataset}: {len(edges)} temporal edges on core of {n} nodes, "
          f"span {(t1 - t0) / 86400:.1f} days", flush=True)

    tau_edges = cfg["tau_steps"] * cfg["macro"]   # decay memory in edges
    sample_every = cfg["macro"] * 50              # edges between lambda samples
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = cfg["prune_w"]
    trk = None                                    # exact-only sampling in this pass
    base = BaselineFull()

    lam_v, sample_ts, v2_v = [], [], []
    pair_w = {}
    steps = 0
    start = time.time()
    batch = []
    for idx, (u, v, ts) in enumerate(edges):
        a, b = (remap[u], remap[v]) if remap[u] < remap[v] else (remap[v], remap[u])
        batch.append((a, b))
        if (idx + 1) % sample_every != 0:
            continue
        # ---- apply decay to all remembered pairs (python loop over dict) ----
        for k in list(pair_w.keys()):
            w = pair_w[k] * gamma
            if w < prune_w:
                del pair_w[k]
            else:
                pair_w[k] = w
        for (a, b) in batch:
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + 1.0
        batch = []
        # ---- synthetic CP injection on the real stream ----
        if args.inject_sample is not None and steps == args.inject_sample:
            if args.inject_scale is not None:
                # monotone CP: global weight scaling -> lambda_2 drops by ~scale
                for kk in list(pair_w.keys()):
                    pair_w[kk] *= args.inject_scale
            else:
                rng = np.random.default_rng(args.inject_seed)
                keys = list(pair_w.keys())
                k = int(round(args.inject_frac * len(keys)))
                for kk in rng.choice(len(keys), k, replace=False):
                    del pair_w[keys[kk]]
        # ---- build L in C from the pair-weight dict ----
        rows, cols, vals = [], [], []
        deg = {}
        for (a, b), w in pair_w.items():
            deg[a] = deg.get(a, 0.0) + w
            deg[b] = deg.get(b, 0.0) + w
            rows += [a, b]; cols += [b, a]; vals += [-w, -w]
        for a, d in deg.items():
            rows.append(a); cols.append(a); vals.append(d)
        Lcsr = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        lam_v.append(base.update(-1, -1, 0, Lcsr))
        v2_v.append(np.asarray(base.v2, dtype=np.float32))
        sample_ts.append(ts)
        steps += 1
        if steps % 50 == 0:
            print(f"  sample {steps} ({time.time()-start:.0f}s, "
                  f"lam2={lam_v[-1]:.4f}, pairs={len(pair_w)})", flush=True)

    lam_trk = np.asarray(lam_v)
    stats = noise_stats(lam_trk, win=min(50, max(10, steps // 5)))
    stats["samples"] = steps
    stats["sample_every_edges"] = sample_every
    stats["baseline_avg_time_ms"] = 1e3 * float(np.mean(base.time_per_event)) if base.time_per_event else None

    tag = args.tag or args.dataset
    out = {"dataset": args.dataset, "stats": stats, "tag": tag,
           "tau_edges": tau_edges, "n_nodes": n,
           "inject_sample": args.inject_sample, "inject_frac": args.inject_frac,
           "inject_scale": args.inject_scale,
           "inject_seed": args.inject_seed}
    os.makedirs("../runs", exist_ok=True)
    np.savez_compressed(f"../runs/real_{tag}_lam.npz",
                        lam=lam_trk, ts=np.asarray(sample_ts),
                        v2=np.asarray(v2_v).T)   # (n_nodes, samples)
    with open(f"../runs/real_{tag}_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
