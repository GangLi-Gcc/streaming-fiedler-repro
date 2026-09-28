"""P0n-part1: per-component spectral tracking for the disconnected regime.

Motivation (P0j finding #1/#2): on CollegeMsg (98% of samples have global
lam2 ~ 0), every per-snapshot GLOBAL spectral channel inherits a degenerate
eigenspace: eigsh noise on the zero eigenvalue's multiplicity is amplified by
the incremental-MAD 1e-12 sig floor -> phantom alarms (lam-z) or elevated fp
(vec-D). A connectivity gate is infeasible (1.9% connected fraction).

Design: track the spectrum of the DOMINANT CONNECTED COMPONENT instead of
the whole graph:
  - components from the support pattern of the weighted Laplacian
    (scipy.sparse.csgraph.connected_components, undirected);
  - lam2_dom / v2_dom = Fiedler pair of the dominant component's induced
    sub-Laplacian (same shift-invert numerics as everywhere else);
  - ncomp = component count (exact integer -- solver-noise-free, equals the
    multiplicity of the zero eigenvalue);
  - ndom = dominant component size (diagnostic).

The stream builder and injections are bit-identical to lad_baseline.replay
(imported from there); this script reuses its global eig pass and adds the
per-component pass per sample. One run per invocation (cache per run).

Output: ../runs/pc_{tag}.npz with lam_dom, v2_dom (zero-padded to n, float32),
mask_dom (bool (samples, n)), ncomp, ndom, plus global lam/spec for reference.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lad_baseline as lb  # noqa: E402
from lad_baseline import snapshot_eigs, set_dataset  # noqa: E402


def dom_eigs(Lcsr):
    """Per-component decomposition + dominant-component Fiedler pair."""
    n = Lcsr.shape[0]
    supp = (Lcsr != 0)
    supp.setdiag(0)
    supp = supp.tocsr()
    supp.data[:] = 1.0
    ncomp, labels = connected_components(supp, directed=False)
    if ncomp <= 1:
        dom = np.arange(n)
    else:
        counts = np.bincount(labels, minlength=ncomp)
        dom = np.where(labels == int(np.argmax(counts)))[0]
    ndom = len(dom)
    if ndom < 2:
        return ncomp, ndom, np.nan, np.zeros(n)
    Ld = Lcsr[dom][:, dom].tocsr()
    try:
        # snapshot_eigs returns (vals, v2_single_vector) -- NOT the full basis
        vals, v2d = snapshot_eigs(Ld, k=min(3, ndom - 1))
        lam2 = float(vals[1]) if len(vals) > 1 else np.nan
        v = np.asarray(v2d, dtype=float).ravel() if len(vals) > 1 else np.zeros(ndom)
    except Exception:
        lam2, v = np.nan, np.zeros(ndom)
    v_full = np.zeros(n)
    v_full[dom] = v
    nv = np.linalg.norm(v_full)
    if nv > 0:
        v_full /= nv
    return ncomp, ndom, lam2, v_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="college-msg")
    ap.add_argument("--run", required=True, help="run tag, e.g. cmsg-inj015a")
    cli = ap.parse_args()
    set_dataset(cli.dataset)
    inject = lb.RUNS[cli.run]
    t0 = time.time()
    # global eig pass: reuse the cached bit-identical replay when available
    cache = f"../runs/real_{cli.run}_spec.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        lam, spec, ts = d["lam"], d["spec"], d["ts"]
    else:
        lam, spec, _, ts = lb.replay(cli.run, inject)
    # per-component pass: replay again but intercept Laplacians cheaply by
    # re-walking the same construction is costly; instead recompute from
    # replay's saved global eig is not enough -> we re-run the builder via
    # replay internals. To avoid duplicating builder logic we monkey-run
    # replay a second time is wasteful; simpler: rebuild Laplacians by
    # replaying the edge stream with the same code path but skipping eigsh.
    edges, nodes = lb.load_edges()
    remap = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    tau_edges = lb.CFG["tau_steps"] * lb.CFG["macro"]
    sample_every = lb.CFG["macro"] * 50
    gamma = float(np.exp(-1.0 / tau_edges))
    prune_w = lb.CFG["prune_w"]
    frac, scale, seed = (inject + (None,))[:3] if inject else (None, None, None)
    lam_dom, v2_dom, mask_dom, ncomp_v, ndom_v = [], [], [], [], []
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
        nc, nd, l2, vfull = dom_eigs(Lcsr)
        ncomp_v.append(nc)
        ndom_v.append(nd)
        lam_dom.append(l2)
        v2_dom.append(vfull.astype(np.float32))
        mask_dom.append(np.abs(vfull) > 0)
        steps += 1
    out = f"../runs/pc_{cli.run}.npz"
    np.savez_compressed(
        out, lam=np.asarray(lam), spec=np.asarray(spec), ts=np.asarray(ts),
        lam_dom=np.asarray(lam_dom, dtype=float),
        v2_dom=np.asarray(v2_dom, dtype=np.float32),
        mask_dom=np.asarray(mask_dom, dtype=bool),
        ncomp=np.asarray(ncomp_v, dtype=int),
        ndom=np.asarray(ndom_v, dtype=int))
    print(f"saved {out}: {steps} samples, "
          f"ncomp range [{min(ncomp_v)},{max(ncomp_v)}], "
          f"ndom range [{min(ndom_v)},{max(ndom_v)}], "
          f"lam_dom finite frac {np.mean(np.isfinite(lam_dom)):.3f} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
