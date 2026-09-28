#!/usr/bin/env python3
"""A3: Scale experiment — MDR scaling at n=500, n=1000, n=2000 SBM.

Tests whether the sqrt(stiffness) design rule holds across scales.
Runs the minimal MDR rig (3 hub levels x 3 p_near x 3 seeds) at each n.
Outputs runs/scale_experiment.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import json, time
import numpy as np
import scipy.sparse as sp
from pathlib import Path

from trackers import fiedler_exact, WarmStartTracker

ROOT = Path(__file__).resolve().parent.parent


def make_sbm(n_per, p_in, p_out, w_hub, seed):
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < n_per) == (j < n_per)
            p = p_in if same else p_out
            if rng.random() < p:
                rows += [i, j]; cols += [j, i]; vals += [1.0, 1.0]
    if w_hub > 0:
        hub0, hub1 = n, n + 1
        n += 2
        for v in range(n_per):
            rows += [hub0, v]; cols += [v, hub0]; vals += [w_hub, w_hub]
        for v in range(n_per, 2 * n_per):
            rows += [hub1, v]; cols += [v, hub1]; vals += [w_hub, w_hub]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    d = np.array(A.sum(axis=1)).ravel()
    return (sp.diags(d) - A).astype(float)


def mdr_for_config(n_per, p_out, w_hub, seed, n_delta=11):
    """Return (theta_star, k, sqrt_stiff, gap) for one configuration."""
    L = make_sbm(n_per=n_per, p_in=0.05, p_out=p_out, w_hub=w_hub, seed=seed)
    n = L.shape[0]
    i0, j0 = 0, n_per  # cross-block edge

    vals0, vecs0 = fiedler_exact(L, k=3)
    stiff = 2 * float(L.diagonal().max()) + 1 - vals0[1]
    gap = vals0[2] - vals0[1]

    tr = WarmStartTracker(restart_tol_ratio=0.03, refine_abs_tol=1e-6, max_refine=30)
    _ = tr.update(0, 1, +1, L)  # seed

    # adaptive delta grid: expand until we see att >= 0.5 or hit delta_max=1000
    delta_max = 10.0
    found = False
    for attempt in range(4):
        delta_grid = np.logspace(-4, np.log10(delta_max), n_delta)
        atts = []
        for delta in delta_grid:
            e = np.zeros(n); e[i0] = 1; e[j0] = -1
            dL = delta * sp.csr_matrix(np.outer(e, e))
            L_new = L + dL
            vals_new, vecs_new = fiedler_exact(L_new, k=3)
            v2_new = vecs_new[:, 1]
            d_exact = float(1 - abs(vecs0[:, 1] @ v2_new))

            tr.v2 = vecs0[:, 1].copy()
            if hasattr(tr, 'lam2'): tr.lam2 = vals0[1]
            if hasattr(tr, 'lam_est'): tr.lam_est = vals0[1]

            tr.update(i0, j0, +1, L_new)
            d_tr = float(1 - abs(vecs0[:, 1] @ tr.v2))
            att = d_tr / d_exact if d_exact > 1e-15 else float('nan')
            atts.append((delta, d_exact, d_tr, att))
            if not np.isnan(att) and att >= 0.5:
                found = True
                break
        if found:
            break
        delta_max *= 10.0  # expand and retry

    # find MDR: smallest delta with att >= 0.5
    theta_star = None
    for delta, d_ex, d_tr, att in atts:
        if not np.isnan(att) and att >= 0.5:
            theta_star = d_ex
            break
    if theta_star is None:
        # no crossing found even at delta=1000: record as nan
        theta_star = float('nan')

    k = theta_star / np.sqrt(stiff) if (stiff > 0 and theta_star == theta_star) else float('nan')
    return dict(n_per=n_per, p_out=p_out, w_hub=w_hub, seed=seed, n=n,
                theta_star=theta_star, k=k, sqrt_stiff=float(np.sqrt(stiff)),
                gap=float(gap), stiff=float(stiff), delta_max_used=float(delta_max/10))


def run():
    configs = []
    for n_per in [250, 500, 1000]:  # n = 500, 1000, 2000
        for p_out in [0.005, 0.02, 0.08]:
            for w_hub in [0.0, 10.0, 30.0]:
                for seed in [0, 1, 2]:
                    configs.append((n_per, p_out, w_hub, seed))

    print(f"Running {len(configs)} configurations...")
    results = []
    for idx, (n_per, p_out, w_hub, seed) in enumerate(configs):
        t0 = time.perf_counter()
        row = mdr_for_config(n_per, p_out, w_hub, seed)
        dt = time.perf_counter() - t0
        results.append(row)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(configs)}] n={row['n']} p={p_out} w={w_hub} "
                  f"seed={seed} k={row['k']:.3g} t={dt:.1f}s")

    out = ROOT / 'runs' / 'scale_experiment.json'
    out.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {len(results)} rows -> {out}")

    print("\n=== Summary: k by n_per and w_hub ===")
    import numpy as np
    for n_per in [250, 500, 1000]:
        for w in [0.0, 10.0, 30.0]:
            rows = [r for r in results if r['n_per'] == n_per and r['w_hub'] == w]
            ks = [r['k'] for r in rows if not np.isnan(r['k']) and r['k'] > 0]
            if ks:
                print(f"  n={2*n_per} w={w}: k median={np.median(ks):.3g} "
                      f"min={min(ks):.3g} max={max(ks):.3g} n={len(ks)}")


if __name__ == '__main__':
    run()
