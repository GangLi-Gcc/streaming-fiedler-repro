#!/usr/bin/env python3
"""Verify LAD smallest-vs-largest signature gives identical alarm sets on email-Eu.

For each snapshot compute both the 6 smallest and 6 largest Laplacian
eigenvalues, run the LAD Z* channel on each signature, calibrate on the control
stream, and compare per-run alarm timestamps (segment indices) and event peaks.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATASETS, build_stream, fiedler_exact, lad_zstar
import scipy.sparse.linalg as spla

TOL = 40


def largest_eigs(L, k=6):
    """k largest eigenvalues of the Laplacian (descending)."""
    n = L.shape[0]
    vals = spla.eigsh(L, k=k, which="LA", return_eigenvectors=False, tol=1e-8)
    return np.sort(vals)[::-1]  # descending


def main():
    cfg = DATASETS["email-eu"]
    stationary0 = cfg["stationary0"]
    cp = cfg["inject_at"] - stationary0

    specs = {"smallest": {}, "largest": {}}
    for tag in ["control"] + list(cfg["runs"].keys()):
        inject = None if tag == "control" else cfg["runs"][tag]
        snapshots, n = build_stream("email-eu", inject=inject)
        small, large = [], []
        for L, ts in snapshots:
            vals, _ = fiedler_exact(L, k=6)      # 6 smallest
            small.append(vals[:6])
            large.append(largest_eigs(L, k=6))   # 6 largest (descending)
        specs["smallest"][tag] = np.asarray(small)
        specs["largest"][tag] = np.asarray(large)

    seg = slice(stationary0, None)

    def run(sign):
        # calibration on control
        zc = lad_zstar(specs[sign]["control"][seg])
        thr = float(np.nanmax(zc)) * (1 + 1e-6)
        out = {}
        for tag in cfg["runs"]:
            z = lad_zstar(specs[sign][tag][seg])
            alarms = [t for t in range(len(z)) if np.isfinite(z[t]) and z[t] > thr]
            det = any(0 <= a - cp <= TOL for a in alarms)
            out[tag] = {"alarms_cp": sorted(a - cp for a in alarms), "det": bool(det)}
        return thr, out

    thr_s, res_s = run("smallest")
    thr_l, res_l = run("largest")
    print(f"smallest thr={thr_s:.6g}   largest thr={thr_l:.6g}")
    identical = True
    for tag in cfg["runs"]:
        a = res_s[tag]; b = res_l[tag]
        same = a["alarms_cp"] == b["alarms_cp"] and a["det"] == b["det"]
        identical &= same
        print(f"  {tag:10s} smallest det={int(a['det'])} alarms_cp={a['alarms_cp']}")
        print(f"  {'':10s} largest  det={int(b['det'])} alarms_cp={b['alarms_cp']}  "
              f"{'IDENTICAL' if same else 'DIFFER'}")
    print(f"\nIDENTICAL ALARM SETS: {identical}")


if __name__ == "__main__":
    main()
