#!/usr/bin/env python3
"""Generate the per-seed MTR crossing table for paper Sec. 4 (Round 10)."""
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recs = json.load(open(os.path.join(ROOT, "rebuild", "results", "mdr_configs.json"),
                      encoding="utf-8"))
cells = defaultdict(list)
for r in recs:
    cells[(r["p_near"], r["w_hub"], r["seed"])].append(r)


def sci(x, nd=1):
    if x is None:
        return r"$\---$"
    if x < 1e-12:
        return r"${\approx}0$"
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"${m:.{nd}f}{{\\times}}10^{{{e}}}$"


rows = []
for (pn, wh, sd) in sorted(cells):
    rs = sorted(cells[(pn, wh, sd)], key=lambda r: r["delta"])
    att0 = rs[0]["att"]
    if att0 >= 0.5:
        r = rs[0]
        rows.append((pn, wh, sd, "L", None, None, r["att"], r["capture_dir"],
                     r["d_end"], "refine"))
    else:
        cross = None
        for i, r in enumerate(rs):
            if r["att"] >= 0.5:
                cross = i
                break
        lo, hi = rs[cross - 1], rs[cross]
        br = "restart" if hi["att"] >= 0.95 else "refine"
        rows.append((pn, wh, sd, "B", lo["theta"], hi["theta"], hi["att"],
                     hi["capture_dir"], hi["d_end"], br))

with open(os.path.join(ROOT, "rebuild", "results", "seed_table.tex"),
          "w", encoding="utf-8") as f:
    for pn, wh, sd, st, thl, thh, att, cap, dend, br in rows:
        thl_s = sci(thl) if thl is not None else r"$\---$"
        thh_s = sci(thh) if thh is not None else r"$\---$"
        br_s = "R" if br == "restart" else "r"
        st_s = "B" if st == "B" else "L"
        f.write(
            f"{pn:.3f} & {wh:3.0f} & {sd} & {st_s} & {thl_s} & {thh_s} & "
            f"${att:.3f}$ & ${cap:.3f}$ & {sci(dend)} & {br_s} \\\\\n")

# summary
B = [r for r in rows if r[3] == "B"]
L = [r for r in rows if r[3] == "L"]
sw = [r for r in B if r[9] == "restart"]
print("bracketed", len(B), "left-censored", len(L))
print("restart crossings", len(sw), "refine crossings", len(B) - len(sw))
print("att* among B: min", min(r[6] for r in B), "max", max(r[6] for r in B))
print("cap* among B: min", min(r[7] for r in B))
print("d_end/d_ex among B: max", max(1 - r[7] for r in B))
print()
print("capture_0 / att_0 median per config:")
for (pn, wh) in sorted(set((r[0], r[1]) for r in rows)):
    caps = [rr["capture_dir"] for rr in recs
            if rr["p_near"] == pn and rr["w_hub"] == wh and rr["delta"] == 0.0001]
    atts = [rr["att"] for rr in recs
            if rr["p_near"] == pn and rr["w_hub"] == wh and rr["delta"] == 0.0001]
    print(f"  p={pn:.3f} w={wh:3.0f}: median cap0={np.median(caps):.3f}  "
          f"median att0={np.median(atts):.3f}")
