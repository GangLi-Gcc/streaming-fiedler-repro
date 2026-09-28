#!/usr/bin/env python3
"""Union-level detector comparison with STRICT false-alarm accounting.

Our detector is a union of channels; LAD and SCPD as published are single-
signature detectors.  Comparing a union against a single signature would
flatter us, so this script reports both readings, and -- unlike the earlier
version of this script -- pairs every detection count with its false-alarm
burden under the corrected accounting of lad_baseline.evaluate() (every alarm
outside [cp, cp+tol] is a false alarm, including late alarms).

Outputs runs/union_comparison.json
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOL = 40

DATASETS = {
    "email-Eu": {
        "lad": "lad_baseline.json",
        "n_samples": 466,
        "runs": ["inj015a", "inj015b", "inj030a", "inj030b",
                 "inj050a", "inj050b", "sc06a", "sc06b"],
    },
    "CollegeMsg": {
        "lad": "lad_baseline-college-msg.json",
        "n_samples": 177,
        "runs": ["cmsg-inj015a", "cmsg-inj015b", "cmsg-inj030a", "cmsg-inj030b",
                 "cmsg-inj050a", "cmsg-inj050b", "cmsg-sc06a", "cmsg-sc06b"],
    },
}

COMBOS = [
    ("Ours: vec-D only",            ["vecD"]),
    ("Ours: $\\lambda_2$-z only",   ["lam"]),
    ("Ours: vec-D $\\cup$ $\\lambda_2$-z", ["vecD", "lam"]),
    ("LAD (replayed)",              ["LAD"]),
    ("SCPD (replayed)",             ["SCPD"]),
    ("LAD $\\cup$ SCPD",            ["LAD", "SCPD"]),
]

out = {}
for dsname, cfg in DATASETS.items():
    lad = {r["run"]: r for r in
           json.loads((ROOT / "runs" / cfg["lad"]).read_text())["rows"]}
    scpd = {r["tag"]: r for r in
            json.loads((ROOT / "runs" / "scpd_baseline.json").read_text())}
    runs = cfg["runs"]
    eligible = cfg["n_samples"] - (TOL + 1)
    total_elig = eligible * len(runs)

    def get(run, key):
        if key == "SCPD":
            s = scpd.get(run)
            return (bool(s["detected"]), int(s["fp"])) if s else (False, 0)
        return (bool(lad[run][key][0]), int(lad[run][key][1]))

    print(f"\n{'='*70}\n{dsname}  (eligible non-CP samples = {total_elig})\n{'='*70}")
    print(f"{'detector':34s} {'det':>5s} {'FP':>5s} {'prec':>6s} {'FAR/1k':>7s}")
    print("-" * 70)

    rows = []
    for name, keys in COMBOS:
        tp = fp = 0
        for r in runs:
            hits = []
            for k in keys:
                d, f = get(r, k)
                hits.append(d)
                fp += f
            tp += any(hits)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        far1k = 1000 * fp / total_elig
        print(f"{name:34s} {tp}/{len(runs)} {fp:>5d} {prec:>6.2f} {far1k:>7.2f}")
        rows.append({"detector": name, "channels": keys, "detected": tp,
                     "n_cps": len(runs), "fp": fp, "precision": prec,
                     "far_per_1000": far1k})

    out[dsname] = {"n_samples": cfg["n_samples"], "tol": TOL,
                   "eligible_total": total_elig, "rows": rows}

(ROOT / "runs" / "union_comparison.json").write_text(json.dumps(out, indent=1))
print(f"\n--> runs/union_comparison.json")
print("\nNote: detection counts are unchanged by the accounting fix; the false-")
print("alarm columns are new and were previously reported as zero because only")
print("pre-changepoint alarms were counted.")
