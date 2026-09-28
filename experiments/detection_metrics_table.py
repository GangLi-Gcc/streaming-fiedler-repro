#!/usr/bin/env python3
"""Full detection metrics (precision / recall / FAR / delay + Wilson CIs).

Uses the CORRECTED false-alarm accounting in lad_baseline.evaluate(): every
alarm outside the match window [cp, cp+tol] is a false alarm, including late
alarms.  The earlier version counted only alarms with a < cp, which reported
fp = 0 for runs that in fact raised many alarms.

Outputs runs/detection_metrics.json and prints paper-ready tables.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOL = 40
CP = 100

DATASETS = {
    "email-Eu": {
        "file": "lad_baseline.json",
        "n_samples": 466,
        "runs": ["inj015a", "inj015b", "inj030a", "inj030b",
                 "inj050a", "inj050b", "sc06a", "sc06b"],
    },
    "CollegeMsg": {
        "file": "lad_baseline-college-msg.json",
        "n_samples": 177,
        "runs": ["cmsg-inj015a", "cmsg-inj015b", "cmsg-inj030a", "cmsg-inj030b",
                 "cmsg-inj050a", "cmsg-inj050b", "cmsg-sc06a", "cmsg-sc06b"],
    },
}
CHANNELS = [("vecD", "vec-D window-$z$"),
            ("lam", "$\\lambda_2$ window-$z$"),
            ("LAD", "LAD (replayed)"),
            ("SCPD", "SCPD (replayed)")]

# SCPD lives in its own results file keyed by tag
SCPD_FILE = "scpd_baseline.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


out = {}
for dsname, cfg in DATASETS.items():
    data = json.loads((ROOT / "runs" / cfg["file"]).read_text())
    rows = {r["run"]: r for r in data["rows"]}

    # merge SCPD (stored separately, keyed by tag) into the same row structure
    scpd = {r["tag"]: r for r in json.loads((ROOT / "runs" / SCPD_FILE).read_text())}
    for run in cfg["runs"]:
        s = scpd.get(run)
        rows[run]["SCPD"] = ([bool(s["detected"]), int(s["fp"]), s["delay"]]
                             if s else [False, 0, None])

    eligible = cfg["n_samples"] - (TOL + 1)   # samples where an alarm is an FP
    n_cp = len(cfg["runs"])

    print(f"\n{'='*78}\n{dsname}  (n_samples={cfg['n_samples']}, tol={TOL}, "
          f"cp={CP}, eligible-FP samples/config={eligible})\n{'='*78}")
    print(f"{'channel':22s} {'rec':>7s} {'95% CI':>14s} {'FP':>4s} "
          f"{'prec':>6s} {'FAR/1k':>7s} {'delay':>12s}")
    print("-" * 78)

    ds = {"n_samples": cfg["n_samples"], "tol": TOL, "cp": CP,
          "eligible_per_config": eligible, "n_cps": n_cp, "channels": {}}

    for key, label in CHANNELS:
        tp = sum(1 for r in cfg["runs"] if rows[r][key][0])
        fp = sum(rows[r][key][1] for r in cfg["runs"])
        delays = [rows[r][key][2] for r in cfg["runs"] if rows[r][key][2] is not None]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_cp
        lo, hi = wilson(tp, n_cp)
        far = fp / (eligible * n_cp)          # per eligible sample, pooled
        dstr = (f"{min(delays)}-{max(delays)}" if delays else "n/a")
        fp_by_run = {r: rows[r][key][1] for r in cfg["runs"] if rows[r][key][1]}

        print(f"{label:22s} {tp}/{n_cp:<5d} [{lo:.2f},{hi:.2f}]  {fp:>4d} "
              f"{prec:>6.2f} {1000*far:>7.2f} {dstr:>12s}")

        ds["channels"][key] = {
            "label": label, "tp": tp, "fp": fp, "n_cps": n_cp,
            "recall": rec, "recall_ci95": [lo, hi], "precision": prec,
            "far_per_sample": far, "far_per_1000": 1000 * far,
            "delays": delays,
            "delay_min": min(delays) if delays else None,
            "delay_max": max(delays) if delays else None,
            "fp_by_run": fp_by_run,
        }
    out[dsname] = ds

    print("\n  false alarms by configuration:")
    for key, label in CHANNELS:
        fpr = ds["channels"][key]["fp_by_run"]
        print(f"    {label:22s} {fpr if fpr else '(none)'}")

(ROOT / "runs" / "detection_metrics.json").write_text(json.dumps(out, indent=1))
print(f"\n--> runs/detection_metrics.json")
