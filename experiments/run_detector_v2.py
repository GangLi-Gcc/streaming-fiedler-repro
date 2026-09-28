"""Task #2 sweep: SelfNormalizingDrift (v2) vs naive CUSUM on planted CPs.

Configs: no-CP control + 3 CP kinds (pin_drop x0.4, split, bridge_cut),
3 seeds each. Metrics: precision/recall/F1, mean detection delay, and
false alarms per 1000 events on the control runs.

Usage: python run_detector_v2.py [--n 1200] [--events 2500] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_baseline import run_stream
from detectors import CUSUMDrift, evaluate_detection
from detectors_v2 import SelfNormalizingDrift, alarms_per_1000

TOL = 300  # alarm-within tolerance for matching planted CPs


def run_detector_on(lam, det):
    for t, v in enumerate(lam):
        det.update(t, v)
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--events", type=int, default=2500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cp-at", type=int, default=1250)
    args = ap.parse_args()

    configs = [
        ("control", None, None),
        ("pin_drop", "pin_drop", 0.4),
        ("split", "split", None),
        ("bridge_cut", "bridge_cut", None),
    ]

    def make_detectors():
        return {
            "cusum_naive": CUSUMDrift(window=240, min_slack=4.0, cooldown=120),
            "selfnorm_v2": SelfNormalizingDrift(win_base=300, win_recent=100,
                                                threshold=4.0, cooldown=120),
        }

    rows = []
    cache_path = "../runs/traj_cache.npz"
    cache = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}

    def get_traj(cname, ckind, cfactor, seed):
        key = f"{cname}_{seed}"
        if key not in cache:
            lam, _, _ = run_stream(args.n, args.events, seed,
                                   cp_at=args.cp_at if ckind else None,
                                   cp_kind=ckind or "pin_drop",
                                   cp_factor=cfactor or 0.4)
            cache[key] = lam
            np.savez_compressed(cache_path, **cache)
        return cache[key]

    for cname, ckind, cfactor in configs:
        for seed in range(args.seeds):
            lam = get_traj(cname, ckind, cfactor, seed)
            for dname, det in make_detectors().items():
                run_detector_on(lam, det)
                cps = [args.cp_at] if ckind else []
                ev = evaluate_detection(det.alarms, cps, tol=TOL)
                rows.append({
                    "config": cname, "seed": seed, "detector": dname,
                    "tp": ev["tp"], "fp": ev["fp"], "fn": ev["fn"],
                    "f1": round(ev["f1"], 3),
                    "mean_delay": ev["mean_delay"],
                    "alarms_per_1000": round(alarms_per_1000(det.alarms, args.events), 2),
                })
                print(f"[{cname} seed{seed} {dname}] "
                      f"tp={ev['tp']} fp={ev['fp']} f1={ev['f1']:.3f} "
                      f"alarms/1k={alarms_per_1000(det.alarms, args.events):.2f}",
                      flush=True)

    # aggregate
    agg = {}
    for r in rows:
        key = (r["config"], r["detector"])
        agg.setdefault(key, []).append(r)
    summary = []
    for (cfg, det), rs in sorted(agg.items()):
        summary.append({
            "config": cfg, "detector": det, "seeds": len(rs),
            "f1_mean": round(float(np.mean([r["f1"] for r in rs])), 3),
            "alarms_per_1000_mean": round(float(np.mean([r["alarms_per_1000"] for r in rs])), 2),
            "delays": [r["mean_delay"] for r in rs],
        })

    out = {"config": vars(args), "summary": summary, "rows": rows}
    os.makedirs("../runs", exist_ok=True)
    with open("../runs/detector_v2_sweep.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
