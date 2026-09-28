"""Change-point detectors on the lambda_2(t) trajectory.

CUSUMDrift : classic CUSUM on a sliding window with a detection threshold;
outputs alarm times. Evaluated against planted changepoints.
"""
from __future__ import annotations

import numpy as np


class CUSUMDrift:
    def __init__(self, window=200, threshold=None, min_slack=1.0, cooldown=50):
        self.window = window
        self.threshold = threshold
        self.min_slack = min_slack
        self.cooldown = cooldown
        self.x = []
        self.alarms = []
        self._last_alarm = -10**9
        self._baseline_mean = None
        self._baseline_std = None

    def update(self, t, val):
        self.x.append(val)
        if len(self.x) < self.window:
            return None
        arr = np.asarray(self.x[-self.window:])
        mu, sd = arr.mean(), arr.std() + 1e-12
        # CUSUM against running baseline estimated from first half of window
        base = np.asarray(self.x[-self.window: -self.window // 2])
        mu0, sd0 = base.mean(), base.std() + 1e-12
        stat = (mu - mu0) / sd0
        thr = self.threshold if self.threshold is not None else self.min_slack
        if abs(stat) > thr and (t - self._last_alarm) > self.cooldown:
            self.alarms.append(t)
            self._last_alarm = t
            return t
        return None


def evaluate_detection(alarms, cps, tol=300):
    """Match alarms to planted changepoints.

    An alarm matches CP c only if it fires in [c, c+tol] -- alarms BEFORE the
    CP are false alarms, not detections (negative-delay "matches" were an
    evaluation artifact that inflated recall in sim_v1).

    Returns dict(tp, fp, fn, delays[list of >=0], f1, precision, recall).
    """
    alarms = sorted(alarms)
    cps = sorted(cps)
    used = set()
    tp = 0
    delays = []
    for a in alarms:
        best, best_d = None, None
        for ci, c in enumerate(cps):
            if ci in used:
                continue
            d = a - c
            if 0 <= d <= tol and (best_d is None or d < best_d):
                best, best_d = ci, d
        if best is not None:
            used.add(best)
            tp += 1
            delays.append(best_d)
        # unmatched alarm counts as FP later
    fp = len(alarms) - tp
    fn = len(cps) - len(used)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn,
            "delays": delays,
            "mean_delay": float(np.mean(delays)) if delays else None,
            "precision": prec, "recall": rec, "f1": f1}
