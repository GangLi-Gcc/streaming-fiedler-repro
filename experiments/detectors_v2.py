"""Detector v2: self-normalizing drift detector on the lambda_2(t) trajectory.

Design rationale (from notes/2026-09-14_任务1_漂移本底.md):
  - lambda_2(t) is a NON-STATIONARY random walk with seed-dependent scale
    (sigma_inc varies 12x across seeds) -> any global threshold fails.
  - Quasi-periodic wander (period 158-750 events) pollutes mean-based
    baselines -> use MEDIANS for both baseline and recent level.
  - Detectability condition: |level shift| > z_thr * sigma_loc * sqrt(L).

Statistic (two-window median comparison):
  base   = median(lambda[t-Wb-L : t-L])      (pre-window)
  recent = median(lambda[t-L : t])           (post-window)
  sigma_loc = 1.4826 * MAD(increments over trailing window)   (robust local scale)
  z = (recent - base) / (sigma_loc * sqrt(L))

Alarm when |z| > threshold, with cooldown. All state is O(window) -- no
recomputation from scratch.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class SelfNormalizingDrift:
    def __init__(self, win_base=300, win_recent=100, mad_window=400,
                 threshold=4.0, cooldown=60):
        self.win_base = win_base
        self.win_recent = win_recent
        self.mad_window = mad_window
        self.threshold = threshold
        self.cooldown = cooldown
        self.x = deque(maxlen=win_base + win_recent + 2)
        self.alarms = []
        self.z_trace = []
        self._last_alarm = -10**9

    def update(self, t, val):
        self.x.append(val)
        need = self.win_base + self.win_recent
        if len(self.x) < need + 1:
            return None
        arr = np.asarray(self.x)
        inc = np.diff(arr[-(self.mad_window + 1):])
        med_inc = np.median(inc)
        sigma_loc = 1.4826 * np.median(np.abs(inc - med_inc)) + 1e-12
        base = np.median(arr[: self.win_base])
        recent = np.median(arr[self.win_base:])
        z = (recent - base) / (sigma_loc * np.sqrt(self.win_recent))
        self.z_trace.append(z)
        if abs(z) > self.threshold and (t - self._last_alarm) > self.cooldown:
            self.alarms.append(t)
            self._last_alarm = t
            return t
        return None


class PersistenceDrift:
    """Detector v3: sustained-deviation statistic with persistence requirement.

    Motivation (task #2 negative result): lambda_2(t) wander has heavy-tailed
    bursts that defeat fixed-window MAD normalization; a changepoint instead
    produces a SUSTAINED level shift. v3 therefore requires |z| to stay above
    a (lower) threshold for K consecutive events with consistent sign before
    alarming. Bursts die out; shifts persist.
    """
    def __init__(self, win_base=300, win_recent=100, mad_window=400,
                 threshold=3.0, persist_k=25, cooldown=120):
        self.win_base = win_base
        self.win_recent = win_recent
        self.mad_window = mad_window
        self.threshold = threshold
        self.persist_k = persist_k
        self.cooldown = cooldown
        self.x = deque(maxlen=win_base + win_recent + 2)
        self.alarms = []
        self.z_trace = []
        self._last_alarm = -10**9
        self._streak = 0
        self._streak_sign = 0

    def update(self, t, val):
        self.x.append(val)
        need = self.win_base + self.win_recent
        if len(self.x) < need + 1:
            return None
        arr = np.asarray(self.x)
        inc = np.diff(arr[-(self.mad_window + 1):])
        med_inc = np.median(inc)
        sigma_loc = 1.4826 * np.median(np.abs(inc - med_inc)) + 1e-12
        base = np.median(arr[: self.win_base])
        recent = np.median(arr[self.win_base:])
        z = (recent - base) / (sigma_loc * np.sqrt(self.win_recent))
        self.z_trace.append(z)
        if abs(z) > self.threshold:
            s = 1 if z > 0 else -1
            if s == self._streak_sign:
                self._streak += 1
            else:
                self._streak_sign = s
                self._streak = 1
        else:
            self._streak = 0
            self._streak_sign = 0
        if (self._streak >= self.persist_k
                and (t - self._last_alarm) > self.cooldown):
            self.alarms.append(t)
            self._last_alarm = t
            self._streak = 0  # require re-accumulation after an alarm
            return t
        return None


def alarms_per_1000(alarms, events):
    return 1000.0 * len(alarms) / max(events, 1)
