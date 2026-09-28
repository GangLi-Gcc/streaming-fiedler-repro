# Streaming Fiedler Tracking — Reproducibility Archive

Code, raw data, and result records for streaming Fiedler-vector (second
Laplacian eigenpair) maintenance and spectral change-point detection on dynamic
graphs.

## Contents

- `rebuild/` — clean re-implementation (authoritative). `core.py` holds the
  warm-started tracker and the detection channels; `run_*.py`, `analyze_*.py`
  and `verify_*.py` regenerate and audit the records in `rebuild/results/`.
- `experiments/` — original evaluation codebase, including the LAD and SCPD
  baselines, the scale experiment, and the gap/MDR rigs.
- `data/` — raw input graphs (SNAP CollegeMsg, email-Eu-core temporal).
- `runs/` — legacy result records (JSON plus NumPy trajectory caches).

## Key result records

- `rebuild/results/detection_email-eu.json`, `detection_college-msg.json`,
  `detection_union_email-eu.json`, `scpd_email-eu.json` — detection metrics.
- `rebuild/results/mdr_configs.json`, `mdr_kband.json`, `gate_audit.json`,
  `gate_audit_crossings.json`, `detectability.json` — minimum-trackable-rotation
  rig and gate audit.
- `rebuild/results/grest_compare.json` — G-REST comparison.
- `runs/lad_baseline-college-msg.json`, `runs/scale_experiment.json` — legacy
  per-run records.

## Requirements

Python 3.13, NumPy, SciPy, networkx. See the import headers of each script.
