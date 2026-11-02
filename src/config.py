"""Central configuration for the API reliability-scoring pipeline.

Every constant that affects results lives here so the thesis can quote exact
settings and so experiments are reproducible.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data source
# RCAEval: real telemetry recorded from three running microservice systems.
# Paper: "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems
# with Telemetry Data" (WWW'25 companion / FSE'26).
HF_DATASET = "phamquiluan/RCAEval"
HF_API = f"https://huggingface.co/api/datasets/{HF_DATASET}"
HF_FILE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

SYSTEM_NAMES = {
    "ob": "Online Boutique",
    "ss": "Sock Shop",
    "tt": "Train Ticket",
}
SUITE_NAMES = {
    "re1": "RE1 (resource + network faults)",
    "re2": "RE2 (multi-source, 6 fault types)",
    "re3": "RE3 (code-level faults)",
}

# ------------------------------------------------------------------ windowing
# Telemetry is sampled at 1 Hz. Windows are the unit of scoring: one reliability
# score per (API, window).
WINDOW_S = 30            # length of an observation window, seconds
STRIDE_S = 15            # hop between consecutive windows, seconds
BASELINE_S = 90          # trailing span used as the causal "normal" baseline
MIN_SAMPLES_IN_WINDOW = 20   # of the 30 expected 1 Hz samples
MIN_BASELINE_COVERAGE = 0.66 # fraction of the baseline span that must be present
MIN_CASE_DURATION_S = 300    # a recording shorter than this cannot show both states

# Label horizon: a window is positive if the API is in a failing state during
# the next HORIZON_S seconds. Features never look past the window end, so there
# is no leakage from the future into the feature vector.
HORIZON_S = 30

# ------------------------------------------------------- reliability scoring
# Weights are taken unchanged from the project proposal.
W_FAILURE_RISK = 0.40
W_ERROR_FREQ = 0.35
W_CONSISTENCY = 0.25

# Status bands. Chosen so the worked example in the proposal (0.73) stays
# "Reliable", as presented.
BAND_RELIABLE = 0.70
BAND_DEGRADED = 0.50

# ------------------------------------------------------------------- training
RANDOM_STATE = 42
N_ESTIMATORS = 400
N_SPLITS = 5             # grouped cross-validation folds (grouped by case)
