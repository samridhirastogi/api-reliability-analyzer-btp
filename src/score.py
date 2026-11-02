"""Combine the model's failure risk with observed runtime metrics into a
single reliability score per API per window.

    Reliability = 0.40 * (1 - FailureRisk)
                + 0.35 * (1 - ErrorFrequency)
                + 0.25 * ResponseConsistency

The three components and their weights are taken from the project proposal.
One definition had to change to make the sum well-formed:

*Response consistency.* The proposal defines it as the inverse of the standard
deviation of latency, 1 / sigma. That is unbounded (a perfectly steady API
gives 1/0) and unit-dependent (milliseconds and seconds give different scores
for identical behaviour), so it cannot be mixed with two terms that live in
[0, 1]. This implementation keeps the intent -- less variable latency means a
more consistent API -- using the coefficient of variation:

    CV          = sigma(latency) / mean(latency)
    Consistency = 1 / (1 + CV)

which is dimensionless, lands in (0, 1], equals 1 for perfectly steady latency
and falls towards 0 as jitter grows. Section "Scoring" of the report discusses
the change.

*Missing components.* Not every service in the benchmark exposes an error-rate
metric. Rather than silently substituting zero -- which would reward an API for
being unmeasured -- the missing term is dropped and the remaining weights are
renormalised to sum to one.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from .config import (BAND_DEGRADED, BAND_RELIABLE, DATA_PROCESSED, MODELS,
                     W_CONSISTENCY, W_ERROR_FREQ, W_FAILURE_RISK)

STATUS_ORDER = ["Failure-prone", "Degraded", "Reliable"]


def load_bundle(path=None):
    return joblib.load(path or (MODELS / "failure_risk.joblib"))


def band(score):
    return pd.cut(score, bins=[-np.inf, BAND_DEGRADED, BAND_RELIABLE, np.inf],
                  labels=STATUS_ORDER).astype(object)


def score_frame(win: pd.DataFrame, bundle) -> pd.DataFrame:
    """Add failure risk, the three components and the reliability score."""
    out = win.copy()
    out["failure_risk"] = bundle["model"].predict_proba(
        out[bundle["features"]])[:, 1]

    # Error frequency: failed requests / total requests, as measured.
    out["error_frequency"] = out["err_mean"].clip(0.0, 1.0)
    # Response consistency: bounded, scale-free stability of response time.
    out["response_consistency"] = 1.0 / (1.0 + out["lat50_cv"].clip(lower=0))

    risk_term = W_FAILURE_RISK * (1.0 - out.failure_risk)
    err_term = W_ERROR_FREQ * (1.0 - out.error_frequency)
    cons_term = W_CONSISTENCY * out.response_consistency

    has_err = out.error_frequency.notna()
    weight = np.where(has_err, W_FAILURE_RISK + W_ERROR_FREQ + W_CONSISTENCY,
                      W_FAILURE_RISK + W_CONSISTENCY)
    total = risk_term + cons_term + err_term.fillna(0.0)

    out["reliability_score"] = (total / weight).clip(0.0, 1.0)
    out["error_measured"] = has_err
    out["status"] = band(out.reliability_score)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_PROCESSED / "labelled.parquet"))
    p.add_argument("--out", default=str(DATA_PROCESSED / "scored.parquet"))
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)

    win = pd.read_parquet(args.data)
    win = win[~win.is_infra].reset_index(drop=True)
    scored = score_frame(win, load_bundle(args.model))
    scored.to_parquet(args.out, index=False)

    print("scored windows      : {:,}".format(len(scored)))
    print("error metric present: {:.1%}".format(scored.error_measured.mean()))

    # Windows of non-injected services during an active fault are ambiguous
    # (the fault propagates to callers), so the summary below - like the
    # report - is computed on the unambiguous ones only.
    scored = scored[scored.usable]
    print("unambiguous windows : {:,}".format(len(scored)))
    print("\nscore by ground truth:")
    print(scored.groupby(scored.label.map({0: "healthy", 1: "faulty"}))
          .reliability_score.describe()[["count", "mean", "std", "50%"]]
          .to_string())
    print("\nstatus vs ground truth (row = true state):")
    print(pd.crosstab(scored.label.map({0: "healthy", 1: "faulty"}),
                      scored.status, normalize="index")
          .reindex(columns=STATUS_ORDER).round(3).to_string())
    print("\nmean component values:")
    comp = scored.groupby(scored.label.map({0: "healthy", 1: "faulty"}))[
        ["failure_risk", "error_frequency", "response_consistency",
         "reliability_score"]].mean().round(4)
    print(comp.to_string())
    print("\nwritten ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
