"""Endpoint-level scoring from raw request spans.

The main pipeline scores one API surface per service, because CPU, memory and
error-rate metrics are only exported per service. For the cases where the
benchmark also ships raw distributed-tracing spans, this module goes a level
finer and scores each individual **endpoint** -- `/hipstershop.CheckoutService/
PlaceOrder` rather than `checkoutservice` -- directly from the requests
themselves.

Per endpoint per window, measured from the spans:

* request count            -- number of spans
* error frequency          -- share of spans with a non-OK status code
* response consistency     -- 1 / (1 + sigma/mu) over span durations
* p50 / p95 latency

The failure-risk term is inherited from the endpoint's parent service, since it
is derived from per-service resource telemetry. That is stated in the dashboard
next to the numbers rather than hidden.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (BAND_DEGRADED, BAND_RELIABLE, DATA_PROCESSED, DATA_RAW,
                     STRIDE_S, W_CONSISTENCY, W_ERROR_FREQ, W_FAILURE_RISK)
from .download import CASE_RE
from .score import band

# Span service names do not always match the metric service names.
SERVICE_ALIASES = {"frontendservice": "frontend"}
MIN_SPANS_IN_WINDOW = 10


def normalise_service(name: str, known: set[str]) -> str:
    if name in known:
        return name
    alias = SERVICE_ALIASES.get(name)
    if alias in known:
        return alias
    stripped = name.replace("service", "")
    for candidate in (stripped, stripped + "service", stripped + "-service"):
        if candidate in known:
            return candidate
    return name


def endpoint_windows(spans: pd.DataFrame, t0: int,
                     known_services: set[str]) -> pd.DataFrame:
    """Aggregate spans into the same 30 s / 15 s window grid as the metrics."""
    df = pd.DataFrame({
        "service": spans.serviceName.map(
            lambda s: normalise_service(s, known_services)),
        "endpoint": spans.operationName.astype(str),
        # Jaeger records span timings in microseconds
        "t": spans.startTimeMillis.astype("float64") / 1000.0,
        "latency": spans.duration.astype("float64") / 1e6,
        "failed": spans.statusCode.fillna(0).astype("float64").ne(0),
    })
    df = df[df.latency.notna() & df.t.notna()]
    df["block"] = ((df.t - t0) // STRIDE_S).astype(int)

    per_block = df.groupby(["service", "endpoint", "block"]).agg(
        n=("latency", "size"),
        total=("latency", "sum"),
        sq=("latency", lambda s: float(np.sum(s.to_numpy() ** 2))),
        p50=("latency", "median"),
        p95=("latency", lambda s: float(np.percentile(s, 95))),
        errors=("failed", "sum"),
    ).reset_index()

    # a window is this block plus the previous one
    g = per_block.sort_values("block").groupby(["service", "endpoint"],
                                               sort=False)
    out = per_block[["service", "endpoint", "block"]].copy()
    roll = lambda col: g[col].transform(  # noqa: E731
        lambda x: x.rolling(2, min_periods=2).sum())
    n = roll("n")
    total = roll("total")
    sq = roll("sq")
    mean = total / n
    var = (sq / n) - mean ** 2

    out["requests"] = n
    out["latency_mean"] = mean
    out["latency_std"] = np.sqrt(var.clip(lower=0))
    out["latency_p50"] = g["p50"].transform(
        lambda x: x.rolling(2, min_periods=2).mean())
    out["latency_p95"] = g["p95"].transform(
        lambda x: x.rolling(2, min_periods=2).max())
    out["error_frequency"] = roll("errors") / n
    out["response_consistency"] = 1.0 / (
        1.0 + (out.latency_std / (out.latency_mean + 1e-9)).clip(lower=0))
    return out[out.requests >= MIN_SPANS_IN_WINDOW].dropna(subset=["latency_mean"])


def score_case(case: str, scored: pd.DataFrame) -> pd.DataFrame | None:
    """Endpoint scores for one case, or None if it has no spans."""
    traces = DATA_RAW / case / "traces.parquet"
    if not traces.exists():
        return None
    svc = scored[scored.case == case]
    if svc.empty:
        return None

    t0 = int(svc.t0.iloc[0])
    spans = pd.read_parquet(
        traces, columns=["serviceName", "operationName", "startTimeMillis",
                         "duration", "statusCode"])
    ep = endpoint_windows(spans, t0, set(svc.service.unique()))
    if ep.empty:
        return None

    # inherit the failure risk of the parent service for the same window
    risk = svc[["service", "window_end", "t0", "failure_risk"]].copy()
    risk["block"] = ((risk.window_end - risk.t0) // STRIDE_S).astype(int)
    ep = ep.merge(risk[["service", "block", "window_end", "failure_risk"]],
                  on=["service", "block"], how="inner")
    if ep.empty:
        return None

    ep["case"] = case
    ep["api"] = ep.service + " · " + ep.endpoint
    ep["reliability_score"] = (
        W_FAILURE_RISK * (1 - ep.failure_risk)
        + W_ERROR_FREQ * (1 - ep.error_frequency)
        + W_CONSISTENCY * ep.response_consistency).clip(0, 1)
    ep["status"] = band(ep.reliability_score)
    return ep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored", default=str(DATA_PROCESSED / "scored.parquet"))
    p.add_argument("--out", default=str(DATA_PROCESSED / "endpoints.parquet"))
    args = p.parse_args(argv)

    cases_with_spans = sorted(
        d.name for d in DATA_RAW.iterdir()
        if d.is_dir() and CASE_RE.match(d.name) and (d / "traces.parquet").exists())
    if not cases_with_spans:
        print("no cases with spans downloaded; fetch some with:\n"
              "    python -m src.download --with-traces 6")
        return 0

    print("scoring endpoints for {} cases with spans".format(len(cases_with_spans)))
    frames = []
    for case in cases_with_spans:
        scored = pd.read_parquet(args.scored, filters=[("case", "==", case)])
        result = score_case(case, scored)
        if result is None:
            print("  {}: no usable spans".format(case))
            continue
        frames.append(result)
        print("  {}: {} endpoints, {:,} windows".format(
            case, result.api.nunique(), len(result)))

    if not frames:
        print("nothing scored")
        return 0

    ep = pd.concat(frames, ignore_index=True)
    ep.to_parquet(args.out, index=False)
    print("\nendpoint windows : {:,}".format(len(ep)))
    print("distinct endpoints: {}".format(ep.api.nunique()))
    print("endpoints ever seeing an error: {}".format(
        int((ep.groupby("api").error_frequency.max() > 0).sum())))
    print("\nlowest-scoring endpoints (mean over all windows):")
    worst = ep.groupby("api").agg(
        score=("reliability_score", "mean"),
        errors=("error_frequency", "mean"),
        consistency=("response_consistency", "mean"),
        p50=("latency_p50", "mean")).sort_values("score").head(10)
    print(worst.round(4).to_string())
    print("\nwritten ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
