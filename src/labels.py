"""Attach ground-truth failure labels to windows.

The label comes from the benchmark's recorded fault-injection timestamp, not
from the telemetry itself, so it is independent of the features the model sees.

A window (service *s*, ending at *e*) is **positive** when *s* is the service a
real fault was injected into and that fault is active at some point during the
next ``HORIZON_S`` seconds.

Non-target services observed while a fault is active are **masked out** rather
than labelled negative. A fault in one service propagates to its callers, so
those windows are genuinely ambiguous: the benchmark records which service was
injected, not which services ended up degraded. Calling them negative would
teach the model to ignore real degradation. ``--propagation-negative`` keeps
them as negatives instead, for the sensitivity study in the report.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import DATA_PROCESSED, HORIZON_S


def attach(win: pd.DataFrame, cases: pd.DataFrame,
           propagation_negative: bool = False) -> pd.DataFrame:
    meta = cases.set_index("case")
    win = win.copy()
    win["target_service"] = win.case.map(meta.target_service)
    win["fault"] = win.case.map(meta.fault)
    win["inject_time"] = win.case.map(meta.inject_time)

    win["is_target"] = win.service == win.target_service
    # seconds between the window end and the moment the fault was injected
    win["lead_s"] = win.window_end - win.inject_time
    # the fault runs from injection to the end of the recording
    fault_in_horizon = (win.lead_s + HORIZON_S) > 0

    win["label"] = (win.is_target & fault_in_horizon).astype(int)
    win["phase"] = pd.cut(
        win.lead_s,
        bins=[-10**9, -HORIZON_S, 0, 10**9],
        labels=["pre_fault", "onset", "during_fault"],
    )

    if propagation_negative:
        win["usable"] = True
    else:
        ambiguous = (~win.is_target) & fault_in_horizon
        win["usable"] = ~ambiguous

    return win


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", default=str(DATA_PROCESSED / "windows.parquet"))
    p.add_argument("--cases", default=str(DATA_PROCESSED / "cases.parquet"))
    p.add_argument("--out", default=str(DATA_PROCESSED / "labelled.parquet"))
    p.add_argument("--propagation-negative", action="store_true")
    args = p.parse_args(argv)

    win = pd.read_parquet(args.windows)
    cases = pd.read_parquet(args.cases)
    out = attach(win, cases, args.propagation_negative)
    out.to_parquet(args.out, index=False)

    usable = out[out.usable]
    print(f"windows total    : {len(out):,}")
    print(f"usable for model : {len(usable):,} "
          f"({len(usable) / len(out):.1%})")
    print(f"positives        : {int(usable.label.sum()):,} "
          f"({usable.label.mean():.2%})")
    print("positives by fault type:")
    print(usable[usable.label == 1].fault.value_counts().to_string())
    print("usable windows by phase:")
    print(usable.phase.value_counts().to_string())
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
