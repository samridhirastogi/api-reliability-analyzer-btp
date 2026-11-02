"""Measure how far a fault actually spreads beyond the service it hit.

This exists because the labelling policy depends on an empirical question that
is usually assumed rather than measured.

When a fault is injected into one service, the services calling it may also
slow down. The benchmark records only *which service was injected*, not which
ones ended up degraded, so the standard defence is to mask every non-injected
service while a fault is active -- treating its windows as neither healthy nor
failing. That defence is only proportionate if propagation is common.

This module checks. For every service in every recording it compares mean p50
latency during the fault against the same service's latency before the
injection, and reports what share of *non-injected* services actually moved.

The measurement is used to justify a labelling decision, not to produce labels:
nothing here feeds the model. Deriving labels from latency would be circular,
since latency is also an input feature.
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import viz  # noqa: E402
from .config import DATA_PROCESSED, FIGURES, REPORTS, SYSTEM_NAMES  # noqa: E402

# A service counts as degraded if its latency rose by at least this much.
DEGRADED_RATIO = 1.2
# Comparison spans, in seconds relative to the injection.
PRE_END = -30
POST_START, POST_END = 30, 330


def measure(labelled: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, service): latency before vs during the fault."""
    d = labelled[~labelled.is_infra]
    pre = (d[d.lead_s < PRE_END]
           .groupby(["case", "system", "fault", "service", "is_target"])
           .lat50_mean.mean().rename("pre"))
    post = (d[(d.lead_s >= POST_START) & (d.lead_s <= POST_END)]
            .groupby(["case", "system", "fault", "service", "is_target"])
            .lat50_mean.mean().rename("post"))
    r = pd.concat([pre, post], axis=1).dropna().reset_index()
    r = r[r.pre > 0].copy()
    r["ratio"] = r.post / r.pre
    r["degraded"] = r.ratio > DEGRADED_RATIO
    return r


def window_accounting(labelled: pd.DataFrame, measured: pd.DataFrame) -> dict:
    """How many masked windows are genuinely ambiguous, and how many are not."""
    d = labelled[~labelled.is_infra]
    affected = set(map(tuple, measured.loc[measured.degraded,
                                           ["case", "service"]].to_numpy()))
    masked = d[~d.usable]
    is_affected = np.fromiter(
        ((c, s) in affected for c, s in zip(masked.case, masked.service)),
        dtype=bool, count=len(masked))
    usable_now = int(d.usable.sum())
    recoverable = int((~is_affected).sum())
    return {
        "total_windows": int(len(d)),
        "masked_windows": int(len(masked)),
        "masked_share": float(len(masked) / len(d)),
        "masked_and_degraded": int(is_affected.sum()),
        "masked_but_clean": recoverable,
        "share_of_masked_that_degraded": float(is_affected.mean()),
        "usable_now": usable_now,
        "usable_if_recovered": usable_now + recoverable,
        "data_gain": float(recoverable / usable_now),
    }


def figure(measured: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    bins = np.linspace(0.5, 3.0, 51)
    tgt = measured[measured.is_target].ratio.clip(0.5, 3.0)
    oth = measured[~measured.is_target].ratio.clip(0.5, 3.0)
    ax1.hist(oth, bins=bins, color=viz.SERIES[0], alpha=0.8, density=True,
             label="other services")
    ax1.hist(tgt, bins=bins, color=viz.SERIES[1], alpha=0.8, density=True,
             label="the injected service")
    ax1.axvline(DEGRADED_RATIO, color=viz.INK_MUTED, ls="--", lw=1.2)
    ax1.text(DEGRADED_RATIO, ax1.get_ylim()[1] * 0.93,
             "  +20%", fontsize=9, color=viz.INK_MUTED)
    ax1.set_xlabel("latency during fault / latency before")
    ax1.set_ylabel("density")
    ax1.set_title("Most services never feel the fault")
    ax1.legend()

    by_sys = (measured[~measured.is_target]
              .groupby("system").degraded.agg(["mean", "size"]))
    by_sys.index = [SYSTEM_NAMES.get(s, s) for s in by_sys.index]
    by_sys = by_sys.sort_values("mean")
    ax2.barh(by_sys.index, by_sys["mean"] * 100, color=viz.SERIES[0],
             height=0.55)
    for y, (v, n) in enumerate(zip(by_sys["mean"] * 100, by_sys["size"])):
        ax2.text(v + 0.5, y, "{:.1f}%  (n={})".format(v, int(n)), va="center",
                 fontsize=9, color=viz.INK_SECONDARY)
    ax2.set_xlim(0, max(by_sys["mean"] * 100) * 1.45)
    ax2.set_xlabel("share of non-injected services degrading >20%")
    ax2.set_title("Propagation rate by system")
    ax2.grid(axis="y", visible=False)

    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    path = FIGURES / "propagation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  figure ->", path.name)


def write_markdown(measured: pd.DataFrame, acct: dict):
    tgt = measured[measured.is_target]
    oth = measured[~measured.is_target]
    thresholds = (1.1, 1.2, 1.5, 2.0)

    lines = [
        "# How far does a fault actually spread?",
        "",
        "Generated by `python -m src.propagation`. For every service in every "
        "recording, mean p50 latency over the {}-{}s *after* the injection is "
        "compared with the same service's latency before it.".format(
            POST_START, POST_END),
        "",
        "## The headline",
        "",
        "| | Injected service | Every other service |",
        "|---|---|---|",
        "| Service-case pairs | {:,} | {:,} |".format(len(tgt), len(oth)),
        "| Median latency ratio | {:.3f} | **{:.3f}** |".format(
            tgt.ratio.median(), oth.ratio.median()),
    ]
    for t in thresholds:
        lines.append("| Share degrading >{:.0f}% | {:.1%} | {:.1%} |".format(
            (t - 1) * 100, (tgt.ratio > t).mean(), (oth.ratio > t).mean()))
    lines += [
        "",
        "**Propagation is real but narrow.** {:.1%} of non-injected services "
        "degrade by more than 20% while a fault is active; the median one does "
        "not move at all (ratio {:.3f}). The injected service, by contrast, "
        "degrades {:.1%} of the time.".format(
            oth.degraded.mean(), oth.ratio.median(),
            (tgt.ratio > DEGRADED_RATIO).mean()),
        "",
        "## Why this matters for labelling",
        "",
        "The masking policy treats *every* non-injected service as ambiguous "
        "while a fault is active. The measurement above says that is far more "
        "conservative than the data warrants:",
        "",
        "| | Windows |",
        "|---|---|",
        "| Total (excluding infrastructure) | {:,} |".format(acct["total_windows"]),
        "| Masked by the policy | {:,} ({:.1%}) |".format(
            acct["masked_windows"], acct["masked_share"]),
        "| ...belonging to a service that actually degraded | {:,} ({:.1%} of masked) |".format(
            acct["masked_and_degraded"], acct["share_of_masked_that_degraded"]),
        "| ...clean negatives being discarded | **{:,}** |".format(
            acct["masked_but_clean"]),
        "",
        "Masking therefore discards {:,} windows to guard against an "
        "ambiguity that affects {:,} of them -- roughly {:.0f}x more data than "
        "the problem justifies. Labelling those services negative instead "
        "takes the training set from {:,} to {:,} windows (+{:.0%}).".format(
            acct["masked_windows"], acct["masked_and_degraded"],
            acct["masked_windows"] / max(acct["masked_and_degraded"], 1),
            acct["usable_now"], acct["usable_if_recovered"], acct["data_gain"]),
        "",
        "That is the evidence behind this project reporting the "
        "propagation-negative labelling as its primary result. The masked "
        "variant is kept as a comparison, and the gap between them -- see "
        "`BASELINES.md` -- bounds what the choice is worth.",
        "",
        "## By system",
        "",
        "| System | Non-injected services | Degrading >20% |",
        "|---|---|---|",
    ]
    by_sys = oth.groupby("system").degraded.agg(["mean", "size"])
    for s, row in by_sys.sort_values("mean", ascending=False).iterrows():
        lines.append("| {} | {:,} | {:.1%} |".format(
            SYSTEM_NAMES.get(s, s), int(row["size"]), row["mean"]))

    lines += [
        "",
        "Online Boutique propagates most, which fits its shape: a single "
        "`frontend` fans out to every other service, so a fault anywhere has a "
        "short path to it. Train Ticket has many more services and longer, "
        "narrower call chains, so a given fault reaches proportionally fewer "
        "of them.",
        "",
        "## By fault type",
        "",
        "| Fault | Non-injected services degrading >20% |",
        "|---|---|",
    ]
    by_fault = oth.groupby("fault").degraded.mean().sort_values(ascending=False)
    for f, v in by_fault.items():
        lines.append("| {} | {:.1%} |".format(f, v))

    lines += [
        "",
        "## What this measurement is not",
        "",
        "A latency ratio is evidence that a service degraded, not proof that "
        "the injected fault caused it: load varies on its own, and some of "
        "these services are noisy. The figure is used to size the labelling "
        "problem and to justify a policy, never to label a window -- latency "
        "is an input feature, so labelling on it would be circular.",
        "",
        "Establishing actual causation needs the call graph, which the "
        "benchmark ships for {} of the recordings. A first look at six of "
        "them was inconclusive: the services that transitively call the "
        "injected one showed no degradation, while a few that cannot reach it "
        "did, and only nine service-level edges were recovered. That is a lead "
        "worth following, not a result.".format(240),
        "",
    ]

    path = REPORTS / "PROPAGATION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("  results ->", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_PROCESSED / "labelled.parquet"),
                   help="the MASKED labelling, whose policy is being assessed")
    args = p.parse_args(argv)

    cols = ["case", "system", "fault", "service", "is_target", "lead_s",
            "lat50_mean", "is_infra", "usable"]
    labelled = pd.read_parquet(args.data, columns=cols)

    measured = measure(labelled)
    acct = window_accounting(labelled, measured)
    measured.to_parquet(DATA_PROCESSED / "propagation.parquet", index=False)

    oth = measured[~measured.is_target]
    print("service-case pairs      : {:,}".format(len(measured)))
    print("non-injected degrading  : {:.1%}".format(oth.degraded.mean()))
    print("median ratio, others    : {:.3f}".format(oth.ratio.median()))
    print("masked windows          : {:,} ({:.1%} of all)".format(
        acct["masked_windows"], acct["masked_share"]))
    print("  genuinely ambiguous   : {:,} ({:.1%} of masked)".format(
        acct["masked_and_degraded"], acct["share_of_masked_that_degraded"]))
    print("  recoverable negatives : {:,}".format(acct["masked_but_clean"]))
    print("training data gain      : +{:.0%}".format(acct["data_gain"]))

    figure(measured)
    write_markdown(measured, acct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
