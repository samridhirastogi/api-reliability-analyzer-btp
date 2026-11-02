"""Produce the figures and results tables for the thesis.

Everything here reads artefacts the pipeline already wrote; nothing is
recomputed from scratch, so the numbers in the report and the numbers in the
dashboard cannot drift apart.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_recall_curve  # noqa: E402

from . import viz  # noqa: E402
from .config import (BAND_DEGRADED, BAND_RELIABLE, DATA_PROCESSED, FIGURES,
                     MODELS, REPORTS, SYSTEM_NAMES)  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": viz.SURFACE,
    "axes.facecolor": viz.SURFACE,
    "axes.edgecolor": viz.BASELINE,
    "axes.labelcolor": viz.INK_SECONDARY,
    "axes.titlecolor": viz.INK_PRIMARY,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": viz.GRIDLINE,
    "grid.linewidth": 0.8,
    "xtick.color": viz.INK_MUTED,
    "ytick.color": viz.INK_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "lines.linewidth": 2,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _finish(fig, name):
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print("  figure ->", path.name)
    return path


# ------------------------------------------------------------------ figures
def fig_dataset(cases: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    pivot = (cases.assign(system_name=cases.system.map(SYSTEM_NAMES))
             .pivot_table(index="suite", columns="system_name",
                          values="case", aggfunc="count").fillna(0))
    bottom = np.zeros(len(pivot))
    for i, col in enumerate(pivot.columns):
        ax1.bar(pivot.index, pivot[col], bottom=bottom, label=col,
                color=viz.SERIES[i], width=0.6, edgecolor=viz.SURFACE, lw=2)
        bottom += pivot[col].to_numpy()
    for x, total in zip(pivot.index, bottom):
        ax1.text(x, total + 4, int(total), ha="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax1.set_title("Recorded failure cases per suite")
    ax1.set_ylabel("cases")
    ax1.legend(loc="upper right")
    ax1.grid(axis="x", visible=False)

    counts = cases.fault.value_counts()
    ax2.barh(counts.index[::-1], counts.values[::-1], color=viz.SERIES[0],
             height=0.6)
    for y, v in enumerate(counts.values[::-1]):
        ax2.text(v + 2, y, str(v), va="center", fontsize=9,
                 color=viz.INK_SECONDARY)
    ax2.set_title("Injected fault types")
    ax2.set_xlabel("cases")
    ax2.grid(axis="y", visible=False)
    return _finish(fig, "dataset_composition.png")


def fig_example_case(scored: pd.DataFrame, cases: pd.DataFrame, case_id: str):
    meta = cases[cases.case == case_id].iloc[0]
    d = scored[scored.case == case_id].copy()
    d["t"] = d.window_end - meta.inject_time
    tgt = d[d.service == meta.target_service].sort_values("t")
    others = d[d.service != meta.target_service]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for svc, g in others.groupby("service"):
        g = g.sort_values("t")
        axes[0].plot(g.t, g.lat50_mean, color=viz.BASELINE, lw=1, zorder=1)
        axes[2].plot(g.t, g.reliability_score, color=viz.BASELINE, lw=1, zorder=1)
    axes[0].plot(tgt.t, tgt.lat50_mean, color=viz.SERIES[0], zorder=3,
                 label=meta.target_service)
    axes[0].set_ylabel("p50 latency (s)")
    axes[0].set_title("What was measured, and what the pipeline reported — "
                      "{}".format(case_id))
    axes[0].legend(loc="upper left")

    axes[1].plot(tgt.t, tgt.failure_risk, color=viz.SERIES[1])
    axes[1].set_ylabel("failure risk")
    axes[1].set_ylim(0, 1)

    axes[2].plot(tgt.t, tgt.reliability_score, color=viz.SERIES[0], zorder=3)
    axes[2].axhspan(0, BAND_DEGRADED, color=viz.STATUS_COLOR["Failure-prone"],
                    alpha=0.07, zorder=0)
    axes[2].axhspan(BAND_DEGRADED, BAND_RELIABLE,
                    color=viz.STATUS_COLOR["Degraded"], alpha=0.07, zorder=0)
    axes[2].axhspan(BAND_RELIABLE, 1, color=viz.STATUS_COLOR["Reliable"],
                    alpha=0.07, zorder=0)
    axes[2].set_ylabel("reliability score")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("seconds relative to fault injection")

    for ax in axes:
        ax.axvline(0, color=viz.STATUS_COLOR["Failure-prone"], ls="--", lw=1.5)
    axes[0].text(0, axes[0].get_ylim()[1], " fault injected", va="top",
                 fontsize=9, color=viz.STATUS_COLOR["Failure-prone"])
    axes[2].text(axes[2].get_xlim()[0], 0.02, " grey lines: the other APIs",
                 fontsize=9, color=viz.INK_MUTED)
    return _finish(fig, "example_case.png")


def fig_model(te: pd.DataFrame, report: dict):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    prec, rec, _ = precision_recall_curve(te.label, te.prob)
    axes[0].plot(rec, prec, color=viz.SERIES[0])
    base = te.label.mean()
    axes[0].axhline(base, color=viz.BASELINE, ls="--", lw=1.2)
    axes[0].text(0.02, base + 0.02, "always-positive baseline ({:.2f})".format(base),
                 fontsize=8, color=viz.INK_MUTED)
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision-recall, held-out cases\nPR-AUC = {:.3f}".format(
        report["holdout"]["pr_auc"]))

    for i, (col, name) in enumerate([("prob_uncalibrated", "raw forest votes"),
                                     ("prob", "after calibration")]):
        if col not in te:
            continue
        x, y = calibration_curve(te.label, te[col], n_bins=10, strategy="quantile")
        axes[1].plot(y, x, marker="o", ms=5, color=viz.SERIES[i], label=name)
    axes[1].plot([0, 1], [0, 1], color=viz.BASELINE, ls="--", lw=1.2)
    axes[1].set_xlabel("predicted failure risk")
    axes[1].set_ylabel("observed failure rate")
    axes[1].set_title("Calibration\nBrier = {:.4f}".format(
        report["holdout"]["brier"]))
    axes[1].legend(loc="upper left")

    healthy = te[te.label == 0]
    faulty = te[te.label == 1]
    bins = np.linspace(0, 1, 41)
    axes[2].hist(healthy.prob, bins=bins, color=viz.SERIES[0], alpha=0.75,
                 label="healthy windows", density=True)
    axes[2].hist(faulty.prob, bins=bins, color=viz.SERIES[1], alpha=0.75,
                 label="failing windows", density=True)
    axes[2].axvline(report["holdout"]["threshold"], color=viz.INK_MUTED, ls="--",
                    lw=1.2)
    axes[2].set_xlabel("predicted failure risk")
    axes[2].set_ylabel("density")
    axes[2].set_title("Risk separates the two states")
    axes[2].legend()
    return _finish(fig, "model_performance.png")


def fig_scores(scored: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 1, 41)
    healthy = scored[scored.label == 0].reliability_score
    faulty = scored[scored.label == 1].reliability_score
    ax1.hist(healthy, bins=bins, color=viz.SERIES[0], alpha=0.75, density=True,
             label="healthy windows")
    ax1.hist(faulty, bins=bins, color=viz.SERIES[1], alpha=0.75, density=True,
             label="failing windows")
    for v in (BAND_DEGRADED, BAND_RELIABLE):
        ax1.axvline(v, color=viz.BASELINE, ls="--", lw=1.2)
    ax1.set_xlabel("reliability score")
    ax1.set_ylabel("density")
    ax1.set_title("Reliability score by true state")
    ax1.legend(loc="upper left")

    ct = pd.crosstab(scored.label.map({0: "healthy", 1: "failing"}),
                     scored.status, normalize="index")
    ct = ct.reindex(columns=viz.STATUS_ORDER).fillna(0)
    left = np.zeros(len(ct))
    for status in viz.STATUS_ORDER:
        ax2.barh(ct.index, ct[status], left=left, height=0.5,
                 color=viz.STATUS_COLOR[status], label=viz.status_label(status),
                 edgecolor=viz.SURFACE, lw=2)
        for y, (l, v) in enumerate(zip(left, ct[status])):
            if v > 0.06:
                ax2.text(l + v / 2, y, "{:.0%}".format(v), ha="center",
                         va="center", fontsize=9, color="white")
        left += ct[status].to_numpy()
    ax2.set_xlabel("share of windows")
    ax2.set_xlim(0, 1)
    ax2.set_title("Status assigned, by true state")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    ax2.grid(axis="y", visible=False)
    return _finish(fig, "score_distribution.png")


def fig_generalisation(report: dict):
    los = report.get("leave_one_system_out", {})
    lof = report.get("leave_one_fault_out", {})
    if not los and not lof:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, data, title in (
            (axes[0], los, "Held-out system (never trained on)"),
            (axes[1], lof, "Held-out fault type (never trained on)")):
        if not data:
            ax.axis("off")
            continue
        names = [SYSTEM_NAMES.get(k, k) for k in data]
        vals = [v["pr_auc"] for v in data.values()]
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order],
                color=viz.SERIES[0], height=0.55)
        for y, i in enumerate(order):
            ax.text(vals[i] + 0.01, y, "{:.3f}".format(vals[i]), va="center",
                    fontsize=9, color=viz.INK_SECONDARY)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("PR-AUC")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    return _finish(fig, "generalisation.png")


def fig_importance():
    path = REPORTS / "feature_importance.csv"
    if not path.exists():
        return None
    imp = pd.read_csv(path).head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.feature, imp.impurity, color=viz.SERIES[0], height=0.6)
    ax.set_xlabel("mean decrease in impurity")
    ax.set_title("Which measurements the model relies on")
    ax.grid(axis="y", visible=False)
    return _finish(fig, "feature_importance.png")


def fig_detection(te: pd.DataFrame, threshold: float):
    delays = []
    for _c, g in te[te.is_target].groupby("case"):
        after = g[g.lead_s >= 0].sort_values("lead_s")
        hit = after[after.prob >= threshold]
        if len(hit):
            delays.append(hit.lead_s.iloc[0])
    if not delays:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(delays, bins=np.arange(0, max(delays) + 30, 15),
            color=viz.SERIES[0], edgecolor=viz.SURFACE, lw=1.5)
    med = float(np.median(delays))
    ax.axvline(med, color=viz.SERIES[1], lw=2)
    ax.text(med, ax.get_ylim()[1] * 0.92, "  median {:.0f}s".format(med),
            color=viz.SERIES[1], fontsize=9)
    ax.set_xlabel("seconds between the real fault and the first flag")
    ax.set_ylabel("held-out cases")
    ax.set_title("How quickly the failing API is flagged")
    ax.grid(axis="x", visible=False)
    return _finish(fig, "detection_delay.png")


# ------------------------------------------------------------------- tables
def _calibration_sentence(report: dict, te: pd.DataFrame | None) -> str:
    """Report the calibration gain on held-out data, not on the split the
    calibrator was fitted to.

    ``metrics.json`` records the before/after Brier on the calibration split
    itself, which is in-sample for the isotonic fit and therefore flattering.
    Where the held-out predictions are available, both numbers are recomputed
    there instead.
    """
    if te is None or "prob_uncalibrated" not in te:
        return ("Calibration changed the Brier score from {:.4f} to {:.4f} on "
                "the calibration split. That split is in-sample for the "
                "isotonic fit, so treat it as an upper bound on the gain."
                .format(report["brier_before_calibration"],
                        report["brier_after_calibration"]))
    before = brier_score_loss(te.label, te.prob_uncalibrated)
    after = brier_score_loss(te.label, te.prob)
    direction = "improves" if after < before else "slightly worsens"
    return (
        "Calibration {} the Brier score on **held-out cases**, from {:.4f} "
        "(raw forest vote share) to {:.4f} (isotonic). The equivalent figures "
        "on the calibration split itself are {:.4f} and {:.4f}, but that split "
        "is in-sample for the isotonic fit, so the held-out pair is the honest "
        "one. The point of calibrating is less the Brier score than the fact "
        "that the reliability score multiplies this value by 0.40 and so needs "
        "it to mean an actual probability.".format(
            direction, before, after, report["brier_before_calibration"],
            report["brier_after_calibration"]))


def write_results(report: dict, scored: pd.DataFrame, cases: pd.DataFrame,
                  te: pd.DataFrame | None = None):
    h, d = report["holdout"], report["detection"]
    cv = report["cross_validation"]
    lines = [
        "# Results",
        "",
        "Generated by `python -m src.report`. Every measurement below was "
        "recorded from a running system; the faults were deliberately injected "
        "in a controlled testbed, which is what makes the labels exact.",
        "",
        "## Corpus",
        "",
        "| | |",
        "|---|---|",
        "| Recorded failure cases used | {} |".format(len(cases)),
        "| Systems | {} |".format(", ".join(sorted(
            cases.system.map(SYSTEM_NAMES).dropna().unique()))),
        "| Fault types | {} |".format(", ".join(sorted(cases.fault.unique()))),
        "| Scored windows | {:,} |".format(len(scored)),
        "| Distinct APIs | {} |".format(scored.service.nunique()),
        "| Window / stride / horizon | {}s / {}s / {}s |".format(
            report["dataset"]["window_s"], report["dataset"]["stride_s"],
            report["dataset"]["horizon_s"]),
        "| Positive (failing) windows | {:.2%} |".format(
            report["dataset"]["positive_rate"]),
        "",
        "## Failure-risk model",
        "",
        "Held-out **cases** (not held-out windows): the model never saw any "
        "window from these recordings.",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| PR-AUC | {:.3f} |".format(h["pr_auc"]),
        "| PR-AUC of a trivial baseline | {:.3f} |".format(h["positive_rate"]),
        "| ROC-AUC | {:.3f} |".format(h["roc_auc"]),
        "| Precision | {:.3f} |".format(h["precision"]),
        "| Recall | {:.3f} |".format(h["recall"]),
        "| F1 | {:.3f} |".format(h["f1"]),
        "| Brier score | {:.4f} |".format(h["brier"]),
        "| Decision threshold | {:.3f} |".format(h["threshold"]),
        "",
        "5-fold cross-validation grouped by case: PR-AUC "
        "{:.3f} +/- {:.3f}, F1 {:.3f}.".format(
            cv["pr_auc_mean"], cv["pr_auc_std"], cv["f1_mean"]),
        "",
        _calibration_sentence(report, te),
        "",
        "## Catching the real fault",
        "",
        "| | |",
        "|---|---|",
        "| Held-out cases where the injected API was flagged | {} / {} |".format(
            d["cases_detected"], d["cases_detected"] + d["cases_missed"]),
        "| Detection rate | {:.1%} |".format(d["detection_rate"]),
        "| Median seconds from injection to first flag | {} |".format(
            "{:.0f}".format(d["median_delay_s"])
            if d["median_delay_s"] is not None else "n/a"),
        "| 90th percentile | {} |".format(
            "{:.0f}".format(d["p90_delay_s"])
            if d["p90_delay_s"] is not None else "n/a"),
        "",
    ]

    # The delay is bounded below by the window geometry, not by the model.
    if te is not None and len(te):
        first = te[te.is_target & (te.lead_s >= 0)].groupby("case").lead_s.min()
        delays = []
        for _c, g in te[te.is_target].groupby("case"):
            a = g[g.lead_s >= 0].sort_values("lead_s")
            hit = a[a.prob >= h["threshold"]]
            if len(hit):
                delays.append(float(hit.lead_s.iloc[0]))
        if delays and len(first):
            ds = pd.Series(delays)
            floor = float(first.min())
            lines += [
                "**This delay is bounded below by the window geometry, not by "
                "the model.** Windows are {}s long on a {}s hop, so the first "
                "window that can contain any post-injection sample ends {:.0f}s "
                "after the injection, and candidate windows arrive every {}s "
                "thereafter. The observed delays are quantised accordingly "
                "({}).".format(
                    report["dataset"]["window_s"], report["dataset"]["stride_s"],
                    floor, report["dataset"]["stride_s"],
                    ", ".join("{:.0f}s".format(v)
                              for v in sorted(ds.unique())[:6])),
                "",
                "| Flagged within | Share of held-out cases |",
                "|---|---|",
            ]
            for t in (floor, floor + 15, floor + 30, floor + 45):
                lines.append("| {:.0f}s | {:.1%} |".format(t, (ds <= t).mean()))
            lines += [
                "",
                "{:.0%} of cases are flagged in the very first window that "
                "could possibly show the fault, and {:.0%} by the second. The "
                "median of {:.0f}s is therefore two windows, close to the "
                "mechanical floor: the model is not the bottleneck. Finer "
                "windows would cut the delay at the cost of noisier features, "
                "and that trade-off -- not model capacity -- is what sets how "
                "early this can fire.".format(
                    (ds <= floor).mean(), (ds <= floor + 15).mean(),
                    float(ds.median())),
                "",
            ]

    los = report.get("leave_one_system_out", {})
    if los:
        lines += ["## Generalising to an unseen system", "",
                  "| Held-out system | PR-AUC | Precision | Recall |",
                  "|---|---|---|---|"]
        for k, v in los.items():
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
                SYSTEM_NAMES.get(k, k), v["pr_auc"], v["precision"], v["recall"]))
        lines.append("")

    lof = report.get("leave_one_fault_out", {})
    if lof:
        n_cases = cases.fault.value_counts()
        lines += ["## Generalising to an unseen fault type", "",
                  "The fault type is removed from training entirely, so the "
                  "model has to recognise a failure mechanism it has never "
                  "seen.", "",
                  "| Held-out fault | Cases | PR-AUC | Precision | Recall |",
                  "|---|---|---|---|---|"]
        for k, v in sorted(lof.items(), key=lambda kv: -kv[1]["pr_auc"]):
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                k, int(n_cases.get(k, 0)), v["pr_auc"], v["precision"],
                v["recall"]))
        lines.append("")

        # A fault whose mechanism is itself one of the input features should be
        # easier to recognise unseen than one that only surfaces as latency.
        direct = {"cpu": "CPU utilisation", "mem": "memory",
                  "socket": "socket count"}
        indirect = {"loss": "packet loss", "delay": "network delay"}
        have_direct = [k for k in direct if k in lof]
        have_indirect = [k for k in indirect if k in lof]
        if have_direct and have_indirect:
            fmt = lambda ks: ", ".join(  # noqa: E731
                "{} {:.3f}".format(k, lof[k]["pr_auc"]) for k in
                sorted(ks, key=lambda k: -lof[k]["pr_auc"]))
            lines += [
                "**What transfers is a fault whose mechanism the model can see "
                "directly.** Every fault type that has its own input feature "
                "scores well unseen ({}): a model that has never trained on a "
                "memory fault still watches memory climb. The fault types with "
                "no such channel surface only as latency and its shape, and "
                "they are where it struggles -- packet loss at {:.3f}, and "
                "network delay which reaches {:.3f} but recovers only {:.0%} "
                "of the fault.".format(
                    fmt(have_direct), lof["loss"]["pr_auc"],
                    lof["delay"]["pr_auc"], lof["delay"]["recall"])
                if {"loss", "delay"} <= set(have_indirect) else
                "**What transfers is a fault whose mechanism the model can see "
                "directly.** Fault types with their own input feature score "
                "well unseen ({}); those without such a channel, surfacing "
                "only as latency, are weaker ({}).".format(
                    fmt(have_direct), fmt(have_indirect)),
                "",
            ]
            if "disk" in lof:
                lines += [
                    "Disk stress sits between the two at {:.3f}, which fits the "
                    "same explanation: disk I/O *is* an input feature, but the "
                    "benchmark exports it for only about a third of services, "
                    "so the channel exists for some APIs and not others."
                    .format(lof["disk"]["pr_auc"]),
                    "",
                ]
        if "delay" in lof:
            lines += [
                "Network delay is the row to read carefully. PR-AUC {:.3f} "
                "looks respectable, but precision {:.3f} against recall {:.3f} "
                "says the model is confident when it fires and misses most of "
                "the fault otherwise. A single averaged number hides that; the "
                "precision/recall pair does not.".format(
                    lof["delay"]["pr_auc"], lof["delay"]["precision"],
                    lof["delay"]["recall"]),
                "",
            ]
        small = sorted((k for k in lof if int(n_cases.get(k, 0)) < 20),
                       key=lambda k: int(n_cases.get(k, 0)))
        if small:
            best_small = max(small, key=lambda k: lof[k]["pr_auc"])
            counts = ["{} ({} cases)".format(k, int(n_cases.get(k, 0)))
                      for k in small]
            joined = (counts[0] if len(counts) == 1
                      else " and ".join([", ".join(counts[:-1]), counts[-1]]))
            lines += [
                "Read the thinly sampled rows with caution -- {} -- since they "
                "carry wide error bars. They are in particular not evidence "
                "that code-level faults are harder as a family: {} scores "
                "{:.3f} on {} cases, among the best in the table, while {} "
                "scores {:.3f} on {}.".format(
                    joined, best_small, lof[best_small]["pr_auc"],
                    int(n_cases.get(best_small, 0)),
                    small[0], lof[small[0]]["pr_auc"],
                    int(n_cases.get(small[0], 0))),
                "",
            ]

    comp = scored.groupby(scored.label.map({0: "healthy", 1: "failing"}))[
        ["failure_risk", "error_frequency", "response_consistency",
         "reliability_score"]].mean()
    lines += ["## Reliability score", "",
              "Mean component values by true state:", "",
              "| State | Failure risk | Error frequency | Response consistency "
              "| Reliability score |", "|---|---|---|---|---|"]
    for state, r in comp.iterrows():
        lines.append("| {} | {:.3f} | {:.4f} | {:.3f} | {:.3f} |".format(
            state, r.failure_risk, r.error_frequency, r.response_consistency,
            r.reliability_score))
    lines.append("")

    ct = pd.crosstab(scored.label.map({0: "healthy", 1: "failing"}),
                     scored.status, normalize="index").reindex(
                         columns=viz.STATUS_ORDER).fillna(0)
    lines += ["Status assigned, by true state:", "",
              "| State | " + " | ".join(viz.STATUS_ORDER) + " |",
              "|---|" + "---|" * len(viz.STATUS_ORDER)]
    for state, r in ct.iterrows():
        lines.append("| {} | ".format(state) +
                     " | ".join("{:.1%}".format(v) for v in r) + " |")
    lines.append("")

    path = REPORTS / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("  results ->", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=None, help="case id for the example figure")
    args = p.parse_args(argv)

    report = json.loads((MODELS / "metrics.json").read_text())
    cases = pd.read_parquet(DATA_PROCESSED / "cases.parquet")
    scored = pd.read_parquet(DATA_PROCESSED / "scored.parquet")
    te = pd.read_parquet(DATA_PROCESSED / "test_predictions.parquet")

    print("writing figures")
    fig_dataset(cases)
    fig_model(te, report)
    fig_scores(scored[scored.usable])
    fig_generalisation(report)
    fig_importance()
    fig_detection(te, report["holdout"]["threshold"])

    case_id = args.case
    if case_id is None:
        # a held-out case with a clear, representative degradation
        candidates = [c for c in report["split"]["test_cases"]
                      if c.startswith("re2")]
        case_id = candidates[0] if candidates else report["split"]["test_cases"][0]
    fig_example_case(scored, cases, case_id)

    write_results(report, scored[scored.usable], cases, te)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
