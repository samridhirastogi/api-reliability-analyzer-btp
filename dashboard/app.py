"""Reliability dashboard.

Replays a recorded failure case window by window and shows what the scoring
pipeline would have reported to an on-call engineer at each moment: every API
ranked by reliability score, the three components behind that score, and the
raw telemetry underneath it.

Run with::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import viz  # noqa: E402
from src.config import (BAND_DEGRADED, BAND_RELIABLE, DATA_PROCESSED, DATA_RAW,
                        MODELS, REPORTS, SYSTEM_NAMES, W_CONSISTENCY,
                        W_ERROR_FREQ, W_FAILURE_RISK)  # noqa: E402
from src.ingest import read_case  # noqa: E402

SCORED = DATA_PROCESSED / "scored.parquet"
CASES = DATA_PROCESSED / "cases.parquet"

st.set_page_config(page_title="API Reliability Scoring", page_icon="\U0001f4e1",
                   layout="wide")

try:  # Altair >= 5.5
    alt.theme.register("btp", enable=True)(viz.altair_theme)
except AttributeError:  # pragma: no cover - older Altair
    alt.themes.register("btp", viz.altair_theme)
    alt.themes.enable("btp")


# --------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_cases() -> pd.DataFrame:
    c = pd.read_parquet(CASES)
    c["system_name"] = c.system.map(SYSTEM_NAMES)
    return c


@st.cache_data(show_spinner=False)
def load_scored(case: str) -> pd.DataFrame:
    df = pd.read_parquet(SCORED, filters=[("case", "==", case)])
    return df.sort_values(["window_end", "service"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_raw(case: str) -> pd.DataFrame:
    result = read_case(DATA_RAW / case)
    return result[0] if result else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_report() -> dict:
    path = MODELS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_importance() -> pd.DataFrame:
    path = REPORTS / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if not SCORED.exists():
    st.error("No scored data yet. Run the pipeline first:\n\n"
             "```\npython -m src.download\npython -m src.features\n"
             "python -m src.labels\npython -m src.train\npython -m src.score\n```")
    st.stop()

cases = load_cases()
report = load_model_report()
test_cases = set(report.get("split", {}).get("test_cases", []))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("API Reliability")
st.sidebar.caption("Real telemetry from the RCAEval benchmark — recorded "
                   "runs of three microservice systems under injected faults.")

only_test = st.sidebar.toggle(
    "Held-out cases only", value=bool(test_cases),
    help="Cases the model never saw during training. Use these to judge it.")
pool = cases[cases.case.isin(test_cases)] if only_test and test_cases else cases

system = st.sidebar.selectbox("System", sorted(pool.system_name.dropna().unique()))
pool = pool[pool.system_name == system]
fault = st.sidebar.selectbox("Fault type", ["(any)"] + sorted(pool.fault.unique()))
if fault != "(any)":
    pool = pool[pool.fault == fault]

if pool.empty:
    st.warning("No cases match those filters.")
    st.stop()

case_id = st.sidebar.selectbox(
    "Recorded case", sorted(pool.case),
    format_func=lambda c: "{}  →  {} / {}".format(
        c, cases.loc[cases.case == c, "target_service"].iloc[0],
        cases.loc[cases.case == c, "fault"].iloc[0]))

meta = cases[cases.case == case_id].iloc[0]
scored = load_scored(case_id)
if scored.empty:
    st.warning("That case produced no scored windows.")
    st.stop()

windows = sorted(scored.window_end.unique())
rel = [int(w - meta.inject_time) for w in windows]

st.sidebar.divider()
top_n = st.sidebar.slider("APIs shown in charts", 3, viz.MAX_SERIES, 6)
st.sidebar.caption(
    "Reliability = {:.2f}·(1−risk) + {:.2f}·(1−errors) "
    "+ {:.2f}·consistency".format(W_FAILURE_RISK, W_ERROR_FREQ,
                                       W_CONSISTENCY))


# ------------------------------------------------------------------- header
st.title("AI-based reliability scoring of backend APIs")
head = st.columns([2, 1, 1, 1])
head[0].metric("System", system)
head[1].metric("Injected API", meta.target_service)
head[2].metric("Fault", meta.fault)
head[3].metric("APIs scored", scored.service.nunique())
if case_id in test_cases:
    st.caption("This case was held out of training — the model has never "
               "seen any window of it.")

ENDPOINTS = DATA_PROCESSED / "endpoints.parquet"


@st.cache_data(show_spinner=False)
def load_endpoints(case: str) -> pd.DataFrame:
    if not ENDPOINTS.exists():
        return pd.DataFrame()
    return pd.read_parquet(ENDPOINTS, filters=[("case", "==", case)])


tab_live, tab_case, tab_ep, tab_model, tab_about = st.tabs(
    ["Live scoring", "Case timeline", "Endpoints", "Model", "Method"])


# -------------------------------------------------------------- live tab
with tab_live:
    idx = st.select_slider(
        "Replay position (seconds relative to the moment the fault was injected)",
        options=list(range(len(windows))),
        value=min(range(len(windows)), key=lambda i: abs(rel[i] - 120)),
        format_func=lambda i: "{:+d}s".format(rel[i]))
    now = windows[idx]
    snap = scored[scored.window_end == now].copy()
    snap["rank"] = snap.reliability_score.rank(method="first")
    snap = snap.sort_values("reliability_score")

    state = "fault active" if rel[idx] >= 0 else "before injection"
    cols = st.columns(4)
    cols[0].metric("Window ends at", "{:+d}s".format(rel[idx]), state)
    cols[1].metric("Failure-prone APIs",
                   int((snap.status == "Failure-prone").sum()))
    cols[2].metric("Lowest score", "{:.2f}".format(snap.reliability_score.min()),
                   snap.service.iloc[0])
    target_row = snap[snap.service == meta.target_service]
    cols[3].metric("Score of injected API",
                   "{:.2f}".format(target_row.reliability_score.iloc[0])
                   if len(target_row) else "n/a")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("APIs ranked by reliability")
        bars = snap.copy()
        bars["label"] = bars.status.map(viz.status_label)
        chart = (
            alt.Chart(bars)
            .mark_bar(height=16, cornerRadiusEnd=4)
            .encode(
                x=alt.X("reliability_score:Q", title="Reliability score",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("service:N", sort=alt.SortField("reliability_score"),
                        title=None),
                color=alt.Color(
                    "status:N", title="Status",
                    scale=alt.Scale(domain=viz.STATUS_ORDER,
                                    range=[viz.STATUS_COLOR[s]
                                           for s in viz.STATUS_ORDER]),
                    legend=alt.Legend(symbolType="square")),
                tooltip=[
                    alt.Tooltip("service:N", title="API"),
                    alt.Tooltip("label:N", title="Status"),
                    alt.Tooltip("reliability_score:Q", title="Score", format=".3f"),
                    alt.Tooltip("failure_risk:Q", title="Failure risk", format=".3f"),
                    alt.Tooltip("error_frequency:Q", title="Error freq", format=".4f"),
                    alt.Tooltip("response_consistency:Q", title="Consistency",
                                format=".3f"),
                ],
            )
            .properties(height=max(200, 26 * len(bars)))
        )
        text = (alt.Chart(bars).mark_text(align="left", dx=6, fontSize=11,
                                          color=viz.INK_SECONDARY)
                .encode(x="reliability_score:Q",
                        y=alt.Y("service:N", sort=alt.SortField("reliability_score")),
                        text=alt.Text("reliability_score:Q", format=".2f")))
        rules = (alt.Chart(pd.DataFrame({"v": [BAND_DEGRADED, BAND_RELIABLE]}))
                 .mark_rule(strokeDash=[4, 4], color=viz.BASELINE)
                 .encode(x="v:Q"))
        st.altair_chart(chart + text + rules, width="stretch")
        st.caption("Dashed lines mark the band edges: below {:.2f} "
                   "failure-prone, above {:.2f} reliable."
                   .format(BAND_DEGRADED, BAND_RELIABLE))

    with right:
        st.subheader("What drives each score")
        worst = snap.head(top_n).copy()
        parts = []
        for _, r in worst.iterrows():
            err = 0.0 if pd.isna(r.error_frequency) else r.error_frequency
            parts += [
                {"service": r.service, "component": "Failure risk (40%)",
                 "value": W_FAILURE_RISK * (1 - r.failure_risk)},
                {"service": r.service, "component": "Error frequency (35%)",
                 "value": W_ERROR_FREQ * (1 - err)},
                {"service": r.service, "component": "Response consistency (25%)",
                 "value": W_CONSISTENCY * r.response_consistency},
            ]
        comp = pd.DataFrame(parts)
        order = ["Failure risk (40%)", "Error frequency (35%)",
                 "Response consistency (25%)"]
        st.altair_chart(
            alt.Chart(comp).mark_bar(height=16, stroke=viz.SURFACE, strokeWidth=2)
            .encode(
                x=alt.X("value:Q", title="Contribution to the score",
                        stack="zero", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("service:N", title=None,
                        sort=list(worst.service)),
                color=alt.Color("component:N", title="Component",
                                scale=alt.Scale(domain=order,
                                                range=viz.SERIES[:3]),
                                sort=order,
                                legend=alt.Legend(orient="bottom", columns=1)),
                order=alt.Order("component:N", sort="ascending"),
                tooltip=["service:N", "component:N",
                         alt.Tooltip("value:Q", format=".3f")],
            ).properties(height=max(180, 26 * len(worst))),
            width="stretch")
        st.caption("The {} lowest-scoring APIs in this window. A short bar "
                   "segment is the component dragging the score down."
                   .format(len(worst)))

    st.subheader("Table view")
    table = snap[["service", "status", "reliability_score", "failure_risk",
                  "error_frequency", "response_consistency", "lat50_mean",
                  "lat90_mean", "wl_mean", "cpu_mean"]].copy()
    table.insert(2, "state", table.status.map(viz.status_label))
    table = table.drop(columns=["status"])
    table.columns = ["API", "State", "Reliability", "Failure risk",
                     "Error freq", "Consistency", "p50 latency (s)",
                     "p90 latency (s)", "Requests/s", "CPU"]
    st.dataframe(table.round(4), width="stretch", hide_index=True)


# ---------------------------------------------------------- case timeline
with tab_case:
    st.subheader("Reliability score through the whole recording")
    span = scored.groupby("service").reliability_score.min().sort_values()
    picks = st.multiselect("APIs", sorted(scored.service.unique()),
                           default=list(span.index[:top_n]))
    if picks:
        ts = scored[scored.service.isin(picks)].copy()
        ts["t"] = ts.window_end - meta.inject_time
        base = alt.Chart(ts).encode(
            x=alt.X("t:Q", title="Seconds relative to fault injection"),
            y=alt.Y("reliability_score:Q", title="Reliability score",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("service:N", title="API",
                            scale=alt.Scale(domain=sorted(picks),
                                            range=viz.SERIES[:len(picks)])),
        )
        hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover",
                                    empty=False)
        line = base.mark_line()
        dots = base.mark_point(filled=True, size=70, opacity=0).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("service:N", title="API"),
                     alt.Tooltip("t:Q", title="t (s)"),
                     alt.Tooltip("reliability_score:Q", title="Score",
                                 format=".3f"),
                     alt.Tooltip("failure_risk:Q", title="Risk", format=".3f")],
        ).add_params(hover)
        inject = (alt.Chart(pd.DataFrame({"t": [0]}))
                  .mark_rule(color=viz.STATUS_COLOR["Failure-prone"],
                             strokeWidth=2, strokeDash=[6, 3])
                  .encode(x="t:Q"))
        label = (alt.Chart(pd.DataFrame({"t": [0], "y": [0.04],
                                         "s": ["fault injected"]}))
                 .mark_text(align="left", dx=6, fontSize=11,
                            color=viz.STATUS_COLOR["Failure-prone"])
                 .encode(x="t:Q", y="y:Q", text="s:N"))
        st.altair_chart((line + dots + inject + label).properties(height=300),
                        width="stretch")

        st.subheader("Raw telemetry behind the score")
        raw = load_raw(case_id)
        if not raw.empty:
            metric = st.radio(
                "Measurement", ["latency_p50", "latency_p90", "error_rate",
                                "workload", "cpu"],
                horizontal=True,
                format_func=lambda m: {
                    "latency_p50": "p50 latency (s)",
                    "latency_p90": "p90 latency (s)",
                    "error_rate": "error rate",
                    "workload": "requests / s",
                    "cpu": "CPU"}[m])
            sub = raw[raw.service.isin(picks)].copy()
            sub["t"] = sub.time - meta.inject_time
            sub = sub.dropna(subset=[metric])
            if sub.empty:
                st.info("That measurement is not exported for the selected APIs "
                        "in this system.")
            else:
                mchart = (alt.Chart(sub).mark_line()
                          .encode(
                              x=alt.X("t:Q", title="Seconds relative to injection"),
                              y=alt.Y(metric + ":Q", title=metric),
                              color=alt.Color(
                                  "service:N", title="API",
                                  scale=alt.Scale(domain=sorted(picks),
                                                  range=viz.SERIES[:len(picks)])),
                              tooltip=["service:N", alt.Tooltip("t:Q"),
                                       alt.Tooltip(metric + ":Q", format=".4f")])
                          .properties(height=260))
                st.altair_chart(mchart + inject, width="stretch")
                with st.expander("Table view of this measurement"):
                    st.dataframe(sub[["t", "service", metric]].round(5),
                                 width="stretch", hide_index=True,
                                 height=260)
    else:
        st.info("Pick at least one API.")


# --------------------------------------------------------------- endpoints
with tab_ep:
    ep = load_endpoints(case_id)
    if ep.empty:
        st.info(
            "No request spans downloaded for this case, so endpoint-level "
            "scoring is unavailable here.\n\n"
            "Fetch spans for a few cases and score them:\n\n"
            "```\npython -m src.download --with-traces 6\n"
            "python -m src.endpoints\n```\n\n"
            "Cases that already have spans appear with an **Endpoints** table.")
    else:
        st.subheader("Individual endpoints, scored from the requests themselves")
        st.caption(
            "Latency, error rate and consistency here are measured directly "
            "from distributed-tracing spans, one row per endpoint. The failure-"
            "risk term is inherited from the endpoint's parent service, because "
            "CPU and memory telemetry is only exported per service.")

        ep = ep.copy()
        ep["t"] = ep.window_end - meta.inject_time
        options = sorted(ep.t.unique())
        pick_t = st.select_slider(
            "Replay position (seconds relative to injection)", options=options,
            value=min(options, key=lambda v: abs(v - 120)),
            format_func=lambda v: "{:+d}s".format(int(v)))
        snap_ep = ep[ep.t == pick_t].sort_values("reliability_score")

        c1, c2 = st.columns([3, 2])
        with c1:
            bars = snap_ep.head(12).copy()
            bars["label"] = bars.status.map(viz.status_label)
            st.altair_chart(
                alt.Chart(bars).mark_bar(height=15, cornerRadiusEnd=4).encode(
                    x=alt.X("reliability_score:Q", title="Reliability score",
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("endpoint:N", sort=list(bars.endpoint), title=None),
                    color=alt.Color(
                        "status:N", title="Status",
                        scale=alt.Scale(domain=viz.STATUS_ORDER,
                                        range=[viz.STATUS_COLOR[s]
                                               for s in viz.STATUS_ORDER]),
                        legend=alt.Legend(symbolType="square")),
                    tooltip=[alt.Tooltip("api:N", title="Endpoint"),
                             alt.Tooltip("label:N", title="Status"),
                             alt.Tooltip("reliability_score:Q", format=".3f"),
                             alt.Tooltip("requests:Q", title="Requests"),
                             alt.Tooltip("error_frequency:Q", format=".4f"),
                             alt.Tooltip("latency_p50:Q", title="p50 (s)",
                                         format=".4f")],
                ).properties(height=max(200, 26 * len(bars))),
                width="stretch")
        with c2:
            st.metric("Endpoints scored", snap_ep.api.nunique())
            st.metric("Requests in this window", int(snap_ep.requests.sum()))
            st.metric("Endpoints with errors",
                      int((snap_ep.error_frequency > 0).sum()))

        tbl = snap_ep[["service", "endpoint", "status", "reliability_score",
                       "failure_risk", "error_frequency", "response_consistency",
                       "latency_p50", "latency_p95", "requests"]].copy()
        tbl.insert(3, "state", tbl.status.map(viz.status_label))
        tbl = tbl.drop(columns=["status"])
        tbl.columns = ["Service", "Endpoint", "State", "Reliability",
                       "Failure risk (service)", "Error freq", "Consistency",
                       "p50 (s)", "p95 (s)", "Requests"]
        st.dataframe(tbl.round(4), width="stretch", hide_index=True)


# ------------------------------------------------------------------- model
with tab_model:
    if not report:
        st.info("Train the model first: `python -m src.train`.")
    else:
        h = report["holdout"]
        d = report["detection"]
        st.subheader("Failure-risk model on held-out cases")
        m = st.columns(5)
        m[0].metric("PR-AUC", "{:.3f}".format(h["pr_auc"]),
                    help="Area under precision-recall. The baseline for this "
                         "class balance is {:.3f}.".format(h["positive_rate"]))
        m[1].metric("ROC-AUC", "{:.3f}".format(h["roc_auc"]))
        m[2].metric("F1", "{:.3f}".format(h["f1"]))
        m[3].metric("Recall", "{:.3f}".format(h["recall"]))
        m[4].metric("Precision", "{:.3f}".format(h["precision"]))

        st.subheader("Catching the real fault")
        m = st.columns(3)
        m[0].metric("Cases detected", "{} / {}".format(
            d["cases_detected"], d["cases_detected"] + d["cases_missed"]))
        m[1].metric("Median time to flag",
                    "{:.0f}s".format(d["median_delay_s"])
                    if d["median_delay_s"] is not None else "n/a")
        m[2].metric("Brier score", "{:.4f}".format(h["brier"]),
                    help="Lower is better; measures how well the risk "
                         "probability is calibrated.")

        gen = report.get("leave_one_system_out", {})
        if gen:
            rows = [{"Held-out system": SYSTEM_NAMES.get(k, k),
                     "PR-AUC": v["pr_auc"], "Recall": v["recall"],
                     "Precision": v["precision"]} for k, v in gen.items()]
            st.subheader("Generalising to a system never seen in training")
            st.dataframe(pd.DataFrame(rows).round(3), width="stretch",
                         hide_index=True)

        imp = load_importance()
        if not imp.empty:
            st.subheader("What the model looks at")
            top = imp.head(15).sort_values("impurity")
            st.altair_chart(
                alt.Chart(top).mark_bar(height=14, color=viz.SERIES[0])
                .encode(
                    x=alt.X("impurity:Q", title="Importance"),
                    y=alt.Y("feature:N", sort=list(top.feature), title=None),
                    tooltip=["feature:N", alt.Tooltip("impurity:Q", format=".4f")])
                .properties(height=360),
                width="stretch")
            with st.expander("Full importance table"):
                st.dataframe(imp.round(5), width="stretch",
                             hide_index=True)


# ------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### Where the data comes from

Every number in this dashboard was measured on a running system. The recordings
come from **RCAEval**, a published benchmark of three open-source microservice
applications (Online Boutique, Sock Shop, Train Ticket) driven by real workload
generators while real faults were injected into one service at a time. For each
case the benchmark records per-second telemetry for every service and the exact
timestamp of the injection, which is used as ground truth. No values are
simulated or generated.

### How a score is produced

1. **Windowing.** Telemetry is cut into 30-second windows every 15 seconds.
2. **Features.** Each window gets level and shape features (mean, spread, tail
   ratio of latency; error rate; request rate; CPU and memory) plus the same
   quantities *relative to that API's own preceding baseline*, so a slow API is
   not penalised for being slow, only for changing.
3. **Failure risk.** A calibrated random forest estimates the probability that
   the API is in a failing state over the next 30 seconds.
4. **Reliability score.** Risk, measured error frequency and response
   consistency are combined with the proposal's weights of 0.40 / 0.35 / 0.25.

### One definition had to change

The proposal defines response consistency as the inverse of the standard
deviation of latency. That is unbounded - a perfectly steady API gives 1/0 - and
depends on whether latency is recorded in seconds or milliseconds, so it cannot
be added to two terms that live between 0 and 1. This implementation keeps the
intent and uses the coefficient of variation instead:

> consistency = 1 / (1 + sigma/mean)

which is dimensionless, sits in (0, 1], equals 1 for perfectly steady latency
and falls towards 0 as jitter grows.

Where an API exposes no error-rate metric, the error term is dropped and the
remaining weights are renormalised rather than substituting zero, which would
otherwise reward an API for being unmeasured.
""")
