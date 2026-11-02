"""Turn 1 Hz telemetry into causal per-window feature vectors.

Design notes
------------
* A window is ``WINDOW_S`` seconds long and starts every ``STRIDE_S`` seconds.
  Windows are built by aggregating non-overlapping ``STRIDE_S`` blocks and then
  combining consecutive blocks, which makes overlapping windows cheap and exact.
* Every feature is computed from samples at or before the window end, so no
  information from the labelled horizon can leak into the feature vector. The
  trailing baseline stops one full window short of the current window, so the
  two never share a sample.
* Absolute levels are not comparable across services -- a catalogue lookup and
  a checkout call have very different normal latencies -- so each raw feature is
  paired with a *relative* feature measured against that service's own recent
  baseline. Those relative features are what let one model generalise across
  services and across systems.
* Cases are processed one at a time: the full benchmark is roughly 40 million
  telemetry rows, which does not fit comfortably in memory, while the resulting
  window table does.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (BASELINE_S, DATA_PROCESSED, DATA_RAW,
                     MIN_BASELINE_COVERAGE, MIN_SAMPLES_IN_WINDOW, STRIDE_S,
                     WINDOW_S)
from .download import CASE_RE
from .ingest import is_infrastructure, read_case

EPS = 1e-9
BLOCKS_PER_WINDOW = WINDOW_S // STRIDE_S
BASELINE_BLOCKS = BASELINE_S // STRIDE_S
METRICS = ["latency_p50", "latency_p90", "error_rate", "workload",
           "cpu", "mem", "socket", "diskio"]
SHORT = {"latency_p50": "lat50", "latency_p90": "lat90", "error_rate": "err",
         "workload": "wl", "cpu": "cpu", "mem": "mem", "socket": "sock",
         "diskio": "disk"}
RELATIVE = ["lat50", "lat90", "wl", "cpu", "mem", "sock", "disk"]

FEATURE_COLUMNS = [
    # level and shape inside the window
    "lat50_mean", "lat50_std", "lat50_max", "lat50_cv", "consistency",
    "lat90_mean", "lat90_max", "tail_ratio", "lat_spread",
    "err_mean", "err_max", "wl_mean", "wl_cv", "cpu_mean", "cpu_max",
    "mem_mean", "sock_mean", "disk_mean",
    # change against the service's own trailing baseline
    "lat50_ratio", "lat50_z", "lat90_ratio", "lat90_z", "cv_ratio",
    "err_delta", "wl_ratio", "wl_z", "cpu_ratio", "cpu_z",
    "mem_ratio", "mem_z", "sock_ratio", "disk_ratio",
]
KEY_COLUMNS = ["case", "suite", "system", "service", "window_end", "t0",
               "is_infra"]
# kept for the dashboard and the scoring step, not fed to the model
CONTEXT_COLUMNS = ["bl_lat50_mean", "bl_err_mean", "bl_cpu_mean", "wl_std",
                   "lat50_min", "lat90_std", "err_std"]


def _block_stats(tel: pd.DataFrame) -> pd.DataFrame:
    """Count / sum / sum-of-squares / min / max per STRIDE_S block."""
    tel = tel.copy()
    t0 = int(tel["time"].min())
    tel["block"] = ((tel["time"] - t0) // STRIDE_S).astype(int)

    agg = {"time": "max"}
    for m in METRICS:
        tel[m + "__sq"] = tel[m] * tel[m]
        tel[m + "__n"] = tel[m].notna().astype("float64")
        tel[m + "__mx"] = tel[m]
        tel[m + "__mn"] = tel[m]
        agg[m] = "sum"
        agg[m + "__sq"] = "sum"
        agg[m + "__n"] = "sum"
        agg[m + "__mx"] = "max"
        agg[m + "__mn"] = "min"

    blocks = tel.groupby(["service", "block"], as_index=False, sort=True).agg(agg)
    blocks["t0"] = t0
    return blocks


def _combine(blocks, group, n_blocks: int, shift: int, prefix: str):
    """Aggregate `n_blocks` consecutive blocks, ending `shift` blocks back."""
    out = {}

    def roll_sum(col):
        return group[col].transform(
            lambda x: x.rolling(n_blocks, min_periods=n_blocks).sum().shift(shift))

    for m in METRICS:
        s = SHORT[m]
        n = roll_sum(m + "__n")
        total = roll_sum(m)
        sq = roll_sum(m + "__sq")
        safe_n = n.replace(0, np.nan)
        mean = total / safe_n
        var = (sq / safe_n) - mean * mean
        out[prefix + s + "_mean"] = mean
        out[prefix + s + "_std"] = np.sqrt(var.clip(lower=0))
        out[prefix + s + "_n"] = n
        if not prefix:
            out[s + "_max"] = group[m + "__mx"].transform(
                lambda x: x.rolling(n_blocks, min_periods=1).max().shift(shift))
            out[s + "_min"] = group[m + "__mn"].transform(
                lambda x: x.rolling(n_blocks, min_periods=1).min().shift(shift))
    return pd.DataFrame(out, index=blocks.index)


def build_case(tel: pd.DataFrame) -> pd.DataFrame:
    """Feature table for one recorded case."""
    blocks = _block_stats(tel)
    group = blocks.groupby("service", sort=False)

    # the window ending at this block, and the baseline ending one window back
    cur = _combine(blocks, group, BLOCKS_PER_WINDOW, 0, "")
    base = _combine(blocks, group, BASELINE_BLOCKS, BLOCKS_PER_WINDOW, "bl_")

    f = pd.concat([blocks[["service"]], cur, base], axis=1)
    f["window_end"] = blocks["time"].to_numpy()
    f["t0"] = blocks["t0"].to_numpy()

    # ---- shape of the latency distribution inside the window
    f["lat50_cv"] = f.lat50_std / (f.lat50_mean + EPS)
    f["consistency"] = 1.0 / (1.0 + f.lat50_cv)
    f["tail_ratio"] = f.lat90_mean / (f.lat50_mean + EPS)
    f["lat_spread"] = (f.lat50_max - f.lat50_min) / (f.lat50_mean + EPS)
    f["wl_cv"] = f.wl_std / (f.wl_mean + EPS)
    f["bl_lat50_cv"] = f.bl_lat50_std / (f.bl_lat50_mean + EPS)
    f["cv_ratio"] = f.lat50_cv / (f.bl_lat50_cv + EPS)

    # ---- change relative to the service's own recent normal
    for s in RELATIVE:
        f[s + "_ratio"] = f[s + "_mean"] / (f["bl_" + s + "_mean"] + EPS)
        f[s + "_z"] = ((f[s + "_mean"] - f["bl_" + s + "_mean"])
                       / (f["bl_" + s + "_std"] + EPS))
    f["err_delta"] = f.err_mean - f.bl_err_mean

    keep = (f.lat50_n >= MIN_SAMPLES_IN_WINDOW) & (
        f.bl_lat50_n >= MIN_BASELINE_COVERAGE * BASELINE_S)
    f = f[keep].copy()
    f["is_infra"] = f.service.map(is_infrastructure)
    return f


def build_all(case_dirs, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, metas, skipped = [], [], []
    for i, d in enumerate(case_dirs, 1):
        try:
            result = read_case(d)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            skipped.append((d.name, str(exc)))
            continue
        if result is None:
            skipped.append((d.name, "missing metrics / inject_time / latency"))
            continue
        tel, meta = result
        if not meta["usable"]:
            reason = ("injected service has no telemetry" if not meta["target_present"]
                      else "injection timestamp outside recording"
                      if not meta["timeline_ok"] else "recording too short")
            skipped.append((d.name, reason))
            continue
        f = build_case(tel)
        if f.empty:
            skipped.append((d.name, "recording too short for one baseline"))
            continue
        f.insert(0, "case", meta["case"])
        f.insert(1, "suite", meta["suite"])
        f.insert(2, "system", meta["system"])
        frames.append(f)
        metas.append(meta)
        if verbose and i % 50 == 0:
            print("  {}/{} cases".format(i, len(case_dirs)), flush=True)

    if skipped:
        print("skipped {} cases; first few: {}".format(len(skipped), skipped[:3]))
    if not frames:
        raise SystemExit("no usable cases - run `python -m src.download` first")

    cols = KEY_COLUMNS + FEATURE_COLUMNS + CONTEXT_COLUMNS
    win = pd.concat(frames, ignore_index=True)
    win = win[[c for c in cols if c in win.columns]]
    floats = win.select_dtypes("float64").columns
    win[floats] = win[floats].astype("float32")
    return win, pd.DataFrame(metas)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DATA_PROCESSED / "windows.parquet"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    case_dirs = sorted(d for d in DATA_RAW.iterdir()
                       if d.is_dir() and CASE_RE.match(d.name))
    if args.limit:
        case_dirs = case_dirs[: args.limit]
    print("building windows from {} recorded cases".format(len(case_dirs)))

    win, cases = build_all(case_dirs)
    win.to_parquet(args.out, index=False)
    cases.to_parquet(DATA_PROCESSED / "cases.parquet", index=False)

    print("cases kept    : {}".format(len(cases)))
    print("windows       : {:,}".format(len(win)))
    print("services      : {}".format(win.service.nunique()))
    print("per case (avg): {:.0f}".format(len(win) / max(len(cases), 1)))
    missing = win[FEATURE_COLUMNS].isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        print("features with missing values:")
        print(missing.head(12).round(3).to_string())
    print("written ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
