"""Normalise one recorded case into a tidy long telemetry table.

Each RCAEval suite exports slightly different column names (RE1 / Online
Boutique uses ``_load`` and a single ``_latency``; the others use ``_workload``
with ``_latency-50`` / ``_latency-90``).  This module hides those differences
behind one schema:

    service | time | latency_p50 | latency_p90 | error_rate | workload |
    cpu | mem | diskio | socket

``time`` is the original Unix timestamp. Nothing is resampled, smoothed or
interpolated, so every value stays a measurement taken from the running system.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (DATA_RAW, MIN_CASE_DURATION_S, SUITE_NAMES, SYSTEM_NAMES)
from .download import CASE_RE

# raw metric suffix -> canonical column name
METRIC_MAP = {
    "latency-50": "latency_p50",
    "latency-90": "latency_p90",
    "latency": "latency_p50",   # RE1 / Online Boutique exports a single latency
    "error": "error_rate",
    "workload": "workload",
    "load": "workload",         # RE1 / Online Boutique name for request rate
    "cpu": "cpu",
    "mem": "mem",
    "diskio": "diskio",
    "socket": "socket",
}
CANONICAL = ["latency_p50", "latency_p90", "error_rate", "workload",
             "cpu", "mem", "diskio", "socket"]

# Datastores, brokers and exporters are not request-serving APIs. They are kept
# out of the scored universe; their metrics remain available as context.
INFRA_SUFFIXES = ("-db", "-mongo", "-mysql", "-redis")
INFRA_EXACT = {"redis", "rabbitmq", "rabbitmq-exporter", "session-db",
               "mongo", "nacos", "kafka", "zookeeper", "jaeger"}


def is_infrastructure(service: str) -> bool:
    return service in INFRA_EXACT or service.endswith(INFRA_SUFFIXES)


def parse_case_id(case: str) -> dict:
    m = CASE_RE.match(case)
    if not m:
        raise ValueError("unrecognised case id: " + case)
    suite, system, service, fault, run = m.groups()
    return {"case": case, "suite": suite, "system": system,
            "target_service": service, "fault": fault, "run": int(run)}


def read_case(case_dir):
    """Return ``(long telemetry frame, case metadata)`` or ``None``."""
    metrics_path = case_dir / "metrics.parquet"
    inject_path = case_dir / "inject_time.txt"
    if not metrics_path.exists() or not inject_path.exists():
        return None

    meta = parse_case_id(case_dir.name)
    meta["inject_time"] = int(inject_path.read_text().strip())

    wide = pd.read_parquet(metrics_path)
    if "time" not in wide.columns or len(wide) == 0:
        return None
    wide = wide.sort_values("time").reset_index(drop=True)

    # column name -> (service, canonical metric)
    by_metric: dict[str, dict[str, str]] = {}
    unknown = set()
    for col in wide.columns:
        if col == "time" or "_" not in col:
            continue
        service, suffix = col.rsplit("_", 1)
        canon = METRIC_MAP.get(suffix)
        if canon is None:
            unknown.add(suffix)
            continue
        by_metric.setdefault(canon, {})[service] = col

    if "latency_p50" not in by_metric:
        return None  # nothing to score without a latency signal

    services = sorted({s for m in by_metric.values() for s in m})
    times = wide["time"].to_numpy()
    n_t = len(times)

    long = pd.DataFrame({
        "service": np.repeat(services, n_t),
        "time": np.tile(times, len(services)),
    })
    blank = np.full(n_t, np.nan)
    for canon in CANONICAL:
        mapping = by_metric.get(canon, {})
        long[canon] = np.concatenate([
            wide[mapping[s]].to_numpy(dtype="float64") if s in mapping else blank
            for s in services
        ])

    # Resolve which service the fault was injected into. The case id is the
    # reliable source; a few root_cause.txt files hold a raw log line rather
    # than a service name, so that file is only trusted when it names a service
    # that actually appears in the telemetry.
    if meta["target_service"] not in services:
        rc = case_dir / "root_cause.txt"
        if rc.exists():
            for tok in rc.read_text().replace(",", " ").split():
                if tok.strip() in services:
                    meta["target_service"] = tok.strip()
                    break

    meta["unknown_suffixes"] = sorted(unknown)
    meta["n_services"] = len(services)
    meta["n_samples"] = n_t
    meta["duration_s"] = int(times[-1] - times[0])
    meta["inject_offset_s"] = int(meta["inject_time"] - times[0])
    meta["target_present"] = meta["target_service"] in services
    # A handful of recordings are truncated, or carry a corrupted injection
    # timestamp that falls outside the recording. They cannot be labelled.
    meta["timeline_ok"] = bool(0 <= meta["inject_offset_s"] <= meta["duration_s"])
    meta["long_enough"] = bool(meta["duration_s"] >= MIN_CASE_DURATION_S)
    meta["usable"] = bool(meta["target_present"] and meta["timeline_ok"]
                          and meta["long_enough"])
    return long, meta


def main(argv=None):
    """Summarise the downloaded corpus (composition table for the report)."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    case_dirs = sorted(d for d in DATA_RAW.iterdir()
                       if d.is_dir() and CASE_RE.match(d.name))
    if args.limit:
        case_dirs = case_dirs[: args.limit]

    rows, bad = [], []
    for d in case_dirs:
        try:
            result = read_case(d)
        except Exception as exc:  # noqa: BLE001
            bad.append((d.name, str(exc)))
            continue
        if result is None:
            bad.append((d.name, "unreadable"))
            continue
        tel, meta = result
        meta["error_metric_services"] = int(
            tel.groupby("service").error_rate.apply(lambda s: s.notna().any()).sum())
        rows.append(meta)

    cases = pd.DataFrame(rows)
    cases["system_name"] = cases.system.map(SYSTEM_NAMES)
    cases["suite_name"] = cases.suite.map(SUITE_NAMES)

    print("readable cases: {} / {}".format(len(cases), len(case_dirs)))
    print("\ncases per system and suite:")
    print(pd.crosstab(cases.suite_name, cases.system_name).to_string())
    print("\nfault types:")
    print(cases.fault.value_counts().to_string())
    print("\nrecording length (s):")
    print(cases.duration_s.describe()[["min", "50%", "max"]].to_string())
    print("\nseconds of normal behaviour before injection:")
    print(cases.inject_offset_s.describe()[["min", "50%", "max"]].to_string())
    print("\nservices per case: median {:.0f}, max {}".format(
        cases.n_services.median(), cases.n_services.max()))
    print("\nvalidity checks:")
    for flag in ("target_present", "timeline_ok", "long_enough", "usable"):
        print("  {:<15} {}/{}".format(flag, int(cases[flag].sum()), len(cases)))
    rejected = cases.loc[~cases.usable, "case"].tolist()
    if rejected:
        print("  rejected: {}".format(rejected))
    if bad:
        print("\nunreadable: {} e.g. {}".format(len(bad), bad[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
