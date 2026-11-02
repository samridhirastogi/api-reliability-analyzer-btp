"""Download real failure cases from the RCAEval benchmark.

No data is generated here. Each case is a recording of a real microservice
system (Online Boutique, Sock Shop or Train Ticket) running under real load
while a real fault was injected into one service, together with the ground
truth timestamp of that injection.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import DATA_RAW, HF_API, HF_FILE

CASE_RE = re.compile(r"^(re[123])(ob|ss|tt)_(.+)_([a-z0-9]+)_(\d+)$")


def list_cases(force: bool = False) -> dict[str, set[str]]:
    """Return {case_id: {available file names}} from the dataset index."""
    cache = DATA_RAW / "_index.json"
    if cache.exists() and not force:
        raw = json.loads(cache.read_text())
    else:
        resp = requests.get(HF_API, timeout=120)
        resp.raise_for_status()
        raw = [s["rfilename"] for s in resp.json().get("siblings", [])]
        cache.write_text(json.dumps(raw))

    cases: dict[str, set[str]] = defaultdict(set)
    for path in raw:
        if "/" in path:
            case, fname = path.split("/", 1)
            if CASE_RE.match(case):
                cases[case].add(fname)
    return dict(cases)


def select(cases: dict[str, set[str]], suites: list[str], systems: list[str],
           per_group: int | None) -> list[str]:
    """Pick case ids, optionally capping how many runs per fault group."""
    chosen: list[str] = []
    groups: dict[tuple, list[str]] = defaultdict(list)
    for case in sorted(cases):
        m = CASE_RE.match(case)
        suite, system, service, fault, _run = m.groups()
        if suite not in suites or system not in systems:
            continue
        groups[(suite, system, service, fault)].append(case)
    for key in sorted(groups):
        runs = sorted(groups[key], key=lambda c: int(CASE_RE.match(c).group(5)))
        chosen.extend(runs if per_group is None else runs[:per_group])
    return chosen


def _fetch(case: str, fname: str) -> tuple[str, str]:
    dest = DATA_RAW / case / fname
    if dest.exists() and dest.stat().st_size > 0:
        return case, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(f"{HF_FILE}/{case}/{fname}", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return case, "downloaded"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suites", default="re1,re2,re3")
    p.add_argument("--systems", default="ob,ss,tt")
    p.add_argument("--per-group", type=int, default=None,
                   help="cap runs per (system, service, fault) group")
    p.add_argument("--with-traces", type=int, default=0,
                   help="also download raw spans for N cases (endpoint-level view)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--refresh-index", action="store_true")
    args = p.parse_args(argv)

    cases = list_cases(force=args.refresh_index)
    picked = select(cases, args.suites.split(","), args.systems.split(","),
                    args.per_group)
    print(f"{len(cases)} cases in benchmark, {len(picked)} selected", flush=True)

    jobs: list[tuple[str, str]] = []
    for case in picked:
        for fname in ("metrics.parquet", "inject_time.txt"):
            if fname in cases[case]:
                jobs.append((case, fname))
        if "root_cause.txt" in cases[case]:
            jobs.append((case, "root_cause.txt"))

    if args.with_traces:
        with_traces = [c for c in picked if "traces.parquet" in cases[c]]
        for case in with_traces[: args.with_traces]:
            jobs.append((case, "traces.parquet"))
        print(f"including spans for {min(args.with_traces, len(with_traces))} cases",
              flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_fetch, c, f): (c, f) for c, f in jobs}
        for fut in as_completed(futs):
            case, fname = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                print(f"  FAILED {case}/{fname}: {exc}", file=sys.stderr, flush=True)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files", flush=True)

    print(f"done: {done - failed} ok, {failed} failed -> {DATA_RAW}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
