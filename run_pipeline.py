"""Run the whole pipeline end to end.

    python run_pipeline.py                 # everything, downloading if needed
    python run_pipeline.py --from train    # re-run from a given stage onwards
    python run_pipeline.py --only report   # one stage
    python run_pipeline.py --with-extras   # also run the baselines and ablations

Stages:

    download     fetch the recorded failure cases
    features     1 Hz telemetry -> causal per-window features
    labels       ground-truth labels, masked policy
    labels-alt   ground-truth labels, propagation-negative policy (primary)
    propagation  measure how far faults actually spread
    train        fit and evaluate the failure-risk model
    score        compute the reliability scores
    report       figures and RESULTS.md

The primary results use the propagation-negative labelling; `reports/
PROPAGATION.md` explains why, and `reports/BASELINES.md` reports both.
"""
from __future__ import annotations

import argparse
import sys
import time

PROPNEG = "data/processed/labelled_propneg.parquet"

# stage name -> (module, default extra arguments)
STAGES: dict[str, tuple[str, list[str]]] = {
    "download": ("download", []),
    "features": ("features", []),
    "labels": ("labels", []),
    "labels-alt": ("labels", ["--propagation-negative", "--out", PROPNEG]),
    "propagation": ("propagation", []),
    "train": ("train", ["--data", PROPNEG]),
    "score": ("score", ["--data", PROPNEG]),
    "report": ("report", []),
}
EXTRAS: dict[str, tuple[str, list[str]]] = {
    "baselines": ("baselines", []),
    "ablation": ("ablation", []),
}
ORDER = list(STAGES)


def run(stage: str, module_name: str, extra: list[str]) -> None:
    module = __import__("src." + module_name, fromlist=["main"])
    headline = (module.__doc__ or module_name).strip().splitlines()[0]
    print("\n" + "=" * 72)
    print("[{}] {}".format(stage, headline))
    print("=" * 72, flush=True)
    started = time.time()
    code = module.main(extra)
    if code:
        raise SystemExit("stage '{}' failed with code {}".format(stage, code))
    print("[{}] finished in {:.0f}s".format(stage, time.time() - started),
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", choices=ORDER, default="download")
    p.add_argument("--only", choices=ORDER + list(EXTRAS), default=None)
    p.add_argument("--with-extras", action="store_true",
                   help="also run the baselines and ablations (slow)")
    p.add_argument("--per-group", type=int, default=None,
                   help="download only N runs per fault group (a quick subset)")
    args = p.parse_args()

    if args.only:
        table = {**STAGES, **EXTRAS}
        plan = [(args.only,) + table[args.only]]
    else:
        plan = [(s,) + STAGES[s] for s in ORDER[ORDER.index(args.start):]]
        if args.with_extras:
            plan += [(s,) + EXTRAS[s] for s in EXTRAS]

    for stage, module_name, extra in plan:
        if stage == "download" and args.per_group:
            extra = extra + ["--per-group", str(args.per_group)]
        run(stage, module_name, extra)

    print("\nAll done. Launch the dashboard with:\n"
          "    streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
