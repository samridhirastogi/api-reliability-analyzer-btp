"""Validate the reliability score's weights instead of asserting them.

The proposal fixes the weights at 0.40 / 0.35 / 0.25 by judgement. A composite
score built that way is only defensible if two questions are answered: what
*else* the weights could have been, and whether the conclusions survive a
different choice. The OECD/JRC *Handbook on Constructing Composite Indicators*
(2008) sets this out as standard practice -- state the weighting scheme, then
run an uncertainty and sensitivity analysis over it -- and that is what this
module does.

Four alternative schemes are compared with the proposal's:

* **equal** -- one third each, the most common choice in practice and the
  honest null hypothesis.
* **entropy** -- Shannon entropy weighting, which gives more weight to the
  component that varies more across windows. Objective, data-driven, and
  criticised for over-weighting indicators that vary for uninteresting reasons.
* **discriminative** -- weights proportional to how far each component
  separates failing windows from healthy ones, in units of the healthy
  standard deviation.
* **learned** -- a logistic regression fitted on the three components, with its
  coefficients normalised into weights.

Then an uncertainty analysis: several hundred weight vectors are drawn from a
flat Dirichlet distribution, and the score is recomputed for each, to see how
much the result actually depends on the choice.

**A caution that belongs with the result.** Choosing weights to maximise
ranking accuracy turns the composite score into a second classifier, which is
not what it is for -- the model already does that job better. The weights exist
to make a score an operator can read and decompose. So the question this module
answers is not "which weights are best" but "does the choice of weights change
the answer", which is what the Handbook's sensitivity analysis is for.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import (DATA_PROCESSED, RANDOM_STATE, REPORTS, W_CONSISTENCY,
                     W_ERROR_FREQ, W_FAILURE_RISK)

TERMS = ["risk_term", "error_term", "consistency_term"]
PRETTY = {"risk_term": "failure risk", "error_term": "error frequency",
          "consistency_term": "response consistency"}
PROPOSAL = np.array([W_FAILURE_RISK, W_ERROR_FREQ, W_CONSISTENCY])


def terms_matrix(scored: pd.DataFrame) -> pd.DataFrame:
    """The three components, each oriented so that higher means more reliable."""
    out = pd.DataFrame(index=scored.index)
    out["risk_term"] = 1.0 - scored.failure_risk
    out["error_term"] = 1.0 - scored.error_frequency.fillna(0.0)
    out["consistency_term"] = scored.response_consistency
    return out.clip(0.0, 1.0)


def entropy_weights(X: pd.DataFrame) -> np.ndarray:
    """Shannon entropy weighting: more weight where the indicator varies more."""
    P = np.clip(X.to_numpy(dtype="float64"), 1e-12, None)
    P = P / P.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(P))
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1.0 - e
    return d / d.sum()


def discriminative_weights(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Weights proportional to each component's class separation."""
    sep = []
    for c in X.columns:
        h = X.loc[y == 0, c]
        f = X.loc[y == 1, c]
        sd = h.std()
        sep.append(abs(f.mean() - h.mean()) / sd if sd > 0 else 0.0)
    sep = np.asarray(sep)
    return sep / sep.sum() if sep.sum() > 0 else np.full(len(sep), 1 / len(sep))


def learned_weights(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Normalised magnitudes of a logistic regression on the three components."""
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=RANDOM_STATE).fit(X, y)
    coef = np.abs(lr.coef_[0])
    return coef / coef.sum() if coef.sum() > 0 else np.full(X.shape[1], 1 / 3)


def evaluate(X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> dict:
    s = X.to_numpy() @ w
    return {
        "pr_auc": float(average_precision_score(y, -s)),
        "roc_auc": float(roc_auc_score(y, -s)),
    }


def rank_agreement(X: pd.DataFrame, keys: pd.DataFrame, w_a, w_b,
                   max_groups: int = 3000) -> float:
    """How similarly two weightings rank the APIs *within the same window*.

    This is the operational question: an on-call engineer reads the ordering of
    APIs at one moment, not the absolute score.
    """
    sa = X.to_numpy() @ w_a
    sb = X.to_numpy() @ w_b
    df = keys.copy()
    df["a"], df["b"] = sa, sb
    rhos = []
    for _k, g in df.groupby(["case", "window_end"], sort=False):
        if len(g) < 3:
            continue
        rho = spearmanr(g.a, g.b).statistic
        if np.isfinite(rho):
            rhos.append(rho)
        if len(rhos) >= max_groups:
            break
    return float(np.mean(rhos)) if rhos else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_PROCESSED / "scored.parquet"))
    p.add_argument("--draws", type=int, default=500)
    p.add_argument("--sample", type=int, default=120000)
    args = p.parse_args(argv)

    scored = pd.read_parquet(args.data)
    scored = scored[scored.usable & ~scored.is_infra].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE)
    if len(scored) > args.sample:
        idx = rng.choice(len(scored), args.sample, replace=False)
        scored = scored.iloc[np.sort(idx)].reset_index(drop=True)

    X = terms_matrix(scored)
    y = scored.label.to_numpy()
    keys = scored[["case", "window_end"]]
    print("windows {:,} | positives {:.2%}".format(len(X), y.mean()))

    schemes = {
        "proposal (0.40/0.35/0.25)": PROPOSAL,
        "equal (1/3 each)": np.full(3, 1 / 3),
        "entropy": entropy_weights(X),
        "discriminative": discriminative_weights(X, y),
        "learned (logistic)": learned_weights(X, y),
    }
    results = {}
    for name, w in schemes.items():
        r = evaluate(X, y, w)
        r["weights"] = {PRETTY[c]: float(v) for c, v in zip(TERMS, w)}
        r["rank_agreement_with_proposal"] = (
            1.0 if name.startswith("proposal")
            else rank_agreement(X, keys, PROPOSAL, w))
        results[name] = r
        print("  {:<26} pr_auc={:.3f}  weights=[{}]  rank_rho={:.3f}".format(
            name, r["pr_auc"],
            " ".join("{:.2f}".format(v) for v in w),
            r["rank_agreement_with_proposal"]), flush=True)

    # ---- uncertainty analysis: how much does the choice matter at all?
    print("\nuncertainty analysis over {} random weightings".format(args.draws))
    draws = rng.dirichlet(np.ones(3), size=args.draws)
    pr = np.array([average_precision_score(y, -(X.to_numpy() @ w)) for w in draws])
    prop = results["proposal (0.40/0.35/0.25)"]["pr_auc"]
    mc = {
        "draws": int(args.draws),
        "pr_auc_min": float(pr.min()), "pr_auc_max": float(pr.max()),
        "pr_auc_mean": float(pr.mean()), "pr_auc_std": float(pr.std()),
        "pr_auc_p05": float(np.percentile(pr, 5)),
        "pr_auc_p95": float(np.percentile(pr, 95)),
        "proposal_pr_auc": float(prop),
        "proposal_percentile": float((pr < prop).mean()),
    }
    print("  PR-AUC across random weightings: {:.3f} - {:.3f} "
          "(mean {:.3f}, sd {:.3f})".format(
              mc["pr_auc_min"], mc["pr_auc_max"], mc["pr_auc_mean"],
              mc["pr_auc_std"]))
    print("  the proposal's weights sit at the {:.0%} percentile".format(
        mc["proposal_percentile"]))

    write_markdown(results, mc)
    (REPORTS / "weights.json").write_text(
        json.dumps({"schemes": results, "uncertainty": mc}, indent=2))
    return 0


def write_markdown(results: dict, mc: dict):
    prop = results["proposal (0.40/0.35/0.25)"]
    best = max(results.items(), key=lambda kv: kv[1]["pr_auc"])
    lines = [
        "# Are the score's weights defensible?",
        "",
        "Generated by `python -m src.weights`. The proposal fixes the weights "
        "at 0.40 / 0.35 / 0.25 by judgement. This is the check that they are "
        "not arbitrary, following the OECD/JRC *Handbook on Constructing "
        "Composite Indicators* (2008), which treats a stated weighting scheme "
        "plus an uncertainty and sensitivity analysis as the minimum standard "
        "for a composite index.",
        "",
        "## Alternative weighting schemes",
        "",
        "| Scheme | Failure risk | Error freq | Consistency | PR-AUC | Rank agreement |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        w = r["weights"]
        lines.append("| {} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} |".format(
            name, w["failure risk"], w["error frequency"],
            w["response consistency"], r["pr_auc"],
            r["rank_agreement_with_proposal"]))
    lines += [
        "",
        "*Rank agreement* is the mean Spearman correlation between that "
        "scheme's ordering of APIs and the proposal's, computed within the "
        "same window -- the operational question, since an engineer reads the "
        "ordering of APIs at one moment rather than the absolute number.",
        "",
        "## What the data says the weights should be",
        "",
        "Every data-driven scheme concentrates weight on the failure-risk term "
        "and all but removes the other two. That is consistent with the "
        "component separation measured in `ABLATIONS.md`: failure risk "
        "separates failing from healthy windows by about 25 standard "
        "deviations, error frequency by 3.4, and response consistency by 0.44. "
        "Weighting a 0.44-sigma signal at 0.25 alongside a 25-sigma signal at "
        "0.40 is what makes the composite rank slightly worse than the risk "
        "term alone.",
        "",
        "## Uncertainty analysis",
        "",
        "{} weight vectors were drawn from a flat Dirichlet distribution and "
        "the score recomputed for each.".format(mc["draws"]),
        "",
        "| | PR-AUC |",
        "|---|---|",
        "| Worst random weighting | {:.3f} |".format(mc["pr_auc_min"]),
        "| 5th percentile | {:.3f} |".format(mc["pr_auc_p05"]),
        "| Mean | {:.3f} |".format(mc["pr_auc_mean"]),
        "| 95th percentile | {:.3f} |".format(mc["pr_auc_p95"]),
        "| Best random weighting | {:.3f} |".format(mc["pr_auc_max"]),
        "| **The proposal's weights** | **{:.3f}** ({:.0%} percentile) |".format(
            mc["proposal_pr_auc"], mc["proposal_percentile"]),
        "",
    ]

    spread = mc["pr_auc_max"] - mc["pr_auc_min"]
    lines += [
        "The spread across *every* weighting from pure-risk to pure-consistency "
        "is {:.3f} PR-AUC. ".format(spread) +
        ("That is wide, so the choice of weights materially changes the result "
         "and must be justified rather than assumed."
         if spread > 0.15 else
         "The result is therefore only weakly sensitive to the weights over "
         "much of the simplex, though the extremes do differ."),
        "",
        "## Conclusion",
        "",
        "The proposal's weights sit at the {:.0%} percentile of random "
        "weightings, so they are a reasonable rather than an unlucky choice, "
        "but they are **not** the accuracy-maximising ones -- {} reaches "
        "{:.3f}. ".format(
            mc["proposal_percentile"], best[0], best[1]["pr_auc"]) +
        "Re-weighting towards the failure-risk term would improve ranking.",
        "",
        "That improvement should be declined, and the reason is the point of "
        "this analysis. Fitting weights to maximise ranking accuracy turns the "
        "composite into a second, worse classifier -- the model already "
        "performs that task, and performs it better. The score's purpose is to "
        "give an operator a number they can decompose into *why*: how much of "
        "this API's poor score is predicted risk, how much is errors it is "
        "actually returning, how much is unstable latency. The honest position "
        "is that 0.40 / 0.35 / 0.25 is a **presentation** choice, now measured "
        "rather than assumed, with the cost of that choice quantified here and "
        "in `ABLATIONS.md`.",
        "",
        "A project that needed the weights to be optimal rather than "
        "interpretable should use the `learned` row, and say so.",
        "",
    ]
    path = REPORTS / "WEIGHTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\nwritten ->", path)


if __name__ == "__main__":
    raise SystemExit(main())
