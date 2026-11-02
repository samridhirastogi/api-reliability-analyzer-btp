"""Train and evaluate the failure-risk model.

The model answers one question: *given the last 30 s of this API's telemetry,
what is the probability it is in a failing state over the next 30 s?* Its
calibrated output becomes the ``Failure Risk`` term of the reliability score,
which is why calibration matters here -- an uncalibrated forest vote share
would make the 0.40 weight meaningless.

Evaluation is always **grouped by case**: all windows of one recorded failure
case sit on the same side of every split, so the model is never scored on
windows from a case it has already seen.
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .config import (DATA_PROCESSED, HORIZON_S, MODELS, N_ESTIMATORS, N_SPLITS,
                     RANDOM_STATE, REPORTS, STRIDE_S, WINDOW_S)
from .features import FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=UserWarning)


def make_model(n_estimators: int = N_ESTIMATORS) -> Pipeline:
    return Pipeline([
        # Some services expose no error-rate metric at all. The indicator
        # columns keep "not measured" distinguishable from "measured as zero".
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def calibrate(base, X_cal, y_cal):
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    except ImportError:  # pragma: no cover - older scikit-learn
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def best_threshold(y_true, prob):
    prec, rec, thr = precision_recall_curve(y_true, prob)
    if not len(thr):
        return 0.5
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def scores(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def detection_delay(df, threshold):
    """How long after the real injection the model first flags the API."""
    delays, missed = [], 0
    for _case, g in df[df.is_target].groupby("case"):
        after = g[g.lead_s >= 0].sort_values("lead_s")
        if not len(after):
            continue
        hit = after[after.prob >= threshold]
        if len(hit):
            delays.append(float(hit.lead_s.iloc[0]))
        else:
            missed += 1
    total = len(delays) + missed
    return {
        "cases_detected": len(delays),
        "cases_missed": missed,
        "detection_rate": len(delays) / max(total, 1),
        "median_delay_s": float(np.median(delays)) if delays else None,
        "p90_delay_s": float(np.percentile(delays, 90)) if delays else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_PROCESSED / "labelled.parquet"))
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--skip-permutation", action="store_true")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.data)
    df = df[df.usable & ~df.is_infra].reset_index(drop=True)
    X_all = df[FEATURE_COLUMNS]
    y_all = df.label.values
    g_all = df.case.values
    print("usable windows {:,} | positives {:.2%} | cases {} | services {}".format(
        len(df), y_all.mean(), df.case.nunique(), df.service.nunique()))

    # ---------------------------------------------------- held-out case split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                            random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X_all, y_all, g_all))
    sub = df.iloc[tr_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25,
                             random_state=RANDOM_STATE)
    fit_rel, cal_rel = next(gss2.split(sub, sub.label, sub.case))
    fit_idx, cal_idx = tr_idx[fit_rel], tr_idx[cal_rel]

    model = make_model().fit(X_all.iloc[fit_idx], y_all[fit_idx])
    calibrated = calibrate(model, X_all.iloc[cal_idx], y_all[cal_idx])

    prob_cal = calibrated.predict_proba(X_all.iloc[cal_idx])[:, 1]
    thr = best_threshold(y_all[cal_idx], prob_cal)
    prob_te = calibrated.predict_proba(X_all.iloc[te_idx])[:, 1]

    results = {
        "dataset": {
            "windows": int(len(df)),
            "cases": int(df.case.nunique()),
            "services": int(df.service.nunique()),
            "positive_rate": float(y_all.mean()),
            "window_s": WINDOW_S, "stride_s": STRIDE_S, "horizon_s": HORIZON_S,
        },
        "split": {
            "fit_cases": sorted(set(df.case.iloc[fit_idx])),
            "cal_cases": sorted(set(df.case.iloc[cal_idx])),
            "test_cases": sorted(set(df.case.iloc[te_idx])),
        },
        "holdout": scores(y_all[te_idx], prob_te, thr),
        "brier_before_calibration": float(brier_score_loss(
            y_all[cal_idx], model.predict_proba(X_all.iloc[cal_idx])[:, 1])),
        "brier_after_calibration": float(brier_score_loss(y_all[cal_idx], prob_cal)),
    }

    te = df.iloc[te_idx].copy()
    te["prob"] = prob_te
    # kept so the report can show what calibration changed
    te["prob_uncalibrated"] = model.predict_proba(X_all.iloc[te_idx])[:, 1]
    results["detection"] = detection_delay(te, thr)
    te.to_parquet(DATA_PROCESSED / "test_predictions.parquet", index=False)
    print("held-out :", json.dumps(results["holdout"]))
    print("detection:", json.dumps(results["detection"]))

    # ------------------------------------------- grouped cross-validation
    cv_rows = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X_all, y_all, g_all)):
        m = make_model().fit(X_all.iloc[a], y_all[a])
        pr = m.predict_proba(X_all.iloc[b])[:, 1]
        cv_rows.append(dict(scores(y_all[b], pr, thr), fold=k))
        print("  cv fold {}: pr_auc={:.3f} f1={:.3f}".format(
            k, cv_rows[-1]["pr_auc"], cv_rows[-1]["f1"]))
    cv = pd.DataFrame(cv_rows)
    results["cross_validation"] = {
        "folds": cv_rows,
        "pr_auc_mean": float(cv.pr_auc.mean()),
        "pr_auc_std": float(cv.pr_auc.std()),
        "roc_auc_mean": float(cv.roc_auc.mean()),
        "f1_mean": float(cv.f1.mean()),
    }

    # -------------------------------- generalisation to an unseen system
    results["leave_one_system_out"] = {}
    for sysid in sorted(df.system.unique()):
        mask = (df.system != sysid).values
        if mask.all() or not mask.any():
            continue
        m = make_model().fit(X_all[mask], y_all[mask])
        pr = m.predict_proba(X_all[~mask])[:, 1]
        results["leave_one_system_out"][sysid] = scores(y_all[~mask], pr, thr)
        print("  unseen system {}: pr_auc={:.3f}".format(
            sysid, results["leave_one_system_out"][sysid]["pr_auc"]))

    # ---------------------------- generalisation to an unseen fault type
    results["leave_one_fault_out"] = {}
    for fault in sorted(df.fault.unique()):
        mask = (df.fault != fault).values
        if (~mask).sum() < 200 or y_all[~mask].sum() < 20:
            continue
        m = make_model().fit(X_all[mask], y_all[mask])
        pr = m.predict_proba(X_all[~mask])[:, 1]
        results["leave_one_fault_out"][fault] = scores(y_all[~mask], pr, thr)
        print("  unseen fault {}: pr_auc={:.3f}".format(
            fault, results["leave_one_fault_out"][fault]["pr_auc"]))

    # ------------------------------------------------------- importances
    rf = model.named_steps["rf"]
    names = list(model.named_steps["impute"].get_feature_names_out(FEATURE_COLUMNS))
    imp = pd.DataFrame({"feature": names, "impurity": rf.feature_importances_})
    if not args.skip_permutation:
        rng = np.random.default_rng(RANDOM_STATE)
        sample = (te_idx if len(te_idx) <= 20000
                  else rng.choice(te_idx, 20000, replace=False))
        perm = permutation_importance(
            calibrated, X_all.iloc[sample], y_all[sample], n_repeats=5,
            scoring="average_precision", random_state=RANDOM_STATE, n_jobs=-1)
        imp = imp.merge(pd.DataFrame({"feature": FEATURE_COLUMNS,
                                      "permutation": perm.importances_mean}),
                        on="feature", how="left")
    imp = imp.sort_values("impurity", ascending=False)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    print("top features:", ", ".join(imp.feature.head(8)))

    joblib.dump({"model": calibrated, "threshold": thr,
                 "features": FEATURE_COLUMNS}, MODELS / "failure_risk.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(results, indent=2))
    print("model ->", MODELS / "failure_risk.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
