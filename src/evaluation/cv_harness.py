"""Shared Group-Aware cross-validation harness (GroupKFold by trip_id).

Why group by trip: GPS points inside a trip are strongly autocorrelated, so a
random row split would leak near-duplicate rows into validation and inflate
scores. All 10 candidate rows of one observation, and all observations of one
trip, always sit in the same fold.

Usage (any member):
    from src.evaluation.cv_harness import run_group_cv, evaluate_holdout
    run_group_cv("random_forest", lambda: MyModel(), train_df, feature_cols)
"""
import csv
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .base import META_COLS, METRICS_CSV, positive_proba
from .metrics import METRIC_COLS, compute_metrics

LOG_FIELDS = (["model", "split", "n_splits", "seed", "fit_seconds"] + METRIC_COLS
              + ["top1_acc_std", "macro_f1_std", "log_loss_std", "continuity_std", "params", "timestamp"])


def log_row(row: dict, path: str = METRICS_CSV):
    """Append one result row to the shared metrics CSV (creates it if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    row = {**{k: "" for k in LOG_FIELDS}, **row, "timestamp": datetime.now().isoformat(timespec="seconds")}
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _prep(model, X_tr, X_va):
    """Scale on the training fold only when the model asks for it."""
    if getattr(model, "needs_scaling", False):
        sc = StandardScaler().fit(X_tr)
        return sc.transform(X_tr), sc.transform(X_va)
    return X_tr, X_va


def _fit_eval(factory, tr, va, feature_cols):
    model = factory()
    X_tr = tr[feature_cols].values.astype(np.float32)
    X_va = va[feature_cols].values.astype(np.float32)
    X_tr, X_va = _prep(model, X_tr, X_va)
    t0 = time.perf_counter()
    model.fit(X_tr, tr["label"].values)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = positive_proba(model.predict_proba(X_va))
    infer_s = time.perf_counter() - t0
    m = compute_metrics(va[META_COLS], va["label"].values, proba, infer_s)
    m["fit_seconds"] = fit_s
    return m, model


def get_folds(df, n_splits=5, trip_subset=None, seed=42):
    """Deterministic GroupKFold splits (same folds for every model => fair comparison).
    trip_subset: optional fraction (0-1) of trips to use (for fast Bayesian tuning)."""
    trips = np.array(sorted(df["trip_id"].unique()))
    if trip_subset and trip_subset < 1.0:
        rng = np.random.RandomState(seed)
        trips = np.sort(rng.choice(trips, size=max(n_splits, int(len(trips) * trip_subset)), replace=False))
        df = df[df["trip_id"].isin(trips)]
    gkf = GroupKFold(n_splits=n_splits)
    return df, list(gkf.split(df, groups=df["trip_id"].values))


def run_group_cv(model_name, factory, train_df, feature_cols, n_splits=5, trip_subset=None,
                 seed=42, params=None, log=True, verbose=True, metrics_path=METRICS_CSV):
    """Group-aware CV. Returns dict with per-fold metrics and mean/std summary."""
    df, folds = get_folds(train_df, n_splits, trip_subset, seed)
    fold_metrics = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        m, _ = _fit_eval(factory, tr, va, feature_cols)
        m["fold"] = i
        fold_metrics.append(m)
        if verbose:
            print(f"[{model_name}] fold {i+1}/{n_splits}  top1={m['top1_acc']:.4f}  macroF1={m['macro_f1']:.4f}  "
                  f"logloss={m['log_loss']:.4f}  cont={m['continuity']:.4f}  fit={m['fit_seconds']:.1f}s")
        if log:
            log_row({"model": model_name, "split": f"cv_fold_{i}", "n_splits": n_splits, "seed": seed,
                     "params": str(params or ""), **m}, metrics_path)
    fm = pd.DataFrame(fold_metrics)
    mean, std = fm.mean(numeric_only=True), fm.std(numeric_only=True, ddof=0)
    summary = {k: float(mean[k]) for k in METRIC_COLS + ["fit_seconds"]}
    for k in ["top1_acc", "macro_f1", "log_loss", "continuity"]:
        summary[k + "_std"] = float(std[k])
    if log:
        log_row({"model": model_name, "split": "cv_mean", "n_splits": n_splits, "seed": seed,
                 "params": str(params or ""), **summary}, metrics_path)
    if verbose:
        print(f"[{model_name}] CV mean: top1={summary['top1_acc']:.4f}±{summary['top1_acc_std']:.4f} "
              f"macroF1={summary['macro_f1']:.4f} logloss={summary['log_loss']:.4f} cont={summary['continuity']:.4f}")
    return {"folds": fold_metrics, "summary": summary}


def evaluate_holdout(model_name, factory, train_df, test_df, feature_cols, seed=42, params=None,
                     log=True, verbose=True, metrics_path=METRICS_CSV):
    """Fit on ALL 80 train trips, evaluate on the 10 student_test trips."""
    m, model = _fit_eval(factory, train_df, test_df, feature_cols)
    if verbose:
        print(f"[{model_name}] student_test: top1={m['top1_acc']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f} cont={m['continuity']:.4f} fit={m['fit_seconds']:.1f}s")
    if log:
        log_row({"model": model_name, "split": "student_test", "seed": seed, "params": str(params or ""), **m}, metrics_path)
    return m, model
