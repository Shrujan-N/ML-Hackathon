"""Bayesian hyper-parameter optimisation (Optuna TPE) for the boosting model.

Objective : mean Group-CV (GroupKFold by trip) score on the training trips only.
            Default = maximise Top-1 accuracy; --metric log_loss minimises log loss instead.
Sampler   : TPE (a Bayesian method: models p(x|good) / p(x|bad) and proposes the next point by EI).
Pruner    : MedianPruner - stops a trial early if its running fold-average is worse than the median.
Leakage   : student_test is NEVER touched here.

  python -m src.tuning.bayes_opt --backend lightgbm --n-trials 30 --folds 3 --trip-subset 0.5
Outputs (outputs/tuning/): trials.csv, best_params.json, optimization_history.png,
                           param_importance.png, study.db (resumable)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

from src.evaluation.base import REPO_ROOT, load_data
from src.evaluation.cv_harness import get_folds, _fit_eval
from src.evaluation.metrics import HIGHER_IS_BETTER
from src.models.boosting.boosting_model import make_model

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "tuning")


def suggest_params(trial, backend):
    if backend == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if backend == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    return dict(  # hgb
        max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    )


def run_study(backend="lightgbm", n_trials=30, n_splits=3, trip_subset=0.5, metric="top1_acc", seed=42,
              timeout=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, _, feature_cols = load_data()
    df, folds = get_folds(train_df, n_splits, trip_subset, seed)
    direction = "maximize" if HIGHER_IS_BETTER[metric] else "minimize"

    def objective(trial):
        params = suggest_params(trial, backend)
        scores = []
        for i, (tr_idx, va_idx) in enumerate(folds):
            m, _ = _fit_eval(lambda: make_model(backend=backend, seed=seed, **params),
                             df.iloc[tr_idx], df.iloc[va_idx], feature_cols)
            scores.append(m[metric])
            trial.report(sum(scores) / len(scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(scores) / len(scores)

    study = optuna.create_study(
        study_name=f"boosting_{backend}_{metric}", direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        storage=f"sqlite:///{os.path.join(OUT_DIR, 'study.db')}", load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    study.trials_dataframe().to_csv(os.path.join(OUT_DIR, "trials.csv"), index=False)
    best = {"backend": backend, "metric": metric, "direction": direction, "best_value": study.best_value,
            "best_params": study.best_params, "n_trials_total": len(study.trials),
            "n_trials_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
            "n_trials_pruned": sum(t.state.name == "PRUNED" for t in study.trials),
            "cv": {"n_splits": n_splits, "trip_subset": trip_subset, "seed": seed}}
    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    make_plots(study, metric)
    print(f"Best {metric}: {study.best_value:.5f}\n{json.dumps(study.best_params, indent=2)}")
    return study


def make_plots(study, metric):
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not done:
        return
    vals = [t.value for t in done]
    best_so_far = []
    for v in vals:
        best_so_far.append(v if not best_so_far else (max if study.direction.name == "MAXIMIZE" else min)(best_so_far[-1], v))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(range(len(vals)), vals, s=28, alpha=0.7, label="Trial value")
    ax.plot(range(len(vals)), best_so_far, color="#dc2626", lw=2, label="Best so far")
    ax.set_xlabel("Completed trial"); ax.set_ylabel(f"CV {metric}")
    ax.set_title("Bayesian Optimisation (Optuna TPE) - Optimisation History")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "optimization_history.png"), dpi=180); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#2563eb")
        ax.set_xlabel("Relative importance"); ax.set_title("Hyper-parameter Importance")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "param_importance.png"), dpi=180); plt.close(fig)
    except Exception as e:  # too few trials for fANOVA
        print("param importance skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "hgb"])
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--trip-subset", type=float, default=0.5)
    ap.add_argument("--metric", default="top1_acc", choices=["top1_acc", "macro_f1", "log_loss", "continuity"])
    ap.add_argument("--timeout", type=int, default=None, help="seconds")
    a = ap.parse_args()
    run_study(a.backend, a.n_trials, a.folds, a.trip_subset, a.metric, timeout=a.timeout)
