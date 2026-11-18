"""CLI: run the shared Group-CV (and optional student_test hold-out) for any model.

  python -m src.evaluation.run_cv --model boosting --backend lightgbm --holdout
  python -m src.evaluation.run_cv --model hgb --holdout
  # any teammate's model (factory returns an object with fit / predict_proba / needs_scaling):
  python -m src.evaluation.run_cv --name random_forest --factory src.models.random_forest.rf_model:make_model --holdout
"""
import argparse
import importlib

from .base import load_data
from .cv_harness import evaluate_holdout, run_group_cv


def resolve_factory(args):
    if args.factory:
        mod, fn = args.factory.split(":")
        f = getattr(importlib.import_module(mod), fn)
        return args.name or fn, (lambda: f())
    from src.models.boosting.boosting_model import make_model
    backend = "hgb" if args.model == "hgb" else args.backend
    return args.name or f"boosting_{backend}", (lambda: make_model(backend=backend))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boosting", choices=["boosting", "hgb"])
    ap.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "hgb"])
    ap.add_argument("--factory", default=None, help="module:callable returning an unfitted model")
    ap.add_argument("--name", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trip-subset", type=float, default=None)
    ap.add_argument("--holdout", action="store_true", help="also fit on all train, score student_test")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    train_df, test_df, feature_cols = load_data()
    name, factory = resolve_factory(args)
    run_group_cv(name, factory, train_df, feature_cols, n_splits=args.folds,
                 trip_subset=args.trip_subset, log=not args.no_log)
    if args.holdout:
        evaluate_holdout(name, factory, train_df, test_df, feature_cols, log=not args.no_log)


if __name__ == "__main__":
    main()
