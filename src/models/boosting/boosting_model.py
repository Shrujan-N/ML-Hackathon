"""Model 4: Gradient Boosting (LightGBM / XGBoost / HistGradientBoosting).

Trees need no feature scaling (needs_scaling=False), so raw features are used.
Plugs into the shared harness via the predict_proba(X) contract.
"""
import os

import joblib
import numpy as np

DEFAULTS = {
    "lightgbm": dict(n_estimators=300, learning_rate=0.1, num_leaves=63, max_depth=-1,
                     min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     reg_alpha=0.0, reg_lambda=1.0),
    "xgboost": dict(n_estimators=300, learning_rate=0.1, max_depth=8, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0),
    "hgb": dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0),
}


class BoostingModel:
    needs_scaling = False

    def __init__(self, backend="lightgbm", seed=42, n_jobs=-1, **params):
        self.backend = backend
        self.seed = seed
        self.n_jobs = n_jobs
        self.params = {**DEFAULTS[backend], **params}
        self.name = f"boosting_{backend}"
        self.clf = None

    def _build(self):
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(objective="binary", random_state=self.seed, n_jobs=self.n_jobs,
                                  verbose=-1, **self.params)
        if self.backend == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(objective="binary:logistic", tree_method="hist", random_state=self.seed,
                                 n_jobs=self.n_jobs, eval_metric="logloss", **self.params)
        if self.backend == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=self.seed, **self.params)
        raise ValueError(f"unknown backend {self.backend}")

    def fit(self, X, y):
        self.clf = self._build().fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def feature_importances(self):
        return getattr(self.clf, "feature_importances_", None)


def make_model(backend="lightgbm", **params):
    """Factory used by the harness: run_group_cv('boosting', lambda: make_model(...), ...)"""
    return BoostingModel(backend=backend, **params)


# ---- Final-model bundle (consumed by Member 4's FastAPI on Laptop 1) ----------
def save_bundle(model, feature_cols, path, scaler=None, meta=None):
    joblib.dump({"model": model, "feature_cols": list(feature_cols), "needs_scaling": model.needs_scaling,
                 "scaler": scaler, "meta": meta or {}}, path, compress=3)


def load_bundle(path):
    return joblib.load(path)


def bundle_predict_proba(bundle, df):
    """df must contain bundle['feature_cols']. Returns 1-D P(label=1), same row order."""
    X = df[bundle["feature_cols"]].values.astype(np.float32)
    if bundle["needs_scaling"] and bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)
    p = bundle["model"].predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p
