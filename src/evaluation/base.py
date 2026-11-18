"""Shared model interface + data loading helpers (contract agreed at Hour 0).

Every model (Members 1-4) must expose:
    fit(X, y)            -> self
    predict_proba(X)     -> array (n,) of P(label=1)  OR  (n, 2) sklearn style
    needs_scaling: bool  -> True for Logistic Regression / KNN / SVM / MLP.
                            The harness then fits a StandardScaler on the
                            *training fold only* and transforms X for you.

Models are passed to the harness as a *factory* (a zero-argument callable that
returns a fresh, unfitted model), so every fold starts from scratch.
"""
import json
import os

import joblib
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCHEMA_PATH = os.path.join(REPO_ROOT, "artifacts", "feature_schema.json")
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train_candidates.pkl")
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "student_test_candidates.pkl")
RESULTS_DIR = os.path.join(REPO_ROOT, "outputs", "evaluation")
METRICS_CSV = os.path.join(RESULTS_DIR, "model_metrics.csv")

META_COLS = ["trip_id", "obs_idx", "candidate_arc_id", "from_node", "to_node", "true_arc_id"]


class BaseCandidateModel:
    """Optional base class; any object with the same methods also works."""

    name = "base"
    needs_scaling = False

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


def positive_proba(p):
    """Normalise predict_proba output to a 1-D P(label=1) vector."""
    p = np.asarray(p)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] == 2 else p.ravel()
    return p.astype(np.float64)


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """Returns (train_df, test_df, feature_cols). Rows are sorted by
    (trip_id, obs_idx) so per-trip sequence metrics (continuity) are valid."""
    schema = load_schema()
    train_df = joblib.load(train_path).sort_values(["trip_id", "obs_idx"], kind="stable").reset_index(drop=True)
    test_df = joblib.load(test_path).sort_values(["trip_id", "obs_idx"], kind="stable").reset_index(drop=True)
    return train_df, test_df, schema["feature_columns"]
