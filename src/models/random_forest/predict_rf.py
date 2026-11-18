"""
predict_rf.py — PathSnap ML Hackathon: Model 3 (Random Forest / Extra Trees) Inference

Loads the trained Random Forest or Extra Trees model from artifacts and provides
convenience functions for batch inference and candidate-level map matching.
"""

import os
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RF_MODEL_PATH = os.path.join(REPO_ROOT, "artifacts", "rf_model.joblib")
ET_MODEL_PATH = os.path.join(REPO_ROOT, "artifacts", "et_model.joblib")
BEST_MODEL_TXT = os.path.join(REPO_ROOT, "artifacts", "best_rf_model.txt")

FEATURE_COLS = [
    "dist_to_arc_m",
    "dist_min_diff_m",
    "dist_ratio_to_min",
    "cand_rank",
    "proj_fraction",
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "arc_length_m",
    "heading_diff_rad",
    "heading_cos",
    "heading_sin",
    "gps_speed_mps",
    "dt_prev_s",
    "prev_gps_dist_to_arc_m",
    "next_gps_dist_to_arc_m",
    "is_connected_from_prev_top1",
]


def load_best_model():
    """Load the best available tree model from artifacts."""
    if os.path.exists(BEST_MODEL_TXT):
        with open(BEST_MODEL_TXT, "r") as f:
            path = f.read().strip()
        if os.path.exists(path):
            return joblib.load(path), os.path.basename(path)

    for p in [RF_MODEL_PATH, ET_MODEL_PATH]:
        if os.path.exists(p):
            return joblib.load(p), os.path.basename(p)

    raise FileNotFoundError("No trained tree model found in artifacts directory.")


def predict_proba(features_df: pd.DataFrame, model=None) -> np.ndarray:
    """
    Given a dataframe with the 16 feature columns, returns class probabilities:
    shape (N, 2), where column 1 is P(label=1).
    """
    if model is None:
        model, _ = load_best_model()

    X = features_df[FEATURE_COLS].values
    return model.predict_proba(X)


def get_top1_arc(candidates_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Given candidate rows for one or multiple trips, scores all candidates and
    returns the predicted arc for each (trip_id, obs_idx) observation.
    
    Returns DataFrame with columns:
        trip_id, obs_idx, predicted_arc_id, confidence, from_node, to_node
    """
    df = candidates_df.copy()
    probs = predict_proba(df, model)[:, 1]
    df["score"] = probs

    # For each observation group, take candidate with maximum score
    best_idx = df.groupby(["trip_id", "obs_idx"])["score"].idxmax()
    best_rows = df.loc[best_idx].copy()

    rename_cols = {"candidate_arc_id": "predicted_arc_id", "score": "confidence"}
    best_rows = best_rows.rename(columns=rename_cols)

    cols_to_keep = [c for c in ["trip_id", "obs_idx", "timestamp", "predicted_arc_id", "confidence", "from_node", "to_node", "true_arc_id", "label"] if c in best_rows.columns]
    return best_rows[cols_to_keep].sort_values(by=["trip_id", "obs_idx"]).reset_index(drop=True)
