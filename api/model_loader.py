"""
model_loader.py — Loads a joblib-serialised scikit-learn model and exposes
a predict() helper that works with a Pandas DataFrame of feature columns.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 16 feature columns the model was trained on, in order.
FEATURE_COLUMNS = [
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


def load_model(path: str) -> Tuple[Optional[object], str]:
    """
    Load a joblib model from *path*.

    Returns
    -------
    (model, model_name)
        model      — the loaded sklearn estimator, or None if file not found.
        model_name — base filename without extension, or "none".
    """
    model_path = Path(path)
    if not model_path.exists():
        logger.warning("Model file not found: %s — running in no-model mode.", path)
        return None, "none"

    try:
        model = joblib.load(model_path)
        model_name = model_path.stem  # e.g. "rf_model" or "et_model"
        logger.info("Loaded model '%s' from %s", model_name, path)
        return model, model_name
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model from %s: %s", path, exc)
        return None, "none"


def predict(model, feature_df: pd.DataFrame) -> np.ndarray:
    """
    Run predict_proba on *feature_df* (must contain FEATURE_COLUMNS).

    Returns
    -------
    np.ndarray of shape (n_rows,)
        Positive-class probability for each candidate row.
    """
    X = feature_df[FEATURE_COLUMNS].values
    proba = model.predict_proba(X)  # shape (n_rows, 2)
    return proba[:, 1]  # positive-class scores


def auto_load_model(artifacts_dir: str = "artifacts") -> Tuple[Optional[object], str]:
    """
    Try to load et_model.joblib first (usually better), fall back to rf_model.joblib.
    Returns (model, model_name).
    """
    for name in ("et_model.joblib", "rf_model.joblib"):
        candidate = os.path.join(artifacts_dir, name)
        model, model_name = load_model(candidate)
        if model is not None:
            return model, model_name
    return None, "none"
