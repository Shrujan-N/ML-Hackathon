import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
    model_path = Path(path)
    if not model_path.exists():
        return None, "none"
    try:
        model = joblib.load(model_path)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        elif isinstance(model, dict) and "model" in model:
            inner = model["model"]
            if hasattr(inner, "n_jobs"):
                inner.n_jobs = 1
            if hasattr(inner, "clf") and hasattr(inner.clf, "n_jobs"):
                inner.clf.n_jobs = 1
        model_name = model_path.stem
        logger.info("Loaded model '%s' from %s", model_name, path)
        return model, model_name
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", path, exc)
        return None, "none"


def predict(model, feature_df: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict) and "model" in model and "feature_cols" in model:
        X = feature_df[model["feature_cols"]].values.astype(np.float32)
        if model.get("needs_scaling") and model.get("scaler") is not None:
            X = model["scaler"].transform(X)
        proba = model["model"].predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba.ravel()

    X = feature_df[FEATURE_COLUMNS].values
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 else proba.ravel()


def auto_load_model(artifacts_dir: str = "artifacts") -> Tuple[Optional[object], str]:
    for name in ("final_model.joblib", "et_model.joblib", "rf_model.joblib"):
        candidate = os.path.join(artifacts_dir, name)
        model, model_name = load_model(candidate)
        if model is not None:
            return model, model_name
    return None, "none"
