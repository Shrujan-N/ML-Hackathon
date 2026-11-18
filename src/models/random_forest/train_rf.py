"""
train_rf.py — PathSnap ML Hackathon: Model 3 (Random Forest / Extra Trees)

Trains a RandomForestClassifier and ExtraTreesClassifier on the map-matching
candidate dataset. Each GPS observation has up to k=10 candidate arcs; the
model scores all candidates and the arc with the highest P(label=1) is selected
as the matched arc (argmax over (trip_id, obs_idx)).

Usage (from repo root):
    python src/models/random_forest/train_rf.py

TODO for final run: replace SAMPLE_CSV with the full 1.86 M-row dataset once a
full CSV export (train_candidates.csv.gz or similar) is available. The current
train_candidates.pkl has a pickle compatibility issue (StringDtype) and cannot
be loaded. The sample CSV covers only 2 trips (50 k rows) — enough for a
functional smoke-test; swap in the full data for competition-grade numbers.
"""

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, log_loss

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

SAMPLE_CSV   = os.path.join(REPO_ROOT, "data", "processed", "train_candidates_sample.csv")
FULL_CSV_GZ  = os.path.join(REPO_ROOT, "data", "processed", "train_candidates.csv.gz")
STUDENT_CSV  = os.path.join(REPO_ROOT, "data", "processed", "student_test_candidates.csv")
SCHEMA_JSON  = os.path.join(REPO_ROOT, "artifacts", "feature_schema.json")

RF_MODEL_PATH = os.path.join(REPO_ROOT, "artifacts", "rf_model.joblib")
ET_MODEL_PATH = os.path.join(REPO_ROOT, "artifacts", "et_model.joblib")
RESULTS_PATH  = os.path.join(REPO_ROOT, "artifacts", "rf_results.json")

# ---------------------------------------------------------------------------
# Feature columns (from feature_schema.json)
# ---------------------------------------------------------------------------
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
TARGET_COL = "label"

# Columns needed for evaluation (arc topology for route continuity)
META_COLS = ["trip_id", "obs_idx", "candidate_arc_id", "from_node", "to_node"]


# ===========================================================================
# Data loading
# ===========================================================================

def load_train_data() -> pd.DataFrame:
    """
    Load training data with priority:
      1. Full gzip CSV  (not yet generated — skipped if missing)
      2. Sample CSV     (always present; covers 2 trips / 50 k rows)

    Returns the dataframe with feature + meta columns.
    """
    if os.path.exists(FULL_CSV_GZ):
        print(f"[data] Loading full training data from {FULL_CSV_GZ} …")
        df = pd.read_csv(FULL_CSV_GZ, compression="gzip")
        print(f"[data] Loaded {len(df):,} rows, {df['trip_id'].nunique()} trips.")
    else:
        print(f"[data] Full CSV not found. Using sample: {SAMPLE_CSV}")
        print("[data] NOTE: sample covers only 2 trips — for final results use full data!")
        df = pd.read_csv(SAMPLE_CSV)
        print(f"[data] Loaded {len(df):,} rows, {df['trip_id'].nunique()} trips.")
    return df


# ===========================================================================
# Train / validation split  (group-aware — no trip leakage)
# ===========================================================================

def group_aware_split(df: pd.DataFrame, val_size: int = 10, random_state: int = 42):
    """
    Split by trip_id.  val_size trips go to validation, the rest to train.
    With the sample (2 trips) we use 1 val / 1 train so the script still runs.
    """
    rng = np.random.default_rng(random_state)
    all_trips = np.array(sorted(df["trip_id"].unique()))
    n_trips   = len(all_trips)

    actual_val = min(val_size, max(1, n_trips - 1))  # always keep ≥1 train trip
    shuffled   = rng.permutation(all_trips)
    val_trips  = set(shuffled[:actual_val])
    train_trips = set(shuffled[actual_val:])

    train_df = df[df["trip_id"].isin(train_trips)].copy()
    val_df   = df[df["trip_id"].isin(val_trips)].copy()

    print(f"[split] train trips: {sorted(train_trips)}  ({len(train_df):,} rows)")
    print(f"[split] val   trips: {sorted(val_trips)}   ({len(val_df):,} rows)")
    return train_df, val_df


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def top1_accuracy(df: pd.DataFrame, proba_col: str = "proba") -> float:
    """
    For each (trip_id, obs_idx) group pick the candidate with the highest
    predicted P(label=1). Return the fraction of observations where that
    candidate's label == 1.
    """
    idx = df.groupby(["trip_id", "obs_idx"])[proba_col].idxmax()
    top1_rows = df.loc[idx]
    return top1_rows[TARGET_COL].mean()


def route_continuity(df: pd.DataFrame, proba_col: str = "proba") -> float:
    """
    Route continuity: for consecutive observation pairs (obs_idx t, t+1) within
    each trip, check whether the predicted arc at t and t+1 share a node
    (i.e. to_node[t] == from_node[t+1]  OR  they are the same arc).

    Returns the fraction of consecutive pairs that are topologically connected.
    """
    # Pick top-1 predicted arc per (trip_id, obs_idx)
    idx = df.groupby(["trip_id", "obs_idx"])[proba_col].idxmax()
    top1 = df.loc[idx, ["trip_id", "obs_idx", "candidate_arc_id", "from_node", "to_node"]].copy()
    top1 = top1.sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    # Shift to get next obs within same trip
    top1["next_trip_id"]    = top1["trip_id"].shift(-1)
    top1["next_from_node"]  = top1["from_node"].shift(-1)
    top1["next_obs_idx"]    = top1["obs_idx"].shift(-1)

    # Only consecutive obs within the same trip
    consecutive = top1[
        (top1["trip_id"] == top1["next_trip_id"]) &
        (top1["next_obs_idx"] == top1["obs_idx"] + 1)
    ]

    if len(consecutive) == 0:
        return float("nan")

    # Connected = to_node of current == from_node of next
    connected = (consecutive["to_node"] == consecutive["next_from_node"]).sum()
    return connected / len(consecutive)


def evaluate(model, df: pd.DataFrame, label: str) -> dict:
    """
    Full evaluation on a candidate dataframe:
      - Top-1 Accuracy
      - Macro F1  (threshold 0.5 on per-row predictions)
      - Log Loss
      - Route Continuity
    """
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_COL].values

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    df = df.copy()
    df["proba"] = proba

    top1_acc   = top1_accuracy(df)
    macro_f1   = f1_score(y, pred, average="macro", zero_division=0)
    ll         = log_loss(y, proba)
    continuity = route_continuity(df)

    print(f"\n  [{label}]")
    print(f"    Top-1 Accuracy : {top1_acc:.4f}")
    print(f"    Macro F1       : {macro_f1:.4f}")
    print(f"    Log Loss       : {ll:.4f}")
    print(f"    Route Continuity: {continuity:.4f}" if not np.isnan(continuity)
          else "    Route Continuity: N/A (need consecutive obs)")

    return {
        "top1_acc"  : round(float(top1_acc), 6),
        "macro_f1"  : round(float(macro_f1), 6),
        "log_loss"  : round(float(ll), 6),
        "continuity": round(float(continuity), 6) if not np.isnan(continuity) else None,
    }


# ===========================================================================
# Main training routine
# ===========================================================================

def train():
    # --- Load data -----------------------------------------------------------
    train_full = load_train_data()

    # --- Split ---------------------------------------------------------------
    train_df, val_df = group_aware_split(train_full, val_size=10)

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df[TARGET_COL].values

    # --- Load student test set -----------------------------------------------
    print(f"\n[data] Loading student test data from {STUDENT_CSV} …")
    student_df = pd.read_csv(STUDENT_CSV)
    print(f"[data] Student test: {len(student_df):,} rows, {student_df['trip_id'].nunique()} trips.")

    # --- Model configs -------------------------------------------------------
    shared_params = dict(
        n_estimators=200,
        class_weight="balanced",
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )

    models = [
        ("RandomForest",  RandomForestClassifier(**shared_params),  RF_MODEL_PATH),
        ("ExtraTrees",    ExtraTreesClassifier(**shared_params),     ET_MODEL_PATH),
    ]

    all_results = []

    for model_name, clf, save_path in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name} …")
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s")

        # Save model
        joblib.dump(clf, save_path)
        print(f"  Saved → {save_path}")

        # Evaluate
        print(f"\n  Evaluating {model_name} …")
        val_metrics     = evaluate(clf, val_df,     f"{model_name} | Validation")
        student_metrics = evaluate(clf, student_df, f"{model_name} | Student Test")

        result = {
            "model_name"            : model_name,
            "val_top1_acc"          : val_metrics["top1_acc"],
            "val_macro_f1"          : val_metrics["macro_f1"],
            "val_log_loss"          : val_metrics["log_loss"],
            "val_route_continuity"  : val_metrics["continuity"],
            "student_test_top1_acc" : student_metrics["top1_acc"],
            "student_test_macro_f1" : student_metrics["macro_f1"],
            "student_test_log_loss" : student_metrics["log_loss"],
            "student_route_continuity": student_metrics["continuity"],
        }
        all_results.append(result)

    # --- Print summary table -------------------------------------------------
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<16} {'Val Top1':>9} {'Val F1':>8} {'Val LL':>8} {'Test Top1':>10} {'Test F1':>8} {'Test LL':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['model_name']:<16} "
            f"{r['val_top1_acc']:>9.4f} "
            f"{r['val_macro_f1']:>8.4f} "
            f"{r['val_log_loss']:>8.4f} "
            f"{r['student_test_top1_acc']:>10.4f} "
            f"{r['student_test_macro_f1']:>8.4f} "
            f"{r['student_test_log_loss']:>8.4f}"
        )

    # --- Save results --------------------------------------------------------
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")

    # --- Pick best model for predict_rf.py -----------------------------------
    best = max(all_results, key=lambda r: r["val_top1_acc"])
    best_path = RF_MODEL_PATH if best["model_name"] == "RandomForest" else ET_MODEL_PATH
    best_marker = os.path.join(REPO_ROOT, "artifacts", "best_rf_model.txt")
    with open(best_marker, "w") as f:
        f.write(best_path)
    print(f"Best model: {best['model_name']} (val Top-1={best['val_top1_acc']:.4f})")
    print(f"Best model path written → {best_marker}")


if __name__ == "__main__":
    train()
