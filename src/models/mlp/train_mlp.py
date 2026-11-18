"""Train Model 5 (PyTorch MLP) on Member 1's processed dataset (Member 3).

Uses 80 training trips (trip-grouped internal validation split for genuine
epoch-wise train/val loss curves & early stopping) and evaluates preliminary
performance on the 10 student-testing trips. Official cross-model GroupKFold CV
comparison is deferred to Member 2's shared evaluation harness (`src/evaluation/`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from src.models.mlp.mlp_model import (
    MLPConfig,
    PathSnapMLPClassifier,
    plot_mlp_training_curves,
)


def compute_obs_top1(df: pd.DataFrame, prob_col: str = "pred_prob") -> float:
    """Observation-level Top-1 accuracy across all (trip_id, obs_idx) groups."""
    idx = df.groupby(["trip_id", "obs_idx"], sort=False)[prob_col].idxmax()
    chosen = df.loc[idx]
    return float((chosen["candidate_arc_id"] == chosen["true_arc_id"]).mean())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    schema = json.loads((repo_root / "artifacts" / "feature_schema.json").read_text(encoding="utf-8"))
    feat_cols = schema["feature_columns"]
    target_col = schema["target_column"]

    print("Loading Member 1 processed candidate datasets...")
    train_df: pd.DataFrame = joblib.load(repo_root / "data" / "processed" / "train_candidates.pkl")
    test_df: pd.DataFrame = joblib.load(repo_root / "data" / "processed" / "student_test_candidates.pkl")

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.int64)
    groups_train = train_df["trip_id"].to_numpy()

    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.int64)

    cfg = MLPConfig(
        hidden_dims=(128, 64, 32),
        dropout=0.15,
        use_batchnorm=True,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=2048,
        max_epochs=30,
        patience=5,
        val_fraction=0.15,
        auto_pos_weight=False,  # Exact calibrated BCE log-loss for optimal Log Loss & HMM emissions
        scale_features=True,
        random_state=42,
        device="auto",
    )

    clf = PathSnapMLPClassifier(cfg)
    print(f"Training Model 5 (MLP) on {len(X_train):,} rows ({train_df['trip_id'].nunique()} trips) using device={clf.device_}...")
    clf.fit(X_train, y_train, groups=groups_train)

    # Inference timing on student_test
    t_inf0 = time.perf_counter()
    test_probs = clf.predict_proba(X_test)[:, 1]
    inf_time_s = time.perf_counter() - t_inf0
    test_preds = (test_probs >= 0.5).astype(np.int64)

    test_eval_df = test_df[["trip_id", "obs_idx", "candidate_arc_id", "true_arc_id", "label"]].copy()
    test_eval_df["pred_prob"] = test_probs.astype(np.float32)

    cm = confusion_matrix(y_test, test_preds).tolist()
    metrics = {
        "model_name": "Model 5: MLP Neural Network",
        "algorithm_family": "Neural networks",
        "device": str(clf.device_),
        "config": cfg.to_dict(),
        "train_rows": int(len(X_train)),
        "train_trips": int(train_df["trip_id"].nunique()),
        "student_test_rows": int(len(X_test)),
        "student_test_trips": int(test_df["trip_id"].nunique()),
        "epochs_trained": int(len(clf.history_)),
        "best_epoch": int(clf.best_epoch_),
        "best_val_loss": float(min(h["val_loss"] for h in clf.history_)),
        "best_val_macro_f1": float(max(h["val_macro_f1"] for h in clf.history_)),
        "train_time_s": float(clf.train_time_s_),
        "student_test_inference_time_s": float(inf_time_s),
        "student_test_accuracy": float(accuracy_score(y_test, test_preds)),
        "student_test_precision_pos": float(precision_score(y_test, test_preds, zero_division=0)),
        "student_test_recall_pos": float(recall_score(y_test, test_preds, zero_division=0)),
        "student_test_macro_f1": float(f1_score(y_test, test_preds, average="macro", zero_division=0)),
        "student_test_pos_f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "student_test_log_loss": float(log_loss(y_test, test_probs)),
        "student_test_obs_top1_accuracy": compute_obs_top1(test_eval_df, "pred_prob"),
        "student_test_confusion_matrix": cm,
        "note": "Preliminary train->student_test run; official 5-fold GroupKFold CV awaiting Member 2 harness.",
    }

    out_dir = repo_root / "outputs" / "member3"
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    clf.save(out_dir / "models" / "mlp_model5.pt")
    pd.DataFrame(clf.history_).to_csv(out_dir / "logs" / "mlp_epoch_history.csv", index=False)
    plot_mlp_training_curves(
        clf.history_,
        out_dir / "plots" / "model5_mlp_loss_curves.png",
        title="Model 5 (MLP) — Real Epoch-wise Train vs Validation Curves (80 Train Trips)",
    )
    joblib.dump(test_eval_df, out_dir / "predictions" / "mlp_student_test_preds.pkl", compress=3)
    (out_dir / "logs" / "mlp_student_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\n=== Model 5 (MLP) Student-Test Results ===")
    for k in [
        "epochs_trained",
        "best_epoch",
        "best_val_loss",
        "best_val_macro_f1",
        "train_time_s",
        "student_test_inference_time_s",
        "student_test_macro_f1",
        "student_test_obs_top1_accuracy",
        "student_test_log_loss",
        "student_test_precision_pos",
        "student_test_recall_pos",
    ]:
        print(f"  {k}: {metrics[k]}")


if __name__ == "__main__":
    main()
