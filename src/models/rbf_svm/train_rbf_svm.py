"""Train Model 6 (Kernel-based RBF SVM) on Member 1's processed dataset (Member 3).

Uses memory-safe Nystroem RBF kernel approximation + chunked L2 hinge-loss SVM
+ Platt sigmoid calibration across 100% of the 1,860,930 training candidate rows,
and evaluates preliminary performance on the 10 student-testing trips. Official
GroupKFold CV comparison is deferred to Member 2's shared evaluation harness.
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

from src.models.rbf_svm.rbf_svm_model import PathSnapRBFSVMClassifier, RBFSVMConfig


def compute_obs_top1(df: pd.DataFrame, prob_col: str = "pred_prob") -> float:
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
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df[target_col].to_numpy(dtype=np.int64)

    cfg = RBFSVMConfig(
        mode="auto",
        C=1.0,
        class_weight=None,  # Preserves calibrated posteriors after Platt scaling
        nystroem_components=256,
        nystroem_fit_samples=50_000,
        chunk_size=50_000,
        sgd_epochs=6,
        calibration_fraction=0.10,
        scale_features=True,
        random_state=42,
    )

    clf = PathSnapRBFSVMClassifier(cfg)
    print(f"Training Model 6 (RBF SVM) on {len(X_train):,} rows ({train_df['trip_id'].nunique()} trips)...")
    clf.fit(X_train, y_train)

    t_inf0 = time.perf_counter()
    test_probs = clf.predict_proba(X_test)[:, 1]
    inf_time_s = time.perf_counter() - t_inf0
    test_preds = (test_probs >= 0.5).astype(np.int64)

    test_eval_df = test_df[["trip_id", "obs_idx", "candidate_arc_id", "true_arc_id", "label"]].copy()
    test_eval_df["pred_prob"] = test_probs.astype(np.float32)

    cm = confusion_matrix(y_test, test_preds).tolist()
    metrics = {
        "model_name": "Model 6: RBF SVM (Nystroem Kernel Approximation)",
        "algorithm_family": "Kernel-based",
        "resolved_mode": clf.resolved_mode_,
        "config": cfg.to_dict(),
        "train_rows": int(len(X_train)),
        "train_trips": int(train_df["trip_id"].nunique()),
        "student_test_rows": int(len(X_test)),
        "student_test_trips": int(test_df["trip_id"].nunique()),
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
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    clf.save(out_dir / "models" / "rbf_svm_model6.joblib")
    joblib.dump(test_eval_df, out_dir / "predictions" / "rbf_svm_student_test_preds.pkl", compress=3)
    (out_dir / "logs" / "rbf_svm_student_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\n=== Model 6 (RBF SVM) Student-Test Results ===")
    for k in [
        "resolved_mode",
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
