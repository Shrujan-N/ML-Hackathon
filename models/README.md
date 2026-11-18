# PathSnap Models Folder

All trained models across the 6 explored algorithm families are collected here (alongside the shared training-fit `StandardScaler` and `feature_schema.json`).

| File | Model | Family | Input Scaling Required? |
| :--- | :--- | :--- | :--- |
| `model1_logistic_regression.joblib` | Model 1: Logistic Regression | Linear and probabilistic | Yes (`feature_scaler.joblib`) |
| `model2_distance_weighted_knn.joblib` | Model 2: Distance-Weighted KNN | Instance-based | Yes (`feature_scaler.joblib`) |
| `model3a_random_forest.joblib` | Model 3a: Random Forest | Ensemble (Bagging) | No (raw 16 features) |
| `model3b_extra_trees.joblib` | Model 3b: Extra Trees | Ensemble (Bagging) | No (raw 16 features) |
| `model4_boosting_lightgbm_tuned.joblib` | Model 4: Tuned LightGBM Bundle | Boosting (Optuna tuned) | Handled inside bundle |
| `model5_mlp_neural_network.pt` | Model 5: PyTorch MLP (128-64-32) | Neural networks | Handled inside `.load()` |
| `model6_rbf_svm_nystroem.joblib` | Model 6: RBF SVM (Nystrom + Platt) | Kernel-based | Handled inside `.load()` |
| `feature_scaler.joblib` | StandardScaler (fit on `train` only) | Preprocessing | - |
| `feature_schema.json` | 16 feature names & candidate stats | Metadata | - |

Source implementations live under `src/models/` (`logistic_regression/`, `knn/`, `random_forest/`, `boosting/`, `mlp/`, `rbf_svm/`), and the unified comparison script is `src/models/compare_all_models.py`.
