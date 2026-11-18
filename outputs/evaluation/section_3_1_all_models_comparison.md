# Unified Model Comparison & Winner Summary (All 6 Families)

## Most Accurate Models
- **Highest Standalone Top-1 Accuracy:** `Model 4b: HistGradientBoosting` — **86.21%** Top-1 Accuracy, **0.0701** Log Loss, **93.68%** Route Continuity
- **Best Calibrated Probabilities (Lowest Log Loss):** `Model 4: LightGBM (Optuna Tuned)` — **0.0697** Log Loss, **85.97%** Top-1 Accuracy
- **Most Accurate End-to-End Pipeline (Classifier + HMM/Viterbi):** `LightGBM Tuned + HMM` — **88.95%** Weighted Top-1 Accuracy, **96.82%** Obs Continuity, **99.52%** Route Continuity, **88.64%** Route LCS F1

## Full Comparison Table (`student_test`: 10 trips / 19,160 GPS observations / 191,600 candidate rows)

| Rank | Model | Family | CV Top-1 | Student-Test Top-1 | Binary Macro F1 | Arc Macro F1 | Route Continuity | Log Loss | Infer Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Model 4b: HistGradientBoosting | Boosting | 84.32% | **86.21%** | N/A (logged) | 0.7643 | 93.68% | 0.0701 | 0.75s |
| 2 | Model 4c: XGBoost | Boosting | 84.41% | **86.08%** | N/A (logged) | 0.7606 | 93.77% | 0.0704 | 0.17s |
| 3 | Model 4d: LightGBM (Untuned) | Boosting | 84.30% | **86.07%** | N/A (logged) | 0.7618 | 93.68% | 0.0703 | 0.51s |
| 4 | Model 4: LightGBM (Optuna Tuned) | Boosting | N/A (full-fit) | **85.97%** | 0.9197 | 0.7614 | 93.66% | 0.0697 | 1.23s |
| 5 | Model 5: MLP Neural Network | Neural networks | N/A (full-fit) | **85.70%** | 0.9186 | 0.7524 | 93.81% | 0.0719 | 0.10s |
| 6 | Model 3a: Random Forest | Ensemble (Bagging) | N/A (full-fit) | **85.59%** | 0.8988 | 0.7561 | 92.34% | 0.1676 | 0.52s |
| 7 | Model 3b: Extra Trees | Ensemble (Bagging) | N/A (full-fit) | **85.35%** | 0.9006 | 0.7510 | 92.18% | 0.1823 | 0.51s |
| 8 | Model 1: Logistic Regression | Linear and probabilistic | 83.31% | **85.33%** | 0.9075 | 0.7362 | 93.08% | 0.0941 | 0.03s |
| 9 | Model 6: RBF SVM (Nystrom) | Kernel-based | N/A (full-fit) | **84.93%** | 0.9038 | 0.7334 | 93.19% | 0.0935 | 1.15s |
| 10 | Model 2: Distance-Weighted KNN | Instance-based | 82.34% | **83.92%** | 0.9021 | 0.7067 | 91.70% | 0.2225 | 12.12s |

## Full Pipeline Comparison (Classifier + HMM/Viterbi on `student_test`)

| Pipeline | Classifier-Only Top-1 | Classifier + HMM Top-1 | Obs Continuity | Full Route Continuity | Route LCS F1 | Decode Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| LightGBM Tuned + HMM | 85.97% | **88.95%** | 96.82% | 99.52% | 88.64% | 4.92s |
| MLP (Model 5) + HMM | 85.70% | **88.32%** | 96.76% | 99.46% | 88.04% | 5.13s |
