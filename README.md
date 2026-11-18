# PathSnap: Distributed Map-Matching & Route Reconstruction

Course: **UE24CS352A: Machine Learning** | Department of CSE, PES University

PathSnap reconstructs realistic vehicle trajectories from noisy 1 Hz GPS logs (`.track`) by matching GPS observations onto a directed road network (`.nodes`, `.arcs`) using candidate generation, supervised ML classifiers across 6 algorithm families, Bayesian hyperparameter optimisation, and HMM/Viterbi sequential decoding, deployed across a two-laptop FastAPI + Streamlit architecture.

---

## Two-Laptop Deployment Architecture

| Laptop | Role | Components |
| :--- | :--- | :--- |
| **Laptop 1** | Model & API Server | Runs FastAPI (`api/main.py`) on port `8000`, hosts trained model weights, HMM/Viterbi decoder, road-network data, and server-side evaluation (`/evaluate` keeps ground-truth labels on server). |
| **Laptop 2** | Evaluator Interface | Runs Streamlit UI (`ui/app.py`), displays the 10 evaluation trips, sends trip candidates to Laptop 1 over LAN (`http://<LAPTOP1_IP>:8000`), and renders predicted routes on interactive Folium maps. |

---

## Repository Structure

```
ML Hackathon/
├── README.md
├── requirements.txt
├── PathSnap_Technical_Report.pdf        # Consolidated team technical report (Sections 2.2 - 8)
├── PathSnap_Technical_Report.md
├── models/                              # All trained models collected in one folder
│   ├── model1_logistic_regression.joblib
│   ├── model2_distance_weighted_knn.joblib
│   ├── model3a_random_forest.joblib
│   ├── model3b_extra_trees.joblib
│   ├── model4_boosting_lightgbm_tuned.joblib
│   ├── model5_mlp_neural_network.pt
│   ├── model6_rbf_svm_nystroem.joblib
│   ├── feature_scaler.joblib
│   └── feature_schema.json
├── artifacts/                           # Runtime model bundles & scalers used by API/scripts
├── src/
│   ├── data/                            # Raw parsing, meter projection, DP alignment & EDA
│   ├── features/                        # cKDTree candidate generation (K=10) & 16 features
│   ├── models/                          # All 6 model families + unified compare_all_models.py
│   ├── evaluation/                      # Shared GroupKFold harness, metrics & comparison plots
│   ├── tuning/                          # Optuna TPE Bayesian optimisation & final export
│   ├── hmm/                             # Road graph, shortest-path cache, HMM & Viterbi decoder
│   └── visualization/                   # Mandatory Folium multi-layer map generator
├── api/                                 # Laptop 1 FastAPI backend (main.py, schemas.py, model_loader.py)
├── ui/                                  # Laptop 2 Streamlit UI (app.py, api_client.py, map_builder.py)
├── notebooks/                           # Phase 1 EDA & feature engineering notebook
├── outputs/                             # Saved EDA tables/plots/maps, model metrics, tuning trials & HMM logs
└── tests/                               # Unit tests for HMM, Member 3 models, and Folium maps
```

---

## Installation

1. Clone the repository and create a Python environment (Python 3.10+):
   ```bash
   git clone https://github.com/Shrujan-N/ML-Hackathon.git
   cd ML-Hackathon
   pip install -r requirements.txt
   ```
2. Place the raw dataset (`train/` and `student_test/` folders) under `data/raw/`.

---

## Execution Pipeline

### 1. Exploratory Data Analysis & Feature Engineering
```bash
python src/eda_analysis.py
python src/feature_engineering.py
```

### 2. Model Training, Group-CV Evaluation & Unified Comparison
```bash
# Member 1 models (Logistic Regression & Distance-Weighted KNN)
python src/train_member1_models.py

# Member 2 Group-CV harness & Boosting models (LightGBM / XGBoost / HGB)
python -m src.evaluation.run_cv --backend lightgbm --holdout
python -m src.evaluation.run_cv --backend xgboost --holdout
python -m src.evaluation.run_cv --backend hgb --holdout

# Member 3 models (PyTorch MLP & RBF SVM)
python -m src.models.mlp.train_mlp
python -m src.models.rbf_svm.train_rbf_svm

# Member 4 models (Random Forest & Extra Trees)
python src/models/random_forest/train_rf.py

# Unified comparison across all models + HMM/Viterbi
python -m src.models.compare_all_models
python -m src.evaluation.plots
```

### 3. Bayesian Hyperparameter Optimisation (Optuna TPE)
```bash
python -m src.tuning.bayes_opt --backend lightgbm --n-trials 30 --folds 3 --trip-subset 0.5
python -m src.tuning.export_final
```

### 4. HMM & Viterbi Route Reconstruction + Folium Maps
```bash
python -m src.hmm.run_hmm_student_test
pytest tests/
```

---

## Two-Laptop LAN Deployment

### On Laptop 1 (API Server)
Start the FastAPI server listening on all interfaces (`0.0.0.0:8000`) and keep the terminal visible to inspect live request logs:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Check Laptop 1's local IP (e.g., `ipconfig` on Windows -> `192.168.x.x`).

### On Laptop 2 (Evaluator UI)
Launch the Streamlit UI and enter Laptop 1's API URL (`http://<LAPTOP1_IP>:8000`) in the sidebar:
```bash
streamlit run ui/app.py
```

---

## Summary of Results (`student_test` — 10 Trips, 19,160 Observations)

- **Best Standalone Classifier (Top-1 Accuracy):** HistGradientBoosting (`86.21%`, Log Loss `0.0701`), closely followed by XGBoost (`86.08%`) and LightGBM (`86.07%`).
- **Best Calibrated Probabilities (Lowest Log Loss):** Optuna-Tuned LightGBM (`0.0697` Log Loss, `85.97%` Top-1 Accuracy, `0.9197` Binary Macro F1).
- **Best End-to-End Pipeline (Classifier + HMM/Viterbi):** **Optuna-Tuned LightGBM + HMM/Viterbi** reaches **`88.95%` Weighted Top-1 Accuracy**, **`96.82%` Observation Continuity**, **`99.52%` Full Route Continuity**, and **`88.64%` Route LCS F1** (improving over `85.97%` classifier-only predictions).
