# Member 2 - Eval harness, boosting model & Bayesian tuning

## New files (drop into the repo root, keep the folder structure)
```
src/evaluation/  base.py  metrics.py  cv_harness.py  run_cv.py  plots.py
src/models/boosting/boosting_model.py        (Model 4)
src/tuning/      bayes_opt.py  export_final.py
```
`pip install lightgbm xgboost optuna` (add to requirements.txt - Member 4 owns that file).

## Run order
```bash
# 1. Harness + Model 4 (5-fold GroupKFold by trip, plus fit-on-train / score-on-student_test)
python -m src.evaluation.run_cv --backend lightgbm --holdout
python -m src.evaluation.run_cv --backend xgboost  --holdout      # optional: compare backends
python -m src.evaluation.run_cv --backend hgb      --holdout      # sklearn fallback

# 2. Teammates plug THEIR models in (same command, their factory)
python -m src.evaluation.run_cv --name logreg --factory src.models.logreg.model:make_model --holdout

# 3. Bayesian optimisation on the winning model (train trips only)
python -m src.tuning.bayes_opt --backend lightgbm --n-trials 30 --folds 3 --trip-subset 0.5

# 4. Freeze final model + export artifacts, then regenerate tables/plots
python -m src.tuning.export_final
python -m src.evaluation.plots
```

## Model contract for teammates (Hour-0 agreement)
A model factory returns an object with `fit(X, y)`, `predict_proba(X)` (shape `(n,)` or `(n,2)`)
and `needs_scaling` (True for LogReg / KNN / SVM / MLP - the harness then fits a StandardScaler
on the *training fold only*). Features/labels come from `artifacts/feature_schema.json`.

## Outputs
| Path | Used for |
|---|---|
| `outputs/evaluation/model_metrics.csv` | single source of truth for every number in the PDF |
| `outputs/evaluation/master_comparison_cv.md/.csv`, `..._student_test.*` | Sec 3.1 table |
| `outputs/evaluation/plots/*.png` | Sec 3.3 charts |
| `outputs/tuning/trials.csv, best_params.json, optimization_history.png, param_importance.png` | Sec 4 |
| `artifacts/final_model.joblib` (bundle), `final_scaler.joblib`, `final_metrics.json` | Member 4's FastAPI |

Member 4 loads the model with:
```python
from src.models.boosting.boosting_model import load_bundle, bundle_predict_proba
bundle = load_bundle("artifacts/final_model.joblib")
p = bundle_predict_proba(bundle, candidates_df)   # P(label=1) per candidate row
```
Predicted arc per observation = candidate with max `p` within each `(trip_id, obs_idx)`;
Member 3's HMM can use `p` as the emission probability.

## Metric definitions (edit only `src/evaluation/metrics.py` if the problem statement differs)
- **Top-1**: % observations where highest-scoring candidate == true arc (true arc outside the K=10 candidates counts as a miss).
- **Macro F1**: macro-average F1 over arc classes (true vs predicted arc id).
- **Log loss**: binary log loss over all candidate rows.
- **Continuity**: % consecutive predictions in a trip that are the same arc or connected (`to_node[t-1] == from_node[t]`).
- **Inference time**: ms per 1000 candidate rows and per observation.

## Viva cheat-sheet
- *Why GroupKFold by trip?* Points in a trip are autocorrelated; random splits leak and inflate scores.
- *Why scaler inside the fold?* Fitting on all data leaks validation statistics. Trees don't need scaling at all.
- *Bayesian opt?* Optuna TPE: models good vs bad hyper-parameter regions, proposes next trial by expected improvement; MedianPruner kills poor trials early. Objective = mean CV Top-1 on train trips only; student_test never used.
- *Why boosting?* Corrects errors sequentially, handles non-linear interactions (distance x heading x connectivity) and mixed scales without preprocessing.
