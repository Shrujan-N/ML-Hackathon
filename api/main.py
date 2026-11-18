"""
main.py — PathSnap FastAPI backend (Member 4 — Laptop 1).

Endpoints
---------
GET  /          — health check
POST /predict   — run ML model on GPS arc candidates
POST /evaluate  — server-side evaluation (labels never leave server)
GET  /logs      — last 100 log entries
"""

import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import FEATURE_COLUMNS, auto_load_model
from api.schemas import (
    ArcScore,
    EvaluateRequest,
    EvaluateResponse,
    ObservationPrediction,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Logging setup — ring buffer of last 100 entries, plus stdout
# ---------------------------------------------------------------------------

LOG_BUFFER: deque = deque(maxlen=100)


class RingBufferHandler(logging.Handler):
    """Appends formatted log records to LOG_BUFFER."""

    def emit(self, record: logging.LogRecord) -> None:
        LOG_BUFFER.append(self.format(record))


_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_ring = RingBufferHandler()
_ring.setFormatter(_fmt)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(_ring)

# Also print to stdout so CMD shows the logs.
_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(_fmt)
root_logger.addHandler(_stdout_handler)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state (populated at startup)
# ---------------------------------------------------------------------------

_model = None
_model_name: str = "none"

# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_name
    logger.info("PathSnap API starting up — loading model …")
    _model, _model_name = auto_load_model(artifacts_dir="artifacts")
    if _model is None:
        logger.warning(
            "No model loaded. /predict will return 503 until a model is placed in artifacts/."
        )
    else:
        logger.info("Model ready: %s", _model_name)
    yield
    logger.info("PathSnap API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PathSnap Map-Matching API",
    version="1.0.0",
    description="ML-powered GPS arc-candidate scoring backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # LAN-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidates_to_df(candidates) -> pd.DataFrame:
    """Convert a list of CandidateFeatures to a DataFrame with FEATURE_COLUMNS."""
    rows = [
        {col: getattr(c, col) for col in FEATURE_COLUMNS}
        for c in candidates
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _compute_route_continuity(predictions: List[ObservationPrediction], observations) -> float:
    """
    Route continuity: fraction of consecutive (obs_i, obs_i+1) pairs where
    the predicted arcs share at least one node (from_node or to_node).
    """
    if len(predictions) < 2:
        return 1.0

    # Build a quick lookup: obs_idx -> (from_node, to_node) of predicted arc
    arc_nodes: dict = {}
    for obs, pred in zip(observations, predictions):
        for cand in obs.candidates:
            if cand.candidate_arc_id == pred.predicted_arc_id:
                arc_nodes[pred.obs_idx] = (cand.from_node, cand.to_node)
                break

    continuity_hits = 0
    pairs = 0
    sorted_preds = sorted(predictions, key=lambda p: p.obs_idx)
    for i in range(len(sorted_preds) - 1):
        idx_a = sorted_preds[i].obs_idx
        idx_b = sorted_preds[i + 1].obs_idx
        if idx_a not in arc_nodes or idx_b not in arc_nodes:
            continue
        nodes_a = set(arc_nodes[idx_a])
        nodes_b = set(arc_nodes[idx_b])
        pairs += 1
        if nodes_a & nodes_b:
            continuity_hits += 1

    return continuity_hits / pairs if pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def health_check():
    """Health-check endpoint."""
    return {
        "status": "ok",
        "model": _model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Score all arc candidates for each GPS observation and return top-1 arc.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place rf_model.joblib or et_model.joblib in artifacts/ and restart.",
        )

    t_start = time.perf_counter()

    predictions: List[ObservationPrediction] = []

    for obs in request.observations:
        if not obs.candidates:
            continue

        feature_df = _candidates_to_df(obs.candidates)

        # Import here to avoid circular dependency at module level
        from api.model_loader import predict as model_predict
        scores = model_predict(_model, feature_df)  # shape (n_cands,)

        arc_ids = [c.candidate_arc_id for c in obs.candidates]
        best_idx = int(np.argmax(scores))

        all_scores = [
            ArcScore(candidate_arc_id=arc_id, score=float(score))
            for arc_id, score in zip(arc_ids, scores)
        ]

        predictions.append(
            ObservationPrediction(
                obs_idx=obs.obs_idx,
                predicted_arc_id=arc_ids[best_idx],
                confidence=float(scores[best_idx]),
                all_scores=all_scores,
            )
        )

    infer_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        "[PREDICT] trip=%s obs=%d infer=%.1fms",
        request.trip_id,
        len(predictions),
        infer_ms,
    )

    return PredictResponse(
        trip_id=request.trip_id,
        predictions=predictions,
        inference_time_ms=round(infer_ms, 2),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(request: EvaluateRequest):
    """
    Server-side evaluation — ground-truth labels stay on the server.
    Computes Top-1 Accuracy, Macro F1, Log Loss, and Route Continuity.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    # Lazily import sklearn metrics to keep startup fast
    from sklearn.metrics import f1_score, log_loss

    all_true_labels: List[int] = []      # binary (1 if label matches predicted)
    all_pred_proba: List[float] = []     # positive-class proba for each candidate
    all_true_binary: List[int] = []      # ground truth binary per observation
    top1_hits = 0
    obs_predictions: List[ObservationPrediction] = []

    from api.model_loader import predict as model_predict

    for obs in request.observations:
        if not obs.candidates:
            continue

        feature_df = _candidates_to_df(obs.candidates)
        scores = model_predict(_model, feature_df)

        arc_ids = [c.candidate_arc_id for c in obs.candidates]
        best_idx = int(np.argmax(scores))
        predicted_arc = arc_ids[best_idx]

        # Ground truth
        is_correct = int(predicted_arc == obs.true_arc_id)
        top1_hits += is_correct

        # Build binary labels per candidate (1 if == true_arc_id)
        for arc_id, score in zip(arc_ids, scores):
            all_true_labels.append(1 if arc_id == obs.true_arc_id else 0)
            all_pred_proba.append(float(score))

        # For macro F1: per-observation binary prediction
        all_true_binary.append(1)  # true answer exists in candidates
        # We only record whether top-1 is right; use binary arrays for f1
        # (ground truth = 1 for the true arc, compare against argmax prediction)

        obs_predictions.append(
            ObservationPrediction(
                obs_idx=obs.obs_idx,
                predicted_arc_id=predicted_arc,
                confidence=float(scores[best_idx]),
                all_scores=[
                    ArcScore(candidate_arc_id=aid, score=float(s))
                    for aid, s in zip(arc_ids, scores)
                ],
            )
        )

    n_obs = len(request.observations)
    top1_acc = top1_hits / n_obs if n_obs > 0 else 0.0

    # Macro F1 on per-candidate binary predictions
    pred_binary = [1 if p >= 0.5 else 0 for p in all_pred_proba]
    macro_f1 = float(
        f1_score(all_true_labels, pred_binary, average="macro", zero_division=0)
    )

    # Log Loss on per-candidate scores
    # Clip probabilities to avoid log(0)
    clipped = [max(1e-7, min(1 - 1e-7, p)) for p in all_pred_proba]
    ll = float(log_loss(all_true_labels, clipped))

    # Route continuity
    route_cont = _compute_route_continuity(obs_predictions, request.observations)

    logger.info(
        "[EVAL] trip=%s top1=%.4f f1=%.4f logloss=%.4f continuity=%.4f",
        request.trip_id,
        top1_acc,
        macro_f1,
        ll,
        route_cont,
    )

    return EvaluateResponse(
        trip_id=request.trip_id,
        top1_accuracy=round(top1_acc, 4),
        macro_f1=round(macro_f1, 4),
        log_loss=round(ll, 4),
        route_continuity=round(route_cont, 4),
    )


@app.get("/logs")
def get_logs():
    """Return the last 100 log entries."""
    return {"logs": list(LOG_BUFFER)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
