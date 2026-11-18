"""
schemas.py — All Pydantic v2 request/response models for the PathSnap FastAPI backend.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class CandidateFeatures(BaseModel):
    """One arc candidate with all 16 feature columns + identifiers."""
    candidate_arc_id: int
    from_node: int
    to_node: int
    # 16 feature columns (in order)
    dist_to_arc_m: float
    dist_min_diff_m: float
    dist_ratio_to_min: float
    cand_rank: float
    proj_fraction: float
    dist_to_from_node_m: float
    dist_to_to_node_m: float
    arc_length_m: float
    heading_diff_rad: float
    heading_cos: float
    heading_sin: float
    gps_speed_mps: float
    dt_prev_s: float
    prev_gps_dist_to_arc_m: float
    next_gps_dist_to_arc_m: float
    is_connected_from_prev_top1: float


class Observation(BaseModel):
    """One GPS observation with its arc candidates."""
    obs_idx: int
    timestamp: float
    candidates: List[CandidateFeatures]


class ObservationWithLabel(BaseModel):
    """One GPS observation with candidates + ground-truth label (for /evaluate)."""
    obs_idx: int
    timestamp: float
    true_arc_id: int  # ground-truth arc — never forwarded to UI
    candidates: List[CandidateFeatures]


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    trip_id: str
    observations: List[Observation]


class ArcScore(BaseModel):
    candidate_arc_id: int
    score: float


class ObservationPrediction(BaseModel):
    obs_idx: int
    predicted_arc_id: int
    confidence: float
    all_scores: List[ArcScore]


class PredictResponse(BaseModel):
    trip_id: str
    predictions: List[ObservationPrediction]
    inference_time_ms: float


# ---------------------------------------------------------------------------
# /evaluate
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    trip_id: str
    observations: List[ObservationWithLabel]


class EvaluateResponse(BaseModel):
    trip_id: str
    top1_accuracy: float
    macro_f1: float
    log_loss: float
    route_continuity: float


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------

class LogsResponse(BaseModel):
    logs: List[str]
