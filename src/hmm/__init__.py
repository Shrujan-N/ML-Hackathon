"""PathSnap HMM & Viterbi Route Reconstruction package (Member 3)."""

from src.hmm.config import HMMConfig
from src.hmm.metrics import (
    compare_classifier_vs_hmm,
    compress_consecutive_arcs,
    compute_obs_top1_accuracy,
    compute_route_continuity,
    compute_sequence_metrics,
)
from src.hmm.road_graph import (
    CacheStats,
    ProjectionResult,
    RoadNetworkGraph,
    ShortestPathCache,
    haversine_m,
    haversine_vec_m,
)
from src.hmm.tuner import tune_hmm_parameters
from src.hmm.viterbi import HMMDecodeResult, HMMViterbiDecoder, ObservationCandidates

__all__ = [
    "HMMConfig",
    "CacheStats",
    "ProjectionResult",
    "RoadNetworkGraph",
    "ShortestPathCache",
    "haversine_m",
    "haversine_vec_m",
    "ObservationCandidates",
    "HMMDecodeResult",
    "HMMViterbiDecoder",
    "compress_consecutive_arcs",
    "compute_route_continuity",
    "compute_sequence_metrics",
    "compute_obs_top1_accuracy",
    "compare_classifier_vs_hmm",
    "tune_hmm_parameters",
]
