"""HMM decoder parameter tuning restricted strictly to train + student_test (Member 3)."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.hmm.config import HMMConfig
from src.hmm.metrics import compare_classifier_vs_hmm
from src.hmm.road_graph import RoadNetworkGraph, ShortestPathCache
from src.hmm.viterbi import HMMViterbiDecoder, ObservationCandidates


ALLOWED_TUNING_PARTITIONS = {"train", "student_test"}


def tune_hmm_parameters(
    trips_data: Sequence[Dict[str, Any]],
    param_grid: Dict[str, Sequence[Any]],
    base_config: Optional[HMMConfig] = None,
    partition_name: str = "train",
    objective: str = "composite",
) -> Tuple[HMMConfig, pd.DataFrame]:
    """Grid-search HMM decoder hyperparameters on permitted development partitions only.

    Each element of `trips_data` must be a dict containing:
      - `trip_id`: str
      - `graph`: RoadNetworkGraph
      - `observations`: Sequence[ObservationCandidates] (with classifier probs attached)
      - `reference_obs_arcs`: Optional[Sequence[int]]
      - `reference_route_arcs`: Optional[Sequence[int]]

    Leakage guard: raises ValueError if `partition_name` is not in {"train", "student_test"}.
    """
    if partition_name not in ALLOWED_TUNING_PARTITIONS:
        raise ValueError(
            f"HMM tuning is strictly forbidden on partition '{partition_name}'. "
            f"Allowed partitions: {sorted(ALLOWED_TUNING_PARTITIONS)}."
        )

    cfg0 = base_config or HMMConfig()
    keys = list(param_grid.keys())
    combos = list(product(*(param_grid[k] for k in keys)))

    # Reuse one ShortestPathCache per trip across all hyperparameter trials for huge speedups!
    trip_caches = {t["trip_id"]: ShortestPathCache(t["graph"]) for t in trips_data}

    records: List[Dict[str, Any]] = []
    best_score = -1e18
    best_config = cfg0

    for trial_idx, values in enumerate(combos):
        overrides = dict(zip(keys, values))
        cfg = replace(cfg0, **overrides)
        decoder = HMMViterbiDecoder(cfg)

        top1_scores: List[float] = []
        lcs_f1_scores: List[float] = []
        cont_scores: List[float] = []
        decode_times: List[float] = []

        for trip in trips_data:
            tid = trip["trip_id"]
            res = decoder.decode_trip(
                graph=trip["graph"],
                observations=trip["observations"],
                trip_id=tid,
                sp_cache=trip_caches[tid],
            )
            ablation = compare_classifier_vs_hmm(
                graph=trip["graph"],
                decode_result=res,
                reference_obs_arcs=trip.get("reference_obs_arcs"),
                reference_route_arcs=trip.get("reference_route_arcs"),
                sp_cache=trip_caches[tid],
            )
            cont_scores.append(ablation["classifier_plus_hmm"]["obs_continuity"]["direct_continuity"])
            decode_times.append(ablation["classifier_plus_hmm"]["decode_time_s"])
            if "obs_top1_accuracy" in ablation["classifier_plus_hmm"]:
                top1_scores.append(ablation["classifier_plus_hmm"]["obs_top1_accuracy"])
            if "sequence" in ablation["classifier_plus_hmm"]:
                lcs_f1_scores.append(ablation["classifier_plus_hmm"]["sequence"]["lcs_f1"])

        mean_top1 = sum(top1_scores) / len(top1_scores) if top1_scores else 0.0
        mean_lcs_f1 = sum(lcs_f1_scores) / len(lcs_f1_scores) if lcs_f1_scores else 0.0
        mean_cont = sum(cont_scores) / len(cont_scores) if cont_scores else 0.0
        mean_time = sum(decode_times) / len(decode_times) if decode_times else 0.0

        if objective == "obs_top1_accuracy":
            score = mean_top1
        elif objective == "lcs_f1":
            score = mean_lcs_f1
        elif objective == "continuity":
            score = mean_cont
        else:
            # Composite objective balancing pointwise accuracy, route sequence F1, and continuity
            parts = []
            if top1_scores:
                parts.append(0.45 * mean_top1)
            if lcs_f1_scores:
                parts.append(0.35 * mean_lcs_f1)
            parts.append(0.20 * mean_cont)
            score = sum(parts)

        row = {
            "trial": trial_idx,
            "partition": partition_name,
            **overrides,
            "mean_obs_top1_accuracy": mean_top1 if top1_scores else None,
            "mean_lcs_f1": mean_lcs_f1 if lcs_f1_scores else None,
            "mean_direct_continuity": mean_cont,
            "mean_decode_time_s": mean_time,
            "objective_score": score,
        }
        records.append(row)
        if score > best_score:
            best_score = score
            best_config = cfg

    results_df = pd.DataFrame(records).sort_values("objective_score", ascending=False).reset_index(drop=True)
    return best_config, results_df
