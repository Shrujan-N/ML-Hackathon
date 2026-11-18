"""HMM & Viterbi sequential route reconstruction decoder (Member 3).

Combines per-observation candidate probabilities from any supervised classifier
(`predict_proba`) with directed road-network transition likelihoods in log-space
using the Viterbi algorithm.

Handles:
- Emission probabilities from classifier scores (with Gaussian distance fallback)
- Transition probabilities via Newson & Krumm (2009) exponential route-vs-GPS discrepancy
- GPS time gaps (dynamic cutoff + relaxed beta + optional sequence break on extreme gaps)
- Missing candidate sets (carry-forward or explicit sentinel without crashing)
- Disconnected road segments / invalid transitions (finite fallback penalty, no -inf collapse)
- Continuous route reconstruction by inserting shortest-path intermediate arcs
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.hmm.config import HMMConfig
from src.hmm.road_graph import (
    ProjectionResult,
    RoadNetworkGraph,
    ShortestPathCache,
    haversine_m,
)


@dataclass
class ObservationCandidates:
    """Candidate road arcs and classifier scores for a single GPS observation."""

    obs_idx: int
    lon: float
    lat: float
    timestamp: float
    candidate_arc_ids: List[int] = field(default_factory=list)
    candidate_probs: List[float] = field(default_factory=list)
    candidate_dists_m: Optional[List[float]] = None


@dataclass
class HMMDecodeResult:
    """JSON-serializable output of HMM/Viterbi decoding for a single trip."""

    trip_id: str
    obs_indices: List[int]
    predicted_obs_arcs: List[int]  # One arc_id per observation (-1 if unmapped)
    classifier_only_obs_arcs: List[int]  # Argmax classifier baseline per observation
    reconstructed_route_arcs: List[int]  # Deduplicated continuous chronological route
    total_log_prob: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Contract-friendly JSON dict for Member 4's FastAPI backend & UI."""
        return {
            "trip_id": self.trip_id,
            "obs_indices": self.obs_indices,
            "predicted_obs_arcs": self.predicted_obs_arcs,
            "classifier_only_obs_arcs": self.classifier_only_obs_arcs,
            "reconstructed_route_arcs": self.reconstructed_route_arcs,
            "total_log_prob": float(self.total_log_prob),
            "diagnostics": self.diagnostics,
        }


class HMMViterbiDecoder:
    """Log-space Hidden Markov Model + Viterbi route decoder."""

    def __init__(self, config: Optional[HMMConfig] = None) -> None:
        self.config = config or HMMConfig()

    def _emission_log_probs(
        self,
        obs: ObservationCandidates,
        projections: List[ProjectionResult],
    ) -> np.ndarray:
        """Compute log emission scores for each candidate arc at observation `obs`."""
        cfg = self.config
        k = len(obs.candidate_arc_ids)
        if k == 0:
            return np.empty(0, dtype=np.float64)

        has_probs = len(obs.candidate_probs) == k and all(
            p is not None and not math.isnan(p) for p in obs.candidate_probs
        )
        if has_probs:
            probs = np.clip(np.asarray(obs.candidate_probs, dtype=np.float64), cfg.prob_eps, 1.0)
            return cfg.emission_weight * np.log(probs)

        if cfg.use_distance_fallback:
            if obs.candidate_dists_m is not None and len(obs.candidate_dists_m) == k:
                dists = np.asarray(obs.candidate_dists_m, dtype=np.float64)
            else:
                dists = np.asarray([p.perp_dist_m for p in projections], dtype=np.float64)
            sigma = max(1e-3, cfg.sigma_z_m)
            return -0.5 * (dists / sigma) ** 2 - math.log(math.sqrt(2.0 * math.pi) * sigma)

        return np.zeros(k, dtype=np.float64)

    def _search_cutoff_m(self, d_gps_m: float, dt_s: float) -> float:
        """Dynamic shortest-path distance cutoff accounting for GPS displacement & time gaps."""
        cfg = self.config
        spatial_bound = d_gps_m * cfg.search_dist_multiplier + cfg.search_dist_buffer_m
        temporal_bound = max(1.0, dt_s) * cfg.max_speed_mps
        raw = max(cfg.min_search_radius_m, spatial_bound, min(temporal_bound, cfg.max_search_radius_m))
        return float(min(cfg.max_search_radius_m, raw))

    def _effective_beta(self, dt_s: float) -> float:
        """Relax transition scale `beta` proportionally during GPS gaps."""
        cfg = self.config
        if dt_s <= 1.0:
            return max(1e-3, cfg.beta_m)
        return max(1e-3, cfg.beta_m * (1.0 + cfg.beta_gap_scale * (dt_s - 1.0)))

    def _transition_log_prob(
        self,
        graph: RoadNetworkGraph,
        sp_cache: ShortestPathCache,
        arc_prev: int,
        proj_prev: ProjectionResult,
        arc_curr: int,
        proj_curr: ProjectionResult,
        d_gps_m: float,
        dt_s: float,
        cutoff_m: float,
        effective_beta: float,
    ) -> Tuple[float, bool]:
        """Compute log transition score P(arc_curr | arc_prev) and whether fallback was triggered."""
        cfg = self.config
        route_dist_m, is_backward, _ = sp_cache.arc_to_arc_distance(
            arc_prev, proj_prev.frac_along, arc_curr, proj_curr.frac_along, cutoff_m
        )

        if route_dist_m is None:
            # Disconnected road segment or exceeds dynamic search radius -> finite fallback penalty
            return -cfg.disconnected_penalty, True

        discrepancy = abs(route_dist_m - d_gps_m)
        log_p = -discrepancy / effective_beta - math.log(effective_beta)

        if arc_prev == arc_curr:
            log_p += cfg.same_arc_bonus
            if is_backward and route_dist_m > 2.0:
                log_p -= cfg.reverse_penalty
        elif graph.is_reverse_arc(arc_prev, arc_curr):
            log_p -= cfg.reverse_penalty

        return log_p, False

    def decode_trip(
        self,
        graph: RoadNetworkGraph,
        observations: Sequence[ObservationCandidates],
        trip_id: str = "unknown",
        sp_cache: Optional[ShortestPathCache] = None,
    ) -> HMMDecodeResult:
        """Run Viterbi decoding over an ordered sequence of GPS observations for one trip."""
        t0 = time.perf_counter()
        cfg = self.config
        cache = sp_cache if sp_cache is not None else ShortestPathCache(graph)

        obs_list = list(observations)
        n_obs = len(obs_list)
        if n_obs == 0:
            return HMMDecodeResult(
                trip_id=trip_id,
                obs_indices=[],
                predicted_obs_arcs=[],
                classifier_only_obs_arcs=[],
                reconstructed_route_arcs=[],
                total_log_prob=0.0,
                diagnostics={"n_obs": 0, "decode_time_s": 0.0, "cache_stats": cache.stats.to_dict()},
            )

        # Classifier-only Top-1 baseline (preserves strict candidate correspondence)
        classifier_only: List[int] = []
        for obs in obs_list:
            if len(obs.candidate_arc_ids) == 0:
                classifier_only.append(-1)
            elif len(obs.candidate_probs) == len(obs.candidate_arc_ids):
                best_idx = int(np.argmax(obs.candidate_probs))
                classifier_only.append(int(obs.candidate_arc_ids[best_idx]))
            else:
                classifier_only.append(int(obs.candidate_arc_ids[0]))

        # Filter observations with at least one valid candidate for Viterbi states
        valid_positions: List[int] = []
        valid_obs: List[ObservationCandidates] = []
        valid_projs: List[List[ProjectionResult]] = []
        n_missing_candidates = 0

        for pos, obs in enumerate(obs_list):
            if len(obs.candidate_arc_ids) == 0:
                n_missing_candidates += 1
                continue
            projs = [
                graph.project_point_onto_arc(obs.lon, obs.lat, int(arc_id))
                for arc_id in obs.candidate_arc_ids
            ]
            valid_positions.append(pos)
            valid_obs.append(obs)
            valid_projs.append(projs)

        if len(valid_obs) == 0:
            elapsed = time.perf_counter() - t0
            return HMMDecodeResult(
                trip_id=trip_id,
                obs_indices=[o.obs_idx for o in obs_list],
                predicted_obs_arcs=[-1] * n_obs,
                classifier_only_obs_arcs=classifier_only,
                reconstructed_route_arcs=[],
                total_log_prob=0.0,
                diagnostics={
                    "n_obs": n_obs,
                    "n_valid_obs": 0,
                    "n_missing_candidates": n_missing_candidates,
                    "n_gps_gaps": 0,
                    "n_sequence_breaks": 0,
                    "n_disconnected_fallbacks": 0,
                    "decode_time_s": elapsed,
                    "cache_stats": cache.stats.to_dict(),
                },
            )

        # Split valid observations into continuous Viterbi segments on extreme GPS breaks
        segments: List[List[int]] = [[0]]
        n_gps_gaps = 0
        n_sequence_breaks = 0

        for i in range(1, len(valid_obs)):
            prev_o = valid_obs[i - 1]
            curr_o = valid_obs[i]
            dt = max(0.0, float(curr_o.timestamp) - float(prev_o.timestamp))
            d_gps = haversine_m(prev_o.lon, prev_o.lat, curr_o.lon, curr_o.lat)
            if dt > cfg.gap_threshold_s:
                n_gps_gaps += 1
            if dt >= cfg.break_gap_s or d_gps >= cfg.break_distance_m:
                n_sequence_breaks += 1
                segments.append([i])
            else:
                segments[-1].append(i)

        # Run log-space Viterbi independently on each continuous segment
        decoded_valid_arcs: List[int] = [-1] * len(valid_obs)
        decoded_valid_projs: List[Optional[ProjectionResult]] = [None] * len(valid_obs)
        total_log_prob = 0.0
        n_disconnected_fallbacks = 0

        for seg in segments:
            seg_len = len(seg)
            first_i = seg[0]
            dp_prev = self._emission_log_probs(valid_obs[first_i], valid_projs[first_i])
            backpointers: List[np.ndarray] = []

            for step in range(1, seg_len):
                prev_i = seg[step - 1]
                curr_i = seg[step]
                o_prev = valid_obs[prev_i]
                o_curr = valid_obs[curr_i]
                p_prev_list = valid_projs[prev_i]
                p_curr_list = valid_projs[curr_i]

                k_prev = len(o_prev.candidate_arc_ids)
                k_curr = len(o_curr.candidate_arc_ids)

                dt = max(0.0, float(o_curr.timestamp) - float(o_prev.timestamp))
                d_gps = haversine_m(o_prev.lon, o_prev.lat, o_curr.lon, o_curr.lat)
                cutoff = self._search_cutoff_m(d_gps, dt)
                eff_beta = self._effective_beta(dt)

                emit_curr = self._emission_log_probs(o_curr, p_curr_list)
                dp_curr = np.full(k_curr, -math.inf, dtype=np.float64)
                bp_curr = np.zeros(k_curr, dtype=np.int64)

                all_disconnected = True
                for j in range(k_curr):
                    arc_j = int(o_curr.candidate_arc_ids[j])
                    proj_j = p_curr_list[j]
                    best_score = -math.inf
                    best_prev = 0
                    for prev_idx in range(k_prev):
                        arc_p = int(o_prev.candidate_arc_ids[prev_idx])
                        t_logp, is_disc = self._transition_log_prob(
                            graph=graph,
                            sp_cache=cache,
                            arc_prev=arc_p,
                            proj_prev=p_prev_list[prev_idx],
                            arc_curr=arc_j,
                            proj_curr=proj_j,
                            d_gps_m=d_gps,
                            dt_s=dt,
                            cutoff_m=cutoff,
                            effective_beta=eff_beta,
                        )
                        if not is_disc:
                            all_disconnected = False
                        score = dp_prev[prev_idx] + t_logp
                        if score > best_score:
                            best_score = score
                            best_prev = prev_idx
                    dp_curr[j] = best_score + emit_curr[j]
                    bp_curr[j] = best_prev

                if all_disconnected:
                    n_disconnected_fallbacks += 1

                backpointers.append(bp_curr)
                dp_prev = dp_curr

            # Backtrack optimal state sequence for this segment
            best_last_state = int(np.argmax(dp_prev))
            total_log_prob += float(dp_prev[best_last_state])
            state_seq = [0] * seg_len
            state_seq[-1] = best_last_state
            for step in range(seg_len - 2, -1, -1):
                state_seq[step] = int(backpointers[step][state_seq[step + 1]])

            for step, valid_idx in enumerate(seg):
                st = state_seq[step]
                decoded_valid_arcs[valid_idx] = int(valid_obs[valid_idx].candidate_arc_ids[st])
                decoded_valid_projs[valid_idx] = valid_projs[valid_idx][st]

        # Map decoded states back to all original observations (handling missing candidates)
        predicted_obs_arcs: List[int] = [-1] * n_obs
        for valid_idx, orig_pos in enumerate(valid_positions):
            predicted_obs_arcs[orig_pos] = decoded_valid_arcs[valid_idx]

        if n_missing_candidates > 0 and cfg.missing_candidate_policy == "carry_forward":
            # Forward-fill, then backward-fill leading missing observations
            last_valid = -1
            for i in range(n_obs):
                if predicted_obs_arcs[i] != -1:
                    last_valid = predicted_obs_arcs[i]
                elif last_valid != -1:
                    predicted_obs_arcs[i] = last_valid
            next_valid = -1
            for i in range(n_obs - 1, -1, -1):
                if predicted_obs_arcs[i] != -1:
                    next_valid = predicted_obs_arcs[i]
                elif next_valid != -1:
                    predicted_obs_arcs[i] = next_valid

        # Reconstruct continuous deduplicated route sequence (with optional intermediate path filling)
        reconstructed_route = self._reconstruct_route_sequence(
            graph=graph,
            sp_cache=cache,
            valid_obs=valid_obs,
            decoded_valid_arcs=decoded_valid_arcs,
            decoded_valid_projs=decoded_valid_projs,
            segments=segments,
        )

        elapsed = time.perf_counter() - t0
        return HMMDecodeResult(
            trip_id=trip_id,
            obs_indices=[o.obs_idx for o in obs_list],
            predicted_obs_arcs=predicted_obs_arcs,
            classifier_only_obs_arcs=classifier_only,
            reconstructed_route_arcs=reconstructed_route,
            total_log_prob=total_log_prob,
            diagnostics={
                "n_obs": n_obs,
                "n_valid_obs": len(valid_obs),
                "n_missing_candidates": n_missing_candidates,
                "n_gps_gaps": n_gps_gaps,
                "n_sequence_breaks": n_sequence_breaks,
                "n_disconnected_fallbacks": n_disconnected_fallbacks,
                "decode_time_s": elapsed,
                "cache_stats": cache.stats.to_dict(),
            },
        )

    def _reconstruct_route_sequence(
        self,
        graph: RoadNetworkGraph,
        sp_cache: ShortestPathCache,
        valid_obs: List[ObservationCandidates],
        decoded_valid_arcs: List[int],
        decoded_valid_projs: List[Optional[ProjectionResult]],
        segments: List[List[int]],
    ) -> List[int]:
        """Build deduplicated chronological arc sequence, filling intermediate shortest-path arcs."""
        cfg = self.config
        route: List[int] = []

        def _append_arc(arc_id: int) -> None:
            if arc_id < 0:
                return
            if not route or route[-1] != arc_id:
                route.append(arc_id)

        for seg in segments:
            for step, v_idx in enumerate(seg):
                curr_arc = decoded_valid_arcs[v_idx]
                if step == 0:
                    _append_arc(curr_arc)
                    continue

                prev_v_idx = seg[step - 1]
                prev_arc = decoded_valid_arcs[prev_v_idx]
                if prev_arc == curr_arc:
                    continue

                if cfg.fill_intermediate_arcs and not graph.is_directly_connected(prev_arc, curr_arc):
                    o_prev = valid_obs[prev_v_idx]
                    o_curr = valid_obs[v_idx]
                    dt = max(0.0, float(o_curr.timestamp) - float(o_prev.timestamp))
                    d_gps = haversine_m(o_prev.lon, o_prev.lat, o_curr.lon, o_curr.lat)
                    cutoff = self._search_cutoff_m(d_gps, dt)
                    p_prev = decoded_valid_projs[prev_v_idx]
                    p_curr = decoded_valid_projs[v_idx]
                    _, _, inter_arcs = sp_cache.arc_to_arc_distance(
                        prev_arc,
                        p_prev.frac_along if p_prev else 0.5,
                        curr_arc,
                        p_curr.frac_along if p_curr else 0.5,
                        cutoff,
                    )
                    if inter_arcs:
                        for mid_arc in inter_arcs:
                            _append_arc(mid_arc)

                _append_arc(curr_arc)

        return route

    def decode_from_dataframe(
        self,
        graph: RoadNetworkGraph,
        candidates_df: pd.DataFrame,
        trip_id: str = "unknown",
        track_df: Optional[pd.DataFrame] = None,
        obs_idx_col: str = "obs_idx",
        arc_id_col: str = "candidate_arc_id",
        prob_col: str = "pred_prob",
        lon_col: str = "lon",
        lat_col: str = "lat",
        ts_col: str = "timestamp",
        dist_col: Optional[str] = None,
        sp_cache: Optional[ShortestPathCache] = None,
    ) -> HMMDecodeResult:
        """Adapter accepting Member 1's candidate DataFrame + classifier probability column.

        Preserves observations with 0 candidates if `track_df` is also supplied.
        """
        grouped = {int(k): grp for k, grp in candidates_df.groupby(obs_idx_col, sort=True)}

        if track_df is not None:
            ordered_obs: List[ObservationCandidates] = []
            for row_i, row in track_df.reset_index(drop=True).iterrows():
                oidx = int(row[obs_idx_col]) if obs_idx_col in track_df.columns else int(row_i)
                lon = float(row[lon_col])
                lat = float(row[lat_col])
                ts = float(row[ts_col]) if ts_col in track_df.columns else float(oidx)
                grp = grouped.get(oidx)
                if grp is None or len(grp) == 0:
                    ordered_obs.append(ObservationCandidates(oidx, lon, lat, ts))
                else:
                    ordered_obs.append(
                        ObservationCandidates(
                            obs_idx=oidx,
                            lon=lon,
                            lat=lat,
                            timestamp=ts,
                            candidate_arc_ids=grp[arc_id_col].astype(int).tolist(),
                            candidate_probs=(
                                grp[prob_col].astype(float).tolist() if prob_col in grp.columns else []
                            ),
                            candidate_dists_m=(
                                grp[dist_col].astype(float).tolist()
                                if (dist_col and dist_col in grp.columns)
                                else None
                            ),
                        )
                    )
        else:
            ordered_obs = []
            for oidx in sorted(grouped.keys()):
                grp = grouped[oidx]
                first = grp.iloc[0]
                ordered_obs.append(
                    ObservationCandidates(
                        obs_idx=oidx,
                        lon=float(first[lon_col]),
                        lat=float(first[lat_col]),
                        timestamp=float(first[ts_col]) if ts_col in grp.columns else float(oidx),
                        candidate_arc_ids=grp[arc_id_col].astype(int).tolist(),
                        candidate_probs=(
                            grp[prob_col].astype(float).tolist() if prob_col in grp.columns else []
                        ),
                        candidate_dists_m=(
                            grp[dist_col].astype(float).tolist()
                            if (dist_col and dist_col in grp.columns)
                            else None
                        ),
                    )
                )

        return self.decode_trip(graph=graph, observations=ordered_obs, trip_id=trip_id, sp_cache=sp_cache)

    def decode_member1_trip(
        self,
        trip: Dict[str, Any],
        trip_candidates_df: pd.DataFrame,
        prob_col: str = "pred_prob",
        graph: Optional[RoadNetworkGraph] = None,
        sp_cache: Optional[ShortestPathCache] = None,
    ) -> HMMDecodeResult:
        """Direct adapter for Member 1's `load_trip()` dict + `data/processed/*_candidates.pkl` schema.

        Member 1's processed candidate tables store `trip_id`, `obs_idx`, `timestamp`,
        `candidate_arc_id`, `dist_to_arc_m`, etc. (without `lon`/`lat`), so this joins
        the raw GPS coordinates from `trip['track']` indexed by `obs_idx`.
        """
        g = graph if graph is not None else RoadNetworkGraph(trip["nodes"], trip["arcs"])
        track = np.asarray(trip["track"], dtype=np.float64)
        trip_id = str(trip.get("trip_id", "unknown"))

        track_df = pd.DataFrame(
            {
                "obs_idx": np.arange(len(track), dtype=np.int64),
                "lon": track[:, 0],
                "lat": track[:, 1],
                "timestamp": track[:, 2],
            }
        )
        return self.decode_from_dataframe(
            graph=g,
            candidates_df=trip_candidates_df,
            trip_id=trip_id,
            track_df=track_df,
            obs_idx_col="obs_idx",
            arc_id_col="candidate_arc_id",
            prob_col=prob_col,
            lon_col="lon",
            lat_col="lat",
            ts_col="timestamp",
            dist_col="dist_to_arc_m" if "dist_to_arc_m" in trip_candidates_df.columns else None,
            sp_cache=sp_cache,
        )
