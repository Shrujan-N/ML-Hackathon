"""Unit & real-graph tests for Member 3's HMM & Viterbi decoder (`src/hmm/`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.hmm import (
    HMMConfig,
    HMMViterbiDecoder,
    ObservationCandidates,
    RoadNetworkGraph,
    ShortestPathCache,
    compare_classifier_vs_hmm,
    compute_route_continuity,
    compute_sequence_metrics,
    tune_hmm_parameters,
)


def _make_parallel_roads_graph() -> RoadNetworkGraph:
    """Build a directed graph with two parallel horizontal roads plus an isolated segment.

    Main road (y = 0.0000):
      node 0 (0.000, 0.000) --arc 0--> node 1 (0.001, 0.000) --arc 1--> node 2 (0.002, 0.000) --arc 2--> node 3 (0.003, 0.000)
    Parallel disconnected road (y = 0.0002, ~22m north):
      node 4 (0.000, 0.0002) --arc 3--> node 5 (0.001, 0.0002) --arc 4--> node 6 (0.002, 0.0002)
    Reverse arc on main road:
      node 1 --arc 5--> node 0
    Isolated component far away:
      node 7 (0.050, 0.050) --arc 6--> node 8 (0.051, 0.050)
    """
    nodes = np.array(
        [
            [0.000, 0.0000],
            [0.001, 0.0000],
            [0.002, 0.0000],
            [0.003, 0.0000],
            [0.000, 0.0002],
            [0.001, 0.0002],
            [0.002, 0.0002],
            [0.050, 0.0500],
            [0.051, 0.0500],
        ],
        dtype=np.float64,
    )
    arcs = np.array(
        [
            [0, 1],  # arc 0
            [1, 2],  # arc 1
            [2, 3],  # arc 2
            [4, 5],  # arc 3 (parallel)
            [5, 6],  # arc 4 (parallel)
            [1, 0],  # arc 5 (reverse of arc 0)
            [7, 8],  # arc 6 (isolated)
        ],
        dtype=np.int64,
    )
    return RoadNetworkGraph(nodes_lonlat=nodes, arcs=arcs)


def test_road_graph_and_shortest_path_cache():
    g = _make_parallel_roads_graph()
    assert g.num_nodes == 9
    assert g.num_arcs == 7
    assert g.is_directly_connected(0, 1)
    assert g.is_directly_connected(0, 0)
    assert not g.is_directly_connected(0, 2)
    assert g.is_reverse_arc(0, 5)

    cache = ShortestPathCache(g)
    # Multi-hop shortest path from node 0 -> node 3 should traverse arcs [0, 1, 2]
    dist, path = cache.shortest_node_path(0, 3, cutoff_m=1000.0)
    assert dist is not None and dist > 300.0
    assert path == [0, 1, 2]

    # Cache hit on second query
    prev_runs = cache.stats.dijkstra_runs
    dist2, path2 = cache.shortest_node_path(0, 2, cutoff_m=1000.0)
    assert cache.stats.dijkstra_runs == prev_runs
    assert path2 == [0, 1] and dist2 < dist

    # Unreachable isolated node
    d_unreach, p_unreach = cache.shortest_node_path(0, 7, cutoff_m=2000.0)
    assert d_unreach is None and p_unreach is None

    # Cutoff expansion: first query with tiny cutoff fails, wider cutoff succeeds
    cache2 = ShortestPathCache(g)
    assert cache2.shortest_node_path(0, 3, cutoff_m=50.0)[0] is None
    d_wide, p_wide = cache2.shortest_node_path(0, 3, cutoff_m=1000.0)
    assert d_wide is not None and p_wide == [0, 1, 2]


def test_viterbi_corrects_noisy_classifier_jump():
    """Classifier wrongly favors disconnected parallel street at t=2; HMM restores main road."""
    g = _make_parallel_roads_graph()
    obs = [
        ObservationCandidates(0, 0.0003, 0.00001, 1.0, [0, 3], [0.85, 0.15]),
        ObservationCandidates(1, 0.0008, 0.00001, 2.0, [0, 3], [0.80, 0.20]),
        # Noisy GPS spike: classifier prefers parallel arc 4 (0.70 vs 0.30)
        ObservationCandidates(2, 0.0014, 0.00015, 3.0, [1, 4], [0.30, 0.70]),
        ObservationCandidates(3, 0.0018, 0.00001, 4.0, [1, 4], [0.85, 0.15]),
        ObservationCandidates(4, 0.0025, 0.00001, 5.0, [2], [0.95]),
    ]

    decoder = HMMViterbiDecoder(HMMConfig(beta_m=5.0, disconnected_penalty=20.0))
    res = decoder.decode_trip(g, obs, trip_id="synth_parallel")

    # Classifier-only jumps to arc 4 at obs 2
    assert res.classifier_only_obs_arcs == [0, 0, 4, 1, 2]
    # HMM keeps continuous path [0, 0, 1, 1, 2]
    assert res.predicted_obs_arcs == [0, 0, 1, 1, 2]
    assert res.reconstructed_route_arcs == [0, 1, 2]

    ablation = compare_classifier_vs_hmm(
        g, res, reference_obs_arcs=[0, 0, 1, 1, 2], reference_route_arcs=[0, 1, 2]
    )
    assert ablation["classifier_plus_hmm"]["obs_top1_accuracy"] == 1.0
    assert ablation["classifier_only"]["obs_top1_accuracy"] == pytest.approx(0.8)
    assert ablation["delta_obs_top1_accuracy"] > 0.0
    assert ablation["classifier_plus_hmm"]["obs_continuity"]["direct_continuity"] == 1.0


def test_missing_candidates_and_intermediate_arc_filling():
    """Verify missing candidate bridging and shortest-path intermediate arc reconstruction."""
    g = _make_parallel_roads_graph()
    # Vehicle observed on arc 0, then obs 1 has NO candidates, then observed on arc 2 (skipping arc 1)
    obs = [
        ObservationCandidates(0, 0.0005, 0.0000, 1.0, [0], [0.9]),
        ObservationCandidates(1, 0.0015, 0.0000, 8.0, [], []),  # Missing candidates + gap
        ObservationCandidates(2, 0.0025, 0.0000, 15.0, [2], [0.9]),
    ]
    decoder = HMMViterbiDecoder(HMMConfig(fill_intermediate_arcs=True, missing_candidate_policy="carry_forward"))
    res = decoder.decode_trip(g, obs, trip_id="synth_gap")

    assert res.diagnostics["n_missing_candidates"] == 1
    assert res.diagnostics["n_gps_gaps"] == 1
    assert res.predicted_obs_arcs == [0, 0, 2]
    # Intermediate arc 1 must be automatically filled in reconstructed_route_arcs!
    assert res.reconstructed_route_arcs == [0, 1, 2]


def test_disconnected_fallback_and_sequence_break():
    """Ensure disconnected arcs and extreme GPS gaps do not produce -inf or crashes."""
    g = _make_parallel_roads_graph()
    obs = [
        ObservationCandidates(0, 0.0005, 0.0000, 1.0, [0], [0.9]),
        # Jump to isolated arc 6 without time break -> triggers disconnected fallback
        ObservationCandidates(1, 0.0505, 0.0500, 3.0, [6], [0.9]),
        # Extreme 180s time break -> triggers sequence break
        ObservationCandidates(2, 0.0015, 0.0000, 183.0, [1], [0.9]),
    ]
    decoder = HMMViterbiDecoder(HMMConfig(break_gap_s=60.0, break_distance_m=10000.0))
    res = decoder.decode_trip(g, obs, trip_id="synth_disc")

    assert np.isfinite(res.total_log_prob)
    assert res.predicted_obs_arcs == [0, 6, 1]
    assert res.diagnostics["n_disconnected_fallbacks"] >= 1
    assert res.diagnostics["n_sequence_breaks"] == 1
    assert isinstance(res.to_dict(), dict)


def test_dataframe_adapter_and_tuner_leakage_guard():
    g = _make_parallel_roads_graph()
    cand_df = pd.DataFrame(
        {
            "obs_idx": [0, 0, 1, 1, 2],
            "candidate_arc_id": [0, 3, 1, 4, 2],
            "pred_prob": [0.9, 0.1, 0.35, 0.65, 0.95],
            "lon": [0.0005, 0.0005, 0.0015, 0.0015, 0.0025],
            "lat": [0.0, 0.0, 0.0001, 0.0001, 0.0],
            "timestamp": [1.0, 1.0, 2.0, 2.0, 3.0],
        }
    )
    decoder = HMMViterbiDecoder()
    res = decoder.decode_from_dataframe(g, cand_df, trip_id="df_trip")
    assert res.predicted_obs_arcs == [0, 1, 2]

    # Leakage guard must block evaluator partition
    with pytest.raises(ValueError, match="strictly forbidden"):
        tune_hmm_parameters(
            trips_data=[],
            param_grid={"beta_m": [5.0]},
            partition_name="evaluator_test",
        )


def test_real_student_test_graph_ground_truth_transitions():
    """Smoke-test real graph loading, transition caching, and Viterbi on student_test/00000007."""
    trip_dir = Path("student_test/00000007")
    if not trip_dir.exists():
        pytest.skip("student_test/00000007 not present")

    g = RoadNetworkGraph.from_raw_files(
        trip_dir / "00000007.nodes",
        trip_dir / "00000007.arcs",
    )
    route = [int(line.strip()) for line in (trip_dir / "00000007.route").open()]

    # Construct synthetic candidate observations centered on each arc of the real ground-truth route
    observations = []
    for t, true_arc in enumerate(route):
        u, v = g.arc_endpoints(true_arc)
        mid_lon = 0.5 * (g.nodes_lonlat[u, 0] + g.nodes_lonlat[v, 0])
        mid_lat = 0.5 * (g.nodes_lonlat[u, 1] + g.nodes_lonlat[v, 1])
        # Include true_arc plus an unrelated distractor arc
        distractor = (true_arc + 100) % g.num_arcs
        # Intentionally flip probabilities every 10th step to confirm Viterbi preserves true path
        probs = [0.35, 0.65] if (t > 0 and t % 10 == 0 and t < len(route) - 1) else [0.85, 0.15]
        observations.append(
            ObservationCandidates(
                obs_idx=t,
                lon=float(mid_lon),
                lat=float(mid_lat),
                timestamp=float(t + 1),
                candidate_arc_ids=[true_arc, distractor],
                candidate_probs=probs,
            )
        )

    decoder = HMMViterbiDecoder(HMMConfig(beta_m=5.0, disconnected_penalty=25.0))
    res = decoder.decode_trip(g, observations, trip_id="00000007")

    cont = compute_route_continuity(g, res.reconstructed_route_arcs)
    seq = compute_sequence_metrics(res.reconstructed_route_arcs, route)
    assert cont["direct_continuity"] == 1.0
    assert seq["lcs_f1"] == 1.0
    assert res.diagnostics["decode_time_s"] < 5.0
