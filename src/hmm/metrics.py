"""Route continuity, sequence-level metrics, and Classifier vs HMM ablation (Member 3)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from src.hmm.road_graph import RoadNetworkGraph, ShortestPathCache
from src.hmm.viterbi import HMMDecodeResult


def compress_consecutive_arcs(arcs: Sequence[int]) -> List[int]:
    """Remove consecutive duplicate arcs and invalid (-1) entries."""
    out: List[int] = []
    for a in arcs:
        a_int = int(a)
        if a_int < 0:
            continue
        if not out or out[-1] != a_int:
            out.append(a_int)
    return out


def compute_route_continuity(
    graph: RoadNetworkGraph,
    arc_sequence: Sequence[int],
    reachable_cutoff_m: float = 250.0,
    sp_cache: Optional[ShortestPathCache] = None,
) -> Dict[str, float]:
    """Compute direct and graph-reachable continuity on a predicted arc sequence.

    - `direct_continuity`: fraction of consecutive valid transitions (a_i -> a_{i+1})
      where either a_i == a_{i+1} or to_node(a_i) == from_node(a_{i+1}).
    - `compressed_direct_continuity`: direct connectivity after deduplicating stays.
    - `reachable_continuity`: fraction of consecutive valid transitions reachable via
      a directed path within `reachable_cutoff_m` meters.
    """
    valid = [int(a) for a in arc_sequence if int(a) >= 0]
    if len(valid) <= 1:
        return {
            "direct_continuity": 1.0 if len(valid) == 1 else 0.0,
            "compressed_direct_continuity": 1.0 if len(valid) == 1 else 0.0,
            "reachable_continuity": 1.0 if len(valid) == 1 else 0.0,
            "n_transitions": 0,
            "n_discontinuities": 0,
        }

    direct_ok = 0
    reachable_ok = 0
    cache = sp_cache if sp_cache is not None else ShortestPathCache(graph)

    for i in range(len(valid) - 1):
        a, b = valid[i], valid[i + 1]
        if graph.is_directly_connected(a, b):
            direct_ok += 1
            reachable_ok += 1
        else:
            dist_m, _ = cache.shortest_node_path(
                int(graph.to_nodes[a]), int(graph.from_nodes[b]), reachable_cutoff_m
            )
            if dist_m is not None:
                reachable_ok += 1

    n_trans = len(valid) - 1
    comp = compress_consecutive_arcs(valid)
    if len(comp) <= 1:
        comp_direct = 1.0
    else:
        comp_ok = sum(
            1 for i in range(len(comp) - 1) if graph.is_directly_connected(comp[i], comp[i + 1])
        )
        comp_direct = comp_ok / (len(comp) - 1)

    return {
        "direct_continuity": direct_ok / n_trans,
        "compressed_direct_continuity": comp_direct,
        "reachable_continuity": reachable_ok / n_trans,
        "n_transitions": n_trans,
        "n_discontinuities": n_trans - direct_ok,
    }


def _lcs_length(seq_a: Sequence[int], seq_b: Sequence[int]) -> int:
    """Order-preserving Longest Common Subsequence length in O(min(len)) memory."""
    if not seq_a or not seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    prev = [0] * (len(seq_b) + 1)
    for x in seq_a:
        curr = [0] * (len(seq_b) + 1)
        for j, y in enumerate(seq_b, start=1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[-1]


def compute_sequence_metrics(
    predicted_route_arcs: Sequence[int],
    reference_route_arcs: Sequence[int],
) -> Dict[str, float]:
    """Compare reconstructed route sequence against `.route` reference sequence.

    Reports:
    - Order-aware LCS Precision, Recall, F1 (most faithful route-sequence metric)
    - Multiset (bag-of-arcs) Precision, Recall, F1
    - Set Jaccard similarity
    """
    pred = compress_consecutive_arcs(predicted_route_arcs)
    ref = compress_consecutive_arcs(reference_route_arcs)

    if not pred and not ref:
        return {
            "lcs_precision": 1.0,
            "lcs_recall": 1.0,
            "lcs_f1": 1.0,
            "bag_precision": 1.0,
            "bag_recall": 1.0,
            "bag_f1": 1.0,
            "jaccard": 1.0,
        }
    if not pred or not ref:
        return {
            "lcs_precision": 0.0,
            "lcs_recall": 0.0,
            "lcs_f1": 0.0,
            "bag_precision": 0.0,
            "bag_recall": 0.0,
            "bag_f1": 0.0,
            "jaccard": 0.0,
        }

    lcs_len = _lcs_length(pred, ref)
    lcs_p = lcs_len / len(pred)
    lcs_r = lcs_len / len(ref)
    lcs_f1 = (2.0 * lcs_p * lcs_r / (lcs_p + lcs_r)) if (lcs_p + lcs_r) > 0 else 0.0

    c_pred = Counter(pred)
    c_ref = Counter(ref)
    overlap = sum((c_pred & c_ref).values())
    bag_p = overlap / len(pred)
    bag_r = overlap / len(ref)
    bag_f1 = (2.0 * bag_p * bag_r / (bag_p + bag_r)) if (bag_p + bag_r) > 0 else 0.0

    s_pred, s_ref = set(pred), set(ref)
    jaccard = len(s_pred & s_ref) / len(s_pred | s_ref)

    return {
        "lcs_precision": lcs_p,
        "lcs_recall": lcs_r,
        "lcs_f1": lcs_f1,
        "bag_precision": bag_p,
        "bag_recall": bag_r,
        "bag_f1": bag_f1,
        "jaccard": jaccard,
    }


def compute_obs_top1_accuracy(
    predicted_obs_arcs: Sequence[int],
    reference_obs_arcs: Sequence[int],
) -> float:
    """Observation-level Top-1 accuracy comparing predicted arc vs reference arc per GPS point."""
    if len(predicted_obs_arcs) != len(reference_obs_arcs):
        raise ValueError("predicted_obs_arcs and reference_obs_arcs must have equal length")
    if len(predicted_obs_arcs) == 0:
        return 0.0
    matches = sum(
        1 for p, r in zip(predicted_obs_arcs, reference_obs_arcs) if int(p) == int(r) and int(r) >= 0
    )
    valid_ref = sum(1 for r in reference_obs_arcs if int(r) >= 0)
    return matches / valid_ref if valid_ref > 0 else 0.0


def compare_classifier_vs_hmm(
    graph: RoadNetworkGraph,
    decode_result: HMMDecodeResult,
    reference_obs_arcs: Optional[Sequence[int]] = None,
    reference_route_arcs: Optional[Sequence[int]] = None,
    sp_cache: Optional[ShortestPathCache] = None,
) -> Dict[str, Any]:
    """Produce structured Classifier-Only vs Classifier+HMM ablation metrics for one trip."""
    cache = sp_cache if sp_cache is not None else ShortestPathCache(graph)

    clf_cont = compute_route_continuity(graph, decode_result.classifier_only_obs_arcs, sp_cache=cache)
    hmm_obs_cont = compute_route_continuity(graph, decode_result.predicted_obs_arcs, sp_cache=cache)
    hmm_route_cont = compute_route_continuity(
        graph, decode_result.reconstructed_route_arcs, sp_cache=cache
    )

    out: Dict[str, Any] = {
        "trip_id": decode_result.trip_id,
        "classifier_only": {"continuity": clf_cont},
        "classifier_plus_hmm": {
            "obs_continuity": hmm_obs_cont,
            "route_continuity": hmm_route_cont,
            "decode_time_s": decode_result.diagnostics.get("decode_time_s", 0.0),
        },
    }

    if reference_obs_arcs is not None:
        clf_acc = compute_obs_top1_accuracy(decode_result.classifier_only_obs_arcs, reference_obs_arcs)
        hmm_acc = compute_obs_top1_accuracy(decode_result.predicted_obs_arcs, reference_obs_arcs)
        out["classifier_only"]["obs_top1_accuracy"] = clf_acc
        out["classifier_plus_hmm"]["obs_top1_accuracy"] = hmm_acc
        out["delta_obs_top1_accuracy"] = hmm_acc - clf_acc

    if reference_route_arcs is not None:
        clf_seq = compute_sequence_metrics(decode_result.classifier_only_obs_arcs, reference_route_arcs)
        hmm_seq = compute_sequence_metrics(
            decode_result.reconstructed_route_arcs, reference_route_arcs
        )
        out["classifier_only"]["sequence"] = clf_seq
        out["classifier_plus_hmm"]["sequence"] = hmm_seq
        out["delta_lcs_f1"] = hmm_seq["lcs_f1"] - clf_seq["lcs_f1"]

    out["delta_direct_continuity"] = (
        hmm_obs_cont["direct_continuity"] - clf_cont["direct_continuity"]
    )
    return out
