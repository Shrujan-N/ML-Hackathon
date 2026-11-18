"""End-to-end HMM/Viterbi ablation & Folium map generation on all 10 student_test trips (Member 3).

Runs using Model 5 (MLP) real candidate probabilities on Member 1's processed
`student_test_candidates.pkl` and raw trip road graphs (`student_test/`).
Will be re-run with Member 2's final Bayesian-tuned winning model once selected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.data.data_loader import load_trip
from src.hmm import (
    HMMConfig,
    HMMViterbiDecoder,
    RoadNetworkGraph,
    ShortestPathCache,
    compare_classifier_vs_hmm,
)
from src.visualization import create_trip_map, save_trip_map


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pred_path = repo_root / "outputs" / "member3" / "predictions" / "mlp_student_test_preds.pkl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Run train_mlp.py first to produce {pred_path}")

    cand_df: pd.DataFrame = joblib.load(repo_root / "data" / "processed" / "student_test_candidates.pkl")
    pred_df: pd.DataFrame = joblib.load(pred_path)
    cand_df = cand_df.copy()
    cand_df["pred_prob"] = pred_df["pred_prob"].to_numpy(dtype=np.float32)

    cfg = HMMConfig(
        emission_weight=1.0,
        beta_m=5.0,
        beta_gap_scale=0.5,
        same_arc_bonus=0.25,
        reverse_penalty=12.0,
        disconnected_penalty=25.0,
        gap_threshold_s=5.0,
        break_gap_s=60.0,
        fill_intermediate_arcs=True,
    )
    decoder = HMMViterbiDecoder(cfg)

    trip_ids = sorted(cand_df["trip_id"].astype(str).unique())
    rows: List[Dict[str, Any]] = []
    maps_dir = repo_root / "outputs" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    # Representative trips to render interactive Folium comparison maps for:
    # 00000000 (standard), 00000007 (clean urban), 00000065 (210s GPS gap), 00000076 (road break + 128 gaps)
    map_trips = {"00000000", "00000007", "00000065", "00000076"}

    print(f"Running HMM/Viterbi decoding on {len(trip_ids)} student_test trips (Model 5 MLP emissions)...")
    for tid in trip_ids:
        trip = load_trip(str(repo_root), "student_test", tid)
        g = RoadNetworkGraph(trip["nodes"], trip["arcs"])
        cache = ShortestPathCache(g)

        t_df = cand_df[cand_df["trip_id"].astype(str) == tid]
        res = decoder.decode_member1_trip(
            trip=trip,
            trip_candidates_df=t_df,
            prob_col="pred_prob",
            graph=g,
            sp_cache=cache,
        )

        # Reference observation-level ground truth aligned by obs_idx from Member 1's table
        obs_true_map = (
            t_df.drop_duplicates("obs_idx").set_index("obs_idx")["true_arc_id"].to_dict()
        )
        ref_obs_arcs = [int(obs_true_map.get(oidx, -1)) for oidx in res.obs_indices]
        ref_route = trip["route"].tolist() if trip["route"] is not None else None

        ablation = compare_classifier_vs_hmm(
            graph=g,
            decode_result=res,
            reference_obs_arcs=ref_obs_arcs,
            reference_route_arcs=ref_route,
            sp_cache=cache,
        )

        row = {
            "trip_id": tid,
            "n_obs": res.diagnostics["n_obs"],
            "n_gps_gaps": res.diagnostics["n_gps_gaps"],
            "n_seq_breaks": res.diagnostics["n_sequence_breaks"],
            "n_disc_fallbacks": res.diagnostics["n_disconnected_fallbacks"],
            "clf_obs_top1": ablation["classifier_only"]["obs_top1_accuracy"],
            "hmm_obs_top1": ablation["classifier_plus_hmm"]["obs_top1_accuracy"],
            "delta_obs_top1": ablation["delta_obs_top1_accuracy"],
            "clf_direct_cont": ablation["classifier_only"]["continuity"]["direct_continuity"],
            "hmm_direct_cont": ablation["classifier_plus_hmm"]["obs_continuity"]["direct_continuity"],
            "hmm_route_direct_cont": ablation["classifier_plus_hmm"]["route_continuity"]["direct_continuity"],
            "delta_direct_cont": ablation["delta_direct_continuity"],
            "clf_lcs_f1": ablation["classifier_only"]["sequence"]["lcs_f1"],
            "hmm_lcs_f1": ablation["classifier_plus_hmm"]["sequence"]["lcs_f1"],
            "delta_lcs_f1": ablation["delta_lcs_f1"],
            "decode_time_s": res.diagnostics["decode_time_s"],
            "dijkstra_runs": res.diagnostics["cache_stats"]["dijkstra_runs"],
        }
        rows.append(row)
        print(
            f"  Trip {tid}: Top1 {row['clf_obs_top1']:.3f} -> {row['hmm_obs_top1']:.3f} "
            f"(+{row['delta_obs_top1']:.3f}) | Cont {row['clf_direct_cont']:.3f} -> {row['hmm_direct_cont']:.3f} | "
            f"LCS-F1 {row['clf_lcs_f1']:.3f} -> {row['hmm_lcs_f1']:.3f} | {row['decode_time_s']:.2f}s"
        )

        if tid in map_trips:
            fmap = create_trip_map(
                graph=g,
                track_lonlat=trip["track"][:, :2],
                timestamps=trip["track"][:, 2],
                trip_id=tid,
                reference_route_arcs=ref_route,
                classifier_only_arcs=res.classifier_only_obs_arcs,
                predicted_route_arcs=res.reconstructed_route_arcs,
                metrics_summary={
                    "Model": "MLP (Model 5) + HMM",
                    "Clf-Only Top-1": f"{row['clf_obs_top1']*100:.2f}%",
                    "Clf+HMM Top-1": f"{row['hmm_obs_top1']*100:.2f}%",
                    "Clf-Only Cont.": f"{row['clf_direct_cont']*100:.2f}%",
                    "Clf+HMM Cont.": f"{row['hmm_direct_cont']*100:.2f}%",
                    "Route LCS F1": f"{row['hmm_lcs_f1']*100:.2f}%",
                },
            )
            save_trip_map(fmap, maps_dir / f"student_test_{tid}_mlp_hmm.html")

    df_out = pd.DataFrame(rows)
    out_logs = repo_root / "outputs" / "member3" / "logs"
    out_logs.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_logs / "hmm_ablation_mlp_student_test.csv", index=False)

    # Weighted observation-level averages + macro trip averages
    total_obs = df_out["n_obs"].sum()
    summary = {
        "source_model": "Model 5: MLP Neural Network (untuned baseline HMM config)",
        "hmm_config": cfg.to_dict(),
        "n_trips": int(len(df_out)),
        "total_obs": int(total_obs),
        "weighted_clf_obs_top1": float((df_out["clf_obs_top1"] * df_out["n_obs"]).sum() / total_obs),
        "weighted_hmm_obs_top1": float((df_out["hmm_obs_top1"] * df_out["n_obs"]).sum() / total_obs),
        "mean_trip_clf_obs_top1": float(df_out["clf_obs_top1"].mean()),
        "mean_trip_hmm_obs_top1": float(df_out["hmm_obs_top1"].mean()),
        "mean_trip_clf_direct_cont": float(df_out["clf_direct_cont"].mean()),
        "mean_trip_hmm_direct_cont": float(df_out["hmm_direct_cont"].mean()),
        "mean_trip_hmm_route_direct_cont": float(df_out["hmm_route_direct_cont"].mean()),
        "mean_trip_clf_lcs_f1": float(df_out["clf_lcs_f1"].mean()),
        "mean_trip_hmm_lcs_f1": float(df_out["hmm_lcs_f1"].mean()),
        "mean_decode_time_s": float(df_out["decode_time_s"].mean()),
        "total_decode_time_s": float(df_out["decode_time_s"].sum()),
    }
    (out_logs / "hmm_ablation_mlp_student_test.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== Aggregate Student-Test Ablation (MLP vs MLP+HMM) ===")
    for k, v in summary.items():
        if k != "hmm_config":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
