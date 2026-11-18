"""
api_client.py — HTTP client for the PathSnap FastAPI backend.

Exposes:
    predict_trip(api_url, trip_id, candidate_df) -> dict
        Sends the candidate features for a trip to POST /predict and returns
        the parsed JSON response dict.
"""

import time
import requests
import pandas as pd
from typing import Dict, Any

# Feature columns expected by the API (order matters for sanity; schema defines them)
FEATURE_COLS = [
    "dist_to_arc_m", "dist_min_diff_m", "dist_ratio_to_min", "cand_rank",
    "proj_fraction", "dist_to_from_node_m", "dist_to_to_node_m", "arc_length_m",
    "heading_diff_rad", "heading_cos", "heading_sin", "gps_speed_mps",
    "dt_prev_s", "prev_gps_dist_to_arc_m", "next_gps_dist_to_arc_m",
    "is_connected_from_prev_top1",
]

IDENTIFIER_COLS = ["candidate_arc_id", "from_node", "to_node"]


def _build_payload(trip_id: str, candidate_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a filtered candidate DataFrame (one trip) into the PredictRequest
    JSON body expected by the backend.

    Parameters
    ----------
    trip_id : str
        Trip identifier string (e.g. "00000001").
    candidate_df : pd.DataFrame
        Rows from student_test_candidates.csv for this trip, with columns
        including obs_idx, timestamp, candidate_arc_id, from_node, to_node,
        and all 16 feature columns.

    Returns
    -------
    dict  — JSON-serialisable payload.
    """
    observations = []
    for obs_idx, grp in candidate_df.groupby("obs_idx", sort=True):
        # Grab timestamp (same for all candidates in this obs)
        ts = float(grp["timestamp"].iloc[0])
        candidates = []
        for _, row in grp.iterrows():
            cand = {col: float(row[col]) for col in FEATURE_COLS}
            cand["candidate_arc_id"] = int(row["candidate_arc_id"])
            cand["from_node"] = int(row["from_node"])
            cand["to_node"] = int(row["to_node"])
            candidates.append(cand)
        observations.append(
            {"obs_idx": int(obs_idx), "timestamp": ts, "candidates": candidates}
        )

    return {"trip_id": str(trip_id), "observations": observations}


def predict_trip(
    api_url: str,
    trip_id,
    candidate_df: pd.DataFrame,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Call POST {api_url}/predict for one trip.

    Parameters
    ----------
    api_url : str
        Base URL of the backend, e.g. "http://localhost:8000".
    trip_id : int | str
        Trip ID value as found in the DataFrame (int or str).
    candidate_df : pd.DataFrame
        Full student_test_candidates DataFrame (all trips OK — filtered here).
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with keys: trip_id, predictions (list), inference_time_ms (float)
         or {"error": <message>} on failure.
    """
    # Filter to this trip
    trip_df = candidate_df[candidate_df["trip_id"] == trip_id].copy()
    if trip_df.empty:
        return {"error": f"No data found for trip_id={trip_id}"}

    payload = _build_payload(str(trip_id), trip_df)
    endpoint = f"{api_url.rstrip('/')}/predict"

    try:
        t0 = time.time()
        response = requests.post(endpoint, json=payload, timeout=timeout)
        elapsed_ms = (time.time() - t0) * 1000
        response.raise_for_status()
        result = response.json()
        # Attach client-side elapsed time if backend didn't provide it
        if "inference_time_ms" not in result:
            result["inference_time_ms"] = elapsed_ms
        return result
    except requests.exceptions.ConnectionError as exc:
        return {"error": f"Connection refused — is the backend running at {api_url}? ({exc})"}
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {timeout}s"}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"HTTP {response.status_code}: {response.text[:300]}"}
    except Exception as exc:
        return {"error": str(exc)}
