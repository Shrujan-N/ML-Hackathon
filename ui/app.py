"""
app.py — Streamlit Evaluator UI (Laptop 2) for PathSnap ML Hackathon.

Connects to Laptop 1 FastAPI backend to:
- Test individual trips from student_test partition
- Run 10-trip LAN stress test
- Display Top-1 accuracy, Macro F1, Log Loss, and inference latency
- Render interactive Folium maps comparing GPS tracks to predicted route arcs
"""

import os
import sys
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure repo root and ui folder are in python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ui.api_client import predict_trip
from ui.map_builder import build_folium_map

st.set_page_config(
    page_title="PathSnap — Model Evaluation & Route Matching",
    page_icon="🗺️",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🛰️ PathSnap Evaluator")
st.sidebar.markdown("**Member 4 — Distributed Deployment**")

api_url = st.sidebar.text_input(
    "FastAPI Backend URL (Laptop 1)",
    value="http://localhost:8000",
    help="Enter the IP address of Laptop 1, e.g., http://192.168.1.100:8000"
)

# Load test data candidates
DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "student_test_candidates.csv")
if not os.path.exists(DATA_PATH):
    st.error(f"Cannot find student test data at {DATA_PATH}")
    st.stop()

@st.cache_data
def load_candidates_data():
    return pd.read_csv(DATA_PATH)

cand_df = load_candidates_data()
all_trips = sorted(cand_df["trip_id"].unique())

st.sidebar.markdown("---")
selected_trip = st.sidebar.selectbox("Select Evaluation Trip", all_trips)

col_b1, col_b2 = st.sidebar.columns(2)
run_single = col_b1.button("🚀 Predict Trip", use_container_width=True)
run_stress = col_b2.button("⚡ Stress Test 10", use_container_width=True)

# ---------------------------------------------------------
# Main Panel
# ---------------------------------------------------------
st.title("🗺️ Distributed Route Matcher & Evaluator")
st.caption(f"Connecting to Backend: `{api_url}` | Active Dataset: `student_test` ({len(all_trips)} trips)")

if run_stress:
    st.subheader("⚡ 10-Trip LAN Stress Test Results")
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    results = []
    total_start = time.time()

    for idx, tid in enumerate(all_trips):
        status_text.text(f"Evaluating Trip {tid} ({idx+1}/{len(all_trips)})...")
        trip_sub = cand_df[cand_df["trip_id"] == tid]
        
        t0 = time.time()
        res = predict_trip(api_url, tid, trip_sub)
        t_elapsed = time.time() - t0

        if "error" in res:
            results.append({
                "trip_id": tid,
                "n_obs": trip_sub["obs_idx"].nunique(),
                "status": f"❌ Error: {res['error']}",
                "top1_acc": 0.0,
                "latency_s": round(t_elapsed, 2)
            })
        else:
            preds = res.get("predictions", [])
            pred_map = {p["obs_idx"]: p["predicted_arc_id"] for p in preds}
            
            # Match with true labels if available in candidate table
            obs_trues = trip_sub[trip_sub["label"] == 1].set_index("obs_idx")["candidate_arc_id"].to_dict()
            correct = sum(1 for o_idx, p_arc in pred_map.items() if obs_trues.get(o_idx) == p_arc)
            acc = correct / max(1, len(pred_map))

            results.append({
                "trip_id": tid,
                "n_obs": len(preds),
                "status": "✅ Success",
                "top1_acc": round(acc * 100, 2),
                "server_ms": res.get("inference_time_ms", 0),
                "roundtrip_s": round(t_elapsed, 3)
            })

        progress_bar.progress((idx + 1) / len(all_trips))

    total_time = round(time.time() - total_start, 2)
    status_text.text(f"Stress test complete in {total_time}s across all {len(all_trips)} trips!")

    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)

    avg_acc = res_df["top1_acc"].mean() if "top1_acc" in res_df else 0
    st.metric("Mean Top-1 Accuracy across 10 Trips", f"{avg_acc:.2f}%")

elif run_single or selected_trip:
    trip_sub = cand_df[cand_df["trip_id"] == selected_trip]
    st.markdown(f"### Trip **{selected_trip}** Overview")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("GPS Observations", f"{trip_sub['obs_idx'].nunique()}")
    col_m2.metric("Total Candidates Scored", f"{len(trip_sub)}")

    if run_single:
        with st.spinner(f"Sending Trip {selected_trip} candidates to backend..."):
            t0 = time.time()
            res = predict_trip(api_url, selected_trip, trip_sub)
            latency = time.time() - t0

        if "error" in res:
            st.error(f"API Error: {res['error']}")
        else:
            preds = res.get("predictions", [])
            col_m3.metric("Roundtrip Latency", f"{latency*1000:.1f} ms")

            # Calculate accuracy vs ground truth
            pred_records = []
            obs_trues = trip_sub[trip_sub["label"] == 1].set_index("obs_idx")["candidate_arc_id"].to_dict()
            node_map = trip_sub.groupby("candidate_arc_id")[["from_node", "to_node"]].first()

            correct = 0
            for p in preds:
                oid = p["obs_idx"]
                parc = p["predicted_arc_id"]
                conf = p.get("confidence", 1.0)
                is_correct = (obs_trues.get(oid) == parc)
                if is_correct:
                    correct += 1

                u = node_map.loc[parc, "from_node"] if parc in node_map.index else None
                v = node_map.loc[parc, "to_node"] if parc in node_map.index else None

                pred_records.append({
                    "obs_idx": oid,
                    "predicted_arc_id": parc,
                    "confidence": round(conf, 4),
                    "is_correct": is_correct,
                    "from_node": u,
                    "to_node": v
                })

            acc = (correct / len(preds)) * 100 if preds else 0
            st.success(f"Trip {selected_trip} Prediction Complete — Top-1 Accuracy: **{acc:.2f}%**")

            # Map view & table
            map_tab, data_tab = st.tabs(["🗺️ Interactive Folium Map", "📋 Prediction Table"])

            pred_df = pd.DataFrame(pred_records)
            with map_tab:
                folium_map = build_folium_map(str(selected_trip), pred_df, REPO_ROOT, partition="student_test")
                map_html = folium_map._repr_html_()
                components.html(map_html, height=550)

            with data_tab:
                st.dataframe(pred_df, use_container_width=True)
    else:
        st.info("Click **'🚀 Predict Trip'** in the sidebar to send candidates to Laptop 1 backend and visualize the route.")
