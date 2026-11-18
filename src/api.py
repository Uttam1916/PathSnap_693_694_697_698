"""
PathSnap - Evaluator API Backend & Interface
Hosts the trained LightGBM model, HMM/Viterbi route reconstructor,
and serves both REST API endpoints and the minimal evaluator UI.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# Internal imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import generate_trip_features, load_trip, build_adjacency
from hmm_viterbi import HMMRouteReconstructor
from visualizations import create_folium_trajectory_map
from model_training import FEATURE_COLS

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="High-performance map-matching and sequential route reconstruction service",
    version="1.0.0"
)

# Enable CORS for two-laptop deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & reconstructor cache
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_tuned.pkl")
STUDENT_TEST_DIR = os.path.join(BASE_DIR, "Student folder", "student_test")
EVALUATOR_TEST_DIR = os.path.join(BASE_DIR, "Student folder", "evaluator_test")
TRAIN_DIR = os.path.join(BASE_DIR, "Student folder", "train")
MAPS_DIR = os.path.join(BASE_DIR, "visualizations", "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

MODEL = None
RECONSTRUCTOR = None


@app.on_event("startup")
def load_artifacts():
    global MODEL, RECONSTRUCTOR
    print("Loading PathSnap model artifacts...")
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        print(f"Loaded tuned model from {MODEL_PATH}")
    elif os.path.exists(os.path.join(BASE_DIR, "models", "xgboost.pkl")):
        MODEL = joblib.load(os.path.join(BASE_DIR, "models", "xgboost.pkl"))
        print("Fallback: Loaded XGBoost model")
    else:
        print("Warning: No model file found in models/")
    
    RECONSTRUCTOR = HMMRouteReconstructor()
    print("HMM Route Reconstructor initialized.")


class EvaluateRequest(BaseModel):
    trip_id: str
    custom_dir: Optional[str] = None


def find_trip_directory(trip_id: str, custom_dir: Optional[str] = None) -> tuple:
    """Finds the dataset directory containing the requested trip_id."""
    clean_id = str(trip_id).zfill(8)
    
    if custom_dir and os.path.exists(os.path.join(custom_dir, clean_id)):
        return custom_dir, clean_id
    
    # 1. Check evaluator_test
    if os.path.exists(os.path.join(EVALUATOR_TEST_DIR, clean_id)):
        return EVALUATOR_TEST_DIR, clean_id
    
    # 2. Check student_test
    if os.path.exists(os.path.join(STUDENT_TEST_DIR, clean_id)):
        return STUDENT_TEST_DIR, clean_id
    
    # 3. Check train
    if os.path.exists(os.path.join(TRAIN_DIR, clean_id)):
        return TRAIN_DIR, clean_id
    
    raise HTTPException(status_code=404, detail=f"Trip ID '{clean_id}' not found in any dataset partition.")


@app.get("/health")
def health_check():
    """Health check endpoint confirming API readiness."""
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "model_class": MODEL.__class__.__name__ if MODEL else None,
        "available_partitions": {
            "evaluator_test": os.path.exists(EVALUATOR_TEST_DIR),
            "student_test": os.path.exists(STUDENT_TEST_DIR),
            "train": os.path.exists(TRAIN_DIR)
        }
    }


@app.get("/api/trips")
def list_available_trips():
    """Lists available trip IDs for evaluator selection."""
    if os.path.exists(EVALUATOR_TEST_DIR):
        target_dir = EVALUATOR_TEST_DIR
        partition_name = "evaluator_test"
    else:
        target_dir = STUDENT_TEST_DIR
        partition_name = "student_test"
    
    trips = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    return {
        "partition": partition_name,
        "count": len(trips),
        "trips": trips
    }


@app.post("/api/evaluate")
def evaluate_trip(req: EvaluateRequest):
    """
    Main evaluation pipeline endpoint:
    1. Loads raw track, nodes, arcs
    2. Generates candidates & 15 features
    3. Runs classifier inference
    4. Runs HMM/Viterbi route reconstruction
    5. Calculates server-side accuracy & continuity
    6. Generates interactive Folium map
    """
    t_start = time.time()
    trip_dir, clean_id = find_trip_directory(req.trip_id, req.custom_dir)
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on server.")
    
    # 1. Feature generation
    try:
        features_df, meta = generate_trip_features(trip_dir, clean_id, max_candidates=15)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating features: {str(e)}")
    
    if len(features_df) == 0:
        raise HTTPException(status_code=400, detail=f"No candidates generated for trip {clean_id}.")
    
    # 2. Classifier inference
    X = features_df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)
        scores = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
    else:
        scores = MODEL.predict(X).astype(float)
    
    features_df['cls_prob'] = scores
    
    # 3. Structure candidate sets for HMM
    obs_list = sorted(features_df['obs_idx'].unique())
    cands_by_obs = {}
    cls_probs_dict = {}
    cls_best_preds = []
    
    for o in obs_list:
        grp = features_df[features_df['obs_idx'] == o]
        best_row = grp.loc[grp['cls_prob'].idxmax()]
        cls_best_preds.append(int(best_row['arc_id']))
        
        cands_by_obs[int(o)] = [(int(r['arc_id']), float(r['distance'])) for _, r in grp.iterrows()]
        for _, r in grp.iterrows():
            cls_probs_dict[(int(o), int(r['arc_id']))] = float(r['cls_prob'])
    
    # 4. HMM / Viterbi sequential decoding
    pred_path = RECONSTRUCTOR.decode(
        obs_list, cands_by_obs, meta['arc_info'], meta['adj'], meta['track'], cls_probs_dict
    )
    
    # 5. Server-side evaluation against reference route
    gt_assigned = meta['gt_assigned']
    accuracy = None
    if gt_assigned is not None and len(gt_assigned) > 0:
        correct = 0
        total = 0
        for i, o in enumerate(obs_list):
            if o < len(gt_assigned):
                target = gt_assigned[o]
                if pred_path[i] == target:
                    correct += 1
                total += 1
        accuracy = float(correct / max(total, 1))
    
    # 6. Route network continuity
    continuity = 1.0
    if len(pred_path) > 1:
        connected = 0
        arc_info = meta['arc_info']
        adj = meta['adj']
        for i in range(len(pred_path) - 1):
            a1, a2 = pred_path[i], pred_path[i + 1]
            if a1 == a2:
                connected += 1
            elif a1 in arc_info and a2 in arc_info:
                to1 = arc_info[a1]['to_node']
                from2 = arc_info[a2]['from_node']
                if to1 == from2 or from2 in [n for n, _ in adj.get(to1, [])]:
                    connected += 1
        continuity = float(connected / (len(pred_path) - 1))
    
    # 7. Generate interactive Folium map
    map_filename = f"eval_{clean_id}_map.html"
    map_path = os.path.join(MAPS_DIR, map_filename)
    create_folium_trajectory_map(trip_dir, clean_id, predictions_arcs=pred_path, output_html=map_path)
    
    elapsed_ms = (time.time() - t_start) * 1000.0
    
    # Deduplicate consecutive arcs for clean sequence display
    compact_route = []
    for aid in pred_path:
        if not compact_route or compact_route[-1] != aid:
            compact_route.append(aid)
    
    return {
        "status": "success",
        "trip_id": clean_id,
        "n_observations": len(obs_list),
        "n_unique_arcs": len(compact_route),
        "pointwise_accuracy": round(accuracy * 100.0, 2) if accuracy is not None else "N/A (No Ground Truth)",
        "route_continuity": round(continuity * 100.0, 2),
        "processing_time_ms": round(elapsed_ms, 1),
        "predicted_route": compact_route,
        "map_url": f"/map/{clean_id}"
    }


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def get_map(trip_id: str):
    """Serves the generated interactive Folium map for the trip."""
    clean_id = str(trip_id).zfill(8)
    map_filename = f"eval_{clean_id}_map.html"
    map_path = os.path.join(MAPS_DIR, map_filename)
    
    # Fallback to test map if eval map not yet generated
    if not os.path.exists(map_path):
        map_path = os.path.join(MAPS_DIR, f"test_trip_{clean_id}_map.html")
    
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    raise HTTPException(status_code=404, detail=f"Map for trip {clean_id} has not been generated yet.")


# Minimal, Clean, Distraction-Free Evaluator UI (No Gradients, No Emojis, Clean Neutral Design)
HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PathSnap Evaluator Interface</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #f9fafb;
      color: #111827;
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    h1 { font-size: 20px; font-weight: 600; color: #111827; }
    .subtitle { font-size: 13px; color: #6b7280; margin-top: 4px; }
    .card {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .form-group {
      display: flex;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
    }
    .form-item { flex: 1; min-width: 220px; }
    label { display: block; font-size: 12px; font-weight: 500; color: #374151; margin-bottom: 6px; }
    select, input, button {
      width: 100%;
      height: 38px;
      padding: 0 12px;
      font-size: 14px;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      background: #ffffff;
      color: #111827;
      outline: none;
    }
    select:focus, input:focus { border-color: #111827; }
    button {
      background-color: #111827;
      color: #ffffff;
      font-weight: 500;
      border: none;
      cursor: pointer;
      width: auto;
      min-width: 160px;
    }
    button:hover { background-color: #374151; }
    button:disabled { background-color: #9ca3af; cursor: not-allowed; }
    
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .stat-box {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      padding: 16px;
    }
    .stat-label { font-size: 11px; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-value { font-size: 24px; font-weight: 600; color: #111827; margin-top: 6px; }
    
    .map-container {
      width: 100%;
      height: 580px;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      overflow: hidden;
      background: #f3f4f6;
    }
    iframe { width: 100%; height: 100%; border: none; }
    
    .log-box {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      background: #111827;
      color: #f3f4f6;
      padding: 16px;
      border-radius: 4px;
      overflow-x: auto;
      max-height: 200px;
    }
    .status-badge {
      display: inline-block;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 500;
      border-radius: 3px;
      background: #e5e7eb;
      color: #374151;
    }
    .badge-ok { background: #dcfce7; color: #166534; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>PathSnap: Map-Matching Evaluator System</h1>
      <div class="subtitle">Distributed Trajectory snaps & sequential route reconstruction · Laptop 2 Interface</div>
    </header>

    <div class="card">
      <div class="form-group">
        <div class="form-item">
          <label for="tripSelect">Select Evaluation Trip</label>
          <select id="tripSelect">
            <option value="">Loading available trips...</option>
          </select>
        </div>
        <div class="form-item" style="max-width: 240px;">
          <label for="manualTripId">Or Enter Trip ID</label>
          <input type="text" id="manualTripId" placeholder="e.g. 00000000">
        </div>
        <div class="form-item" style="flex: none;">
          <button id="evalBtn" onclick="runEvaluation()">Evaluate Trip</button>
        </div>
      </div>
    </div>

    <div id="resultsSection" style="display: none;">
      <div class="grid">
        <div class="stat-box">
          <div class="stat-label">Pointwise Accuracy</div>
          <div class="stat-value" id="resAccuracy">--</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Route Continuity</div>
          <div class="stat-value" id="resContinuity">--</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">GPS Observations</div>
          <div class="stat-value" id="resObs">--</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Latency (ms)</div>
          <div class="stat-value" id="resTime">--</div>
        </div>
      </div>

      <div class="card" style="padding: 0; overflow: hidden; margin-bottom: 20px;">
        <div class="map-container">
          <iframe id="mapFrame" src="about:blank"></iframe>
        </div>
      </div>

      <div class="card">
        <label style="margin-bottom: 8px;">Server Response & Decoded Route Preview</label>
        <pre class="log-box" id="jsonOutput">// Evaluator API logs will appear here</pre>
      </div>
    </div>
  </div>

  <script>
    const API_BASE = window.location.origin;

    async function loadTrips() {
      try {
        const res = await fetch(`${API_BASE}/api/trips`);
        const data = await res.json();
        const sel = document.getElementById('tripSelect');
        sel.innerHTML = '';
        data.trips.forEach(t => {
          const opt = document.createElement('option');
          opt.value = t;
          opt.textContent = `Trip ${t} (${data.partition})`;
          sel.appendChild(opt);
        });
      } catch (err) {
        document.getElementById('tripSelect').innerHTML = '<option value="00000000">00000000 (Default)</option>';
      }
    }

    async function runEvaluation() {
      const selectVal = document.getElementById('tripSelect').value;
      const manualVal = document.getElementById('manualTripId').value.trim();
      const tripId = manualVal || selectVal || "00000000";

      const btn = document.getElementById('evalBtn');
      btn.disabled = true;
      btn.textContent = 'Processing...';

      const resultsDiv = document.getElementById('resultsSection');
      resultsDiv.style.display = 'block';

      try {
        const response = await fetch(`${API_BASE}/api/evaluate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ trip_id: tripId })
        });

        const data = await response.json();
        btn.disabled = false;
        btn.textContent = 'Evaluate Trip';

        if (!response.ok) {
          alert('Evaluation error: ' + (data.detail || 'Server error'));
          return;
        }

        document.getElementById('resAccuracy').textContent = typeof data.pointwise_accuracy === 'number' ? data.pointwise_accuracy + '%' : data.pointwise_accuracy;
        document.getElementById('resContinuity').textContent = data.route_continuity + '%';
        document.getElementById('resObs').textContent = data.n_observations.toLocaleString();
        document.getElementById('resTime').textContent = data.processing_time_ms + ' ms';

        document.getElementById('mapFrame').src = `${API_BASE}${data.map_url}`;
        document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);

      } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Evaluate Trip';
        alert('Network request failed: ' + e.message);
      }
    }

    window.onload = loadTrips;
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the evaluator user interface."""
    return HTMLResponse(content=HTML_UI)


if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so Laptop 2 on the local network can connect
    print("Starting PathSnap API Server on http://0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
