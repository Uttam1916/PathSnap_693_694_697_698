"""
PathSnap - Evaluator API Backend & Interface
Hosts the trained LightGBM model, HMM/Viterbi route reconstructor,
and serves both REST API endpoints and the minimal evaluator UI.
Strict Zero-Leakage Policy: Evaluator data is used purely for inference and evaluation.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Request
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

# Global paths & caching
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_tuned.pkl")

# Path discovery for partitions
if os.path.exists(os.path.join(BASE_DIR, "evaluator_test")):
    EVALUATOR_TEST_DIR = os.path.join(BASE_DIR, "evaluator_test")
else:
    EVALUATOR_TEST_DIR = os.path.join(BASE_DIR, "Student folder", "evaluator_test")

if os.path.exists(os.path.join(BASE_DIR, "Student folder", "student_test")):
    STUDENT_TEST_DIR = os.path.join(BASE_DIR, "Student folder", "student_test")
else:
    STUDENT_TEST_DIR = os.path.join(BASE_DIR, "student_test")

if os.path.exists(os.path.join(BASE_DIR, "Student folder", "train")):
    TRAIN_DIR = os.path.join(BASE_DIR, "Student folder", "train")
else:
    TRAIN_DIR = os.path.join(BASE_DIR, "train")

MAPS_DIR = os.path.join(BASE_DIR, "visualizations", "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

MODEL = None
RECONSTRUCTOR = None
EVALUATION_CACHE = {}


@app.on_event("startup")
def load_artifacts():
    global MODEL, RECONSTRUCTOR
    print("=" * 65)
    print(" PATHSNAP DISTRIBUTED MAP-MATCHING SYSTEM — SERVER STARTUP")
    print("=" * 65)
    print("Loading PathSnap model artifacts...")
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        print(f" [OK] Loaded tuned model from {MODEL_PATH}")
    elif os.path.exists(os.path.join(BASE_DIR, "models", "xgboost.pkl")):
        MODEL = joblib.load(os.path.join(BASE_DIR, "models", "xgboost.pkl"))
        print(" [OK] Fallback: Loaded XGBoost model")
    else:
        print(" [WARN] No model file found in models/")
    
    RECONSTRUCTOR = HMMRouteReconstructor()
    print(" [OK] HMM Route Reconstructor initialized.")
    print(f" [OK] Evaluator test directory: {EVALUATOR_TEST_DIR}")
    print(f" [OK] Student test directory:   {STUDENT_TEST_DIR}")
    print("=" * 65)


class EvaluateRequest(BaseModel):
    trip_id: str
    custom_dir: Optional[str] = None
    force_refresh: Optional[bool] = False


class BatchEvaluateRequest(BaseModel):
    partition: Optional[str] = "evaluator_test"
    force_refresh: Optional[bool] = False


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


@app.api_route("/health", methods=["GET", "HEAD"])
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
def list_available_trips(partition: Optional[str] = Query(None)):
    """Lists available trip IDs for evaluator selection."""
    if partition == "student_test" and os.path.exists(STUDENT_TEST_DIR):
        target_dir = STUDENT_TEST_DIR
        partition_name = "student_test"
    elif partition == "train" and os.path.exists(TRAIN_DIR):
        target_dir = TRAIN_DIR
        partition_name = "train"
    elif os.path.exists(EVALUATOR_TEST_DIR):
        target_dir = EVALUATOR_TEST_DIR
        partition_name = "evaluator_test"
    else:
        target_dir = STUDENT_TEST_DIR
        partition_name = "student_test"
    
    trips = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    
    available_parts = []
    if os.path.exists(EVALUATOR_TEST_DIR):
        available_parts.append("evaluator_test")
    if os.path.exists(STUDENT_TEST_DIR):
        available_parts.append("student_test")
    if os.path.exists(TRAIN_DIR):
        available_parts.append("train")
        
    return {
        "partition": partition_name,
        "count": len(trips),
        "trips": trips,
        "available_partitions": available_parts
    }


def _run_single_trip_inference(trip_dir: str, clean_id: str) -> dict:
    """Internal helper to execute full pipeline inference on a trip."""
    t_start = time.time()
    
    # 1. Feature generation
    features_df, meta = generate_trip_features(trip_dir, clean_id, max_candidates=15)
    if len(features_df) == 0:
        raise HTTPException(status_code=400, detail=f"No candidate segments generated for trip {clean_id}.")
    
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
    
    for o in obs_list:
        grp = features_df[features_df['obs_idx'] == o]
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
    
    acc_val = round(accuracy * 100.0, 2) if accuracy is not None else None
    cont_val = round(continuity * 100.0, 2)
    
    result = {
        "status": "success",
        "trip_id": clean_id,
        "partition": os.path.basename(trip_dir),
        "n_observations": len(obs_list),
        "n_unique_arcs": len(compact_route),
        "pointwise_accuracy": acc_val if acc_val is not None else "N/A (No Ground Truth)",
        "route_continuity": cont_val,
        "processing_time_ms": round(elapsed_ms, 1),
        "predicted_route": compact_route,
        "map_url": f"/map/{clean_id}"
    }
    return result


@app.post("/api/evaluate")
def evaluate_trip(req: EvaluateRequest):
    """
    Main evaluation pipeline endpoint:
    Processes requested trip, runs LightGBM + HMM route reconstruction,
    computes server-side metrics, and logs live feedback to stdout.
    """
    trip_dir, clean_id = find_trip_directory(req.trip_id, req.custom_dir)
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on server.")
    
    # Check cache unless force_refresh is requested
    cache_key = f"{clean_id}_{os.path.basename(trip_dir)}"
    if not req.force_refresh and cache_key in EVALUATION_CACHE:
        cached = EVALUATION_CACHE[cache_key].copy()
        print(f"[CACHE HIT] Trip {clean_id} served in <1ms (Accuracy: {cached.get('pointwise_accuracy')}%, Continuity: {cached.get('route_continuity')}%)")
        return cached

    print(f"\n[EVALUATION REQUEST] Processing Trip: {clean_id} | Partition: {os.path.basename(trip_dir)}")
    t0 = time.time()
    
    try:
        result = _run_single_trip_inference(trip_dir, clean_id)
        EVALUATION_CACHE[cache_key] = result
        
        print(f" -> Observations: {result['n_observations']} | Reconstructed Arcs: {result['n_unique_arcs']}")
        print(f" -> Pointwise Accuracy: {result['pointwise_accuracy']}% | Route Continuity: {result['route_continuity']}%")
        print(f" -> Latency: {result['processing_time_ms']} ms | Map: {result['map_url']}")
        print(f"[COMPLETED] Trip {clean_id} evaluated successfully in {time.time() - t0:.2f}s\n")
        return result
    except Exception as e:
        print(f"[ERROR] Evaluation failed for Trip {clean_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline evaluation error: {str(e)}")


@app.post("/api/evaluate_all")
def evaluate_all_trips(req: BatchEvaluateRequest):
    """
    Batch evaluation endpoint:
    Runs inference sequentially across all trips in the partition and returns summary metrics.
    """
    if req.partition == "student_test" and os.path.exists(STUDENT_TEST_DIR):
        target_dir = STUDENT_TEST_DIR
        part_name = "student_test"
    elif os.path.exists(EVALUATOR_TEST_DIR):
        target_dir = EVALUATOR_TEST_DIR
        part_name = "evaluator_test"
    else:
        target_dir = STUDENT_TEST_DIR
        part_name = "student_test"
        
    trips = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    print(f"\n[BATCH EVALUATION] Running batch inference across {len(trips)} trips in '{part_name}'...")
    
    results = []
    accuracies = []
    continuities = []
    
    for t in trips:
        cache_key = f"{t}_{part_name}"
        if not req.force_refresh and cache_key in EVALUATION_CACHE:
            res = EVALUATION_CACHE[cache_key]
        else:
            res = _run_single_trip_inference(target_dir, t)
            EVALUATION_CACHE[cache_key] = res
        
        results.append({
            "trip_id": res["trip_id"],
            "n_observations": res["n_observations"],
            "n_unique_arcs": res["n_unique_arcs"],
            "pointwise_accuracy": res["pointwise_accuracy"],
            "route_continuity": res["route_continuity"],
            "processing_time_ms": res["processing_time_ms"]
        })
        
        if isinstance(res["pointwise_accuracy"], (int, float)):
            accuracies.append(res["pointwise_accuracy"])
        if isinstance(res["route_continuity"], (int, float)):
            continuities.append(res["route_continuity"])
            
    mean_acc = round(float(np.mean(accuracies)), 2) if accuracies else None
    mean_cont = round(float(np.mean(continuities)), 2) if continuities else None
    
    print(f"[BATCH COMPLETE] Partition: {part_name} | Mean Accuracy: {mean_acc}% | Mean Continuity: {mean_cont}%\n")
    
    return {
        "status": "success",
        "partition": part_name,
        "total_trips": len(trips),
        "mean_accuracy": mean_acc,
        "mean_continuity": mean_cont,
        "results": results
    }


@app.api_route("/map/{trip_id}", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_map(trip_id: str):
    """Serves the generated interactive Folium map for the trip."""
    clean_id = str(trip_id).zfill(8)
    map_filename = f"eval_{clean_id}_map.html"
    map_path = os.path.join(MAPS_DIR, map_filename)
    
    # Fallback to test map if eval map not yet generated
    if not os.path.exists(map_path):
        map_path = os.path.join(MAPS_DIR, f"test_trip_{clean_id}_map.html")
    
    # If still not found, generate the trajectory map dynamically
    if not os.path.exists(map_path):
        try:
            trip_dir, clean_id = find_trip_directory(clean_id)
            create_folium_trajectory_map(trip_dir, clean_id, predictions_arcs=None, output_html=map_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate map for trip {clean_id}: {str(e)}")
    
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            content = f.read()
        resp = HTMLResponse(content=content)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        return resp
    
    raise HTTPException(status_code=404, detail=f"Map for trip {clean_id} could not be loaded.")


# Minimal, Clean, Distraction-Free Evaluator UI (Loaded from src/ui.html)
UI_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def serve_ui():
    """Serves the evaluator user interface with no-cache headers."""
    if os.path.exists(UI_HTML_PATH):
        with open(UI_HTML_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        resp = HTMLResponse(content=content)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    return HTMLResponse(content="<h1>PathSnap Evaluator Interface</h1><p>ui.html not found.</p>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so Laptop 2 on the local network can connect
    print("Starting PathSnap API Server on http://0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
