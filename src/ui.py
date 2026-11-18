"""
PathSnap - Laptop 2 Evaluator Client
Connects to Laptop 1 API, displays available trips, and triggers evaluation requests.
Can be executed as a command-line evaluator tool or used to open the browser UI.
"""

import sys
import os
import time
import json
import urllib.request
import urllib.parse
import webbrowser

DEFAULT_API_URL = "http://localhost:8000"


def print_header():
    print("=" * 80)
    print("PATHSNAP EVALUATOR INTERFACE (LAPTOP 2)")
    print("=" * 80)


def test_connection(api_url):
    print(f"Connecting to API at {api_url}/health ...")
    try:
        req = urllib.request.Request(f"{api_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print("Connection successful! Server status: OK")
            print(f"Loaded Model: {data.get('model_class')}")
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure Laptop 1 is running: uvicorn src.api:app --host 0.0.0.0 --port 8000")
        return False


def get_available_trips(api_url):
    try:
        req = urllib.request.Request(f"{api_url}/api/trips")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get("trips", [])
    except Exception:
        return []


def evaluate_trip(api_url, trip_id):
    print(f"\nSubmitting evaluation request for Trip [{trip_id}]...")
    t0 = time.time()
    payload = json.dumps({"trip_id": trip_id}).encode('utf-8')
    req = urllib.request.Request(
        f"{api_url}/api/evaluate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - t0
            print("-" * 80)
            print("EVALUATION RESULTS RECEIVED:")
            print(f"  Trip ID:             {data['trip_id']}")
            print(f"  GPS Observations:    {data['n_observations']:,}")
            print(f"  Pointwise Accuracy:  {data['pointwise_accuracy']}%")
            print(f"  Route Continuity:    {data['route_continuity']}%")
            print(f"  Server Latency:      {data['processing_time_ms']} ms (Total roundtrip: {elapsed*1000:.1f} ms)")
            print(f"  Predicted Arcs:      {len(data['predicted_route'])} unique segments")
            print(f"  Interactive Map URL: {api_url}{data['map_url']}")
            print("-" * 80)
            return data
    except urllib.error.HTTPError as he:
        err = he.read().decode('utf-8')
        print(f"API Error ({he.code}): {err}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def main():
    print_header()
    api_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_API_URL
    
    if not test_connection(api_url):
        sys.exit(1)
    
    trips = get_available_trips(api_url)
    print(f"\nAvailable Evaluation Trips ({len(trips)} found):")
    for i, t in enumerate(trips[:10]):
        print(f"  [{i+1}] {t}")
    
    if len(sys.argv) > 2:
        selected_trip = sys.argv[2]
        evaluate_trip(api_url, selected_trip)
    else:
        print("\nOptions:")
        print(f"1. Open Web UI in browser: {api_url}")
        print("2. Run evaluation in console for first trip")
        choice = input("\nEnter trip ID to evaluate (or press Enter to open Web UI): ").strip()
        
        if not choice:
            print(f"Opening {api_url} in your default browser...")
            webbrowser.open(api_url)
        else:
            clean_id = str(choice).zfill(8)
            evaluate_trip(api_url, clean_id)


if __name__ == "__main__":
    main()
