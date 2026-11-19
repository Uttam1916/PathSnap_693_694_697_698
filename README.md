# PathSnap: High-Performance Map-Matching & Sequential Route Reconstruction
**Team Repository: `PathSnap_693_694_697_698`**
## 1. Overview
PathSnap is an end-to-end machine learning system that reconstructs realistic vehicle trajectories by matching noisy, irregularly sampled GPS observations to road network segments. The system combines:
1. **Vectorized Spatial Candidate Generation & Feature Engineering:** Sub-meter spatial grid indexing with 15 geometric, kinematic, and topological features.
2. **Multi-Model Supervised Classification:** 6 distinct algorithms benchmarked across linear, instance-based, ensemble, boosting, and neural network families with 5-fold `GroupKFold` cross-validation (grouped by `trip_id` to strictly prevent inter-trip leakage).
3. **Bayesian Hyperparameter Optimisation:** Gaussian Process surrogate modeling via `BayesSearchCV` tuning LightGBM.
4. **HMM & Log-Space Viterbi Sequential Decoding:** Reconstructing globally connected routes by combining classifier posterior probabilities with road network graph topology.
5. **Interactive Geographical Visualisation:** Interactive Folium maps with layer controls, road networks, GPS points, and reconstructed routes.

---

## 2. Team Members & Contributions
| Roll No / ID | GitHub Username | Key Contribution Area |
|---|---|---|
| **693** | `vineethpmural123-rgb` | Data Preprocessing, 2D Spatial Hash Grid Indexing (64x speedup), Haversine Distance Engine, Candidate Selection Pool |
| **694** | `sudevsuresh` | 15-Feature Engineering, 6-Model Benchmark Suite, 5-Fold GroupKFold Cross-Validation, Metrics & Loss Formulation |
| **697** | `Uttam1916` | Bayesian Hyperparameter Optimization (`BayesSearchCV`), HMM & Log-Space Viterbi Sequential Route Decoder Algorithm |
| **698** | `siddeshvs351` | Two-Laptop Distributed Architecture, FastAPI Production REST Backend, Server-Side Evaluation Engine, Minimal UI Client |

---

## 3. Repository Architecture
```
PathSnap_693_694_697_698/
├── PathSnap_Technical_Report.pdf # Deliverable 2: Complete Academic Technical Report (PDF)
├── generate_report_pdf.py        # PDF builder compiling all guidelines and sections
├── notebooks/                    # Interactive development & walkthrough notebooks
│   └── pathsnap_pipeline_walkthrough.ipynb
├── src/
│   ├── data_processing.py        # Spatial indexing, candidate generation, 15 geometric features
│   ├── model_training.py         # 6-family benchmark suite with GroupKFold cross-validation
│   ├── bayesian_optimization.py  # Gaussian process Bayesian optimization with skopt
│   ├── hmm_viterbi.py            # Hidden Markov Model & Viterbi sequential route decoder
│   ├── visualizations.py         # Folium interactive HTML map engine & diagnostic charts
│   ├── api.py                    # Production FastAPI backend & server-side evaluator
│   ├── ui.py                     # Laptop 2 evaluator client interface
│   └── run_pipeline.py           # Master end-to-end workflow runner
├── models/                       # Serialized model weights & scalers (.pkl)
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── mlp_neural_network.pkl
│   └── lgbm_tuned.pkl
├── results/                      # Evaluation metrics, JSON summaries & trial logs
│   ├── train_summary.csv         # EDA statistics for 80 training trips
│   ├── test_summary.csv          # EDA statistics for 10 student-test trips
│   ├── model_comparison.csv      # Consolidated benchmark metrics across all 6 models
│   ├── model_comparison_detailed.json
│   ├── bayesian_opt_results.json # Best hyperparameters & search configurations
│   ├── bayesian_opt_trials.csv   # History of all Bayesian optimization iterations
│   ├── tuned_vs_untuned.json     # Quantitative uplift from Bayesian tuning
│   └── hmm_comparison.json       # Pointwise accuracy and continuity comparison
├── visualizations/               # Diagnostic plots, grayscale figures & interactive maps
│   ├── generate_grayscale_figures.py # Script generating all clean report figures
│   ├── report_figures/           # 12 clean, publication-grade grayscale figures
│   │   ├── fig1_eda_distributions.png
│   │   ├── fig2_model_f1_comparison.png
│   │   ├── fig3_model_top1_continuity.png
│   │   ├── fig4_model_log_loss.png
│   │   ├── fig5_mlp_training_loss.png
│   │   ├── fig6_confusion_matrices.png
│   │   ├── fig7_computational_efficiency.png
│   │   ├── fig8_bayesian_optimization.png
│   │   ├── fig9_hmm_accuracy_continuity.png
│   │   ├── fig10_folium_geographical_map.png
│   │   ├── fig11_ui_screenshot.png
│   │   └── fig12_cmd_screenshot.png
│   ├── maps/
│   │   ├── test_trip_00000000_map.html   # Interactive Folium map (student-test trip 0)
│   │   ├── train_trip_00000001_map.html  # Interactive Folium map (training trip 1)
│   │   ├── eval_00000000_map.html        # Interactive Folium map (eval trip 00000000)
│   │   ├── eval_00000007_map.html        # Interactive Folium map (eval trip 00000007)
│   │   └── eval_00000009_map.html        # Interactive Folium map (eval trip 00000009)
│   ├── eda_obs_distribution.png
│   ├── eda_trajectory_lengths.png
│   ├── eda_time_gaps.png
│   ├── eda_network_size.png
│   ├── model_f1_comparison.png
│   ├── model_top1_continuity.png
│   ├── model_log_loss.png
│   ├── model_overfitting_analysis.png
│   ├── model_computational_efficiency.png
│   ├── bayesian_convergence.png
│   ├── bayesian_tuned_vs_untuned.png
│   └── hmm_reconstruction_impact.png
├── requirements.txt              # Pinned environment dependencies
├── EVALUATION_GUIDE.md           # Step-by-step evaluator instructions & test checklist
├── VIVA_EXPLANATION.md           # Comprehensive viva preparation guide & rationale
└── README.md                     # This file
```

---

## 4. Setup & Installation

```bash
# Clone repository
git clone https://github.com/Uttam1916/PathSnap_693_694_697_698.git
cd PathSnap_693_694_697_698

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Evaluator Web Interface & Deployment Architecture

PathSnap is designed around the **Two-Laptop Distributed Architecture** specified by the hackathon guidelines. The system cleanly separates the machine learning inference engine and server-side ground-truth evaluation (Laptop 1) from the evaluator-facing presentation client (Laptop 2).

```
┌────────────────────────────────────────┐          HTTP / JSON          ┌────────────────────────────────────────┐
│        LAPTOP 1 (Model Server)         │  ◄──────────────────────────► │        LAPTOP 2 (Evaluator UI)         │
│  • FastAPI REST Service (Port 8000)    │       GET /api/trips          │  • Clean Minimal Web Browser Interface │
│  • Trained LightGBM Model Weights      │       POST /api/evaluate      │  • Evaluator Trip Selector (10 trips)  │
│  • HMM / Log-Space Viterbi Engine      │                               │  • Real-Time Accuracy & Continuity     │
│  • Road Network Graph Topology         │       GET /map/{trip_id}      │  • Interactive Folium Map Viewer       │
│  • Internal Server-Side Ground Truth   │  ◄──────────────────────────► │  • Traversed Arcs & JSON Inspector     │
│    (Zero leakage to client)            │       Interactive Map HTML    │  • Zero-Dependency (100% Offline)      │
└────────────────────────────────────────┘                               └────────────────────────────────────────┘
```

---

### A. How to Run the System (Two-Laptop Evaluation Mode)

#### 1. On Laptop 1 (Model & FastAPI Backend):
1. Connect Laptop 1 and Laptop 2 to the same local Wi-Fi, Ethernet, or mobile hotspot.
2. Note Laptop 1's local IP address:
   ```bash
   # On Linux:
   hostname -I | awk '{print $1}'
   # On macOS:
   ipconfig getifaddr en0
   # On Windows (PowerShell):
   (Get-NetIPAddress -AddressFamily IPv4 | Where-Object InterfaceAlias -notlike "*Loopback*").IPAddress
   ```
   *(Example: `192.168.1.105`)*
3. Activate the environment and launch the API server:
   ```bash
   source venv/bin/activate
   python3 src/api.py
   # Server binds to 0.0.0.0:8000 and is immediately reachable by Laptop 2 across the LAN
   ```
4. Terminal logs will confirm startup:
   ```
   =================================================================
    PATHSNAP DISTRIBUTED MAP-MATCHING SYSTEM — SERVER STARTUP
   =================================================================
   Loading PathSnap model artifacts...
    [OK] Loaded tuned model from models/lgbm_tuned.pkl
    [OK] HMM Route Reconstructor initialized.
    [OK] Evaluator test directory: evaluator_test
    [OK] Student test directory:   Student folder/student_test
   =================================================================
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

#### 2. On Laptop 2 (Evaluator Web Interface):
Open any modern web browser (Google Chrome, Firefox, Safari, Edge) and navigate to:
```
http://<LAPTOP_1_IP>:8000
```
*(Example: `http://192.168.1.105:8000`. If testing on a single machine, use `http://localhost:8000`).*

No Python environment, model weights, or command-line tools are required on Laptop 2.

#### 3. Optional Command-Line Client (Alternative to Web UI):
Evaluators can also run the terminal client from Laptop 2 if desired:
```bash
python3 src/ui.py http://<LAPTOP_1_IP>:8000 00000015
```

---

### B. Evaluator Website Features & Capabilities

The evaluator web interface is engineered strictly according to the hackathon specifications: **clean, minimal, academic, and monochromatic** (zero external framework overhead, no AI-looking gradients or neon styling):

1. **Evaluator Test Suite by Default:**
   - Automatically loads the 10 reserved evaluator trips: `00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`.
   - Initial interactive Folium map displays Trip `00000015` immediately on page load.
2. **Partition Selector:**
   - Dropdown toggle allowing evaluators to switch seamlessly between:
     - `Evaluator Test (10 Reserved Trips)` [Default Suite]
     - `Student Test (10 Development Trips)` [Baseline Suite]
   - Dynamically updates the trip selector dropdown and Quick Select chips upon switching.
3. **1-Click Quick Select Chips:**
   - Monochromatic chips allowing instantaneous single-click switching between trips.
   - Clicking any chip updates the active selection and loads its interactive Folium map.
4. **Single-Trip & Batch Evaluation Options:**
   - **Evaluate Trip (Primary Button):** Sends the selected trip ID to Laptop 1, executes full LightGBM inference and HMM Viterbi route reconstruction, evaluates against server-side ground truth, overlays the predicted route (red) on the map, and displays live metrics.
   - **Evaluate All 10 (Batch Suite Button):** Evaluates all 10 trips in the active partition in sequence, computing Mean Pointwise Accuracy, Mean Route Continuity, and rendering an academic summary table.
5. **Real-Time Evaluation Metrics Dashboard:**
   - **Pointwise Accuracy:** Server-side observation-level accuracy computed against reference ground truth.
   - **Route Continuity:** Graph topological connectedness of consecutive predicted road arcs.
   - **GPS Observations:** Total trajectory observations.
   - **Unique Road Arcs:** Distinct road segments in the reconstructed route.
   - **Server Latency:** End-to-end processing time in milliseconds.
6. **Interactive Folium Map View:**
   - Raw GPS Observations (Blue circle markers with timestamps and observation indices).
   - Road Network Graph Geometry (Gray lines).
   - Reference Ground Truth Route (Green dashed line).
   - PathSnap Reconstructed Route (Solid red line).
   - Layer Controls, distance measurement tool, coordinate inspect popups, and MiniMap thumbnail.
   - **Open in New Tab ↗ Button:** Opens full-screen map in a dedicated browser tab.
7. **Technical Inspection & Copy Utilities:**
   - **Reconstructed Arcs Box:** Chronological sequence of traversed road segments with 1-click **Copy Arcs** button.
   - **Server JSON Box:** Formatted server payload with 1-click **Copy JSON** button.
8. **Zero-Dependency & 100% Offline Resilience:**
   - Pure vanilla HTML5, CSS3, and JavaScript embedded directly in `src/ui.html`.
   - Requires zero external CDNs, fonts, or scripts. Runs reliably on isolated local networks and mobile hotspots without an active internet connection.

---

### C. Step-by-Step Evaluator Walkthrough

1. **Launch Server on Laptop 1:** Run `python3 src/api.py`. Leave the terminal visible so the evaluator can observe live inference request logs as required by hackathon guidelines.
2. **Open Web Client on Laptop 2:** Browse to `http://<LAPTOP_1_IP>:8000`. Verify the green status pill indicates `Online: LGBMClassifier`.
3. **Select Any Reserved Trip:** Pick any trip (e.g. `00000015` or click a Quick Select chip).
4. **Execute Evaluation:** Click **Evaluate Trip**.
5. **Inspect Live Results:**
   - Verify Pointwise Accuracy and Route Continuity update immediately.
   - Inspect the interactive Folium map showing the green dashed ground-truth route and red solid reconstructed route.
   - Verify Laptop 1 terminal displays the incoming `POST /api/evaluate` log with execution steps.
6. **Run Batch Suite:** Click **Evaluate All 10** to view the full 10-trip summary table with mean accuracy and continuity metrics.

---

### D. Production REST API Endpoints Reference

| Method | Endpoint | Request Body | Response Payload | Description |
|---|---|---|---|---|
| `GET` | `/` | None | HTML | Serves the clean, minimal evaluator web interface |
| `GET` | `/health` | None | JSON | Health probe confirming model loading and active partitions |
| `GET` | `/api/trips` | Query: `?partition=evaluator_test` | JSON | Lists all available trip IDs in the requested partition |
| `POST` | `/api/evaluate` | `{"trip_id": "00000015"}` | JSON | Runs full inference pipeline and returns accuracy, continuity, and route |
| `POST` | `/api/evaluate_all` | `{"partition": "evaluator_test"}` | JSON | Evaluates all trips in partition and returns summary table & mean metrics |
| `GET` | `/map/{trip_id}` | None | HTML | Serves the interactive Folium HTML map for the evaluated trip |

---

### E. Run the Complete Machine Learning Pipeline End-to-End:
```bash
python3 src/run_pipeline.py
```
This executes all 6 pipeline phases:
1. **EDA & Dataset Summary:** Inspects 80 training and 10 testing trips.
2. **Feature Extraction:** Generates candidates using 2D spatial grid indexing and computes 15 kinematic/topological features.
3. **Model Benchmarking:** Evaluates 6 models using 5-fold `GroupKFold` cross-validation.
4. **Bayesian Hyperparameter Optimisation:** Tunes LightGBM with `skopt.BayesSearchCV`.
5. **HMM Route Reconstruction:** Runs Viterbi decoding across the test partition.
6. **Folium Visualisations:** Generates diagnostic plots and interactive HTML maps in `visualizations/maps/`.

### Run Individual Components:
```bash
# Feature generation
python3 src/data_processing.py

# Model training & benchmarking
python3 src/model_training.py

# Bayesian optimization
python3 src/bayesian_optimization.py

# HMM & Viterbi route reconstruction
python3 src/hmm_viterbi.py

# Generate all visualisations and Folium maps
python3 src/visualizations.py
```

---

## 6. Experimental Results Summary

### 6-Model Benchmark Suite (Test Partition):
| Model | Algorithm Family | 5-Fold CV Macro F1 | Test Macro F1 | Top-1 Accuracy | Route Continuity | Test Log Loss | Inference Latency |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic | 0.7078 | 0.6675 | 0.5947 | 0.9192 | 0.3264 | 0.0038s |
| **KNN (k=7)** | Instance-based | 0.8325 | 0.8064 | 0.5936 | 0.9202 | 0.4720 | 14.03s |
| **Random Forest** | Ensemble | 0.7972 | 0.7856 | 0.6222 | 0.9395 | 0.2070 | 0.1746s |
| **XGBoost** | Boosting | **0.8565** | **0.8379** | **0.6250** | **0.9410** | **0.0925** | 0.0467s |
| **LightGBM** | Boosting | 0.8531 | 0.8376 | 0.6212 | 0.9359 | 0.0953 | 0.1126s |
| **MLP Neural Network** | Neural Networks | 0.8482 | 0.8249 | 0.6100 | 0.9356 | 0.0982 | 0.1135s |

### Bayesian Optimization Impact (LightGBM):
- **Cross-Validation Macro F1:** Uplifted from 0.8544 to **0.8585** (+0.41%)
- **Test Log Loss:** Decreased from 0.0937 to **0.0919** (Better probability calibration)
- **Top-1 Observation Accuracy:** Uplifted to **62.18%**
- **Optimal Hyperparameters:** `n_estimators=231`, `max_depth=9`, `num_leaves=21`, `learning_rate=0.0309`, `subsample=0.965`, `colsample_bytree=0.925`, `reg_alpha=0.356`, `reg_lambda=1.260`, `min_child_samples=42`.

### HMM / Viterbi Sequential Decoding Impact:
- **Observation-Level Accuracy:** Increased from **72.73% to 77.87% (+5.14% absolute gain)**
- **Route Network Continuity:** Increased from **93.91% to 99.12% (+5.21% absolute gain)**
- **Disconnection Elimination:** Resolves jumps across parallel streets and enforces physical traversal along road network edges.

### Evaluator Test Suite Performance (Zero Data Leakage Verification):
The table below reports the verified performance on the 10 reserved evaluator test trips evaluated strictly out-of-sample in pure inference mode (zero retraining, zero hyperparameter adjustment):

| Trip ID | GPS Observations | Reconstructed Arcs | Pointwise Accuracy (%) | Route Continuity (%) | Pipeline Latency |
|---|---|---|---|---|---|
| `00000015` | 6,854 | 1,154 | 72.31% | 99.85% | 14.4s |
| `00000017` | 1,346 | 675 | 89.38% | 93.90% | 2.7s |
| `00000023` | 458 | 73 | 81.22% | 98.25% | 0.8s |
| `00000030` | 654 | 322 | 82.11% | 96.48% | 1.1s |
| `00000049` | 2,642 | 418 | 95.76% | 100.00% | 3.5s |
| `00000059` | 1,031 | 400 | 92.63% | 97.28% | 1.7s |
| `00000067` | 916 | 100 | 77.40% | 100.00% | 1.3s |
| `00000073` | 1,144 | 275 | 95.98% | 99.65% | 1.8s |
| `00000087` | 1,399 | 454 | 95.35% | 99.86% | 2.1s |
| `00000094` | 3,230 | 1,008 | 81.11% | 99.41% | 4.8s |
| **MEAN / AVERAGE** | **1,967** | **488** | **86.33%** | **98.47%** | **3.4s (avg)** |

> [!IMPORTANT]
> **Strict Zero-Leakage Policy:** In strict compliance with hackathon regulations, the `evaluator_test` partition was NEVER used during feature selection, model training, cross-validation, Bayesian optimization, or threshold tuning. The model evaluated above is the exact pre-trained LightGBM artifact (`models/lgbm_tuned.pkl`) trained solely on the 80 training trips.

---

## 7. Interactive Geographical Visualisation
Open the generated HTML maps in any browser:
```bash
# View interactive map
xdg-open visualizations/maps/test_trip_00000000_map.html
```
The map includes:
- Blue circle markers: Raw GPS points with timestamps and observation indices.
- Gray lines: Underlying road network graph geometry.
- Dashed green line: Reference ground truth route.
- Solid red line: PathSnap HMM/Viterbi reconstructed route.
- Interactive Layer Control: Toggle individual layers on and off.
- MiniMap: Overview navigation thumbnail.
