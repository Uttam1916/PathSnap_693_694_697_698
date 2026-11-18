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
| **693** | `vineethpmural123-rgb` | Data Preprocessing, Spatial Grid Indexing, Vectorized Feature Engineering, EDA Distributions |
| **694** | `sudevsuresh` | 6-Model Benchmark Suite, 5-Fold GroupKFold CV, Model Evaluation Metrics, Model Weights |
| **697** | `siddeshvs351` | Bayesian Hyperparameter Optimization (`BayesSearchCV`), HMM & Log-Space Viterbi Sequential Route Decoder |
| **698** | `Uttam1916` | Master Pipeline Orchestration, Interactive Folium Visualizations, Documentation & Viva Preparation |

---

## 3. Repository Architecture
```
PathSnap_693_694_697_698/
├── src/
│   ├── data_processing.py        # Spatial indexing, candidate generation, 15 geometric features
│   ├── model_training.py         # 6-family benchmark suite with GroupKFold cross-validation
│   ├── bayesian_optimization.py  # Gaussian process Bayesian optimization with skopt
│   ├── hmm_viterbi.py            # Hidden Markov Model & Viterbi sequential route decoder
│   ├── visualizations.py         # Folium interactive HTML map engine & diagnostic charts
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
├── visualizations/               # Diagnostic plots and interactive maps
│   ├── maps/
│   │   ├── test_trip_00000000_map.html   # Interactive Folium map (student-test trip 0)
│   │   └── train_trip_00000001_map.html  # Interactive Folium map (training trip 1)
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

## 5. How to Run the System

### A. Two-Laptop Deployment Architecture (Evaluation Mode)
According to the hackathon guidelines, the system runs across two machines on the same local network:

#### On Laptop 1 (Model & FastAPI Backend):
Hosts the trained LightGBM model, HMM/Viterbi engine, and server-side evaluation.
```bash
source venv/bin/activate
python3 src/api.py
# Server starts listening on http://0.0.0.0:8000
```

#### On Laptop 2 (Evaluator User Interface):
1. **Web Interface (Recommended):** Open any browser and navigate to:
   ```
   http://<LAPTOP_1_IP>:8000
   ```
   (If running locally on a single machine, open `http://localhost:8000`).
   - Select any of the 10 evaluation trips from the dropdown.
   - Click **Evaluate Trip**.
   - Displays real-time pointwise accuracy, route continuity, latency, and the embedded interactive Folium map.

2. **Command-Line Interface:**
   ```bash
   python3 src/ui.py http://<LAPTOP_1_IP>:8000 [OPTIONAL_TRIP_ID]
   ```

---

### B. Run the Complete Pipeline End-to-End:
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
