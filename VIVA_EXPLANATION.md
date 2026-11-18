# PathSnap - Master Viva & Evaluation Guide
### Complete Reference for Evaluators & Students (Simple Words, Deep Technical Detail)
**Team Repository:** `PathSnap_693_694_697_698`  

### Team Member Responsibilities & Technical Contributions
| SRN / Roll No | Team Member | GitHub Username | Core Technical & Machine Learning Responsibilities |
|---|---|---|---|
| **693** | **Vineeth P Mural** | `vineethpmural123-rgb` | **Data Preprocessing & Spatial Indexing:** 2D Spatial Hash Grid (~300m cells, 64x candidate search acceleration, $O(1)$ spatial queries), Haversine spherical distance formulas, Candidate Generation (200m radius, Top-15), EDA distribution analysis. |
| **694** | **Sudev Suresh** | `sudevsuresh` | **15-Feature Engineering & 6-Model Benchmark Suite:** Vectorized extraction of 15 kinematic/geometric features, training and evaluation across 6 distinct model families, 5-Fold `GroupKFold` cross-validation grouped by `trip_id` to strictly prevent data leakage. |
| **697** | **Uttam K R** | `Uttam1916` | **Bayesian Optimization & HMM/Viterbi Sequential Decoder:** Gaussian Process Bayesian Optimization (`skopt.BayesSearchCV`) tuning 9 hyperparameters on LightGBM using Expected Improvement, mathematical formulation of HMM emission & transition matrices on road graph, Log-Space Viterbi dynamic programming recurrence algorithm, and sensor dropout gap handling (+5.14% accuracy gain, +5.21% continuity gain). |
| **698** | **Siddesh V S** | `siddeshvs351` | **Two-Laptop Distributed Deployment & FastAPI Engine:** High-performance asynchronous FastAPI REST backend (`src/api.py`), server-side ground truth verification engine, interactive Folium geospatial rendering engine (`src/visualizations.py`), and the zero-dependency evaluator UI client (`src/ui.py`). |

---

## Table of Contents
1. [The 60-Second Evaluator Pitch](#1-the-60-second-evaluator-pitch)
2. [Concept-by-Concept Deep Dive (Explained in Simple Words)](#2-concept-by-concept-deep-dive-explained-in-simple-words)
   - [What is Map-Matching?](#what-is-map-matching)
   - [Road Network Graph Structure](#road-network-graph-structure)
   - [The 2D Spatial Hash Grid & 64x Speedup](#the-2d-spatial-hash-grid--64x-speedup)
   - [Haversine vs Euclidean Distance](#haversine-vs-euclidean-distance)
   - [Candidate Generation & Candidate Miss Rate](#candidate-generation--candidate-miss-rate)
   - [The Severe Class Imbalance Problem](#the-severe-class-imbalance-problem)
   - [Data Leakage Prevention: GroupKFold by Trip ID](#data-leakage-prevention-groupkfold-by-trip-id)
   - [The 15 Engineered Features & Physical Rationale](#the-15-engineered-features--physical-rationale)
3. [The 6 Benchmark Models - Detailed Breakdown & Viva Defense](#3-the-6-benchmark-models---detailed-breakdown--viva-defense)
   - [Model 1: Logistic Regression (Linear Baseline)](#model-1-logistic-regression-linear-baseline)
   - [Model 2: K-Nearest Neighbors (KNN - Instance-Based)](#model-2-k-nearest-neighbors-knn---instance-based)
   - [Model 3: Random Forest (Bagged Ensembles)](#model-3-random-forest-bagged-ensembles)
   - [Model 4: XGBoost (Gradient Boosting)](#model-4-xgboost-gradient-boosting)
   - [Model 5: LightGBM (Selected for Bayesian Optimization)](#model-5-lightgbm-selected-for-bayesian-optimization)
   - [Model 6: Multi-Layer Perceptron (Deep Neural Network)](#model-6-multi-layer-perceptron-deep-neural-network)
4. [Master Model Comparison & Architectural Face-Offs](#4-master-model-comparison--architectural-face-offs)
   - [Consolidated Performance Benchmark Table](#consolidated-performance-benchmark-table)
   - [Linear vs Non-Linear Models](#linear-vs-non-linear-models)
   - [Random Forest vs Gradient Boosting (XGBoost & LightGBM)](#random-forest-vs-gradient-boosting-xgboost--lightgbm)
   - [XGBoost vs LightGBM: Why LightGBM Won the Deployment Choice](#xgboost-vs-lightgbm-why-lightgbm-won-the-deployment-choice)
   - [Neural Networks vs Tree Boosters for Geospatial Data](#neural-networks-vs-tree-boosters-for-geospatial-data)
   - [Standalone Classifier vs Classifier + HMM Sequence Decoder](#standalone-classifier-vs-classifier--hmm-sequence-decoder)
5. [Bayesian Hyperparameter Optimization Explained Simply](#5-bayesian-hyperparameter-optimization-explained-simply)
   - [Why Bayesian Search Beats Grid & Random Search](#why-bayesian-search-beats-grid--random-search)
   - [The 9 Tuned Parameters on LightGBM](#the-9-tuned-parameters-on-lightgbm)
   - [Quantitative Uplift: Untuned vs Tuned LightGBM](#quantitative-uplift-untuned-vs-tuned-lightgbm)
6. [HMM & Viterbi Sequential Route Reconstruction](#6-hmm--viterbi-sequential-route-reconstruction)
   - [Why Pure ML Classifiers Fail: The Teleportation Problem](#why-pure-ml-classifiers-fail-the-teleportation-problem)
   - [The Phone Autocorrect Analogy](#the-phone-autocorrect-analogy)
   - [Hidden States, Observations, Emission & Transition Modeling](#hidden-states-observations-emission--transition-modeling)
   - [The Viterbi Dynamic Programming Algorithm in Log-Space](#the-viterbi-dynamic-programming-algorithm-in-log-space)
   - [Handling Sensor Gaps & Tunnels](#handling-sensor-gaps--tunnels)
   - [Measured Per-Trip Impact Across All 10 Test Trips](#measured-per-trip-impact-across-all-10-test-trips)
7. [Two-Laptop Distributed Deployment Architecture](#7-two-laptop-distributed-deployment-architecture)
   - [Role of Laptop 1 (Model Server & API)](#role-of-laptop-1-model-server--api)
   - [Role of Laptop 2 (Minimal Evaluator UI)](#role-of-laptop-2-minimal-evaluator-ui)
   - [Network Protocol & REST Endpoints](#network-protocol--rest-endpoints)
   - [Evaluator UI Design Compliance](#evaluator-ui-design-compliance)
8. [Our Conclusion Till Now: Why Our System is Better Than Others](#8-our-conclusion-till-now-why-our-system-is-better-than-others)
   - [The Winning Pipeline Formulation](#the-winning-pipeline-formulation)
   - [Detailed Comparison Against Alternative Paradigms](#detailed-comparison-against-alternative-paradigms)
9. [Caveats & System Limitations (Honest Technical Appraisal)](#9-caveats--system-limitations-honest-technical-appraisal)
10. [How We Can Improve (Actionable Future Roadmap)](#10-how-we-can-improve-actionable-future-roadmap)
11. [Cheat Sheet: Top 10 Questions Evaluators Will Ask & Winning Answers](#11-cheat-sheet-top-10-questions-evaluators-will-ask--winning-answers)

---

## 1. The 60-Second Evaluator Pitch
> "Good morning/afternoon, evaluators. Our project is **PathSnap**, an end-to-end distributed machine learning system that solves **Map-Matching** — snapping noisy, erratic GPS coordinates to the true underlying road network.
>
> 1. **Data & Preprocessing:** A single trip contains over 165,000 road segments. Naive distance search took ~10 seconds per trip. We implemented a **2D Spatial Hash Grid** (~300m cells) that reduced search time by **64x** (down to 0.15s per trip) and extracted **15 physical features** capturing distance, vehicle heading, speed, and road geometry.
> 2. **6-Model Benchmark:** We evaluated 6 diverse models across linear, instance-based, ensemble, boosting, and neural families using 5-Fold **`GroupKFold` cross-validation** grouped strictly by `trip_id` to guarantee zero data leakage. Gradient boosting proved clearly superior, achieving **0.838 Macro F1** and **94.1% continuity**.
> 3. **Bayesian Optimization:** We tuned LightGBM across 9 hyperparameters using **Gaussian Process Bayesian Optimization (`skopt`)**, pushing CV Macro F1 to **0.8585** and lowering test log loss to **0.0919**.
> 4. **HMM + Viterbi Sequence Decoder:** Because classifiers treat each GPS dot independently — causing cars to 'teleport' across parallel roads — we built a **Hidden Markov Model with log-space Viterbi decoding**. This combines classifier confidence with road graph topology, increasing pointwise accuracy from **72.73% to 77.87% (+5.14% gain)** and route continuity from **93.91% to 99.12% (+5.21% gain)**.
> 5. **Distributed Deployment:** The system is deployed across two laptops: Laptop 1 runs the FastAPI inference engine and server-side ground truth verification, while Laptop 2 runs a clean, minimal UI allowing you to evaluate any of the 10 evaluation trips in real time."

---

## 2. Concept-by-Concept Deep Dive (Explained in Simple Words)

### What is Map-Matching?
Imagine driving through a city with your smartphone GPS on. The phone's GPS receiver calculates your position by measuring the travel time of radio signals from satellites 20,000 km in space. However, in the real world:
- Tall concrete and glass buildings reflect satellite signals before they reach your phone (called the **multipath effect**).
- Trees, tunnels, and atmospheric weather introduce delays and signal drift.
- As a result, the recorded GPS coordinates are noisy. The GPS might say your car is 15 meters to the left — inside a grocery store, swimming in a canal, or driving on a parallel one-way service road in the wrong direction!

**Map-Matching** is the computational task of taking this noisy sequence of raw GPS coordinates and determining the exact sequence of road segments that the vehicle physically drove on.

### Road Network Graph Structure
The map is mathematically represented as a **directed graph** $G = (V, E)$:
- **Nodes ($V$):** Road intersections or sharp curves, stored as `.nodes` files with `(longitude, latitude)` coordinates.
- **Arcs ($E$):** Road segments connecting two nodes, stored as `.arcs` files with `(from_node_id, to_node_id)`. Each row number is a unique `arc_id`.
- Because arcs are directional, a two-way street is represented as two separate arcs: one from node $A \rightarrow B$, and one from node $B \rightarrow A$.
- **Ground Truth (`.route`):** The exact list of sequential `arc_id` values traversed by the vehicle during that trip.

### The 2D Spatial Hash Grid & 64x Speedup
- **The Problem:** In a typical trip, there are ~2,500 GPS points and up to 237,000 road arcs. If you compute the distance from every GPS point to every road segment, you have to perform:
  $$2,500 \times 237,000 = 592,500,000 \text{ distance calculations per trip!}$$
  On a modern laptop CPU, this took ~9.6 to 15 seconds per trip (over 20 minutes for the entire dataset).
- **The Solution (Spatial Bucketing):** We divide the geographical bounding box of the map into a regular grid of square cells, where each cell is **0.003 degrees in size (approximately 300 meters)**.
  - When loading the road network, each road segment is indexed into the grid cells that its bounding box intersects.
  - When a GPS point arrives, we only inspect roads inside its immediate grid cell and adjacent neighboring cells.
- **The Result:** The search complexity drops from $O(N \times M)$ to amortized $O(1)$ per GPS point. Total lookup time collapsed from **9.62s down to 0.15s per trip — an exact 64x speedup!**

### Haversine vs Euclidean Distance
- **Why Euclidean distance ($x^2 + y^2$) fails:** The Earth is a 3-dimensional sphere (ellipsoid), not a flat sheet of paper. One degree of latitude is roughly constant (~111 km everywhere), but one degree of longitude shrinks as you move away from the equator:
  $$\text{Distance per degree longitude} = 111.32 \times \cos(\text{latitude}) \text{ km}$$
  In Germany (where this dataset was recorded, $\approx 49.5^\circ$ North), 1 degree of latitude is ~111 km, but 1 degree of longitude is only ~72 km! Using standard Euclidean distance distorts distances by more than 35%, completely breaking perpendicular distance calculations.
- **The Haversine Formula:** We calculate exact great-circle surface distance in meters using:
  $$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
  $$d = 2 R \cdot \arctan2\left(\sqrt{a}, \sqrt{1-a}\right)$$
  where $R = 6,371,000\text{ m}$ (mean Earth radius), $\phi$ is latitude, and $\lambda$ is longitude in radians.

### Candidate Generation & Candidate Miss Rate
- For each GPS point, we search within a **200-meter radius** and retrieve the top **15 closest road segments** (candidates).
- **Candidate Miss Rate:** What happens if the true road segment is NOT among these 15 candidates? If the true road is missing from the candidate pool, no downstream machine learning classifier or HMM can ever match it!
  - **Train Partition Miss Rate:** **8.36%** (Candidate Pool Recall = **91.64%**).
  - **Student Test Miss Rate:** **20.64%** (Candidate Pool Recall = **79.36%**).
  - *Why does miss rate happen?* In some test trips (such as Trip `00000056`), severe GPS sensor drift pushed points >200m away from the true road into rural fields, or road graph segments were missing in local areas. Acknowledging this candidate recall ceiling is crucial for defending system boundaries.

### The Severe Class Imbalance Problem
- For every GPS observation, our spatial query retrieves up to 15 road candidates.
- However, the car is physically driving on **only ONE road** at that exact second!
- This creates an extreme **1-to-14 class imbalance** (~6.7% positive class `1`, ~93.3% negative class `0`).
- **Why Raw Accuracy is Deceptive:** If a dumb model simply outputs `0` (negative) for every single candidate, it would achieve **93.3% accuracy**, yet it would predict zero roads and be completely useless!
- **How We Solve It:**
  1. We evaluate models using **Macro F1-Score** (which treats the positive and negative class with equal weight).
  2. We measure **Observation-level Top-1 Accuracy** (for each GPS point, does the highest-ranked candidate match the true ground truth road?).
  3. We measure **Route Continuity** (what percentage of consecutive matched road segments physically connect to each other in the road graph?).
  4. We configure classifiers with balanced class weighting or tuned positive class scale weights (`scale_pos_weight`).

### Data Leakage Prevention: GroupKFold by Trip ID
- **The Threat:** Trajectory data is sequential and spatio-temporally correlated. If you drive down an expressway for 5 minutes, 300 consecutive GPS points are recorded on that same stretch of highway.
- **Standard Random K-Fold Leakage:** If you use standard `KFold(shuffle=True)`, point #100 might end up in the training fold, while point #101 ends up in the validation fold. The model easily 'cheats' because it has already seen the exact same car and same road segment half a second earlier! This produces artificially high validation scores that collapse on unseen trips.
- **The GroupKFold Solution:** We group our cross-validation strictly by **`trip_id`** across the 80 training trips (16 trips per fold). The model is trained on 64 complete trips and validated on 16 entirely unseen trips with zero overlap in geography or time. This guarantees realistic, leak-free generalisation estimates.

### The 15 Engineered Features & Physical Rationale
For every candidate road segment $(i)$ associated with a GPS observation $(t)$, we extract a rich 15-dimensional feature vector:

| # | Feature Name | Formula / Definition | Physical Intuition & Meaning |
|---|---|---|---|
| 1 | `distance` | Haversine perpendicular distance $(m)$ | Closer roads are intuitively much more likely to be the true road. |
| 2 | `bearing_diff` | $|\theta_{\text{car}} - \theta_{\text{road}}| \in [0^\circ, 180^\circ]$ | A car driving North ($0^\circ$) cannot physically be on an East-West road ($90^\circ$). |
| 3 | `arc_length` | Road segment length $(m)$ | Distinguishes long arterial expressways from tiny 5-meter intersection connector links. |
| 4 | `gps_speed` | Distance from prior point / $\Delta t$ $(m/s)$ | A vehicle travelling at $100\text{ km/h}$ belongs on a highway, not a narrow alleyway. |
| 5 | `candidate_rank` | Distance rank $(1, 2, \dots, 15)$ | Provides tree models an explicit discrete ordinal ranking signal. |
| 6 | `norm_distance` | $d_i / \max(d_{\text{candidates}})$ | Normalizes distance variations across dense city centers versus sparse rural roads. |
| 7 | `dist_ratio` | $d_i / d_{\min}$ | If candidate 1 is 4m away and candidate 2 is 80m away, the ratio ($20\times$) strongly suppresses candidate 2. |
| 8 | `n_candidates` | Count of candidates found | Informs the model whether we are at a complex 12-way junction or on an isolated country road. |
| 9 | `cos_bearing_diff` | $\cos(\Delta \theta) \in [-1.0, 1.0]$ | Smooth circular trigonometric representation that avoids boundary discontinuities at $0^\circ / 360^\circ$. |
| 10 | `speed_compatibility` | $\text{speed} / \max(\text{arc\_length}, 1.0)$ | Flags physically impossible maneuvers (e.g. driving $90\text{ km/h}$ through a tight 4m turn node). |
| 11 | `dist_to_from_node` | Haversine distance to start junction $(m)$ | Identifies whether the vehicle is entering or just joined the road segment. |
| 12 | `dist_to_to_node` | Haversine distance to end junction $(m)$ | Identifies whether the vehicle is about to exit onto an intersecting street. |
| 13 | `frac_position` | Projection distance along arc / arc length | Tracks linear progress along the street ($0.0 = \text{entry}, 1.0 = \text{exit}$), enforcing one-way travel. |
| 14 | `log_distance` | $\ln(1 + d)$ | Compresses long-tail distance outliers, preventing extreme errors from skewing linear weights or MLP gradients. |
| 15 | `bear_diff_squared` | $(\Delta \theta)^2$ | Imposes a steep quadratic penalty on vehicles attempting to travel opposite the legal road direction. |

---

## 3. The 6 Benchmark Models - Detailed Breakdown & Viva Defense

To adhere strictly to hackathon guidelines, we implemented, trained, and benchmarked **6 distinct models** spanning 5 algorithm families.

```
                    +------------------------------------------------+
                    |             6 BENCHMARK MODEL SUITE            |
                    +------------------------------------------------+
                                            |
         +------------------+---------------+----------------+------------------+
         |                  |                                |                  |
+-----------------+ +----------------+                      +-----------------+ +-----------------+
|     LINEAR      | | INSTANCE-BASED |                      |    ENSEMBLE     | | DEEP LEARNING   |
| Logistic Reg.   | | KNN (k=7)      |                      | Random Forest   | | MLP (64, 32)    |
| (Fast Baseline) | | (Memory Heavy) |                      | (54MB Bagging)  | | (Neural Net)    |
+-----------------+ +----------------+                      +-----------------+ +-----------------+
                                            |
                                 +--------------------+
                                 |  GRADIENT BOOSTING |
                                 |  - XGBoost (Hist)  |
                                 |  - LightGBM (Leaf) |
                                 +--------------------+
```

### Model 1: Logistic Regression (Linear Baseline)
- **Algorithm Family:** Linear & Probabilistic.
- **Concept:** Models the log-odds of a road candidate being correct as a linear combination of features:
  $$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Loss Function:** Binary Cross-Entropy (Log Loss) with $L_2$ regularization:
  $$\mathcal{L}_{\text{LR}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right] + \frac{1}{2C} \|\mathbf{w}\|^2$$
- **Hyperparameters:** $C = 1.0$, L-BFGS solver, `class_weight='balanced'`, `max_iter=500`.
- **Performance:** 5-Fold CV F1: **0.7078**, Test Macro F1: **0.6675**, Top-1 Accuracy: **59.47%**, Continuity: **91.92%**, Test Log Loss: **0.3264**, Inference Latency: **0.0038s**.
- **Strengths:** Blazing fast inference (under 4 milliseconds for 250,000 candidates), highly interpretable regression weights, zero memory overhead.
- **Limitations:** Completely incapable of learning non-linear feature interactions (such as the joint relationship between candidate rank and bearing difference).
- **Diagnosis (Underfitting vs Overfitting):** **Underfitting**. Train accuracy was 87.3% and test accuracy was 85.3%, but Macro F1 was only 0.6675. A straight hyper-plane cannot carve out curved roads or complex road intersections.

### Model 2: K-Nearest Neighbors (KNN - Instance-Based)
- **Algorithm Family:** Instance-Based / Non-Parametric.
- **Concept:** When a road candidate arrives, KNN searches through the entire historical training dataset to find the $k=7$ most geometrically similar candidate feature vectors, weighting their votes inversely by Minkowski distance:
  $$w_j = \frac{1}{d(\mathbf{x}, \mathbf{x}_j) + \epsilon}$$
- **Hyperparameters:** $k = 7$, distance weighting (`weights='distance'`), Minkowski metric ($p=2$), standardized inputs (`StandardScaler`).
- **Performance:** 5-Fold CV F1: **0.8325**, Test Macro F1: **0.8064**, Top-1 Accuracy: **59.36%**, Continuity: **92.02%**, Test Log Loss: **0.4720**, Inference Latency: **14.03s**.
- **Strengths:** Makes zero assumptions about data distribution; adapts naturally to local spatial pockets.
- **Limitations:** **Fatal inference latency (14.03 seconds per test batch!)** because it performs brute-force distance comparisons against hundreds of thousands of stored training vectors. Also generated the worst Log Loss (**0.4720**) because neighbor distance votes produce poorly calibrated posterior probabilities.
- **Diagnosis:** **Overfitting**. Memorized the training data ($100\%$ train accuracy) but dropped sharply on unseen trips.

### Model 3: Random Forest (Bagged Ensembles)
- **Algorithm Family:** Bagged Decision Tree Ensemble.
- **Concept:** Trains 100 deep, unpruned decision trees in parallel. Each tree is trained on a bootstrap sample of the data (`max_samples=0.25`) with random feature subsampling at each split. The final prediction is the average probability across all 100 trees:
  $$\hat{P}(y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
- **Loss / Split Criterion:** Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=0}^1 p_k^2 = 2 p (1 - p)$$
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `max_samples=0.25`, `class_weight='balanced'`.
- **Performance:** 5-Fold CV F1: **0.7972**, Test Macro F1: **0.7856**, Top-1 Accuracy: **62.22%**, Continuity: **93.95%**, Test Log Loss: **0.2070**, Inference Latency: **0.1746s**, Training Time: **49.61s**.
- **Strengths:** Highly resilient to noisy GPS outliers; bagging effectively dampens individual tree variance. Handles tabular feature thresholds with ease.
- **Limitations:** Extremely bloated model artifact (**54.2 MB** file on disk!) and high training time (almost 50 seconds).
- **Diagnosis:** **Well-Balanced Generalization**. Capping `max_depth=12` prevented individual trees from memorizing specific GPS coordinates.

### Model 4: XGBoost (Gradient Boosting)
- **Algorithm Family:** Sequential Gradient Boosted Decision Trees.
- **Concept:** Builds shallow decision trees sequentially. Each new tree $f_t(\mathbf{x})$ fits the negative gradient (pseudo-residuals) of the loss function from all prior trees:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
  where $g_i = \frac{\partial \mathcal{L}}{\partial \hat{y}_i}$ and $h_i = \frac{\partial^2 \mathcal{L}}{\partial \hat{y}_i^2}$ are exact first- and second-order Taylor expansions.
- **Hyperparameters:** `n_estimators=150`, `max_depth=6`, `learning_rate=0.1`, `tree_method='hist'`.
- **Performance:** 5-Fold CV F1: **0.8565**, Test Macro F1: **0.8379**, Top-1 Accuracy: **62.50%**, Continuity: **94.10%**, Test Log Loss: **0.0925**, Inference Latency: **0.0467s**, Training Time: **0.598s**.
- **Strengths:** Outstanding accuracy and top-1 ranking ability; second-order optimization provides exceptionally well-calibrated probabilities (Log Loss: 0.0925).
- **Limitations:** Level-wise tree depth growth creates symmetric trees that evaluate redundant splits on sparse geospatial features.
- **Diagnosis:** **Minimal Overfitting**. Train accuracy 97.75% vs Test accuracy 96.36% — a tiny 1.39% generalization gap.

### Model 5: LightGBM (Selected for Bayesian Optimization)
- **Algorithm Family:** Fast Leaf-Wise Gradient Boosted Decision Trees.
- **Concept:** Uses **Leaf-Wise (Best-First) Tree Growth** rather than level-wise growth. It splits the leaf node that yields the maximum delta loss reduction across the entire tree, converging much faster to optimal decision boundaries. It also groups continuous features into 256 discrete integer bins (Histogram-based algorithm), slashing memory bandwidth.
- **Loss Function:** Binary Logarithmic Loss:
  $$\mathcal{L} = -\sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$
- **Hyperparameters (Default Benchmark):** `n_estimators=200`, `max_depth=8`, `num_leaves=31`, `learning_rate=0.1`, `subsample=0.8`.
- **Performance:** 5-Fold CV F1: **0.8531**, Test Macro F1: **0.8376**, Top-1 Accuracy: **62.12%**, Continuity: **93.59%**, Test Log Loss: **0.0953**, Inference Latency: **0.1126s**, Model Size: **708 KB**.
- **Strengths:** Unmatched deployment efficiency: 75x smaller model file than Random Forest (708 KB vs 54 MB), ultra-fast training, superb probability calibration, native handling of missing values.
- **Limitations:** Leaf-wise growth can overfit on small sample leaves if `min_child_samples` is not properly regularized (which we addressed via Bayesian optimization).
- **Diagnosis:** **Superior Generalization & Deployment Candidate**.

### Model 6: Multi-Layer Perceptron (Deep Neural Network)
- **Algorithm Family:** Deep Artificial Neural Networks.
- **Concept:** A feedforward deep architecture consisting of an input layer (15 standardized features) $\rightarrow$ Hidden Layer 1 (64 neurons, ReLU) $\rightarrow$ Hidden Layer 2 (32 neurons, ReLU) $\rightarrow$ Output Layer (1 neuron, Sigmoid). Trained via mini-batch stochastic gradient descent using the Adam optimizer with backpropagation:
  $$\mathbf{z}^{(1)} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{z}^{(2)} = \text{ReLU}(\mathbf{W}_2 \mathbf{z}^{(1)} + \mathbf{b}_2), \quad \hat{y} = \sigma(\mathbf{w}_3^T \mathbf{z}^{(2)} + b_3)$$
- **Hyperparameters:** Architecture: `(64, 32)`, Activation: ReLU, Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$), Batch Size: 512, Early Stopping: `patience=10`, Epochs: up to 200.
- **Performance:** 5-Fold CV F1: **0.8482**, Test Macro F1: **0.8249**, Top-1 Accuracy: **61.00%**, Continuity: **93.56%**, Test Log Loss: **0.0982**, Inference Latency: **0.1135s**, Model Size: **84 KB**.
- **Strengths:** Ultra-compact model weights (only 84 KB), smooth non-linear manifolds, outputs well-scaled sigmoid probabilities.
- **Limitations:** Highly sensitive to input scaling; training converged slower than gradient boosting; cannot split discrete ordinal features (like candidate rank $1, 2, 3$) as sharply as tree algorithms.
- **Diagnosis:** **Regularized via Early Stopping**. Training automatically halted at epoch 24 when validation loss ceased improving, preventing overfitting.

---

## 4. Master Model Comparison & Architectural Face-Offs

### Consolidated Performance Benchmark Table
*(All models trained on 80 training trips under 5-Fold GroupKFold CV and evaluated on 10 student-test trips)*

| Model | Algorithm Family | 5-Fold CV Macro F1 | Test Macro F1 | Top-1 Accuracy | Route Continuity | Test Log Loss | Inference Latency | Model Weight Size | Overfitting Status |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic | 0.7078 $\pm$ 0.008 | 0.6675 | 59.47% | 91.92% | 0.3264 | **0.0038s** | **1.2 KB** | Underfitting |
| **KNN (k=7)** | Instance-based | 0.8325 $\pm$ 0.036 | 0.8064 | 59.36% | 92.02% | 0.4720 | 14.0316s | 13.4 MB | Severe Overfitting |
| **Random Forest** | Ensemble (Bagging) | 0.7972 $\pm$ 0.065 | 0.7856 | 62.22% | 93.95% | 0.2070 | 0.1746s | 54.2 MB | Well-Balanced |
| **XGBoost** | Boosting | **0.8565 $\pm$ 0.040** | **0.8379** | **62.50%** | **94.10%** | **0.0925** | 0.0467s | 1.8 MB | Minimal Overfit |
| **LightGBM** | Boosting | 0.8531 $\pm$ 0.046 | 0.8376 | 62.12% | 93.59% | 0.0953 | 0.1126s | **708 KB** | Excellent Fit |
| **MLP Neural Net** | Neural Networks | 0.8482 $\pm$ 0.038 | 0.8249 | 61.00% | 93.56% | 0.0982 | 0.1135s | 84 KB | Well-Balanced |

---

### Linear vs Non-Linear Models
- **Logistic Regression (Linear)** produced a poor Macro F1 of **0.6675** and Log Loss of **0.3264**.
- **The Core Flaw:** A linear decision boundary assumes each feature acts independently:
  $$\text{Score} = w_1 \cdot \text{distance} + w_2 \cdot \text{bearing\_diff} + \dots$$
  In real-world driving, features have strong **non-linear conditional thresholds**: if vehicle speed is $100\text{ km/h}$, a distance of 15m is normal for a multi-lane expressway, but if speed is $5\text{ km/h}$, 15m implies you are parked off-road. Linear models cannot represent these conditional gates without manual polynomial feature combinations.
- **Tree-based models** create orthogonal splits ($x_1 > 12.5 \text{ AND } x_2 < 30^\circ$), easily capturing these geographic realities.

### Random Forest vs Gradient Boosting (XGBoost & LightGBM)
- **Bagging (Random Forest):** Builds deep trees independently in parallel, then averages them. It primarily reduces **variance**.
  - *Drawback:* Trees are large, resulting in a **54.2 MB** binary file and a 50-second training time.
- **Boosting (XGBoost & LightGBM):** Builds shallow trees sequentially, where each tree specifically focuses on the mistakes (residuals) made by previous trees. It reduces both **bias and variance**.
  - *Result:* Macro F1 increased from **0.7856 $\rightarrow$ 0.8379**, Top-1 accuracy improved to **62.5%**, and Log Loss dropped from **0.2070 $\rightarrow$ 0.0925** (a $55\%$ error reduction!).

### XGBoost vs LightGBM: Why LightGBM Won the Deployment Choice
Evaluators will notice that default XGBoost scored **0.8379 Macro F1** while default LightGBM scored **0.8376** (a microscopic difference of 0.0003). Why did we select LightGBM for our production pipeline?
1. **Histogram-Based Binning:** LightGBM bins continuous floating-point features into 256 discrete integer buckets. This reduces memory usage by $80\%$ and avoids continuous sorting of coordinates.
2. **Leaf-Wise Growth:** LightGBM splits leaf nodes with the highest loss reduction, achieving higher accuracy in fewer tree iterations.
3. **Model Footprint:** LightGBM's trained model is only **708 KB**, compared to 1.8 MB for XGBoost and 54.2 MB for Random Forest. It loads in 12 milliseconds in our FastAPI server.
4. **Bayesian Headroom:** Because LightGBM has rich structural hyperparameters (`num_leaves`, `min_child_samples`, `subsample`), Bayesian optimization was able to push its CV Macro F1 to **0.8585** (surpassing baseline XGBoost!).

### Neural Networks vs Tree Boosters for Geospatial Data
- Our 2-layer MLP achieved a strong **0.8249 Macro F1** and **0.0982 Log Loss**, proving deep learning is viable.
- However, Gradient Boosted Decision Trees beat the Neural Network (**0.8379 vs 0.8249**):
  - Tabular spatial data contains discrete ordinal features (like `candidate_rank` $\in \{1, 2, \dots, 15\}$) and extreme long-tail distributions. Decision trees naturally slice discrete thresholds without continuous gradient saturation.
  - Neural nets require careful feature scaling (`StandardScaler`), are prone to getting trapped in saddle points on tabular manifolds, and take $3\times$ longer to train.

### Standalone Classifier vs Classifier + HMM Sequence Decoder
This is the most critical comparison of the entire project:

```
+---------------------------------------------------------------------------------------------------+
| PIPELINE STAGE COMPARISON (Evaluated on all 10 Student-Test Trips)                                |
+-----------------------------+-----------------------+-----------------------+---------------------+
| Metric                      | Classifier Alone      | Classifier + HMM      | Absolute Gain       |
+-----------------------------+-----------------------+-----------------------+---------------------+
| Observation-Level Accuracy  | 72.73%                | 77.87%                | +5.14%              |
| Sequential Route Continuity | 93.91%                | 99.12%                | +5.21%              |
+-----------------------------+-----------------------+-----------------------+---------------------+
```

- **Classifier Alone:** Chooses roads point-by-point. It suffers from the **'Teleportation Problem'**: at highway overpasses or parallel service roads, it jumps back and forth between disconnected roads, causing broken routes.
- **Classifier + HMM:** Uses the road network graph to enforce physical drivability. It rejects physically impossible jumps, boosting continuity to **99.12%** and rescuing complex trips (e.g. Trip `00000077` gained **+11.86% accuracy**!).

---

## 5. Bayesian Hyperparameter Optimization Explained Simply

### Why Bayesian Search Beats Grid & Random Search
When tuning machine learning models with many parameters, traditional methods are deeply flawed:
- **Grid Search:** Tries every single combination like cracking a 9-digit safe combination. With 9 parameters, testing 5 values each requires $5^9 = 1,953,125$ training runs! At 5 seconds per run, it would take **113 days**!
- **Random Search:** Picks random combinations blindly. It runs faster, but it is completely 'memoryless': if a learning rate of $0.001$ failed miserably, it might stupidly try $0.0009$ next because it doesn't learn from prior trials.
- **Bayesian Optimization (`skopt.BayesSearchCV`):** Works like an intelligent scientist using a **Gaussian Process (GP)** surrogate model:
  1. It builds a smooth probabilistic map of how hyperparameters affect the validation Macro F1 score.
  2. For the next trial, it uses an **Acquisition Function (Expected Improvement - EI)** to balance *exploitation* (testing near parameters that scored high) and *exploration* (testing regions with high uncertainty).
  3. It locates the optimal global parameter pocket in just **10 intelligent trials**!

```
Iteration 1-3: Broad Exploration (Builds GP prior across search space)
Iteration 4-7: Exploits promising high-F1 regions (Narrows learning rate & tree depth)
Iteration 8-10: Fine-tunes L1/L2 leaf regularization (Finds global maximum)
```

### The 9 Tuned Parameters on LightGBM
We configured an 80-trip GroupKFold cross-validation search over 9 critical hyperparameters:

| Hyperparameter | Search Space | Prior Distribution | Best Value Found | Why We Tuned It |
|---|---|---|---|---|
| `n_estimators` | $[100, 350]$ | Uniform Integer | **231** | Number of sequential boosting rounds. |
| `max_depth` | $[4, 12]$ | Uniform Integer | **9** | Maximum tree depth to prevent deep tree overfitting. |
| `learning_rate` | $[0.02, 0.25]$ | Log-Uniform Real | **0.0309** | Shrinkage step size (lower rate = smoother learning). |
| `num_leaves` | $[15, 75]$ | Uniform Integer | **21** | Maximum leaves per tree (controls model capacity). |
| `subsample` | $[0.6, 1.0]$ | Uniform Real | **0.9648** | Fraction of rows sampled per tree (bagging fraction). |
| `colsample_bytree` | $[0.6, 1.0]$ | Uniform Real | **0.9250** | Fraction of features sampled per tree (feature bagging). |
| `min_child_samples` | $[10, 50]$ | Uniform Integer | **42** | Minimum samples required in a leaf (prevents singleton memorization). |
| `reg_alpha` | $[10^{-4}, 5.0]$ | Log-Uniform Real | **0.3560** | $L_1$ leaf weight regularization (enforces sparsity). |
| `reg_lambda` | $[10^{-4}, 5.0]$ | Log-Uniform Real | **1.2603** | $L_2$ leaf weight regularization (prevents extreme leaf weights). |

### Quantitative Uplift: Untuned vs Tuned LightGBM
The optimization completed in **7.33 seconds** across 10 iterations. The refit model demonstrated clear quantitative improvements:

| Metric | Default LightGBM | Bayesian-Tuned LightGBM | Measured Improvement |
|---|---|---|---|
| **5-Fold CV Macro F1** | 0.8544 $\pm$ 0.027 | **0.8585 $\pm$ 0.027** | **+0.41% absolute gain** |
| **Test Log Loss** | 0.0937 | **0.0919** | **-0.0018 (Better probability calibration)** |
| **Observation Top-1 Accuracy** | 62.14% | **62.18%** | **+0.04% gain** |
| **Route Continuity** | 93.69% | **93.91%** | **+0.22% gain** |

---

## 6. HMM & Viterbi Sequential Route Reconstruction

### Why Pure ML Classifiers Fail: The Teleportation Problem
A supervised machine learning classifier evaluates each GPS point in complete isolation.
- Suppose point $t=1$ is 3 meters away from Road $A$ (an elevated freeway).
- Point $t=2$ has a small GPS multipath error that places it 2 meters away from Road $B$ (a residential frontage street 15 meters below the freeway).
- The classifier greedy picks Road $A$ at $t=1$, and Road $B$ at $t=2$.
- In the real physical world, **cars cannot jump off elevated bridges or teleport through concrete barriers!** The classifier produces a broken, zig-zagging route with terrible route continuity.

### The Phone Autocorrect Analogy
Think of the HMM like smartphone keyboard autocorrect:
- When typing on a touch screen, your thumb hits messy coordinates on the glass.
- Each messy tap is like a **noisy GPS observation**.
- The letters directly under your thumb are like **candidate road segments**.
- If you tap near `T`, then `H`, then `E`, autocorrect doesn't just evaluate each tap independently — it looks at the sequence of letters to see if they form a real, valid English word in the dictionary!
- **HMM does the exact same thing for cars:** It treats the road network graph like a 'dictionary of drivable routes'.

```
                      +------------------------------------------------+
                      |         HMM VITERBI DECODING PIPELINE          |
                      +------------------------------------------------+

Observation t-1:    GPS Point (t-1)                   GPS Point (t)
                         |                                 |
                         v                                 v
Candidates:       Road A    Road B                  Road A    Road C
                   (p=0.8)   (p=0.2)                 (p=0.4)   (p=0.6)
                     \       /                         \       /
                      \     /                           \     /
                       \   /                             \   /
Graph Transition:       \ /                               \ /
                         X       Does Road A physically    X
                        / \      connect to Road C?       / \
                       /   \     - YES: High Prob        /   \
                      v     v    - NO: Penalty (1e-7)   v     v
                  [Optimal Path Calculation via Dynamic Programming]
```

### Hidden States, Observations, Emission & Transition Modeling
In our mathematical formulation:
1. **Hidden State ($s_t$):** The true road segment ($\text{arc\_id}$) the car was physically driving on at second $t$.
2. **Observation ($o_t$):** The measured GPS coordinate $(\text{lon}_t, \text{lat}_t)$.
3. **Emission Probability ($P(o_t \mid s_t)$):** The likelihood of seeing this GPS coordinate given the vehicle is on road arc $s_t$. We combine the physical distance Gaussian distribution with our Bayesian-tuned LightGBM classifier posterior confidence:
   $$P(o_t \mid s_t) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{d(o_t, s_t)^2}{2\sigma^2}\right) \times \left[ P_{\text{LGBM}}(y=1 \mid \mathbf{x}_{t, s_t}) \right]^{1.2}$$
   where $\sigma = 45.0\text{ meters}$ (measured empirical GPS standard deviation in urban canyons).
4. **Transition Probability ($P(s_t \mid s_{t-1})$):** How physically probable is it to drive from road $s_{t-1}$ to road $s_t$?
   - **Case 1: Same Road ($s_t = s_{t-1}$):** Transition probability = $0.60$ (vehicles usually stay on the same road segment for several seconds).
   - **Case 2: Directly Connected in Road Graph ($s_{t-1} \rightarrow s_t$):** Transition probability = $0.35 \times \exp\left(-\frac{|\Delta d|}{25.0}\right)$, where $\Delta d = |d_{\text{great\_circle}} - d_{\text{projected}}|$.
   - **Case 3: Disconnected in Road Graph:** A severe penalty $\epsilon = 10^{-7}$ is applied.

### The Viterbi Dynamic Programming Algorithm in Log-Space
Computing the probabilities of all possible route combinations across 2,500 observations would take $15^{2,500}$ operations — which exceeds the number of atoms in the known universe!
The **Viterbi Algorithm** solves this in $O(T \times K^2)$ time using dynamic programming:
- Let $V_t(k)$ be the maximum log-probability of any valid route ending at candidate $k$ at time step $t$.
- **The Log-Space Recurrence:** We compute everything in negative log-space to completely prevent floating-point arithmetic underflow:
  $$V_t(k) = \ln P(o_t \mid s_t = k) + \max_{j \in \text{Candidates}_{t-1}} \left[ V_{t-1}(j) + \ln P(s_t = k \mid s_{t-1} = j) \right]$$
- We store a backpointer $\text{ptr}_t(k) = \arg\max_j [\dots]$ at each step.
- At final time step $T$, we pick the candidate with the highest log-probability and backtrack through the pointers to reconstruct the exact globally optimal route.

### Handling Sensor Gaps & Tunnels
In the real world, vehicles drive through tunnels or under dense overpasses where GPS signals vanish for 30 seconds to several minutes.
- When $\Delta t > 30\text{ seconds}$, consecutive GPS observations are spaced far apart.
- Under strict transition penalties, the decoder would get stuck because the car obviously traversed intermediate unobserved road segments.
- **Our Adaptive Gap Strategy:** When $\Delta t > 30\text{s}$, we scale down the disconnection penalty by $100\times$ and allow transition probability to decay smoothly as a function of average vehicle velocity ($v = \Delta d / \Delta t$). This allows the decoder to bridge across gaps smoothly without breaking trajectory continuity.

### Measured Per-Trip Impact Across All 10 Test Trips
Here is the exact trip-by-trip experimental proof that HMM/Viterbi dramatically outperforms standalone classifiers:

| Trip ID | Observations | Classifier Accuracy | HMM Viterbi Accuracy | Accuracy Delta | Classifier Continuity | HMM Continuity | Continuity Delta |
|---|---|---|---|---|---|---|---|
| `00000000` | 2,458 | 71.36% | **72.86%** | +1.51% | 96.58% | **98.98%** | +2.40% |
| `00000007` | 1,354 | 89.59% | **96.45%** | **+6.87%** | 99.41% | **99.78%** | +0.37% |
| `00000009` | 1,049 | 81.70% | **86.08%** | **+4.39%** | 91.22% | **98.19%** | **+6.97%** |
| `00000040` | 2,995 | 81.97% | **85.58%** | **+3.61%** | 97.96% | **99.40%** | +1.44% |
| `00000052` | 262 | 93.13% | **96.56%** | **+3.44%** | 95.02% | **98.85%** | +3.83% |
| `00000056` | 3,958 | 19.76% | **20.54%** | +0.78% | 89.56% | **99.29%** | **+9.73%** |
| `00000065` | 768 | 75.65% | **82.03%** | **+6.38%** | 92.44% | **99.87%** | **+7.43%** |
| `00000076` | 1,486 | 76.51% | **83.85%** | **+7.34%** | 90.37% | **98.72%** | **+8.35%** |
| `00000077` | 3,911 | 54.62% | **66.48%** | **+11.86%!** | 96.39% | **99.69%** | +3.30% |
| `00000091` | 874 | 83.07% | **88.22%** | **+5.15%** | 90.15% | **98.40%** | **+8.25%** |
| **AVERAGE** | **1,911** | **72.73%** | **77.87%** | **+5.14%** | **93.91%** | **99.12%** | **+5.21%** |

*(Notice Trip `00000077`: accuracy increased by **+11.86%**, and on Trip `00000056` continuity jumped by **+9.73%**!)*

---

## 7. Two-Laptop Distributed Deployment Architecture

As required in Section 3 of the hackathon guidelines, PathSnap operates as a distributed client-server architecture communicating over local Wi-Fi / Ethernet:

```
+-------------------------------------------------------+               HTTP POST /api/evaluate               +-------------------------------------------------------+
|                       LAPTOP 1                        | <-------------------------------------------------- |                       LAPTOP 2                        |
|                  (Model & API Host)                   |                                                     |                 (Evaluator Interface)                 |
|                                                       | --------------------------------------------------> |                                                       |
| - Hosts trained Bayesian-Tuned LightGBM model         |          JSON Response Payload:                     | - Clean, minimal Web UI (http://<LAPTOP1_IP>:8000)    |
| - Hosts Road Network Graph & Spatial Grid Index       |          {                                          | - Evaluator selects any of the 10 Reserved Trips     |
| - Runs on-the-fly 15-Feature Engineering Engine       |            "pointwise_accuracy": 0.9645,            | - Displays Pointwise Accuracy & Route Continuity      |
| - Executes Log-Space HMM/Viterbi Route Decoder        |            "route_continuity": 0.9978,             | - Renders Interactive Folium HTML Map                 |
| - Computes Server-Side Ground Truth Verification      |            "latency_ms": 2475.7,                   | - Displays Real-Time Terminal Activity Logs           |
| - Generates Interactive Folium Map HTML               |            "predicted_arcs": [120, 125, ...]       | - Completely Free of Gradients, Emojis, and Clutter   |
+-------------------------------------------------------+          }                                          +-------------------------------------------------------+
```

### Role of Laptop 1 (Model Server & API)
- Runs the FastAPI production server on `http://0.0.0.0:8000`.
- Keeps the trained Bayesian-tuned LightGBM weights and spatial index resident in RAM.
- When an evaluation request arrives, Laptop 1:
  1. Loads the raw `.track`, `.nodes`, and `.arcs` files for the selected trip.
  2. Queries the spatial hash grid to extract candidates.
  3. Computes the 15 feature vectors vectorially with NumPy.
  4. Runs LightGBM inference to get class posterior probabilities.
  5. Decodes the globally optimal sequence using HMM/Viterbi.
  6. Compares predictions against internal server-side ground truth (`.route`) to compute exact accuracy and continuity.
  7. Renders the interactive Folium HTML map.
  8. Returns the verified JSON metrics.

### Role of Laptop 2 (Minimal Evaluator UI)
- The evaluator sits at Laptop 2 and interacts exclusively with the Web UI or CLI client.
- The UI fetches available trips from `GET /api/trips`.
- The evaluator selects any trip (e.g., `00000007`) from a dropdown menu and clicks **'Run Evaluation'**.
- Within 1 to 2 seconds, the dashboard displays:
  - **Pointwise Accuracy** (e.g. `96.45%`)
  - **Route Continuity** (e.g. `99.78%`)
  - **GPS Observations** (e.g. `1,354`)
  - **Server Processing Latency** (e.g. `2475.7 ms`)
  - **Interactive Folium Map** showing the snapped route directly on the road network.

### Network Protocol & REST Endpoints
| Endpoint | Method | Input Parameters | Output Response |
|---|---|---|---|
| `/` | `GET` | None | Serves the clean Evaluator Web Dashboard HTML |
| `/health` | `GET` | None | `{"status": "healthy", "model": "LGBMClassifier", "tuned": true}` |
| `/api/trips` | `GET` | None | `{"trips": ["00000000", "00000007", ...]}` |
| `/api/evaluate` | `POST` | `{"trip_id": "00000007"}` | JSON containing accuracy, continuity, arc list, latency |
| `/map/{trip_id}` | `GET` | URL path `trip_id` | Returns self-contained interactive Folium HTML map |

### Evaluator UI Design Compliance
In accordance with strict professional academic criteria:
- **Zero Gradients:** Clean, solid flat visual design.
- **Zero Emojis:** Completely professional, distraction-free typography.
- **Zero Loud / Saturated Colors:** Neutral background (`#f9fafb`) with crisp white cards (`#ffffff`) and subtle borders (`#e5e7eb`).
- **High Contrast Typography:** Dark neutral text (`#111827`) using system sans-serif fonts.

---

## 8. Our Conclusion Till Now: Why Our System is Better Than Others

### The Winning Pipeline Formulation
After rigorous empirical testing across 6 model families, Bayesian optimization, and probabilistic graphical modeling, our final production architecture is:
$$\mathbf{Raw\ GPS} \xrightarrow{\text{2D Spatial Grid (300m)}} \mathbf{15\ Features} \xrightarrow{\text{Bayesian LightGBM}} \mathbf{Posterior\ Confidences} \xrightarrow{\text{HMM Viterbi Decoder}} \mathbf{Optimal\ Route}$$

### Detailed Comparison Against Alternative Paradigms

#### 1. Why It is Better Than Pure Geometric / Nearest-Road Heuristics:
- Pure geometric heuristics snap each GPS point to the nearest road segment in Euclidean distance.
- *Why they fail:* At highway flyovers, overpasses, and one-way grids, the nearest road is often going the wrong direction or belongs to an unreachable lower road. Pure heuristics ignore vehicle heading, velocity, and connectivity, producing chaotic, illegal paths.
- *Why PathSnap is better:* Our 15 features capture bearing alignment and speed compatibility, while HMM transition penalties make illegal turns mathematically impossible.

#### 2. Why It is Better Than Standalone Machine Learning Classifiers:
- Standalone ML models (even with 97% training accuracy) predict each second independently.
- *Why they fail:* Minor sensor noise causes them to alternate between parallel roads, yielding broken routes with 93.9% continuity and teleportation artifacts.
- *Why PathSnap is better:* The HMM adds road network topology constraints. It boosted continuity to **99.12%** and pointwise accuracy by **+5.14%**.

#### 3. Why It is Better Than Classical HMM Heuristics (e.g. Newson & Krumm 2009):
- Classic map-matching papers use simple hand-tuned Gaussian emissions based only on GPS distance.
- *Why they fail:* In dense urban canyons with high-rise buildings, distance alone is ambiguous because multiple candidate roads are within 10 meters of each other.
- *Why PathSnap is better:* Instead of hand-tuned rules, our emission probabilities are powered by a **Bayesian-tuned LightGBM classifier** that jointly weighs 15 non-linear physical interactions, achieving much sharper discriminative power.

#### 4. Why It is Better Than Heavy Deep Neural Networks:
- Deep neural networks require high GPU resources, are slow to converge on tabular data, and consume tens of megabytes of memory.
- *Why PathSnap is better:* Our LightGBM model weights occupy only **708 KB**, evaluate in **0.11 seconds**, require zero specialized GPU hardware, and outperform our MLP neural network on every single metric (**0.8376 vs 0.8249 F1**).

#### 5. Why It is 100% Leak-Free:
- By enforcing **GroupKFold cross-validation grouped by Trip ID**, we ensure the model generalises across completely unseen geographic regions, guaranteeing honest, reproducible scores.

---

## 9. Caveats & System Limitations (Honest Technical Appraisal)

A hallmark of a winning engineering team is acknowledging real-world limitations. Here are the 5 caveats of our current system:

1. **Candidate Generation Recall Ceiling (Candidate Miss Rate):**
   - If the true road segment is further than 200m away from the GPS observation (due to catastrophic satellite drift or missing roads in the `.arcs` file), it will not be included in our candidate set.
   - On student test trips, the candidate miss rate was **20.64%** (principally concentrated in Trip `00000056`). Because our classifier and HMM only rank retrieved candidates, they cannot predict roads that were never retrieved in the initial query.
2. **Extended GPS Outages / Tunnels:**
   - While our adaptive gap handler relaxes transition penalties for dropouts $>30\text{s}$, when signals disappear for thousands of seconds, the true trajectory between those points is inherently ambiguous without dead-reckoning (IMU/odometry) sensors.
3. **2D Representation of 3D Overpasses (Lack of Z-Axis Elevation):**
   - The road network coordinates are provided in 2D `(longitude, latitude)`. At stacked double-decker highways or underground subway interchanges with identical 2D footprints, the model must rely solely on speed and heading to distinguish upper from lower levels.
4. **Near-Zero Velocity Heading Jitter:**
   - When a car stops at a red light or in traffic, consecutive GPS coordinates jitter around the same spot due to noise. Computing $\theta = \arctan2(\Delta \text{lat}, \Delta \text{lon})$ on near-zero displacements causes the vehicle heading to spin erratically between $0^\circ$ and $360^\circ$, producing temporary noisy bearing features.
5. **Computational Complexity on Massive State Spaces:**
   - The Viterbi algorithm scales with $O(T \times K^2)$. If candidate count $K$ is expanded from 15 to 50 across a 5,000-point trip, the transition matrix expands quadratically ($2,500$ multiplications per step), increasing latency.

---

## 10. How We Can Improve (Actionable Future Roadmap)

If given more time or moving to production, here are the 6 concrete engineering improvements we would implement:

1. **Adaptive Dynamic Candidate Search Radius:**
   - Instead of a fixed 200m radius, dynamically adjust the search radius based on vehicle speed and satellite Dilution of Precision (DOP):
     $$R_{\text{search}} = R_{\text{base}} + v_{\text{gps}} \times \Delta t$$
   - At high highway speeds ($120\text{ km/h}$) or during GPS dropouts, expand to 500m to eliminate candidate misses.
2. **True Network Shortest-Path Distance for Transitions (Dijkstra / A\*):**
   - Currently, our transition probability uses straight-line distance difference between road endpoints.
   - *Upgrade:* Run Dijkstra or A\* shortest path on the road network graph between candidate road $s_{t-1}$ and candidate road $s_t$. Transition probability will then evaluate:
     $$P(s_t \mid s_{t-1}) \propto \exp\left( -\frac{|d_{\text{graph\_shortest\_path}} - d_{\text{great\_circle}}|}{\beta} \right)$$
     This matches the gold-standard Newson & Krumm formulation and completely eliminates impossible shortcuts.
3. **Extended Kalman Filter (EKF) Pre-Smoothing:**
   - Pass the raw GPS coordinates through an Extended Kalman Filter before candidate extraction. The EKF fuses GPS coordinates with a constant-velocity bicycle motion model, filtering out high-frequency multipath jitter and maintaining stable vehicle heading even when stopped at red lights.
4. **Graph Neural Networks (GNN / ST-GNN):**
   - Model the road network directly as a Spatial-Temporal Graph Neural Network. Nodes learn structural embeddings of the road hierarchy, allowing the network to learn global routing patterns end-to-end.
5. **Incorporating Road Metadata & Legal Constraints:**
   - Integrate OpenStreetMap road tags: speed limits, road type (motorway vs residential vs footway), and one-way restrictions. If a candidate road is a pedestrian footpath and vehicle speed is $80\text{ km/h}$, its emission probability is forced to zero.
6. **Online / Streaming Viterbi with Fixed-Lag Decoding:**
   - The standard Viterbi algorithm requires the entire trip to finish before backtracking. For live in-car navigation, implement **Fixed-Lag Viterbi**: backtrack and emit matched roads after a sliding window of $W = 10\text{ seconds}$, enabling real-time low-latency streaming map-matching.

---

## 11. Cheat Sheet: Top 10 Questions Evaluators Will Ask & Winning Answers

### Q1: Why did you use Macro F1-score as your primary selection metric instead of simple Accuracy?
> **Answer:** "Because candidate classification suffers from severe class imbalance: for every GPS point, we generate 15 candidates, but only 1 is correct. 93.3% of the dataset is negative (class 0). A naive classifier that predicts '0' for everything would achieve 93.3% accuracy while matching zero roads! Macro F1 calculates the unweighted mean of F1 for both classes, heavily penalizing models that fail on the minority positive class."

### Q2: Why did tree boosting (XGBoost / LightGBM) outperform the Multi-Layer Perceptron (Neural Network)?
> **Answer:** "Geospatial tabular data contains discrete step-functions — for example, candidate distance rank (rank 1 vs rank 2) and sharp angle differences. Decision trees naturally place orthogonal axis-aligned splits on discrete thresholds. Neural networks, on the other hand, use continuous smooth activation functions that require extensive feature scaling and struggle to model sharp tabular boundaries without millions of parameters."

### Q3: What specific problem does the HMM solve that the machine learning classifier cannot?
> **Answer:** "The classifier evaluates each GPS point in isolation. If a car drives on an elevated expressway and a GPS point jitters 5 meters sideways near a parallel surface road, the classifier jumps to the surface road. Cars cannot teleport! The HMM incorporates road network graph topology. Transitioning between disconnected road segments carries a severe penalty ($\epsilon = 10^{-7}$). The Viterbi algorithm finds the globally optimal continuous path, boosting route continuity to 99.12%."

### Q4: How did you guarantee that there is absolutely zero data leakage in your cross-validation?
> **Answer:** "We used `GroupKFold` cross-validation grouped strictly by `trip_id`. Trajectory points from the same trip are heavily auto-correlated in space and time. If we used standard random K-Fold, consecutive seconds from the same car would appear in both train and validation folds, artificially inflating scores. By grouping by trip ID, validation folds were evaluated exclusively on completely unseen trips in different areas."

### Q5: How did you achieve a 64x speedup in candidate road generation?
> **Answer:** "A naive distance calculation requires checking 2,500 GPS points against up to 237,000 road segments — over 500 million checks per trip, which took ~10 seconds. We implemented a 2D spatial hash grid with 300-meter cells (~0.003 degrees). Road segments are indexed into grid buckets on load, reducing candidate search to an amortized $O(1)$ spatial lookup. Search latency dropped to 0.15s per trip."

### Q6: Why did you select LightGBM over XGBoost for final deployment when XGBoost had a slightly higher default F1?
> **Answer:** "While default XGBoost was marginally higher by 0.0003 F1, LightGBM uses histogram binning and leaf-wise tree growth, reducing memory usage and producing a trained model file of only 708 KB (vs 54 MB for Random Forest). Furthermore, after Bayesian hyperparameter optimization on LightGBM, its CV Macro F1 reached 0.8585 and log loss dropped to 0.0919, surpassing baseline XGBoost while maintaining sub-second inference."

### Q7: How does Bayesian Hyperparameter Optimization work, and why not use Grid Search?
> **Answer:** "Grid search over 9 continuous and integer hyperparameters would require over 1.9 million trials, which is computationally intractable. Bayesian optimization uses a Gaussian Process surrogate model to build a probabilistic curve of our validation F1 score. It uses an Expected Improvement acquisition function to intelligently test hyperparameters where it expects the highest performance uplift, finding the optimal parameters in just 10 trials."

### Q8: What happens when the vehicle enters a tunnel or GPS drops out for several minutes?
> **Answer:** "In our HMM transition matrix, we implemented adaptive gap handling: whenever $\Delta t > 30\text{ seconds}$, the strict graph disconnection penalty ($\epsilon = 10^{-7}$) is relaxed by $100\times$, and transition probability decays smoothly as a function of the vehicle's implied velocity. This allows the Viterbi decoder to bridge across missing GPS observations without breaking route continuity."

### Q9: Why was Euclidean distance inappropriate for distance calculations?
> **Answer:** "Because the Earth is a sphere, degrees of longitude shrink as you move away from the equator according to $\cos(\text{latitude})$. At our dataset's German latitude (~49.5° N), 1 degree of latitude is ~111 km, but 1 degree of longitude is only ~72 km. Standard Euclidean distance ($x^2 + y^2$) distorts distances by over 35%. We used the spherical Haversine formula to calculate true great-circle surface distance in meters."

### Q10: How does your two-laptop deployment ensure evaluation integrity?
> **Answer:** "Laptop 1 runs the FastAPI inference engine and stores the road graph, trained models, and ground truth reference routes. Laptop 2 runs the evaluator UI and connects over HTTP. When the evaluator picks a trip, Laptop 2 sends only the trip ID to Laptop 1. Laptop 1 computes the entire prediction pipeline and evaluates accuracy internally. Ground truth reference labels are never exposed to Laptop 2 or the client UI, preventing any possibility of client-side cheating or hardcoded outputs."
