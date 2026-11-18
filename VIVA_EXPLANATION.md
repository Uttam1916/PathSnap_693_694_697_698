# PathSnap: Complete Student Viva & Evaluation Guide
### Plain-English Explanations, Design Rationales, and Quantitative Evidence
**Repository:** `PathSnap_693_694_697_698`  
**Evaluation Target:** First Phase Evaluation (11:30 AM)  
**Authors:** Team 693_694_697_698 (`vineethpmural123-rgb`, `sudevsuresh`, `siddeshvs351`, `Uttam1916`)

---

## Table of Contents
1. [Executive Summary & 2-Minute Viva Pitch](#1-executive-summary--2-minute-viva-pitch)
2. [Problem Understanding: What is Map-Matching?](#2-problem-understanding-what-is-map-matching)
3. [Dataset Exploration & Preprocessing (Guide Section 2.2)](#3-dataset-exploration--preprocessing-guide-section-22)
   - Missing-Value Analysis
   - Trajectory & Network Characteristics
   - Spatial Grid Indexing Rationale (64x Speedup)
   - Ground Truth Alignment & Candidate Miss Rate
4. [Feature Engineering Rationale (15 Vectorized Features)](#4-feature-engineering-rationale-15-vectorized-features)
5. [Multi-Model Benchmarking Suite (Guide Section 3.1 & 3.2)](#5-multi-model-benchmarking-suite-guide-section-31--32)
   - Consolidated Benchmark Table (All 6 Models)
   - Compulsory Individual Model Breakdown:
     - 5.1 Logistic Regression (Linear & Probabilistic)
     - 5.2 K-Nearest Neighbors (Instance-Based)
     - 5.3 Random Forest (Ensemble)
     - 5.4 XGBoost (Gradient Boosting)
     - 5.5 LightGBM (Gradient Boosting)
     - 5.6 MLP Neural Network (Deep Learning)
6. [Bayesian Hyperparameter Optimisation (Guide Section 4)](#6-bayesian-hyperparameter-optimisation-guide-section-4)
7. [HMM & Viterbi Sequential Route Reconstruction (Guide Section 5)](#7-hmm--viterbi-sequential-route-reconstruction-guide-section-5)
8. [Interactive Geographical Visualisations with Folium (Guide Section 6)](#8-interactive-geographical-visualisations-with-folium-guide-section-6)
9. [Academic Integrity & Data Leakage Prevention (Guide Section 8)](#9-academic-integrity--data-leakage-prevention-guide-section-8)
10. [Viva Quick Cheat Sheet: Anticipated Questions & Answers](#10-viva-quick-cheat-sheet-anticipated-questions--answers)

---

## 1. Executive Summary & 2-Minute Viva Pitch

### The 2-Minute Elevator Pitch:
> "Our project solves the **Map-Matching Problem** — determining the true road path a vehicle traveled given noisy, discrete GPS points.
>
> We built a clean, leak-free, 4-stage technical pipeline:
> 1. **Spatial Candidate Generation & Vectorized Feature Engineering:** Because road networks have over 165,000 arcs per trip, naive search takes 15 minutes. We created a 2D spatial grid index (~300m cells) that reduced search time by **64x** (to 0.15s per trip) while extracting **15 physical, geometric, and kinematic features** per candidate.
> 2. **Rigorous Multi-Model Comparison across 6 Families:** We evaluated 6 distinct models using 5-Fold `GroupKFold` cross-validation grouped strictly by `trip_id` so no trip was ever split across folds. Tree-based boosting models (`XGBoost` and `LightGBM`) dominated with Macro F1 of **0.838**, Top-1 Accuracy of **62.5%**, and Route Continuity of **94.1%**.
> 3. **Bayesian Hyperparameter Optimisation:** Using `skopt.BayesSearchCV` with Gaussian Process surrogates over 9 hyperparameters, we tuned LightGBM, boosting CV Macro F1 from 0.854 to **0.858** and reducing log loss to **0.0919**.
> 4. **HMM & Viterbi Sequential Trajectory Decoding:** Classifiers evaluate points independently, leading to impossible jumps across parallel streets. We implemented a Hidden Markov Model with log-space Viterbi dynamic programming that merges Gaussian GPS spatial emissions with road network topological transition penalties. This boosted observation accuracy from **72.73% to 77.87% (+5.14%)** and route continuity from **93.91% to 99.12% (+5.21%)**."

---

## 2. Problem Understanding: What is Map-Matching?

### In Very Simple Words:
Imagine you are walking down a street with your phone. GPS chips in phones and cars have physical noise (caused by tall buildings, satellite atmospheric delays, and multi-path reflections). The GPS might say you are 20 meters to the left, inside someone's living room or on an adjacent one-way street going the wrong way.

**Map-Matching** is the process of snapping these noisy GPS coordinates onto the real, physically connected road network graph.

### Why is this hard?
1. **Parallel Roads:** A highway might run right next to a frontage road. The GPS points land in between them.
2. **Dense Intersections:** At a 5-way junction, 10 different road segments exist within 20 meters of one GPS point.
3. **Temporal Gaps:** The car might go through a tunnel or lose signal for 60 seconds. When it reappears, we must deduce which path it took.
4. **Scale:** A single trip has up to 16,000 GPS points and 237,000 road segments. Checking every road segment for every point would require $16,000 \times 237,000 \approx 3.8 \times 10^9$ calculations per trip!

---

## 3. Dataset Exploration & Preprocessing (Guide Section 2.2)

### Partitioning Rules (Strictly Enforced):
- **Training Partition (80 trips):** Used exclusively for exploratory analysis, candidate extraction, and 5-fold cross-validation. Total: **208,417 GPS observations**, average **2,605 obs/trip**, average **48,461 arcs/trip**.
- **Student-Testing Partition (10 trips):** Used solely for development-stage model comparison and decoder evaluation. Total: **19,160 GPS observations**.
- **Evaluator-Testing Partition (10 trips):** Untouched and reserved for live evaluator grading.

### Missing-Value Analysis:
We inspected all four file types (`.track`, `.nodes`, `.arcs`, `.route`):
1. **Coordinates & IDs:** Zero missing values (`NaN` or `null`) across all nodes, arcs, and tracks. Every record contains valid float coordinates and integer identifiers.
2. **Temporal Regularity:** The nominal sampling frequency is **1 Hz (1 second intervals)**, covering **97.4%** of consecutive readings.
3. **Sampling Anomalies:** Several trips contain severe temporal dropouts where satellite signal was lost. The maximum recorded time gap was **153,064 seconds** (over 42 hours), with typical dropout clusters between 15s and 120s. Our pipeline handles these gaps by dynamically adapting the transition probability matrix in the HMM.

### Spatial Grid Indexing Rationale (The 64x Speedup):
- **The Problem:** Naive distance computation compares $N$ GPS observations against $M$ road arcs ($O(N \times M)$). On large trips with 237,000 arcs, processing a single trip took ~10 seconds, meaning the full dataset would take 25+ minutes.
- **Our Solution:** We built a 2D spatial hash grid with a bucket size of **`0.003 degrees` (~300 meters)**. Every road segment is hashed into all grid cells its bounding box touches.
- **The Result:** Querying candidates for a GPS point is reduced to $O(1)$ grid lookup plus checking only the ~50 arcs within neighboring cells. Single-trip candidate generation dropped from **9.6 seconds to 0.15 seconds (64x speedup)**.
- **Candidate Pool Constraint:** Radius $R = 200\text{ meters}$, keeping at most $K = 15$ closest candidates per observation epoch.

### Ground Truth Alignment & Candidate Miss Rate:
- The `.route` file provides an ordered sequence of traversed arc IDs, but does NOT provide a direct 1-to-1 timestamp mapping to `.track`.
- Vehicles stop at red lights and travel at variable speeds, meaning a car spends 30 observations on a 100m street and only 2 observations on a fast 200m highway segment.
- **Our Alignment Algorithm:** We implemented an advancing chronological projection window that tracks progress along the route arcs, achieving **98.8%** alignment accuracy with a median spatial projection error of **4.5 meters**.
- **Candidate Miss Rate:**
  - **Training Partition:** **8.36% Miss Rate (91.64% Hit Rate)** across 3,060,582 candidate pairs.
  - **Student-Test Partition:** **20.64% Miss Rate (79.36% Hit Rate)** across 272,486 candidate pairs.

---

## 4. Feature Engineering Rationale (15 Vectorized Features)

For every pair of (GPS observation $t$, Candidate road arc $i$), we compute **15 features**:

| # | Feature Name | Description | Physical / Machine Learning Rationale |
|---|---|---|---|
| 1 | `distance` | Perpendicular distance from GPS point to road segment (meters) | Primary signal. GPS noise follows a normal distribution; closer roads have exponentially higher likelihood. |
| 2 | `bearing_diff` | Angular difference between vehicle heading and road segment orientation | Directional consistency. A car driving North ($0^\circ$) cannot be on an East-West street ($90^\circ$). |
| 3 | `arc_length` | Physical length of the road segment (meters) | Longer segments represent main avenues and highways; shorter segments represent turn lanes and alleyways. |
| 4 | `gps_speed` | Instantaneous vehicle speed calculated between consecutive GPS points (m/s) | Distinguishes urban streets ($20-40\text{ km/h}$) from grade-separated motorways ($80-130\text{ km/h}$). |
| 5 | `candidate_rank` | Rank of this candidate arc sorted by distance ($1 = \text{closest}$) | Invariant to absolute GPS drift; gives trees a discrete priority signal. |
| 6 | `norm_distance` | Distance divided by the maximum distance in the candidate set | Normalizes across wide-open rural areas versus dense urban street canyons. |
| 7 | `dist_ratio` | Distance divided by the minimum candidate distance ($d_i / d_{\min}$) | Directly measures how much worse this road is compared to the #1 closest candidate. |
| 8 | `n_candidates` | Total candidate arcs found within 200m for this observation | Measures local road density. Dense junctions require more conservative decision thresholds. |
| 9 | `cos_bearing_diff` | $\cos(\theta_{\text{diff}})$ | Smooth, continuous periodic representation of angular difference. Avoids boundary discontinuity at $180^\circ$. |
| 10 | `speed_compatibility`| Ratio of GPS speed to segment length ($v / L$) | Flags physically impossible situations (e.g. driving $120\text{ km/h}$ over a 5-meter turning arc). |
| 11 | `dist_to_from_node`| Distance from GPS point to the road's start junction | Identifies whether the vehicle is entering, traversing, or exiting the road segment. |
| 12 | `dist_to_to_node` | Distance from GPS point to the road's end junction | Crucial for predicting upcoming transitions to connected segments. |
| 13 | `frac_position` | Normalized projection along the segment ($0.0 = \text{start}, 1.0 = \text{end}$) | Prevents reverse travel and confirms monotonic forward traversal along one-way arcs. |
| 14 | `log_distance` | $\ln(1.0 + \text{distance})$ | Compresses heavy-tailed spatial outliers, linearizing distance decay for neural networks and logistic regression. |
| 15 | `bear_diff_squared`| $(\theta_{\text{diff}})^2$ | Heavily penalizes candidates with opposite driving directions (e.g. traveling the wrong way on a dual carriageway). |

---

## 5. Multi-Model Benchmarking Suite (Guide Section 3.1 & 3.2)

### Consolidated Model Performance Table:
*Evaluated on 3,060,582 Training Candidate Rows (80 trips) and 272,486 Student-Testing Rows (10 trips).*

| Model | Algorithm Family | Key Configuration | 5-Fold CV Macro F1 | Student-Test Macro F1 | Top-1 Accuracy | Route Continuity | Test Log Loss | Inference Latency |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear & Probabilistic | `C=1.0, solver=lbfgs, class_weight=balanced` | 0.7078 | 0.6675 | 0.5947 | 0.9192 | 0.3264 | **0.0038s** |
| **K-Nearest Neighbors** | Instance-Based | `k=7, weights=distance, metric=minkowski` | 0.8325 | 0.8064 | 0.5936 | 0.9202 | 0.4720 | 14.03s |
| **Random Forest** | Ensemble | `n_estimators=100, max_depth=12, balanced` | 0.7972 | 0.7856 | 0.6222 | 0.9395 | 0.2070 | 0.1746s |
| **XGBoost** | Boosting | `n_estimators=150, max_depth=6, lr=0.1, hist` | **0.8565** | **0.8379** | **0.6250** | **0.9410** | **0.0925** | 0.0467s |
| **LightGBM** | Boosting | `n_estimators=200, max_depth=8, lr=0.1, subsample=0.8` | 0.8531 | 0.8376 | 0.6212 | 0.9359 | 0.0953 | 0.1126s |
| **MLP Neural Network** | Neural Networks | `hidden=(64, 32), adam, early_stopping=True` | 0.8482 | 0.8249 | 0.6100 | 0.9356 | 0.0982 | 0.1135s |

---

### Compulsory Individual Model Breakdown (All 6 Models):

#### 5.1 Logistic Regression (Linear and Probabilistic Family)
- **Suitability:** Serves as the fundamental linear benchmark. It determines how well candidate road segments can be separated based purely on linear combinations of distance, bearing difference, and speed.
- **Strengths:** Ultra-fast training (8.67s) and fastest inference latency (0.0038s). Model coefficients are directly interpretable as odds ratios. Produces smooth, continuous probability estimates.
- **Limitations:** Incapable of capturing nonlinear feature boundaries (e.g. candidate rank interacts strongly with distance in a non-linear step-function). Underperforms when features exhibit complex multicollinearity.
- **Important Hyperparameters:** Inverse regularization strength $C=1.0$, L-BFGS quasi-Newton optimization solver, `class_weight='balanced'` to offset the 95:5 class imbalance.
- **Loss Function:** Binary Cross-Entropy (Negative Log-Likelihood).
- **Student-Test Results:** Macro F1: **0.6675**, Top-1 Accuracy: **0.5947**, Route Continuity: **0.9192**, Test Log Loss: **0.3264**.
- **Overfitting vs Underfitting:** **Severe Underfitting**. Training accuracy was 87.3% and test accuracy was 85.3%, with a low test Macro F1 of 0.6675. The linear decision surface cannot capture the complex spatial geometry of road intersections.

#### 5.2 K-Nearest Neighbors (Instance-Based Family)
- **Suitability:** Map-matching has strong local spatial similarity: if a GPS observation has a similar distance, bearing difference, and speed to known correct matches in training data, it should receive the same positive classification.
- **Strengths:** Truly non-parametric; no explicit functional form assumed. Distance weighting ensures nearest neighbors exert higher influence.
- **Limitations:** Extremely high inference latency (**14.03 seconds** on test set) because it must query high-dimensional KD-trees across millions of instances. Memory-intensive ($13.3\text{ MB}$ serialized).
- **Important Hyperparameters:** $k=7$ neighbors (odd number to avoid ties; $k < 5$ was overly noisy, $k > 10$ over-smoothed small turns), `weights='distance'`, Minkowski metric ($p=2$, Euclidean) on StandardScaler normalized features.
- **Optimization Objective:** Nearest-neighbor metric distance minimization.
- **Student-Test Results:** Macro F1: **0.8064**, Top-1 Accuracy: **0.5936**, Route Continuity: **0.9202**, Test Log Loss: **0.4720**.
- **Overfitting vs Underfitting:** **Overfitting**. The model achieved a trivial 100% training accuracy on stored points, but dropped to 0.8064 Macro F1 and had the worst Log Loss (0.4720) because empirical neighborhood vote fractions produce poorly calibrated probabilities.

#### 5.3 Random Forest (Ensemble Family)
- **Suitability:** Ensembles of decorrelated decision trees reduce variance and naturally handle non-linear relationships, step functions, and mixed scale tabular features without requiring monotonic transformations.
- **Strengths:** Robust against noisy GPS outliers. Low variance across cross-validation folds. Bagging prevents individual trees from memorizing anomalous GPS coordinates.
- **Limitations:** Substantial memory footprint (**54 MB** serialized model file). Slower training (**49.6 seconds**) due to deep orthogonal axis splits. Probability outputs tend to peak around discrete bucket fractions.
- **Important Hyperparameters:** `n_estimators=100`, `max_depth=12` (depth-limited to prevent leaf memorization), `max_samples=0.25` (subsampling for tree diversity), `class_weight='balanced'`.
- **Loss Function:** Gini Impurity for split criteria.
- **Student-Test Results:** Macro F1: **0.7856**, Top-1 Accuracy: **0.6222**, Route Continuity: **0.9395**, Test Log Loss: **0.2070**.
- **Overfitting vs Underfitting:** **Well-balanced with slight generalisation gap**. Training accuracy was 95.3% and test accuracy was 93.3%. Restricting `max_depth=12` successfully prevented the 100% memorization seen in unconstrained decision trees.

#### 5.4 XGBoost (Gradient Boosting Family)
- **Suitability:** Gradient boosted decision trees sequentially correct residual errors. In map-matching, ambiguous parallel roads require sequential, fine-grained additive boundaries that boosting creates with high precision.
- **Strengths:** **Highest overall performance** across all initial models (Macro F1: **0.8379**, Top-1: **0.6250**, Route Continuity: **0.9410**, Log Loss: **0.0925**). Fast training via histogram binning (`tree_method='hist'`) and built-in L1/L2 leaf regularization.
- **Limitations:** Sensitive to learning rate and tree depth; requires careful tuning of leaf shrinkage to avoid over-weighting isolated GPS errors.
- **Important Hyperparameters:** `n_estimators=150`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=5.0`.
- **Loss Function:** Second-order Taylor expanded Binary Cross-Entropy with Hessian regularization.
- **Student-Test Results:** Macro F1: **0.8379**, Top-1 Accuracy: **0.6250**, Route Continuity: **0.9410**, Test Log Loss: **0.0925**, Inference Latency: **0.0467s**.
- **Overfitting vs Underfitting:** **Excellent Generalisation**. Training accuracy was 97.7% and test accuracy was 96.4%. The small gap (1.3%) and very low log loss (0.0925) demonstrate that L1/L2 leaf penalties prevented overfitting to individual training trips.

#### 5.5 LightGBM (Gradient Boosting Family)
- **Suitability:** Selected as our primary candidate for Bayesian optimization. LightGBM uses leaf-wise (best-first) tree growth and Gradient-based One-Side Sampling (GOSS), delivering near-identical accuracy to XGBoost with superior training speed and lower memory usage.
- **Strengths:** Highly compact serialized model ($708\text{ KB}$ untuned, $564\text{ KB}$ tuned). Rapid inference latency (0.112s for 272k rows). Native handling of class imbalance and histogram-based binning.
- **Limitations:** Unconstrained leaf-wise growth can lead to overfitting on smaller partitions if `num_leaves` is not carefully bounded.
- **Important Hyperparameters:** `n_estimators=200`, `max_depth=8`, `num_leaves=31`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Loss Function:** Binary Log Loss (Logarithmic Loss).
- **Student-Test Results:** Macro F1: **0.8376**, Top-1 Accuracy: **0.6212**, Route Continuity: **0.9359**, Test Log Loss: **0.0953**.
- **Overfitting vs Underfitting:** **Well-regularized**. Train accuracy was 97.5% vs Test accuracy 96.3%. The model generalizes stably across all 10 student-test trips.

#### 5.6 Multi-Layer Perceptron (Neural Network Family)
- **Suitability:** Evaluates whether deep nonlinear functional representations and learned hidden layers can capture continuous coordinate geometries better than orthogonal axis-aligned decision trees.
- **Strengths:** Compact model representation ($84\text{ KB}$). Fast inference (0.113s). Smooth, well-calibrated sigmoid probability outputs suitable for downstream Bayesian and HMM integration.
- **Limitations:** Requires strict feature scaling (sensitive to unnormalized coordinates). Lacks the inductive bias of tree ensembles on tabular features.
- **Important Hyperparameters:** Architecture: 2 hidden layers with `(64, 32)` ReLU neurons, `batch_size=512`, `learning_rate_init=0.001`, Adam optimizer, `early_stopping=True` with 10% validation holdout, $L_2$ penalty $\alpha=10^{-4}$.
- **Loss Function:** Binary Cross-Entropy with Backpropagation.
- **Student-Test Results:** Macro F1: **0.8249**, Top-1 Accuracy: **0.6100**, Route Continuity: **0.9356**, Test Log Loss: **0.0982**.
- **Overfitting vs Underfitting:** **Prevented by Early Stopping**. Training stopped at epoch 24 when validation loss flattened, preventing divergence between train (97.0%) and test (96.1%) accuracy.

---

## 6. Bayesian Hyperparameter Optimisation (Guide Section 4)

### Why Bayesian Optimisation Over Grid or Random Search?
- **Grid Search:** Brute-force exhaustive evaluation across a discrete grid. For 9 hyperparameters with 5 values each, this requires $5^9 \approx 1,953,125$ evaluations — computationally impossible within the hackathon timeframe.
- **Random Search:** Samples points independently at random. Faster, but completely blind: it does not use the results of previous evaluations to decide where to look next.
- **Bayesian Optimisation (`skopt.BayesSearchCV`):** Builds a **Gaussian Process (GP) surrogate model** of the unknown objective function. It balances **exploration** (searching areas of high uncertainty) and **exploitation** (searching near known optima) using the **Expected Improvement (EI)** acquisition function:
$$\text{EI}(x) = \mathbb{E}\left[\max(0, f(x) - f(x^+))\right]$$
This identifies the optimal hyperparameter region in dramatically fewer trials.

### Search Space & Distributions Considered:
```python
search_spaces = {
    'n_estimators': Integer(100, 350),                     # Number of boosting stages
    'max_depth': Integer(4, 12),                           # Tree depth ceiling
    'learning_rate': Real(0.02, 0.25, prior='log-uniform'),# Step shrinkage
    'num_leaves': Integer(15, 75),                         # Max leaves per tree
    'subsample': Real(0.6, 1.0),                           # Row bagging ratio
    'colsample_bytree': Real(0.6, 1.0),                    # Feature bagging ratio
    'min_child_samples': Integer(10, 50),                  # Minimum data in one leaf
    'reg_alpha': Real(1e-4, 5.0, prior='log-uniform'),     # L1 regularization
    'reg_lambda': Real(1e-4, 5.0, prior='log-uniform'),    # L2 regularization
}
```

### Optimisation Configuration & Outcome:
- **Optimization Objective:** Maximize Macro F1-score under 3-Fold `GroupKFold` cross-validation on the training partition.
- **Number of Trials:** 10 Bayesian iterations.
- **Best Hyperparameters Found:**
  - `n_estimators`: **231**
  - `max_depth`: **9**
  - `num_leaves`: **21**
  - `learning_rate`: **0.0309** (lower learning rate with more trees improved stability)
  - `subsample`: **0.965**
  - `colsample_bytree`: **0.925**
  - `min_child_samples`: **42** (higher leaf threshold prevented leaf memorization)
  - `reg_alpha` ($L_1$): **0.356**
  - `reg_lambda` ($L_2$): **1.260** (strong $L_2$ penalty smoothed prediction margins)

### Tuned vs Untuned Empirical Comparison:
| Metric | Untuned LightGBM Baseline | Bayesian Tuned LightGBM | Quantitative Uplift / Delta |
|---|---|---|---|
| **5-Fold CV Macro F1** | 0.8544 | **0.8585** | **+0.41%** |
| **Student-Test Top-1 Accuracy** | 0.6214 | **0.6218** | **+0.04%** |
| **Student-Test Log Loss** | 0.0937 | **0.0919** | **-0.0018** (Better calibration) |
| **Student-Test Route Continuity** | 0.9369 | **0.9391** | **+0.22%** |
| **Inference Time** | 0.1685s | 0.2444s | +0.0759s |

---

## 7. HMM & Viterbi Sequential Route Reconstruction (Guide Section 5)

### Why is a Classifier Alone Not Enough?
A supervised classifier scores each GPS point independently. If observation $t$ is slightly closer to Road $A$, and observation $t+1$ is slightly closer to Road $B$ (which does not connect to Road $A$), a greedy argmax classifier will predict $A \rightarrow B$. In reality, a car cannot teleport across city blocks.

An **HMM (Hidden Markov Model)** solves this by enforcing physical topological continuity over the road network graph.

### Mathematical Formulation:
1. **Hidden States ($S_t$):** The set of candidate road segments $a_i \in C(z_t)$ within 200m of GPS observation $z_t$.
2. **Observations ($z_t$):** The sequence of raw GPS measurements $(\text{lon}_t, \text{lat}_t, \text{time}_t)$.
3. **Emission Probability:**
$$P(z_t \mid s_t = a_i) \propto \exp\left(-\frac{1}{2}\left(\frac{d(z_t, a_i)}{\sigma_{\text{gps}}}\right)^2\right) \times \left[P_{\text{classifier}}(a_i \mid z_t)\right]^{1.2}$$
   - $\sigma_{\text{gps}} = 45\text{ meters}$: Empirical Gaussian GPS measurement noise.
   - Combined with the classifier posterior: If the classifier is confident that $a_i$ is correct based on bearing and speed, emission probability increases.
4. **Transition Probability:**
$$P(s_t = a_j \mid s_{t-1} = a_i) = \begin{cases}
0.55 & \text{if } a_i = a_j \text{ (car persists on same arc)} \\
\exp\left(-\frac{|d_{\text{gps}} - d_{\text{road}}|}{\beta}\right) & \text{if } a_i \rightarrow a_j \text{ connected (direct or 1-2 hops)} \\
\epsilon = 10^{-7} & \text{if disconnected (severe penalty)}
\end{cases}$$
   - **Distance Discrepancy:** If the Euclidean GPS distance between consecutive points is 30m, but the road network distance between segments is 800m, the transition probability drops exponentially ($\beta = 25\text{m}$).

### Log-Space Viterbi Recurrence:
To prevent numerical underflow over thousands of observations, computations are performed in log-space:
$$V(t, j) = \max_i \left[ V(t-1, i) + \ln P(a_j \mid a_i) \right] + \ln P(z_t \mid a_j)$$
After reaching epoch $T$, the globally optimal trajectory is recovered by backtracking:
$$s_T^* = \arg\max_j V(T, j), \quad s_{t-1}^* = \text{backpointer}(t, s_t^*)$$

### Handling Edge Cases:
- **Missing Candidates:** If an observation epoch has zero candidate arcs (e.g. car in an unmapped tunnel), the Viterbi state holds the previous valid segment and connects forward to the next observed epoch.
- **Large GPS Dropouts ($\Delta t > 30\text{s}$):** When satellites drop signal for minutes, the vehicle traverses intermediate unobserved links. We relax the disconnect penalty by a factor of 100 ($\epsilon_{\text{gap}} = 10^{-5}$) to allow the path to bridge across the unobserved gap.

### Quantitative Experimental Results (Classifier vs HMM):
*Evaluated across all 10 held-out student-testing trips:*

| Trip ID | Observations | Classifier Accuracy | HMM Accuracy | Accuracy Lift | Classifier Continuity | HMM Continuity | Continuity Lift |
|---|---|---|---|---|---|---|---|
| `00000000` | 2,458 | 71.36% | **72.86%** | **+1.51%** | 96.58% | **98.98%** | **+2.40%** |
| `00000007` | 1,354 | 89.59% | **96.45%** | **+6.87%** | 99.41% | **99.78%** | **+0.37%** |
| `00000009` | 1,049 | 81.70% | **86.08%** | **+4.39%** | 91.22% | **98.19%** | **+6.97%** |
| `00000040` | 2,995 | 81.97% | **85.58%** | **+3.61%** | 97.96% | **99.40%** | **+1.44%** |
| `00000052` | 262 | 93.13% | **96.56%** | **+3.44%** | 95.02% | **98.85%** | **+3.83%** |
| `00000056` | 3,958 | 19.76% | **20.54%** | **+0.78%** | 89.56% | **99.29%** | **+9.73%** |
| `00000065` | 768 | 75.65% | **82.03%** | **+6.38%** | 92.44% | **99.87%** | **+7.43%** |
| `00000076` | 1,486 | 76.51% | **83.85%** | **+7.34%** | 90.37% | **98.72%** | **+8.35%** |
| `00000077` | 3,911 | 54.62% | **66.48%** | **+11.86%** | 96.39% | **99.69%** | **+3.30%** |
| `00000091` | 874 | 83.07% | **88.22%** | **+5.15%** | 90.15% | **98.40%** | **+8.25%** |
| **OVERALL MEAN** | **19,115** | **72.73%** | **77.87%** | **+5.14%** | **93.91%** | **99.12%** | **+5.21%** |

---

## 8. Interactive Geographical Visualisations with Folium (Guide Section 6)

Our visualization engine generates rich, interactive HTML maps using `Folium` (Leaflet.js):
- **Map File:** [`visualizations/maps/test_trip_00000000_map.html`](file:///home/uttam/Documents/mlhack/visualizations/maps/test_trip_00000000_map.html)
- **Map File:** [`visualizations/maps/train_trip_00000001_map.html`](file:///home/uttam/Documents/mlhack/visualizations/maps/train_trip_00000001_map.html)

### Visual Layers & Color Coding:
1. **Raw GPS Observations (Blue Circle Markers):** Plotted with radius 3.5 and popup tooltips showing observation index and timestamp. Visualizes the raw noisy sensor scatter.
2. **Road Network Geometry (Gray Polylines):** Background graph representation showing all drivable road segments in the bounding box.
3. **Ground Truth Route (Dashed Green Polyline):** The true sequence of traversed arcs (width 4.5, opacity 0.85).
4. **PathSnap HMM Reconstructed Route (Solid Red Polyline):** The final predicted trajectory decoded by Viterbi (width 3.0, opacity 0.85).
5. **Start & Stop Markers:** Green `play` icon for trip origin, Red `stop` icon for destination.
6. **Interactive Controls:** Includes a collapsible LayerControl to toggle layers on/off and a dynamic MiniMap thumbnail.

---

## 9. Academic Integrity & Data Leakage Prevention (Guide Section 8)

### Strict Anti-Leakage Measures:
1. **Trip-Level Splitting (`GroupKFold`):** Standard random K-Fold randomly splits rows, putting observations from the same trip into both train and test folds. Because consecutive GPS points are temporally correlated, this causes severe data leakage. We grouped splits strictly by `trip_id`, ensuring all folds evaluated unseen trips.
2. **Untouched Evaluator Partition:** The 10 evaluator trips in `Student folder/evaluator_test` were never opened, inspected, or processed during training or tuning.
3. **Scaler Preprocessing Isolation:** The `StandardScaler` was fitted strictly on the training partition and serialized (`models/scaler.pkl`). Test candidate features were only transformed.
4. **Zero Hardcoded Outputs:** All predictions are generated on-the-fly through feature extraction $\rightarrow$ classifier $\rightarrow$ HMM decoding.
5. **Authentic Experimental Execution:** All metrics, confusion matrices, and figures were computed directly from our codebase execution and pushed to the public team repository.

---

## 10. Viva Quick Cheat Sheet: Anticipated Questions & Answers

### Q1: Why did you choose Macro F1 instead of Accuracy as your primary evaluation metric?
> **Answer:** "Because the candidate classification problem is extremely imbalanced (~95% negative, ~5% positive). If a model simply predicts 0 for every candidate, it would get 95% accuracy while being completely useless. Macro F1 calculates the unweighted mean of F1 across both positive and negative classes, penalizing models that fail to identify the true road."

### Q2: What is the purpose of Top-1 Accuracy and Route Continuity?
> **Answer:** "Macro F1 only evaluates row-by-row classification. **Top-1 Accuracy** evaluates the map-matching objective: for each GPS observation epoch, does the highest-scoring candidate match the true road? **Route Continuity** measures physical feasibility: what fraction of consecutive predictions are connected by an edge in the road graph?"

### Q3: Why is Euclidean distance incorrect for GPS coordinates?
> **Answer:** "The Earth is an oblate spheroid, not a flat 2D plane. At Germany's latitude (~49.5°N), lines of longitude converge: 1 degree of latitude is ~111 km, but 1 degree of longitude is only ~72 km. Euclidean distance treats degrees equally, distorting East-West distances by over 35%. We used the **Haversine formula** to compute true orthodromic surface distances."

### Q4: Why did LightGBM and XGBoost outperform Neural Networks and Random Forest?
> **Answer:** "Gradient boosted decision trees naturally excel at tabular data with heterogeneous feature scales, step functions (like candidate rank), and non-linear interactions. Neural networks required extensive feature normalization and struggled with discrete ranking thresholds, while Random Forest had higher variance and larger memory consumption."

### Q5: How does HMM fix errors made by the classifier?
> **Answer:** "At intersections or parallel highways, a classifier might oscillate between two adjacent roads because of GPS jitter. The HMM adds a graph transition penalty: transitioning to an adjacent road requires traversing a huge graph detour, which carries a heavy transition penalty $\exp(-|\Delta d|/\beta)$. The Viterbi algorithm finds the path that maximizes the global product of emission and transition probabilities, filtering out zig-zags and enforcing a single continuous route."

### Q6: How did you handle large GPS dropouts?
> **Answer:** "In our dataset, sampling is normally 1 Hz, but some trips have signal loss exceeding 60 seconds. In standard HMMs, a large gap would trigger disconnect penalties and break the path. We detect gaps where $\Delta t > 30\text{s}$ and relax the disconnect penalty by a factor of 100, allowing the Viterbi decoder to bridge across unobserved road segments without penalizing the valid trajectory."
