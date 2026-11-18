# PathSnap - Student Viva & Evaluation Guide
### Everything Explained in Simple Words for Your Evaluation (11:30 AM)
**Team Repository:** `PathSnap_693_694_697_698`  
**Teammates:** Vineeth (`vineethpmural123-rgb`), Sudev (`sudevsuresh`), Siddesh (`siddeshvs351`), Uttam (`Uttam1916`)

---

## Quick Reference: 30-Second Summary to Say to Evaluators
> "Our project solves **Map-Matching** — taking noisy GPS dots from a car and snapping them to the real road network.
>
> 1. **Data & Preprocessing:** We indexed over 165,000 roads per trip using a **2D spatial grid** (~300m cells). This sped up candidate road finding by **64x** (from 10 seconds down to 0.15s per trip) and extracted **15 physical features** like distance, car heading vs road direction, and speed.
> 2. **6-Model Benchmark:** We trained and compared **6 different models** across multiple algorithm families using 5-Fold **`GroupKFold` cross-validation** (grouped by trip ID to strictly prevent data leakage). Tree boosting models (**XGBoost** and **LightGBM**) performed best, achieving **0.838 Macro F1**, **62.5% Top-1 accuracy**, and **94.1% continuity**.
> 3. **Bayesian Optimization:** We tuned LightGBM over 9 hyperparameters using **Gaussian Process Bayesian Optimization (`skopt`)**, improving CV Macro F1 to **0.8585** and lowering log loss to **0.0919**.
> 4. **HMM + Viterbi Decoding:** Classifiers alone pick roads independently, causing cars to 'teleport' across parallel roads. Our HMM dynamic programming combines classifier probabilities with road network connections, boosting accuracy from **72.73% to 77.87% (+5.14%)** and route continuity from **93.91% to 99.12% (+5.21%)**."

---

## 1. What is the Problem? (In Super Simple Words)

When you drive a car with a phone or GPS receiver, the GPS signal is not 100% accurate. 
- Tall buildings reflect the satellite signal (called "multipath effect").
- Trees, tunnels, and bad weather cause drift.
- The GPS might say your car is 15 meters to the right — inside someone's house, or on a parallel one-way street going the wrong way!

**Map-Matching** means: Take this messy stream of GPS points and figure out the exact sequence of road segments the car actually drove on.

---

## 2. The Dataset (What We Were Given)

The dataset contains three separate folders:
1. **`train/` (80 trips):** Used to train our models and run cross-validation.
2. **`student_test/` (10 trips):** Held-out trips used to evaluate and compare our models during development.
3. **`evaluator_test/` (10 trips):** **Untouched!** Reserved solely for final evaluator grading.

### What are the 4 raw file types inside each trip?
- **`.track` (GPS observations):** Contains 3 columns: `longitude`, `latitude`, `timestamp`. Usually records 1 point per second (1 Hz).
- **`.nodes` (Road intersections):** Contains road junction coordinates: `longitude`, `latitude`.
- **`.arcs` (Road segments):** Contains connections: `from_node_id` and `to_node_id`. Each row index is an `arc_id` (a road).
- **`.route` (Ground truth route):** The exact list of `arc_id`s the car actually took in order (provided for training trips).

### What did our Exploratory Data Analysis (EDA) find?
- **Average Trip:** ~2,605 GPS points, ~24,169 road nodes, and ~48,461 road arcs. Trajectories average ~37 km in length.
- **Missing Values:** Zero missing coordinates (`NaN` or empty fields). All points have valid numbers.
- **Sampling Gaps:** 97.4% of readings are exactly 1 second apart. But some trips have satellite dropouts where the signal vanished for minutes (up to 153,064 seconds in one extreme case!). Our HMM is designed specifically to handle these gaps.
- **Candidate Miss Rate:** When looking within 200m of each GPS point, the true road was present in our candidate pool **91.64% of the time on train** and **79.36% on test**.

---

## 3. How We Preprocessed the Data & The 64x Speedup Trick

### Why naive search was too slow:
If a trip has 2,500 GPS points and 165,000 road segments, checking every road for every point takes $2,500 \times 165,000 \approx 412,000,000$ distance calculations. That took ~10 seconds per trip (over 20 minutes for all trips!).

### The 2D Spatial Grid Solution:
We divided the map into a grid of small buckets (each cell is **0.003 degrees $\approx$ 300 meters**). 
- When loading roads, we put each road segment into the grid buckets it touches.
- When finding candidates for a GPS point, we only check roads inside the nearby grid buckets.
- **Result:** Search time dropped from **9.6 seconds down to 0.15 seconds per trip (64x faster!)**.
- For each GPS point, we keep the top **15 closest road segments** within a **200m radius**.

### Why Haversine Distance (Not Euclidean):
You cannot use flat Euclidean geometry ($x^2 + y^2$) for latitude and longitude because the Earth is round. In Germany (~49.5° North), 1 degree of latitude is ~111 km, but 1 degree of longitude is only ~72 km! Euclidean distance distorts distances by over 35%. We used the **Haversine formula** to get true ground distance in meters.

---

## 4. The 15 Features We Created (And Why)

For every candidate road near a GPS point, we calculate 15 features:

1. **`distance`:** Perpendicular distance from GPS to road (meters). *Why:* Closer roads are obviously more likely.
2. **`bearing_diff`:** Angle difference between vehicle heading and road direction. *Why:* If you drive North, you can't be on an East-West road.
3. **`arc_length`:** Length of the road segment (meters). *Why:* Distinguishes long highways from tiny turn-lanes.
4. **`gps_speed`:** Speed of the car (m/s). *Why:* Driving 100 km/h means you are on a highway, not a parking lot alley.
5. **`candidate_rank`:** Is this the #1 closest road, #2, or #10? *Why:* Gives tree models an easy ranking signal.
6. **`norm_distance`:** Distance divided by the furthest candidate's distance. *Why:* Normalizes wide rural areas vs tight cities.
7. **`dist_ratio`:** Ratio of distance compared to the closest candidate ($d / d_{\min}$). *Why:* If one road is 5m away and another is 80m, this ratio (16x) clearly flags the second road as unlikely.
8. **`n_candidates`:** Number of roads nearby. *Why:* Tells the model if we are in a dense intersection or an empty highway.
9. **`cos_bearing_diff`:** $\cos(\text{angle difference})$. *Why:* Smooth mathematical representation that avoids angle wrap-around issues.
10. **`speed_compatibility`:** Speed divided by segment length. *Why:* Flags impossible maneuvers (e.g. driving 80 km/h across a 5m junction arc).
11. **`dist_to_from_node`:** Distance to the start of the road segment. *Why:* Knows if the car is just entering the road.
12. **`dist_to_to_node`:** Distance to the end of the road segment. *Why:* Knows if the car is about to transition to the next road.
13. **`frac_position`:** How far along the road the projection lands ($0.0 = \text{start}, 1.0 = \text{end}$). *Why:* Confirms the car is moving forward along one-way streets.
14. **`log_distance`:** $\ln(1 + \text{distance})$. *Why:* Compresses big distance outliers so they don't break linear models or neural nets.
15. **`bear_diff_squared`:** Squared angle difference. *Why:* Strongly penalizes driving in the exact opposite direction.

---

## 5. Model Benchmarking: 6 Models Compared

We evaluated 6 distinct models across different algorithm families.

### The Official Comparison Table:
*(Evaluated using 5-Fold GroupKFold CV on 80 training trips, tested on 10 student-test trips)*

| Model | Family | 5-Fold CV Macro F1 | Test Macro F1 | Top-1 Accuracy | Route Continuity | Test Log Loss | Inference Latency |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Probabilistic | 0.7078 | 0.6675 | 59.47% | 91.92% | 0.3264 | **0.0038s** |
| **KNN (k=7)** | Instance-based | 0.8325 | 0.8064 | 59.36% | 92.02% | 0.4720 | 14.03s |
| **Random Forest** | Ensemble | 0.7972 | 0.7856 | 62.22% | 93.95% | 0.2070 | 0.1746s |
| **XGBoost** | Boosting | **0.8565** | **0.8379** | **62.50%** | **0.9410** | **0.0925** | 0.0467s |
| **LightGBM** | Boosting | 0.8531 | 0.8376 | 62.12% | 93.59% | 0.0953 | 0.1126s |
| **MLP Neural Network** | Neural Networks | 0.8482 | 0.8249 | 61.00% | 93.56% | 0.0982 | 0.1135s |

---

### Compulsory Viva Questions for EVERY Single Model:

#### 1. Logistic Regression
- **Why suitable?** Simplest baseline to check if the road candidates can be separated with simple linear weighting.
- **Strengths:** Super fast (3 milliseconds inference!), highly interpretable weights, outputs smooth probabilities.
- **Limitations:** Cannot learn nonlinear interactions (e.g., candidate rank and bearing difference interact in a non-linear step-function).
- **Hyperparameters:** $C=1.0$, L-BFGS solver, `class_weight='balanced'`.
- **Loss:** Binary cross-entropy (Log Loss).
- **Test Performance:** Macro F1 `0.6675`, Top-1 `59.47%`, Log Loss `0.3264`.
- **Overfitting/Underfitting:** **Underfitting**. Train accuracy 87.3% and test accuracy 85.3% with low F1 (0.6675). Straight lines simply cannot model curved street intersections.

#### 2. K-Nearest Neighbors (KNN)
- **Why suitable?** Based on spatial similarity: if a GPS point has similar distance and direction as known good matches, it should belong to the same road.
- **Strengths:** No assumptions about data shape; distance weighting gives nearby points more vote.
- **Limitations:** **Extremely slow inference (14 seconds!)** because it must search millions of stored rows. Memory heavy (13 MB).
- **Hyperparameters:** $k=7$ neighbors, distance-weighted, Minkowski metric on standardized features.
- **Test Performance:** Macro F1 `0.8064`, Top-1 `59.36%`, Log Loss `0.4720`.
- **Overfitting/Underfitting:** **Overfitting**. Got 100% training accuracy by memorizing points, but dropped to 80.6% on test data and had the worst log loss (0.4720) because neighborhood voting gives poor probability calibration.

#### 3. Random Forest
- **Why suitable?** Ensembles of decision trees handle non-linear tabular features without needing feature normalization.
- **Strengths:** Very resistant to GPS noise outliers; bagging reduces variance.
- **Limitations:** Large file size (**54 MB** model weight file!) and slower training (50 seconds).
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `class_weight='balanced'`.
- **Loss:** Gini impurity.
- **Test Performance:** Macro F1 `0.7856`, Top-1 `62.22%`, Continuity `93.95%`, Log Loss `0.2070`.
- **Overfitting/Underfitting:** **Well-balanced**. Limiting depth to 12 prevented trees from memorizing individual trips.

#### 4. XGBoost
- **Why suitable?** Gradient boosting sequentially corrects mistakes of prior trees, creating fine-grained decision boundaries.
- **Strengths:** **Highest overall performance** (Macro F1: `0.8379`, Top-1: `62.50%`, Log Loss: `0.0925`). Very fast inference (0.046s).
- **Limitations:** Needs careful regularization tuning so it doesn't over-weight noisy GPS glitches.
- **Hyperparameters:** `n_estimators=150`, `max_depth=6`, `learning_rate=0.1`, `tree_method='hist'`.
- **Loss:** Binary log loss with second-order Taylor expansion and L1/L2 leaf regularization.
- **Test Performance:** Macro F1 `0.8379`, Top-1 `62.50%`, Continuity `94.10%`, Log Loss `0.0925`.
- **Overfitting/Underfitting:** **Minimal overfitting**. Train accuracy was 97.7% vs Test 96.4% — a tiny 1.3% gap.

#### 5. LightGBM (Selected for Bayesian Tuning)
- **Why suitable?** Delivers gradient boosting accuracy like XGBoost but uses histogram binning and leaf-wise growth, making it much more lightweight ($708\text{ KB}$ vs $54\text{ MB}$ for Random Forest).
- **Strengths:** Compact model size, fast training and inference (0.112s), excellent probability calibration.
- **Limitations:** Leaf-wise growth can overfit if tree depth and leaf counts are not bounded.
- **Hyperparameters:** `n_estimators=200`, `max_depth=8`, `num_leaves=31`, `learning_rate=0.1`.
- **Loss:** Binary log loss.
- **Test Performance:** Macro F1 `0.8376`, Top-1 `62.12%`, Continuity `93.59%`, Log Loss `0.0953`.
- **Overfitting/Underfitting:** **Healthy generalisation**. Train accuracy 97.5% vs Test 96.3%.

#### 6. Multi-Layer Perceptron (Neural Network)
- **Why suitable?** Tests whether deep neural layers with ReLU activations can learn continuous road coordinate manifolds better than decision trees.
- **Strengths:** Very compact ($84\text{ KB}$), smooth sigmoid probabilities.
- **Limitations:** Sensitive to feature scales and requires longer convergence time.
- **Hyperparameters:** 2 hidden layers `(64, 32)` ReLU neurons, Adam optimizer, `batch_size=512`, `early_stopping=True`.
- **Loss:** Binary cross-entropy with backpropagation.
- **Test Performance:** Macro F1 `0.8249`, Top-1 `61.00%`, Continuity `93.56%`, Log Loss `0.0982`.
- **Overfitting/Underfitting:** **Controlled by Early Stopping**. Training automatically halted at epoch 24 when validation loss flattened, preventing overfitting.

---

## 6. Bayesian Hyperparameter Optimisation Explained Simply

### Why Bayesian instead of Grid or Random Search?
- **Grid Search:** Tries every combination like a combination lock. With 9 parameters, that would take over 1.9 million trials — impossible in a hackathon!
- **Random Search:** Guesses parameters randomly. Faster, but completely blind: it doesn't learn from previous attempts.
- **Bayesian Optimisation (`skopt.BayesSearchCV`):** Uses a **Gaussian Process surrogate model**. It looks at which parameter settings worked well previously, builds a probabilistic curve, and picks the next parameters where it expects the biggest improvement (Expected Improvement acquisition function). It finds the best settings in just 10 trials!

### What We Tuned on LightGBM:
- Search space: 9 parameters (`n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha` L1, `reg_lambda` L2).
- Objective: Maximize Macro F1 under GroupKFold CV.

### Optimal Hyperparameters Found:
- `n_estimators`: **231** | `max_depth`: **9** | `num_leaves`: **21**
- `learning_rate`: **0.0309** (lower rate + more trees = much smoother predictions)
- `subsample`: **0.965** | `colsample_bytree`: **0.925** | `min_child_samples`: **42**
- `reg_alpha` (L1): **0.356** | `reg_lambda` (L2): **1.260** (strong L2 penalty stopped tree overconfidence)

### Quantitative Uplift (Untuned vs Tuned):
- **CV Macro F1:** Increased from 0.8544 $\rightarrow$ **0.8585 (+0.41%)**
- **Test Log Loss:** Dropped from 0.0937 $\rightarrow$ **0.0919 (Better probability calibration)**
- **Top-1 Accuracy:** Increased to **62.18%**

---

## 7. HMM & Viterbi Explained Like Phone Autocorrect

### Why the ML Classifier Alone is Not Enough:
The classifier scores each GPS point independently. If GPS point 1 is slightly closer to Road A, and GPS point 2 is slightly closer to Road B (even if Road B is an elevated highway with no ramp from Road A!), a greedy classifier will jump $A \rightarrow B$. In real life, cars cannot teleport across fences.

### The Autocorrect Analogy:
Think of the HMM like your phone's autocorrect:
- Each messy keyboard tap = a noisy GPS point.
- The letters nearby = candidate road segments.
- Autocorrect doesn't just pick the closest letter for each tap — it picks words that make sense as a sequence!

Similarly:
1. **Hidden States:** The actual road segments the car was on.
2. **Observations:** The noisy GPS points.
3. **Emission Probability:** How close is the GPS point to this road, multiplied by the classifier's confidence:
   $$P(\text{GPS} \mid \text{Road}) = \text{Gaussian}(d, \sigma=45\text{m}) \times [P_{\text{classifier}}]^{1.2}$$
4. **Transition Probability:** Does Road A physically connect to Road B in the road network graph?
   - If connected: High probability, decayed by distance difference $\exp(-|\Delta d| / 25\text{m})$.
   - If disconnected: Heavy penalty ($\epsilon = 10^{-7}$).
5. **Viterbi Algorithm:** Uses dynamic programming to find the single best continuous path from start to end without checking billions of combinations.

### Handling Gaps:
If GPS drops for $>30\text{ seconds}$ (e.g. car in a tunnel), we relax the disconnect penalty by $100\times$, allowing the decoder to bridge across unobserved road segments without breaking the route.

### Measured Impact on the 10 Student-Test Trips:
- **Mean Observation Accuracy:** Jumped from **72.73% $\rightarrow$ 77.87% (+5.14% absolute gain!)**
- **Mean Route Continuity:** Jumped from **93.91% $\rightarrow$ 99.12% (+5.21% absolute gain!)**
- On Trip `00000077`, accuracy improved by **+11.86%** and continuity rose to **99.69%**!

---

## 8. Interactive Folium Maps (How to Show Evaluators)

We built interactive maps using Folium. You can open them directly in any browser:
- `visualizations/maps/test_trip_00000000_map.html`
- `visualizations/maps/train_trip_00000001_map.html`

### What the Colors Mean:
- **Blue Circle Markers:** Raw GPS points (with popup showing time and observation index).
- **Gray Lines:** The entire underlying road network graph.
- **Dashed Green Line:** True vehicle route (ground truth).
- **Solid Red Line:** PathSnap predicted route (decoded by HMM/Viterbi).
- **Green Play / Red Stop Icons:** Start and end of the trip.
- **Layer Control:** Checkbox menu in the top right to turn individual layers on or off.

---

## 9. Academic Integrity & Why Our Code is 100% Leak-Free

If the evaluator asks how you ensured no data leakage:
1. **GroupKFold by Trip ID:** We never used standard random K-Fold. Standard K-Fold splits rows randomly, putting consecutive GPS points from the same trip into both train and validation folds. We grouped strictly by `trip_id` so validation always tested completely unseen trips.
2. **Untouched Evaluator Partition:** The 10 evaluator trips in `evaluator_test` were never opened, touched, or used.
3. **Preprocessing Fit on Train Only:** Our `StandardScaler` was fit strictly on the 80 training trips, and only `.transform()` was called on test data.
4. **Zero Hardcoded Routes:** All routes are generated dynamically through feature extraction $\rightarrow$ classifier $\rightarrow$ HMM decoding.

---

## 10. Cheat Sheet: Questions Evaluators Might Ask & Winning Answers

### Q: Why Macro F1 instead of simple Accuracy?
> **Answer:** "Because only ~5% of candidate roads are correct (95% are negatives). If a model predicted '0' for everything, it would get 95% accuracy while being completely useless! Macro F1 calculates the average score for both classes equally, punishing models that fail to find the true road."

### Q: Why did tree boosting (XGBoost/LightGBM) beat Neural Networks?
> **Answer:** "Tabular map-matching features have heterogeneous scales and discrete step-functions (like candidate rank $1, 2, 3$). Decision trees naturally split on discrete thresholds without needing smooth gradients, whereas neural nets required heavy normalization and were slower to converge."

### Q: What did HMM do that the classifier couldn't?
> **Answer:** "The classifier evaluates each GPS point in isolation, so it often jumps back and forth between parallel streets. The HMM adds graph connectivity constraints: transitioning to a disconnected road carries a massive penalty ($\epsilon = 10^{-7}$). Viterbi finds the globally optimal continuous path, increasing route continuity to 99.12%."

### Q: How did you prevent the pipeline from hanging on large road networks?
> **Answer:** "With 237,000 arcs per trip, naive distance checks take 15 minutes. We implemented a 2D spatial hash grid with 300m cells. This reduced candidate road lookups to $O(1)$, speeding up processing by **64x** (down to 0.15s per trip)."
