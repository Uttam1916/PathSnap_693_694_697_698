# PathSnap - Complete Viva Explanation Guide
### Everything Explained in Simple Words

---

## 1. What is the Problem? (Problem Understanding)

**In simple words:** Cars have GPS but GPS is not perfectly accurate. When a car drives on a road, the GPS might show it on a nearby parallel road, or even jumping across buildings. Our job is to take these messy GPS points and figure out which exact roads the car actually drove on.

**The technical name:** This is called **Map Matching** — matching GPS observations to road segments on a map.

**What we're given:**
- `.track` files = GPS points (longitude, latitude, timestamp) — where the GPS *thinks* the car was
- `.nodes` files = intersection points on the road network (like road junctions)
- `.arcs` files = road segments connecting nodes (the actual roads)
- `.route` files = the ground truth — which roads the car actually took (for training)

---

## 2. How is the Data Organised? (Dataset Understanding)

We have **three partitions** (never mix them!):

| Partition | Trips | Purpose |
|-----------|-------|---------|
| Training | 80 trips | Train models, cross-validate |
| Student Test | 10 trips | Compare models during development |
| Evaluator Test | 10 trips | Final evaluation only (DON'T TOUCH) |

**Key observations from our EDA:**
- Average ~2605 GPS observations per trip
- Average ~24,169 road nodes and ~48,461 road arcs per trip
- Average route length: ~392 arcs
- Most time gaps are 1 second (97%), but some have large gaps up to 153,064s
- Trajectories average ~37 km in length
- Data is from Germany (Mannheim area, coordinates around 8.5°E, 49.5°N)

**Why this matters:** Large road networks mean lots of possible candidate roads for each GPS point. We need to be smart about narrowing down candidates.

---

## 3. How Did We Process the Data? (Data Processing)

### Step 1: Load Raw Data
We read all four file types for each trip. No headers in the files — we know the columns from the documentation.

### Step 2: Build the Road Network Graph
We create an **adjacency list** from the arcs file. For each road segment (arc), we calculate:
- Its physical length (using haversine formula for lat/lon)
- Its compass bearing (direction)

**Why adjacency list?** We need to know which roads connect to which roads — this is crucial for route continuity later.

### Step 3: Generate Candidates
For each GPS observation, we find nearby road segments that the car *could* be on.

**How?** We use a **spatial grid index** (dividing the map into ~300m grid cells) for efficiency. For each GPS point, we look at nearby grid cells and calculate the perpendicular distance to each arc in those cells.

**Why spatial grid?** Without it, we'd check every single arc for every GPS point — that would be O(n×m) which is way too slow with 165,000+ arcs and 1,300+ GPS points per trip. The grid reduces this dramatically.

**Parameters:**
- `radius_m = 200`: Only consider arcs within 200 meters
- `max_candidates = 15`: Keep at most 15 closest candidates per observation

### Step 4: Engineer Features
For each (GPS_observation, candidate_arc) pair, we compute **15 features**:

| # | Feature | Why This Feature? |
|---|---------|-------------------|
| 1 | `distance` | How far is the GPS point from this road? Closer = more likely |
| 2 | `bearing_diff` | Is the car heading in the same direction as the road? |
| 3 | `arc_length` | Longer roads have more chance of being traversed |
| 4 | `gps_speed` | Speed of the car (useful for distinguishing highway vs local road) |
| 5 | `candidate_rank` | Is this the closest candidate or further down the list? |
| 6 | `norm_distance` | Distance relative to the furthest candidate |
| 7 | `dist_ratio` | How many times further than the closest candidate? |
| 8 | `n_candidates` | Number of nearby roads (urban areas have more) |
| 9 | `cos_bearing_diff` | Cosine of direction difference (smoother than raw angle) |
| 10 | `speed_compatibility` | Is the GPS speed reasonable for this type of road? |
| 11 | `dist_to_from_node` | Distance to the start of the road segment |
| 12 | `dist_to_to_node` | Distance to the end of the road segment |
| 13 | `frac_position` | Where along the road segment is the car? (0 = start, 1 = end) |
| 14 | `log_distance` | Log-transformed distance (reduces impact of outliers) |
| 15 | `bear_diff_squared` | Squared bearing diff (penalizes large direction mismatches more) |

**Why these features?** Each feature captures a different aspect of "how likely is it that this GPS point corresponds to this road segment":
- **Distance features (1,5,6,7,14):** The main signal — GPS noise follows a roughly Gaussian pattern, so closer roads are more likely
- **Direction features (2,9,15):** A car moving north is unlikely to be on an east-west road
- **Context features (3,4,8,10):** Additional information about the road and driving behavior
- **Position features (11,12,13):** Where exactly on the road the projection falls

### Step 5: Assign Labels
For training data, we know the ground truth route (sequence of arc IDs). We map each GPS observation to its corresponding ground truth arc by distributing observations proportionally across the route.

Label = 1 if the candidate arc matches the ground truth, 0 otherwise.

**Important:** The data is very imbalanced (~99% negative, ~1% positive) because only one of the ~15 candidates per observation is correct.

---

## 4. Why Did We Choose These Distance Calculations? (Geographical Considerations)

**Haversine distance** instead of Euclidean distance because:
- Lat/lon are on a sphere, not a flat plane
- 1 degree of longitude ≈ 72 km at this latitude, 1 degree of latitude ≈ 111 km
- Using Euclidean would give wrong distances

**Point-to-segment distance** because:
- A GPS point should be matched to the *closest point on the road*, not just the endpoints
- We project the GPS point perpendicularly onto the road segment line

---

## 5. Model Selection & Experimentation (6 Models)

We trained **6 models** from different algorithm families:

### Model 1: Logistic Regression (Linear/Probabilistic)
- **Why suitable?** Simple baseline that shows if the problem is linearly separable
- **Strengths:** Fast, interpretable, gives calibrated probabilities
- **Limitations:** Assumes linear decision boundary — GPS noise patterns may not be linear
- **Config:** C=1.0, L-BFGS solver, balanced class weights
- **Loss function:** Cross-entropy (log loss)

### Model 2: KNN (Instance-based)
- **Why suitable?** Similar GPS patterns should match similar road segments
- **Strengths:** No training needed, captures local patterns
- **Limitations:** Slow inference (needs to compare with all training points), sensitive to feature scaling
- **Config:** k=7, distance-weighted, minkowski metric
- **Why k=7?** Small k is noisy, large k is too smooth. 7 is a reasonable balance.

### Model 3: Random Forest (Ensemble)
- **Why suitable?** Can capture nonlinear relationships between features, robust to outliers
- **Strengths:** Resistant to overfitting, handles imbalanced data well with class weights
- **Limitations:** Can be memory-intensive, produces "blocky" probability estimates
- **Config:** 200 trees, max_depth=15, balanced class weights
- **Loss function:** Gini impurity

### Model 4: XGBoost (Boosting)
- **Why suitable?** Sequential error correction often gives best performance on tabular data
- **Strengths:** Handles imbalanced data, built-in regularization, very fast
- **Limitations:** More hyperparameters to tune, can overfit if not regularized
- **Config:** 200 trees, learning_rate=0.1, max_depth=8, subsample=0.8
- **Loss function:** Binary log loss

### Model 5: LightGBM (Boosting)
- **Why suitable?** Similar to XGBoost but faster on large datasets due to histogram-based splitting
- **Strengths:** Very fast training, memory efficient, handles large feature spaces
- **Limitations:** Can overfit on small datasets, leaf-wise growth can be unstable
- **Config:** 200 trees, learning_rate=0.1, max_depth=10, subsample=0.8
- **Loss function:** Binary log loss

### Model 6: MLP Neural Network
- **Why suitable?** Can learn complex nonlinear feature interactions
- **Strengths:** Universal function approximator, can capture hidden patterns
- **Limitations:** Needs careful tuning, can overfit, slower to train
- **Config:** 3 hidden layers (128, 64, 32 neurons), adaptive learning rate, early stopping
- **Loss function:** Cross-entropy

### How We Evaluated:
- **GroupKFold Cross-Validation** (5 folds) on training data — groups by trip_id so no data leakage
- **Student-test evaluation** on the 10 held-out test trips
- Metrics: Accuracy, Precision, Recall, Macro F1, Log Loss, Top-1 Accuracy, Route Continuity

### Why GroupKFold?
Regular K-Fold might put some GPS points from the same trip in both train and validation — that's data leakage! GroupKFold ensures entire trips stay together.

---

## 6. Bayesian Hyperparameter Optimisation

### Why LightGBM?
We selected LightGBM for optimisation because it showed the best balance of speed and performance during initial experiments.

### Why Bayesian instead of Grid Search?
- **Grid Search:** Tries every combination — exponential in number of parameters
- **Random Search:** Faster but doesn't learn from previous trials
- **Bayesian (skopt):** Uses a probabilistic model (Gaussian Process) to predict which hyperparameters might work best, then focuses search there. Gets better results in fewer trials.

### Hyperparameters Tuned:

| Parameter | Range | Why This Range? |
|-----------|-------|-----------------|
| `n_estimators` | 100-500 | More trees = better but slower |
| `max_depth` | 5-20 | Controls tree complexity |
| `learning_rate` | 0.01-0.3 (log) | Lower = more precise but needs more trees |
| `num_leaves` | 20-100 | LightGBM leaf count (more = more complex) |
| `subsample` | 0.6-1.0 | Row subsampling for regularization |
| `colsample_bytree` | 0.6-1.0 | Column subsampling for regularization |
| `min_child_samples` | 5-50 | Minimum data per leaf (prevents overfitting) |
| `reg_alpha` | 1e-6 to 10 (log) | L1 regularization |
| `reg_lambda` | 1e-6 to 10 (log) | L2 regularization |

### Process:
1. 30 Bayesian optimization trials
2. Each trial does 5-fold GroupKFold cross-validation
3. Objective: Maximize macro F1-score
4. Refit best model on all training data
5. Compare tuned vs untuned on student-test set

---

## 7. HMM & Viterbi Route Reconstruction

### Why Do We Need This?
The classifier picks the best road for each GPS point *independently*. But a car drives on *connected* roads! The classifier might predict:
- GPS point 1 → Road A
- GPS point 2 → Road B (which doesn't connect to Road A!)

That gives us a "broken" route. HMM fixes this.

### How HMM Works (Simple Explanation):

Think of it like autocorrect on your phone:
- Each key you press = GPS observation
- Each word suggestion = candidate road segment
- The phone picks words that make sense in sequence, not just the closest key

Similarly:
- **Hidden States:** The actual road segments the car is on (what we want to find)
- **Observations:** The GPS coordinates we see
- **Emission Probability:** How likely is this GPS point given we're on this road? (Based on distance + classifier probability)
- **Transition Probability:** How likely is it to go from Road A to Road B? (Based on road connectivity)

### Emission Probability:
```
P(GPS_obs | road_segment) = Gaussian(distance, σ=50m) × classifier_probability
```
- **Why Gaussian?** GPS errors roughly follow a normal distribution
- **Why σ=50m?** Typical GPS error in urban areas is 10-50 meters
- **Why multiply by classifier?** The classifier learned more complex patterns than just distance

### Transition Probability:
```
P(road_B | road_A) depends on:
1. Are road_A and road_B directly connected? (High probability)
2. Is the route distance consistent with GPS distance? (Higher if consistent)
3. Is there a large time gap? (Allow more flexibility)
```
- **Same arc:** 0.5 (car is still on the same road)
- **Connected arcs:** Exponential decay based on distance discrepancy
- **Disconnected arcs:** Very low penalty (1e-8)
- **Large GPS gaps:** Relaxed constraints (car may have traveled through unknown roads)

### Viterbi Algorithm:
Instead of trying all possible combinations (exponential!), Viterbi uses **dynamic programming**:
1. Start: Score each candidate for the first GPS point
2. For each next GPS point: For each candidate, find the best previous candidate (considering both emission and transition)
3. Backtrack: Follow the best path backward from the end

**Time complexity:** O(T × S²) where T = number of observations, S = max candidates per observation

### Handling Edge Cases:
- **Missing candidates:** Skip observations with no candidates, interpolate
- **Large GPS gaps:** Relax transition constraints (the car may have taken a path we can't trace)
- **Disconnected segments:** Apply a small penalty rather than zero (avoid eliminating valid paths)

### Results:
- **Classifier-only:** Gets the top-1 prediction right for each point independently
- **Classifier + HMM:** Produces coherent connected routes with improved route continuity

---

## 8. Feature Engineering Decisions Explained

### Why distance is the most important feature?
GPS noise is essentially random displacement. The true road is almost always one of the closest candidates. Distance alone gives us ~70%+ accuracy.

### Why bearing difference matters?
If you're driving north, you're on a north-south road, not an east-west one. This eliminates many false candidates, especially at intersections where multiple roads cross.

### Why we use both raw and transformed features?
- `distance` + `log_distance`: Log handles the long tail (some candidates are far away)
- `bearing_diff` + `cos_bearing_diff` + `bear_diff_squared`: Different representations help different models learn the pattern
- `dist_ratio`: If the closest candidate is 5m away and this one is 50m, the ratio (10) is very informative

### Why candidate_rank?
Instead of just distance, the rank tells the model "this is the 1st closest" vs "this is the 10th closest" — a simpler signal that some models can use more easily.

---

## 9. How We Prevent Data Leakage

1. **GroupKFold:** Entire trips stay together during cross-validation
2. **Separate partitions:** Student-test data never used for training
3. **Evaluator-test data:** Never touched during development
4. **Scaler fit on training only:** StandardScaler fitted on training data, applied to test
5. **No hardcoded predictions:** All predictions go through the ML pipeline

---

## 10. Evaluation Metrics Explained

| Metric | What It Measures | Why We Use It |
|--------|-----------------|---------------|
| **Accuracy** | % of predictions correct | Basic but misleading with imbalanced data |
| **Precision** | Of predicted positives, how many are truly positive? | Avoids false positives |
| **Recall** | Of actual positives, how many did we find? | Avoids missing the true road |
| **F1 (Macro)** | Harmonic mean of precision & recall, averaged equally across classes | **Primary metric** — balances both |
| **Log Loss** | Penalizes confident wrong predictions | Measures probability quality |
| **Top-1 Accuracy** | For each GPS point, is the highest-scored candidate correct? | Direct measure of map-matching quality |
| **Candidate Miss Rate** | % of GPS points where true road isn't in candidates | Shows if candidate generation is good enough |
| **Route Continuity** | % of consecutive predictions that are connected | Measures route coherence |

### Why Macro F1 as Primary Metric?
- Accuracy alone is misleading (predicting all 0s gives ~99% accuracy!)
- Macro F1 treats both classes equally
- Reflects both precision and recall

---

## 11. Overfitting Analysis

**Signs of overfitting:** Train accuracy >> Test accuracy
- If a model gets 99% on train but 60% on test → overfitting
- Random Forest and KNN are prone to this

**How we prevent it:**
- Cross-validation (5-fold with group awareness)
- Regularization (L1/L2 in LightGBM, early stopping in MLP)
- Balanced class weights
- Subsample parameters in XGBoost/LightGBM

---

## 12. Project Architecture Summary

```
Raw Data (.track, .nodes, .arcs, .route)
    ↓
Data Processing (spatial grid, haversine)
    ↓
Candidate Generation (nearby arcs per GPS point)
    ↓
Feature Engineering (15 features per candidate)
    ↓
Model Training (6 models compared)
    ↓
Bayesian Optimisation (tune best model)
    ↓
HMM/Viterbi Decoding (coherent route)
    ↓
Evaluation & Visualisation
```

---

## 13. Key Design Decisions Summary

| Decision | Why |
|----------|-----|
| Spatial grid indexing | Speed: O(1) lookup instead of O(n) per GPS point |
| 200m candidate radius | Covers 99%+ of GPS errors while keeping candidate count manageable |
| 15 max candidates | Enough to include true road while keeping computation fast |
| 15 features | Comprehensive coverage of distance, direction, speed, and context signals |
| 6 different model families | Shows systematic experimentation across algorithm types |
| GroupKFold | Prevents trip-level data leakage |
| Bayesian optimization | More efficient than grid/random search |
| LightGBM as final model | Best speed-performance tradeoff on tabular data |
| HMM with Viterbi | Ensures route connectivity using road network structure |
| Gaussian emission model (σ=50m) | Matches empirical GPS error distribution |

---

*This document covers all topics that may come up during the viva evaluation.*
