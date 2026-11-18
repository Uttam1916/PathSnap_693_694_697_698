"""
PathSnap - Multi-Model Training, Evaluation, and Comparison Pipeline
Trains and rigorously benchmarks 6 distinct models from multiple algorithm families:
1. Linear/Probabilistic: Logistic Regression
2. Instance-based: K-Nearest Neighbors (KNN)
3. Ensemble: Random Forest
4. Boosting: XGBoost
5. Boosting: LightGBM
6. Neural Network: Multi-Layer Perceptron (MLP)

Uses GroupKFold cross-validation on 80 training trips (grouped by trip_id)
to strictly prevent spatial-temporal data leakage, and evaluates on 10 student-test trips.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, log_loss, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEATURE_COLS = [
    'distance', 'bearing_diff', 'arc_length', 'gps_speed',
    'candidate_rank', 'norm_distance', 'dist_ratio', 'n_candidates',
    'cos_bearing_diff', 'speed_compatibility', 'dist_to_from_node',
    'dist_to_to_node', 'frac_position', 'log_distance', 'bear_diff_squared'
]


def prepare_dataset(features_df):
    """Prepares feature matrix X, binary targets y, and group identifiers."""
    labeled = features_df[features_df['label'] >= 0].copy()
    X = labeled[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = labeled['label'].values.astype(int)
    groups = labeled['trip_id'].values
    return X, y, groups, labeled


def compute_top1_accuracy(model, features_df, scaler=None):
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation, select the candidate road segment with the highest
    predicted probability and verify whether it matches the ground truth reference arc.
    """
    df = features_df.copy()
    X = df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    if scaler is not None:
        X = scaler.transform(X)
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        scores = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
    else:
        scores = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict(X)
    
    df['pred_score'] = scores
    
    # Per-observation argmax
    correct = 0
    total = 0
    for (trip_id, obs_idx), group in df.groupby(['trip_id', 'obs_idx']):
        total += 1
        best_cand = group.loc[group['pred_score'].idxmax()]
        # Check if best candidate has label == 1
        if best_cand['label'] == 1:
            correct += 1
    
    return float(correct / max(total, 1))


def compute_route_continuity(model, features_df, test_trip_data, scaler=None):
    """
    Route Continuity:
    Measures the fraction of consecutive GPS observation predictions that form
    a continuous, valid path on the road network graph.
    """
    df = features_df.copy()
    X = df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    if scaler is not None:
        X = scaler.transform(X)
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        df['score'] = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
    else:
        df['score'] = model.predict(X).astype(float)
    
    continuity_scores = []
    for trip_id, t_df in df.groupby('trip_id'):
        lookup_id = str(trip_id).zfill(8)
        if lookup_id not in test_trip_data:
            continue
        meta = test_trip_data[lookup_id]
        adj = meta['adj']
        arc_info = meta['arc_info']
        
        # Sort chronologically by observation
        sorted_obs = t_df.sort_values('obs_idx')
        best_preds = sorted_obs.loc[sorted_obs.groupby('obs_idx')['score'].idxmax()]
        pred_arcs = best_preds['arc_id'].values
        
        connected = 0
        pairs = max(len(pred_arcs) - 1, 1)
        for i in range(len(pred_arcs) - 1):
            a1, a2 = pred_arcs[i], pred_arcs[i + 1]
            if a1 == a2:
                connected += 1
            elif a1 in arc_info and a2 in arc_info:
                to1 = arc_info[a1]['to_node']
                from2 = arc_info[a2]['from_node']
                if to1 == from2:
                    connected += 1
                else:
                    # 1-hop neighbor
                    nbrs = [n for n, _ in adj.get(to1, [])]
                    if from2 in nbrs:
                        connected += 1
        continuity_scores.append(connected / pairs)
    
    return float(np.mean(continuity_scores)) if continuity_scores else 0.0


def get_models_suite():
    """Initializes the 6 distinct model specifications across algorithmic families."""
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                max_iter=500, C=1.0, solver='lbfgs',
                class_weight='balanced', random_state=42
            ),
            'family': 'Linear and Probabilistic',
            'config': 'C=1.0, solver=lbfgs, class_weight=balanced, max_iter=500',
            'scale': True,
            'subsample_train': None
        },
        'KNN': {
            'model': KNeighborsClassifier(
                n_neighbors=7, weights='distance', metric='minkowski', n_jobs=-1
            ),
            'family': 'Instance-based',
            'config': 'k=7, weights=distance, metric=minkowski',
            'scale': True,
            'subsample_train': 50000  # Fast inference sample
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100, max_depth=12, max_samples=0.25,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'family': 'Ensemble',
            'config': 'n_estimators=100, max_depth=12, max_samples=0.25, balanced',
            'scale': False,
            'subsample_train': None
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                tree_method='hist', random_state=42, n_jobs=-1,
                eval_metric='logloss'
            ),
            'family': 'Boosting',
            'config': 'n_estimators=150, max_depth=6, lr=0.1, tree_method=hist',
            'scale': False,
            'subsample_train': 250000
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            ),
            'family': 'Boosting',
            'config': 'n_estimators=200, max_depth=8, lr=0.1, subsample=0.8',
            'scale': False,
            'subsample_train': None
        },
        'MLP Neural Network': {
            'model': MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=30, batch_size=512,
                early_stopping=True, random_state=42
            ),
            'family': 'Neural Networks',
            'config': 'hidden=(64, 32), batch_size=512, early_stopping=True',
            'scale': True,
            'subsample_train': 100000
        }
    }
    return models


def train_and_evaluate_all(train_features, test_features, train_trip_data, test_trip_data):
    """Executes full GroupKFold cross-validation, test evaluation, and metrics extraction."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    X_train, y_train, groups_train, train_labeled = prepare_dataset(train_features)
    X_test, y_test, groups_test, test_labeled = prepare_dataset(test_features)
    
    # Fit scaler on training set strictly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    models_dict = get_models_suite()
    all_results = []
    best_model_name = None
    best_f1 = -1.0
    
    print("\n" + "=" * 80)
    print("STARTING BENCHMARK ACROSS 6 ALGORITHM FAMILIES")
    print("=" * 80)
    
    cv = GroupKFold(n_splits=5)
    
    for name, spec in models_dict.items():
        print(f"\nEvaluating Model: [{name}] (Family: {spec['family']})")
        model = spec['model']
        scale = spec['scale']
        subsample = spec['subsample_train']
        
        X_tr = X_train_scaled if scale else X_train
        X_te = X_test_scaled if scale else X_test
        
        # 1. 5-Fold GroupKFold Cross-Validation on Training Trips
        print(f"  Running 5-Fold GroupKFold CV...")
        cv_f1_scores = []
        cv_splits = list(cv.split(X_tr, y_train, groups_train))
        for fold, (tr_idx, val_idx) in enumerate(cv_splits):
            if subsample and len(tr_idx) > subsample:
                rng = np.random.RandomState(42 + fold)
                sub_idx = tr_idx[rng.choice(len(tr_idx), subsample, replace=False)]
            else:
                sub_idx = tr_idx
            
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_tr[sub_idx], y_train[sub_idx])
            val_preds = fold_model.predict(X_tr[val_idx])
            f1_m = f1_score(y_train[val_idx], val_preds, average='macro', zero_division=0)
            cv_f1_scores.append(f1_m)
        
        cv_f1_mean = float(np.mean(cv_f1_scores))
        cv_f1_std = float(np.std(cv_f1_scores))
        print(f"  CV Macro F1: {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})")
        
        # 2. Fit on Full Permitted Training Partition
        start_fit = time.time()
        if subsample and len(X_tr) > subsample:
            rng = np.random.RandomState(42)
            fit_idx = rng.choice(len(X_tr), subsample, replace=False)
            model.fit(X_tr[fit_idx], y_train[fit_idx])
        else:
            model.fit(X_tr, y_train)
        fit_time = time.time() - start_fit
        print(f"  Fit time: {fit_time:.2f}s")
        
        # 3. Evaluate on Student-Testing Partition
        start_inf = time.time()
        y_pred_test = model.predict(X_te)
        inf_time = time.time() - start_inf
        
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        
        # Training accuracy for overfitting analysis
        # Sample for quick train accuracy check
        train_sample_idx = np.random.RandomState(42).choice(len(X_tr), min(50000, len(X_tr)), replace=False)
        train_acc = accuracy_score(y_train[train_sample_idx], model.predict(X_tr[train_sample_idx]))
        
        # Log loss
        if hasattr(model, 'predict_proba'):
            try:
                probs_test = model.predict_proba(X_te)
                test_ll = float(log_loss(y_test, probs_test))
            except:
                test_ll = float('nan')
        else:
            test_ll = float('nan')
        
        # Observation-level Top-1 Accuracy & Route Continuity
        top1_acc = compute_top1_accuracy(model, test_labeled, scaler if scale else None)
        route_cont = compute_route_continuity(model, test_labeled, test_trip_data, scaler if scale else None)
        
        cm = confusion_matrix(y_test, y_pred_test).tolist()
        
        print(f"  Test Accuracy:        {test_acc:.4f}")
        print(f"  Test Macro F1:        {test_f1:.4f}")
        print(f"  Top-1 Accuracy:       {top1_acc:.4f}")
        print(f"  Route Continuity:     {route_cont:.4f}")
        print(f"  Inference Time:       {inf_time:.4f}s")
        
        # Save model artifact
        slug = name.replace(" ", "_").lower()
        joblib.dump(model, f"models/{slug}.pkl")
        
        res_record = {
            'model': name,
            'family': spec['family'],
            'config': spec['config'],
            'cv_f1_macro': cv_f1_mean,
            'cv_f1_std': cv_f1_std,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1_macro': float(test_f1),
            'test_log_loss': test_ll,
            'top1_accuracy': top1_acc,
            'route_continuity': route_cont,
            'train_time': float(fit_time),
            'inference_time': float(inf_time),
            'confusion_matrix': cm
        }
        all_results.append(res_record)
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name
    
    # Save full comparison table
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    with open('results/model_comparison_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"SELECTION OUTCOME: Best Model is '{best_model_name}' (Test F1: {best_f1:.4f})")
    print("=" * 80)
    
    return results_df, best_model_name


if __name__ == "__main__":
    from data_processing import process_all_trips_fast

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "=" * 80)
    print("PATHSNAP: 6-MODEL BENCHMARK SUITE & GROUPKFOLD CROSS-VALIDATION")
    print("=" * 80)
    
    if not (os.path.exists("results/train_features.csv") and os.path.exists("results/test_features.csv")):
        print("Error: Feature files not found! Please run python3 src/data_processing.py first.")
        sys.exit(1)
        
    print("Loading preprocessed candidate features...")
    train_features = pd.read_csv("results/train_features.csv")
    test_features = pd.read_csv("results/test_features.csv")
    
    print("Loading test trajectory metadata for Top-1 accuracy and route continuity...")
    _, test_meta = process_all_trips_fast("Student folder/student_test", max_candidates=15, verbose=False)
    
    print("\nTraining and evaluating 6 models across 5 algorithm families...")
    results_df, best_model_name = train_and_evaluate_all(
        train_features, test_features, {}, test_meta
    )
    
    print("\nConsolidated Performance Benchmark:")
    print(results_df[['model', 'family', 'cv_f1_macro', 'test_f1_macro', 'top1_accuracy', 'route_continuity', 'test_log_loss', 'train_time']].to_string(index=False))

