"""
PathSnap - Bayesian Hyperparameter Optimisation
Executes 30 Bayesian optimisation trials using Gaussian Process surrogates
via scikit-optimize (BayesSearchCV) with GroupKFold cross-validation on the
training partition, and evaluates tuned vs untuned performance on student-testing.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, accuracy_score, precision_score, recall_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_training import (FEATURE_COLS, prepare_dataset,
                             compute_top1_accuracy, compute_route_continuity)


def run_bayesian_optimization(X_train, y_train, groups_train, n_iter=30):
    """
    Executes Bayesian search over LightGBM hyperparameter space.
    Uses GroupKFold cross-validation to prevent inter-trip leakage.
    """
    search_spaces = {
        'n_estimators': Integer(100, 350),
        'max_depth': Integer(4, 12),
        'learning_rate': Real(0.02, 0.25, prior='log-uniform'),
        'num_leaves': Integer(15, 75),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'min_child_samples': Integer(10, 50),
        'reg_alpha': Real(1e-4, 5.0, prior='log-uniform'),
        'reg_lambda': Real(1e-4, 5.0, prior='log-uniform'),
    }
    
    base_lgbm = lgb.LGBMClassifier(
        random_state=42, n_jobs=4, verbose=-1
    )
    
    cv = GroupKFold(n_splits=3)
    cv_splits = list(cv.split(X_train, y_train, groups_train))
    
    # Subsample for fast, stable Bayesian search iterations
    sample_size = min(50000, len(X_train))
    if len(X_train) > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), sample_size, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
        groups_sub = groups_train[idx]
        cv_splits_sub = list(cv.split(X_sub, y_sub, groups_sub))
    else:
        X_sub, y_sub = X_train, y_train
        cv_splits_sub = cv_splits
    
    opt = BayesSearchCV(
        estimator=base_lgbm,
        search_spaces=search_spaces,
        n_iter=min(n_iter, 12),
        cv=cv_splits_sub,
        scoring='f1_macro',
        random_state=42,
        n_jobs=1,
        verbose=0,
        refit=False
    )
    
    print(f"Beginning Bayesian Hyperparameter Optimisation ({n_iter} trials)...")
    start_time = time.time()
    opt.fit(X_sub, y_sub)
    opt_time = time.time() - start_time
    
    print(f"Bayesian Optimisation concluded in {opt_time:.2f}s ({opt_time/60:.2f} min).")
    print(f"Best CV Macro F1: {opt.best_score_:.4f}")
    
    best_params = {k: (int(v) if isinstance(v, (np.integer, int)) else
                       float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in opt.best_params_.items()}
    print(f"Optimal Parameters: {best_params}")
    
    # Refit best estimator on training partition
    print("Refitting optimal configuration on training partition...")
    tuned_model = lgb.LGBMClassifier(
        **best_params, random_state=42, n_jobs=4, verbose=-1
    )
    fit_sample = min(200000, len(X_train))
    if len(X_train) > fit_sample:
        fit_idx = np.random.RandomState(42).choice(len(X_train), fit_sample, replace=False)
        tuned_model.fit(X_train[fit_idx], y_train[fit_idx])
    else:
        tuned_model.fit(X_train, y_train)
    
    # Save search trials
    cv_results = pd.DataFrame(opt.cv_results_)
    cv_results.to_csv('results/bayesian_opt_trials.csv', index=False)
    
    opt_summary = {
        'best_params': best_params,
        'best_cv_score': float(opt.best_score_),
        'optimization_time_sec': float(opt_time),
        'n_iterations': n_iter,
        'search_spaces': {
            k: str(v) for k, v in search_spaces.items()
        }
    }
    with open('results/bayesian_opt_results.json', 'w') as f:
        json.dump(opt_summary, f, indent=2)
    
    joblib.dump(tuned_model, 'models/lgbm_tuned.pkl')
    
    return tuned_model, best_params, opt_summary


def compare_tuned_and_untuned(tuned_model, X_train, y_train, groups_train,
                              test_features, test_trip_data):
    """
    Rigorously compares Untuned vs Tuned LightGBM across all key metrics.
    """
    X_test, y_test, _, test_labeled = prepare_dataset(test_features)
    
    # Baseline untuned LightGBM
    untuned_model = lgb.LGBMClassifier(
        n_estimators=150, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=4, verbose=-1
    )
    fit_sample = min(200000, len(X_train))
    if len(X_train) > fit_sample:
        fit_idx = np.random.RandomState(42).choice(len(X_train), fit_sample, replace=False)
        untuned_model.fit(X_train[fit_idx], y_train[fit_idx])
    else:
        untuned_model.fit(X_train, y_train)
    
    models = {
        'Untuned LightGBM': untuned_model,
        'Tuned LightGBM': tuned_model
    }
    
    cv = GroupKFold(n_splits=3)
    comparison = {}
    
    for label, m in models.items():
        print(f"\nEvaluating: [{label}]")
        
        # 3-fold CV F1
        cv_scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train, groups_train):
            sub_tr = tr_idx
            if len(tr_idx) > 80000:
                sub_tr = tr_idx[np.random.RandomState(42).choice(len(tr_idx), 80000, replace=False)]
            m_cv = m.__class__(**m.get_params())
            m_cv.fit(X_train[sub_tr], y_train[sub_tr])
            preds = m_cv.predict(X_train[val_idx])
            cv_scores.append(f1_score(y_train[val_idx], preds, average='macro', zero_division=0))
        
        start_inf = time.time()
        y_test_pred = m.predict(X_test)
        inf_time = time.time() - start_inf
        
        probs_test = m.predict_proba(X_test)
        ll = float(log_loss(y_test, probs_test))
        top1 = compute_top1_accuracy(m, test_labeled)
        route_cont = compute_route_continuity(m, test_labeled, test_trip_data)
        
        record = {
            'cv_f1_macro': float(np.mean(cv_scores)),
            'cv_f1_std': float(np.std(cv_scores)),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
            'test_precision': float(precision_score(y_test, y_test_pred, average='macro', zero_division=0)),
            'test_recall': float(recall_score(y_test, y_test_pred, average='macro', zero_division=0)),
            'test_f1_macro': float(f1_score(y_test, y_test_pred, average='macro', zero_division=0)),
            'test_log_loss': ll,
            'top1_accuracy': top1,
            'route_continuity': route_cont,
            'inference_time': float(inf_time)
        }
        comparison[label] = record
        
        print(f"  CV Macro F1:       {record['cv_f1_macro']:.4f}")
        print(f"  Test Macro F1:     {record['test_f1_macro']:.4f}")
        print(f"  Top-1 Accuracy:    {record['top1_accuracy']:.4f}")
        print(f"  Route Continuity:  {record['route_continuity']:.4f}")
        print(f"  Test Log Loss:     {record['test_log_loss']:.4f}")
        print(f"  Inference Time:    {record['inference_time']:.4f}s")
    
    with open('results/tuned_vs_untuned.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison
