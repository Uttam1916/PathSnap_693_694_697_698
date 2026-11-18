"""
PathSnap - Master Pipeline Orchestrator
Executes the full end-to-end PathSnap workflow:
1. Dataset Exploration and Summary Generation
2. Candidate Generation, Ground Truth Alignment, and Feature Engineering
3. Multi-Model Benchmark (6 Models across Algorithm Families) with GroupKFold CV
4. Bayesian Hyperparameter Optimisation (30 Iterations)
5. HMM + Viterbi Sequential Trajectory Decoding
6. Comprehensive Visualization and Folium Interactive Map Generation
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np

# Set project working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from data_processing import dataset_summary, process_all_trips_fast
from model_training import train_and_evaluate_all, prepare_dataset
from bayesian_optimization import run_bayesian_optimization, compare_tuned_and_untuned
from hmm_viterbi import evaluate_route_reconstruction, HMMRouteReconstructor
from visualizations import (plot_eda_summary, plot_model_comparison,
                             plot_bayesian_optimization_charts,
                             plot_hmm_reconstruction_charts,
                             create_folium_trajectory_map)


def main():
    t_start = time.time()
    
    train_dir = "Student folder/train"
    test_dir = "Student folder/student_test"
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/maps', exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PATHSNAP MASTER WORKFLOW INITIATED")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: DATASET EXPLORATION & SUMMARY
    # =========================================================================
    print("\n[STEP 1/6] Running Exploratory Data Analysis & Partition Summaries...")
    train_summary = dataset_summary(train_dir)
    test_summary = dataset_summary(test_dir)
    
    train_summary.to_csv("results/train_summary.csv", index=False)
    test_summary.to_csv("results/test_summary.csv", index=False)
    
    print(f"  Training Trips:         {len(train_summary)} trips")
    print(f"  Training Observations:  {train_summary['n_observations'].sum():,} GPS points")
    print(f"  Avg Obs / Trip:         {train_summary['n_observations'].mean():.1f}")
    print(f"  Avg Road Arcs / Trip:   {train_summary['n_arcs'].mean():.1f}")
    print(f"  Nominal Sampling Rate:  {train_summary['mean_time_gap'].mean():.2f}s (~1 Hz)")
    print(f"  Student Test Trips:     {len(test_summary)} trips")
    print(f"  Test Observations:      {test_summary['n_observations'].sum():,} GPS points")
    
    # =========================================================================
    # STEP 2: DATA PROCESSING & FEATURE EXTRACTION
    # =========================================================================
    print("\n[STEP 2/6] Candidate Generation & Vectorized Feature Engineering...")
    t_proc = time.time()
    
    if os.path.exists("results/train_features.csv") and os.path.exists("results/test_features.csv"):
        print("  Loading existing preprocessed features from disk...")
        train_features = pd.read_csv("results/train_features.csv")
        test_features = pd.read_csv("results/test_features.csv")
        print("  Loading metadata for test trips...")
        _, test_meta = process_all_trips_fast(test_dir, max_candidates=15, verbose=False)
        train_meta = {}
    else:
        print("  Processing 80 Training Trips...")
        train_features, train_meta = process_all_trips_fast(train_dir, max_candidates=15, verbose=True)
        train_features.to_csv("results/train_features.csv", index=False)
        print("  Processing 10 Student Test Trips...")
        test_features, test_meta = process_all_trips_fast(test_dir, max_candidates=15, verbose=True)
        test_features.to_csv("results/test_features.csv", index=False)
    
    print(f"  Feature Preparation Completed in {time.time() - t_proc:.2f}s")
    print(f"  Training Candidates:    {len(train_features):,} rows across 15 features")
    print(f"  Test Candidates:        {len(test_features):,} rows across 15 features")
    
    # =========================================================================
    # STEP 3: MODEL TRAINING & BENCHMARKING (6 ALGORITHM FAMILIES)
    # =========================================================================
    print("\n[STEP 3/6] Multi-Model Training & GroupKFold Cross-Validation...")
    if os.path.exists("results/model_comparison.csv"):
        print("  Loading benchmark results from results/model_comparison.csv...")
        results_df = pd.read_csv("results/model_comparison.csv")
    else:
        results_df, best_model_name = train_and_evaluate_all(
            train_features, test_features, train_meta, test_meta
        )
    
    print("\nConsolidated Performance Summary:")
    print(results_df[[
        'model', 'family', 'cv_f1_macro', 'test_f1_macro',
        'top1_accuracy', 'route_continuity', 'test_log_loss', 'train_time'
    ]].to_string(index=False))
    
    # =========================================================================
    # STEP 4: BAYESIAN HYPERPARAMETER OPTIMISATION
    # =========================================================================
    print("\n[STEP 4/6] Bayesian Hyperparameter Optimisation on Best Model...")
    X_train, y_train, groups_train, _ = prepare_dataset(train_features)
    
    tuned_lgbm, best_params, opt_summary = run_bayesian_optimization(
        X_train, y_train, groups_train, n_iter=10
    )
    
    comparison = compare_tuned_and_untuned(
        tuned_lgbm, X_train, y_train, groups_train, test_features, test_meta
    )
    
    # =========================================================================
    # STEP 5: HMM / VITERBI SEQUENTIAL ROUTE RECONSTRUCTION
    # =========================================================================
    print("\n[STEP 5/6] HMM / Viterbi Sequential Trajectory Decoding...")
    hmm_summary = evaluate_route_reconstruction(
        tuned_lgbm, test_features, test_meta
    )
    
    # =========================================================================
    # STEP 6: VISUALIZATION & FOLIUM MAPS
    # =========================================================================
    print("\n[STEP 6/6] Generating Folium Interactive Maps & Diagnostic Visualizations...")
    plot_eda_summary('results/train_summary.csv', 'results/test_summary.csv')
    plot_model_comparison('results/model_comparison.csv')
    plot_bayesian_optimization_charts('results/bayesian_opt_trials.csv', 'results/tuned_vs_untuned.json')
    plot_hmm_reconstruction_charts('results/hmm_comparison.json')
    
    # Interactive Folium map for student test trip 00000000
    test_trip_id = "00000000"
    if test_trip_id in test_meta:
        # Reconstruct route for map
        reconstructor = HMMRouteReconstructor()
        t_df = test_features[test_features['trip_id'].astype(str).str.zfill(8) == str(test_trip_id).zfill(8)]
        meta = test_meta[test_trip_id]
        obs_list = sorted(t_df['obs_idx'].unique())
        
        X_sub = t_df[test_features.columns.intersection(train_features.columns)].copy()
        X_sub_mat = np.nan_to_num(t_df[['distance', 'bearing_diff', 'arc_length', 'gps_speed',
                                        'candidate_rank', 'norm_distance', 'dist_ratio', 'n_candidates',
                                        'cos_bearing_diff', 'speed_compatibility', 'dist_to_from_node',
                                        'dist_to_to_node', 'frac_position', 'log_distance', 'bear_diff_squared']].values)
        probs = tuned_lgbm.predict_proba(X_sub_mat)[:, 1]
        
        cands_by_obs = {}
        cls_probs_dict = {}
        for idx, (_, r) in enumerate(t_df.iterrows()):
            o = int(r['obs_idx'])
            a = int(r['arc_id'])
            d = float(r['distance'])
            p = float(probs[idx])
            cands_by_obs.setdefault(o, []).append((a, d))
            cls_probs_dict[(o, a)] = p
        
        pred_path = reconstructor.decode(
            obs_list, cands_by_obs, meta['arc_info'], meta['adj'], meta['track'], cls_probs_dict
        )
        
        create_folium_trajectory_map(
            test_dir, test_trip_id, predictions_arcs=pred_path,
            output_html='visualizations/maps/test_trip_00000000_map.html'
        )
    
    # Also create map for training trip 00000001
    create_folium_trajectory_map(
        train_dir, "00000001", predictions_arcs=None,
        output_html='visualizations/maps/train_trip_00000001_map.html'
    )
    
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"PATHSNAP MASTER WORKFLOW COMPLETED SUCCESSFULLY IN {total_elapsed:.1f}s ({total_elapsed/60:.2f} MIN)")
    print("All artifacts, model weights, logs, plots, and interactive maps are generated.")
    print("=" * 80)


if __name__ == '__main__':
    main()
