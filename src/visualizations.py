"""
PathSnap - Comprehensive Visualization and Mapping Engine
Generates all mandatory EDA plots, model comparison charts, confusion matrices,
Bayesian optimization convergence curves, HMM reconstruction gains, and
interactive Folium geographical maps.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MiniMap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import load_trip, build_adjacency

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_eda_summary(train_summary_path, test_summary_path, output_dir='visualizations'):
    """Generates all exploratory data analysis distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.read_csv(train_summary_path)
    test_df = pd.read_csv(test_summary_path)
    
    # 1. GPS Observation Distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(train_df['n_observations'], bins=25, color='#3498db', alpha=0.75, label=f'Train (n={len(train_df)})', edgecolor='white')
    ax.hist(test_df['n_observations'], bins=10, color='#e67e22', alpha=0.8, label=f'Student Test (n={len(test_df)})', edgecolor='white')
    ax.set_title('Distribution of GPS Observations per Trip', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of GPS Measurements')
    ax.set_ylabel('Trip Frequency')
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_obs_distribution.png', dpi=200)
    plt.close()
    
    # 2. Trajectory Physical Lengths
    fig, ax = plt.subplots(figsize=(9, 4.5))
    lengths_km = train_df['trajectory_length_m'] / 1000.0
    ax.hist(lengths_km, bins=25, color='#2ecc71', alpha=0.75, edgecolor='white')
    ax.axvline(lengths_km.median(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Median: {lengths_km.median():.1f} km')
    ax.set_title('Physical Trajectory Length Distribution (Training Partition)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Trajectory Length (km)')
    ax.set_ylabel('Trip Frequency')
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_trajectory_lengths.png', dpi=200)
    plt.close()
    
    # 3. Sampling Gap Analysis
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(range(len(train_df)), train_df['max_time_gap'], color='#9b59b6', s=45, alpha=0.8, label='Max Time Gap per Trip')
    ax.axhline(1.0, color='#27ae60', linestyle='--', linewidth=1.5, label='Nominal 1 Hz (1s)')
    ax.set_yscale('log')
    ax.set_title('Sampling Regularity & Maximum Temporal Gaps', fontsize=13, fontweight='bold')
    ax.set_xlabel('Trip Index (1-80)')
    ax.set_ylabel('Max Time Gap (Seconds, Log Scale)')
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_time_gaps.png', dpi=200)
    plt.close()
    
    # 4. Road Network Size Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.hist(train_df['n_nodes'] / 1000.0, bins=20, color='#1abc9c', alpha=0.75, edgecolor='white')
    ax1.set_title('Road Network Graph Nodes', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Thousands of Nodes')
    ax1.set_ylabel('Trip Frequency')
    
    ax2.hist(train_df['n_arcs'] / 1000.0, bins=20, color='#f39c12', alpha=0.75, edgecolor='white')
    ax2.set_title('Road Network Graph Arcs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Thousands of Arcs')
    ax2.set_ylabel('Trip Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eda_network_size.png', dpi=200)
    plt.close()
    print("EDA plots successfully generated.")


def plot_model_comparison(results_csv_path, output_dir='visualizations'):
    """Generates all comparative charts across the 6 model families."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv_path)
    models = df['model'].values
    x = np.arange(len(models))
    w = 0.35
    
    # 1. Macro F1: CV vs Test
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, df['cv_f1_macro'], w, label='5-Fold CV F1 (Train)', color='#2980b9', alpha=0.85)
    ax.bar(x + w/2, df['test_f1_macro'], w, label='Macro F1 (Student Test)', color='#e67e22', alpha=0.85)
    ax.set_title('Cross-Validation vs Student-Test Macro F1-Score', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=11)
    ax.set_ylabel('Macro F1-Score')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_f1_comparison.png', dpi=200)
    plt.close()
    
    # 2. Top-1 Accuracy & Route Continuity
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, df['top1_accuracy'], w, label='Top-1 Accuracy', color='#27ae60', alpha=0.85)
    ax.bar(x + w/2, df['route_continuity'], w, label='Route Continuity', color='#8e44ad', alpha=0.85)
    ax.set_title('Observation-Level Top-1 Accuracy & Road Network Continuity', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=11)
    ax.set_ylabel('Metric Score [0.0 - 1.0]')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_top1_continuity.png', dpi=200)
    plt.close()
    
    # 3. Log Loss
    valid_ll = ~np.isnan(df['test_log_loss'])
    if np.any(valid_ll):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(models[valid_ll], df['test_log_loss'][valid_ll], color='#d35400', alpha=0.85, width=0.45)
        ax.set_title('Multi-Model Test Log Loss (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Binary Cross-Entropy (Log Loss)')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_log_loss.png', dpi=200)
        plt.close()
    
    # 4. Overfitting Analysis: Train vs Test Accuracy
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, df['train_accuracy'], w, label='Training Accuracy', color='#34495e', alpha=0.85)
    ax.bar(x + w/2, df['test_accuracy'], w, label='Test Accuracy', color='#16a085', alpha=0.85)
    ax.set_title('Generalisation & Overfitting Analysis: Train vs Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=11)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_overfitting_analysis.png', dpi=200)
    plt.close()
    
    # 5. Computational Efficiency (Train Time & Inference Time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.barh(models, df['train_time'], color='#2c3e50', alpha=0.85)
    ax1.set_title('Model Training Time (Seconds)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Seconds')
    
    ax2.barh(models, df['inference_time'] * 1000.0, color='#c0392b', alpha=0.85)
    ax2.set_title('Inference Latency on Test Partition (Milliseconds)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Milliseconds')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_computational_efficiency.png', dpi=200)
    plt.close()

    # 6. Confusion Matrix Heatmaps (All 6 Models)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, row in df.iterrows():
        m_name = row['model']
        cm_raw = eval(str(row['confusion_matrix']))
        cm = np.array(cm_raw)
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        ax.set_title(f"{m_name}\n(Macro F1: {row['test_f1_macro']:.4f})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.suptitle('Confusion Matrices Across All 6 Models (Student-Test Partition)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_confusion_matrices.png', dpi=200)
    plt.close()
    print("Model comparison charts & confusion matrices generated.")


def plot_bayesian_optimization_charts(trials_csv_path, comparison_json_path, output_dir='visualizations'):
    """Generates Bayesian optimization convergence and tuned-vs-untuned comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convergence curve
    if os.path.exists(trials_csv_path):
        trials = pd.read_csv(trials_csv_path)
        if 'mean_test_score' in trials.columns:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            scores = trials['mean_test_score'].values
            best_cum = np.maximum.accumulate(scores)
            ax.plot(range(1, len(scores) + 1), scores, 'o--', color='#bdc3c7', alpha=0.7, label='Trial Score')
            ax.plot(range(1, len(best_cum) + 1), best_cum, 'r-', linewidth=2.5, label='Global Optimum Progression')
            ax.set_title('Bayesian Hyperparameter Optimisation: Convergence Trace', fontsize=13, fontweight='bold')
            ax.set_xlabel('Bayesian Search Iteration')
            ax.set_ylabel('GroupKFold Macro F1')
            ax.legend(frameon=True)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/bayesian_convergence.png', dpi=200)
            plt.close()
    
    # Tuned vs Untuned comparison
    if os.path.exists(comparison_json_path):
        with open(comparison_json_path) as f:
            data = json.load(f)
        
        metrics = ['cv_f1_macro', 'test_f1_macro', 'top1_accuracy', 'route_continuity']
        metric_labels = ['CV Macro F1', 'Test Macro F1', 'Top-1 Accuracy', 'Route Continuity']
        
        untuned_scores = [data['Untuned LightGBM'][m] for m in metrics]
        tuned_scores = [data['Tuned LightGBM'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, untuned_scores, w, label='Untuned Baseline (Default)', color='#95a5a6', alpha=0.85)
        ax.bar(x + w/2, tuned_scores, w, label='Bayesian Tuned LightGBM', color='#27ae60', alpha=0.9)
        ax.set_title('Performance Uplift: Untuned vs Bayesian Tuned LightGBM', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel('Score [0.0 - 1.0]')
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bayesian_tuned_vs_untuned.png', dpi=200)
        plt.close()
        print("Bayesian optimization figures generated.")


def plot_hmm_reconstruction_charts(hmm_json_path, output_dir='visualizations'):
    """Generates charts illustrating the quantitative value added by HMM/Viterbi."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(hmm_json_path):
        return
    
    with open(hmm_json_path) as f:
        data = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    methods = ['Classifier-Only\n(Greedy Argmax)', 'Classifier +\nHMM / Viterbi']
    accs = [
        data['classifier_only']['avg_pointwise_accuracy'] * 100.0,
        data['hmm_viterbi']['avg_pointwise_accuracy'] * 100.0
    ]
    conts = [
        data['classifier_only']['avg_route_continuity'] * 100.0,
        data['hmm_viterbi']['avg_route_continuity'] * 100.0
    ]
    
    colors = ['#e74c3c', '#2ecc71']
    
    b1 = ax1.bar(methods, accs, color=colors, width=0.45, alpha=0.9)
    ax1.set_title('Pointwise Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 105)
    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 1.5, f"{h:.2f}%", ha='center', fontweight='bold')
    
    b2 = ax2.bar(methods, conts, color=colors, width=0.45, alpha=0.9)
    ax2.set_title('Route Network Continuity (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Continuity (%)')
    ax2.set_ylim(0, 105)
    for rect in b2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h + 1.5, f"{h:.2f}%", ha='center', fontweight='bold')
    
    plt.suptitle('Sequential Trajectory Reconstruction: Classifier vs HMM/Viterbi', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hmm_reconstruction_impact.png', dpi=200)
    plt.close()
    print("HMM reconstruction charts generated.")


def create_folium_trajectory_map(trip_dir, trip_id, predictions_arcs=None,
                                 output_html='visualizations/maps/trip_map.html'):
    """
    Creates rich interactive Folium map displaying:
    1. Raw noisy GPS observations (blue circle markers with timestamps)
    2. Road network segment geometry (gray background lines)
    3. Ground truth reference route (dashed green line)
    4. Reconstructed / Predicted route (solid red line)
    Includes layer control, scale, and tooltips.
    """
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    track_df, nodes_df, arcs_df, route = load_trip(trip_dir, trip_id)
    _, arc_info = build_adjacency(arcs_df, nodes_df)
    
    center_lat = float(track_df['lat'].mean())
    center_lon = float(track_df['lon'].mean())
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB positron')
    
    # 1. Road Network Layer (Background)
    road_layer = folium.FeatureGroup(name='Road Network Geometry', show=False)
    # Sample bounding box arcs
    min_lon, max_lon = track_df['lon'].min() - 0.01, track_df['lon'].max() + 0.01
    min_lat, max_lat = track_df['lat'].min() - 0.01, track_df['lat'].max() + 0.01
    
    for aid, info in arc_info.items():
        if (min_lon <= info['from_lon'] <= max_lon and min_lat <= info['from_lat'] <= max_lat):
            folium.PolyLine(
                locations=[[info['from_lat'], info['from_lon']], [info['to_lat'], info['to_lon']]],
                color='#7f8c8d', weight=1.2, opacity=0.4
            ).add_to(road_layer)
    road_layer.add_to(m)
    
    # 2. Ground Truth Route Layer
    if route is not None:
        gt_layer = folium.FeatureGroup(name='Ground Truth Route (Reference)', show=True)
        gt_coords = []
        for aid in route:
            if aid in arc_info:
                info = arc_info[aid]
                gt_coords.extend([[info['from_lat'], info['from_lon']], [info['to_lat'], info['to_lon']]])
        if gt_coords:
            folium.PolyLine(
                locations=gt_coords, color='#27ae60', weight=4.5, opacity=0.85,
                dash_array='8, 8', tooltip='Ground Truth Trajectory'
            ).add_to(gt_layer)
        gt_layer.add_to(m)
    
    # 3. Predicted Route Layer
    if predictions_arcs is not None:
        pred_layer = folium.FeatureGroup(name='PathSnap Predicted Route (HMM Decoded)', show=True)
        pred_coords = []
        for aid in predictions_arcs:
            if aid in arc_info:
                info = arc_info[aid]
                pred_coords.extend([[info['from_lat'], info['from_lon']], [info['to_lat'], info['to_lon']]])
        if pred_coords:
            folium.PolyLine(
                locations=pred_coords, color='#e74c3c', weight=3.0, opacity=0.85,
                tooltip='Reconstructed Trajectory'
            ).add_to(pred_layer)
        pred_layer.add_to(m)
    
    # 4. Raw GPS Observations Layer
    gps_layer = folium.FeatureGroup(name='Raw GPS Observations', show=True)
    # Downsample points slightly for smooth rendering if track is very large
    step = max(1, len(track_df) // 500)
    for i in range(0, len(track_df), step):
        row = track_df.iloc[i]
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3.5,
            color='#2980b9',
            fill=True,
            fill_color='#3498db',
            fill_opacity=0.75,
            popup=f"Obs {int(row['obs_idx'])} | T: {row['timestamp']:.0f}s"
        ).add_to(gps_layer)
    gps_layer.add_to(m)
    
    # Start and End markers
    folium.Marker(
        [track_df.iloc[0]['lat'], track_df.iloc[0]['lon']],
        popup="Trajectory Start", icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    folium.Marker(
        [track_df.iloc[-1]['lat'], track_df.iloc[-1]['lon']],
        popup="Trajectory End", icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    m.save(output_html)
    print(f"Interactive Folium map saved to: {output_html}")
    return output_html


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "=" * 80)
    print("PATHSNAP: GEOSPATIAL VISUALIZATIONS & DIAGNOSTIC PLOTS GENERATOR")
    print("=" * 80)
    
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/maps', exist_ok=True)
    
    print("\n1. Generating EDA distribution plots...")
    if os.path.exists('results/train_summary.csv') and os.path.exists('results/test_summary.csv'):
        plot_eda_summary('results/train_summary.csv', 'results/test_summary.csv')
    else:
        print("  Summary files not found, skipping EDA plots.")
        
    print("\n2. Generating Model Comparison benchmark charts...")
    if os.path.exists('results/model_comparison.csv'):
        plot_model_comparison('results/model_comparison.csv')
    else:
        print("  results/model_comparison.csv not found, skipping model charts.")
        
    print("\n3. Generating Bayesian Optimization convergence plots...")
    if os.path.exists('results/bayesian_opt_trials.csv') and os.path.exists('results/tuned_vs_untuned.json'):
        plot_bayesian_optimization_charts('results/bayesian_opt_trials.csv', 'results/tuned_vs_untuned.json')
    else:
        print("  Bayesian logs not found, skipping Bayesian plots.")
        
    print("\n4. Generating HMM sequential impact charts...")
    if os.path.exists('results/hmm_comparison.json'):
        plot_hmm_reconstruction_charts('results/hmm_comparison.json')
    else:
        print("  HMM results not found, skipping HMM charts.")
        
    print("\n5. Generating Interactive Folium HTML Maps...")
    create_folium_trajectory_map(
        'Student folder/student_test', '00000000',
        output_html='visualizations/maps/test_trip_00000000_map.html'
    )
    create_folium_trajectory_map(
        'Student folder/train', '00000001',
        output_html='visualizations/maps/train_trip_00000001_map.html'
    )
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS AND MAPS GENERATED SUCCESSFULLY!")
    print("=" * 80)

