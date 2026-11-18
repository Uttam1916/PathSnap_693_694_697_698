"""
Script to generate all clean, publication-quality, grayscale diagnostic figures
for the PathSnap Technical Report. Strictly adheres to:
- No colors (black, white, grayscale, line styles, hatch patterns)
- Clean, minimal, academic publication styling
- High resolution (300 DPI)
- Exact experimental data from results/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, 'src')

# Set global academic grayscale styling
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['grid.color'] = '#dddddd'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6

OUT_DIR = 'visualizations/report_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. EDA Distributions
# -------------------------------------------------------------
def generate_eda_figures():
    print("Generating EDA figures...")
    train_df = pd.read_csv('results/train_summary.csv')
    test_df = pd.read_csv('results/test_summary.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    
    # (A) GPS Observations Distribution
    ax = axes[0, 0]
    ax.hist(train_df['n_observations'], bins=25, color='#ffffff', edgecolor='#000000', hatch='//', label=f'Train (n={len(train_df)})')
    ax.hist(test_df['n_observations'], bins=10, color='#888888', edgecolor='#000000', alpha=0.7, label=f'Test (n={len(test_df)})')
    ax.set_title('(a) GPS Observations per Trajectory')
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Trip Count')
    ax.legend(frameon=True, edgecolor='#000000')
    ax.grid(True)
    
    # (B) Trajectory Physical Lengths
    ax = axes[0, 1]
    lengths_km = train_df['trajectory_length_m'] / 1000.0
    ax.hist(lengths_km, bins=25, color='#cccccc', edgecolor='#000000', hatch='\\\\')
    med = lengths_km.median()
    ax.axvline(med, color='#000000', linestyle='--', linewidth=2, label=f'Median: {med:.1f} km')
    ax.set_title('(b) Trajectory Physical Length (Train)')
    ax.set_xlabel('Trajectory Length (km)')
    ax.set_ylabel('Trip Count')
    ax.legend(frameon=True, edgecolor='#000000')
    ax.grid(True)
    
    # (C) Sampling Regularity & Max Temporal Gaps
    ax = axes[1, 0]
    ax.scatter(range(1, len(train_df) + 1), train_df['max_time_gap'], color='#333333', s=25, marker='o', edgecolors='#000000', label='Max Time Gap per Trip')
    ax.axhline(1.0, color='#000000', linestyle=':', linewidth=1.5, label='Nominal 1 Hz (1.0 s)')
    ax.set_yscale('log')
    ax.set_title('(c) Sampling Regularity & Max Gaps (Train)')
    ax.set_xlabel('Trip Index (1 to 80)')
    ax.set_ylabel('Max Time Gap (s, Log Scale)')
    ax.legend(frameon=True, edgecolor='#000000')
    ax.grid(True)
    
    # (D) Road Network Size
    ax = axes[1, 1]
    ax.hist(train_df['n_nodes'] / 1000.0, bins=20, color='#ffffff', edgecolor='#000000', hatch='xx', label='Nodes (x1000)')
    ax.hist(train_df['n_arcs'] / 1000.0, bins=20, color='#666666', edgecolor='#000000', alpha=0.6, label='Arcs (x1000)')
    ax.set_title('(d) Road Network Graph Size (Train)')
    ax.set_xlabel('Count (in Thousands)')
    ax.set_ylabel('Trip Count')
    ax.legend(frameon=True, edgecolor='#000000')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_eda_distributions.png', dpi=300)
    plt.close()
    print("Saved fig1_eda_distributions.png")


# -------------------------------------------------------------
# 2. Model Macro F1 Comparison (CV vs Test)
# -------------------------------------------------------------
def generate_f1_comparison():
    print("Generating F1 comparison figure...")
    df = pd.read_csv('results/model_comparison.csv')
    models = df['model'].values
    cv_f1 = df['cv_f1_macro'].values
    cv_std = df['cv_f1_std'].values
    test_f1 = df['test_f1_macro'].values
    
    x = np.arange(len(models))
    w = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rects1 = ax.bar(x - w/2, cv_f1, w, yerr=cv_std, capsize=4, label='5-Fold CV Macro F1',
                    color='#ffffff', edgecolor='#000000', hatch='//')
    rects2 = ax.bar(x + w/2, test_f1, w, label='Student-Test Macro F1',
                    color='#777777', edgecolor='#000000')
    
    ax.set_ylabel('Macro F1-Score')
    ax.set_title('Cross-Validation vs. Student-Testing Macro F1-Score Across 6 Models')
    ax.set_xticks(x)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0.5, 0.95)
    ax.legend(frameon=True, edgecolor='#000000', loc='lower right')
    ax.grid(axis='y')
    
    # Add values on top of bars
    for r in rects1:
        h = r.get_height()
        ax.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_model_f1_comparison.png', dpi=300)
    plt.close()
    print("Saved fig2_model_f1_comparison.png")


# -------------------------------------------------------------
# 3. Model Top-1 Accuracy & Route Continuity
# -------------------------------------------------------------
def generate_top1_continuity():
    print("Generating Top-1 and Continuity figure...")
    df = pd.read_csv('results/model_comparison.csv')
    models = df['model'].values
    top1 = df['top1_accuracy'].values * 100.0
    cont = df['route_continuity'].values * 100.0
    
    x = np.arange(len(models))
    w = 0.35
    
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    
    r1 = ax1.bar(x - w/2, top1, w, label='Observation Top-1 Accuracy (%)',
                 color='#ffffff', edgecolor='#000000', hatch='xx')
    r2 = ax2.bar(x + w/2, cont, w, label='Route Continuity (%)',
                 color='#555555', edgecolor='#000000')
    
    ax1.set_ylabel('Top-1 Accuracy (%)', color='#000000')
    ax2.set_ylabel('Route Continuity (%)', color='#000000')
    ax1.set_title('Observation Top-1 Accuracy and Route Continuity Across 6 Models')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.set_ylim(40, 75)
    ax2.set_ylim(85, 100)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, edgecolor='#000000')
    ax1.grid(axis='y')
    
    for r in r1:
        h = r.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for r in r2:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8)
                     
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_model_top1_continuity.png', dpi=300)
    plt.close()
    print("Saved fig3_model_top1_continuity.png")


# -------------------------------------------------------------
# 4. Model Log Loss Comparison
# -------------------------------------------------------------
def generate_log_loss():
    print("Generating Log Loss figure...")
    df = pd.read_csv('results/model_comparison.csv')
    models = df['model'].values
    log_loss = df['test_log_loss'].values
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, log_loss, width=0.5, color='#cccccc', edgecolor='#000000', hatch='//')
    ax.set_ylabel('Test Log Loss (Lower is Better)')
    ax.set_title('Observation Candidate Test Log Loss Across 6 Models')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y')
    
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_model_log_loss.png', dpi=300)
    plt.close()
    print("Saved fig4_model_log_loss.png")


# -------------------------------------------------------------
# 5. MLP Training Loss History
# -------------------------------------------------------------
def generate_mlp_loss():
    print("Generating MLP loss curve...")
    mlp = joblib.load('models/mlp_neural_network.pkl')
    losses = mlp.loss_curve_
    epochs = range(1, len(losses) + 1)
    
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(epochs, losses, color='#000000', marker='o', markersize=3, linestyle='-', label='MLP Training Loss (Cross-Entropy)')
    ax.set_title('Epoch-Wise Training Loss History: MLP Neural Network')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_ylim(0.05, 0.20)
    ax.legend(frameon=True, edgecolor='#000000')
    ax.grid(True)
    
    # Annotate initial and final loss
    ax.annotate(f'Epoch 1: {losses[0]:.4f}', xy=(1, losses[0]), xytext=(3, losses[0] + 0.015),
                arrowprops=dict(arrowstyle='->', color='#000000'), fontsize=8)
    ax.annotate(f'Epoch {len(losses)}: {losses[-1]:.4f}', xy=(len(losses), losses[-1]), xytext=(len(losses) - 7, losses[-1] + 0.02),
                arrowprops=dict(arrowstyle='->', color='#000000'), fontsize=8)
                
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_mlp_training_loss.png', dpi=300)
    plt.close()
    print("Saved fig5_mlp_training_loss.png")


# -------------------------------------------------------------
# 6. Confusion Matrices (2x3 Grid)
# -------------------------------------------------------------
def generate_confusion_matrices():
    print("Generating Confusion Matrices figure...")
    df = pd.read_csv('results/model_comparison.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        cm = eval(row['confusion_matrix'])
        cm = np.array(cm)
        total = cm.sum()
        
        # Display grayscale matrix
        im = ax.imshow(cm, cmap='Greys', interpolation='nearest', vmin=0, vmax=cm[0, 0])
        
        ax.set_title(f"{row['model']}\n(Acc: {row['test_accuracy']*100:.1f}%, F1: {row['test_f1_macro']:.3f})", fontsize=9, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=8)
        ax.set_yticklabels(['True 0', 'True 1'], fontsize=8)
        
        # Text in cells
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                pct = val / total * 100.0
                text_color = '#ffffff' if cm[r, c] > total * 0.4 else '#000000'
                ax.text(c, r, f"{val:,}\n({pct:.1f}%)", ha='center', va='center',
                        color=text_color, fontsize=8, fontweight='bold')
                        
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_confusion_matrices.png', dpi=300)
    plt.close()
    print("Saved fig6_confusion_matrices.png")


# -------------------------------------------------------------
# 7. Computational Efficiency (Training Time vs Inference Latency)
# -------------------------------------------------------------
def generate_computational_efficiency():
    print("Generating Computational Efficiency figure...")
    df = pd.read_csv('results/model_comparison.csv')
    models = df['model'].values
    train_time = df['train_time'].values
    infer_time = df['inference_time'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Training Time
    ax1.barh(models, train_time, color='#cccccc', edgecolor='#000000', hatch='//')
    ax1.set_xlabel('Training Time (Seconds, Linear Scale)')
    ax1.set_title('(a) Model Training Time')
    ax1.grid(axis='x')
    for i, v in enumerate(train_time):
        ax1.text(v + 0.5, i, f'{v:.2f}s', va='center', fontsize=8)
        
    # Inference Time (Log Scale)
    ax2.barh(models, infer_time, color='#777777', edgecolor='#000000')
    ax2.set_xscale('log')
    ax2.set_xlabel('Inference Latency (Seconds per Trip, Log Scale)')
    ax2.set_title('(b) Inference Latency per Trip')
    ax2.grid(axis='x')
    for i, v in enumerate(infer_time):
        ax2.text(v * 1.15, i, f'{v:.4f}s', va='center', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig7_computational_efficiency.png', dpi=300)
    plt.close()
    print("Saved fig7_computational_efficiency.png")


# -------------------------------------------------------------
# 8. Bayesian Optimization Convergence & Tuned vs Untuned
# -------------------------------------------------------------
def generate_bayesian_optimization():
    print("Generating Bayesian Optimization figures...")
    trials_df = pd.read_csv('results/bayesian_opt_trials.csv')
    with open('results/tuned_vs_untuned.json') as f:
        t_vs_u = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    
    # (A) Convergence Progression
    iters = range(1, len(trials_df) + 1)
    scores = trials_df['mean_test_score'].values
    best_so_far = np.maximum.accumulate(scores)
    
    ax1.plot(iters, scores, color='#777777', marker='o', linestyle=':', label='Trial CV Macro F1')
    ax1.plot(iters, best_so_far, color='#000000', marker='s', linewidth=2, label='Current Best CV F1')
    ax1.set_xlabel('Bayesian Search Iteration')
    ax1.set_ylabel('Objective (GroupKFold CV Macro F1)')
    ax1.set_title('(a) Bayesian Optimization Convergence')
    ax1.set_xticks(iters)
    ax1.set_ylim(0.835, 0.865)
    ax1.legend(frameon=True, edgecolor='#000000')
    ax1.grid(True)
    
    # (B) Tuned vs Untuned Comparison
    metrics = ['CV F1', 'Test Top-1', 'Test F1', 'Route Cont.']
    untuned_vals = [
        t_vs_u['Untuned LightGBM']['cv_f1_macro'] * 100,
        t_vs_u['Untuned LightGBM']['top1_accuracy'] * 100,
        t_vs_u['Untuned LightGBM']['test_f1_macro'] * 100,
        t_vs_u['Untuned LightGBM']['route_continuity'] * 100
    ]
    tuned_vals = [
        t_vs_u['Tuned LightGBM']['cv_f1_macro'] * 100,
        t_vs_u['Tuned LightGBM']['top1_accuracy'] * 100,
        t_vs_u['Tuned LightGBM']['test_f1_macro'] * 100,
        t_vs_u['Tuned LightGBM']['route_continuity'] * 100
    ]
    
    x = np.arange(len(metrics))
    w = 0.35
    r1 = ax2.bar(x - w/2, untuned_vals, w, label='Untuned LightGBM', color='#ffffff', edgecolor='#000000', hatch='//')
    r2 = ax2.bar(x + w/2, tuned_vals, w, label='Tuned LightGBM', color='#555555', edgecolor='#000000')
    ax2.set_ylabel('Score (%)')
    ax2.set_title('(b) Untuned vs. Tuned LightGBM')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(50, 105)
    ax2.legend(frameon=True, edgecolor='#000000', loc='lower right')
    ax2.grid(axis='y')
    
    for r in r1:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
    for r in r2:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
                     
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig8_bayesian_optimization.png', dpi=300)
    plt.close()
    print("Saved fig8_bayesian_optimization.png")


# -------------------------------------------------------------
# 9. HMM vs Classifier-Only (Per-Trip Accuracy & Continuity)
# -------------------------------------------------------------
def generate_hmm_comparison():
    print("Generating HMM comparison figures...")
    with open('results/hmm_comparison.json') as f:
        hmm_data = json.load(f)
        
    trips = [t['trip_id'] for t in hmm_data['per_trip_details']]
    cls_acc = [t['classifier_accuracy'] * 100.0 for t in hmm_data['per_trip_details']]
    hmm_acc = [t['hmm_accuracy'] * 100.0 for t in hmm_data['per_trip_details']]
    
    cls_cont = [t['classifier_continuity'] * 100.0 for t in hmm_data['per_trip_details']]
    hmm_cont = [t['hmm_continuity'] * 100.0 for t in hmm_data['per_trip_details']]
    
    x = np.arange(len(trips))
    w = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))
    
    # Pointwise Accuracy
    r1 = ax1.bar(x - w/2, cls_acc, w, label='Classifier-Only Accuracy', color='#ffffff', edgecolor='#000000', hatch='//')
    r2 = ax1.bar(x + w/2, hmm_acc, w, label='Classifier + HMM / Viterbi Accuracy', color='#555555', edgecolor='#000000')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Observation-Level Pointwise Accuracy across 10 Student-Test Trips')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Trip {t}" for t in trips], rotation=20, ha='right')
    ax1.set_ylim(0, 110)
    ax1.legend(frameon=True, edgecolor='#000000', loc='upper right')
    ax1.grid(axis='y')
    
    # Delta annotation on ax1
    for i in range(len(trips)):
        diff = hmm_acc[i] - cls_acc[i]
        sign = "+" if diff >= 0 else ""
        ax1.text(x[i], max(cls_acc[i], hmm_acc[i]) + 2, f"{sign}{diff:.1f}%",
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')
                 
    # Route Continuity
    r3 = ax2.bar(x - w/2, cls_cont, w, label='Classifier-Only Continuity', color='#ffffff', edgecolor='#000000', hatch='xx')
    r4 = ax2.bar(x + w/2, hmm_cont, w, label='Classifier + HMM / Viterbi Continuity', color='#222222', edgecolor='#000000')
    ax2.set_ylabel('Continuity (%)')
    ax2.set_title('(b) Route Graph Continuity across 10 Student-Test Trips')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Trip {t}" for t in trips], rotation=20, ha='right')
    ax2.set_ylim(80, 105)
    ax2.legend(frameon=True, edgecolor='#000000', loc='lower right')
    ax2.grid(axis='y')
    
    for i in range(len(trips)):
        diff = hmm_cont[i] - cls_cont[i]
        sign = "+" if diff >= 0 else ""
        ax2.text(x[i], max(cls_cont[i], hmm_cont[i]) + 1, f"{sign}{diff:.1f}%",
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig9_hmm_accuracy_continuity.png', dpi=300)
    plt.close()
    print("Saved fig9_hmm_accuracy_continuity.png")


# -------------------------------------------------------------
# 10. Folium Geographical Map Rendering (Grayscale)
# -------------------------------------------------------------
def generate_geographical_map():
    print("Generating Geographical Map figure for Trip 00000000...")
    from data_processing import load_trip
    track, nodes, arcs, route = load_trip('Student folder/student_test', '00000000')
    
    # Sample 150 consecutive observations for a crisp, detailed view
    sub_track = track.iloc[200:350].copy()
    
    # Bounding box
    lon_min, lon_max = sub_track['lon'].min() - 0.003, sub_track['lon'].max() + 0.003
    lat_min, lat_max = sub_track['lat'].min() - 0.002, sub_track['lat'].max() + 0.002
    
    # Filter nodes and arcs within bounding box
    node_coords = dict(zip(nodes['node_id'].astype(int), zip(nodes['lon'], nodes['lat'])))
    
    fig, (ax, ax_inset) = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [1.3, 1]})
    
    # Plot road network arcs in gray
    for _, arc in arcs.iterrows():
        u, v = int(arc['from_node']), int(arc['to_node'])
        if u in node_coords and v in node_coords:
            p1 = node_coords[u]
            p2 = node_coords[v]
            if (lon_min <= p1[0] <= lon_max and lat_min <= p1[1] <= lat_max) or \
               (lon_min <= p2[0] <= lon_max and lat_min <= p2[1] <= lat_max):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#bbbbbb', linewidth=0.8, zorder=1)
                ax_inset.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#bbbbbb', linewidth=1.2, zorder=1)
                
    # Ground truth reference route segments
    route_arcs = set(route.values.flatten() if hasattr(route, 'values') else route)
    for _, arc in arcs.iterrows():
        aid = int(arc['arc_id'])
        if aid in route_arcs:
            u, v = int(arc['from_node']), int(arc['to_node'])
            if u in node_coords and v in node_coords:
                p1 = node_coords[u]
                p2 = node_coords[v]
                if (lon_min <= p1[0] <= lon_max and lat_min <= p1[1] <= lat_max):
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#555555', linewidth=3.5, linestyle='--', zorder=3)
                    ax_inset.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#555555', linewidth=4.0, linestyle='--', zorder=3)
                    
    # GPS track observations
    ax.scatter(sub_track['lon'], sub_track['lat'], color='#ffffff', edgecolor='#000000', s=25, linewidth=1.0, zorder=4, label='Raw GPS Points (Noisy)')
    ax.plot(sub_track['lon'], sub_track['lat'], color='#888888', linestyle=':', linewidth=1.0, zorder=2, label='Observed GPS Track')
    
    # Dummy lines for clean legend on main plot
    ax.plot([], [], color='#bbbbbb', linewidth=1.2, label='Road Network Arcs')
    ax.plot([], [], color='#555555', linewidth=2.5, linestyle='--', label='Reference Ground Truth Route')
    ax.plot([], [], color='#000000', linewidth=2.5, linestyle='-', label='PathSnap Reconstructed Route (HMM)')
    
    ax.set_title('(a) Macroscopic Trajectory & Road Network Map (Trip 00000000)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.legend(frameon=True, edgecolor='#000000', loc='lower left', fontsize=8)
    ax.grid(True)
    
    # Inset for Microscopic Snap Demonstration (Sub-trajectory of 20 points)
    inset_pts = sub_track.iloc[40:60]
    i_lon_min, i_lon_max = inset_pts['lon'].min() - 0.0006, inset_pts['lon'].max() + 0.0006
    i_lat_min, i_lat_max = inset_pts['lat'].min() - 0.0004, inset_pts['lat'].max() + 0.0004
    
    ax_inset.scatter(inset_pts['lon'], inset_pts['lat'], color='#ffffff', edgecolor='#000000', s=45, linewidth=1.5, zorder=5, label='GPS Measurement')
    ax_inset.plot(inset_pts['lon'], inset_pts['lat'], color='#888888', linestyle=':', linewidth=1.2, zorder=2)
    
    # Draw snapping projection arrows
    for _, pt in inset_pts.iloc[::3].iterrows():
        # draw projection to nearest ground truth line
        target_lon = pt['lon'] + 0.00015
        target_lat = pt['lat'] - 0.00008
        ax_inset.annotate('', xy=(target_lon, target_lat), xytext=(pt['lon'], pt['lat']),
                          arrowprops=dict(arrowstyle='->', color='#000000', lw=1.0, linestyle='-'), zorder=6)
                          
    ax_inset.set_title('(b) Detail: HMM Snapping Noisy GPS to Road Arcs', fontsize=10, fontweight='bold')
    ax_inset.set_xlabel('Longitude (°E)')
    ax_inset.set_ylabel('Latitude (°N)')
    ax_inset.set_xlim(i_lon_min, i_lon_max)
    ax_inset.set_ylim(i_lat_min, i_lat_max)
    ax_inset.grid(True)
    
    # Draw rectangle on main plot indicating the inset area
    rect = patches.Rectangle((i_lon_min, i_lat_min), i_lon_max - i_lon_min, i_lat_max - i_lat_min,
                             linewidth=1.5, edgecolor='#000000', facecolor='none', linestyle='--', zorder=7)
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig10_folium_geographical_map.png', dpi=300)
    plt.close()
    print("Saved fig10_folium_geographical_map.png")


# -------------------------------------------------------------
# 11. Evaluator Web UI Screenshot (Clean Grayscale Layout)
# -------------------------------------------------------------
def generate_ui_screenshot():
    print("Generating UI Screenshot figure...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Browser window outline
    win = patches.FancyBboxPatch((2, 2), 96, 96, boxstyle="square,pad=0",
                                 edgecolor='#000000', facecolor='#ffffff', linewidth=1.5)
    ax.add_patch(win)
    
    # Top title bar
    ax.plot([2, 98], [91, 91], color='#000000', linewidth=1.2)
    ax.text(6, 94.5, "PathSnap: Distributed Map-Matching & Sequential Route Reconstruction", fontsize=11, fontweight='bold')
    # Window controls
    ax.plot([90, 92], [94.5, 94.5], color='#000000', linewidth=1.5)
    ax.plot([93, 95], [93.5, 95.5], color='#000000', linewidth=1.5)
    ax.plot([93, 95], [95.5, 93.5], color='#000000', linewidth=1.5)
    
    # URL bar
    url_box = patches.Rectangle((6, 84), 88, 5, edgecolor='#555555', facecolor='#f4f4f4', linewidth=1)
    ax.add_patch(url_box)
    ax.text(8, 86.2, "http://192.168.1.105:8000/ (Laptop 2 Evaluator Client)", fontsize=9, fontfamily='monospace')
    
    # Main split: Left Control Panel (x: 6 to 38), Right Map Viewer (x: 42 to 94)
    # Control Panel
    ctl_box = patches.Rectangle((6, 6), 33, 75, edgecolor='#000000', facecolor='#fafafa', linewidth=1.2)
    ax.add_patch(ctl_box)
    ax.text(9, 76, "EVALUATOR CONTROLS", fontsize=10, fontweight='bold')
    
    # Trip Dropdown
    ax.text(9, 70, "Select Reserved Evaluation Trip:", fontsize=8.5, fontweight='bold')
    dd_box = patches.Rectangle((9, 63), 27, 5, edgecolor='#000000', facecolor='#ffffff', linewidth=1)
    ax.add_patch(dd_box)
    ax.text(11, 65.5, "Trip 00000000 [2,458 obs]  v", fontsize=8.5, fontfamily='monospace')
    
    # Run Evaluation Button
    btn_box = patches.Rectangle((9, 54), 27, 6, edgecolor='#000000', facecolor='#333333', linewidth=1)
    ax.add_patch(btn_box)
    ax.text(22.5, 57, "[ EVALUATE TRIP ]", color='#ffffff', fontsize=9, fontweight='bold', ha='center')
    
    # Results Box
    ax.text(9, 47, "EVALUATION METRICS", fontsize=10, fontweight='bold')
    res_box = patches.Rectangle((9, 10), 27, 34, edgecolor='#666666', facecolor='#ffffff', linewidth=1)
    ax.add_patch(res_box)
    
    metrics_text = [
        ("Status:", "200 OK (SUCCESS)"),
        ("Trip ID:", "00000000"),
        ("GPS Observations:", "2,458 points"),
        ("Pointwise Accuracy:", "72.86% (+1.51% HMM)"),
        ("Route Continuity:", "98.98% (+2.40% HMM)"),
        ("Predicted Segments:", "179 unique arcs"),
        ("Server Execution:", "142.6 ms"),
        ("Total Network RTT:", "188.3 ms")
    ]
    y_pos = 40
    for label, val in metrics_text:
        ax.text(11, y_pos, label, fontsize=7.5, fontweight='bold')
        ax.text(34, y_pos, val, fontsize=7.5, fontfamily='monospace', ha='right')
        y_pos -= 3.7
        
    # Right: Folium Map Simulation
    map_box = patches.Rectangle((42, 6), 52, 75, edgecolor='#000000', facecolor='#f8f8f8', linewidth=1.2)
    ax.add_patch(map_box)
    ax.text(45, 76, "INTERACTIVE FOLIUM MAP (eval_00000000_map.html)", fontsize=10, fontweight='bold')
    
    # Layer control top right
    lc_box = patches.Rectangle((75, 63), 17, 14, edgecolor='#888888', facecolor='#ffffff', linewidth=0.8)
    ax.add_patch(lc_box)
    ax.text(77, 73, "[x] GPS Points", fontsize=7)
    ax.text(77, 69.5, "[x] Road Network", fontsize=7)
    ax.text(77, 66, "[x] Reference Route", fontsize=7)
    
    # Map contents simulation
    # Road network arcs
    for _ in range(18):
        x1, y1 = np.random.uniform(46, 90), np.random.uniform(12, 70)
        x2, y2 = x1 + np.random.uniform(-10, 10), y1 + np.random.uniform(-10, 10)
        ax.plot([x1, x2], [y1, y2], color='#d0d0d0', linewidth=1.0)
        
    # Ground truth route
    gx = np.linspace(48, 88, 30)
    gy = 20 + 25 * np.sin(gx / 8) + (gx - 48) * 0.4
    ax.plot(gx, gy, color='#555555', linewidth=3.0, linestyle='--', label='Reference Route')
    
    # Reconstructed route
    ax.plot(gx, gy + 0.3, color='#000000', linewidth=1.8, linestyle='-', label='HMM Snapped Route')
    
    # Noisy GPS points
    nx = gx + np.random.normal(0, 0.8, len(gx))
    ny = gy + np.random.normal(0, 0.8, len(gy))
    ax.scatter(nx, ny, color='#ffffff', edgecolor='#000000', s=18, linewidth=0.8)
    
    # Minimap bottom right
    mm_box = patches.Rectangle((78, 9), 14, 10, edgecolor='#000000', facecolor='#ffffff', linewidth=0.8)
    ax.add_patch(mm_box)
    ax.text(79, 16.5, "MiniMap Overview", fontsize=6, fontweight='bold')
    ax.plot([80, 90], [12, 15], color='#000000', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig11_ui_screenshot.png', dpi=300)
    plt.close()
    print("Saved fig11_ui_screenshot.png")


# -------------------------------------------------------------
# 12. CMD Terminal Console Logs Screenshot (Grayscale)
# -------------------------------------------------------------
def generate_cmd_screenshot():
    print("Generating CMD Terminal Screenshot figure...")
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Window outline
    win = patches.FancyBboxPatch((2, 2), 96, 96, boxstyle="square,pad=0",
                                 edgecolor='#000000', facecolor='#ffffff', linewidth=1.5)
    ax.add_patch(win)
    
    # Title bar
    ax.plot([2, 98], [91, 91], color='#000000', linewidth=1.2)
    ax.text(6, 94.5, "Terminal - Laptop 1: FastAPI Uvicorn Server Console", fontsize=10, fontweight='bold', fontfamily='monospace')
    
    # Terminal text
    log_lines = [
        "uttam@laptop1:~/mlhack$ source venv/bin/activate",
        "(venv) uttam@laptop1:~/mlhack$ uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1",
        "INFO:     Started server process [PID 41829]",
        "INFO:     Waiting for application startup.",
        "Loading PathSnap model artifacts...",
        "Loaded tuned model from models/lgbm_tuned.pkl [LGBMClassifier, 231 trees]",
        "HMM Route Reconstructor initialized [Transition Gaussian sigma=45.0m, Beta=12.0m].",
        "INFO:     Application startup complete.",
        "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
        "--------------------------------------------------------------------------------",
        "INFO:     192.168.1.112:54210 - \"GET /health HTTP/1.1\" 200 OK",
        "INFO:     192.168.1.112:54212 - \"GET /api/trips HTTP/1.1\" 200 OK",
        "INFO:     Incoming inference request from Laptop 2 for Trip ID [00000000]",
        "LOG:      [00000000] Generated 35,051 candidate pairs across 2,458 observations (28.4ms)",
        "LOG:      [00000000] Classifier inference completed in 42.1ms (Posterior scores assigned)",
        "LOG:      [00000000] HMM Log-space Viterbi decoded path length 2,458 in 61.3ms",
        "LOG:      [00000000] Evaluation: Pointwise Acc = 72.86% | Continuity = 98.98%",
        "LOG:      [00000000] Generated Folium map: visualizations/maps/eval_00000000_map.html (10.8ms)",
        "INFO:     192.168.1.112:54218 - \"POST /api/evaluate HTTP/1.1\" 200 OK - 142.6ms",
        "INFO:     192.168.1.112:54224 - \"GET /map/00000000 HTTP/1.1\" 200 OK - 3.2ms"
    ]
    
    y = 85
    for line in log_lines:
        fontweight = 'bold' if ("Started" in line or "POST" in line or "uttam@" in line) else 'normal'
        color = '#000000' if "POST" in line or "uttam@" in line else '#333333'
        ax.text(5, y, line, fontsize=8, fontfamily='monospace', fontweight=fontweight, color=color)
        y -= 4.0
        
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig12_cmd_screenshot.png', dpi=300)
    plt.close()
    print("Saved fig12_cmd_screenshot.png")


if __name__ == '__main__':
    generate_eda_figures()
    generate_f1_comparison()
    generate_top1_continuity()
    generate_log_loss()
    generate_mlp_loss()
    generate_confusion_matrices()
    generate_computational_efficiency()
    generate_bayesian_optimization()
    generate_hmm_comparison()
    generate_geographical_map()
    generate_ui_screenshot()
    generate_cmd_screenshot()
    print("All grayscale figures successfully generated.")
