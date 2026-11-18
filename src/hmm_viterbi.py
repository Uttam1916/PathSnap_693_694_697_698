"""
PathSnap - Hidden Markov Model (HMM) & Viterbi Sequential Route Reconstruction
Reconstructs globally optimal, physically coherent trajectories on the road network
by combining ML classifier emission probabilities with graph topological transition probabilities.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import haversine_vectorized, load_trip, build_adjacency
from model_training import FEATURE_COLS


class HMMRouteReconstructor:
    """
    Hidden Markov Model decoder with Viterbi dynamic programming.
    
    States: Candidate road segments (arcs) at each observation epoch t.
    Observations: Sequential GPS measurements (lon, lat, timestamp).
    
    Emission Model:
        P(z_t | s_t = a_i) = N(d(z_t, a_i); 0, sigma^2) * P_classifier(a_i | z_t)
    
    Transition Model:
        P(s_t = a_j | s_{t-1} = a_i) based on road connectivity and distance discrepancy:
        exp(-|d_gps - d_road| / beta) for connected arcs,
        disconnect_penalty for disconnected transitions.
    """
    def __init__(self, sigma_gps=45.0, beta_transition=25.0,
                 disconnect_penalty=1e-7, max_gap_seconds=30.0):
        self.sigma_gps = sigma_gps
        self.beta_transition = beta_transition
        self.disconnect_penalty = disconnect_penalty
        self.max_gap_seconds = max_gap_seconds
    
    def calculate_emission(self, distance, cls_prob):
        """Computes combined Gaussian GPS emission and classifier posterior."""
        # Gaussian spatial decay
        spatial_p = np.exp(-0.5 * (distance / self.sigma_gps) ** 2)
        # Combine with classifier probability
        cls_p = max(cls_prob, 1e-4) if cls_prob is not None else 0.5
        combined = spatial_p * (cls_p ** 1.2)
        return max(combined, 1e-30)
    
    def calculate_transition(self, a_prev, a_curr, arc_info, adj, gps_dist, dt):
        """Computes topological and distance-congruent transition probability."""
        if a_prev == a_curr:
            # Vehicle persists on the same road segment
            return 0.55
        
        if a_prev not in arc_info or a_curr not in arc_info:
            return self.disconnect_penalty
        
        info_prev = arc_info[a_prev]
        info_curr = arc_info[a_curr]
        
        to_prev = info_prev['to_node']
        from_curr = info_curr['from_node']
        
        # Check graph adjacency (direct connection or 1-2 hops)
        connected = False
        road_dist = 0.0
        
        if to_prev == from_curr:
            connected = True
            road_dist = info_curr['length']
        else:
            # 1-hop check
            nbrs_1 = {n for n, _ in adj.get(to_prev, [])}
            if from_curr in nbrs_1:
                connected = True
                road_dist = info_prev['length'] + info_curr['length']
            else:
                # 2-hop check
                for n1 in nbrs_1:
                    nbrs_2 = {n for n, _ in adj.get(n1, [])}
                    if from_curr in nbrs_2:
                        connected = True
                        road_dist = info_prev['length'] + 2.0 * info_curr['length']
                        break
        
        if connected:
            dist_discrepancy = abs(gps_dist - road_dist)
            trans_p = np.exp(-dist_discrepancy / self.beta_transition)
            return max(trans_p, 1e-15)
        else:
            # Disconnected segment
            if dt > self.max_gap_seconds:
                # Vehicle could have traversed intermediate unobserved links
                return self.disconnect_penalty * 100.0
            return self.disconnect_penalty
    
    def decode(self, obs_indices, candidates_by_obs, arc_info, adj, track_df, cls_probs_dict):
        """
        Executes log-space Viterbi forward pass and backward trace.
        """
        valid_obs = [o for o in obs_indices if len(candidates_by_obs.get(o, [])) > 0]
        if not valid_obs:
            return [0] * len(obs_indices)
        
        T = len(valid_obs)
        track_lons = track_df['lon'].values
        track_lats = track_df['lat'].values
        track_times = track_df['timestamp'].values
        
        # Viterbi tables
        # V[t][arc_id] = (best_log_prob, best_prev_arc_id)
        V = [{} for _ in range(T)]
        
        # Epoch 0
        o0 = valid_obs[0]
        for arc_id, dist in candidates_by_obs[o0]:
            cp = cls_probs_dict.get((o0, arc_id), 0.5)
            e_p = self.calculate_emission(dist, cp)
            V[0][arc_id] = (np.log(e_p), None)
        
        # Forward pass
        for t in range(1, T):
            curr_o = valid_obs[t]
            prev_o = valid_obs[t - 1]
            
            gps_d = haversine_vectorized(
                track_lons[prev_o], track_lats[prev_o],
                track_lons[curr_o], track_lats[curr_o]
            )
            dt = max(track_times[curr_o] - track_times[prev_o], 0.1)
            
            for curr_arc, curr_dist in candidates_by_obs[curr_o]:
                cp = cls_probs_dict.get((curr_o, curr_arc), 0.5)
                e_p = self.calculate_emission(curr_dist, cp)
                log_e = np.log(e_p)
                
                best_lp = -1e30
                best_prev = None
                
                for prev_arc, (prev_lp, _) in V[t - 1].items():
                    tr_p = self.calculate_transition(
                        prev_arc, curr_arc, arc_info, adj, gps_d, dt
                    )
                    total_lp = prev_lp + np.log(tr_p) + log_e
                    if total_lp > best_lp:
                        best_lp = total_lp
                        best_prev = prev_arc
                
                V[t][curr_arc] = (best_lp, best_prev)
        
        # Backtracking
        best_final_arc = max(V[T - 1].items(), key=lambda x: x[1][0])[0]
        decoded_path = [best_final_arc]
        
        for t in range(T - 1, 0, -1):
            prev_arc = V[t][decoded_path[-1]][1]
            if prev_arc is None:
                # Fallback to closest candidate
                prev_cands = candidates_by_obs.get(valid_obs[t - 1], [])
                prev_arc = prev_cands[0][0] if prev_cands else decoded_path[-1]
            decoded_path.append(prev_arc)
        
        decoded_path.reverse()
        
        # Map back to all observation indices
        obs_to_arc = {valid_obs[i]: decoded_path[i] for i in range(T)}
        full_decoded = []
        last_known = decoded_path[0]
        for o in obs_indices:
            if o in obs_to_arc:
                last_known = obs_to_arc[o]
            full_decoded.append(last_known)
        
        return full_decoded


def evaluate_route_reconstruction(model, test_features, test_trip_data):
    """
    Evaluates Classifier-Only (Argmax) vs Classifier + HMM/Viterbi
    across all student-testing trips against the true chronological ground truth.
    """
    reconstructor = HMMRouteReconstructor()
    
    X_test = test_features[FEATURE_COLS].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
        scores = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
    else:
        scores = model.predict(X_test).astype(float)
    
    test_features_copy = test_features.copy()
    test_features_copy['cls_prob'] = scores
    
    per_trip_cls_acc = []
    per_trip_hmm_acc = []
    per_trip_cls_cont = []
    per_trip_hmm_cont = []
    
    trip_reports = []
    
    print("\n" + "=" * 80)
    print("SEQUENTIAL ROUTE RECONSTRUCTION BENCHMARK (TEST PARTITION)")
    print("=" * 80)
    
    for trip_id, meta in test_trip_data.items():
        t_df = test_features_copy[test_features_copy['trip_id'].astype(str).str.zfill(8) == str(trip_id).zfill(8)]
        if len(t_df) == 0:
            continue
        
        track_df = meta['track']
        arc_info = meta['arc_info']
        adj = meta['adj']
        gt_assigned = meta['gt_assigned']
        
        obs_list = sorted(t_df['obs_idx'].unique())
        
        # 1. Classifier-Only (Greedy Argmax)
        cls_preds = []
        cands_by_obs = {}
        cls_probs_dict = {}
        
        for o in obs_list:
            grp = t_df[t_df['obs_idx'] == o]
            best_row = grp.loc[grp['cls_prob'].idxmax()]
            cls_preds.append(int(best_row['arc_id']))
            
            c_list = [(int(r['arc_id']), float(r['distance'])) for _, r in grp.iterrows()]
            cands_by_obs[int(o)] = c_list
            for _, r in grp.iterrows():
                cls_probs_dict[(int(o), int(r['arc_id']))] = float(r['cls_prob'])
        
        # 2. Classifier + HMM / Viterbi Decoding
        hmm_preds = reconstructor.decode(
            obs_list, cands_by_obs, arc_info, adj, track_df, cls_probs_dict
        )
        
        # Accuracy comparison against reference ground truth
        cls_correct = 0
        hmm_correct = 0
        total_eval = 0
        
        for i, o in enumerate(obs_list):
            if gt_assigned is not None and o < len(gt_assigned):
                target = gt_assigned[o]
                if cls_preds[i] == target:
                    cls_correct += 1
                if hmm_preds[i] == target:
                    hmm_correct += 1
                total_eval += 1
        
        cls_acc = cls_correct / max(total_eval, 1)
        hmm_acc = hmm_correct / max(total_eval, 1)
        
        # Continuity comparison
        def calc_continuity(seq):
            if len(seq) <= 1:
                return 1.0
            conn = 0
            for i in range(len(seq) - 1):
                a1, a2 = seq[i], seq[i + 1]
                if a1 == a2:
                    conn += 1
                elif a1 in arc_info and a2 in arc_info:
                    to1 = arc_info[a1]['to_node']
                    from2 = arc_info[a2]['from_node']
                    if to1 == from2 or from2 in [n for n, _ in adj.get(to1, [])]:
                        conn += 1
            return conn / (len(seq) - 1)
        
        cls_cont = calc_continuity(cls_preds)
        hmm_cont = calc_continuity(hmm_preds)
        
        per_trip_cls_acc.append(cls_acc)
        per_trip_hmm_acc.append(hmm_acc)
        per_trip_cls_cont.append(cls_cont)
        per_trip_hmm_cont.append(hmm_cont)
        
        trip_reports.append({
            'trip_id': trip_id,
            'n_observations': total_eval,
            'classifier_accuracy': float(cls_acc),
            'hmm_accuracy': float(hmm_acc),
            'accuracy_delta': float(hmm_acc - cls_acc),
            'classifier_continuity': float(cls_cont),
            'hmm_continuity': float(hmm_cont),
            'continuity_delta': float(hmm_cont - cls_cont)
        })
        
        print(f"Trip [{trip_id}]:")
        print(f"  Pointwise Acc:  Classifier = {cls_acc*100:.2f}% | HMM/Viterbi = {hmm_acc*100:.2f}% (Delta: {(hmm_acc-cls_acc)*100:+.2f}%)")
        print(f"  Route Contin.:  Classifier = {cls_cont*100:.2f}% | HMM/Viterbi = {hmm_cont*100:.2f}% (Delta: {(hmm_cont-cls_cont)*100:+.2f}%)")
    
    summary = {
        'classifier_only': {
            'avg_pointwise_accuracy': float(np.mean(per_trip_cls_acc)),
            'avg_route_continuity': float(np.mean(per_trip_cls_cont))
        },
        'hmm_viterbi': {
            'avg_pointwise_accuracy': float(np.mean(per_trip_hmm_acc)),
            'avg_route_continuity': float(np.mean(per_trip_hmm_cont))
        },
        'improvements': {
            'accuracy_gain': float(np.mean(per_trip_hmm_acc) - np.mean(per_trip_cls_acc)),
            'continuity_gain': float(np.mean(per_trip_hmm_cont) - np.mean(per_trip_cls_cont))
        },
        'per_trip_details': trip_reports
    }
    
    with open('results/hmm_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("OVERALL HMM / VITERBI IMPACT:")
    print(f"  Mean Accuracy:   {summary['classifier_only']['avg_pointwise_accuracy']*100:.2f}% -> {summary['hmm_viterbi']['avg_pointwise_accuracy']*100:.2f}%")
    print(f"  Mean Continuity: {summary['classifier_only']['avg_route_continuity']*100:.2f}% -> {summary['hmm_viterbi']['avg_route_continuity']*100:.2f}%")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    import joblib
    from data_processing import process_all_trips_fast

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "=" * 80)
    print("PATHSNAP: HMM & LOG-SPACE VITERBI SEQUENTIAL ROUTE RECONSTRUCTION")
    print("=" * 80)
    
    model_path = "models/lgbm_tuned.pkl"
    if not os.path.exists(model_path):
        model_path = "models/lightgbm.pkl"
        
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found! Please run model training first.")
        sys.exit(1)
        
    print(f"Loading trained classifier from {model_path}...")
    model = joblib.load(model_path)
    
    print("Loading test features and metadata...")
    test_features = pd.read_csv("results/test_features.csv")
    _, test_meta = process_all_trips_fast("Student folder/student_test", max_candidates=15, verbose=False)
    
    print("\nDecoding trajectories and comparing Classifier-Only vs Classifier + HMM...")
    summary = evaluate_route_reconstruction(model, test_features, test_meta)

