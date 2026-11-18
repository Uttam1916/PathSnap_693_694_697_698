"""
PathSnap - High-Performance Vectorized Data Processing and Feature Engineering
Loads raw trip data (.track, .nodes, .arcs, .route), generates candidates via
spatial grid indexing, aligns ground truth routes chronologically, and engineers
15 domain-specific geographical and kinetic features.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

EARTH_RADIUS = 6371000.0  # meters
GRID_SIZE = 0.003          # ~200-300m cells for fast spatial query


def load_trip(trip_dir, trip_id):
    """Load raw trip files: .track, .nodes, .arcs, and optional .route."""
    prefix = os.path.join(trip_dir, trip_id, trip_id)
    
    # 1. Track: lon, lat, timestamp
    track = np.loadtxt(f"{prefix}.track")
    track_df = pd.DataFrame(track, columns=['lon', 'lat', 'timestamp'])
    track_df['obs_idx'] = np.arange(len(track_df), dtype=int)
    
    # 2. Nodes: lon, lat (0-indexed row = node_id)
    nodes = np.loadtxt(f"{prefix}.nodes")
    nodes_df = pd.DataFrame(nodes, columns=['lon', 'lat'])
    nodes_df['node_id'] = np.arange(len(nodes_df), dtype=int)
    
    # 3. Arcs: from_node, to_node (0-indexed row = arc_id)
    arcs = np.loadtxt(f"{prefix}.arcs", dtype=int)
    arcs_df = pd.DataFrame(arcs, columns=['from_node', 'to_node'])
    arcs_df['arc_id'] = np.arange(len(arcs_df), dtype=int)
    
    # 4. Route (ground truth): sequence of traversed arc_ids
    route_path = f"{prefix}.route"
    if os.path.exists(route_path):
        route = np.loadtxt(route_path, dtype=int)
        if route.ndim == 0:
            route = np.array([int(route)])
    else:
        route = None
    
    return track_df, nodes_df, arcs_df, route


def build_adjacency(arcs_df, nodes_df):
    """Build fast adjacency structures and arc geometric metadata."""
    fn = arcs_df['from_node'].values
    tn = arcs_df['to_node'].values
    
    fn_lons = nodes_df.iloc[fn]['lon'].values
    fn_lats = nodes_df.iloc[fn]['lat'].values
    tn_lons = nodes_df.iloc[tn]['lon'].values
    tn_lats = nodes_df.iloc[tn]['lat'].values
    
    # Arc lengths
    dlon = np.radians(tn_lons - fn_lons)
    dlat = np.radians(tn_lats - fn_lats)
    a = (np.sin(dlat / 2.0)**2 + 
         np.cos(np.radians(fn_lats)) * np.cos(np.radians(tn_lats)) * np.sin(dlon / 2.0)**2)
    arc_lengths = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * EARTH_RADIUS
    
    # Arc bearings
    y = np.sin(dlon) * np.cos(np.radians(tn_lats))
    x = (np.cos(np.radians(fn_lats)) * np.sin(np.radians(tn_lats)) -
         np.sin(np.radians(fn_lats)) * np.cos(np.radians(tn_lats)) * np.cos(dlon))
    arc_bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    
    # Graph adjacency: from_node -> list of (to_node, arc_id)
    adj = defaultdict(list)
    arc_info = {}
    for aid in range(len(arcs_df)):
        f, t = int(fn[aid]), int(tn[aid])
        adj[f].append((t, aid))
        arc_info[aid] = {
            'from_node': f, 'to_node': t,
            'from_lon': fn_lons[aid], 'from_lat': fn_lats[aid],
            'to_lon': tn_lons[aid], 'to_lat': tn_lats[aid],
            'length': arc_lengths[aid],
            'bearing': arc_bearings[aid]
        }
    
    return adj, arc_info


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in meters."""
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = (np.sin(dlat / 2.0)**2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2)
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * EARTH_RADIUS


def point_to_segment_vectorized(px, py, ax, ay, bx, by):
    """Vectorized distance and projection from point P to segments AB."""
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    denom_safe = np.where(denom == 0, 1e-12, denom)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / denom_safe, 0.0, 1.0)
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = haversine_vectorized(px, py, proj_x, proj_y)
    return dist, proj_x, proj_y, t


def align_ground_truth_chronological(track_df, route, arc_info):
    """
    Chronologically aligns GPS observations to the ground truth route sequence.
    Returns array of ground truth arc_id for each observation.
    """
    if route is None or len(route) == 0:
        return None
    
    track_lons = track_df['lon'].values
    track_lats = track_df['lat'].values
    n_obs = len(track_df)
    n_route = len(route)
    
    # Pre-extract route arc coordinates
    r_fn_lons = np.array([arc_info[a]['from_lon'] for a in route])
    r_fn_lats = np.array([arc_info[a]['from_lat'] for a in route])
    r_tn_lons = np.array([arc_info[a]['to_lon'] for a in route])
    r_tn_lats = np.array([arc_info[a]['to_lat'] for a in route])
    
    assigned_gt = np.zeros(n_obs, dtype=int)
    r_ptr = 0
    
    for i in range(n_obs):
        px, py = track_lons[i], track_lats[i]
        # Search window around current route pointer: [r_ptr - 2, r_ptr + 20]
        w_start = max(0, r_ptr - 2)
        w_end = min(n_route, r_ptr + 25)
        
        ax = r_fn_lons[w_start:w_end]
        ay = r_fn_lats[w_start:w_end]
        bx = r_tn_lons[w_start:w_end]
        by = r_tn_lats[w_start:w_end]
        
        d, _, _, _ = point_to_segment_vectorized(px, py, ax, ay, bx, by)
        best_rel_idx = np.argmin(d)
        r_ptr = w_start + best_rel_idx
        assigned_gt[i] = route[r_ptr]
    
    return assigned_gt


def compute_track_kinematics(track_df):
    """Compute heading (bearing) and speed (m/s) for each GPS point."""
    lons = track_df['lon'].values
    lats = track_df['lat'].values
    ts = track_df['timestamp'].values
    n = len(track_df)
    
    headings = np.zeros(n, dtype=float)
    speeds = np.zeros(n, dtype=float)
    
    if n > 1:
        # Forward differences
        dlon = np.radians(lons[1:] - lons[:-1])
        dlat = np.radians(lats[1:] - lats[:-1])
        y = np.sin(dlon) * np.cos(np.radians(lats[1:]))
        x = (np.cos(np.radians(lats[:-1])) * np.sin(np.radians(lats[1:])) -
             np.sin(np.radians(lats[:-1])) * np.cos(np.radians(lats[1:])) * np.cos(dlon))
        fwd_bearings = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        
        dt = np.maximum(ts[1:] - ts[:-1], 0.1)
        fwd_dist = haversine_vectorized(lons[:-1], lats[:-1], lons[1:], lats[1:])
        fwd_speeds = fwd_dist / dt
        
        headings[:-1] = fwd_bearings
        headings[-1] = fwd_bearings[-1]
        
        speeds[:-1] = fwd_speeds
        speeds[-1] = fwd_speeds[-1]
    
    return headings, speeds


def generate_trip_features(trip_dir, trip_id, max_candidates=15):
    """
    Process one trip end-to-end:
    - Candidate generation via spatial grid
    - Feature extraction (15 features)
    - Ground truth label assignment
    """
    track_df, nodes_df, arcs_df, route = load_trip(trip_dir, trip_id)
    adj, arc_info = build_adjacency(arcs_df, nodes_df)
    
    fn = arcs_df['from_node'].values
    tn = arcs_df['to_node'].values
    fn_lons = nodes_df.iloc[fn]['lon'].values
    fn_lats = nodes_df.iloc[fn]['lat'].values
    tn_lons = nodes_df.iloc[tn]['lon'].values
    tn_lats = nodes_df.iloc[tn]['lat'].values
    
    arc_lengths = np.array([arc_info[a]['length'] for a in range(len(arcs_df))])
    arc_bearings = np.array([arc_info[a]['bearing'] for a in range(len(arcs_df))])
    
    # Build spatial grid index
    grid = defaultdict(list)
    cell_x = (fn_lons / GRID_SIZE).astype(int)
    cell_y = (fn_lats / GRID_SIZE).astype(int)
    for aid in range(len(arcs_df)):
        grid[(cell_x[aid], cell_y[aid])].append(aid)
    
    # Ground truth alignment
    gt_arc_per_obs = align_ground_truth_chronological(track_df, route, arc_info)
    
    # Kinematics
    gps_headings, gps_speeds = compute_track_kinematics(track_df)
    
    track_lons = track_df['lon'].values
    track_lats = track_df['lat'].values
    n_obs = len(track_df)
    
    cand_obs = []
    cand_arc = []
    cand_dist = []
    cand_rank = []
    cand_norm_dist = []
    cand_dist_ratio = []
    cand_n_cands = []
    cand_dist_from = []
    cand_dist_to = []
    cand_frac_pos = []
    cand_label = []
    
    for i in range(n_obs):
        px, py = track_lons[i], track_lats[i]
        cx, cy = int(px / GRID_SIZE), int(py / GRID_SIZE)
        
        # 5x5 grid window around GPS point
        nearby = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nearby.extend(grid.get((cx + dx, cy + dy), []))
        
        if not nearby:
            # Fallback 9x9
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    nearby.extend(grid.get((cx + dx, cy + dy), []))
        
        nearby = np.unique(nearby)
        if len(nearby) == 0:
            continue
        
        ax = fn_lons[nearby]
        ay = fn_lats[nearby]
        bx = tn_lons[nearby]
        by = tn_lats[nearby]
        
        dists, _, _, t_frac = point_to_segment_vectorized(px, py, ax, ay, bx, by)
        
        k = min(len(dists), max_candidates)
        if len(dists) > k:
            top_idx = np.argpartition(dists, k)[:k]
            top_idx = top_idx[np.argsort(dists[top_idx])]
        else:
            top_idx = np.argsort(dists)
        
        selected_arcs = nearby[top_idx]
        selected_dists = dists[top_idx]
        selected_fracs = t_frac[top_idx]
        
        min_d = selected_dists[0]
        max_d = max(selected_dists[-1], 1e-6)
        n_c = len(selected_dists)
        
        d_from = haversine_vectorized(px, py, fn_lons[selected_arcs], fn_lats[selected_arcs])
        d_to = haversine_vectorized(px, py, tn_lons[selected_arcs], tn_lats[selected_arcs])
        
        gt_target = gt_arc_per_obs[i] if gt_arc_per_obs is not None else -1
        
        for r, aid in enumerate(selected_arcs):
            cand_obs.append(i)
            cand_arc.append(aid)
            cand_dist.append(selected_dists[r])
            cand_rank.append(r)
            cand_norm_dist.append(selected_dists[r] / max_d)
            cand_dist_ratio.append(selected_dists[r] / max(min_d, 1e-6))
            cand_n_cands.append(n_c)
            cand_dist_from.append(d_from[r])
            cand_dist_to.append(d_to[r])
            cand_frac_pos.append(selected_fracs[r])
            cand_label.append(1 if (gt_target != -1 and aid == gt_target) else 0)
    
    # Vectorized feature computation across all candidates
    cand_obs = np.array(cand_obs, dtype=int)
    cand_arc = np.array(cand_arc, dtype=int)
    cand_dist = np.array(cand_dist, dtype=float)
    cand_rank = np.array(cand_rank, dtype=int)
    cand_norm_dist = np.array(cand_norm_dist, dtype=float)
    cand_dist_ratio = np.array(cand_dist_ratio, dtype=float)
    cand_n_cands = np.array(cand_n_cands, dtype=int)
    cand_dist_from = np.array(cand_dist_from, dtype=float)
    cand_dist_to = np.array(cand_dist_to, dtype=float)
    cand_frac_pos = np.array(cand_frac_pos, dtype=float)
    cand_label = np.array(cand_label, dtype=int)
    
    # Bearing diff
    arc_b = arc_bearings[cand_arc]
    gps_b = gps_headings[cand_obs]
    raw_diff = np.abs(gps_b - arc_b)
    bearing_diff = np.minimum(raw_diff, 360.0 - raw_diff)
    
    # Arc length & speed
    cand_arc_len = arc_lengths[cand_arc]
    cand_speed = gps_speeds[cand_obs]
    
    # Non-linear transformations
    cos_bear_diff = np.cos(np.radians(bearing_diff))
    bear_diff_sq = (bearing_diff / 180.0) ** 2
    speed_compat = np.clip(cand_speed / 15.0, 0.0, 3.0)
    log_dist = np.log1p(cand_dist)
    
    features_df = pd.DataFrame({
        'obs_idx': cand_obs,
        'arc_id': cand_arc,
        'trip_id': trip_id,
        'distance': cand_dist,
        'bearing_diff': bearing_diff,
        'arc_length': cand_arc_len,
        'gps_speed': cand_speed,
        'candidate_rank': cand_rank,
        'norm_distance': cand_norm_dist,
        'dist_ratio': cand_dist_ratio,
        'n_candidates': cand_n_cands,
        'cos_bearing_diff': cos_bear_diff,
        'speed_compatibility': speed_compat,
        'dist_to_from_node': cand_dist_from,
        'dist_to_to_node': cand_dist_to,
        'frac_position': cand_frac_pos,
        'log_distance': log_dist,
        'bear_diff_squared': bear_diff_sq,
        'label': cand_label
    })
    
    trip_meta = {
        'track': track_df,
        'nodes': nodes_df,
        'arcs': arcs_df,
        'route': route,
        'gt_assigned': gt_arc_per_obs,
        'arc_info': arc_info,
        'adj': adj
    }
    
    return features_df, trip_meta


def process_all_trips_fast(base_dir, max_candidates=15, verbose=True):
    """Processes all trips in a directory using vectorized pipelines."""
    trip_ids = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))])
    
    features_list = []
    trip_data = {}
    
    for i, tid in enumerate(trip_ids):
        if verbose and (i % 10 == 0 or i == len(trip_ids) - 1):
            print(f"  Processed {i+1}/{len(trip_ids)} trips...")
        try:
            f_df, meta = generate_trip_features(base_dir, tid, max_candidates)
            features_list.append(f_df)
            trip_data[tid] = meta
        except Exception as e:
            print(f"Error processing trip {tid}: {e}")
    
    combined_features = pd.concat(features_list, ignore_index=True)
    return combined_features, trip_data


def dataset_summary(base_dir):
    """Computes full exploratory statistics across trips."""
    trip_ids = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))])
    
    summary = []
    for tid in trip_ids:
        prefix = os.path.join(base_dir, tid, tid)
        track = np.loadtxt(f"{prefix}.track")
        nodes = np.loadtxt(f"{prefix}.nodes")
        arcs = np.loadtxt(f"{prefix}.arcs", dtype=int)
        
        route_path = f"{prefix}.route"
        route_len = len(np.loadtxt(route_path, dtype=int)) if os.path.exists(route_path) else 0
        
        timestamps = track[:, 2]
        time_diffs = np.diff(timestamps)
        
        total_dist = np.sum(haversine_vectorized(
            track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1]
        ))
        
        summary.append({
            'trip_id': tid,
            'n_observations': len(track),
            'n_nodes': len(nodes),
            'n_arcs': len(arcs),
            'n_route_arcs': route_len,
            'duration_s': timestamps[-1] - timestamps[0],
            'mean_time_gap': float(np.mean(time_diffs)),
            'max_time_gap': float(np.max(time_diffs)),
            'min_time_gap': float(np.min(time_diffs)),
            'trajectory_length_m': float(total_dist),
            'lon_min': float(track[:, 0].min()),
            'lon_max': float(track[:, 0].max()),
            'lat_min': float(track[:, 1].min()),
            'lat_max': float(track[:, 1].max()),
        })
    
    return pd.DataFrame(summary)
