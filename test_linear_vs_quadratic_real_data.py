"""
TEST: Linear vs Quadratic on Real Data (AIDS, IMDB, Linux)
===========================================================

Testa Linear vs Quadratic su grafi REALI usando True_GED values.

Per ogni dataset (AIDS, IMDB, Linux):
1. Carica TUTTE le coppie di grafi da Dataset/[name]/[name]_graphs.csv
2. Carica True_GED values da True_GED/[name]/[name]_ged.csv
3. Split 80/20 train/test
4. Applica Linear → MAE_L, R²_L, time_L
5. Applica Quadratic → MAE_Q, R²_Q, time_Q
6. Output: tabella comparativa con tempi

Key metrics:
- MAE on test set (lower is better)
- Execution time (Linear vs Quadratic ratio)
- Improvement percentage
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time
import ast
from datetime import datetime

from ablation_linear_surrogate_simplified import (
    get_graph_features,
    solve_ged_fw
)
from ged_computation import extract_ot_features, compare_and_swap_graphs


def load_graphs_from_csv(dataset_name):
    """Carica i grafi da CSV e ritorna dict[graph_id -> nx.Graph]"""
    
    csv_path = f"Dataset/{dataset_name}/{dataset_name}_graphs.csv"
    print(f"  Loading graphs from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    graphs = {}
    
    for idx, row in df.iterrows():
        graph_id = row['graph_id']
        edges = ast.literal_eval(row['graph_edge_list'])
        
        # Numero di nodi = max node id + 1
        if edges:
            n_nodes = max(max(e) for e in edges) + 1
        else:
            n_nodes = 1
        
        # Crea grafo
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        
        graphs[graph_id] = G
        
        if (idx + 1) % 1000 == 0:
            print(f"    Loaded {idx+1}/{len(df)} graphs", flush=True)
    
    print(f"  Total graphs loaded: {len(graphs)}")
    return graphs


def load_ged_pairs(dataset_name, graphs):
    """
    Carica True_GED pairs e ritorna lista di (G1, G2, true_ged).
    Se un grafo non esiste, lo salta.
    """
    
    ged_path = f"True_GED/{dataset_name}/{dataset_name}_ged.csv"
    print(f"  Loading GED pairs from {ged_path}...")
    
    df_ged = pd.read_csv(ged_path)
    pairs = []
    skipped = 0
    
    for idx, row in df_ged.iterrows():
        id_1 = row['id_1']
        id_2 = row['id_2']
        true_ged = row['true_ged']
        
        if id_1 in graphs and id_2 in graphs:
            G1 = graphs[id_1]
            G2 = graphs[id_2]
            pairs.append((G1, G2, true_ged))
        else:
            skipped += 1
        
        if (idx + 1) % 50000 == 0:
            print(f"    Processed {idx+1}/{len(df_ged)} pairs", flush=True)
    
    print(f"  Total valid pairs: {len(pairs)} (skipped {skipped})")
    return pairs


def extract_features(G1, G2, mu=0.5, max_iter=30):
    """Estrae feature (GW_score + OT_features) per Linear e Quadratic."""
    
    try:
        # Skip very large graphs
        if len(G1) > 100 or len(G2) > 100:
            return None, None
        
        # Pad graphs to same size (critical for graphs with different node counts)
        G1, G2 = compare_and_swap_graphs(G1, G2)
            
        F1 = get_graph_features(G1)
        F2 = get_graph_features(G2)
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)
        
        # Compute cross matrix as element-wise product of adjacency matrices
        cross_matrix = A1 * A2
        
        # LINEAR
        try:
            Pi_L, obj_L = solve_ged_fw(A1, A2, F1, F2, mu, max_iter, formulation='linear')
            ot_L = extract_ot_features(Pi_L, cross_matrix, A1, A2)
            ot_L['gw_score'] = obj_L
        except Exception as e:
            print(f"      ERROR Linear: {str(e)[:50]}")
            ot_L = None
        
        # QUADRATIC
        try:
            Pi_Q, obj_Q = solve_ged_fw(A1, A2, F1, F2, mu, max_iter, formulation='quadratic')
            ot_Q = extract_ot_features(Pi_Q, cross_matrix, A1, A2)
            ot_Q['gw_score'] = obj_Q
        except Exception as e:
            print(f"      ERROR Quadratic: {str(e)[:50]}")
            ot_Q = None
        
        if ot_L is not None and ot_Q is not None:
            return ot_L, ot_Q
        return None, None
    except Exception as e:
        print(f"      ERROR extract_features: {str(e)[:50]}")
        return None, None


def test_dataset(dataset_name, sample_size=None):
    """Test Linear vs Quadratic su un dataset (con sampling opzionale)."""
    
    print(f"\n{'='*90}")
    print(f" {dataset_name.upper()}")
    print(f"{'='*90}")
    
    # Load graphs
    print(f"\n[1] Loading graphs...")
    graphs = load_graphs_from_csv(dataset_name)
    
    # Load GED pairs
    print(f"\n[2] Loading GED pairs...")
    pairs = load_ged_pairs(dataset_name, graphs)
    
    if len(pairs) == 0:
        print(f"  ERROR: No valid pairs found, skipping {dataset_name}")
        return None
    
    # Sample random subset (se specificato)
    if sample_size is not None and len(pairs) > sample_size:
        print(f"\n[2.5] Sampling {sample_size} pairs from {len(pairs)} total...")
        sample_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
        pairs = [pairs[i] for i in sample_indices]
        print(f"  Sampled pairs: {len(pairs)}")
    else:
        print(f"  Using all {len(pairs)} pairs")
    
    # Extract features
    print(f"\n[3] Extracting features...")
    features_L = []
    features_Q = []
    true_geds = []
    skipped_pairs = 0
    
    for idx, (G1, G2, true_ged) in enumerate(pairs):
        if (idx + 1) % 100 == 0:
            print(f"    [{idx+1}/{len(pairs)}]", flush=True)
        
        ot_L, ot_Q = extract_features(G1, G2, mu=0.5, max_iter=50)
        
        if ot_L is not None and ot_Q is not None:
            features_L.append(ot_L)
            features_Q.append(ot_Q)
            true_geds.append(true_ged)
        else:
            skipped_pairs += 1
    
    print(f"    Valid pairs for training: {len(features_L)} (skipped {skipped_pairs})")
    
    if len(features_L) == 0:
        print(f"  ERROR: No valid features extracted, skipping {dataset_name}")
        return None
    
    # Convert to DataFrames
    df_L = pd.DataFrame(features_L)
    df_Q = pd.DataFrame(features_Q)
    y = np.array(true_geds)
    
    # Split and Train
    print(f"\n[4] Training GB models (80/20 split)...")
    X_L_train, X_L_test, y_train, y_test = train_test_split(
        df_L, y, test_size=0.2, random_state=42
    )
    X_Q_train, X_Q_test, _, _ = train_test_split(
        df_Q, y, test_size=0.2, random_state=42
    )
    
    # Linear
    print(f"    Training Linear GB...")
    t0 = time.time()
    gb_L = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_L.fit(X_L_train, y_train)
    t_train_L = time.time() - t0
    
    t0 = time.time()
    y_pred_L = gb_L.predict(X_L_test)
    t_pred_L = time.time() - t0
    
    mae_L = mean_absolute_error(y_test, y_pred_L)
    r2_L = r2_score(y_test, y_pred_L)
    time_L = t_train_L + t_pred_L
    
    # Quadratic
    print(f"    Training Quadratic GB...")
    t0 = time.time()
    gb_Q = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_Q.fit(X_Q_train, y_train)
    t_train_Q = time.time() - t0
    
    t0 = time.time()
    y_pred_Q = gb_Q.predict(X_Q_test)
    t_pred_Q = time.time() - t0
    
    mae_Q = mean_absolute_error(y_test, y_pred_Q)
    r2_Q = r2_score(y_test, y_pred_Q)
    time_Q = t_train_Q + t_pred_Q
    
    # Results
    improvement = ((mae_L - mae_Q) / mae_L * 100) if mae_L > 0 else 0
    time_ratio = time_Q / time_L if time_L > 0 else 0
    winner = "Quadratic" if mae_Q < mae_L else "Linear"
    
    print(f"\n  Results:")
    print(f"    Linear:    MAE={mae_L:.6f}, R²={r2_L:.6f}, time={time_L:.2f}s")
    print(f"    Quadratic: MAE={mae_Q:.6f}, R²={r2_Q:.6f}, time={time_Q:.2f}s")
    print(f"    Winner: {winner} ({abs(improvement):.2f}% {'better' if improvement > 0 else 'worse'})")
    print(f"    Time ratio (Q/L): {time_ratio:.2f}×")
    
    return {
        'dataset': dataset_name,
        'n_valid_pairs': len(features_L),
        'n_test_samples': len(y_test),
        'mae_linear': mae_L,
        'r2_linear': r2_L,
        'time_linear': time_L,
        'mae_quadratic': mae_Q,
        'r2_quadratic': r2_Q,
        'time_quadratic': time_Q,
        'improvement_pct': improvement,
        'time_ratio_q_l': time_ratio,
        'winner': winner
    }


def main():
    """Main execution."""
    
    print("\n" + "="*90)
    print(" TEST: LINEAR vs QUADRATIC on REAL GRAPHS (AIDS, IMDB, Linux)")
    print("="*90)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for dataset in ['AIDS', 'IMDB', 'Linux']:
        result = test_dataset(dataset)
        if result:
            all_results.append(result)
    
    # Summary Table
    print("\n" + "="*90)
    print(" SUMMARY TABLE")
    print("="*90)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print("\n" + df_results[['dataset', 'n_valid_pairs', 'mae_linear', 'mae_quadratic', 
                                   'improvement_pct', 'time_ratio_q_l', 'winner']].to_string(index=False))
        
        # Save CSV
        csv_path = "results/real_data_comparison.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\n[*] Results saved to: {csv_path}")
        
        # Detailed table
        print("\n" + "="*90)
        print(" DETAILED METRICS")
        print("="*90)
        print(df_results.to_string(index=False))
    
    print("\n" + "="*90)
    print(" TEST COMPLETE")
    print("="*90)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    results = main()
