"""
ABLATION EXPERIMENT: Linear vs Quadratic on GED Prediction
===========================================================

Esperimento unico e completo:
- 2 topologie: BA e ER (separate)
- 50 nodi (fisso)
- BA: m ∈ {1,2,3} casuale
- 16 configurazioni: (p,q) ∈ {0,5,10,20}²
- Per OGNI config: 200 coppie, split 80/20 train/test
- Misura: MAE su test set per Linear e Quadratic

Output:
- Matrice 4×4 MAE per ogni topologia
- Tabella comparativa
- Visualizzazioni
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

from ablation_linear_surrogate_simplified import (
    get_graph_features,
    solve_ged_fw
)
from ged_computation import extract_ot_features


def generate_graph_pair_with_edits(n_nodes, p_add_pct, q_remove_pct, topology='barabasi_albert', seed=None):
    """
    Genera una coppia di grafi G1, G2 con controllo del numero di edit.
    
    Ritorna: (G1, G2, true_ged)
    dove true_ged = numero di archi aggiunti + numero di archi rimossi
    """
    if seed is not None:
        np.random.seed(seed)
    
    if topology == 'barabasi_albert':
        # m casuale ∈ {1,2,3}
        m = np.random.randint(1, 4)
        G1 = nx.barabasi_albert_graph(n_nodes, m)
    else:  # erdos_renyi
        G1 = nx.erdos_renyi_graph(n_nodes, 0.05, seed=seed)
    
    G2 = G1.copy()
    
    # === ADD EDGES ===
    non_edges = list(nx.non_edges(G1))
    n_add = min(int(len(non_edges) * p_add_pct / 100.0), len(non_edges))
    
    added_edges = []
    if n_add > 0:
        indices = np.random.choice(len(non_edges), size=n_add, replace=False)
        added_edges = [non_edges[i] for i in indices]
        for u, v in added_edges:
            G2.add_edge(u, v)
    
    # === REMOVE EDGES ===
    added_edges_set = set(added_edges) | set((v, u) for u, v in added_edges)
    available_edges = [e for e in G2.edges() 
                      if e not in added_edges_set and (e[1], e[0]) not in added_edges_set]
    
    n_remove = min(int(len(available_edges) * q_remove_pct / 100.0), len(available_edges))
    
    removed_edges = []
    if n_remove > 0:
        indices = np.random.choice(len(available_edges), size=n_remove, replace=False)
        removed_edges = [available_edges[i] for i in indices]
        for u, v in removed_edges:
            G2.remove_edge(u, v)
    
    true_ged = len(added_edges) + len(removed_edges)
    
    return G1, G2, true_ged


def generate_dataset(n_nodes, n_pairs, p_add_pct, q_remove_pct, topology='barabasi_albert', seed_start=0):
    """Genera dataset di n_pairs coppie di grafi."""
    
    data = []
    for i in range(n_pairs):
        seed = seed_start + i
        G1, G2, true_ged = generate_graph_pair_with_edits(
            n_nodes, p_add_pct, q_remove_pct, topology, seed
        )
        
        F1 = get_graph_features(G1)
        F2 = get_graph_features(G2)
        A1 = nx.to_numpy_array(G1)
        A2 = nx.to_numpy_array(G2)
        
        data.append({
            'A1': A1, 'A2': A2,
            'F1': F1, 'F2': F2,
            'true_ged': true_ged
        })
    
    return data


def extract_features(dataset, mu=0.5, max_iter=50):
    """Estrae feature (GW_score + 8 OT_features) per Linear e Quadratic."""
    
    features_L, features_Q = [], []
    
    for idx, item in enumerate(dataset):
        if (idx + 1) % 50 == 0:
            print(f"      [{idx+1}/{len(dataset)}]", flush=True)
        
        A1, A2, F1, F2 = item['A1'], item['A2'], item['F1'], item['F2']
        
        try:
            # LINEAR
            Pi_L, obj_L = solve_ged_fw(A1, A2, F1, F2, mu, max_iter, formulation='linear')
            ot_L = extract_ot_features(Pi_L, A1 @ A2, A1, A2)
            ot_L['gw_score'] = obj_L
            features_L.append(ot_L)
            
            # QUADRATIC
            Pi_Q, obj_Q = solve_ged_fw(A1, A2, F1, F2, mu, max_iter, formulation='quadratic')
            ot_Q = extract_ot_features(Pi_Q, A1 @ A2, A1, A2)
            ot_Q['gw_score'] = obj_Q
            features_Q.append(ot_Q)
        except:
            # In caso di errore, salta
            continue
    
    return pd.DataFrame(features_L), pd.DataFrame(features_Q)


def train_and_evaluate(features_L, features_Q, true_geds):
    """Addestra GB su features e ritorna MAE e R² su test set."""
    
    # Split
    X_L_train, X_L_test, y_train, y_test = train_test_split(
        features_L, true_geds, test_size=0.2, random_state=42
    )
    X_Q_train, X_Q_test, _, _ = train_test_split(
        features_Q, true_geds, test_size=0.2, random_state=42
    )
    
    # GB Models
    gb_L = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_Q = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    gb_L.fit(X_L_train, y_train)
    gb_Q.fit(X_Q_train, y_train)
    
    # Predictions
    y_pred_L = gb_L.predict(X_L_test)
    y_pred_Q = gb_Q.predict(X_Q_test)
    
    # Metrics
    mae_L = mean_absolute_error(y_test, y_pred_L)
    r2_L = r2_score(y_test, y_pred_L)
    mae_Q = mean_absolute_error(y_test, y_pred_Q)
    r2_Q = r2_score(y_test, y_pred_Q)
    
    return mae_L, r2_L, mae_Q, r2_Q


def run_experiment():
    """Main experiment runner."""
    
    print("\n" + "="*90)
    print(" ABLATION EXPERIMENT: LINEAR vs QUADRATIC for GED Prediction")
    print("="*90)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    n_nodes = 50
    n_pairs = 200
    p_values = [0, 5, 10, 20]
    q_values = [0, 5, 10, 20]
    topologies = ['barabasi_albert', 'erdos_renyi']
    topology_labels = {'barabasi_albert': 'Barabási-Albert', 'erdos_renyi': 'Erdős-Rényi'}
    
    # Store results
    all_results = []
    mae_matrices = {}
    r2_matrices = {}
    
    for topology in topologies:
        print(f"\n{'='*90}")
        print(f" {topology_labels[topology].upper()}")
        print(f"{'='*90}")
        
        # Initialize matrices
        mae_L_matrix = np.zeros((len(p_values), len(q_values)))
        mae_Q_matrix = np.zeros((len(p_values), len(q_values)))
        r2_L_matrix = np.zeros((len(p_values), len(q_values)))
        r2_Q_matrix = np.zeros((len(p_values), len(q_values)))
        
        seed_offset = 0
        
        for p_idx, p in enumerate(p_values):
            for q_idx, q in enumerate(q_values):
                config_label = f"p={p}%, q={q}%"
                print(f"\n  [{config_label}]")
                
                print(f"    Generating {n_pairs} pairs...")
                dataset = generate_dataset(n_nodes, n_pairs, p, q, topology, seed_start=seed_offset)
                seed_offset += n_pairs
                
                true_geds = np.array([item['true_ged'] for item in dataset])
                print(f"      GED range: {true_geds.min()}-{true_geds.max()}, mean={true_geds.mean():.2f}")
                
                print(f"    Extracting features...")
                features_L, features_Q = extract_features(dataset, mu=0.5, max_iter=50)
                
                print(f"    Training GB models (80/20 split)...")
                mae_L, r2_L, mae_Q, r2_Q = train_and_evaluate(features_L, features_Q, true_geds)
                
                # Store in matrices
                mae_L_matrix[p_idx, q_idx] = mae_L
                mae_Q_matrix[p_idx, q_idx] = mae_Q
                r2_L_matrix[p_idx, q_idx] = r2_L
                r2_Q_matrix[p_idx, q_idx] = r2_Q
                
                improvement = ((mae_L - mae_Q) / mae_L * 100) if mae_L > 0 else 0
                winner = "Q" if mae_Q < mae_L else "L"
                
                print(f"      Linear:   MAE={mae_L:.4f}, R²={r2_L:.4f}")
                print(f"      Quadratic: MAE={mae_Q:.4f}, R²={r2_Q:.4f}")
                print(f"      Winner: {winner} ({abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'})")
                
                all_results.append({
                    'topology': topology_labels[topology],
                    'p': p,
                    'q': q,
                    'mae_linear': mae_L,
                    'r2_linear': r2_L,
                    'mae_quadratic': mae_Q,
                    'r2_quadratic': r2_Q,
                    'improvement_pct': improvement
                })
        
        mae_matrices[topology] = (mae_L_matrix, mae_Q_matrix)
        r2_matrices[topology] = (r2_L_matrix, r2_Q_matrix)
    
    # === SUMMARY TABLE ===
    print("\n" + "="*90)
    print(" SUMMARY TABLE")
    print("="*90)
    
    df_results = pd.DataFrame(all_results)
    
    for topology in topologies:
        print(f"\n{topology_labels[topology]}:")
        print("-" * 90)
        df_topo = df_results[df_results['topology'] == topology_labels[topology]]
        print(df_topo[['p', 'q', 'mae_linear', 'mae_quadratic', 'improvement_pct']].to_string(index=False))
    
    # === HEATMAPS ===
    print("\n[*] Generating visualizations...")
    
    for topology in topologies:
        mae_L, mae_Q = mae_matrices[topology]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Linear heatmap
        sns.heatmap(mae_L, annot=True, fmt='.4f', cmap='YlOrRd', 
                    xticklabels=[f'q={q}' for q in q_values],
                    yticklabels=[f'p={p}' for p in p_values],
                    ax=axes[0], cbar_kws={'label': 'MAE'})
        axes[0].set_title(f'Linear - {topology_labels[topology]}', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Edit Rate (remove %)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Edit Rate (add %)', fontsize=11, fontweight='bold')
        
        # Quadratic heatmap
        sns.heatmap(mae_Q, annot=True, fmt='.4f', cmap='YlGnBu',
                    xticklabels=[f'q={q}' for q in q_values],
                    yticklabels=[f'p={p}' for p in p_values],
                    ax=axes[1], cbar_kws={'label': 'MAE'})
        axes[1].set_title(f'Quadratic - {topology_labels[topology]}', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Edit Rate (remove %)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Edit Rate (add %)', fontsize=11, fontweight='bold')
        
        plt.suptitle(f'GED Prediction MAE: {topology_labels[topology]} (200 pairs per config)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        fig_path = f"figures/ablation_mae_{topology}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path}")
        plt.close()
        
        # Improvement heatmap
        improvement_matrix = ((mae_L - mae_Q) / mae_L) * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    xticklabels=[f'q={q}' for q in q_values],
                    yticklabels=[f'p={p}' for p in p_values],
                    ax=ax, cbar_kws={'label': 'Improvement (%)'})
        ax.set_title(f'Quadratic Improvement over Linear - {topology_labels[topology]}\n(positive = Q better)',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Edit Rate (remove %)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Edit Rate (add %)', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        fig_path = f"figures/ablation_improvement_{topology}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {fig_path}")
        plt.close()
    
    # === SAVE CSV ===
    csv_path = "results/ablation_experiment_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*90)
    print(" EXPERIMENT COMPLETE")
    print("="*90)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal configurations: {len(p_values) * len(q_values) * len(topologies)}")
    print(f"Total pairs generated: {len(p_values) * len(q_values) * len(topologies) * n_pairs}")
    print(f"\nResults saved to: results/ablation_experiment_results.csv")
    print(f"Visualizations saved to: figures/ablation_mae_*.png, ablation_improvement_*.png")
    
    return df_results


if __name__ == "__main__":
    import sys
    # Tee output to both console and log file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file = open("ablation_full_output.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    
    results = run_experiment()
    
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
