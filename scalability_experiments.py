import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
import os
from scipy.optimize import linear_sum_assignment

from ablation_linear_surrogate_simplified import (
    get_graph_features,
    get_permutation_matrix,
)

# ==========================================
# CONFIGURAZIONE STILE PLOT
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. FUNZIONI DI UTILITÀ
# ==========================================
# get_graph_features and get_permutation_matrix are imported from
# ablation_linear_surrogate_simplified

def generate_synthetic_pair(n_nodes, m_edges, n_edits):
    """Genera coppia di grafi con differenze."""
    G1 = nx.barabasi_albert_graph(n_nodes, m_edges)
    G2 = G1.copy()
    
    existing_edges = list(G1.edges())
    non_edges = list(nx.non_edges(G1))
    
    edits_done = 0
    attempts = 0
    max_attempts = n_edits * 10
    
    while edits_done < n_edits and attempts < max_attempts:
        attempts += 1
        op_type = np.random.choice(['add', 'delete'], p=[0.5, 0.5])
        
        if op_type == 'delete' and len(existing_edges) > 0:
            idx = np.random.randint(len(existing_edges))
            u, v = existing_edges[idx]
            if G2.has_edge(u, v):
                G2.remove_edge(u, v)
                existing_edges.remove((u, v))
                non_edges.append((u, v))
                edits_done += 1
        elif op_type == 'add' and len(non_edges) > 0:
            idx = np.random.randint(len(non_edges))
            u, v = non_edges[idx]
            G2.add_edge(u, v)
            non_edges.remove((u, v))
            existing_edges.append((u, v))
            edits_done += 1
            
    return G1, G2

# ==========================================
# 2. SOLUTORE GED-SF (Versione Lineare Finale)
# ==========================================

def solve_ged_fw_linear(A1, A2, F1, F2, mu, max_iter=100):
    """
    Versione ottimizzata del solver usando il Surrogato Lineare.
    """
    n = A1.shape[0]
    
    # Precomputazione Termini Edge (GW Gradient)
    V1 = np.sum(A1**2, axis=1).reshape(-1, 1)
    V2 = np.sum(A2**2, axis=1).reshape(-1, 1)
    ones_col = np.ones((n, 1))
    
    # Inizializzazione Pi
    Pi = np.ones((n, n)) / n
    
    # Precomputazione Termini Strutturali (Statica)
    Lambda = np.sum(F1**2, axis=1, keepdims=True) + \
             np.sum(F2**2, axis=1) - \
             2 * (F1 @ F2.T)

    start_time = time.time()
    
    for t in range(max_iter):
        # Gradiente Edge
        term_A = (V1 @ ones_col.T) + (ones_col @ V2.T)
        term_B = 2 * (A1 @ Pi @ A2.T)
        grad_edge = term_A - term_B
        
        # Gradiente Strutturale (Lineare)
        grad_struct = mu * Lambda
            
        # Gradiente Totale
        full_grad = grad_edge + grad_struct
        
        # LMO (Algoritmo Ungherese)
        row_ind, col_ind = linear_sum_assignment(full_grad)
        S = get_permutation_matrix((row_ind, col_ind), n)
        
        # Aggiornamento Frank-Wolfe
        gamma = 2.0 / (t + 2)
        Pi = (1 - gamma) * Pi + gamma * S
        
    elapsed = time.time() - start_time
    return elapsed

# ==========================================
# 3. ESPERIMENTI DI SCALABILITÀ
# ==========================================

def run_scalability_experiments():
    print("=" * 70)
    print(" ESPERIMENTO DI SCALABILITÀ: THROUGHPUT + ABLATION STRUTTURALE")
    print("=" * 70)
    
    max_iter_fw = 50
    
    # ==========================================
    # ESPERIMENTO: Throughput (Dataset Load) - Con e Senza Features Strutturali
    # ==========================================
    print("\n--- Throughput Analysis: Baseline vs With Structural Features ---")
    
    # Generiamo un pool di grafi "piccoli" (tipo AIDS) per simulare il dataset
    POOL_SIZE = 200
    N_FIXED = 20
    print(f"  Generating dataset pool of {POOL_SIZE} graphs (N={N_FIXED})...")
    
    # Precomputiamo tutto per evitare overhead non correlato all'algoritmo
    graph_pool = []
    for _ in range(POOL_SIZE):
        G = nx.barabasi_albert_graph(N_FIXED, 2)
        graph_pool.append({
            'F': get_graph_features(G),
            'A': nx.to_numpy_array(G)
        })
    
    workloads = [50, 200, 500, 1000] # Numero di coppie da processare
    n_runs = 5  # Numero di simulazioni per ogni workload
    
    # ===== CASO A: Senza Features Strutturali (mu=0) =====
    print("\n  Case A: Without Structural Features (mu=0)...")
    times_baseline = []
    mu_baseline = 0.0
    
    for k_pairs in workloads:
        run_times = []
        for run_idx in range(n_runs):
            start = time.time()
            
            for _ in range(k_pairs):
                idx1, idx2 = random.sample(range(POOL_SIZE), 2)
                data1 = graph_pool[idx1]
                data2 = graph_pool[idx2]
                
                # Calcolo senza features strutturali
                _ = solve_ged_fw_linear(data1['A'], data2['A'], data1['F'], data2['F'], mu_baseline, max_iter_fw)
            
            total_t = time.time() - start
            run_times.append(total_t)
        
        avg_t = np.mean(run_times)
        times_baseline.append(avg_t)
        print(f"    Processed {k_pairs} pairs in {avg_t:.4f}s avg ({k_pairs/avg_t:.1f} pairs/sec)")

    # ===== CASO B: Con Features Strutturali (mu=0.5) =====
    print("\n  Case B: With Structural Features (mu=0.5)...")
    times_with_features = []
    mu_with_features = 0.5
    
    for k_pairs in workloads:
        run_times = []
        for run_idx in range(n_runs):
            start = time.time()
            
            for _ in range(k_pairs):
                idx1, idx2 = random.sample(range(POOL_SIZE), 2)
                data1 = graph_pool[idx1]
                data2 = graph_pool[idx2]
                
                # Calcolo con features strutturali
                _ = solve_ged_fw_linear(data1['A'], data2['A'], data1['F'], data2['F'], mu_with_features, max_iter_fw)
            
            total_t = time.time() - start
            run_times.append(total_t)
        
        avg_t = np.mean(run_times)
        times_with_features.append(avg_t)
        print(f"    Processed {k_pairs} pairs in {avg_t:.4f}s avg ({k_pairs/avg_t:.1f} pairs/sec)")
    
    # ==========================================
    # PLOTTING: Throughput Comparison
    # ==========================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot entrambe le curve
    ax.plot(workloads, times_baseline, 'o-', label='Without Structural Features (μ=0)', 
            color='tab:red', linewidth=2.5, markersize=8)
    ax.plot(workloads, times_with_features, 's-', label='With Structural Features (μ=0.5)', 
            color='tab:blue', linewidth=2.5, markersize=8)
    
    # Titolo rimosso
    ax.set_xlabel('Number of Graph Pairs Processed', fontsize=12)
    ax.set_ylabel('Total Time (seconds)', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # SALVATAGGIO IN FIGURES
    os.makedirs("figures", exist_ok=True)
    filename_png = "figures/throughput_structural_features_ablation.png"
    
    plt.savefig(filename_png, format='png', bbox_inches='tight', dpi=150)
    
    print(f"\n[*] Plot saved to: {filename_png}")
    plt.show()

if __name__ == "__main__":
    run_scalability_experiments()