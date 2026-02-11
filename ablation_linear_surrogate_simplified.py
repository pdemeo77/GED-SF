"""
ABLATION STUDY SEMPLIFICATO: Linear Surrogate vs Exact Quadratic
==================================================================
Due esperimenti:
1. Runtime vs Graph Size (100-500 nodi)
2. Divergence vs Edit Rate (500 nodi, 1%-20% edit rate)
"""

import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_graph_features(G):
    """Estrae 3 feature: Degree, Clustering, PageRank"""
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()])
    if n > 1:
        deg = deg / (n - 1)
    else:
        deg = np.zeros(n)
    
    cc = np.array(list(nx.clustering(G).values()))
    pr = np.array(list(nx.pagerank(G).values()))
    return np.column_stack((deg, cc, pr))

def get_permutation_matrix(indices, n):
    """Converte output Hungarian algorithm in matrice di permutazione"""
    Pi = np.zeros((n, n))
    Pi[indices] = 1
    return Pi

def generate_synthetic_pair(n_nodes, n_edits, topology='barabasi_albert', **kwargs):
    """Genera coppia di grafi sintetici"""
    if topology == 'barabasi_albert':
        m_edges = kwargs.get('m_edges', 2)
        G1 = nx.barabasi_albert_graph(n_nodes, m_edges)
    elif topology == 'erdos_renyi':
        p = kwargs.get('p', 0.05)
        G1 = nx.erdos_renyi_graph(n_nodes, p)
    else:
        raise ValueError(f"Topologia '{topology}' non supportata")
    
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

def solve_ged_fw(A1, A2, F1, F2, mu, max_iter=150, formulation='linear'):
    """Risolve problema GED-SF usando Frank-Wolfe"""
    n = A1.shape[0]
    
    # Precomputazioni GW term
    V1 = np.sum(A1**2, axis=1).reshape(-1, 1)
    V2 = np.sum(A2**2, axis=1).reshape(-1, 1)
    ones_col = np.ones((n, 1))
    
    # Inizializzazione Pi
    Pi = np.ones((n, n)) / n
    
    # Precomputazioni termine strutturale
    if formulation == 'linear':
        Lambda = np.sum(F1**2, axis=1, keepdims=True) + \
                 np.sum(F2**2, axis=1) - \
                 2 * (F1 @ F2.T)
    else:  # quadratic
        M_quad = F2 @ F2.T
        C_quad_const = F1 @ F2.T
    
    start_time = time.time()
    
    # Loop Frank-Wolfe
    for t in range(max_iter):
        # Gradiente GW
        term_A = (V1 @ ones_col.T) + (ones_col @ V2.T)
        term_B = 2 * (A1 @ Pi @ A2.T)
        grad_edge = term_A - term_B
        
        # Gradiente strutturale
        if formulation == 'linear':
            grad_struct = mu * Lambda
        else:  # quadratic
            grad_struct = 2 * mu * ((Pi @ M_quad) - C_quad_const)
        
        # Gradiente totale
        full_grad = grad_edge + grad_struct
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(full_grad)
        S = get_permutation_matrix((row_ind, col_ind), n)
        
        # Aggiornamento Frank-Wolfe
        gamma = 2.0 / (t + 2)
        Pi = (1 - gamma) * Pi + gamma * S
    
    elapsed = time.time() - start_time
    return Pi, elapsed

# ==========================================
# ESPERIMENTO 1: RUNTIME vs SIZE
# ==========================================

def experiment_runtime_vs_size():
    """Misura tempo esecuzione al variare della dimensione dei grafi"""
    print("\n" + "="*80)
    print(" ESPERIMENTO 1: RUNTIME vs GRAPH SIZE")
    print("="*80)
    
    node_sizes = [100, 200, 300, 400, 500]
    n_pairs_per_size = 3
    max_iter_fw = 50
    mu = 0.5
    
    topologies = {
        'Barabási-Albert (Scale-Free)': {'name': 'barabasi_albert', 'kwargs': {'m_edges': 2}},
        'Erdős-Rényi (Random)': {'name': 'erdos_renyi', 'kwargs': {'p': 0.05}}
    }
    
    results_exp1 = {}
    
    for topo_label, topo_config in topologies.items():
        print(f"\nProcessing {topo_label}...")
        topo_name = topo_config['name']
        topo_kwargs = topo_config['kwargs']
        
        times_lin = []
        times_quad = []
        
        for n in node_sizes:
            print(f"  Size n={n}...", end=" ", flush=True)
            t_lin_accum = 0.0
            t_quad_accum = 0.0
            n_edits = int(n * 0.05)
            
            for i in range(n_pairs_per_size):
                G1, G2 = generate_synthetic_pair(n, n_edits, topology=topo_name, **topo_kwargs)
                F1, F2 = get_graph_features(G1), get_graph_features(G2)
                A1, A2 = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
                
                _, time_lin = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation='linear')
                _, time_quad = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation='quadratic')
                
                t_lin_accum += time_lin
                t_quad_accum += time_quad
            
            times_lin.append(t_lin_accum / n_pairs_per_size)
            times_quad.append(t_quad_accum / n_pairs_per_size)
            print(f"Linear={times_lin[-1]:.4f}s, Quadratic={times_quad[-1]:.4f}s")
        
        results_exp1[topo_label] = {
            'times_lin': times_lin,
            'times_quad': times_quad
        }
    
    # ===== PLOTTING ESPERIMENTO 1 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'Barabási-Albert (Scale-Free)': ('tab:blue', 'tab:red'),
        'Erdős-Rényi (Random)': ('tab:orange', 'tab:green')
    }
    
    for i, (topo_label, times_dict) in enumerate(results_exp1.items()):
        ax = axes[i]
        times_lin = times_dict['times_lin']
        times_quad = times_dict['times_quad']
        
        ax.plot(node_sizes, times_lin, marker='o', label='Linear Surrogate',
                linewidth=2.5, color=colors[topo_label][0], markersize=8)
        ax.plot(node_sizes, times_quad, marker='s', label='Exact Quadratic',
                linewidth=2.5, color=colors[topo_label][1], linestyle='--', markersize=8)
        
        ax.set_xlabel('Graph Size (Nodes)', fontsize=11)
        ax.set_ylabel('Avg Runtime (seconds)', fontsize=11)
        ax.set_title(f'{topo_label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = "figures/experiment1_runtime_vs_size.png"
    plt.savefig(fig_path, format='png', bbox_inches='tight', dpi=150)
    print(f"\n[*] Saved: {fig_path}")
    plt.close()

# ==========================================
# ESPERIMENTO 2: DIVERGENCE vs EDIT RATE
# ==========================================

def experiment_divergence_vs_editrate():
    """Misura divergenza soluzioni al variare dell'edit rate (n=500)"""
    print("\n" + "="*80)
    print(" ESPERIMENTO 2: DIVERGENCE vs EDIT RATE (n=500)")
    print("="*80)
    
    n_fixed = 500
    edit_rates = np.array([0.01, 0.06, 0.11, 0.16])  # 1%, 6%, 11%, 16%
    n_runs = 2
    max_iter_fw = 30
    mu = 0.5
    
    topologies = {
        'Barabási-Albert (Scale-Free)': {'name': 'barabasi_albert', 'kwargs': {'m_edges': 2}},
        'Erdős-Rényi (Random)': {'name': 'erdos_renyi', 'kwargs': {'p': 0.05}}
    }
    
    results_exp2 = {}
    
    for topo_label, topo_config in topologies.items():
        print(f"\nProcessing {topo_label}...")
        topo_name = topo_config['name']
        topo_kwargs = topo_config['kwargs']
        
        divergences = []
        
        for rate in edit_rates:
            print(f"  Edit Rate={rate*100:.0f}%...", end=" ", flush=True)
            diff_accum = 0.0
            n_edits = max(1, int(n_fixed * rate))
            
            for i in range(n_runs):
                G1, G2 = generate_synthetic_pair(n_fixed, n_edits, topology=topo_name, **topo_kwargs)
                F1, F2 = get_graph_features(G1), get_graph_features(G2)
                A1, A2 = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
                
                Pi_lin, _ = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation='linear')
                Pi_quad, _ = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation='quadratic')
                
                diff_norm = np.linalg.norm(Pi_quad - Pi_lin, 'fro') / np.sqrt(n_fixed)
                diff_accum += diff_norm
            
            divergences.append(diff_accum / n_runs)
            print(f"Divergence={divergences[-1]:.6f}")
        
        results_exp2[topo_label] = divergences
    
    # ===== PLOTTING ESPERIMENTO 2 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    bar_colors = {
        'Barabási-Albert (Scale-Free)': 'tab:blue',
        'Erdős-Rényi (Random)': 'tab:orange'
    }
    
    edit_labels = [f"{r*100:.0f}%" for r in edit_rates]
    x = np.arange(len(edit_labels))
    
    for i, (topo_label, divergences) in enumerate(results_exp2.items()):
        ax = axes[i]
        ax.bar(x, divergences, color=bar_colors[topo_label], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(edit_labels)
        ax.set_xlabel('Edge Edit Rate (%)', fontsize=11)
        ax.set_ylabel('Normalized Frobenius Distance', fontsize=11)
        ax.set_title(f'{topo_label}\n(n=500)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = "figures/experiment2_divergence_vs_editrate.png"
    plt.savefig(fig_path, format='png', bbox_inches='tight', dpi=150)
    print(f"\n[*] Saved: {fig_path}")
    plt.close()

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" ABLATION STUDY: LINEAR SURROGATE vs EXACT QUADRATIC")
    print("="*80)
    
    experiment_runtime_vs_size()
    experiment_divergence_vs_editrate()
    
    print("\n" + "="*80)
    print(" COMPLETED")
    print("="*80)
