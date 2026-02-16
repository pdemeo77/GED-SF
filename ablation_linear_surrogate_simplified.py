"""
ABLATION STUDY: Linear Surrogate vs Exact Quadratic - Versione Pulita
=====================================================================

OBIETTIVO DELLO STUDIO:
=======================
Confrontare due formulazioni dell'algoritmo GED-SF (Graph Edit Distance con Structural Features):
  1. LINEAR SURROGATE: Approssima il termine strutturale con un'espressione lineare (veloce)
  2. EXACT QUADRATIC: Formula quadratica esatta (accurata)

DOMANDE SCIENTIFICHE:
=====================
1. PERFORMANCE: Quanto è veloce Linear rispetto a Quadratic? (Experimento 1)
2. QUALITA': Quale formulazione predice meglio il GED vero con ML? (Esperimenti A, B, C)

ESPERIMENTI PIANIFICATI:
========================
EXPERIMENT 1 - RUNTIME vs GRAPH SIZE (Completo ✓):
  - Topologie: BA, ER
  - Sizes: 100-500 nodi
  - Misura: tempo di esecuzione
  - Output: Grafico speed-up

EXPERIMENT A - QUALITY vs (p,q) - SINGLE CONFIG (WIP):
  - Dataset: 200 coppie BA+ER, n=50
  - Config: Una sola coppia (p=5%, q=10%)
  - Valutazione: MAE con GradientBoosting
  - Output: MAE_Linear vs MAE_Quadratic

EXPERIMENT B - QUALITY vs ALL CONFIGS (WIP):
  - Dataset: 200 coppie per ogni (p,q)
  - Configs: p,q ∈ {0, 5, 10, 20} = 16 totali
  - Topologie: BA separate, ER separate
  - Output: Matrici 4×4 con MAE per L e Q

EXPERIMENT C - TOPOLOGY COMPARISON (WIP):
  - Dataset: 200 coppie con m variabile
  - Topologie: BA + ER + Random Regular
  - Valutazione: MAE medio per topologia
  - Output: Confronto robustezza su topologie

STRUTTURA DEL FILE:
====================
1. UTILITY FUNCTIONS (get_graph_features, get_permutation_matrix, etc.)
2. CORE ALGORITHM (solve_ged_fw con linear/quadratic)
3. EXPERIMENT 1 (runtime_vs_size - MANTIENI)
4. EXPERIMENT A-C PLACEHOLDERS (per nuovi esperimenti)
5. MAIN (orchestrazione)
"""

import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Stile grafico
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_graph_features(G):
    """
    Estrae 3 feature strutturali per ogni nodo del grafo.
    
    SCOPO:
    ------
    Calcolare misure di centralità che catturano proprietà topologiche importanti.
    Queste feature vengono usate per costruire il termine strutturale della FGW.
    
    FEATURE ESTRATTE:
    -----------------
    1. DEGREE CENTRALITY (normalizzato):
       - Misura: numero di archi connessi a ogni nodo
       - Normalizzazione: diviso per (n-1) per scala [0, 1]
       - Intuizione: nodi con alto grado sono "hub" nella rete
       - Formula: degree(v) / (n - 1)
    
    2. CLUSTERING COEFFICIENT:
       - Misura: probabilità che i vicini di un nodo siano vicini tra loro
       - Scala: [0, 1] (0 = nessun triangolo, 1 = sottografo completo)
       - Intuizione: mette in evidenza la tendenza a formare cluster/comunità
       - Formula: |{(u,w) : u,w ∈ N(v) e (u,w) ∈ E}| / C(k_v, 2)
                  dove N(v) = vizini di v, k_v = grado di v
    
    3. PAGERANK:
       - Misura: importanza del nodo nella struttura globale del grafo
       - Scala: [0, 1] approssimativamente (dipende dalla struttura)
       - Intuizione: nodi raggiungibili da molti altri nodi importanti = più importanti
       - Formula: iterativa (come nell'algoritmo Google PageRank)
    
    COME VENGONO USATE:
    -------------------
    Le 3 feature vengono organizzate in una matrice n×3:
        F = [degree    clustering  pagerank
             ...]
    
    Questa matrice serve per calcolare le distanze strutturali tra nodi:
        D_struct[i,j] = distanza euclidea tra feature di nodo i e nodo j
    
    Args:
        G (nx.Graph): Grafo NetworkX
    
    Returns:
        np.ndarray: Matrice di shape (n_nodes, 3) contenente le 3 feature
                    Una riga per nodo: [degree, clustering, pagerank]
    
    Esempio:
        >>> G = nx.karate_club_graph()
        >>> F = get_graph_features(G)
        >>> print(F.shape)  # (34, 3)
        >>> print(F[0])     # [0.1, 0.45, 0.023] (approx)
    
    NOTE IMPLEMENTATIVE:
    --------------------
    - Se il grafo è disconnesso: pagerank() e clustering() gestiscono comunque bene
    - Se n=1: il grafo isolato ha degree=0 (eviteremmo divisione per 0)
    - Non normalizzamo clustering e pagerank: range naturali diversi sono OK
      perché poi verranno normalizzate dalla StandardScaler nei modelli ML
    """
    n = G.number_of_nodes()
    
    # 1. DEGREE CENTRALITY - normalizzato
    deg = np.array([d for _, d in G.degree()])  # lista dei gradi: [d(v1), d(v2), ...]
    if n > 1:
        deg = deg / (n - 1)  # normalizza in [0, 1]
    else:
        deg = np.zeros(n)  # caso degenere: grafo con 1 nodo isolato
    
    # 2. CLUSTERING COEFFICIENT - non normalizzato (scala ~[0, 1] comunque)
    cc = np.array(list(nx.clustering(G).values()))  # lista dei clustering coefficient
    
    # 3. PAGERANK - non normalizzato (scala ~[0, 1] naturalmente)
    pr = np.array(list(nx.pagerank(G).values()))  # lista dei pagerank
    
    # Organizza in matrice n×3
    return np.column_stack((deg, cc, pr))

def get_permutation_matrix(indices, n):
    """
    Converte l'output dell'algoritmo Hungarian (matching bipartito ottimale)
    in una matrice di permutazione (one-hot).
    
    CONTESTO: HUNGARIAN ALGORITHM
    ==============================
    L'algoritmo Hungarian risolve il problema dell'assegnamento ottimale:
    minimizza il costo totale di matching tra due insiemi di n elementi.
    
    Output di linear_sum_assignment(C):
        row_ind: [0, 1, 2, ..., n-1]  (sempre in ordine)
        col_ind: [i1, i2, i3, ..., in]  (permutazione dell'output)
    
    Significato: riga i è assegnata alla colonna col_ind[i]
    
    MATRICE DI PERMUTAZIONE
    =======================
    Una matrice di permutazione P è una matrice binaria n×n tale che:
    - Contiene esattamente un 1 per ogni riga e colonna
    - P[i, j] = 1 significa "nodo i è allineato con nodo j"
    
    Geometricamente: rappresenta un matching perfetto bipartito (one-to-one).
    
    CONVERSIONE:
    ============
    Input: indices = (row_ind, col_ind)  # tuple di due array
    Output: P matrice binaria n×n dove P[i, col_ind[i]] = 1
    
    Esempio:
        >>> col_ind = [2, 0, 1]  # matching: 0→2, 1→0, 2→1
        >>> P = get_permutation_matrix(([0,1,2], [2,0,1]), n=3)
        >>> print(P)
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]
    
    USO NELL'ALGORITMO FRANK-WOLFE:
    ================================
    Nel ciclo di ottimizzazione, cerchiamo di trovare la permutazione P
    che minimizza il costo di allineamento tra due grafi.
    
    Ad ogni iterazione:
    1. Calcoliamo il gradiente (matrice di costo)
    2. Risolviamo assignment problem con Hungarian
    3. Convertiamo in matrice di permutazione (questo)
    4. Aggiorniamo la soluzione con media mobile (FW step)
    
    Args:
        indices (tuple): (row_ind, col_ind) dall'output di linear_sum_assignment
        n (int): dimensione della matrice (n×n)
    
    Returns:
        np.ndarray: Matrice binaria n×n di permutazione
    """
    Pi = np.zeros((n, n))
    Pi[indices] = 1  # Metti 1 alle posizioni specificate da indices
    return Pi

def generate_graph_pair_with_edits(n_nodes, p_add_pct, q_remove_pct, topology='erdos_renyi', seed=None):
    """
    Genera una coppia di grafi (G1, G2) dove G2 è ottenuto da G1 applicando edit controllati.
    
    LOGICA DELL'ALGORITMO:
    ======================
    
    1. GENERA G1 casuale con topologia specificata
    2. COPIA G1 in G2
    3. FASE 1 - AGGIUNTE:
       - Calcola numero di archi in G1: n_edges_in_G1
       - Calcola numero di non-archi: n_non_edges
       - Numero di archi da aggiungere: n_add = ceil(p_add_pct% × n_non_edges)
       - Aggiungi n_add archi casuali da non-edges, registra quali sono stati aggiunti
    
    4. FASE 2 - RIMOZIONI:
       - Calcola numero di archi disponibili per rimozione da G2:
         n_edges_available = # archi in G2 attuali - # archi appena aggiunti
       - Numero di archi da rimuovere: n_remove = ceil(q_remove_pct% × n_edges_available)
       - Rimuovi n_remove archi (escludendo quelli appena aggiunti)
    
    5. RITORNA:
       - G1, G2
       - n_actually_added: numero di archi effettivamente aggiunti
       - n_actually_removed: numero di archi effettivamente rimossi
       - GED_true = n_actually_added + n_actually_removed (numero totale di edit)
    
    NOTA IMPORTANTE SULLA GED:
    ==========================
    La GED vera tra G1 e G2 è semplicemente:
    - GED = (numero di archi aggiunti) + (numero di archi rimossi)
    
    Perché? Perché la distanza edit minima è il numero di operazioni di edit,
    dove ogni operazione ha costo 1.
    
    Args:
        n_nodes (int): Numero di nodi del grafo (es. 100, 50)
        p_add_pct (float): Percentuale di non-archi da aggiungere (es. 5, 10, 20)
                          Nota: è percentuale rispetto al numero totale di non-archi disponibili
        q_remove_pct (float): Percentuale di archi da rimuovere (es. 5, 10, 20)
                             Nota: è percentuale degli archi DISPONIBILI per rimozione
                             (cioè escludendo quelli appena aggiunti)
        topology (str): 'erdos_renyi' o 'barabasi_albert'
        seed (int, optional): Random seed per riproducibilità. Se None, usa uno casuale
    
    Returns:
        tuple: (G1, G2, n_actually_added, n_actually_removed, GED_true)
            G1 (nx.Graph): Grafo originale
            G2 (nx.Graph): Grafo modificato
            n_actually_added (int): Numero di archi effettivamente aggiunti
            n_actually_removed (int): Numero di archi effettivamente rimossi
            GED_true (int): GED vera = n_actually_added + n_actually_removed
    
    Esempio:
        >>> G1, G2, n_add, n_rem, ged = generate_graph_pair_with_edits(
        ...     n_nodes=100, p_add_pct=5, q_remove_pct=10, 
        ...     topology='erdos_renyi', seed=42)
        >>> print(f"GED vera = {ged}")  # GED vera = n_add + n_rem
        >>> print(f"G1 archi: {G1.number_of_edges()}, G2 archi: {G2.number_of_edges()}")
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # ========================
    # STEP 1: Genera G1
    # ========================
    
    if topology == 'erdos_renyi':
        p = 0.05  # Probabilità di arco (default)
        G1 = nx.erdos_renyi_graph(n_nodes, p)
    elif topology == 'barabasi_albert':
        # Opzione A: m fisso (2)
        m = 2  
        # Opzione C: m casuale in [1, 2, 3] per maggiore varietà
        # Decommentare la riga sottostante per usare m casuale:
        # m = np.random.randint(1, 4)  # m ∈ {1, 2, 3}
        G1 = nx.barabasi_albert_graph(n_nodes, m)
    else:
        raise ValueError(f"Topologia '{topology}' non supportata. Usa 'erdos_renyi' o 'barabasi_albert'")
    
    # ========================
    # STEP 2: Copia G1 in G2
    # ========================
    
    G2 = G1.copy()
    
    # ========================
    # STEP 3: FASE 1 - AGGIUNTE
    # ========================
    
    n_non_edges_before = len(list(nx.non_edges(G1)))  # Numero di non-archi disponibili
    n_add_target = max(0, int(np.ceil(p_add_pct / 100.0 * n_non_edges_before)))
    
    # Seleziona non-archi da aggiungere
    non_edges_list = list(nx.non_edges(G1))
    if n_add_target > len(non_edges_list):
        n_add_target = len(non_edges_list)  # Non posso aggiungere più di quelli disponibili
    
    # Scegli casualmente quali non-archi aggiungere
    indices_to_add = np.random.choice(len(non_edges_list), size=n_add_target, replace=False)
    added_edges = [non_edges_list[i] for i in indices_to_add]
    
    # Aggiungi effettivamente gli archi
    for u, v in added_edges:
        G2.add_edge(u, v)
    
    n_actually_added = len(added_edges)
    
    # ========================
    # STEP 4: FASE 2 - RIMOZIONI
    # ========================
    
    # Calcola gli archi disponibili per rimozione (escludendo quelli appena aggiunti)
    added_edges_set = set(added_edges) | set((v, u) for u, v in added_edges)  # Considera entrambi gli ordini
    
    # Seleziona archi da rimuovere (solo da G2, escludendo quelli aggiunti)
    edges_available_for_removal = [e for e in G2.edges() if e not in added_edges_set and (e[1], e[0]) not in added_edges_set]
    
    n_remove_target = max(0, int(np.ceil(q_remove_pct / 100.0 * len(edges_available_for_removal))))
    
    if n_remove_target > len(edges_available_for_removal):
        n_remove_target = len(edges_available_for_removal)
    
    # Scegli casualmente quali archi rimuovere
    if len(edges_available_for_removal) > 0:
        indices_to_remove = np.random.choice(len(edges_available_for_removal), size=n_remove_target, replace=False)
        edges_to_remove = [edges_available_for_removal[i] for i in indices_to_remove]
        
        # Rimuovi effettivamente gli archi
        for u, v in edges_to_remove:
            G2.remove_edge(u, v)
        
        n_actually_removed = len(edges_to_remove)
    else:
        n_actually_removed = 0
    
    # ========================
    # STEP 5: Calcola GED vera
    # ========================
    
    GED_true = n_actually_added + n_actually_removed
    
    return G1, G2, n_actually_added, n_actually_removed, GED_true


def generate_synthetic_pair(n_nodes, n_edits, topology='barabasi_albert', **kwargs):
    """
    Genera una coppia di grafi sintetici disturbati per testare l'algoritmo.
    
    MOTIVAZIONE:
    ============
    Vogliamo controllare precisamente:
    - La dimensione dei grafi (reproducibilità)
    - Il numero di edit (controllo della difficoltà)
    - La topologia (test su diversi tipi di rete)
    
    Per questo usiamo grafi sintetici invece di dati reali.
    
    PIPELINE:
    =========
    1. Genera grafo G1 con topologia scelta
    2. Copia G1 in G2
    3. Applica random edits a G2 (add/delete archi)
    4. Ritorna (G1, G2) - coppia per testare approssimazione GED
    
    TOPOLOGIE SUPPORTATE:
    =====================
    
    1. BARABÁSI-ALBERT (Scale-Free Networks):
       - Modello di preferential attachment
       - Proprietà: pochi nodi con grado alto (hubs), molti con grado basso
       - Realismo: simula reti biologiche, sociali, web
       - Parametri: n_nodes, m_edges (archi nuovi per step)
       - Complessità: O(n*m)
       
       Esempio: BA(100, 2) → media ~4 archi per nodo, ma range ampio [0, 50+]
    
    2. ERDŐS-RÉNYI (Random Networks):
       - Ogni coppia di nodi ha probabilità p di essere connessa
       - Proprietà: distribuzione gradi è binomiale (più omogenea)
       - Realismo: baseline "nullo" per confronti
       - Parametri: n_nodes, p (probabilità arco)
       - Complessità: O(n²)
       
       Esempio: ER(100, 0.05) → media ~5 archi per nodo
    
    EDIT OPERATIONS:
    ================
    Ad ogni step, scegliamo randomly tra:
    - ADD edge: aggiunge un arco non presente (se possibile)
    - DELETE edge: rimuove un arco presente (se possibile)
    
    Probabilità: 50% add, 50% delete
    
    Logica: Un grafo G → G' con esattamente n_edits differenze
    
    Args:
        n_nodes (int): Numero di nodi nel grafo
        n_edits (int): Numero di edit operation da applicare (add/delete)
        topology (str): 'barabasi_albert' o 'erdos_renyi'
        **kwargs: Parametri specifici della topologia
                  - Per BA: m_edges (default=2)
                  - Per ER: p (default=0.05)
    
    Returns:
        tuple: (G1, G2)
            G1: Grafo originale generato
            G2: Grafo disturbato con n_edits differenze
    
    Esempio:
        >>> G1, G2 = generate_synthetic_pair(100, 10, 'barabasi_albert', m_edges=2)
        >>> print(f"Differ by {abs(G1.number_of_edges() - G2.number_of_edges())} edges")
        
        >>> G1, G2 = generate_synthetic_pair(100, 5, 'erdos_renyi', p=0.05)
    
    NOTE:
    -----
    - Se non riusciamo a trovare sufficienti edge/non-edge in max_attempts,
      la funzione ritorna comunque (potrebbe essere n_edits_effettivi < n_edits richiesti)
    - max_attempts = 10 * n_edits per evitare loop infiniti
    """
    # Genera grafo G1 con topologia scelta
    if topology == 'barabasi_albert':
        m_edges = kwargs.get('m_edges', 2)  # Default: 2 archi per nuovo nodo
        G1 = nx.barabasi_albert_graph(n_nodes, m_edges)
    elif topology == 'erdos_renyi':
        p = kwargs.get('p', 0.05)  # Default: 5% di probabilità arco
        G1 = nx.erdos_renyi_graph(n_nodes, p)
    else:
        raise ValueError(f"Topologia '{topology}' non supportata")
    
    # Copia per disturbamento
    G2 = G1.copy()
    existing_edges = list(G1.edges())
    non_edges = list(nx.non_edges(G1))
    
    # Applicazione edits fino a raggiungere n_edits (con timeout)
    edits_done = 0
    attempts = 0
    max_attempts = n_edits * 10  # Limite sulla ricerca per evitare loop infiniti
    
    while edits_done < n_edits and attempts < max_attempts:
        attempts += 1
        
        # Scegli casualmente tra aggiungere o rimuovere arco
        op_type = np.random.choice(['add', 'delete'], p=[0.5, 0.5])
        
        # OPERAZIONE 1: DELETE - rimuovi un arco casuale
        if op_type == 'delete' and len(existing_edges) > 0:
            idx = np.random.randint(len(existing_edges))
            u, v = existing_edges[idx]
            if G2.has_edge(u, v):  # Verifica che esista ancora
                G2.remove_edge(u, v)
                existing_edges.remove((u, v))
                non_edges.append((u, v))  # Ora è un non-edge
                edits_done += 1
        
        # OPERAZIONE 2: ADD - aggiungi un arco ne non aggiunto
        elif op_type == 'add' and len(non_edges) > 0:
            idx = np.random.randint(len(non_edges))
            u, v = non_edges[idx]
            G2.add_edge(u, v)
            non_edges.remove((u, v))
            existing_edges.append((u, v))  # Ora è un edge
            edits_done += 1
    
    return G1, G2

def solve_ged_fw(A1, A2, F1, F2, mu, max_iter=150, formulation='linear'):
    """
    Risolve il problema di Graph Edit Distance usando l'algoritmo Frank-Wolfe.
    
    PROBLEMA DA RISOLVERE:
    ======================
    Vogliamo trovare la matrice di allineamento ottimale P (permutazione stocastica)
    che minimizza:
    
        min_P  f(P) = (edge_cost) + μ * (structural_cost)
    
    Dove P è il coupling tra nodi dei due grafi (P[i,j] = "quanto il nodo i è allineato con j")
    
    FORMULAZIONE EDGE COST (GW - Gromov-Wasserstein):
    ==================================================
    
        f_edge(P) = sum_ij sum_kl (|A1[i,k] - A2[j,l]|² * P[i,j] * P[k,l])
    
    Significato: penalizza gli allineamenti dove la struttura dei grafi non corrisponde.
    Esempio: Se P dice "nodo 1→a", allora gli archi attorno a 1 devono corrispondere
             agli archi attorno ad a.
    
    Per efficienza computazionale, questo viene riscritto come:
    
        f_edge(P) = tr(V1^T diag(ones) P) + tr(diag(ones)^T V2 P^T)
                    - 2 * tr(A1 P A2^T P)
    
    Dove V1[i] = sum_k A1[i,k]² (norma al quadrato di ogni riga)
    
    FORMULAZIONE STRUCTURAL COST:
    =============================
    Due formulazioni alternative:
    
    1. LINEAR SURROGATE (Fast):
       ========================
       Approssima il termine strutturale con una forma lineare:
       
           f_struct_lin(P) = sum_ij (Lambda[i,j] * P[i,j])
       
       Dove Lambda[i,j] = ||F1[i]||² + ||F2[j]||² - 2<F1[i], F2[j]>
                        (distanza euclidea al quadrato normalizzata)
       
       Vantaggio: Gradiente è semplice (solo Lambda), no dipendenza da P
                  Computazione: O(n²)
       
       Svantaggio: Non cattura le correlazioni tra allineamenti
                   Il gradiente è statico in tutto l'algoritmo
    
    2. EXACT QUADRATIC (Accurate):
       ============================
       Usa la formulazione quadratica completa:
       
           f_struct_quad(P) = tr(P F1 F2^T F2 F1^T P^T) - 2*tr(P F1 F2^T)
       
       Questo è più simile al termine GW, catturando l'interdipendenza degli allineamenti.
       
       Vantaggio: Formulazione teoricamente corretta, gradiente dipende da P attuale
       
       Svantaggio: Calcolo O(n³) per il gradiente (moltiplicazioni matriciali)
                   Più lento soprattutto per grafi grandi
    
    ALGORITMO FRANK-WOLFE:
    ======================
    È un algoritmo di ottimizzazione convessa che procede per approssimazioni lineari locali.
    
    Pseudocode:
    -----------
    
    1. INIZIALIZZA: Pi ← (1/n) * ones (allineamento uniforme)
    
    2. LOOP t = 0 a max_iter:
       
       a) CALCOLA GRADIENTE:
          ∇f(Pi) = [grad_edge] + [grad_struct]
       
       b) RISOLVI ASSIGNMENT PROBLEM:
          S* ← argmin_S <∇f(Pi), S> 
          (trovare permutazione S che minimizza il prodotto scalare)
          Usato: Hungarian algorithm (O(n³) ma tipicamente O(n² log n) in pratica)
       
       c) AGGIORNA SOLUZIONE (Media mobile):
          Pi ← (1 - γ_t) * Pi + γ_t * S*
          Dove γ_t = 2/(t+2)  (learning rate decrescente)
       
       d) VERIFICA CONVERGENZA:
          Se ||Pi - S*|| < epsilon: break
    
    L'algoritmo converge verso un minimo locale (per problemi non convessi come il nostro,
    rimane locale ma è comunque una buona approssimazione).
    
    DIFFERENZA LINEARE vs QUADRATICA:
    ==================================
    
    Lineare:   Il gradiente è STATICO: grad_struct = μ * Lambda (calcolato una volta)
    Quadratica: Il gradiente CAMBIA: grad_struct = 2μ * ((P @ M_quad) - C_quad_const)
                Dipende da Pi attuale
    
    Implicazione:
    - Lineare: Sceglie la stessa direzione ad ogni step (più veloce, meno preciso)
    - Quadratica: Adatta la direzione in base al progresso (più lento, più preciso)
    
    Args:
        A1 (np.ndarray): Matrice di adiacenza grafo 1 (n×n)
        A2 (np.ndarray): Matrice di adiacenza grafo 2 (n×n)
        F1 (np.ndarray): Feature strutturali grafo 1 (n×3)
        F2 (np.ndarray): Feature strutturali grafo 2 (n×3)
        mu (float): Peso della regolarizzazione strutturale (0.5 = balance)
        max_iter (int): Numero d'iterazioni Frank-Wolfe (default=150)
        formulation (str): 'linear' o 'quadratic' (quale termine strutturale usare)
    
    Returns:
        tuple: (Pi, elapsed)
            Pi (np.ndarray): Matrice di allineamento ottimale (n×n stocastica)
            elapsed (float): Tempo totale di esecuzione in secondi
    
    Esempio di output:
        >>> A1 = nx.to_numpy_array(G1)
        >>> A2 = nx.to_numpy_array(G2)
        >>> F1 = get_graph_features(G1)
        >>> F2 = get_graph_features(G2)
        >>> Pi, time = solve_ged_fw(A1, A2, F1, F2, mu=0.5, max_iter=100, formulation='linear')
        >>> print(f"Esecuzione: {time:.4f}s")
        >>> print(f"Allineamento nodo 0 verso: {np.argmax(Pi[0, :])}")
    
    INTERPRETAZIONE DELLA SOLUZIONE Pi:
    ====================================
    Pi è una matrice stocastica (doppiamente stocastica per FGW idealmente):
    - Riga: pi(i, :) sono i pesi di allineamento del nodo i del grafo 1
    - Colonna: pi(:, j) sono i pesi di allineamento verso il nodo j del grafo 2
    
    Se Pi fosse una permutazione pura:
        pi[i, j] ∈ {0, 1} con un solo 1 per riga/colonna
        → allineamento one-to-one esatto
    
    Ma FGW ritorna un "flusso" stocastico:
        pi[i, j] ∈ [0, 1] indica probabilità/peso dell'allineamento
        → interpretabile come "soft matching"
    """
    n = A1.shape[0]
    
    # ========================================
    # PRECOMPUTAZIONI TERMINE GW (STATIC)
    # ========================================
    
    # V1[i] = sum_k A1[i,k]²  (norma L2 al quadrato della riga i)
    V1 = np.sum(A1**2, axis=1).reshape(-1, 1)  # shape: (n, 1)
    
    # V2[j] = sum_l A2[j,l]²  (norma L2 al quadrato della colonna j)
    V2 = np.sum(A2**2, axis=1).reshape(-1, 1)  # shape: (n, 1)
    
    # ones_col = [1, 1, ..., 1]^T  (vettore colonna di uni)
    ones_col = np.ones((n, 1))
    
    # INIZIALIZZA allineamento: coppia uniforme
    Pi = np.ones((n, n)) / n  # shape: (n, n), ogni elemento = 1/n
    
    # ========================================
    # PRECOMPUTAZIONI TERMINE STRUTTURALE
    # ========================================
    
    if formulation == 'linear':
        # FORMULAZIONE LINEARE: calcola Lambda UNA VOLTA
        #
        # Lambda[i,j] = ||F1[i]||² + ||F2[j]||² - 2*<F1[i], F2[j]>
        #             = distanza euclidea al quadrato
        #
        # Equivalente a: Lambda = (F1 @ F1^T) * ones - 2*(F1 @ F2^T) + ones' * (F2 @ F2^T)
        # Qui sfruttiamo NumPy broadcasting:
        
        Lambda = np.sum(F1**2, axis=1, keepdims=True) + \
                 np.sum(F2**2, axis=1) - \
                 2 * (F1 @ F2.T)
        # shape: (n, n) mediante broadcasting
        # (n,1) + (n,) - (n,n) = (n,n)
    
    else:  # formulation == 'quadratic'
        # FORMULAZIONE QUADRATICA: precomputa M e C che dipenderanno da P all'iterazione
        #
        # Gradiente strutturale quadratico:
        #   ∇f_quad = 2*μ*(P @ (F2 @ F2^T) - F1 @ F2^T)
        #
        # Precomputa i termini fissi:
        
        M_quad = F2 @ F2.T  # shape: (n, n) - covariance-like
        C_quad_const = F1 @ F2.T  # shape: (n, n) - cross-correlazione
    
    # ========================================
    # INIZIO DELL'ALGORITMO FRANK-WOLFE
    # ========================================
    
    start_time = time.time()  # Azzera il timer
    
    for t in range(max_iter):  # Loop principale
        
        # --------
        # STEP 1: CALCOLA GRADIENTE DEL TERMINE GW
        # --------
        # f_edge = tr(V1^T Pi) + tr(Pi V2^T) - 2*tr(A1 Pi A2^T Pi)
        # ∇f_edge = (V1 @ ones^T + ones @ V2^T) - 2(A1 @ Pi @ A2^T)
        
        term_A = (V1 @ ones_col.T) + (ones_col @ V2.T)
        # Risultato: (n, n) + (n, n) = (n, n)
        # term_A[i,j] ≈ V1[i] + V2[j]  (component-wise constant)
        
        term_B = 2 * (A1 @ Pi @ A2.T)
        # Moltiplicazione matriciale: (n,n) @ (n,n) @ (n,n) = (n,n)
        # Questa è la parte che dipende da Pi (adattiva)
        
        grad_edge = term_A - term_B
        # Gradiente del termine di struttura dei grafi
        
        # --------
        # STEP 2: CALCOLA GRADIENTE DEL TERMINE STRUTTURALE
        # --------
        
        if formulation == 'linear':
            # Lineare: il gradiente è semplicemente λ (statico)
            grad_struct = mu * Lambda
        
        else:  # quadratic
            # Quadratica: dipende da Pi attuale
            grad_struct = 2 * mu * ((Pi @ M_quad) - C_quad_const)
            # Calcolo: (n,n) @ (n,n) = (n,n) ✓
            # M_quad = F2 @ F2.T è (n,3) @ (3,n) = (n,n)
            # Pi @ M_quad = (n,n) @ (n,n) = (n,n)
        
        # --------
        # STEP 3: GRADIENTE TOTALE
        # --------
        
        full_grad = grad_edge + grad_struct
        # Matrice (n,n) di costi associati a ogni coppia (i,j)
        
        # --------
        # STEP 4: RISOLVI ASSIGNMENT PROBLEM (Hungarian Algorithm)
        # --------
        # Troviamo la permutazione S che minimizza <full_grad, S>
        # Sotto vincolo che S sia una matrice di permutazione (0-1, una 1 per riga/colonna)
        
        row_ind, col_ind = linear_sum_assignment(full_grad)
        # Ritorna: row_ind = [0,1,...,n-1], col_ind = permutazione ottimale
        # Tempo: O(n³) worst case, ma tipicamente O(n² log n)
        
        S = get_permutation_matrix((row_ind, col_ind), n)
        # S è una matrice di permutazione (0-1, one-hot)
        
        # --------
        # STEP 5: AGGIORNAMENTO FRANK-WOLFE (CONVEX COMBINATION)
        # --------
        # La regola FW combina la soluzione attuale con la nuova direzione:
        # Pi_new = (1 - γ) * Pi_old + γ * S
        # 
        # dove γ = 2/(t+2)  è il learning rate (decrescente nel tempo)
        # t=0: γ=1 (primo passo: Pi ← S)
        # t=1: γ=2/3
        # t=2: γ=1/2
        # ...
        # t→∞: γ→0 (aggiornamenti sempre più piccoli)
        
        gamma = 2.0 / (t + 2)  # formula standard di Frank-Wolfe
        Pi = (1 - gamma) * Pi + gamma * S
        
        # Nota: Pi rimane una distribuzione (elementi in [0,1], somma righe/colonne ~1)
        #       grazie alla convessità della combinazione
    
    elapsed = time.time() - start_time  # Tempo totale
    
    return Pi, elapsed

# ==========================================
# ESPERIMENTO 1: RUNTIME vs SIZE
# ==========================================

def experiment_runtime_vs_size():
    """
    ESPERIMENTO 1: Misura il tempo di esecuzione dei due formulation
    al variare della dimensione dei grafi.
    
    OBIETTIVO:
    ==========
    Capire come scalano i due algoritmi e se il Linear Surrogate offre
    un significativo beneficio computazionale senza perdere troppo in accuratezza.
    
    DESIGN SPERIMENTALE:
    ====================
    
    1. VARIA: Dimensione del grafo (n = 100, 200, 300, 400, 500 nodi)
    2. FISSO: Edit rate = 5% (n_edits = 0.05 * n)
    3. FISSO: Numero iterazioni = 50 (max_iter_fw = 50)
    4. FISSO: Regolarizzazione = 0.5 (μ = 0.5, balance entre edge e struct)
    5. MISURA: Tempo medio su n_pairs_per_size = 3 coppie per ogni size
    
    TOPOLOGIE TESTATE:
    ==================
    
    1. Barabási-Albert (scale-free):
       - Modella reti biologiche, sociali, web
       - Ha hubs (pochi nodi ad alto grado)
       - Feature strutturali: altamente variabili
    
    2. Erdős-Rényi (random):
       - Baseline omogenea
       - Gradi più uniformi
       - Feature strutturali: meno variabili
    
    RISULTATI ATTESI:
    =================
    
    Linear Surrogate: Sempre più veloce (5-15× più veloce per n=500)
    - Ragione: Il gradiente è statico (calcolato una volta)
    - Non dipende da Pi all'interno del loop
    
    Exact Quadratic: Più lento (ma teoricamente più accurato)
    - Ragione: Gradiente dipende da Pi, ricomputo ad ogni iterazione
    - Moltiplicazioni matriciali extra: (Pi @ M_quad)
    
    APPROSSIMAZIONE DELLA COMPLESSITÀ:
    ==================================
    Per ogni iterazione t:
    
    Linear:    ~O(n²) per Hungarian + O(n) per misc
    Quadratic: ~O(n²) per Hungarian + O(n³) per (Pi @ M_quad @ Pi^T)
    
    Quindi per max_iter iterazioni:
    Linear:    O(max_iter * n²)
    Quadratic: O(max_iter * n³)
    
    Differenza: ~n × volte più lento il quadratic per n grande
    """
    print("\n" + "="*80)
    print(" ESPERIMENTO 1: RUNTIME vs GRAPH SIZE")
    print("="*80)
    
    node_sizes = [100, 200, 300, 400, 500]  # Dimensioni da testare
    n_pairs_per_size = 3  # Numero di coppie per mdia
    max_iter_fw = 50  # Iterazioni dell'algoritmo
    mu = 0.5  # Regolarizzazione (balance)
    
    # Definisci le topologie da testare
    topologies = {
        'Barabási-Albert (Scale-Free)': {'name': 'barabasi_albert', 'kwargs': {'m_edges': 2}},
        'Erdős-Rényi (Random)': {'name': 'erdos_renyi', 'kwargs': {'p': 0.05}}
    }
    
    results_exp1 = {}  # Dizionario per raccogliere i risultati
    
    for topo_label, topo_config in topologies.items():
        print(f"\nProcessing {topo_label}...")
        topo_name = topo_config['name']
        topo_kwargs = topo_config['kwargs']
        
        times_lin = []  # Tempi Linear per ogni size
        times_quad = []  # Tempi Quadratic per ogni size
        
        for n in node_sizes:
            print(f"  Size n={n}...", end=" ", flush=True)
            t_lin_accum = 0.0
            t_quad_accum = 0.0
            n_edits = int(n * 0.05)  # 5% degli archi come edit
            
            # Esegui 3 volte per ogni size e fai media
            for i in range(n_pairs_per_size):
                # Genera coppia di grafi sintetici
                G1, G2 = generate_synthetic_pair(n, n_edits, topology=topo_name, **topo_kwargs)
                F1, F2 = get_graph_features(G1), get_graph_features(G2)
                A1, A2 = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
                
                # Esegui entrambe le formulazioni e cronometra
                _, time_lin = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation='linear')
                _, time_quad = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation='quadratic')
                
                t_lin_accum += time_lin
                t_quad_accum += time_quad
            
            # Media dei tempi
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
        
        # Plot entrambi i formulation su stesso grafico
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

# ==========================================
# EXPERIMENT A-C PLACEHOLDERS
# ==========================================
# Questi esperimenti verranno implementati negli script separati:
# - step4_train_gb_and_evaluate.py (Experiment A)
# - (Futuri: Experiment B e C)

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    """
    ENTRY POINT PRINCIPALE
    =======================
    
    Questo script esegue gli esperimenti di ablation per confrontare
    la formulazione lineare vs quadratica dell'algoritmo FGW.
    
    ESPERIMENTI:
    - EXPERIMENT 1: Runtime vs Graph Size (speed-up)
    - EXPERIMENT A-C: Quality via GradientBoosting (negli script separati)
    
    COSA ACCADE:
    ============
    1. Esegui Esperimento 1: misura il tempo di esecuzione
       al variare della dimensione dei grafi (100-500 nodi)
       
       Output: Grafico con 2 subplot (BA e ER), che mostra runtime
       come funzione di n per entrambi i formulation
       
       File salvato: figures/experiment1_runtime_vs_size.png
    
    NOTE SUI NUOVI ESPERIMENTI (A, B, C):
    ====================================
    Gli esperimenti di qualità che sfruttano GradientBoosting
    sono implementati negli script separati:
    
    - step4_train_gb_and_evaluate.py: EXPERIMENT A
    - (Future: Experiment B per griglia completa p,q)
    - (Future: Experiment C per topologie multiple)
    
    Questi permettono di misurare la capacità dei due algoritmi
    di predire il vero GED tramite regressione ML.
    
    INTERPRETAZIONE ESPERIMENTO 1:
    ==============================
    - Linear dovrebbe essere notevolmente più veloce (5-15× a n=500)
    - Quadratic probabilmente cresce più rapidamente con n
    - Per grafi grandi: Linear conviene almeno dal lato computazionale
    """
    print("\n" + "="*80)
    print(" ABLATION STUDY: LINEAR SURROGATE vs EXACT QUADRATIC")
    print("="*80)
    
    # Esegui EXPERIMENT 1 (runtime)
    experiment_runtime_vs_size()
    
    print("\n" + "="*80)
    print(" COMPLETED")
    print("="*80)
