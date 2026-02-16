"""
Calcolo della Graph Edit Distance usando la Teoria del Trasporto Ottimale
==========================================================================

IDEA CENTRALE DEL METODO:
--------------------------
Questo modulo implementa un'approssimazione innovativa della Graph Edit Distance (GED)
che combina la distanza di Fused Gromov-Wasserstein (FGW) con feature strutturali dei grafi.

PROBLEMA:
---------
Calcolare la GED esatta tra due grafi è NP-difficile. Per grafi anche di dimensioni moderate,
il calcolo esatto diventa computazionalmente proibitivo.

SOLUZIONE PROPOSTA:
-------------------
Usiamo la distanza di Fused Gromov-Wasserstein, che:
1. Confronta la STRUTTURA dei grafi (attraverso le matrici di adiacenza)
2. Confronta le CARATTERISTICHE dei nodi (etichette + centralità)
3. Trova un "trasporto ottimale" che allinea i nodi dei due grafi minimizzando
   una combinazione pesata di costi strutturali e di caratteristiche

INNOVAZIONE:
------------
La regolarizzazione con feature strutturali (degree, PageRank, clustering, betweenness)
migliora significativamente l'accuratezza dell'approssimazione, catturando proprietà
topologiche importanti che non sono evidenti solo dalle matrici di adiacenza.

FUNZIONI CHIAVE:
----------------
- compute_ged_GW: Calcola l'approssimazione GED usando la distanza FGW
- compute_structural_features: Estrae misure di centralità dai grafi
- extract_ot_features: Estrae feature informative dalla matrice di coupling
- compare_and_swap_graphs: Normalizza le dimensioni dei grafi per il confronto

PIPELINE TIPICA:
----------------
1. Normalizza i grafi (stessa dimensione)
2. Calcola distanze tra etichette dei nodi
3. Calcola distanze tra feature strutturali
4. Combina le distanze con peso μ
5. Applica FGW per ottenere l'approssimazione GED
"""

import os
import json
import networkx as nx
import numpy as np
import pandas as pd
from time import time
import random

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils import label_dict_construction
from ot.gromov import (
    fused_gromov_wasserstein,
    entropic_fused_gromov_wasserstein,
    BAPG_fused_gromov_wasserstein,
)



def get_graph_by_id(graphs_df, graph_id):
    """
    Recupera un grafo da un dataframe tramite il suo ID.
    
    SCOPO:
    ------
    Fornisce un accesso efficiente ai grafi memorizzati nel DataFrame CSV.
    Evita di dover caricare tutti i grafi in memoria contemporaneamente.
    
    Args:
        graphs_df (pd.DataFrame): DataFrame con i dati dei grafi, deve contenere:
                                   - 'graph_id': ID univoco del grafo
                                   - 'graph_edge_list': lista di archi in formato JSON
        graph_id (int): L'ID del grafo da recuperare
    
    Returns:
        nx.Graph: Il grafo richiesto come oggetto NetworkX
        None: Se il grafo non viene trovato nel DataFrame
    
    Esempio:
        >>> graphs_df = pd.read_csv('Dataset/AIDS/AIDS_graphs.csv')
        >>> G = get_graph_by_id(graphs_df, 42)
        >>> if G is not None:
        >>>     print(f"Il grafo 42 ha {G.number_of_nodes()} nodi")
    
    Note:
        - Il chiamante deve gestire il valore None se il grafo non esiste
        - La lista di archi viene parsata da JSON e convertita in grafo NetworkX
    """
    graph_row = graphs_df[graphs_df['graph_id'] == graph_id]
    if graph_row.empty:
        # Grafo non trovato - il chiamante deve gestire il valore None
        return None
    
    # Parsa la lista di archi dal JSON e costruisce il grafo NetworkX
    edge_list = json.loads(graph_row.iloc[0]['graph_edge_list'])
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G
    


def compare_and_swap_graphs(G_S, G_T, labels_S=None, labels_T=None):
    """
    Normalizza due grafi per avere lo stesso numero di nodi aggiungendo nodi dummy.
    Assicura che il grafo più piccolo sia sempre G_S e quello più grande sia G_T.
    
    PROBLEMA DA RISOLVERE:
    ----------------------
    Gli algoritmi di trasporto ottimale (come Gromov-Wasserstein) richiedono che
    i due grafi abbiano lo stesso numero di nodi. Quando confrontiamo grafi di
    dimensioni diverse, dobbiamo "pareggiare" le dimensioni.
    
    SOLUZIONE:
    ----------
    1. Se G_S ha più nodi di G_T, li scambiamo (convenzione: S = small, T = target)
    2. Aggiungiamo nodi "dummy" isolati (senza archi) al grafo più piccolo
    3. Se presenti, aggiungiamo anche etichette dummy ("?") per i nodi aggiunti
    
    INTERPRETAZIONE:
    ----------------
    I nodi dummy rappresentano nodi "inesistenti" che devono essere eliminati
    per trasformare un grafo nell'altro. Il loro costo di allineamento sarà alto,
    contribuendo correttamente al calcolo della GED.
    
    Args:
        G_S (nx.Graph): Grafo sorgente
        G_T (nx.Graph): Grafo target
        labels_S (list, optional): Etichette dei nodi per il grafo sorgente
        labels_T (list, optional): Etichette dei nodi per il grafo target
    
    Returns:
        tuple: (G_S, G_T) se non ci sono etichette
               (G_S, G_T, labels_S, labels_T) se le etichette sono fornite
               G_S avrà nodi <= G_T, con nodi dummy aggiunti se necessario
    
    Esempio:
        >>> # G1 ha 5 nodi, G2 ha 8 nodi
        >>> G1_norm, G2_norm = compare_and_swap_graphs(G1, G2)
        >>> print(G1_norm.number_of_nodes())  # 8 (3 nodi dummy aggiunti)
        >>> print(G2_norm.number_of_nodes())  # 8
    """
    # Scambia se il sorgente è più grande del target
    if G_S.number_of_nodes() > G_T.number_of_nodes():
        G_S, G_T = G_T, G_S
        if labels_S is not None and labels_T is not None:
            labels_S, labels_T = labels_T, labels_S
    
    # Aggiungi nodi dummy al grafo più piccolo per uguagliare le dimensioni
    existing_nodes = set(G_S.nodes())
    next_dummy_id = max(existing_nodes, default=-1) + 1 if existing_nodes else 0

    while G_S.number_of_nodes() < G_T.number_of_nodes():
        dummy_node = next_dummy_id
        G_S.add_node(dummy_node)  # Nodo dummy isolato senza archi
        next_dummy_id += 1
    
    # Aggiungi etichette dummy se fornite
    if labels_S is not None and labels_T is not None:
        while len(labels_S) < len(labels_T):
            labels_S.append("?")  # Etichetta dummy per i nodi aggiunti
        return G_S, G_T, labels_S, labels_T
    else:
        return G_S, G_T
    
def compute_label_distance(labels_S, labels_T, number_of_nodes):
    """
    Calcola la matrice di distanza tra le etichette dei nodi.
    
    CONTESTO:
    ---------
    Per calcolare la GED, dobbiamo sapere quanto "costa" abbinare un nodo di un grafo
    con un nodo dell'altro grafo. Se i nodi hanno etichette (es. tipi di atomo nel
    dataset AIDS), il costo dipende dalla compatibilità delle etichette.
    
    LOGICA:
    -------
    - Se le etichette sono identiche (es. entrambi Carbonio): costo = 0
    - Se le etichette sono diverse (es. Carbonio vs Ossigeno): costo = 2
    - Se i grafi non hanno etichette: costo uniforme = 1 per tutte le coppie
    
    PERCHÉ COSTO 2 PER MISMATCH?
    -----------------------------
    Una sostituzione di etichetta equivale concettualmente a:
        eliminare un nodo + inserire un nodo nuovo = 2 operazioni
    
    Args:
        labels_S (list or None): Etichette per i nodi del grafo sorgente
        labels_T (list or None): Etichette per i nodi del grafo target
        number_of_nodes (int): Numero di nodi in ciascun grafo (devono essere uguali)
    
    Returns:
        np.ndarray: Matrice di distanze di forma (number_of_nodes, number_of_nodes)
                    Elemento [i,j] = costo di abbinare il nodo i con il nodo j
    
    Raises:
        ValueError: Se le liste di etichette hanno lunghezze diverse
    
    Esempio:
        >>> labels_1 = ['C', 'O', 'N']  # Carbonio, Ossigeno, Azoto
        >>> labels_2 = ['C', 'C', 'O']
        >>> dist = compute_label_distance(labels_1, labels_2, 3)
        >>> print(dist)
        [[0. 0. 2.]    # C vs C=0, C vs C=0, C vs O=2
         [2. 2. 0.]    # O vs C=2, O vs C=2, O vs O=0
         [2. 2. 2.]]   # N vs C=2, N vs C=2, N vs O=2
    """
    # Per grafi non etichettati, usa distanza uniforme
    if labels_S is None and labels_T is None:
        return np.ones((number_of_nodes, number_of_nodes))
    
    # Valida la lunghezza delle etichette
    if len(labels_S) != len(labels_T):
        raise ValueError(f"labels_S e labels_T devono avere la stessa lunghezza. "
                        f"Ricevuto {len(labels_S)} e {len(labels_T)}.")
    
    # Calcola le distanze tra coppie di etichette (0 se uguali, 2 se diverse)
    label_distance_matrix = np.zeros((len(labels_S), len(labels_T)))
    for i, label_S in enumerate(labels_S):
        for j, label_T in enumerate(labels_T):
            label_distance_matrix[i][j] = 0.0 if label_S == label_T else 2.0
    
    return label_distance_matrix

    
def compute_structural_features(graph, list_of_centrality_indices=None):
    """
    Calcola le feature strutturali (misure di centralità) per i nodi del grafo.
    
    IDEA CHIAVE:
    ------------
    Due nodi "simili" in grafi diversi dovrebbero avere ruoli simili nella loro
    rete. Le misure di centralità quantificano l'importanza/ruolo dei nodi.
    
    FEATURE DISPONIBILI:
    --------------------
    - 'Deg': Degree centrality (quante connessioni ha il nodo?)
    - 'PR': PageRank centrality (quanto è "importante" il nodo nella rete?)
    - 'CC': Clustering coefficient (quanto i vicini del nodo sono connessi tra loro?)
    - 'Betw': Betweenness centrality (quanto spesso il nodo si trova sui cammini minimi?)
    
    INTERPRETAZIONE:
    ----------------
    Queste misure catturano proprietà complementari:
    - Deg: popolarità locale
    - PR: importanza globale
    - CC: tendenza a formare cluster
    - Betw: ruolo di "ponte" nella rete
    
    Args:
        graph (nx.Graph): Grafo di input
        list_of_centrality_indices (list, optional): Lista delle feature da calcolare.
                                                      Se None, calcola tutte e quattro.
    
    Returns:
        np.ndarray: Matrice di feature di forma (n_nodi, n_feature)
                    Ogni riga rappresenta un nodo, ogni colonna una feature
    
    Raises:
        ValueError: Se vengono forniti nomi di indici di centralità non validi
    
    Esempio:
        >>> G = nx.karate_club_graph()
        >>> features = compute_structural_features(G, ['Deg', 'PR', 'CC'])
        >>> print(features.shape)  # (34, 3) - 34 nodi, 3 feature
        >>> print(features[0])     # Feature del nodo 0
        [0.515  0.096  0.150]      # Deg=0.515, PR=0.096, CC=0.150
    
    Note:
        - Betweenness usa un algoritmo approssimato (basato su campionamento) per efficienza
        - Dimensione campione k = min(100, n_nodi) offre un buon bilanciamento tra
          accuratezza e velocità su grafi di grandi dimensioni
        - Tutte le misure sono normalizzate tra 0 e 1
    """
    if list_of_centrality_indices is None:
        # Comportamento di default: calcola tutte e quattro le feature
        degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
        pagerank_centrality = np.array(list(nx.pagerank(graph).values()))
        clustering_coefficient = np.array(list(nx.clustering(graph).values()))
        
        # Approssima la betweenness per efficienza
        n_nodes = graph.number_of_nodes()
        k_sample = min(100, n_nodes)  # Dimensione campione per l'approssimazione
        betweenness_centrality = np.array(list(
            nx.betweenness_centrality(graph, k=k_sample, normalized=True).values()
        ))
        
        # Impila le feature come colonne (righe = nodi, colonne = feature)
        return np.vstack((degree_centrality, pagerank_centrality, 
                         clustering_coefficient, betweenness_centrality)).T
    else:
        # Valida la lista di input
        valid_indices = {'Deg', 'PR', 'CC', 'Betw'}
        if not set(list_of_centrality_indices).issubset(valid_indices):
            raise ValueError(f"list_of_centrality_indices può contenere solo 'Deg', 'PR', 'CC', e 'Betw'. "
                           f"Ricevuto: {list_of_centrality_indices}")
        
        features_list = []
        
        # Calcola solo le feature richieste
        if 'Deg' in list_of_centrality_indices:
            degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
            features_list.append(degree_centrality)
        
        if 'PR' in list_of_centrality_indices:
            pagerank_centrality = np.array(list(nx.pagerank(graph).values()))
            features_list.append(pagerank_centrality)
        
        if 'CC' in list_of_centrality_indices:
            clustering_coefficient = np.array(list(nx.clustering(graph).values()))
            features_list.append(clustering_coefficient)
        
        if 'Betw' in list_of_centrality_indices:
            # Approssima la betweenness per efficienza
            n_nodes = graph.number_of_nodes()
            k_sample = min(100, n_nodes)  # Dimensione campione per l'approssimazione
            betweenness_centrality = np.array(list(
                nx.betweenness_centrality(graph, k=k_sample, normalized=True).values()
            ))
            features_list.append(betweenness_centrality)
        
        # Impila le feature come colonne
        return np.vstack(features_list).T


def compute_cross_matrix_with_structural_features(G1, G2, list_of_centrality_indices=None):
    """
    Calcola la matrice di distanze tra le feature strutturali di due grafi.
    
    MATRICE DI COSTO DEI FEATURE:
    -----------------------------
    Questo è uno dei due componenti principali della nostra formulazione FGW:
    - Esso misura quanto i nodi sono diversi in termini di caratteristiche strutturali
    - Complementa la misura di distanza tra etichette
    
    FORMULA UTILIZZATA (efficiente):
    --------------------------------
    Anziché calcolare ||S1[i] - S2[j]||² = Σ(S1[i][k] - S2[j][k])²  per ogni coppia,
    usiamo la formula:
        ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
    
    Questo è molto più veloce usando operazioni matriciali:
        ||S1[i] - S2[j]||² = ||S1[i]||² + ||S2[j]||² - 2 * S1 @ S2.T
    
    COMPLESSITÀ:
    -----------
    Tempo: O(N1 * N2 * D) dove D = numero di feature
    Invece di O(N1 * N2 * D) con il calcolo naive
    
    Args:
        G1 (nx.Graph): Primo grafo
        G2 (nx.Graph): Secondo grafo
        list_of_centrality_indices (list, optional): Quali feature usare
    
    Returns:
        np.ndarray: Matrice di distanze di forma (N1, N2) dove N1, N2 sono i numeri
                    di nodi. Elemento [i,j] = distanza euclidea tra i vettori di
                    feature del nodo i in G1 e del nodo j in G2
    
    Raises:
        ValueError: Se le dimensioni dei feature non corrispondono tra i due grafi
    
    Esempio:
        >>> G1 = nx.karate_club_graph()
        >>> G2 = nx.complete_graph(5)
        >>> dist_matrix = compute_cross_matrix_with_structural_features(G1, G2, ['Deg', 'PR'])
        >>> print(dist_matrix.shape)  # (34, 5)
    """
    N1 = G1.number_of_nodes()
    N2 = G2.number_of_nodes()
    
    # Estrai le feature strutturali da entrambi i grafi
    S1 = compute_structural_features(G1, list_of_centrality_indices)
    S2 = compute_structural_features(G2, list_of_centrality_indices)
    
    # Valida che le dimensioni dei feature coincidono
    if S1.shape[1] != S2.shape[1]:
        raise ValueError(f"Le feature strutturali di G1 e G2 devono avere lo stesso numero di colonne. "
                        f"Ricevuto {S1.shape[1]} e {S2.shape[1]}.")
    
    # Calcola le distanze euclidee per coppie usando operazioni matriciali efficienti
    # ||s1[i] - s2[j]||² = ||s1[i]||² + ||s2[j]||² - 2⟨s1[i], s2[j]⟩
    structural_cross_matrix = (
        np.sum(S1**2, axis=1, keepdims=True).dot(np.ones((1, N2))) +
        np.ones((N1, 1)).dot(np.sum(S2**2, axis=1, keepdims=True).T) -
        2 * S1.dot(S2.T)
    )
    
    return structural_cross_matrix


    
def calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu, include_structural_features):
    """
    Combina le distanze tra etichette e le distanze tra feature strutturali.
    
    FUSIONE BILANCIATA:
    -------------------
    La nostra innovazione principale è combinare questi due tipi di informazione:
    1. Compatibilità delle etichette (proprietà locali/chimiche)
    2. Compatibilità strutturale (ruolo nella topologia)
    
    FORMULA:
    --------
    M = D_label + μ × D_structural
    
    Dove:
        - D_label: matrice di distanza tra etichette
        - D_structural: matrice di distanza tra feature strutturali
        - μ: parametro di regolarizzazione che controlla il peso relativo
    
    EFFETTO DI μ:
    ---------------
    - μ = 0: usa solo le etichette (GW classico)
    - μ = 0.5: bilancia etichette e struttura (valore tipico)
    - μ = 1.0: peso uguale
    - μ > 1: enfatizza la struttura rispetto alle etichette
    
    INTERPRETAZIONE FISICA:
    -----------------------
    Valori alti di μ significano che vogliamo abbinare nodi che svolgono
    ruoli simili nella rete, indipendentemente dalle loro etichette.
    Valori bassi enfatizzano la compatibilità chimica/label.
    
    Args:
        label_distance_matrix (np.ndarray): Distanze tra le etichette dei nodi
        structural_cross_matrix (np.ndarray): Distanze tra le feature strutturali
        mu (float): Parametro di regolarizzazione (non-negativo). Valori tipici: 0-1.
                    Higher values emphasize structure over labels.
        include_structural_features (bool): Se False, ignora le feature strutturali
                                            e ritorna solo label_distance_matrix
    
    Returns:
        np.ndarray: Matrice di costo combinata per il trasporto ottimale
    
    Raises:
        ValueError: Se μ è negativo
    
    Esempio:
        >>> # Caso 1: solo etichette (μ non usato)
        >>> M = calculate_cross_matrix(D_label, D_struct, mu=0.5, include_structural_features=False)
        >>> # M == D_label
        >>> 
        >>> # Caso 2: combina etichette e struttura con μ=0.5
        >>> M = calculate_cross_matrix(D_label, D_struct, mu=0.5, include_structural_features=True)
        >>> # M == D_label + 0.5 * D_struct
    """
    if not include_structural_features:
        return label_distance_matrix
    
    # Valida il parametro μ
    if mu < 0:
        raise ValueError(f"μ (mu) deve essere non-negativo. Ricevuto {mu}")
    
    return label_distance_matrix + mu * structural_cross_matrix



def support_vector_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Support Vector Regression models with different kernels.
    
    Tests three kernel types: RBF, Linear, and Polynomial.
    Feature scaling is applied to both X and y for better SVR performance.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features (GW scores)
        y_train (pd.Series or np.ndarray): Training targets (true GED)
        X_test (pd.DataFrame or np.ndarray): Test features
        y_test (pd.Series or np.ndarray): Test targets
    
    Prints:
        Performance metrics for each kernel:
        - MAE: Mean Absolute Error
        - Accuracy: Percentage of predictions within ±1 of true value
        - Spearman correlation: Rank correlation coefficient
        - Kendall's tau: Another rank correlation measure
    """
    # Feature scaling is critical for SVR performance
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale features and targets
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    # Test different SVR kernels
    for kernel_type in ['rbf', 'linear', 'poly']:
        print(f"Using kernel: {kernel_type}")
        
        # Train SVR model
        # C=1.0: regularization parameter
        # epsilon=0.1: epsilon-tube width for epsilon-insensitive loss
        model = SVR(kernel=kernel_type, C=1.0, epsilon=0.1)
        model.fit(X_train_scaled, y_train_scaled)
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        mae = mean_absolute_error(y_test_scaled, y_pred)
        print(f"SVR Mean Absolute Error: {mae:.4f}")
        
        # Accuracy: predictions within ±1 edit distance
        accuracy = np.mean(np.abs(np.round(y_pred) - y_test_scaled) <= 1) * 100
        print(f"Accuracy (±1 edit distance): {accuracy:.2f}%")
        
        # Rank correlation measures (assess monotonic relationship)
        spearman_corr = pd.Series(y_test_scaled).corr(pd.Series(y_pred), method='spearman')
        print(f"Spearman's Correlation Coefficient: {spearman_corr:.4f}")

        kendall_tau = pd.Series(y_test_scaled).corr(pd.Series(y_pred), method='kendall')
        print(f"Kendall's Tau Coefficient: {kendall_tau:.4f}")
        print(10 * '*')


def compute_ged_GW(G1, G2, cross_matrix):
    """
    Calcola l'approssimazione della Graph Edit Distance usando la distanza FGW.
    
    ALGORITMO CENTRALE:
    ====================
    
    Questo è il cuore dell'intera metodologia. Usa la teoria del trasporto ottimale
    (Optimal Transport, OT) per trovare l'allineamento ottimale tra i nodi dei due grafi.
    
    FUSED GROMOV-WASSERSTEIN (FGW):
    --------------------------------
    FGW è una generalizzazione della distanza di Gromov-Wasserstein che combina:
    
    1. TRASPORTO NEI FEATURE (componente "fusa"):
       Costo del trasporto della massa tra i nodi sulla base delle loro caratteristiche
       → Controllato da cross_matrix (contiene etichette + feature strutturali)
    
    2. DISTANZA STRUTTURALE (componente Gromov-Wasserstein):
       Mette in penalità allineamenti che distruggono la struttura del grafo
       → Controllato da C1 e C2 (matrici di adiacenza)
    
    3. BILANCIAMENTO (parametro alpha):
       α = 0.5 significa che diamo peso UGUALE ai feature e alla struttura
       - α = 1.0: solo feature (diventa Trasporto di Wasserstein classico)
       - α = 0.0: solo struttura (Gromov-Wasserstein puro)
    
    FORMULAZIONE MATEMATICA:
    -------------------------
    FGW_α(C1, C2, M) = min_{π ∈ Π} { 
        (1-α) * <π, M> + α * GW(C1, C2, π)
    }
    
    Dove:
        - π è la matrice di coupling (probabilità di accoppiare nodo i con nodo j)
        - M è la nostra matrice di costo (cross_matrix)
        - GW è la distanza di Gromov-Wasserstein
        - <·,·> è il prodotto scalare
    
    INTERPRETAZIONE DEL RISULTATO:
    --------------------------------
    Il valore FGW è una misura di quanto costa trasformare G1 in G2.
    Valori bassi = grafi simili
    Valori alti = grafi diversi
    
    Args:
        G1 (nx.Graph): Primo grafo
        G2 (nx.Graph): Secondo grafo
        cross_matrix (np.ndarray): Matrice di costo dei feature (etichette + struttura)
    
    Returns:
        tuple: (fgw_distance, coupling_matrix)
            - fgw_distance (float): L'approssimazione FGW della GED
            - coupling_matrix (np.ndarray): Matrice di coupling ottimale
    
    Raises:
        ValueError: Se i grafi non hanno lo stesso numero di nodi
    
    Note:
        - Graphs must have equal node counts (use compare_and_swap_graphs first)
        - Uses uniform distributions over nodes (h1, h2) because we care equally
          about all nodes
        - alpha=0.5 balances equally between features and structure
        - tol_rel=1e-9: tight convergence tolerance for accuracy
    
    Esempio:
        >>> G1 = nx.karate_club_graph()
        >>> G2 = nx.complete_graph(G1.number_of_nodes())
        >>> cross_matrix = np.ones((G1.number_of_nodes(), G2.number_of_nodes()))
        >>> fgw_dist, coupling = compute_ged_GW(G1, G2, cross_matrix)
        >>> print(f"GED approximation: {fgw_dist:.4f}")
    """
    # Valida che i grafi abbiano lo stesso numero di nodi
    if G1.number_of_nodes() != G2.number_of_nodes():
        raise ValueError(f"G1 e G2 devono avere lo stesso numero di nodi. "
                        f"Ricevuto {G1.number_of_nodes()} e {G2.number_of_nodes()}.")
    
    # Estrai le matrici di adiacenza (informazione strutturale)
    C1 = nx.to_numpy_array(G1)
    C2 = nx.to_numpy_array(G2)
    
    # Usa distribuzioni uniformi sui nodi
    # Questo significa che diamo lo stesso peso a tutti i nodi
    h1 = np.ones(G1.number_of_nodes()) / G1.number_of_nodes()
    h2 = np.ones(G2.number_of_nodes()) / G2.number_of_nodes()

    # Calcola la distanza di Fused Gromov-Wasserstein
    coupling, log_cg = fused_gromov_wasserstein(
        cross_matrix, C1, C2, h1, h2, 
        loss="square_loss",  # Perdita quadratica per il confronto strutturale
        alpha=0.5,           # Bilancia equamente tra feature e struttura
        tol_rel=1e-9,        # Tolleranza di convergenza stretta per accuratezza
        verbose=False, 
        log=True
    )
    
    return log_cg['fgw_dist'], coupling

def extract_ot_features(coupling, cross_matrix, C1, C2):
    """
    Estrae feature informative dalla matrice di coupling del trasporto ottimale.
    
    MOTIVAZIONE:
    =============
    La matrice di coupling π contiene ricche informazioni sull'allineamento ottimale
    tra i nodi che vanno OLTRE al semplice valore della distanza FGW.
    
    Interpretazione della matrice coupling:
        π[i,j] = probabilità/massa di accoppiare il nodo i di G1 con il nodo j di G2
        
    Analizzando questi valori, possiamo misurare:
    - Quanto è "sicuro" l'allineamento proposto
    - Quanti nodi hanno allineamenti ambigui
    - Quanto la struttura è preservata
    - Come i costi si distribuiscono
    
    FEATURE ESTRATTE:
    =================
    
    1. alignment_entropy: Shannon entropy della matrice coupling
       - Misura l'incertezza nell'allineamento
       - Bassa entropia = allineamento chiaro e univoco
       - Alta entropia = molte scelte di allineamento igualmente buone
       - Formula: -Σ π[i,j] * log(π[i,j])
    
    2. alignment_confidence: Media dei massimi per riga di π
       - Per ogni nodo in G1, quanto è sicuro il match migliore?
       - Valori alti (vicino a 1) = corrispondenze forti
       - Valori bassi = molte opzioni equivalenti
       - Formula: mean(max_j π[i,j])
    
    3. transport_cost: Costo totale del trasporto
       - Somma pesata delle distanze delle feature
       - Diverso da FGW che include il termine strutturale
       - Formula: Σ π[i,j] * cross_matrix[i,j]
    
    4. marginal_balance: Quanto il coupling rispetta i margini uniformi
       - Perfetto se ogni nodo ha probabilità 1/n di matching
       - Misura lo "squilibrio" dell'allineamento
       - Valori bassi = allineamento ben bilanciato
       - Formula: ||Σ_j π[i,j] - 1/n||_1 + ||Σ_i π[i,j] - 1/m||_1
    
    5. coupling_sparsity: Frazione di elementi quasi-zero nel coupling
       - Sparse = chiaro matching di tipo uno-a-uno
       - Denso = molti soft-matching
       - Formula: count(π < 1e-3) / total_elements
    
    6. max_coupling: Valore massimo nella matrice coupling
       - Esiste almeno un match molto forte?
       - Valori alti = matching forte
       - Valori bassi = matching ambiguo
    
    7. coupling_variance: Varianza dei valori nella matrice coupling
       - Distribuzione uniforme vs concentrata
       - Alta varianza = alcuni match molto forti, altri deboli
       - Bassa varianza = matching uniforme
    
    8. structural_mismatch: Differenza nei gradi dei nodi abbinati
       - Nodi con gradi simili dovrebbero essere abbinati
       - Misura quanto i match violano questa intuizione
       - Formula: Σ π[i,j] * |degree[i] - degree[j]| / Σ π[i,j]
    
    UTILIZZO:
    =========
    Queste 8 feature catturano aspetti complementari della qualità dell'allineamento
    e possono essere usate in un modello di machine learning per:
    - Prevedere il GED vero (migration da soli feature basati su FGW)
    - Fornire fiducia nelle previsioni
    - Aiutare nella selezione del modello
    
    Args:
        coupling (np.ndarray): Matrice di coupling ottimale di forma (n x m)
        cross_matrix (np.ndarray): Matrice di costo dei feature (n x m)
        C1 (np.ndarray): Matrice di adiacenza del primo grafo (n x n)
        C2 (np.ndarray): Matrice di adiacenza del secondo grafo (m x m)
    
    Returns:
        dict: Dizionario con 8 feature OT-derived:
    
    Esempio di output:
        {
            'ot_alignment_entropy': 1.234,
            'ot_alignment_confidence': 0.856,
            'ot_transport_cost': 45.6,
            'ot_marginal_balance': 0.023,
            'ot_coupling_sparsity': 0.782,
            'ot_max_coupling': 0.924,
            'ot_coupling_variance': 0.156,
            'ot_structural_mismatch': 1.234
        }
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    
    # 1. Alignment Entropy - measures uncertainty in node matching
    # High entropy = many possible matches, low entropy = clear matching
    non_zero_coupling = coupling[coupling > eps]
    if len(non_zero_coupling) > 0:
        alignment_entropy = -np.sum(non_zero_coupling * np.log(non_zero_coupling + eps))
    else:
        alignment_entropy = 0.0
    
    # 2. Alignment Confidence - how confidently are nodes matched?
    # For each source node, what's the max probability of matching?
    alignment_confidence = np.max(coupling, axis=1).mean()
    
    # 3. Transport Cost - actual cost of transporting mass
    # This is different from FGW distance which includes structure
    transport_cost = np.sum(coupling * cross_matrix)
    
    # 4. Marginal Balance - how well do row/column sums match uniform?
    # Perfect coupling should have uniform marginals (1/n for each node)
    n, m = coupling.shape
    target_row_marginal = np.ones(n) / n
    target_col_marginal = np.ones(m) / m
    row_marginal = coupling.sum(axis=1)
    col_marginal = coupling.sum(axis=0)
    marginal_balance = (
        np.abs(row_marginal - target_row_marginal).sum() +
        np.abs(col_marginal - target_col_marginal).sum()
    ) / 2.0
    
    # 5. Coupling Sparsity - what fraction of entries are near-zero?
    # Sparse coupling = clear node-to-node matching
    sparsity_threshold = 1e-3
    coupling_sparsity = (coupling < sparsity_threshold).sum() / coupling.size
    
    # 6. Max Coupling - is there at least one strong match?
    max_coupling = np.max(coupling)
    
    # 7. Coupling Variance - are coupling values uniform or peaked?
    coupling_variance = np.var(coupling)
    
    # 8. Structural Mismatch - compare matched node neighborhoods
    # For each matched pair, compare their degrees/connectivity (vectorized)
    degrees_1 = C1.sum(axis=1)  # shape: (n,)
    degrees_2 = C2.sum(axis=1)  # shape: (m,)
    degree_diff_matrix = np.abs(degrees_1[:, np.newaxis] - degrees_2[np.newaxis, :])  # shape: (n, m)
    
    # Mask by coupling > eps for numerical stability
    mask = coupling > eps
    weighted_diff = np.sum(coupling[mask] * degree_diff_matrix[mask])
    total_mass = np.sum(coupling[mask])
    
    structural_mismatch = weighted_diff / total_mass if total_mass > 0 else 0.0
    
    return {
        'ot_alignment_entropy': alignment_entropy,
        'ot_alignment_confidence': alignment_confidence,
        'ot_transport_cost': transport_cost,
        'ot_marginal_balance': marginal_balance,
        'ot_coupling_sparsity': coupling_sparsity,
        'ot_max_coupling': max_coupling,
        'ot_coupling_variance': coupling_variance,
        'ot_structural_mismatch': structural_mismatch
    }

def GED_prediction_regression(file_name):
    """
    Esegue la previsione della GED utilizzando la Regressione mediante Support Vector Machine.
    
    SCOPO:
    ------
    Questo è il passaggio finale della pipeline di apprendimento:
    1. Carichiamo i punteggi FGW approssimati da un file CSV
    2. Addestriamo un modello SVR per imparare la mappatura: FGW_Score → True_GED
    3. Valutiamo le prestazioni del modello
    
    IDEA PRINCIPALE:
    ----------------
    La distanza FGW non è una stima perfetta della GED vera. Il modello SVR
    impara a correggere sistematicamente i bias della nostra approssimazione
    usando i dati di training.
    
    PIPELINE DI APPRENDIMENTO:
    --------------------------
    CSV Input → SVR Training → SVR Testing → Metriche di Valutazione
    
    Il file CSV deve contenere:
        - GW_Score: l'approssimazione FGW che abbiamo calcolato
        - True_GED: il valore vero da un dataset di benchmark (TaGED)
    
    Args:
        file_name (str): Percorso del file CSV con le colonne 'GW_Score' e 'True_GED'
    
    Returns:
        None (stampa i risultati sulla console)
    
    Output Stampato:
        Per ogni kernel SVR (rbf, linear, poly):
        - SVR Mean Absolute Error: quanto è sbagliato in media
        - Accuracy (±1 edit distance): percentuale di previsioni entro ±1
        - Spearman's Correlation: correlazione di rango
        - Kendall's Tau: altra misura di correlazione di rango
    
    Note:
        - Il dataset viene diviso 80% training / 20% test
        - Lo scaling delle feature è critico per le performance di SVR
        - La correlazione di rango è importante per GED (ordine è più critico che valore esatto)
    """
    # Load the data
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return
    
    # Validate required columns exist
    if 'GW_Score' not in df.columns or 'True_GED' not in df.columns:
        print(f"Error: File must contain 'GW_Score' and 'True_GED' columns.")
        return
    
    # Split the data into training and test sets (80% training, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare features and targets
    X_train = train_df[['GW_Score']]
    y_train = train_df['True_GED']
    X_test = test_df[['GW_Score']]
    y_test = test_df['True_GED']

    # Train and evaluate SVR models
    support_vector_regression(X_train, y_train, X_test, y_test)
