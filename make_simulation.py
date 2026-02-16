"""
Simulazione e Valutazione della Graph Edit Distance
====================================================

PIPELINE PRINCIPALE:
====================

Questo modulo orchestra gli esperimenti di approssimazione della GED:
    1. Processa coppie di grafi con/senza feature strutturali
    2. Esegue simulazioni su più dataset
    3. Applica regressione SVR per predire GED dai punteggi GW
    4. Calcola metriche di valutazione complessive (MAE, MSE, RMSE, correlazioni)

FLUSSO DATI:
============
Input: Dataset CSV di grafi + Ground Truth GED
    ↓
Process: Per ogni coppia di grafi:
    - Normalizza dimensioni
    - Calcola feature strutturali
    - Combina con distanze di etichetta
    - Applica FGW
    - Estrai OT features
    ↓
Output: CSV con colonne:
    - GW_Score (approssimazione FGW)
    - OT features (8 feature derivate dal coupling)
    - True_GED (ground truth dal dataset)

ESPERIMENTI CONDOTTI:
====================
Per ogni dataset (AIDS, IMDB, Linux):
    - Esecuzione con feature strutturali = No
    - Esecuzione con feature strutturali = Sì
    
Permettendo il confronto diretto della qualità dell'approssimazione.

FUNZIONI CHIAVE:
================
- process_graph_pair: Processa una singola coppia di grafi
- make_simulation: Orchestra gli esperimenti su tutti i dataset
- apply_svr_and_compute_metrics: Addestra SVR e calcola metriche
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ged_computation import (
    get_graph_by_id,
    compare_and_swap_graphs,
    compute_label_distance,
    compute_cross_matrix_with_structural_features,
    calculate_cross_matrix,
    compute_ged_GW,
    extract_ot_features,
    label_dict_construction,
)


def process_graph_pair(G_1, G_2, labels_1, labels_2, true_ged, dataset, list_of_indices=None, mu=0.5):
    """
    Processa una singola coppia di grafi e calcola i punteggi GW con/senza feature strutturali.
    
    PIPELINE DI PROCESSAMENTO:
    ==========================
    
    Questa funzione implementa l'intera pipeline per una coppia di grafi:
    
    Step 1: NORMALIZZAZIONE DIMENSIONI
    -----------------------------------
    - Controlla che i grafi abbiano lo stesso numero di nodi
    - Se no, aggiunge nodi dummy al grafo più piccolo
    - Aggiusta le etichette di conseguenza
    
    Step 2: VALIDAZIONE ETICHETTE
    ------------------------------
    - Controlla che le etichette corrispondano al numero di nodi
    - Se mancano etichette (es. per nodi dummy), aggiunge "?" come placeholder
    
    Step 3: CALCOLO MATRICI DI DISTANZA
    -----------------------------------
    - Etichette: calcola distanze tra etichette dei nodi
    - Struttura: calcola distanze tra feature strutturali (Deg, PR, CC, Betw)
    
    Step 4: DOPPIO CALCOLO FGW
    -----------------------------------
    a) Senza feature strutturali:
       - Usa solo le distanze di etichetta
       - Baseline: GW classico
    
    b) Con feature strutturali:
       - Combina etichette + struttura con peso μ
       - Metodo proposto
    
    Step 5: ESTRAZIONE OT FEATURES
    --------------------------------
    - Dalla matrice di coupling ottimale, estrae 8 feature informative
    - Queste feature possono migliorare la previsione del GED vero
    
    Args:
        G_1 (nx.Graph): Primo grafo
        G_2 (nx.Graph): Secondo grafo
        labels_1 (list or None): Etichette per il primo grafo
        labels_2 (list or None): Etichette per il secondo grafo
        true_ged (float): Valore vero della GED da ground truth
        dataset (str): Nome del dataset ('AIDS', 'IMDB', o 'Linux')
        list_of_indices (list, optional): Quali feature strutturali usare
                                          Es: ['Deg', 'PR', 'CC'] o None per tutte
        mu (float): Peso della regolarizzazione strutturale (default: 0.5)
    
    Returns:
        dict or None: Dizionario con due chiavi:
            - 'no_structural_features': {
                'GW_Score': float,
                'True_GED': float,
                'ot_alignment_entropy': float,
                'ot_alignment_confidence': float,
                ... (altre OT features)
              }
            - 'with_structural_features': {
                'GW_Score': float,
                'True_GED': float,
                'ot_alignment_entropy': float,
                ... (altre OT features)
              }
        
        Ritorna None solo se i grafi non vengono trovati nel dataset
    
    GESTIONE ERRORI:
    ----------------
    La funzione è robusta a:
    - Lunghezze di etichetta che non corrispondono ai nodi
    - Grafi di dimensioni diverse
    - Dataset non etichettati (AIDS, IMDB, Linux)
    
    Nota Importante:
    ----------------
    Se le etichette sono mancanti o scorrette, vengono aggiunti placeholder ("?")
    automaticamente. Questo garantisce che la funzione non fallisca, ma i risultati
    potrebbero essere sub-ottimali se l'input è corrotto.
    """
    # Gestisci il dataset AIDS con etichette
    if dataset == 'AIDS':
        # Correggi mismatches etichetta-nodo aggiungendo etichette dummy se necessario
        if labels_1 is None:
            labels_1 = ["?"] * G_1.number_of_nodes()
        elif len(labels_1) < G_1.number_of_nodes():
            # Aggiungi etichette dummy per i nodi mancanti
            labels_1 = labels_1 + ["?"] * (G_1.number_of_nodes() - len(labels_1))
        elif len(labels_1) > G_1.number_of_nodes():
            # Tronca le etichette in eccesso (non dovrebbe accadere, ma gestisci comunque)
            labels_1 = labels_1[:G_1.number_of_nodes()]
        
        if labels_2 is None:
            labels_2 = ["?"] * G_2.number_of_nodes()
        elif len(labels_2) < G_2.number_of_nodes():
            # Aggiungi etichette dummy per i nodi mancanti
            labels_2 = labels_2 + ["?"] * (G_2.number_of_nodes() - len(labels_2))
        elif len(labels_2) > G_2.number_of_nodes():
            # Tronca le etichette in eccesso (non dovrebbe accadere, ma gestisci comunque)
            labels_2 = labels_2[:G_2.number_of_nodes()]
        
        # Normalizza le dimensioni dei grafi e adatta le etichette
        # Questo aggiungerà più etichette "?" se i grafi hanno dimensioni diverse
        G_1, G_2, labels_1, labels_2 = compare_and_swap_graphs(G_1, G_2, labels_1, labels_2)
        
        # Calcola la matrice di distanza tra etichette
        label_distance_matrix = compute_label_distance(labels_1, labels_2, G_1.number_of_nodes())
    else:
        # Grafi non etichettati (IMDB, Linux)
        G_1, G_2 = compare_and_swap_graphs(G_1, G_2)
        label_distance_matrix = compute_label_distance(None, None, G_1.number_of_nodes())
    
    # Calcola le distanze tra feature strutturali
    structural_cross_matrix = compute_cross_matrix_with_structural_features(
        G_1, G_2, list_of_centrality_indices=list_of_indices
    )
    
    # Calcola i punteggi GW per entrambe le configurazioni
    results = {}
    for include_features in [False, True]:
        cross_matrix = calculate_cross_matrix(
            label_distance_matrix, 
            structural_cross_matrix, 
            mu=mu, 
            include_structural_features=include_features
        )
        # Ottieni il punteggio GW E la matrice di coupling per le OT features
        gw_score, coupling = compute_ged_GW(G_1, G_2, cross_matrix)
        
        # Estrai le OT features dalla matrice di coupling
        C1 = nx.to_numpy_array(G_1)
        C2 = nx.to_numpy_array(G_2)
        ot_features = extract_ot_features(coupling, cross_matrix, C1, C2)
        
        key = 'with_structural_features' if include_features else 'no_structural_features'
        results[key] = {
            'GW_Score': gw_score, 
            'True_GED': true_ged,
            **ot_features  # Unpacka le OT features nel dict dei risultati
        }
    
    return results


def make_simulation(n_sample=1500, output_dir='risultati', list_of_indices=None, mu=0.5):
    """
    Esegue la simulazione di approssimazione della GED su tutti i dataset.
    
    OVERVIEW DELLA SIMULAZIONE:
    ============================
    
    Per ogni dataset (AIDS, IMDB, Linux):
        1. Campiona n_sample coppie di grafi dal ground truth
        2. Per ogni coppia:
           a) Calcola GW_Score SENZA feature strutturali (baseline)
           b) Calcola GW_Score CON feature strutturali (metodo proposto)
           c) Estrai 8 OT features dalla matrice di coupling
        3. Salva i risultati in CSV
    
    SCOPO:
    ------
    Questa simulazione consente il confronto sperimentale diretto:
        - Qualità dell'approssimazione: FGW score vs true GED
        - Beneficio delle feature strutturali: delta tra due configurazioni
        - Informazioni diagnostiche: OT features per analisi post-hoc
    
    PARAMETRI CONFIGURABILI:
    ========================
    
    n_sample (int):
        Numero di coppie di grafi da processare per dataset
        - Valori tipici: 500, 1500, 2000
        - Valori alti: migliore coverage ma più lento
    
    list_of_indices (list):
        Quali feature strutturali usare:
        - None: usa tutte (Deg, PR, CC, Betw) - completo
        - ['Deg']: solo degree centrality - minimale
        - ['PR', 'CC']: PageRank + Clustering - bilanciato
        Esperimenti mostrano che meno non è sempre meglio: ci può essere rumore
    
    mu (float):
        Peso della regolarizzazione strutturale
        - μ = 0: solo label distance (GW classico)
        - μ = 0.5: bilancia label e struttura (default)
        - μ = 1.0: peso uguale a label e struttura
        Esperimenti guidati mostrano che μ ≈ 0.5 è spesso ottimale
    
    OUTPUT:
    =======
    Per ogni dataset vengono salvati due file CSV:
        - {dataset}_no_structural_features_{indices}.csv
        - {dataset}_with_structural_features_{indices}.csv
    
    Ogni file contiene le colonne:
        - GW_Score: l'approssimazione FGW
        - ot_alignment_entropy: entropia dell'allineamento
        - ot_alignment_confidence: fiducia nel match
        - ot_transport_cost: costo del trasporto
        - ot_marginal_balance: bilanciamento dei margini
        - ot_coupling_sparsity: sparsità del coupling
        - ot_max_coupling: valore max del coupling
        - ot_coupling_variance: varianza del coupling
        - ot_structural_mismatch: mismatch strutturale
        - True_GED: il valore vero dal ground truth
    
    ESEMPIO DI UTILIZZO:
    ====================
    
    # Simulazione completa con tutte le feature
    make_simulation(n_sample=2000)
    
    # Simulazione con solo PageRank
    make_simulation(n_sample=2000, list_of_indices=['PR'], mu=0.7)
    
    # Simulazione baseline senza struttura
    make_simulation(n_sample=500, list_of_indices=[], mu=0.0)
    
    TIMING E COMPLESSITÀ:
    =====================
    Tempo per grafo (approssimativo):
        - Grafi piccoli (AIDS ~10-15 nodi): 50-100ms per coppia
        - Grafi medi (IMDB ~20-50 nodi): 200-500ms per coppia  
        - Grafi grandi (Linux ~50+ nodi): 1-5 secondi per coppia
    
    Tempo totale ≈ n_sample * time_per_coppia
    Per n_sample=2000 su AIDS: ~2-3 minuti
    """
    dataset_names = ['AIDS', 'IMDB', 'Linux']
    
    for dataset in dataset_names:
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset}")
        print(f"{'='*60}")
        
        # Initialize results DataFrames
        results = {
            'no_structural_features': pd.DataFrame(columns=['GW_Score', 'True_GED']),
            'with_structural_features': pd.DataFrame(columns=['GW_Score', 'True_GED'])
        }
        
        # Define dataset paths
        dataset_paths = {
            'true_ged': os.path.join("True_GED", dataset, f"{dataset}_ged.csv"),
            'graphs': os.path.join("Dataset", dataset, f"{dataset}_graphs.csv")
        }
        
        # Load datasets
        try:
            true_ged_df = pd.read_csv(dataset_paths['true_ged']).sample(n=n_sample, random_state=42)
            graphs_df = pd.read_csv(dataset_paths['graphs'])
            print(f"Loaded {len(true_ged_df)} graph pairs from {dataset}")
        except FileNotFoundError as e:
            print(f"❌ Dataset file not found: {e}")
            continue
        except ValueError as e:
            print(f"❌ Sampling error (not enough data?): {e}")
            continue
        
        # Pre-load labels for AIDS dataset
        labels_dict = label_dict_construction() if dataset == 'AIDS' else None
        
        # Process each graph pair
        batch_results = {'no_structural_features': [], 'with_structural_features': []}
        processed_count = 0
        skipped_count = 0
        
        for _, row in true_ged_df.iterrows():
            id_1, id_2, true_ged = row['id_1'], row['id_2'], row['true_ged']
            
            # Retrieve graphs
            G_1 = get_graph_by_id(graphs_df, id_1)
            G_2 = get_graph_by_id(graphs_df, id_2)
            
            if G_1 is None or G_2 is None:
                skipped_count += 1
                continue
            
            # Get labels if applicable
            labels_1 = labels_dict.get(id_1, None) if labels_dict else None
            labels_2 = labels_dict.get(id_2, None) if labels_dict else None
            
            # Process the graph pair
            pair_results = process_graph_pair(
                G_1, G_2, labels_1, labels_2, true_ged, dataset, list_of_indices, mu
            )
            
            # Only add results if processing was successful
            if pair_results is not None:
                for key in batch_results:
                    batch_results[key].append(pair_results[key])
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"✓ Processed: {processed_count} pairs")
        if skipped_count > 0:
            print(f"⚠ Skipped: {skipped_count} pairs (graphs not found in dataset)")
        
        # Convert to DataFrames and save
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename suffix based on indices used
        if list_of_indices is None:
            indices_suffix = "baseline"
        else:
            indices_suffix = "_".join(sorted(list_of_indices))
        
        # Save results
        for key in results:
            if batch_results[key]:
                results[key] = pd.DataFrame(batch_results[key])
                filename = f'{dataset}_{key}_{indices_suffix}.csv'
                filepath = os.path.join(output_dir, filename)
                results[key].to_csv(filepath, index=False)
                print(f"  Saved: {filename} ({len(results[key])} samples)")
    
    print(f"\n{'='*60}")
    print("Simulation Complete!")
    print(f"{'='*60}")

def apply_svr_and_compute_metrics(df, dataset_name, feature_type, indices_suffix):
    """
    Apply Support Vector Regression to predict GED from GW scores and compute metrics.
    
    Trains SVR models with three different kernels (RBF, Linear, Polynomial) and
    evaluates their performance using multiple metrics.
    
    Args:
        df (pd.DataFrame): DataFrame with 'GW_Score' and 'True_GED' columns
        dataset_name (str): Name of the dataset
        feature_type (str): 'No Structural Features' or 'With Structural Features'
        indices_suffix (str): Description of centrality indices used
    
    Returns:
        list or None: List of result dictionaries (one per kernel), or None if insufficient data
        
    Each result dictionary contains:
        - MAE: Mean Absolute Error
        - MSE: Mean Squared Error  
        - RMSE: Root Mean Squared Error
        - Accuracy: Percentage within ±1 edit distance
        - Spearman_Correlation: Rank correlation
        - Kendall_Tau: Another rank correlation measure
    
    Notes:
        - Requires at least 10 samples for train/test split
        - Uses 80/20 train/test split with random_state=42
        - Feature scaling applied for SVR stability
    """
    # Validate sufficient data
    if len(df) < 10:
        print(f"  ⚠ Warning: Insufficient data ({len(df)} samples) for {dataset_name}_{feature_type}")
        return None
    
    # Split data into training and test sets (80/20 split)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    X_train = train_df[['GW_Score']]
    y_train = train_df['True_GED']
    X_test = test_df[['GW_Score']]
    y_test = test_df['True_GED']
    
    # Feature scaling is critical for SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    results = []
    
    # Test different SVR kernels
    for kernel_type in ['rbf', 'linear', 'poly']:
        print(f"    Testing kernel: {kernel_type}")
        
        # Train SVR model
        # C=1.0: Regularization parameter
        # epsilon=0.1: Width of epsilon-insensitive tube
        model = SVR(kernel=kernel_type, C=1.0, epsilon=0.1)
        model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        
        # Inverse transform predictions back to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Compute evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Accuracy: percentage within ±1 edit distance
        accuracy = np.mean(np.abs(np.round(y_pred) - y_test) <= 1) * 100
        
        # Rank correlation measures
        spearman_corr = pd.Series(y_test.values).corr(pd.Series(y_pred), method='spearman')
        kendall_tau = pd.Series(y_test.values).corr(pd.Series(y_pred), method='kendall')
        
        # Print results
        print(f"      MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")
        print(f"      Accuracy: {accuracy:.2f}% | Spearman: {spearman_corr:.4f} | Kendall: {kendall_tau:.4f}")
        print("    " + "-" * 30)
        
        # Store results
        results.append({
            'Dataset': dataset_name,
            'Features': feature_type,
            'Indices': indices_suffix,
            'Kernel': kernel_type,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Accuracy': accuracy,
            'Spearman_Correlation': spearman_corr,
            'Kendall_Tau': kendall_tau,
            'Samples': len(df)
        })
    
    return results


def run_regressions_with_svr(output_dir='risultati'):
    """Run SVR on all CSV files and compute metrics based on SVR predictions."""
    try:
        # Find all CSV files in subdirectories
        all_results = []
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.csv') and 'summary' not in file.lower():
                    file_path = os.path.join(root, file)
                    
                    # Extract info from filename and path
                    filename = os.path.basename(file)
                    dir_name = os.path.basename(root)
                    
                    # Parse dataset and feature type from filename
                    parts = filename.replace('.csv', '').split('_')
                    if len(parts) >= 3:
                        dataset = parts[0]
                        if 'no_structural_features' in filename:
                            feature_type = 'No Structural Features'
                        elif 'with_structural_features' in filename:
                            feature_type = 'With Structural Features'
                        else:
                            feature_type = 'Unknown'
                        
                        # Extract indices from directory name or filename
                        if 'no_indices' in dir_name or 'baseline' in filename:
                            indices_suffix = 'Baseline'
                        else:
                            indices_part = dir_name.replace('ablation_indices_', '').replace('ablation_', '')
                            indices_suffix = indices_part.replace('_', '+')
                        
                        print(f"\n📊 Processing: {dataset} | {feature_type} | {indices_suffix}")
                        print(f"    File: {file_path}")
                        
                        try:
                            df = pd.read_csv(file_path)
                            if 'GW_Score' in df.columns and 'True_GED' in df.columns:
                                results = apply_svr_and_compute_metrics(df, dataset, feature_type, indices_suffix)
                                if results:
                                    all_results.extend(results)
                            else:
                                print(f"    Warning: Required columns not found in {file}")
                        except Exception as e:
                            print(f"    Error processing {file}: {str(e)}")
        
        # Save comprehensive results
        if all_results:
            results_df = pd.DataFrame(all_results)
            summary_file = os.path.join(output_dir, 'svr_metrics_summary.csv')
            results_df.to_csv(summary_file, index=False)
            print(f"\n📝 SVR metrics summary saved to: {summary_file}")
            
            # Print best results summary
            print("\n" + "="*80)
            print("BEST SVR PERFORMANCE BY DATASET")
            print("="*80)
            
            for dataset in results_df['Dataset'].unique():
                dataset_data = results_df[results_df['Dataset'] == dataset]
                if not dataset_data.empty:
                    # Best correlation
                    best_corr = dataset_data.loc[dataset_data['Spearman_Correlation'].idxmax()]
                    # Best MAE
                    best_mae = dataset_data.loc[dataset_data['MAE'].idxmin()]
                    # Best MSE
                    best_mse = dataset_data.loc[dataset_data['MSE'].idxmin()]
                    
                    print(f"\n{dataset}:")
                    print(f"  Best Correlation: {best_corr['Spearman_Correlation']:.3f} | "
                          f"{best_corr['Indices']} | {best_corr['Features']} | {best_corr['Kernel']}")
                    print(f"  Best MAE: {best_mae['MAE']:.3f} | "
                          f"{best_mae['Indices']} | {best_mae['Features']} | {best_mae['Kernel']}")
                    print(f"  Best MSE: {best_mse['MSE']:.3f} | "
                          f"{best_mse['Indices']} | {best_mse['Features']} | {best_mse['Kernel']}")
        
    except FileNotFoundError:
        print(f"Output directory '{output_dir}' not found.")


def run_regressions(output_dir='risultati'):
    """Legacy function - now redirects to SVR-based metrics computation."""
    print("Note: Using SVR-based metrics computation instead of legacy regression.")
    run_regressions_with_svr(output_dir)


# if __name__ == "__main__":
#     make_simulation()
#     run_regressions()
         
