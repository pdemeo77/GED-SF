"""
Esperimento di Sensibilità del Parametro μ per Graph Edit Distance
===================================================================

DESCRIZIONE ESPERIMENTO:
========================
Questo script analizza come il parametro μ (che controlla il trade-off tra
distanza di etichetta e distanza strutturale) influenza l'accuratezza della
previsione della Graph Edit Distance nel dataset AIDS.

PROCEDURA:
==========
1. Per ogni valore di μ in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]:
   a) Genera i dati AIDS usando FGW con quel μ
   b) Estrae feature strutturali (Degree, PageRank, Clustering Coefficient)
   c) Estrae le 8 feature OT dalla matrice di coupling
   d) Addestra 5 regressori (RandomForest, GradientBoosting, SVR, Huber, Ensemble)
   e) Calcola predizioni e salva i risultati

2. Per ogni (μ, modello), calcola metriche:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - Accuracy (% predizioni entro ±1 di GED vera)
   - Correlazione di Spearman
   - Correlazione di Kendall

3. Genera un file Markdown con:
   - Tabella consolidata (μ come righe, modelli come colonne)
   - Risultati pronti per LaTeX

OUTPUT:
=======
- CSV per ogni μ: results/mu_sensitivity_results/mu_X.XX.csv
  Colonne: id_1, id_2, true_ged, rf_pred, gb_pred, svr_pred, huber_pred, ensemble_pred
  
- Markdown: results/mu_sensitivity_results/ESPERIMENTO_SENSIBILITA_MU.md
  Contiene tabelle con le metriche consolidate

FEATURE UTILIZZATE:
===================
Strutturali: Degree (Deg), PageRank (PR), Clustering Coefficient (CC)
OT: 8 feature dalla matrice di coupling
NO Betweenness (come richiesto)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Import project modules
from make_simulation import make_simulation
from utils import build_ged_ground_truth_dataframe

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Valori di μ da testare
MU_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]

# Dataset
DATASET = 'AIDS'

# Feature strutturali da usare (NO Betweenness)
STRUCTURAL_FEATURES = ['Deg', 'PR', 'CC']

# Numero di campioni
N_SAMPLES = 2000

# Random state per reproducibilità
RANDOM_STATE = 42

# Directory di output
OUTPUT_DIR = Path("results/mu_sensitivity_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configurazione dei modelli
MODELS = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        loss='squared_error'
    ),
    'SVR': SVR(
        kernel='rbf',
        C=10,
        gamma='scale',
        epsilon=0.1
    ),
    'Huber': HuberRegressor(
        epsilon=1.35,
        max_iter=200,
        alpha=0.0001
    )
}

# ============================================================================
# FUNZIONI AUSILIARIE
# ============================================================================

def load_and_prepare_data(mu_value):
    """
    Carica o genera i dati AIDS con il parametro μ specificato.
    
    Nota: make_simulation non include μ nel nome del file, quindi generiamo
    il file con make_simulation e poi lo rinominiamo aggiungendo μ.
    
    Returns:
        pd.DataFrame: DataFrame con le colonne:
            - GW_Score
            - ot_alignment_entropy, ot_alignment_confidence, ot_transport_cost,
              ot_marginal_balance, ot_coupling_sparsity, ot_max_coupling,
              ot_coupling_variance, ot_structural_mismatch
            - True_GED
    """
    features_str = "_".join(sorted(STRUCTURAL_FEATURES))
    
    # Cartella da cui caricamento i dati generati
    generated_dir = Path("generated_data")
    generated_dir.mkdir(exist_ok=True)
    
    # Nome del file generato da make_simulation (senza μ)
    original_filename = f"AIDS_with_structural_features_{features_str}.csv"
    original_filepath = Path(original_filename)
    
    # Nome del file con μ incluso (per tracking dei risultati per ogni μ)
    # Salva in generated_data/
    filename_with_mu = f"AIDS_with_structural_features_{features_str}_mu_{mu_value:.2f}.csv"
    filepath_with_mu = generated_dir / filename_with_mu
    
    # Genera i dati se non existono (sia il vecchio nome che il nuovo)
    if not filepath_with_mu.exists():
        print(f"    Generando dati AIDS con μ={mu_value}, feature={STRUCTURAL_FEATURES}...")
        make_simulation(
            n_sample=N_SAMPLES,
            output_dir='.',  # salva nella root con il vecchio nome
            list_of_indices=STRUCTURAL_FEATURES,
            mu=mu_value
        )
        
        # Rinomina il file aggiungendo μ per evitare confusione
        if original_filepath.exists():
            import shutil
            shutil.copy(original_filepath, filepath_with_mu)
    
    # Carica i dati dal file con μ
    df = pd.read_csv(filepath_with_mu)
    
    return df

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):
    """
    Addestra un modello e calcola predizioni + metriche.
    
    Returns:
        tuple: (y_pred_test, mae, mse, accuracy, spearman_r, kendall_tau)
    """
    from sklearn.base import clone
    
    # Clona il modello per evitare modifiche agli iperparametri di base
    model = clone(MODELS[model_name])
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    # Calcola le metriche
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    # Accuracy: percentuale predizioni entro ±1 di GED vera
    accuracy = np.mean(np.abs(y_pred_test - y_test) <= 1) * 100
    
    # Correlazioni di Spearman e Kendall
    spearman_r, _ = spearmanr(y_test, y_pred_test)
    kendall_tau, _ = kendalltau(y_test, y_pred_test)
    
    return y_pred_test, mae, mse, accuracy, spearman_r, kendall_tau

def process_mu_value(mu_value):
    """
    Processa un singolo valore di μ:
    - Carica/genera i dati
    - Addestra i regressori
    - Calcola metriche
    - Salva CSV
    
    Returns:
        dict: Risultati per questo μ
              Chiavi: nomi dei modelli + 'Ensemble'
              Valori: dict con metriche (mae, mse, accuracy, spearman_r, kendall_tau)
    """
    print(f"\n{'='*70}")
    print(f"Processando μ = {mu_value}")
    print(f"{'='*70}")
    
    # Carica i dati
    print(f"  Caricamento dati AIDS...")
    df = load_and_prepare_data(mu_value)
    
    # Separa features e target
    feature_cols = [col for col in df.columns if col.startswith('ot_')]
    feature_cols = ['GW_Score'] + feature_cols  # Includi GW_Score
    
    X = df[feature_cols].values
    y = df['True_GED'].values
    
    # Split train/test
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Dizionario per le predizioni (per salvare CSV)
    predictions_dict = {
        'id_train': train_idx,
        'id_test': test_idx,
        'y_test': y_test
    }
    
    # Dizionario per le metriche
    metrics_dict = {}
    
    # Addestra ogni modello
    print(f"  Addestrando modelli...")
    for model_name in MODELS.keys():
        y_pred, mae, mse, accuracy, spearman_r, kendall_tau_val = \
            train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test)
        
        predictions_dict[f'{model_name}_pred'] = y_pred
        
        metrics_dict[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'Accuracy': accuracy,
            'Spearman_r': spearman_r,
            'Kendall_tau': kendall_tau_val
        }
        
        print(f"    {model_name:20s} → MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Ensemble: media dei 3 modelli migliori (come in run_ot_features_experiments.py)
    rf_pred = predictions_dict['RandomForest_pred']
    gb_pred = predictions_dict['GradientBoosting_pred']
    huber_pred = predictions_dict['Huber_pred']
    ensemble_pred = (0.5 * rf_pred + 0.3 * gb_pred + 0.2 * huber_pred)
    
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_accuracy = np.mean(np.abs(ensemble_pred - y_test) <= 1) * 100
    ensemble_spearman, _ = spearmanr(y_test, ensemble_pred)
    ensemble_kendall, _ = kendalltau(y_test, ensemble_pred)
    
    predictions_dict['Ensemble_pred'] = ensemble_pred
    
    metrics_dict['Ensemble'] = {
        'MAE': ensemble_mae,
        'MSE': ensemble_mse,
        'Accuracy': ensemble_accuracy,
        'Spearman_r': ensemble_spearman,
        'Kendall_tau': ensemble_kendall
    }
    
    print(f"    {'Ensemble':20s} → MAE: {ensemble_mae:.4f}, Accuracy: {ensemble_accuracy:.2f}%")
    
    # Salva CSV con le predizioni
    csv_filename = OUTPUT_DIR / f"mu_{mu_value:.2f}.csv"
    
    # Crea DataFrame per CSV
    csv_data = {
        'test_index': test_idx,
        'true_ged': y_test,
        'rf_pred': rf_pred,
        'gb_pred': gb_pred,
        'svr_pred': predictions_dict['SVR_pred'],
        'huber_pred': huber_pred,
        'ensemble_pred': ensemble_pred
    }
    
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(csv_filename, index=False)
    print(f"  ✓ Salvato CSV: {csv_filename}")
    
    return metrics_dict

def format_markdown_table(all_results):
    """
    Formatta i risultati in una tabella Markdown consolidata.
    
    args:
        all_results: dict di dict
            all_results[μ][modello] = {'MAE': x, 'MSE': y, ...}
    
    Returns:
        str: Markdown tabellare
    """
    markdown = "# Esperimento di Sensibilità del Parametro μ\n\n"
    markdown += f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown += f"Dataset: AIDS\n"
    markdown += f"Feature strutturali: {', '.join(STRUCTURAL_FEATURES)}\n"
    markdown += f"Feature OT: 8 feature dalla matrice di coupling\n"
    markdown += f"Campioni: {N_SAMPLES}\n\n"
    
    # Tabelle per ogni metrica
    metrics = ['MAE', 'MSE', 'Accuracy', 'Spearman_r', 'Kendall_tau']
    models = list(next(iter(all_results.values())).keys())
    
    for metric in metrics:
        markdown += f"## {metric}\n\n"
        markdown += "| μ | " + " | ".join(models) + " |\n"
        markdown += "|---|" + "|".join(["---" for _ in models]) + "|\n"
        
        for mu in MU_VALUES:
            mu_str = f"{mu:.2f}"
            row = f"| {mu_str} |"
            for model in models:
                value = all_results[mu][model][metric]
                # Formattazione diversa per Accuracy
                if metric == 'Accuracy':
                    row += f" {value:.2f}% |"
                else:
                    row += f" {value:.4f} |"
            markdown += row + "\n"
        
        markdown += "\n"
    
    # Statistiche riassuntive
    markdown += "## Statistiche Riassuntive\n\n"
    markdown += "| Metrica | Miglior μ | Miglior Modello | Valore |\n"
    markdown += "|---------|-----------|-----------------|--------|\n"
    
    # Miglior MAE
    best_mae = float('inf')
    best_mu_mae = None
    best_model_mae = None
    for mu in MU_VALUES:
        for model in models:
            mae = all_results[mu][model]['MAE']
            if mae < best_mae:
                best_mae = mae
                best_mu_mae = mu
                best_model_mae = model
    markdown += f"| MAE | {best_mu_mae:.2f} | {best_model_mae} | {best_mae:.4f} |\n"
    
    # Miglior Accuracy
    best_acc = 0
    best_mu_acc = None
    best_model_acc = None
    for mu in MU_VALUES:
        for model in models:
            acc = all_results[mu][model]['Accuracy']
            if acc > best_acc:
                best_acc = acc
                best_mu_acc = mu
                best_model_acc = model
    markdown += f"| Accuracy | {best_mu_acc:.2f} | {best_model_acc} | {best_acc:.2f}% |\n"
    
    # Correlazione media per μ
    markdown += "\n### Correlazione Media per μ\n\n"
    markdown += "| μ | Avg Spearman_r | Avg Kendall_tau |\n"
    markdown += "|---|---|---|\n"
    for mu in MU_VALUES:
        avg_spearman = np.mean([all_results[mu][model]['Spearman_r'] for model in models])
        avg_kendall = np.mean([all_results[mu][model]['Kendall_tau'] for model in models])
        markdown += f"| {mu:.2f} | {avg_spearman:.4f} | {avg_kendall:.4f} |\n"
    
    return markdown

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue l'esperimento di sensibilità di μ."""
    print("\n" + "="*70)
    print("ESPERIMENTO DI SENSIBILITÀ DEL PARAMETRO μ")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Feature strutturali: {STRUCTURAL_FEATURES}")
    print(f"Valori μ: {MU_VALUES}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    # Processa ogni valore di μ
    all_results = {}
    for mu_value in MU_VALUES:
        results_mu = process_mu_value(mu_value)
        all_results[mu_value] = results_mu
    
    # Genera Markdown
    print(f"\nGenerando Markdown...")
    markdown_content = format_markdown_table(all_results)
    
    markdown_file = OUTPUT_DIR / "ESPERIMENTO_SENSIBILITA_MU.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    print(f"✓ Salvato Markdown: {markdown_file}")
    
    print("\n" + "="*70)
    print("ESPERIMENTO COMPLETATO!")
    print("="*70)
    print(f"\nOutput salvati in: {OUTPUT_DIR}/")
    print(f"  - CSV per ogni μ: mu_X.XX.csv")
    print(f"  - Markdown: ESPERIMENTO_SENSIBILITA_MU.md")

if __name__ == '__main__':
    main()
