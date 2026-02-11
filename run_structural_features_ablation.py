"""
Studio di Ablazione delle Feature Strutturali per Graph Edit Distance
=====================================================================

DESCRIZIONE ESPERIMENTO:
========================
Analizza come diverse combinazioni di feature strutturali influenzano 
l'accuratezza della previsione della Graph Edit Distance su tre dataset:
AIDS, IMDB e Linux.

PROCEDURA:
==========
1. Per ogni dataset:
   Per ogni configurazione di feature (7 total):
   a) Genera i dati AIDS usando FGW con quelle feature e μ=0.5
   b) Per IMDB e Linux, μ non ha effetto (no labels)
   c) Estrae feature strutturali sottinsiemi (Deg, CC, PR singoli e coppie)
   d) Estrae le 8 feature OT dalla matrice di coupling
   e) Addestra 5 regressori (RandomForest, GradientBoosting, SVR, Huber, Ensemble)
   f) Calcola predizioni e salva i risultati

2. Per ogni (dataset, configurazione, modello), calcola metriche:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - Accuracy (% predizioni entro ±1 di GED vera)
   - Correlazione di Spearman
   - Correlazione di Kendall

3. Genera un file Markdown per ogni dataset con:
   - Tabella TOP 5 configurazioni ordinate per MAE crescente
   - Risultati pronti per LaTeX

FEATURE CONFIGURAZIONI:
======================
1. Deg (degree only)
2. CC (clustering coefficient only)
3. PR (pagerank only)
4. Deg + CC
5. Deg + PR
6. CC + PR
7. Deg + CC + PR (baseline completo)

OUTPUT:
=======
- Markdown per dataset: results/ablation_results/{DATASET}_ablation_results.md
  Contiene tabella TOP 5 configurazioni ordinate per MAE

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

# Dataset da testare
DATASETS = ['AIDS', 'IMDB', 'Linux']

# μ fisso per AIDS (unico con etichette), ignorato per IMDB e Linux
MU = 0.5

# Configurazioni di feature da testare (7 totali)
FEATURE_CONFIGURATIONS = {
    'Deg': ['Deg'],
    'CC': ['CC'],
    'PR': ['PR'],
    'Deg+CC': ['Deg', 'CC'],
    'Deg+PR': ['Deg', 'PR'],
    'CC+PR': ['CC', 'PR'],
    'Deg+CC+PR': ['Deg', 'CC', 'PR']
}

# Numero di campioni
N_SAMPLES = 2000

# Random state per reproducibilità
RANDOM_STATE = 42

# Directory di output
OUTPUT_DIR = Path("results/ablation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cartella per dati generati
GENERATED_DATA_DIR = Path("generated_data")
GENERATED_DATA_DIR.mkdir(exist_ok=True)

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

def load_and_prepare_data(dataset_name, feature_config_name, feature_list):
    """
    Carica o genera i dati per un dataset con una configurazione di feature specifica.
    
    Returns:
        pd.DataFrame: DataFrame con le colonne:
            - GW_Score
            - ot_alignment_entropy, ot_alignment_confidence, ...
            - True_GED
    """
    features_str = "_".join(sorted(feature_list))
    
    # Nome del file generato da make_simulation (senza config name)
    original_filename = f"{dataset_name}_with_structural_features_{features_str}.csv"
    original_filepath = Path(original_filename)
    
    # Nome del file con config name incluso (per tracking dei risultati)
    filename_with_config = f"{dataset_name}_with_structural_features_{feature_config_name}.csv"
    filepath_with_config = GENERATED_DATA_DIR / filename_with_config
    
    # Genera i dati se non existono
    if not filepath_with_config.exists():
        print(f"    Generando dati {dataset_name} con feature {feature_config_name}...")
        make_simulation(
            n_sample=N_SAMPLES,
            output_dir='.',  # salva nella root con il vecchio nome
            list_of_indices=feature_list,
            mu=MU  # Usato solo per AIDS (ha etichette), ignorato per IMDB/Linux
        )
        
        # Rinomina il file aggiungendo config name per evitare confusione
        if original_filepath.exists():
            import shutil
            shutil.copy(original_filepath, filepath_with_config)
    
    # Carica i dati dal file con config name
    df = pd.read_csv(filepath_with_config)
    
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

def process_configuration(dataset_name, config_name, feature_list):
    """
    Processa una singola configurazione di feature per un dataset:
    - Carica/genera i dati
    - Addestra i regressori
    - Calcola metriche
    
    Returns:
        dict: Risultati per questa configurazione
              Chiavi: nomi dei modelli + 'Ensemble'
              Valori: dict con metriche (mae, mse, accuracy, spearman_r, kendall_tau)
    """
    print(f"  {config_name:15s}", end=" ", flush=True)
    
    # Carica i dati
    df = load_and_prepare_data(dataset_name, config_name, feature_list)
    
    # Separa features e target
    feature_cols = [col for col in df.columns if col.startswith('ot_')]
    feature_cols = ['GW_Score'] + feature_cols  # Includi GW_Score
    
    X = df[feature_cols].values
    y = df['True_GED'].values
    
    # Split train/test
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Dizionario per le metriche
    metrics_dict = {}
    
    # Addestra ogni modello
    for model_name in MODELS.keys():
        y_pred, mae, mse, accuracy, spearman_r, kendall_tau_val = \
            train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test)
        
        metrics_dict[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'Accuracy': accuracy,
            'Spearman_r': spearman_r,
            'Kendall_tau': kendall_tau_val
        }
    
    # Ensemble: media dei 3 modelli migliori
    rf_pred = train_and_evaluate_model('RandomForest', X_train, y_train, X_test, y_test)[0]
    gb_pred = train_and_evaluate_model('GradientBoosting', X_train, y_train, X_test, y_test)[0]
    huber_pred = train_and_evaluate_model('Huber', X_train, y_train, X_test, y_test)[0]
    ensemble_pred = (0.5 * rf_pred + 0.3 * gb_pred + 0.2 * huber_pred)
    
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_accuracy = np.mean(np.abs(ensemble_pred - y_test) <= 1) * 100
    ensemble_spearman, _ = spearmanr(y_test, ensemble_pred)
    ensemble_kendall, _ = kendalltau(y_test, ensemble_pred)
    
    metrics_dict['Ensemble'] = {
        'MAE': ensemble_mae,
        'MSE': ensemble_mse,
        'Accuracy': ensemble_accuracy,
        'Spearman_r': ensemble_spearman,
        'Kendall_tau': ensemble_kendall
    }
    
    print(f"→ MAE: {ensemble_mae:.4f}")
    
    return metrics_dict

def format_markdown_table_top5(dataset_name, all_results):
    """
    Formatta i risultati in Markdown con TOP 5 configurazioni per MAE.
    
    Args:
        dataset_name: nome del dataset
        all_results: dict di dict
            all_results[config][modello] = {'MAE': x, ...}
    
    Returns:
        str: Markdown tabellare
    """
    markdown = f"# Ablazione Feature Strutturali - Dataset {dataset_name}\n\n"
    markdown += f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown += f"Dataset: {dataset_name}\n"
    markdown += f"Numero configurazioni testate: {len(all_results)}\n"
    markdown += f"Parametro μ: {MU} (fisso)\n\n"
    
    # Calcola MAE migliore per ogni configurazione (media su modelli?)
    # Oppure prendi il miglior MAE tra tutti i modelli per quella configurazione
    config_mae = {}
    for config, models_results in all_results.items():
        # Prendi il miglior MAE tra i modelli per questa configurazione
        best_mae_for_config = min([m['MAE'] for m in models_results.values()])
        config_mae[config] = best_mae_for_config
    
    # Ordina per MAE crescente (migliore primo)
    sorted_configs = sorted(config_mae.items(), key=lambda x: x[1])
    
    # Prendi TOP 5
    top5_configs = sorted_configs[:5]
    
    markdown += "## TOP 5 Configurazioni per MAE\n\n"
    markdown += "| Rank | Configurazione | Miglior MAE | Miglior Modello |\n"
    markdown += "|------|----------------|-------------|----------------|\n"
    
    for rank, (config, mae) in enumerate(top5_configs, 1):
        # Trova quale modello ha raggiunto questo MAE per questa configurazione
        best_model_for_config = min(
            all_results[config].items(),
            key=lambda x: x[1]['MAE']
        )
        model_name, metrics = best_model_for_config
        markdown += f"| {rank} | {config:20s} | {mae:.4f} | {model_name:15s} |\n"
    
    markdown += "\n---\n\n"
    
    # Tabella dettagliata per TOP 5
    markdown += "## Metriche Complete per TOP 5\n\n"
    
    for rank, (config, mae) in enumerate(top5_configs, 1):
        markdown += f"### #{rank} - Configurazione: {config}\n\n"
        markdown += "| Modello | MAE | MSE | Accuracy | Spearman_r | Kendall_tau |\n"
        markdown += "|---------|-----|-----|----------|------------|-------------|\n"
        
        for model, metrics in all_results[config].items():
            markdown += f"| {model:15s} | {metrics['MAE']:.4f} | {metrics['MSE']:.4f} | {metrics['Accuracy']:.2f}% | {metrics['Spearman_r']:.4f} | {metrics['Kendall_tau']:.4f} |\n"
        
        markdown += "\n"
    
    return markdown

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue lo studio di ablazione delle feature strutturali."""
    print("\n" + "="*80)
    print("STUDIO DI ABLAZIONE DELLE FEATURE STRUTTURALI")
    print("="*80)
    print(f"Dataset: {', '.join(DATASETS)}")
    print(f"Configurazioni feature: {len(FEATURE_CONFIGURATIONS)}")
    print(f"Modelli: {len(MODELS)} + Ensemble")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Processa ogni dataset
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        
        # Processa ogni configurazione di feature
        all_results = {}
        for config_name, feature_list in FEATURE_CONFIGURATIONS.items():
            results_config = process_configuration(dataset, config_name, feature_list)
            all_results[config_name] = results_config
        
        # Genera Markdown con TOP 5
        print(f"\nGenerando Markdown...")
        markdown_content = format_markdown_table_top5(dataset, all_results)
        
        markdown_file = OUTPUT_DIR / f"{dataset}_ablation_results.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        print(f"✓ Salvato Markdown: {markdown_file}")
    
    print("\n" + "="*80)
    print("ESPERIMENTO COMPLETATO!")
    print("="*80)
    print(f"\nOutput salvati in: {OUTPUT_DIR}/")
    for dataset in DATASETS:
        print(f"  - {dataset}_ablation_results.md")

if __name__ == '__main__':
    main()
