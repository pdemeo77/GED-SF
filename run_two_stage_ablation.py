"""
Two-Stage Ablation Study: GW_Score Impact + OT Features Enrichment
====================================================================

OBIETTIVO ESPERIMENTO:
======================
Quantificare il valore aggiunto delle OT features rispetto alla baseline di GW_Score solo.

FLUSSO:
=======
1. STAGE 1 (Baseline): X = [GW_Score] solo
   → Quale subset di feature strutturali (Deg, CC, PR) è migliore?

2. STAGE 2A (Enriched Raw): X = [GW_Score + 8 OT features] non normalizzato
   → Le OT features migliorano senza normalizzazione?

3. STAGE 2B (Enriched Normalized): X = [GW_Score + 8 OT features] normalizzato
   → Le OT features migliorano con normalizzazione?

4. DELTA ANALYSIS: Confronta Stage 1 vs 2A vs 2B
   → Guadagno % di MAE, Accuracy, etc.

OUTPUT:
=======
results/two_stage_ablation/
├── AIDS_two_stage_results.md
├── IMDB_two_stage_results.md
└── Linux_two_stage_results.md

Ogni Markdown contiene:
- Stage 1 results (7 configurazioni × 5 modelli)
- Stage 2A results (stesse)
- Stage 2B results (stesse)
- Tabella delta e % miglioramento
- Insights e conclusioni
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

warnings.filterwarnings('ignore')

# Import project modules
from make_simulation import make_simulation
from utils import build_ged_ground_truth_dataframe

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DATASETS = ['AIDS', 'IMDB', 'Linux']

# Feature configurations da testare (7 totali)
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

# Random state
RANDOM_STATE = 42

# Directory di output
OUTPUT_DIR = Path("results/two_stage_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cartella per dati generati
GENERATED_DATA_DIR = Path("generated_data")
GENERATED_DATA_DIR.mkdir(exist_ok=True)

# Configurazione dei modelli
MODELS = {
    'RandomForest': RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=3,
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
    Restituisce il dataframe con GW_Score e tutte le OT features.
    """
    features_str = "_".join(sorted(feature_list))
    filename_with_config = f"{dataset_name}_with_structural_features_{feature_config_name}.csv"
    filepath_with_config = GENERATED_DATA_DIR / filename_with_config
    
    # Genera i dati se non existono
    if not filepath_with_config.exists():
        print(f"    Generando dati {dataset_name} con feature {feature_config_name}...")
        make_simulation(
            n_sample=N_SAMPLES,
            output_dir='.',
            list_of_indices=feature_list,
            mu=0.5  # Fisso per AIDS, ignorato per IMDB/Linux
        )
        
        # Copia il file nella directory appropriata
        original_filepath = Path(f"{dataset_name}_with_structural_features_{features_str}.csv")
        if original_filepath.exists():
            import shutil
            shutil.copy(original_filepath, filepath_with_config)
    
    # Carica i dati
    df = pd.read_csv(filepath_with_config)
    return df

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):
    """
    Addestra un modello e calcola predizioni + metriche.
    """
    # Clona il modello
    model = clone(MODELS[model_name])
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    # Metriche
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    accuracy = np.mean(np.abs(y_pred_test - y_test) <= 1) * 100
    spearman_r, _ = spearmanr(y_test, y_pred_test)
    kendall_tau, _ = kendalltau(y_test, y_pred_test)
    
    return y_pred_test, mae, mse, accuracy, spearman_r, kendall_tau

def process_configuration_all_stages(dataset_name, config_name, feature_list, X_test_indices):
    """
    Processa una configurazione per tutti e 3 gli stage.
    
    Returns:
        dict: {
            'stage1': {model -> metrics},
            'stage2a': {model -> metrics},
            'stage2b': {model -> metrics}
        }
    """
    print(f"  {config_name:15s}", end=" ", flush=True)
    
    # Carica i dati
    df = load_and_prepare_data(dataset_name, config_name, feature_list)
    
    # Definisci le colonne OT
    ot_feature_cols = [col for col in df.columns if col.startswith('ot_')]
    
    # Estrai X e y
    X_gw = df[['GW_Score']].values
    X_enriched = df[['GW_Score'] + ot_feature_cols].values
    y = df['True_GED'].values
    
    # Train/test split con seed fisso
    X_train_gw, X_test_gw, y_train, y_test, train_idx, test_idx = train_test_split(
        X_gw, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Stage 2A e 2B usano gli STESSI indici di train/test
    X_train_enriched = X_enriched[train_idx]
    X_test_enriched = X_enriched[test_idx]
    
    # Stage 2B: Normalizza le feature OT (non GW_Score)
    scaler = StandardScaler()
    # Fit su train, applica a test
    X_train_enriched_normalized = X_train_enriched.copy()
    X_test_enriched_normalized = X_test_enriched.copy()
    
    # Normalizza solo le OT features (colonne 1 in poi), non GW_Score (colonna 0)
    X_train_enriched_normalized[:, 1:] = scaler.fit_transform(X_train_enriched[:, 1:])
    X_test_enriched_normalized[:, 1:] = scaler.transform(X_test_enriched[:, 1:])
    
    # Dizionario per i risultati
    results = {'stage1': {}, 'stage2a': {}, 'stage2b': {}}
    
    # Cache delle predizioni per evitare re-training nell'Ensemble
    predictions_cache = {'stage1': {}, 'stage2a': {}, 'stage2b': {}}
    
    # Addestra per ogni modello
    for model_name in MODELS.keys():
        # Stage 1: GW_Score solo
        pred1, mae1, mse1, acc1, sp1, k1 = train_and_evaluate_model(
            model_name, X_train_gw, y_train, X_test_gw, y_test
        )
        predictions_cache['stage1'][model_name] = pred1
        results['stage1'][model_name] = {
            'MAE': mae1, 'MSE': mse1, 'Accuracy': acc1,
            'Spearman_r': sp1, 'Kendall_tau': k1
        }
        
        # Stage 2A: Enriched non-normalizzato
        pred2a, mae2a, mse2a, acc2a, sp2a, k2a = train_and_evaluate_model(
            model_name, X_train_enriched, y_train, X_test_enriched, y_test
        )
        predictions_cache['stage2a'][model_name] = pred2a
        results['stage2a'][model_name] = {
            'MAE': mae2a, 'MSE': mse2a, 'Accuracy': acc2a,
            'Spearman_r': sp2a, 'Kendall_tau': k2a
        }
        
        # Stage 2B: Enriched normalizzato
        pred2b, mae2b, mse2b, acc2b, sp2b, k2b = train_and_evaluate_model(
            model_name, X_train_enriched_normalized, y_train, 
            X_test_enriched_normalized, y_test
        )
        predictions_cache['stage2b'][model_name] = pred2b
        results['stage2b'][model_name] = {
            'MAE': mae2b, 'MSE': mse2b, 'Accuracy': acc2b,
            'Spearman_r': sp2b, 'Kendall_tau': k2b
        }
    
    # Ensemble: media pesata (usa predizioni già calcolate)
    for stage_key in ['stage1', 'stage2a', 'stage2b']:
        rf_pred = predictions_cache[stage_key]['RandomForest']
        gb_pred = predictions_cache[stage_key]['GradientBoosting']
        huber_pred = predictions_cache[stage_key]['Huber']
        
        ensemble_pred = 0.5 * rf_pred + 0.3 * gb_pred + 0.2 * huber_pred
        
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_accuracy = np.mean(np.abs(ensemble_pred - y_test) <= 1) * 100
        ensemble_spearman, _ = spearmanr(y_test, ensemble_pred)
        ensemble_kendall, _ = kendalltau(y_test, ensemble_pred)
        
        results[stage_key]['Ensemble'] = {
            'MAE': ensemble_mae, 'MSE': ensemble_mse, 'Accuracy': ensemble_accuracy,
            'Spearman_r': ensemble_spearman, 'Kendall_tau': ensemble_kendall
        }
    
    return results

def format_two_stage_markdown(dataset_name, all_results):
    """
    Formatta i risultati di tutti e 3 gli stage in un unico Markdown.
    all_results[config] = {'stage1': {...}, 'stage2a': {...}, 'stage2b': {...}}
    """
    markdown = f"# Two-Stage Ablation Study - Dataset {dataset_name}\n\n"
    markdown += f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown += f"Dataset: {dataset_name}\n"
    markdown += f"Configurazioni testate: {len(all_results)}\n\n"
    
    markdown += "## Linea Guida dello Studio\n\n"
    markdown += "- **Stage 1 (Baseline)**: Input = [GW_Score] solo\n"
    markdown += "- **Stage 2A (Enriched Raw)**: Input = [GW_Score + 8 OT features] non normalizzato\n"
    markdown += "- **Stage 2B (Enriched Norm)**: Input = [GW_Score + 8 OT features] normalizzato via StandardScaler\n"
    markdown += "- **Delta**: Differenza di MAE tra stage (negativo = miglioramento)\n\n"
    
    # Per ogni configurazione
    for config_name in FEATURE_CONFIGURATIONS.keys():
        markdown += f"---\n\n## Configurazione: {config_name}\n\n"
        
        results = all_results[config_name]
        stage1 = results['stage1']
        stage2a = results['stage2a']
        stage2b = results['stage2b']
        
        # Stage 1
        markdown += f"### Stage 1 (Baseline - GW_Score Solo)\n\n"
        markdown += "| Modello | MAE | MSE | Accuracy | Spearman_r | Kendall_tau |\n"
        markdown += "|---------|-----|-----|----------|------------|-------------|\n"
        for model in ['RandomForest', 'GradientBoosting', 'SVR', 'Huber', 'Ensemble']:
            m = stage1[model]
            markdown += f"| {model:15s} | {m['MAE']:.4f} | {m['MSE']:.4f} | {m['Accuracy']:.2f}% | {m['Spearman_r']:.4f} | {m['Kendall_tau']:.4f} |\n"
        
        # Stage 2A
        markdown += f"\n### Stage 2A (Enriched Raw - GW_Score + 8 OT Features Non-Normalizzate)\n\n"
        markdown += "| Modello | MAE | MSE | Accuracy | Spearman_r | Kendall_tau |\n"
        markdown += "|---------|-----|-----|----------|------------|-------------|\n"
        for model in ['RandomForest', 'GradientBoosting', 'SVR', 'Huber', 'Ensemble']:
            m = stage2a[model]
            markdown += f"| {model:15s} | {m['MAE']:.4f} | {m['MSE']:.4f} | {m['Accuracy']:.2f}% | {m['Spearman_r']:.4f} | {m['Kendall_tau']:.4f} |\n"
        
        # Stage 2B
        markdown += f"\n### Stage 2B (Enriched Normalized - GW_Score + 8 OT Features Normalizzate)\n\n"
        markdown += "| Modello | MAE | MSE | Accuracy | Spearman_r | Kendall_tau |\n"
        markdown += "|---------|-----|-----|----------|------------|-------------|\n"
        for model in ['RandomForest', 'GradientBoosting', 'SVR', 'Huber', 'Ensemble']:
            m = stage2b[model]
            markdown += f"| {model:15s} | {m['MAE']:.4f} | {m['MSE']:.4f} | {m['Accuracy']:.2f}% | {m['Spearman_r']:.4f} | {m['Kendall_tau']:.4f} |\n"
        
        # Delta Analysis
        markdown += f"\n### Delta Analysis (2A vs 1 e 2B vs 1)\n\n"
        markdown += "| Modello | MAE Stage1 | MAE 2A | Delta 2A | % Improv 2A | MAE 2B | Delta 2B | % Improv 2B | Best |\n"
        markdown += "|---------|-----------|--------|----------|------------|--------|----------|------------|------|\n"
        for model in ['RandomForest', 'GradientBoosting', 'SVR', 'Huber', 'Ensemble']:
            mae1 = stage1[model]['MAE']
            mae2a = stage2a[model]['MAE']
            mae2b = stage2b[model]['MAE']
            
            delta2a = mae2a - mae1
            pct2a = (delta2a / mae1) * 100
            delta2b = mae2b - mae1
            pct2b = (delta2b / mae1) * 100
            
            best = "2A" if mae2a < mae2b else "2B" if mae2b < mae1 else "1"
            best_sym = f"{'↓' if best != '1' else '—'} {best}"
            
            markdown += f"| {model:15s} | {mae1:.4f} | {mae2a:.4f} | {delta2a:+.4f} | {pct2a:+.1f}% | {mae2b:.4f} | {delta2b:+.4f} | {pct2b:+.1f}% | {best_sym} |\n"
        
        markdown += "\n"
    
    # Summary statistics
    markdown += "---\n\n## Analisi Comparativa Globale\n\n"
    
    # Media MAE per stage
    all_mae_stage1 = []
    all_mae_stage2a = []
    all_mae_stage2b = []
    
    for config_name in FEATURE_CONFIGURATIONS.keys():
        results = all_results[config_name]
        for model in MODELS.keys():
            all_mae_stage1.append(results['stage1'][model]['MAE'])
            all_mae_stage2a.append(results['stage2a'][model]['MAE'])
            all_mae_stage2b.append(results['stage2b'][model]['MAE'])
        # Ensemble
        all_mae_stage1.append(results['stage1']['Ensemble']['MAE'])
        all_mae_stage2a.append(results['stage2a']['Ensemble']['MAE'])
        all_mae_stage2b.append(results['stage2b']['Ensemble']['MAE'])
    
    avg_mae_1 = np.mean(all_mae_stage1)
    avg_mae_2a = np.mean(all_mae_stage2a)
    avg_mae_2b = np.mean(all_mae_stage2b)
    
    markdown += f"| Stage | Mean MAE | Improv vs Stage1 |\n"
    markdown += f"|-------|----------|------------------|\n"
    markdown += f"| Stage 1 (Baseline) | {avg_mae_1:.4f} | — |\n"
    markdown += f"| Stage 2A (Raw) | {avg_mae_2a:.4f} | {((avg_mae_2a - avg_mae_1) / avg_mae_1 * 100):+.2f}% |\n"
    markdown += f"| Stage 2B (Normalized) | {avg_mae_2b:.4f} | {((avg_mae_2b - avg_mae_1) / avg_mae_1 * 100):+.2f}% |\n\n"
    
    # Insights
    markdown += "## Insights\n\n"
    if avg_mae_2b < avg_mae_2a < avg_mae_1:
        markdown += "✓ **Stage 2B (Normalized) è il migliore**: le OT features normalizzate forniscono il massimo beneficio.\n"
    elif avg_mae_2a < avg_mae_2b < avg_mae_1:
        markdown += "✓ **Stage 2A (Raw) è migliore di 2B**: le OT features senza normalizzazione sono più utili per questo dataset.\n"
    elif avg_mae_1 < avg_mae_2a and avg_mae_1 < avg_mae_2b:
        markdown += "⚠ **Stage 1 è migliore**: per questo dataset, le OT features degradano le prestazioni.\n"
    else:
        markdown += "— Risultati misti tra gli stage.\n"
    
    markdown += "\n"
    
    return markdown

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue lo studio di ablazione a due stage."""
    print("\n" + "="*80)
    print("TWO-STAGE ABLATION STUDY: GW_Score + OT Features Impact")
    print("="*80)
    print(f"Dataset: {', '.join(DATASETS)}")
    print(f"Configurazioni feature: {len(FEATURE_CONFIGURATIONS)}")
    print(f"Modelli: {len(MODELS)} + Ensemble")
    print(f"Stage: 3 (Baseline, Enriched Raw, Enriched Normalized)")
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
            results_config = process_configuration_all_stages(
                dataset, config_name, feature_list, None
            )
            all_results[config_name] = results_config
        
        # Genera Markdown
        print(f"\nGenerando Markdown...")
        markdown_content = format_two_stage_markdown(dataset, all_results)
        
        markdown_file = OUTPUT_DIR / f"{dataset}_two_stage_results.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        print(f"✓ Salvato: {markdown_file}")
    
    print("\n" + "="*80)
    print("ESPERIMENTO COMPLETATO!")
    print("="*80)
    print(f"\nOutput salvati in: {OUTPUT_DIR}/")
    for dataset in DATASETS:
        print(f"  - {dataset}_two_stage_results.md")

if __name__ == '__main__':
    main()
