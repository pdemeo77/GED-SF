"""
Esperimenti con Feature di Trasporto Ottimale
==============================================

MOTIVAZIONE:
============
Questo script implementa un esperimento avanzato che combina:
1. Feature Gromov-Wasserstein (GW_Score + OT features da FGW)
2. Modelli di machine learning sophisticated (GB, RF, Huber, SVR)
3. Ensemble learning per maggiore robustezza

IPOTESI:
--------
Aggiungendo le 8 feature derivate dalla matrice di coupling ottimale,
possiamo migliorare significativamente la previsione del GED vero.

Precedentemente:
    Utilizziamo solo: [GW_Score]
    Risultato: MAE ≈ 1.5 - 2.0

Con questo metodo:
    Utilizziamo: [GW_Score, ot_alignment_entropy, ot_alignment_confidence, 
                  ot_transport_cost, ot_marginal_balance, ot_coupling_sparsity,
                  ot_max_coupling, ot_coupling_variance, ot_structural_mismatch]
    Risultato atteso: MAE ≈ 1.3 - 1.7 (+8-10% di miglioramento)

MIGLIORI CONFIGURAZIONI TESTATE:
==================================
AIDS:
    - Feature: PageRank only
    - Modello: GradientBoosting
    - MAE migliore: 1.537
    
IMDB:
    - Feature: Degree + Clustering Coefficient
    - Modello: RandomForest
    - MAE migliore: 0.131
    
Linux:
    - Feature: PageRank only
    - Modello: GradientBoosting
    - MAE migliore: 0.347

INTERESSANTE OSSERVAZIONE:
==========================
Il betweenness centrality NON aiuta e a volte peggiora i risultati!
Motivo possibile: 
    - Betweenness è approssimato, introduce rumore
    - Betweenness è computazionalmente costoso per grafi grandi
    - Grado e PageRank catturano già l'informazione importante

MODELLI TESTATI:
================
1. GradientBoosting: Spesso migliore per interpretabilità + precisione
2. RandomForest: Buona robustezza, veloce da addestrare
3. Huber: Robusto agli outlier, semplice
4. SVR: Baseline classico del machine learning

ENSEMBLE FINALE:
================
Combina i 3 migliori modelli:
    y_pred = 0.5 * RF + 0.3 * GB + 0.2 * Huber
Generalmente più robusto dei singoli modelli.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings('ignore')

# Import project modules
from make_simulation import make_simulation
from utils import build_ged_ground_truth_dataframe

# ============================================================================
# CONFIGURAZIONE - Usando le MIGLIORI configurazioni da esperimenti precedenti
# ============================================================================

N_SAMPLES = 2000  # Numero di coppie di grafi da testare per dataset
OUTPUT_DIR = "ot_features_experiments"  # Cartella dove salvare risultati
RANDOM_STATE = 42  # Per riproducibilità

# MIGLIORI configurazioni (senza betweenness!)
# Questi sono i risultati di hyperparameter tuning estensivo
BEST_CONFIGS = {
    'AIDS': ['PR'],              # Solo PageRank - MAE migliore: 1.537
    'IMDB': ['Deg', 'CC'],       # Degree + Clustering - MAE migliore: 0.131
    'Linux': ['PR']              # Solo PageRank - MAE migliore: 0.347
}

DATASETS = ['AIDS', 'IMDB', 'Linux']

# Configurazioni dei modelli provate estensivamente
MODELS = {
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,      # Numero di alberi
        learning_rate=0.1,     # Tasso di apprendimento
        max_depth=5,           # Profondità massima
        random_state=RANDOM_STATE,
        loss='squared_error'
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,          # Alberi più profondi per complessità maggiore
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1  # Usa tutti i core della CPU
    ),
    'Huber': HuberRegressor(
        epsilon=1.35,          # Robusto agli outlier
        max_iter=200,
        alpha=0.0001           # Regolarizzazione L2
    ),
    'SVR': SVR(
        kernel='rbf',          # Kernel gaussiano
        C=10,                  # Forza di regolarizzazione
        gamma='scale',         # Parametro del kernel
        epsilon=0.1            # Larghezza dell'epsilon-tube
    )
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_output_directory():
    """Create output directory structure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/text_results", exist_ok=True)
    print(f"✓ Output directory created: {OUTPUT_DIR}/\n")

def load_and_prepare_data(dataset_name, indices, mu=0.5):
    """
    Load or generate data for a dataset with OT features.
    
    Returns DataFrame with columns:
    - GW_Score
    - ot_alignment_entropy
    - ot_alignment_confidence
    - ot_transport_cost
    - ot_marginal_balance
    - ot_coupling_sparsity
    - ot_max_coupling
    - ot_coupling_variance
    - ot_structural_mismatch
    - True_GED
    """
    # Sort indices alphabetically for consistent filenames
    indices_sorted = sorted(indices) if indices else []
    indices_str = "_".join(indices_sorted) if indices_sorted else "Baseline"
    
    # Check if data already exists
    filename = f"{OUTPUT_DIR}/{dataset_name}_with_structural_features_{indices_str}.csv"
    
    if not os.path.exists(filename):
        print(f"    Generating data for {dataset_name} with {indices_str}...")
        make_simulation(
            n_sample=N_SAMPLES,
            output_dir=OUTPUT_DIR,
            list_of_indices=indices,
            mu=mu
        )
    
    # Load the data (already includes True_GED from make_simulation)
    df = pd.read_csv(filename)
    
    return df

def train_model_with_ot_features(X_train, y_train, X_test, y_test, model_name, dataset_name):
    """
    Train a model using GW_Score + OT features.
    """
    from sklearn.base import clone
    
    model = clone(MODELS[model_name])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (within ±1 edit distance)
    accuracy = np.mean(np.abs(y_pred - y_test) <= 1) * 100
    
    # Calculate rank correlations
    spearman_corr, _ = spearmanr(y_test, y_pred)
    kendall_corr, _ = kendalltau(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': accuracy,
        'Spearman': spearman_corr,
        'Kendall': kendall_corr
    }
    
    return model, y_pred, metrics

def train_ensemble(X_train, y_train, X_test, y_test, dataset_name):
    """Train ensemble with 3 best models."""
    # Train individual models
    models_to_ensemble = ['RandomForest', 'GradientBoosting', 'Huber']
    predictions = []
    
    for model_name in models_to_ensemble:
        model, y_pred, _ = train_model_with_ot_features(
            X_train, y_train, X_test, y_test, model_name, dataset_name
        )
        predictions.append(y_pred)
    
    # Weighted ensemble (same weights as before)
    y_pred_ensemble = (
        0.5 * predictions[0] +  # RandomForest
        0.3 * predictions[1] +  # GradientBoosting
        0.2 * predictions[2]    # Huber
    )
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_ensemble)
    mse = mean_squared_error(y_test, y_pred_ensemble)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_ensemble)
    accuracy = np.mean(np.abs(y_pred_ensemble - y_test) <= 1) * 100
    spearman_corr, _ = spearmanr(y_test, y_pred_ensemble)
    kendall_corr, _ = kendalltau(y_test, y_pred_ensemble)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': accuracy,
        'Spearman': spearman_corr,
        'Kendall': kendall_corr
    }
    
    return None, y_pred_ensemble, metrics

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_ot_experiments():
    """Run experiments with OT features on best configurations."""
    print("\n" + "="*70)
    print("OPTIMAL TRANSPORT FEATURES EXPERIMENTS")
    print("Testing OT-enhanced predictions on BEST configurations")
    print("="*70)
    print(f"Sample size: {N_SAMPLES} graphs per dataset")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Models: {', '.join(MODELS.keys())} + Ensemble")
    print(f"\nBest Configurations:")
    for dataset, indices in BEST_CONFIGS.items():
        print(f"  {dataset}: {' + '.join(indices)}")
    print("="*70)
    
    results = []
    total_experiments = len(DATASETS) * (len(MODELS) + 1)  # +1 for ensemble
    current_exp = 0
    
    for dataset_name in DATASETS:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Configuration: {' + '.join(BEST_CONFIGS[dataset_name])}")
        print(f"{'='*70}")
        
        # Load data with OT features
        df = load_and_prepare_data(dataset_name, BEST_CONFIGS[dataset_name], mu=0.5)
        
        # Prepare features: GW_Score + all OT features
        ot_feature_cols = [col for col in df.columns if col.startswith('ot_')]
        feature_cols = ['GW_Score'] + ot_feature_cols
        
        X = df[feature_cols].values
        y = df['True_GED'].values
        
        print(f"  Features: GW_Score + {len(ot_feature_cols)} OT features")
        print(f"  Total samples: {len(df)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Train individual models
        for model_name in MODELS.keys():
            current_exp += 1
            print(f"  [{current_exp}/{total_experiments}] Training {model_name}...", end=' ')
            
            model, y_pred, metrics = train_model_with_ot_features(
                X_train, y_train, X_test, y_test, model_name, dataset_name
            )
            
            print(f"MAE: {metrics['MAE']:.3f}")
            
            results.append({
                'Dataset': dataset_name,
                'Configuration': '+'.join(BEST_CONFIGS[dataset_name]),
                'Model': f'{model_name}_OT',
                **metrics
            })
        
        # Train ensemble
        current_exp += 1
        print(f"  [{current_exp}/{total_experiments}] Training Ensemble...", end=' ')
        
        _, y_pred_ensemble, metrics = train_ensemble(
            X_train, y_train, X_test, y_test, dataset_name
        )
        
        print(f"MAE: {metrics['MAE']:.3f} ⭐")
        
        results.append({
            'Dataset': dataset_name,
            'Configuration': '+'.join(BEST_CONFIGS[dataset_name]),
            'Model': 'Ensemble_OT',
            **metrics
        })
    
    return pd.DataFrame(results)

def generate_comparison_report(df_results):
    """Generate comparison report with previous best results."""
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS BEST RESULTS")
    print("="*70)
    
    # Previous best results (without OT features)
    previous_best = {
        'AIDS': {'config': 'PR', 'model': 'Huber', 'mae': 1.537},
        'IMDB': {'config': 'Deg+CC', 'model': 'GradientBoosting', 'mae': 0.131},
        'Linux': {'config': 'PR', 'model': 'RandomForest', 'mae': 0.347}
    }
    
    for dataset in DATASETS:
        print(f"\n{dataset}:")
        prev = previous_best[dataset]
        
        # Get best result with OT features
        dataset_results = df_results[df_results['Dataset'] == dataset]
        best_ot = dataset_results.nsmallest(1, 'MAE').iloc[0]
        
        print(f"  Previous Best (no OT): {prev['config']} + {prev['model']}")
        print(f"    MAE: {prev['mae']:.3f}")
        
        print(f"  New Best (with OT):    {best_ot['Configuration']} + {best_ot['Model']}")
        print(f"    MAE: {best_ot['MAE']:.3f}")
        
        improvement = ((prev['mae'] - best_ot['MAE']) / prev['mae']) * 100
        print(f"  Improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"  Status: ✓ OT features help! 🎉")
        else:
            print(f"  Status: ✗ No improvement")

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Setup
    setup_output_directory()
    
    # Run experiments
    print("\n🚀 Starting OT features experiments...\n")
    df_results = run_ot_experiments()
    
    # Save results
    csv_path = f"{OUTPUT_DIR}/ot_features_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate comparison report
    generate_comparison_report(df_results)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Experiments complete!")
    print(f"  Total experiments: {len(df_results)}")
    print(f"  Runtime: {elapsed/60:.1f} minutes")
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
