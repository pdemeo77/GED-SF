#!/usr/bin/env python
"""
Script di Verifica Rapida dell'Ambiente
========================================

SCOPO:
======
Testa tutte le dipendenze richieste e la funzionalità core del progetto.

COSA CONTROLLA:
================
1. Versione Python (richiede 3.11+)
2. Pacchetti principali:
   - networkx: Libreria per grafi
   - numpy: Algebra lineare e array numerici
   - pandas: Analisi e manipolazione dati
   - scikit-learn: Machine learning
   - POT (Python Optimal Transport): Algoritmi di trasporto ottimale
   - scipy: Funzioni scientifiche

3. Funzionalità core:
   - Caricamento dei grafi
   - Calcolo dei feature strutturali
   - Estrazione delle OT features
   - Computation della distanza FGW

UTILIZZO:
=========
    python verify_environment.py

OUTPUT ATTESO:
==============
✓ Python 3.11+ OK
✓ networkx VERSION_NUMBER
✓ numpy VERSION_NUMBER
✓ pandas VERSION_NUMBER
✓ scikit-learn VERSION_NUMBER
✓ POT VERSION_NUMBER
✓ scipy VERSION_NUMBER
✓ Core functionality test PASSED

SE QUALCOSA FALLISCE:
=====================
1. Controlla che l'ambiente conda/venv sia attivato
2. Esegui: pip install -r requirements.txt
3. Se POT non installa, usa: pip install POT
4. Controlla la compatibilità Python (deve essere 3.11+)
"""

import sys
import importlib

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Need 3.11+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} NOT FOUND - {str(e)}")
        return False

def test_ot_features():
    """Test OT features extraction."""
    try:
        import networkx as nx
        from ged_computation import (
            compute_ged_GW, 
            extract_ot_features,
            compute_cross_matrix_with_structural_features
        )
        
        # Create two simple graphs
        G1 = nx.karate_club_graph()
        G2 = nx.karate_club_graph()
        
        # Compute cross matrix
        cross_matrix = compute_cross_matrix_with_structural_features(G1, G2, list_of_centrality_indices=['Deg'])
        
        # Get adjacency matrices
        C1 = nx.to_numpy_array(G1)
        C2 = nx.to_numpy_array(G2)
        
        # Compute GED and extract features
        fgw_dist, coupling = compute_ged_GW(G1, G2, cross_matrix)
        features = extract_ot_features(coupling, cross_matrix, C1, C2)
        
        if len(features) == 8:
            print(f"\n✓ OT Features Extraction Test PASSED")
            print(f"  Extracted {len(features)} features: {list(features.keys())}")
            return True
        else:
            print(f"\n✗ OT Features Extraction Test FAILED - Expected 8 features, got {len(features)}")
            return False
            
    except Exception as e:
        print(f"\n✗ OT Features Extraction Test FAILED - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("GED Structural Features - Environment Verification")
    print("=" * 60)
    
    print("\n1. Checking Python Version...")
    checks = [check_python_version()]
    
    print("\n2. Checking Core Dependencies...")
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('networkx', 'networkx'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('POT', 'ot'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    for package_name, import_name in packages:
        checks.append(check_package(package_name, import_name))
    
    # Check optional packages (don't fail if missing)
    print("\n   Optional packages:")
    try:
        import tqdm
        print(f"  ✓ tqdm                {tqdm.__version__} (optional)")
    except ImportError:
        print(f"  ○ tqdm                not installed (optional - for progress bars)")
    
    print("\n3. Testing Project Modules...")
    try:
        from ged_computation import compute_ged_GW, extract_ot_features
        print("✓ ged_computation module OK")
        checks.append(True)
    except ImportError as e:
        print(f"✗ ged_computation module FAILED - {str(e)}")
        checks.append(False)
    
    try:
        from make_simulation import make_simulation
        print("✓ make_simulation module OK")
        checks.append(True)
    except ImportError as e:
        print(f"✗ make_simulation module FAILED - {str(e)}")
        checks.append(False)
    
    print("\n4. Testing OT Features Extraction...")
    checks.append(test_ot_features())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        print("\n🎉 Environment is ready! You can now run experiments.")
        print("\nNext steps:")
        print("  • Run experiments: python run_ot_features_experiments.py")
        print("  • Run ablation: python run_structural_features_ablation.py")
        print("  • Read docs: REPRODUCTION_GUIDE.md")
        return 0
    else:
        print(f"✗ SOME CHECKS FAILED ({passed}/{total} passed)")
        print("\nPlease fix the issues above before running experiments.")
        print("See ENVIRONMENT_SETUP.md for troubleshooting help.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
