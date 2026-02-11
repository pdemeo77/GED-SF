# Environment Setup Guide

This guide provides step-by-step instructions for setting up the computational environment needed to run the GED experiments.

## Prerequisites

- **Conda/Miniconda/Anaconda** installed on your system
  - Download from: https://docs.conda.io/en/latest/miniconda.html
  - Or use Anaconda: https://www.anaconda.com/download

## Quick Setup (Recommended)

### Option 1: Create Environment from YAML (Easiest)

```bash
# Clone the repository
git clone https://github.com/pdemeo77/GED_structural_features.git
cd GED_structural_features

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate ged_structural_features

# Verify installation
python -c "from ged_computation import extract_ot_features; print('✓ Setup complete!')"
```

### Option 2: Manual Environment Creation

```bash
# Create a new conda environment with Python 3.11
conda create -n ged_structural_features python=3.11 -y

# Activate the environment
conda activate ged_structural_features

# Install dependencies from conda-forge
conda install -c conda-forge numpy scipy networkx pandas scikit-learn matplotlib seaborn -y

# Install Optimal Transport library via pip
pip install POT tqdm

# Verify installation
python -c "from ged_computation import extract_ot_features; print('✓ Setup complete!')"
```

## Platform-Specific Instructions

### Windows (PowerShell)

```powershell
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate ged_structural_features

# Run experiments
python run_ot_features_experiments.py
```

### Linux / macOS (Bash)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate ged_structural_features

# Run experiments
python run_ot_features_experiments.py
```

## Verifying Your Installation

After creating the environment, verify that all components work correctly:

```bash
# Activate environment
conda activate ged_structural_features

# Test 1: Check Python version
python --version
# Expected: Python 3.11.x

# Test 2: Test core imports
python -c "import numpy, scipy, networkx, pandas, sklearn; print('✓ Core libraries OK')"

# Test 3: Test Optimal Transport
python -c "import ot; print('✓ POT library OK')"

# Test 4: Test visualization
python -c "import matplotlib, seaborn; print('✓ Visualization libraries OK')"

# Test 5: Test project modules
python -c "from ged_computation import compute_ged_GW, extract_ot_features; print('✓ Project modules OK')"

# Test 6: Run quick feature extraction test
python -c "
import networkx as nx
from ged_computation import compute_ged_GW, extract_ot_features

# Create two simple graphs
G1 = nx.karate_club_graph()
G2 = nx.karate_club_graph()

# Compute GED and extract features
fgw_dist, coupling = compute_ged_GW(G1, G2, node_feature='degree')
features = extract_ot_features(coupling, G1, G2)
print(f'✓ Extracted {len(features)} OT features successfully!')
print(f'  Features: {list(features.keys())}')
"
```

Expected output:
```
✓ Extracted 8 OT features successfully!
  Features: ['ot_alignment_entropy', 'ot_alignment_confidence', 'ot_transport_cost', 'ot_marginal_balance', 'ot_coupling_sparsity', 'ot_max_coupling', 'ot_coupling_variance', 'ot_structural_mismatch']
```

## Running Your First Experiment

Once the environment is set up, you can run the optimal transport features experiments:

```bash
# Activate environment
conda activate ged_structural_features

# Run OT features experiments (produces breakthrough results)
python run_ot_features_experiments.py

# Results will be saved in: ot_features_experiments/
```

**Expected runtime:** ~30-60 minutes depending on your hardware

## Package Versions

The environment includes:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11.x | Core language |
| NumPy | ≥1.26.0 | Numerical computing |
| SciPy | ≥1.11.0 | Scientific computing |
| NetworkX | ≥3.1 | Graph data structures |
| Pandas | ≥2.2.0 | Data manipulation |
| Scikit-learn | ≥1.4.0 | Machine learning |
| POT | ≥0.9.0 | Optimal Transport |
| Matplotlib | ≥3.8.0 | Plotting |
| Seaborn | ≥0.13.0 | Statistical visualization |
| tqdm | ≥4.66.0 | Progress bars |

## Managing Your Environment

### List all conda environments
```bash
conda env list
```

### Activate the environment
```bash
conda activate ged_structural_features
```

### Deactivate the environment
```bash
conda deactivate
```

### Update the environment (after pulling new changes)
```bash
conda env update -f environment.yml --prune
```

### Export your environment (to share with others)
```bash
conda env export > environment_exact.yml
```

### Remove the environment (if needed)
```bash
conda env remove -n ged_structural_features
```

## Troubleshooting

### Issue: "conda: command not found"
**Solution:** Conda is not installed or not in your PATH. Install Miniconda/Anaconda and restart your terminal.

### Issue: "Solving environment: failed with initial frozen solve"
**Solution:** Update conda and try again:
```bash
conda update -n base conda
conda env create -f environment.yml
```

### Issue: POT installation fails
**Solution:** Install POT separately with pip:
```bash
conda activate ged_structural_features
pip install POT --no-cache-dir
```

### Issue: Import errors when running scripts
**Solution:** Ensure the environment is activated:
```bash
conda activate ged_structural_features
python -c "import sys; print(sys.executable)"
# Should show path with 'ged_structural_features' in it
```

### Issue: "Module not found" errors
**Solution:** Reinstall dependencies:
```bash
conda activate ged_structural_features
conda install -c conda-forge numpy scipy networkx pandas scikit-learn matplotlib seaborn -y
pip install POT tqdm
```

## Testing on a New Machine

After cloning the repository on a new machine:

```bash
# 1. Clone repository
git clone https://github.com/pdemeo77/GED_structural_features.git
cd GED_structural_features

# 2. Create environment
conda env create -f environment.yml

# 3. Activate environment
conda activate ged_structural_features

# 4. Run verification tests
python -c "from ged_computation import extract_ot_features; print('✓ Ready!')"

# 5. Run a quick experiment to test everything
python run_ot_features_experiments.py

# 6. Check results
ls ot_features_experiments/
```

## Hardware Requirements

- **Minimum:**
  - 4 GB RAM
  - 2 CPU cores
  - 2 GB disk space

- **Recommended:**
  - 8+ GB RAM (for larger graphs)
  - 4+ CPU cores (parallel processing)
  - 5 GB disk space (including datasets)

- **For full experiments:**
  - 16 GB RAM
  - 8+ CPU cores
  - 10 GB disk space

## Performance Notes

- **OT features experiments:** ~30-60 minutes on a modern laptop
- **Betweenness experiments:** ~2-4 hours (500× slower due to betweenness computation)
- **Memory usage:** Typically 1-4 GB depending on dataset size

## Alternative: Using pip only (without conda)

If you prefer to use pip with a Python virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from ged_computation import extract_ot_features; print('✓ Ready!')"
```

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Verify your conda installation: `conda --version`
3. Check Python version: `python --version` (should be 3.11.x)
4. Create an issue on GitHub with:
   - Your OS and version
   - Conda version
   - Python version
   - Complete error message

## Next Steps

Once your environment is set up:

1. ✅ Verify installation with test scripts above
2. 📊 Run OT features experiments: `python run_ot_features_experiments.py`
3. 📈 Generate figures: `python generate_publication_figures.py`
4. 📖 Read `docs/COMPREHENSIVE_RESULTS_WRITEUP.md` for detailed analysis
5. 🎯 Adapt the code for your own graph datasets

---

**Environment ready? Start exploring the breakthrough results! 🚀**
