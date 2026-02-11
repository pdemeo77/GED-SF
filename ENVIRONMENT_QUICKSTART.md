# Quick Reference: Environment Setup

## For You (Current Machine)
```bash
# Already verified ✓
python verify_environment.py
# Output: ✓ ALL CHECKS PASSED (12/12)
```

## For Another Machine (Testing Reproduction)

### Windows
```powershell
# 1. Clone repository
git clone https://github.com/pdemeo77/GED_structural_features.git
cd GED_structural_features

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate ged_structural_features

# 4. Verify setup
python verify_environment.py
# Expected: ✓ ALL CHECKS PASSED (12/12)

# 5. Run experiments
python run_ot_features_experiments.py
```

### Linux / macOS
```bash
# 1. Clone repository
git clone https://github.com/pdemeo77/GED_structural_features.git
cd GED_structural_features

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate ged_structural_features

# 4. Verify setup
python verify_environment.py
# Expected: ✓ ALL CHECKS PASSED (12/12)

# 5. Run experiments
python run_ot_features_experiments.py
```

## What's Included in environment.yml

- **Python 3.11** (compatible with 3.11+)
- **NumPy, SciPy** (numerical computing)
- **NetworkX** (graph structures)
- **Pandas** (data manipulation)
- **Scikit-learn** (machine learning)
- **POT** (Optimal Transport)
- **Matplotlib, Seaborn** (visualization)
- **tqdm** (progress bars, optional)

## Troubleshooting

### Problem: "conda: command not found"
**Solution:** Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

### Problem: "Solving environment: failed"
**Solution:** 
```bash
conda update -n base conda
conda env create -f environment.yml
```

### Problem: POT installation fails
**Solution:**
```bash
conda activate ged_structural_features
pip install POT --no-cache-dir
```

### Problem: Import errors
**Solution:**
```bash
# Ensure environment is activated
conda activate ged_structural_features
python -c "import sys; print(sys.executable)"
# Should show path with 'ged_structural_features' in it
```

## Verification Checklist

Run `python verify_environment.py` and check for:

- ✓ Python 3.11+ installed
- ✓ All 8 core packages installed
- ✓ Project modules import successfully
- ✓ OT features extraction works
- ✓ 8 features extracted: 
  - ot_alignment_entropy
  - ot_alignment_confidence
  - ot_transport_cost
  - ot_marginal_balance
  - ot_coupling_sparsity
  - ot_max_coupling
  - ot_coupling_variance
  - ot_structural_mismatch

## Expected Verification Output

```
============================================================
GED Structural Features - Environment Verification
============================================================

1. Checking Python Version...
✓ Python 3.11.x OK

2. Checking Core Dependencies...
✓ numpy                1.26.x
✓ scipy                1.11.x
✓ networkx             3.1.x
✓ pandas               2.2.x
✓ scikit-learn         1.4.x
✓ POT                  0.9.x
✓ matplotlib           3.8.x
✓ seaborn              0.13.x

   Optional packages:
  ○ tqdm                (optional - for progress bars)

3. Testing Project Modules...
✓ ged_computation module OK
✓ make_simulation module OK

4. Testing OT Features Extraction...
✓ OT Features Extraction Test PASSED
  Extracted 8 features: [...]

============================================================
✓ ALL CHECKS PASSED (12/12)

🎉 Environment is ready! You can now run experiments.
```

## Files Reference

- **`environment.yml`** - Conda environment specification (one-command setup)
- **`ENVIRONMENT_SETUP.md`** - Comprehensive guide (troubleshooting, details)
- **`verify_environment.py`** - Automated verification script
- **`requirements.txt`** - Pip-compatible dependencies (alternative to conda)

## Quick Commands

```bash
# List environments
conda env list

# Activate
conda activate ged_structural_features

# Deactivate
conda deactivate

# Update environment
conda env update -f environment.yml --prune

# Remove environment
conda env remove -n ged_structural_features

# Export exact environment (to share)
conda env export > environment_exact.yml
```

---

**Complete Guide:** See `ENVIRONMENT_SETUP.md` for full documentation
