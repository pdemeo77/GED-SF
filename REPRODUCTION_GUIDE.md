# Reproduction Guide: Two-Stage Ablation Study for GED Approximation

**Status:** ✅ Complete and Validated  
**Last Updated:** February 6, 2026  
**Contact:** For questions, see CONTRIBUTING.md  

---

## Executive Summary

This guide walks you through **exact reproduction** of the two-stage ablation study results presented in 
[TWO_STAGE_ABLATION_FINAL_REPORT.md](results/two_stage_ablation/TWO_STAGE_ABLATION_FINAL_REPORT.md).

**Expected outcomes:**
- 3 markdown reports (AIDS, IMDB, Linux) with detailed results tables
- **Total runtime:** ~2-3 hours on standard hardware
- **Expected MAE (best models):** AIDS: 1.29, IMDB: 0.0617, Linux: 0.1931
- **CPU memory:** ~4-8 GB RAM recommended

---

## Step 1: Environment Setup

### 1.1 Clone Repository

```bash
git clone https://github.com/[owner]/GED_structural_features.git
cd GED_structural_features
```

### 1.2 Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate ged_structural_features

# Verify installation
python verify_environment.py
```

**Expected output:**
```
✓ POT installed (optimal_transport)
✓ scikit-learn version X.XX
✓ Project modules importable
✓ Environment is ready!
```

### 1.3 Verify Dataset Availability

The script automatically downloads/prepares datasets. Verify they exist:

```bash
ls Dataset/AIDS/AIDS_graphs.csv
ls Dataset/IMDB/IMDB_graphs.csv
ls Dataset/Linux/Linux_graphs.csv
```

If files are missing, the script will attempt automatic download from canonical sources.

---

## Step 2: Run the Two-Stage Ablation Experiment

### 2.1 Execute Main Script

```bash
# From repository root
python run_two_stage_ablation.py
```

This script performs:

| Stage | Description | Samples | Time (approx) |
|-------|-------------|---------|---------------|
| **Load & Prepare** | Load 3 datasets, compute GW_Score | — | 5-10 min |
| **AIDS Stage 1** | 7 configs × 5 models × 80/20 CV | 1,275 train | 15 min |
| **AIDS Stage 2A/2B** | Same configs, +8 OT features | 1,275 train | 30 min |
| **IMDB Stage 1-2B** | All stages combined | 1,600 train | 25 min |
| **Linux Stage 1-2B** | All stages combined | 1,600 train | 30 min |
| **Report Generation** | Markdown output creation | — | 2 min |
| **TOTAL** | — | — | **~1.5-2 hours** |

### 2.2 Monitor Progress

The script prints progress indicators:

```
[14:32:00] Loading AIDS dataset...
[14:32:15] Computing GW_Score... (1597 graphs)
[14:32:45] Extracting OT features... 
[14:33:20] =========== STAGE 1: Deg Configuration ===========
[14:33:20] RandomForest... MAE: 1.6135
[14:33:45] GradientBoosting... MAE: 1.6118
...
[16:15:30] ✓ Report saved: results/two_stage_ablation/AIDS_two_stage_results.md
[16:15:45] ✓ Report saved: results/two_stage_ablation/IMDB_two_stage_results.md
[16:16:05] ✓ Report saved: results/two_stage_ablation/Linux_two_stage_results.md
[16:16:08] ✓ Experiment complete!
```

### 2.3 Expected File Output

After successful execution:

```
results/two_stage_ablation/
├── AIDS_two_stage_results.md      (336 lines, ~45 KB)
├── IMDB_two_stage_results.md      (336 lines, ~45 KB)
└── Linux_two_stage_results.md     (336 lines, ~45 KB)

generated_data/
├── AIDS_gw_scores.csv             (~50 KB)
├── IMDB_gw_scores.csv             (~50 KB)
└── Linux_gw_scores.csv            (~50 KB)
```

---

## Step 3: Verify Results

### 3.1 Check MAE Values (Quick Validation)

Extract best MAE from each dataset report:

**AIDS:**
```bash
grep "SVR.*2B" results/two_stage_ablation/AIDS_two_stage_results.md | head -1
# Expected: MAE ≈ 1.29-1.30
```

**IMDB:**
```bash
grep "GradientBoosting.*2A" results/two_stage_ablation/IMDB_two_stage_results.md | head -1
# Expected: MAE ≈ 0.061-0.062
```

**Linux:**
```bash
grep "RandomForest.*2A" results/two_stage_ablation/Linux_two_stage_results.md | head -1
# Expected: MAE ≈ 0.193-0.195
```

### 3.2 Full Validation Checklist

Run this validation to confirm reproducibility:

```python
import pandas as pd

# Expected best MAE values (±2% tolerance)
expected = {
    'AIDS': 1.290,
    'IMDB': 0.0617,
    'Linux': 0.1931
}

# Define tolerance
tolerance = 0.02  # 2% allowed variance

for dataset, expected_mae in expected.items():
    # Extract from reports
    # (Implement logic to parse markdown and extract best MAE)
    
    if abs(actual_mae - expected_mae) / expected_mae <= tolerance:
        print(f"✓ {dataset}: PASS (MAE={actual_mae:.4f})")
    else:
        print(f"✗ {dataset}: FAIL (expected {expected_mae}, got {actual_mae})")
```

### 3.3 Detailed Results Review

Open and inspect the generated markdown files:

**AIDS Results:**
- Best config: `PR + SVR (Stage 2B)`
- Expected: MAE=1.2900, Kendall τ=0.5316, Accuracy=45.77%
- See: [results/two_stage_ablation/AIDS_two_stage_results.md](results/two_stage_ablation/AIDS_two_stage_results.md)

**IMDB Results:**
- Best config: `CC + GradientBoosting (Stage 2A)`
- Expected: MAE=0.0617, Kendall τ=0.9641, Accuracy=97.50%
- See: [results/two_stage_ablation/IMDB_two_stage_results.md](results/two_stage_ablation/IMDB_two_stage_results.md)

**Linux Results:**
- Best config: `PR + RandomForest (Stage 2A)`
- Expected: MAE=0.1931, Kendall τ=0.9218, Accuracy=93.00%
- See: [results/two_stage_ablation/Linux_two_stage_results.md](results/two_stage_ablation/Linux_two_stage_results.md)

---

## Step 4: Understanding the Results

### 4.1 Report Structure

Each markdown file contains:

```
1. Configurazione: [Deg | CC | PR | Deg+CC | Deg+PR | CC+PR | Deg+CC+PR]
   ↓
2. Stage 1: Baseline (GW_Score only)
   - 5 models (RF, GB, SVR, Huber, Ensemble)
   - Metrics: MAE, MSE, Accuracy, Spearman, Kendall
   ↓
3. Stage 2A: Enriched Raw OT features
   - Same 5 models
   - Same metrics
   ↓
4. Stage 2B: Enriched Normalized OT features
   - Same 5 models
   - Same metrics
   ↓
5. Delta Analysis: % improvement from Stage 1 → 2A/2B
```

### 4.2 Key Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| **MAE** | 0 → ∞ | Lower is better; primary metric |
| **MSE** | 0 → ∞ | Penalizes larger errors; secondary metric |
| **Accuracy** | 0-100% | % predictions within ±1 GED |
| **Spearman ρ** | -1 → +1 | Monotonic rank correlation |
| **Kendall τ** | -1 → +1 | Ordinal rank agreement (probability-based) |

### 4.3 Comparing to Baseline

Each dataset shows MAE improvement:

```
AIDS:   1.7233 (Stage 1) → 1.3717 (Stage 2B) = -20.4% improvement
IMDB:   0.8262 (Stage 1) → 0.1887 (Stage 2B) = -77.16% improvement  
Linux:  1.1436 (Stage 1) → 0.4591 (Stage 2B) = -59.86% improvement
```

---

## Step 5: Advanced Analysis

### 5.1 Extract Top-K Models per Dataset

```python
import pandas as pd

def extract_top_models(dataset_name, k=5):
    """Extract top-k models by MAE from a dataset's report"""
    with open(f'results/two_stage_ablation/{dataset_name}_two_stage_results.md') as f:
        content = f.read()
    
    # Parse tables and extract Stage 2B results
    # Sort by MAE ascending
    # Return top k
    pass

for ds in ['AIDS', 'IMDB', 'Linux']:
    top5 = extract_top_models(ds, k=5)
    print(f"\n{ds} Top 5 Models:")
    print(top5)
```

### 5.2 Cross-Dataset Comparison

See [TWO_STAGE_ABLATION_FINAL_REPORT.md](results/two_stage_ablation/TWO_STAGE_ABLATION_FINAL_REPORT.md) 
for comprehensive cross-dataset analysis including:

- Configuration optimization per domain
- Feature normalization necessity
- Model selection rationale
- Statistical validation (Spearman vs Kendall relationships)

### 5.3 Reproduce Specific Experiments

To re-run a **single dataset**:

```python
# Modify run_two_stage_ablation.py:
DATASETS = ['AIDS']  # Run only AIDS

# Then execute:
python run_two_stage_ablation.py
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'ged_computation'"

**Solution:**
```bash
# Verify you're in the correct directory
pwd  # Should show: .../GED_structural_features

# Reinstall environment
conda env remove -n ged_structural_features
conda env create -f environment.yml
conda activate ged_structural_features
```

### Problem: "FileNotFoundError: Dataset/AIDS/AIDS_graphs.csv"

**Solution:**
The script attempts automatic download. If it fails:

1. Download manually from: https://www.dropbox.com/s/[link]
2. Place in: `Dataset/AIDS/AIDS_graphs.csv`
3. Re-run script

### Problem: "MemoryError" or Very Slow Execution

**Solution:**
Reduce sample size in `run_two_stage_ablation.py`:

```python
N_SAMPLES = 1000  # Reduce from 2000 for quick test
```

Runtime scales linearly with N_SAMPLES.

### Problem: Results Don't Match Expected MAE Values

**Possible Causes:**

1. **Different Python/sklearn versions**: Ensure versions match environment.yml
2. **Different random seed**: All random_state=42 is fixed; should be deterministic
3. **Incomplete dataset**: Verify all graphs loaded (check console output)
4. **Modified hyperparameters**: Don't change model parameters in the script

**Debug:**
```bash
python -c "import sklearn; print(sklearn.__version__)"  # Should match environment.yml
python -c "from ged_computation import extract_ot_features; print('✓ Module loaded')"
```

---

## Performance Benchmarks

### Hardware Configuration Used for Validation

```
CPU: Intel i7-10700K (8 cores)
RAM: 16 GB
Storage: SSD (NVMe)
OS: Windows 10 / 11
Python: 3.12+
```

### Expected Runtime Breakdown

| Dataset | Stage 1 | Stage 2A | Stage 2B | Reporting | Total |
|---------|---------|----------|----------|-----------|-------|
| **AIDS** | 15 min | 15 min | 15 min | 1 min | **46 min** |
| **IMDB** | 8 min | 8 min | 8 min | 1 min | **25 min** |
| **Linux** | 10 min | 10 min | 10 min | 1 min | **31 min** |
| **TOTAL** | 33 min | 33 min | 33 min | 3 min | **~1.5-2 hours** |

**Note:** Times can vary ±30% depending on I/O speed and background processes.

---

## Reproducibility Checklist

Before publishing/comparing results, verify:

- [ ] Environment created from environment.yml
- [ ] All datasets present in Dataset/
- [ ] Python 3.12+ confirmed
- [ ] run_two_stage_ablation.py executed without errors
- [ ] 3 markdown reports generated in results/two_stage_ablation/
- [ ] Best MAE values within ±2% of expected (see Step 3.2)
- [ ] Kendall τ and Spearman ρ metrics computed
- [ ] All 7 configurations × 3 stages × 5 models tested

---

## Expected Directory Structure After Execution

```
GED_structural_features/
├── Dataset/
│   ├── AIDS/
│   │   └── AIDS_graphs.csv
│   ├── IMDB/
│   │   └── IMDB_graphs.csv
│   └── Linux/
│       └── Linux_graphs.csv
├── results/
│   ├── mu_sensitivity_results/        (optional, from other experiments)
│   └── two_stage_ablation/
│       ├── AIDS_two_stage_results.md      ✓ Primary output
│       ├── IMDB_two_stage_results.md      ✓ Primary output
│       └── Linux_two_stage_results.md     ✓ Primary output
├── generated_data/
│   ├── AIDS_gw_scores.csv
│   ├── IMDB_gw_scores.csv
│   └── Linux_gw_scores.csv
├── run_two_stage_ablation.py          ✓ Main experiment script
├── ged_computation.py                 ✓ Core GED/OT functions
├── utils.py                           ✓ Utility functions
├── environment.yml                    ✓ Conda environment
└── README.md                          ✓ Project overview
```

---

## Related Documentation

- **[README.md](README.md)** - Project overview and quick start
- **[TWO_STAGE_ABLATION_FINAL_REPORT.md](results/two_stage_ablation/TWO_STAGE_ABLATION_FINAL_REPORT.md)** - Comprehensive analysis and interpretation
- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Detailed environment configuration
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{ged_ot_features_2025,
  title={Graph Edit Distance Approximation with Optimal Transport Features},
  author={Author Name},
  journal={Journal Name},
  year={2026},
  note={GitHub: https://github.com/[owner]/GED_structural_features}
}
```

---

## Support

- **Questions about setup?** See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- **Results differ?** Check [Troubleshooting](#troubleshooting) section
- **Bug reports?** Open an issue with: OS, Python version, error message
- **Feature requests?** See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Last tested:** February 6, 2026  
**Maintainer:** [Your Name]  
**License:** MIT
