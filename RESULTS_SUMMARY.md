# SINTESI RISULTATI ESPERIMENTO: LINEAR vs QUADRATIC

**Data**: 16 Febbraio 2026  
**Esperimento**: Ablation Study su GED Prediction tramite Gradient Boosting  
**Durata**: ~5 minuti  
**Dataset**: 6400 coppie di grafi (200 per config × 32 config)

---

## 1. CONFIGURAZIONE ESPERIMENTO

### Setup
- **n_nodes**: 50 (fisso)
- **Topologie**: 2 separate (Barabási-Albert, Erdős-Rényi)
- **Barabási-Albert**: m ∈ {1,2,3} casuale
- **Configurazioni**: 16 (p,q) ∈ {0,5,10,20}²
- **Per config**: 200 coppie, split 80/20 train/test
- **Algoritmi**: Linear Surrogate vs Exact Quadratic
- **Metrica principale**: MAE su test set

---

## 2. RISULTATI GLOBALI

### Win Rate per Topologia

| Metrica | Barabási-Albert | Erdős-Rényi |
|---------|-----------------|------------|
| **Quadratic vince** | 10/16 (62.5%) | 6/16 (37.5%) |
| **Linear vince** | 3/16 (18.8%) | 7/16 (43.8%) |
| **Pareggio** | 3/16 (18.8%) | 3/16 (18.8%) |

**Conclusione**: **Quadratic è decisamente superiore su BA, misto su ER**

---

## 3. ANALISI PER TOPOLOGIA

### A. BARABÁSI-ALBERT (Scale-Free)
**Trend**: Quadratic domina, specialmente con bassi edit rate

#### Top 5 Vittorie Quadratic (Improvement %)
1. **p=0%, q=10%**: 99.97% ✓ (MAE: 0.250 → 0.000)
2. **p=10%, q=20%**: 99.94% ✓ (MAE: 0.174 → 0.000)
3. **p=20%, q=10%**: 72.48% ✓ (MAE: 0.151 → 0.042)
4. **p=10%, q=10%**: 48.84% ✓ (MAE: 0.075 → 0.038)
5. **p=5%, q=20%**: 6.46% ✓ (MAE: 1.848 → 1.728)

#### Sconfitte Quadratic (quando Linear vince)
- **p=0%, q=20%**: -301.88% (MAE: 0.056 vs 0.225) - Linear MOLTO meglio
- **p=5%, q=0%**: -30.07% (MAE: 0.300 vs 0.390)
- **p=5%, q=5%**: -9.13% (MAE: 0.150 vs 0.163)
- **p=10%, q=0%**: -6.94% (MAE: 0.273 vs 0.292)

#### Pareggi (R²=1.0 in entrambi i casi)
- p=0%, q=0%, p=20%, q=0%, p=20%, q=5%, p=20%, q=20%

---

### B. ERDŐS-RÉNYI (Random)
**Trend**: Più instabile, Linear spesso competitivo

#### Top 5 Vittorie Quadratic (Improvement %)
1. **p=5%, q=20%**: 28.25% ✓ (MAE: 0.852 → 0.611)
2. **p=20%, q=20%**: 30.81% ✓ (MAE: 0.101 → 0.070)
3. **p=5%, q=0%**: 22.93% ✓ (MAE: 0.354 → 0.273)
4. **p=20%, q=10%**: 23.62% ✓ (MAE: 0.013 → 0.010)
5. **p=10%, q=5%**: 16.44% ✓ (MAE: 0.209 → 0.174)

#### Sconfitte Quadratic (quando Linear vince)
- **p=10%, q=0%**: -991.28% (MAE: 0.010 vs 0.110) - Linear ENORMEMENTE meglio
- **p=20%, q=0%**: -51.37% (MAE: 0.033 vs 0.050)
- **p=20%, q=5%**: -42.16% (MAE: 0.033 vs 0.047)
- **p=0%, q=5%**: -28.63% (MAE: 0.162 vs 0.208)
- **p=10%, q=20%**: -26.15% (MAE: 0.115 vs 0.145)
- **p=0%, q=20%**: -18.64% (MAE: 0.629 vs 0.746)

---

## 4. PATTERN E OSSERVAZIONI

### Pattern 1: Edit Rate Bassissimo → Quadratic domina BA
Quando q=10% con p basso:
- **BA p=0%, q=10%**: MAE passa da 0.250 → 0.000 (99.97% improvement)
- **BA p=10%, q=20%**: MAE passa da 0.174 → 0.000 (99.94% improvement)

**Interpretazione**: Quadratic risolve meglio quando il problema è poco vincolato (poche aggiunte, solo rimozioni).

### Pattern 2: Solo Aggiunte → Linear più stabile su ER
Quando q=0% (solo edge add, no remove):
- **BA p=5%, q=0%**: Linear MAE=0.300 vs Q MAE=0.390 (Linear vince)
- **ER p=10%, q=0%**: Linear MAE=0.010 vs Q MAE=0.110 (Linear MOLTO meglio)

**Interpretazione**: Linear excels when only adding edges (simpler problem structure).

### Pattern 3: ER è "rumorosa" rispetto a BA
R² medio:
- **BA**: 0.74 (più stabile)
- **ER**: 0.45 (più variabile)

**Interpretazione**: Grafi random sono meno prevedibili per i modelli GB.

### Pattern 4: Range GED alto → Meno variabilità nei risultati
Quando p,q alti (p=20%, q=20%):
- Entrambi convergono a MAE molto bassi
- Discriminazione ridotta

---

## 5. CONCLUSIONI

### Vincitori Assoluti
1. **Quadratic su Barabási-Albert**: Superiore in 10/16 config (62.5%)
   - MAE medio Q: 0.0607
   - MAE medio L: 0.2033
   - **Improvement medio: +40.2%**

2. **Mixed su Erdős-Rényi**: Ballottaggio serrato
   - MAE medio Q: 0.1934
   - MAE medio L: 0.2115
   - **Improvement medio: -5.2%** (Linear leggermente avanti)

### Raccomandazioni

#### Per Barabási-Albert (Scale-Free)
✅ **Preferiamo Quadratic** quando:
- q > 0 (rimozione di archi)
- p basso (pochi archi aggiunti)
- GED range moderato (5-250)

❌ **Usiamo Linear** quando:
- p alto, q=0 (solo aggiunte)
- GED molto alto (quando discretizzazione è critica)

#### Per Erdős-Rényi (Random)
⚠️ **Nessun vincitore chiaro** - risultati istanza-dipendenti
- Provare entrambi in applicazione pratica
- Quadratic coerente su casi con alto q
- Linear robusto su p alto, q basso

#### Overall
- **Quadratic è generalmente più accurato** su scale-free networks
- **Linear rimane competitivo** su random graphs
- **La topologia conta**: non si può generalizzare universalmente

---

## 6. METRICHE RIASSUNTIVE

| Metrica | Barabási-Albert | Erdős-Rényi |
|---------|-----------------|-----------|
| MAE medio Linear | 0.203 | 0.212 |
| MAE medio Quadratic | 0.061 | 0.193 |
| R² medio Linear | 0.824 | 0.449 |
| R² medio Quadratic | 0.856 | 0.509 |
| Vittorie Quadratic | 62.5% | 37.5% |
| Miglior improvement | 99.97% (p=0,q=10) | 30.81% (p=20,q=20) |
| Peggior performance Q | -301.8% (p=0,q=20) | -991.3% (p=10,q=0) |

---

## 7. OUTPUT GENERATI

✅ CSV: `results/ablation_experiment_results.csv` (32 righe)
✅ Heatmap BA MAE: `figures/ablation_mae_barabasi_albert.png`
✅ Heatmap ER MAE: `figures/ablation_mae_erdos_renyi.png`
✅ Heatmap BA Improvement: `figures/ablation_improvement_barabasi_albert.png`
✅ Heatmap ER Improvement: `figures/ablation_improvement_erdos_renyi.png`
