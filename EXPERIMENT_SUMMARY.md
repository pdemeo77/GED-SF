# Ablation Study: Linear Surrogate vs. Exact Quadratic Formulation in GED-SF

## Abstract
This report presents an ablation study evaluating two optimization formulations for approximating Graph Edit Distance (GED) using structural features (GED-SF). We compare a **Linear Surrogate** approach, which utilizes a static structural gradient, against an **Exact Quadratic** formulation. Our experiments, conducted across different graph topologies (Barabási-Albert and Erdős-Rényi) and varying scales (100–500 nodes), demonstrate that the Linear Surrogate significantly reduces computational runtime while maintaining a high degree of alignment fidelity.

## 1. Introduction
Graph Edit Distance is a fundamental metric for graph similarity, but its exact computation is NP-hard. The GED-SF algorithm leverages Fused Gromov-Wasserstein (FGW) and structural node features (Degree, PageRank, Clustering Coefficient) to provide efficient approximations. This study investigates whether approximating the structural term with a linear surrogate—thereby simplifying the Frank-Wolfe optimization steps—offers a viable trade-off between speed and accuracy.

## 2. Experimental Setup
The study was structured into three main phases:
- **Scalability Analysis (Runtime)**: Evaluated average runtime for graph sizes $n \in \{100, 200, 300, 400, 500\}$ with a fixed edit rate of 5%.
- **Robustness Analysis (Divergence)**: Evaluated the solution divergence (Normalized Frobenius Distance) at a fixed size of $n=100$ while varying the edge edit rate from **1% to 12%**.
- **Sensitivity Analysis**: Evaluated the impact of the structural regularization parameter $\mu \in [0.1, 5.0]$ on a fixed graph size ($n=100$).

Two topologies were considered:
1. **Barabási-Albert (BA)**: Representing scale-free networks with prominent hubs.
2. **Erdős-Rényi (ER)**: Representing random networks with uniform edge distribution.

## 3. Results and Discussion

### 3.1 Computational Efficiency (Size Scaling)
The **Linear Surrogate** consistently outperformed the Exact Quadratic formulation in terms of execution time across all sizes. As the graph size increases, the computational overhead of the Quadratic approach becomes prohibitive, whereas the Linear Surrogate maintains a more tractable scaling.

### 3.2 Approximation Fidelity (Edge Edit Robustness)
The solution divergence was analyzed as a function of the edge edit rate at $n=100$. 
- For both topologies, the divergence remains remarkably stable even as the number of structural edits increases up to 12%.
- The **Barabási-Albert** graphs show near-perfect alignment (minimal divergence) due to the strongly distinct structural signatures of hubs.
- The **Erdős-Rényi** graphs exhibit slightly higher but still negligible divergence, confirming that the Linear Surrogate is a robust approximation regardless of the noise level introduced by edits.

### 3.3 Mu Sensitivity
Varying the parameter $\mu$ showed that while a higher weight on structural features increases the influence of the surrogate term, the alignment remain accurate. The stability of the Linear Surrogate across a wide range of $\mu$ suggests it is robust for various levels of structural regularization.

## 4. Conclusion
The experimental results confirm that the **Linear Surrogate** formulation is a highly effective optimization strategy for GED-SF. It provides a substantial computational advantage, especially as graph size scales, without compromising the quality of the graph alignment. This formulation is recommended for large-scale graph comparison tasks.

---
*Figures generated during this study can be found in the `figures/` directory.*
