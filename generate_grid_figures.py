"""
Generate compact 4x4 table figures for BA and ER ablation results.
Each cell (p, q) shows MAE Linear / MAE Quadratic with color-coded background.
Rows = p values (0%, 5%, 10%, 20%), Columns = q values (0%, 5%, 10%, 20%).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Load data
df = pd.read_csv("results/ablation_experiment_results.csv")

p_values = [0, 5, 10, 20]
q_values = [0, 5, 10, 20]


def generate_table_figure(df_topo, topology_label, filename):
    """Generate a compact 4x4 color-coded table."""

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5.6)
    ax.axis('off')

    # Title
    ax.text(2.5, 5.35, f'{topology_label}',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(2.5, 5.1, '200 pairs per config · 50 nodes · GradientBoosting · 80/20 split',
            ha='center', va='center', fontsize=9, color='gray')

    # Column / row dimensions
    col_w = 1.0    # cell width
    row_h = 0.95   # cell height
    x0 = 0.6       # left offset for data cells
    y0 = 0.35      # bottom offset

    # --- Header row (q values) ---
    ax.text(x0 - 0.15, y0 + 4 * row_h + row_h * 0.65, 'p \\ q',
            ha='center', va='center', fontsize=11, fontweight='bold', style='italic')
    for j, q in enumerate(q_values):
        cx = x0 + j * col_w + col_w / 2
        cy = y0 + 4 * row_h + row_h * 0.65
        ax.text(cx, cy, f'q = {q}%', ha='center', va='center',
                fontsize=11, fontweight='bold')

    # --- Row labels (p values) ---
    for i, p in enumerate(p_values):
        row_idx = 3 - i  # top row = p=0
        cy = y0 + row_idx * row_h + row_h / 2
        ax.text(x0 - 0.15, cy, f'p = {p}%', ha='center', va='center',
                fontsize=11, fontweight='bold')

    # Color maps: blue shades for Linear wins, orange shades for Quadratic wins
    # We'll use background color intensity based on |improvement|
    blue_cm = plt.cm.Blues
    orange_cm = plt.cm.Oranges

    for i, p in enumerate(p_values):
        row_idx = 3 - i  # invert so p=0 is top row
        for j, q in enumerate(q_values):
            row = df_topo[(df_topo['p'] == p) & (df_topo['q'] == q)]
            if len(row) == 0:
                continue

            mae_L = row['mae_linear'].values[0]
            mae_Q = row['mae_quadratic'].values[0]

            cx = x0 + j * col_w + col_w / 2
            cy = y0 + row_idx * row_h + row_h / 2

            # Determine winner and background color
            if mae_L < 1e-6 and mae_Q < 1e-6:
                bg_color = '#f0f0f0'  # gray for tie at ~0
                winner = 'tie'
            elif abs(mae_L - mae_Q) < 1e-6:
                bg_color = '#f0f0f0'
                winner = 'tie'
            elif mae_Q < mae_L:
                # Quadratic wins — orange tint
                imp = (mae_L - mae_Q) / mae_L
                intensity = min(imp * 1.5, 0.8)  # cap
                bg_color = orange_cm(0.15 + intensity * 0.55)
                winner = 'Q'
            else:
                # Linear wins — blue tint
                imp = (mae_Q - mae_L) / mae_Q
                intensity = min(imp * 1.5, 0.8)
                bg_color = blue_cm(0.15 + intensity * 0.55)
                winner = 'L'

            # Draw cell background
            rect = plt.Rectangle((x0 + j * col_w, y0 + row_idx * row_h),
                                  col_w, row_h,
                                  facecolor=bg_color, edgecolor='white', linewidth=2)
            ax.add_patch(rect)

            # Format MAE values
            def fmt(v):
                if v < 0.0005:
                    return '≈0'
                elif v < 0.01:
                    return f'{v:.4f}'
                else:
                    return f'{v:.3f}'

            mae_L_str = fmt(mae_L)
            mae_Q_str = fmt(mae_Q)

            # Bold the winner value
            if winner == 'L':
                ax.text(cx, cy + 0.15, f'L: {mae_L_str}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#1a1a1a')
                ax.text(cx, cy - 0.15, f'Q: {mae_Q_str}', ha='center', va='center',
                        fontsize=10, fontweight='normal', color='#555555')
            elif winner == 'Q':
                ax.text(cx, cy + 0.15, f'L: {mae_L_str}', ha='center', va='center',
                        fontsize=10, fontweight='normal', color='#555555')
                ax.text(cx, cy - 0.15, f'Q: {mae_Q_str}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#1a1a1a')
            else:
                ax.text(cx, cy + 0.15, f'L: {mae_L_str}', ha='center', va='center',
                        fontsize=10, color='#555555')
                ax.text(cx, cy - 0.15, f'Q: {mae_Q_str}', ha='center', va='center',
                        fontsize=10, color='#555555')

            # Small improvement percentage
            if winner != 'tie':
                imp_pct = ((mae_L - mae_Q) / max(mae_L, 1e-10)) * 100
                if abs(imp_pct) > 1000:
                    imp_label = f'Δ>999%'
                else:
                    imp_label = f'Δ={abs(imp_pct):.0f}%'
                ax.text(cx, cy - 0.38, imp_label,
                        ha='center', va='center', fontsize=7, color='#888888')

    # Legend
    legend_elements = [
        Patch(facecolor=blue_cm(0.35), edgecolor='gray', label='Linear wins'),
        Patch(facecolor=orange_cm(0.35), edgecolor='gray', label='Quadratic wins'),
        Patch(facecolor='#f0f0f0', edgecolor='gray', label='Tie (MAE ≈ 0)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10, frameon=True)

    # Axis labels
    ax.text(2.6, y0 + 4 * row_h + row_h * 1.15,
            'Edge removal rate (q)', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#444444')
    ax.text(x0 - 0.55, y0 + 2 * row_h, 'Edge addition\nrate (p)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#444444', rotation=90)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


# Generate both figures
for topo_key, topo_label in [('Barabási-Albert', 'Barabási-Albert (Scale-Free)'),
                               ('Erdős-Rényi', 'Erdős-Rényi (Random)')]:
    df_topo = df[df['topology'] == topo_key]
    fname = f"figures/grid_mae_{topo_key.lower().replace('á', 'a').replace('é', 'e').replace('ő', 'o').replace('-', '_')}.png"
    generate_table_figure(df_topo, topo_label, fname)

print("\nDone! Generated 2 table figures.")
