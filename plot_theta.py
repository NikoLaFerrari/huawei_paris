import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_regression_analysis(results, input_dims):
    # 1. Extract Global Data
    t_vals = np.array([r['T_i'] for r in results])
    p_vals = np.array([r['P_i'] for r in results])
    
    # Extract dimensions for coloring
    dims_to_plot = ['dp', 'mp', 'pp', 'ep']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Calculate Global Stats for the header
    slope, intercept, r_value, _, _ = linregress(t_vals, p_vals)
    global_r2 = r_value**2
    line = slope * t_vals + intercept

    for i, dim_name in enumerate(dims_to_plot):
        ax = axes[i]
        dim_values = [r['dims'].get(dim_name, 1) for r in results]
        unique_vals = sorted(list(set(dim_values)))
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(unique_vals)))

        # Plot grouped scatter
        for j, val in enumerate(unique_vals):
            mask = np.array([v == val for v in dim_values])
            ax.scatter(t_vals[mask], p_vals[mask], color=colors[j],
                       label=f'{dim_name.upper()}={val}', alpha=0.7, edgecolors='w', s=50)

        # Plot the global fit line on every subplot for reference
        ax.plot(t_vals, line, color='red', linestyle='--', alpha=0.4)
        
        ax.set_title(f'Scaling Sensitivity: {dim_name.upper()}', fontweight='bold')
        ax.set_xlabel('Predictor Score (T_i)')
        ax.set_ylabel('ND Score (P_i)')
        ax.legend(title=dim_name.upper(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f'Grey-Box Validation Diagnostic (Global R²={global_r2:.4f})\n'
                 f'Vertical Stacks indicate Dimension Inefficiency\n'
                 f'Input: {input_dims}\n'
                 , fontsize=16, fontweight='bold')
    
    print(f"[regression] Global R-squared: {global_r2:.4f}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('scaling_diagnostic.png')
    plt.show()
