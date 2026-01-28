"""
Generate publication-quality plots for CIFAR-10 HPO experiment results.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif'
})

def load_results(results_dir):
    """Load all experiment results."""
    results_dir = Path(results_dir)
    
    # Load Random Search history
    rs_history_path = results_dir / 'random_search' / 'run_42' / 'optimization_history.json'
    with open(rs_history_path, 'r') as f:
        rs_history = json.load(f)
    
    # Load PSO history
    pso_history_path = results_dir / 'pso' / 'run_42' / 'optimization_history.json'
    with open(pso_history_path, 'r') as f:
        pso_history = json.load(f)
    
    # Load summary
    summary_path = results_dir / 'summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return rs_history, pso_history, summary


def plot_comparison_boxplot(summary, output_dir):
    """Figure 1: Comparison boxplot of final test accuracies."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rs_acc = summary['random_search']['test_accuracies']
    pso_acc = summary['pso']['test_accuracies']
    
    data = [rs_acc, pso_acc]
    labels = ['Random Search', 'PSO']
    colors = ['#3498db', '#e74c3c']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.8, color='black', s=50, zorder=5)
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Comparison of Optimization Methods')
    ax.set_ylim([min(min(rs_acc), min(pso_acc)) - 1, max(max(rs_acc), max(pso_acc)) + 1])
    
    # Add mean values as text
    for i, (d, label) in enumerate(zip(data, labels)):
        mean_val = np.mean(d)
        ax.annotate(f'μ = {mean_val:.2f}%', 
                   xy=(i + 1, mean_val), 
                   xytext=(i + 1.3, mean_val),
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_comparison_boxplot.png')
    plt.savefig(output_dir / 'fig1_comparison_boxplot.pdf')
    plt.close()
    print("✅ Saved: fig1_comparison_boxplot.png/pdf")


def plot_convergence_curves(rs_history, pso_history, output_dir):
    """Figure 2: Convergence curves for both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random Search - Trial scores
    ax1 = axes[0]
    trials = [h['metadata']['trial'] for h in rs_history]
    scores = [h['score'] for h in rs_history]
    best_so_far = np.maximum.accumulate(scores)
    
    ax1.bar(trials, scores, alpha=0.6, color='#3498db', label='Trial Score')
    ax1.plot(trials, best_so_far, 'r-', linewidth=2.5, marker='o', 
             markersize=6, label='Best So Far')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Random Search Convergence')
    ax1.legend(loc='lower right')
    ax1.set_ylim([65, 95])
    ax1.grid(True, alpha=0.3)
    
    # PSO - Iteration scores
    ax2 = axes[1]
    
    # Group PSO by iteration
    iterations = []
    iteration_scores = []
    global_bests = []
    
    current_iter = 1
    iter_scores = []
    global_best = 0
    
    for h in pso_history:
        iter_num = h['metadata']['iteration']
        if iter_num != current_iter:
            iterations.append(current_iter)
            iteration_scores.append(np.mean(iter_scores))
            global_bests.append(global_best)
            current_iter = iter_num
            iter_scores = []
        
        iter_scores.append(h['score'])
        global_best = max(global_best, h['score'])
    
    # Add last iteration
    iterations.append(current_iter)
    iteration_scores.append(np.mean(iter_scores))
    global_bests.append(global_best)
    
    ax2.bar(iterations, iteration_scores, alpha=0.6, color='#e74c3c', 
            label='Mean Particle Score')
    ax2.plot(iterations, global_bests, 'b-', linewidth=2.5, marker='s', 
             markersize=8, label='Global Best')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('PSO Convergence')
    ax2.legend(loc='lower right')
    ax2.set_ylim([65, 95])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_convergence_curves.png')
    plt.savefig(output_dir / 'fig2_convergence_curves.pdf')
    plt.close()
    print("✅ Saved: fig2_convergence_curves.png/pdf")


def plot_hyperparameter_distributions(rs_history, pso_history, output_dir):
    """Figure 3: Hyperparameter distributions with performance."""
    
    # Combine all data
    all_configs = []
    all_scores = []
    all_methods = []
    
    for h in rs_history:
        all_configs.append(h['config'])
        all_scores.append(h['score'])
        all_methods.append('Random Search')
    
    for h in pso_history:
        all_configs.append(h['config'])
        all_scores.append(h['score'])
        all_methods.append('PSO')
    
    df = pd.DataFrame(all_configs)
    df['score'] = all_scores
    df['method'] = all_methods
    
    # Create subplots for each hyperparameter
    hyperparams = ['learning_rate', 'batch_size', 'conv_channels_base', 
                   'num_conv_blocks', 'fc_hidden', 'dropout', 'weight_decay']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = {'Random Search': '#3498db', 'PSO': '#e74c3c'}
    
    for idx, hp in enumerate(hyperparams):
        ax = axes[idx]
        
        for method in ['Random Search', 'PSO']:
            mask = df['method'] == method
            ax.scatter(df.loc[mask, hp], df.loc[mask, 'score'], 
                      c=colors[method], alpha=0.6, s=60, label=method)
        
        ax.set_xlabel(hp.replace('_', ' ').title())
        ax.set_ylabel('Accuracy (%)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for some parameters
        if hp in ['learning_rate', 'weight_decay']:
            ax.set_xscale('log')
    
    # Remove extra subplot
    axes[-1].axis('off')
    
    plt.suptitle('Hyperparameter vs Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_hyperparameter_distributions.png')
    plt.savefig(output_dir / 'fig3_hyperparameter_distributions.pdf')
    plt.close()
    print("✅ Saved: fig3_hyperparameter_distributions.png/pdf")


def plot_hyperparameter_importance(rs_history, pso_history, output_dir):
    """Figure 4: Hyperparameter importance (correlation with performance)."""
    
    # Combine all data
    all_configs = []
    all_scores = []
    
    for h in rs_history + pso_history:
        all_configs.append(h['config'])
        all_scores.append(h['score'])
    
    df = pd.DataFrame(all_configs)
    df['score'] = all_scores
    
    # Calculate correlations
    correlations = {}
    for col in df.columns:
        if col != 'score':
            correlations[col] = df[col].corr(df['score'])
    
    # Sort by absolute correlation
    sorted_correlations = dict(sorted(correlations.items(), 
                                       key=lambda x: abs(x[1]), reverse=True))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [k.replace('_', '\n') for k in sorted_correlations.keys()]
    values = list(sorted_correlations.values())
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Correlation with Accuracy')
    ax.set_title('Hyperparameter Importance Analysis')
    ax.set_xlim([-1, 1])
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.05 if val >= 0 else val - 0.05
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
               f'{val:.3f}', va='center', ha=ha, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_hyperparameter_importance.png')
    plt.savefig(output_dir / 'fig4_hyperparameter_importance.pdf')
    plt.close()
    print("✅ Saved: fig4_hyperparameter_importance.png/pdf")


def plot_correlation_heatmap(rs_history, pso_history, output_dir):
    """Figure 5: Correlation heatmap of hyperparameters."""
    
    all_configs = []
    all_scores = []
    
    for h in rs_history + pso_history:
        all_configs.append(h['config'])
        all_scores.append(h['score'])
    
    df = pd.DataFrame(all_configs)
    df['accuracy'] = all_scores
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    
    ax.set_title('Hyperparameter Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_correlation_heatmap.png')
    plt.savefig(output_dir / 'fig5_correlation_heatmap.pdf')
    plt.close()
    print("✅ Saved: fig5_correlation_heatmap.png/pdf")


def plot_best_configs_comparison(rs_history, pso_history, output_dir):
    """Figure 6: Radar chart comparing best configurations."""
    
    # Find best configs
    rs_best = max(rs_history, key=lambda x: x['score'])
    pso_best = max(pso_history, key=lambda x: x['score'])
    
    # Normalize hyperparameters to [0, 1] range
    param_ranges = {
        'learning_rate': (0.0001, 0.01),
        'batch_size': (32, 256),
        'conv_channels_base': (32, 128),
        'num_conv_blocks': (2, 4),
        'fc_hidden': (64, 512),
        'dropout': (0.1, 0.5),
        'weight_decay': (0.00001, 0.001)
    }
    
    def normalize(config):
        normalized = {}
        for key, (min_val, max_val) in param_ranges.items():
            val = config[key]
            # Use log scale for log-scale parameters
            if key in ['learning_rate', 'weight_decay']:
                val = np.log10(val)
                min_val = np.log10(min_val)
                max_val = np.log10(max_val)
            normalized[key] = (val - min_val) / (max_val - min_val)
        return normalized
    
    rs_norm = normalize(rs_best['config'])
    pso_norm = normalize(pso_best['config'])
    
    # Radar chart
    categories = list(param_ranges.keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    rs_values = [rs_norm[cat] for cat in categories]
    rs_values += rs_values[:1]
    
    pso_values = [pso_norm[cat] for cat in categories]
    pso_values += pso_values[:1]
    
    ax.plot(angles, rs_values, 'o-', linewidth=2, label=f'Random Search ({rs_best["score"]:.2f}%)', 
            color='#3498db')
    ax.fill(angles, rs_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, pso_values, 's-', linewidth=2, label=f'PSO ({pso_best["score"]:.2f}%)', 
            color='#e74c3c')
    ax.fill(angles, pso_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], size=10)
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Best Configuration Comparison\n(Normalized Values)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_radar_comparison.png')
    plt.savefig(output_dir / 'fig6_radar_comparison.pdf')
    plt.close()
    print("✅ Saved: fig6_radar_comparison.png/pdf")


def main():
    # Results directory
    results_dir = Path('experiments/experiments/results/CIFAR10_CNN_20260127_170821')
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please update the path to your results directory.")
        return
    
    # Create output directory
    output_dir = results_dir / 'publication_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING PUBLICATION PLOTS")
    print("="*60)
    
    # Load results
    rs_history, pso_history, summary = load_results(results_dir)
    print(f"✅ Loaded {len(rs_history)} Random Search trials")
    print(f"✅ Loaded {len(pso_history)} PSO evaluations")
    
    # Generate all plots
    plot_comparison_boxplot(summary, output_dir)
    plot_convergence_curves(rs_history, pso_history, output_dir)
    plot_hyperparameter_distributions(rs_history, pso_history, output_dir)
    plot_hyperparameter_importance(rs_history, pso_history, output_dir)
    plot_correlation_heatmap(rs_history, pso_history, output_dir)
    plot_best_configs_comparison(rs_history, pso_history, output_dir)
    
    print("="*60)
    print(f"✅ All plots saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()