"""
Visualization utilities for generating publication-quality plots.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json


# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def load_results(results_dir):
    """Load experiment results from directory."""
    results_dir = Path(results_dir)
    
    with open(results_dir / 'summary.json', 'r') as f:
        summary = json.load(f)
    
    all_results = {
        'random_search': [],
        'pso': []
    }
    
    # Load detailed results
    for opt_name in ['random_search', 'pso']:
        opt_dir = results_dir / opt_name
        if opt_dir.exists():
            for run_dir in sorted(opt_dir.iterdir()):
                if run_dir.is_dir():
                    with open(run_dir / 'optimization_history.json', 'r') as f:
                        opt_history = json.load(f)
                    with open(run_dir / 'training_history.json', 'r') as f:
                        train_history = json.load(f)
                    
                    all_results[opt_name].append({
                        'optimization_history': opt_history,
                        'training_history': train_history
                    })
    
    return summary, all_results


def plot_comparison_boxplot(summary, save_path):
    """Create comparison boxplot of test accuracies."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rs_accs = summary['random_search']['test_accuracies']
    pso_accs = summary['pso']['test_accuracies']
    
    data = {
        'Random Search': rs_accs,
        'PSO': pso_accs
    }
    
    positions = [1, 2]
    labels = ['Random Search', 'PSO']
    
    bp = ax.boxplot([rs_accs, pso_accs], positions=positions, 
                    labels=labels, patch_artist=True, widths=0.6)
    
    # Color boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Hyperparameter Optimization Comparison\nCIFAR-10 CNN', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    for i, (label, accs) in enumerate(data.items()):
        mean_acc = np.mean(accs)
        ax.plot(positions[i], mean_acc, 'D', color='black', markersize=8, label='Mean' if i == 0 else '')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved comparison boxplot to {save_path}")


def plot_convergence_curves(all_results, save_path):
    """Plot convergence curves for optimization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for opt_name, color, label in [('random_search', '#3498db', 'Random Search'),
                                    ('pso', '#e74c3c', 'PSO')]:
        if opt_name not in all_results or len(all_results[opt_name]) == 0:
            continue
        
        all_scores = []
        max_trials = 0
        
        for run in all_results[opt_name]:
            history = run['optimization_history']
            scores = [h['score'] for h in history]
            all_scores.append(scores)
            max_trials = max(max_trials, len(scores))
        
        # Pad to same length
        padded_scores = []
        for scores in all_scores:
            padded = scores + [scores[-1]] * (max_trials - len(scores))
            padded_scores.append(padded)
        
        # Calculate mean and std
        padded_scores = np.array(padded_scores)
        mean_scores = np.mean(padded_scores, axis=0)
        std_scores = np.std(padded_scores, axis=0)
        
        x = np.arange(1, max_trials + 1)
        ax.plot(x, mean_scores, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, 
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Trial Number', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Optimization Convergence Curves', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved convergence curves to {save_path}")


def plot_training_curves(all_results, save_path):
    """Plot training curves for best models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for opt_name, color, label in [('random_search', '#3498db', 'Random Search'),
                                    ('pso', '#e74c3c', 'PSO')]:
        if opt_name not in all_results or len(all_results[opt_name]) == 0:
            continue
        
        all_train_accs = []
        all_val_accs = []
        max_epochs = 0
        
        for run in all_results[opt_name]:
            history = run['training_history']
            train_accs = history['train_acc']
            val_accs = history['val_acc']
            
            all_train_accs.append(train_accs)
            all_val_accs.append(val_accs)
            max_epochs = max(max_epochs, len(train_accs))
        
        # Pad to same length
        def pad_list(lst, target_len):
            if len(lst) < target_len:
                return lst + [lst[-1]] * (target_len - len(lst))
            return lst[:target_len]
        
        padded_train = [pad_list(accs, max_epochs) for accs in all_train_accs]
        padded_val = [pad_list(accs, max_epochs) for accs in all_val_accs]
        
        padded_train = np.array(padded_train)
        padded_val = np.array(padded_val)
        
        mean_train = np.mean(padded_train, axis=0)
        mean_val = np.mean(padded_val, axis=0)
        std_train = np.std(padded_train, axis=0)
        std_val = np.std(padded_val, axis=0)
        
        x = np.arange(1, len(mean_train) + 1)
        
        # Plot accuracy
        ax1.plot(x, mean_train, label=f'{label} (Train)', color=color, linestyle='--', linewidth=2)
        ax1.plot(x, mean_val, label=f'{label} (Val)', color=color, linewidth=2)
        ax1.fill_between(x, mean_train - std_train, mean_train + std_train, 
                        alpha=0.1, color=color)
        ax1.fill_between(x, mean_val - std_val, mean_val + std_val, 
                        alpha=0.1, color=color)
    
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss (if available)
    for opt_name, color, label in [('random_search', '#3498db', 'Random Search'),
                                    ('pso', '#e74c3c', 'PSO')]:
        if opt_name not in all_results or len(all_results[opt_name]) == 0:
            continue
        
        all_train_losses = []
        all_val_losses = []
        max_epochs = 0
        
        for run in all_results[opt_name]:
            history = run['training_history']
            if 'train_loss' in history and 'val_loss' in history:
                train_losses = history['train_loss']
                val_losses = history['val_loss']
                
                all_train_losses.append(train_losses)
                all_val_losses.append(val_losses)
                max_epochs = max(max_epochs, len(train_losses))
        
        if len(all_train_losses) > 0:
            def pad_list(lst, target_len):
                if len(lst) < target_len:
                    return lst + [lst[-1]] * (target_len - len(lst))
                return lst[:target_len]
            
            padded_train = [pad_list(losses, max_epochs) for losses in all_train_losses]
            padded_val = [pad_list(losses, max_epochs) for losses in all_val_losses]
            
            padded_train = np.array(padded_train)
            padded_val = np.array(padded_val)
            
            mean_train = np.mean(padded_train, axis=0)
            mean_val = np.mean(padded_val, axis=0)
            
            x = np.arange(1, len(mean_train) + 1)
            
            ax2.plot(x, mean_train, label=f'{label} (Train)', color=color, linestyle='--', linewidth=2)
            ax2.plot(x, mean_val, label=f'{label} (Val)', color=color, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_hyperparameter_importance(all_results, save_path):
    """Plot hyperparameter importance based on correlation with performance."""
    # Collect all configurations and scores
    all_configs = []
    all_scores = []
    
    for opt_name in ['random_search', 'pso']:
        if opt_name not in all_results:
            continue
        
        for run in all_results[opt_name]:
            history = run['optimization_history']
            for h in history:
                all_configs.append(h['config'])
                all_scores.append(h['score'])
    
    if len(all_configs) == 0:
        print("No data for hyperparameter importance plot")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_configs)
    df['score'] = all_scores
    
    # Calculate correlations
    param_names = [col for col in df.columns if col != 'score']
    correlations = []
    
    for param in param_names:
        corr = df[param].corr(df['score'])
        correlations.append(abs(corr))  # Use absolute correlation
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_indices = np.argsort(correlations)[::-1]
    sorted_params = [param_names[i] for i in sorted_indices]
    sorted_corrs = [correlations[i] for i in sorted_indices]
    
    bars = ax.barh(range(len(sorted_params)), sorted_corrs, color='steelblue')
    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(sorted_params)
    ax.set_xlabel('Absolute Correlation with Performance', fontsize=14)
    ax.set_title('Hyperparameter Importance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, corr) in enumerate(zip(bars, sorted_corrs)):
        ax.text(corr + 0.01, i, f'{corr:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved hyperparameter importance plot to {save_path}")


def generate_all_plots(results_dir, output_dir=None):
    """Generate all publication plots."""
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / 'publication_plots'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    summary, all_results = load_results(results_dir)
    
    # Generate plots
    plot_comparison_boxplot(summary, output_dir / 'fig1_comparison_boxplot.png')
    plot_convergence_curves(all_results, output_dir / 'fig2_convergence_curves.png')
    plot_training_curves(all_results, output_dir / 'fig3_training_curves.png')
    plot_hyperparameter_importance(all_results, output_dir / 'fig4_hyperparameter_importance.png')
    
    print(f"\nAll plots generated in {output_dir}")

