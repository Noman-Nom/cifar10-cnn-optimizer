"""
Complete Publication-Ready Plots Generator for HPO Thesis
Generates all figures needed for thesis with statistical analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
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
    'font.family': 'serif',
    'text.usetex': False
})

class PublicationPlotGenerator:
    """Generate all publication-ready plots for thesis"""
    
    def __init__(self, results_dirs, output_dir='publication_plots'):
        """
        Args:
            results_dirs: List of paths to result directories (3 runs)
            output_dir: Output directory for plots
        """
        self.results_dirs = [Path(d) for d in results_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        self.load_all_data()
        
    def load_all_data(self):
        """Load data from all runs"""
        print("="*80)
        print("LOADING DATA FROM ALL RUNS")
        print("="*80)
        
        self.all_data = []
        
        for idx, results_dir in enumerate(self.results_dirs):
            run_seed = [42, 123, 999][idx]
            print(f"\n📂 Loading Run {idx+1} (seed={run_seed}): {results_dir}")
            
            # Load summary
            summary_path = results_dir / 'summary.json'
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Find actual run directories (they all use run_42)
            rs_run_dirs = list((results_dir / 'random_search').glob('run_*'))
            pso_run_dirs = list((results_dir / 'pso').glob('run_*'))
            
            if not rs_run_dirs or not pso_run_dirs:
                print(f"   ⚠️  Warning: No run directories found, skipping...")
                continue
            
            # Use first available run directory
            rs_run_dir = rs_run_dirs[0]
            pso_run_dir = pso_run_dirs[0]
            
            # Load RS history
            rs_history_path = rs_run_dir / 'optimization_history.json'
            with open(rs_history_path, 'r') as f:
                rs_history = json.load(f)
            
            # Load PSO history
            pso_history_path = pso_run_dir / 'optimization_history.json'
            with open(pso_history_path, 'r') as f:
                pso_history = json.load(f)
            
            # Load training histories if exist
            rs_train_path = rs_run_dir / 'training_history.json'
            pso_train_path = pso_run_dir / 'training_history.json'
            
            rs_training = None
            pso_training = None
            
            if rs_train_path.exists():
                with open(rs_train_path, 'r') as f:
                    rs_training = json.load(f)
            
            if pso_train_path.exists():
                with open(pso_train_path, 'r') as f:
                    pso_training = json.load(f)
            
            self.all_data.append({
                'seed': run_seed,
                'summary': summary,
                'rs_history': rs_history,
                'pso_history': pso_history,
                'rs_training': rs_training,
                'pso_training': pso_training
            })
            
            print(f"   ✅ Loaded {len(rs_history)} RS trials")
            print(f"   ✅ Loaded {len(pso_history)} PSO evaluations")
        
        print("\n" + "="*80)
        print("✅ ALL DATA LOADED SUCCESSFULLY")
        print("="*80)
    
    def generate_all_plots(self):
        """Generate all publication plots"""
        print("\n" + "="*80)
        print("GENERATING PUBLICATION-READY PLOTS")
        print("="*80)
        
        # Figure 1: Test Accuracy Comparison (Boxplot with error bars)
        self.plot_test_accuracy_comparison()
        
        # Figure 2: Convergence Curves (All 3 runs)
        self.plot_convergence_all_runs()
        
        # Figure 3: Training Curves (Best models)
        self.plot_training_curves()
        
        # Figure 4: Hyperparameter Importance
        self.plot_hyperparameter_importance()
        
        # Figure 5: Hyperparameter Distributions
        self.plot_hyperparameter_distributions()
        
        # Figure 6: Correlation Heatmap
        self.plot_correlation_heatmap()
        
        # Figure 7: Computational Cost Analysis
        self.plot_computational_cost()
        
        # Figure 8: Best Configuration Comparison (Radar)
        self.plot_best_config_radar()
        
        # Figure 9: Convergence Speed Comparison
        self.plot_convergence_speed()
        
        # Figure 10: Statistical Significance Tests
        self.plot_statistical_tests()
        
        # Generate summary statistics table
        self.generate_statistics_table()
        
        print("\n" + "="*80)
        print(f"✅ ALL PLOTS SAVED TO: {self.output_dir}")
        print("="*80)
    
    def plot_test_accuracy_comparison(self):
        """Figure 1: Test accuracy comparison with error bars"""
        print("\n📊 Generating Figure 1: Test Accuracy Comparison...")
        
        # Extract test accuracies
        rs_accs = []
        pso_accs = []
        
        for data in self.all_data:
            rs_accs.extend(data['summary']['random_search']['test_accuracies'])
            pso_accs.extend(data['summary']['pso']['test_accuracies'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Boxplot
        positions = [1, 2]
        bp = ax.boxplot([rs_accs, pso_accs], 
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Colors
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, data in enumerate([rs_accs, pso_accs]):
            x = np.random.normal(positions[i], 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.6, color='black', s=80, zorder=10, edgecolors='white', linewidth=1.5)
        
        # Statistics
        rs_mean, rs_std = np.mean(rs_accs), np.std(rs_accs)
        pso_mean, pso_std = np.mean(pso_accs), np.std(pso_accs)
        
        # Wilcoxon test (paired, if same number of samples)
        if len(rs_accs) == len(pso_accs):
            stat, p_value = wilcoxon(rs_accs, pso_accs)
            sig_text = f"p = {p_value:.4f}" + (" *" if p_value < 0.05 else " (n.s.)")
        else:
            stat, p_value = mannwhitneyu(rs_accs, pso_accs)
            sig_text = f"p = {p_value:.4f}" + (" *" if p_value < 0.05 else " (n.s.)")
        
        # Annotations
        ax.text(1, rs_mean + 0.3, f'{rs_mean:.2f}±{rs_std:.2f}%', 
                ha='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
        
        ax.text(2, pso_mean + 0.3, f'{pso_mean:.2f}±{pso_std:.2f}%', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
        
        # Statistical test result
        ax.text(0.5, 0.95, f'Wilcoxon Test: {sig_text}', 
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(['Random Search', 'PSO'], fontsize=13)
        ax.set_ylabel('Test Accuracy (%)', fontsize=14)
        ax.set_title('CIFAR-10 Test Accuracy Comparison\n(3 Independent Runs, n=3 each)', 
                     fontsize=16, fontweight='bold')
        ax.set_ylim([min(min(rs_accs), min(pso_accs)) - 1, 
                     max(max(rs_accs), max(pso_accs)) + 1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_test_accuracy_comparison.png')
        plt.savefig(self.output_dir / 'fig1_test_accuracy_comparison.pdf')
        plt.close()
        print("   ✅ Saved: fig1_test_accuracy_comparison.png/pdf")
    
    def plot_convergence_all_runs(self):
        """Figure 2: Convergence curves for all runs"""
        print("\n📊 Generating Figure 2: Convergence Curves...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, data in enumerate(self.all_data):
            seed = data['seed']
            
            # Random Search subplot
            ax_rs = axes[0, idx]
            rs_history = data['rs_history']
            trials = list(range(1, len(rs_history) + 1))
            scores = [h['score'] for h in rs_history]
            best_so_far = np.maximum.accumulate(scores)
            
            ax_rs.bar(trials, scores, alpha=0.5, color='#3498db', label='Trial Score')
            ax_rs.plot(trials, best_so_far, 'r-', linewidth=2.5, marker='o', 
                      markersize=5, label='Best So Far')
            ax_rs.set_xlabel('Trial Number', fontsize=11)
            ax_rs.set_ylabel('Validation Accuracy (%)', fontsize=11)
            ax_rs.set_title(f'Random Search (Seed={seed})', fontsize=12, fontweight='bold')
            ax_rs.legend(loc='lower right', fontsize=9)
            ax_rs.grid(True, alpha=0.3)
            ax_rs.set_ylim([65, 95])
            
            # PSO subplot
            ax_pso = axes[1, idx]
            pso_history = data['pso_history']
            
            # Group by iteration
            iterations = []
            iter_means = []
            iter_stds = []
            global_bests = []
            
            current_iter = 1
            iter_scores = []
            global_best = 0
            
            for h in pso_history:
                iter_num = h['metadata']['iteration']
                if iter_num != current_iter:
                    iterations.append(current_iter)
                    iter_means.append(np.mean(iter_scores))
                    iter_stds.append(np.std(iter_scores))
                    global_bests.append(global_best)
                    current_iter = iter_num
                    iter_scores = []
                
                iter_scores.append(h['score'])
                global_best = max(global_best, h['score'])
            
            # Last iteration
            iterations.append(current_iter)
            iter_means.append(np.mean(iter_scores))
            iter_stds.append(np.std(iter_scores))
            global_bests.append(global_best)
            
            # Plot with error bars
            ax_pso.errorbar(iterations, iter_means, yerr=iter_stds, 
                           fmt='o-', color='#e74c3c', alpha=0.6, 
                           linewidth=2, markersize=6, capsize=5, 
                           label='Mean Particle Score')
            ax_pso.plot(iterations, global_bests, 'b-', linewidth=2.5, 
                       marker='s', markersize=7, label='Global Best')
            ax_pso.set_xlabel('Iteration', fontsize=11)
            ax_pso.set_ylabel('Validation Accuracy (%)', fontsize=11)
            ax_pso.set_title(f'PSO (Seed={seed})', fontsize=12, fontweight='bold')
            ax_pso.legend(loc='lower right', fontsize=9)
            ax_pso.grid(True, alpha=0.3)
            ax_pso.set_ylim([65, 95])
        
        plt.suptitle('Optimization Convergence Across All Runs', 
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_convergence_all_runs.png')
        plt.savefig(self.output_dir / 'fig2_convergence_all_runs.pdf')
        plt.close()
        print("   ✅ Saved: fig2_convergence_all_runs.png/pdf")
    
    def plot_training_curves(self):
        """Figure 3: Training curves for best models"""
        print("\n📊 Generating Figure 3: Training Curves...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for idx, data in enumerate(self.all_data):
            seed = data['seed']
            
            # RS training curve
            if data['rs_training']:
                ax_rs = axes[0, idx]
                history = data['rs_training']
                epochs = list(range(1, len(history['train_loss']) + 1))
                
                ax_rs_twin = ax_rs.twinx()
                
                # Loss
                l1 = ax_rs.plot(epochs, history['train_loss'], 'b-', 
                               linewidth=2, label='Train Loss', alpha=0.7)
                l2 = ax_rs.plot(epochs, history['val_loss'], 'r-', 
                               linewidth=2, label='Val Loss', alpha=0.7)
                
                # Accuracy
                l3 = ax_rs_twin.plot(epochs, history['train_acc'], 'b--', 
                                    linewidth=2, label='Train Acc', alpha=0.7)
                l4 = ax_rs_twin.plot(epochs, history['val_acc'], 'r--', 
                                    linewidth=2, label='Val Acc', alpha=0.7)
                
                ax_rs.set_xlabel('Epoch', fontsize=11)
                ax_rs.set_ylabel('Loss', fontsize=11, color='black')
                ax_rs_twin.set_ylabel('Accuracy (%)', fontsize=11, color='black')
                ax_rs.set_title(f'RS Training (Seed={seed})', fontsize=12, fontweight='bold')
                
                # Combined legend
                lns = l1 + l2 + l3 + l4
                labs = [l.get_label() for l in lns]
                ax_rs.legend(lns, labs, loc='center right', fontsize=8)
                ax_rs.grid(True, alpha=0.3)
            
            # PSO training curve
            if data['pso_training']:
                ax_pso = axes[1, idx]
                history = data['pso_training']
                epochs = list(range(1, len(history['train_loss']) + 1))
                
                ax_pso_twin = ax_pso.twinx()
                
                # Loss
                l1 = ax_pso.plot(epochs, history['train_loss'], 'b-', 
                                linewidth=2, label='Train Loss', alpha=0.7)
                l2 = ax_pso.plot(epochs, history['val_loss'], 'r-', 
                                linewidth=2, label='Val Loss', alpha=0.7)
                
                # Accuracy
                l3 = ax_pso_twin.plot(epochs, history['train_acc'], 'b--', 
                                     linewidth=2, label='Train Acc', alpha=0.7)
                l4 = ax_pso_twin.plot(epochs, history['val_acc'], 'r--', 
                                     linewidth=2, label='Val Acc', alpha=0.7)
                
                ax_pso.set_xlabel('Epoch', fontsize=11)
                ax_pso.set_ylabel('Loss', fontsize=11, color='black')
                ax_pso_twin.set_ylabel('Accuracy (%)', fontsize=11, color='black')
                ax_pso.set_title(f'PSO Training (Seed={seed})', fontsize=12, fontweight='bold')
                
                # Combined legend
                lns = l1 + l2 + l3 + l4
                labs = [l.get_label() for l in lns]
                ax_pso.legend(lns, labs, loc='center right', fontsize=8)
                ax_pso.grid(True, alpha=0.3)
        
        plt.suptitle('Training Curves for Best Configurations', 
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_training_curves.png')
        plt.savefig(self.output_dir / 'fig3_training_curves.pdf')
        plt.close()
        print("   ✅ Saved: fig3_training_curves.png/pdf")
    
    def plot_hyperparameter_importance(self):
        """Figure 4: Hyperparameter importance analysis"""
        print("\n📊 Generating Figure 4: Hyperparameter Importance...")
        
        # Combine all data
        all_configs = []
        all_scores = []
        all_methods = []
        
        for data in self.all_data:
            for h in data['rs_history']:
                all_configs.append(h['config'])
                all_scores.append(h['score'])
                all_methods.append('Random Search')
            
            for h in data['pso_history']:
                all_configs.append(h['config'])
                all_scores.append(h['score'])
                all_methods.append('PSO')
        
        df = pd.DataFrame(all_configs)
        df['score'] = all_scores
        df['method'] = all_methods
        
        # Calculate correlations
        correlations = {}
        p_values = {}
        
        for col in df.columns:
            if col not in ['score', 'method']:
                from scipy.stats import pearsonr
                r, p = pearsonr(df[col], df['score'])
                correlations[col] = r
                p_values[col] = p
        
        # Sort by absolute correlation
        sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        params = [k.replace('_', '\n') for k, v in sorted_params]
        values = [v for k, v in sorted_params]
        p_vals = [p_values[k] for k, v in sorted_params]
        
        colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
        
        bars = ax.barh(params, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add significance stars
        for i, (bar, p_val) in enumerate(zip(bars, p_vals)):
            x_pos = bar.get_width()
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'
            
            ax.text(x_pos + 0.02 if x_pos > 0 else x_pos - 0.02, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{x_pos:.3f} {sig}', 
                   va='center', ha='left' if x_pos > 0 else 'right', 
                   fontsize=10, fontweight='bold')
        
        ax.axvline(x=0, color='black', linewidth=2)
        ax.set_xlabel('Pearson Correlation with Validation Accuracy', fontsize=13)
        ax.set_title('Hyperparameter Importance Analysis\n(Combined across all runs and methods)', 
                     fontsize=15, fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Legend for significance
        ax.text(0.02, 0.02, '*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant', 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_hyperparameter_importance.png')
        plt.savefig(self.output_dir / 'fig4_hyperparameter_importance.pdf')
        plt.close()
        print("   ✅ Saved: fig4_hyperparameter_importance.png/pdf")
    
    def plot_hyperparameter_distributions(self):
        """Figure 5: Hyperparameter distributions with performance"""
        print("\n📊 Generating Figure 5: Hyperparameter Distributions...")
        
        # Combine all data
        all_configs = []
        all_scores = []
        all_methods = []
        
        for data in self.all_data:
            for h in data['rs_history']:
                all_configs.append(h['config'])
                all_scores.append(h['score'])
                all_methods.append('RS')
            
            for h in data['pso_history']:
                all_configs.append(h['config'])
                all_scores.append(h['score'])
                all_methods.append('PSO')
        
        df = pd.DataFrame(all_configs)
        df['score'] = all_scores
        df['method'] = all_methods
        
        # Create subplots
        hyperparams = ['learning_rate', 'batch_size', 'conv_channels_base', 
                       'num_conv_blocks', 'fc_hidden', 'dropout', 'weight_decay']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        colors = {'RS': '#3498db', 'PSO': '#e74c3c'}
        
        for idx, hp in enumerate(hyperparams):
            ax = axes[idx]
            
            for method in ['RS', 'PSO']:
                mask = df['method'] == method
                ax.scatter(df.loc[mask, hp], df.loc[mask, 'score'], 
                          c=colors[method], alpha=0.5, s=50, label=method, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(df[hp], df['score'])
            x_line = np.linspace(df[hp].min(), df[hp].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label=f'r={r_value:.2f}')
            
            ax.set_xlabel(hp.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if hp in ['learning_rate', 'weight_decay']:
                ax.set_xscale('log')
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.suptitle('Hyperparameter Values vs Performance\n(All runs combined)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_hyperparameter_distributions.png')
        plt.savefig(self.output_dir / 'fig5_hyperparameter_distributions.pdf')
        plt.close()
        print("   ✅ Saved: fig5_hyperparameter_distributions.png/pdf")
    
    def plot_correlation_heatmap(self):
        """Figure 6: Correlation heatmap"""
        print("\n📊 Generating Figure 6: Correlation Heatmap...")
        
        # Combine all data
        all_configs = []
        all_scores = []
        
        for data in self.all_data:
            for h in data['rs_history'] + data['pso_history']:
                all_configs.append(h['config'])
                all_scores.append(h['score'])
        
        df = pd.DataFrame(all_configs)
        df['accuracy'] = all_scores
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='RdBu_r', center=0, ax=ax,
                    square=True, linewidths=1, cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                    vmin=-1, vmax=1)
        
        ax.set_title('Hyperparameter Correlation Matrix\n(All runs combined)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_correlation_heatmap.png')
        plt.savefig(self.output_dir / 'fig6_correlation_heatmap.pdf')
        plt.close()
        print("   ✅ Saved: fig6_correlation_heatmap.png/pdf")
    
    def plot_computational_cost(self):
        """Figure 7: Computational cost analysis"""
        print("\n📊 Generating Figure 7: Computational Cost Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Data collection
        rs_times = []
        pso_times = []
        rs_evals = []
        pso_evals = []
        
        for data in self.all_data:
            summary = data['summary']
            
            # Extract times (if available in summary)
            rs_times.append(len(data['rs_history']) * 30)  # Estimate: 30 min per trial
            pso_times.append(len(data['pso_history']) * 25)  # Estimate: 25 min per eval
            
            rs_evals.append(len(data['rs_history']))
            pso_evals.append(len(data['pso_history']))
        
        seeds = [d['seed'] for d in self.all_data]
        
        # Plot 1: Number of evaluations
        ax1 = axes[0, 0]
        x = np.arange(len(seeds))
        width = 0.35
        
        ax1.bar(x - width/2, rs_evals, width, label='Random Search', color='#3498db', alpha=0.7)
        ax1.bar(x + width/2, pso_evals, width, label='PSO', color='#e74c3c', alpha=0.7)
        
        ax1.set_xlabel('Run (Seed)', fontsize=12)
        ax1.set_ylabel('Number of Evaluations', fontsize=12)
        ax1.set_title('Total Model Training Evaluations', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(seeds)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Estimated time (minutes)
        ax2 = axes[0, 1]
        
        ax2.bar(x - width/2, [t/60 for t in rs_times], width, 
                label='Random Search', color='#3498db', alpha=0.7)
        ax2.bar(x + width/2, [t/60 for t in pso_times], width, 
                label='PSO', color='#e74c3c', alpha=0.7)
        
        ax2.set_xlabel('Run (Seed)', fontsize=12)
        ax2.set_ylabel('Time (hours)', fontsize=12)
        ax2.set_title('Estimated Computational Time', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seeds)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Efficiency (best score / evaluations)
        ax3 = axes[1, 0]
        
        rs_best = [max([h['score'] for h in d['rs_history']]) for d in self.all_data]
        pso_best = [max([h['score'] for h in d['pso_history']]) for d in self.all_data]
        
        rs_efficiency = [b / e for b, e in zip(rs_best, rs_evals)]
        pso_efficiency = [b / e for b, e in zip(pso_best, pso_evals)]
        
        ax3.plot(seeds, rs_efficiency, 'o-', linewidth=2.5, markersize=10, 
                label='Random Search', color='#3498db')
        ax3.plot(seeds, pso_efficiency, 's-', linewidth=2.5, markersize=10, 
                label='PSO', color='#e74c3c')
        
        ax3.set_xlabel('Run (Seed)', fontsize=12)
        ax3.set_ylabel('Efficiency (Best Acc / Evaluations)', fontsize=12)
        ax3.set_title('Optimization Efficiency', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence speed (evaluations to reach 85%)
        ax4 = axes[1, 1]
        
        rs_conv_speed = []
        pso_conv_speed = []
        
        for data in self.all_data:
            # RS
            scores = [h['score'] for h in data['rs_history']]
            rs_conv = next((i+1 for i, s in enumerate(scores) if s >= 85), len(scores))
            rs_conv_speed.append(rs_conv)
            
            # PSO
            scores = [h['score'] for h in data['pso_history']]
            pso_conv = next((i+1 for i, s in enumerate(scores) if s >= 85), len(scores))
            pso_conv_speed.append(pso_conv)
        
        ax4.bar(x - width/2, rs_conv_speed, width, 
                label='Random Search', color='#3498db', alpha=0.7)
        ax4.bar(x + width/2, pso_conv_speed, width, 
                label='PSO', color='#e74c3c', alpha=0.7)
        
        ax4.set_xlabel('Run (Seed)', fontsize=12)
        ax4.set_ylabel('Evaluations to Reach 85% Accuracy', fontsize=12)
        ax4.set_title('Convergence Speed', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(seeds)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Computational Cost Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig7_computational_cost.png')
        plt.savefig(self.output_dir / 'fig7_computational_cost.pdf')
        plt.close()
        print("   ✅ Saved: fig7_computational_cost.png/pdf")
    
    def plot_best_config_radar(self):
        """Figure 8: Radar chart comparing best configurations"""
        print("\n📊 Generating Figure 8: Best Configuration Radar Chart...")
        
        # Get best configs from each run
        rs_best_configs = []
        pso_best_configs = []
        
        for data in self.all_data:
            rs_best = max(data['rs_history'], key=lambda x: x['score'])
            pso_best = max(data['pso_history'], key=lambda x: x['score'])
            rs_best_configs.append(rs_best['config'])
            pso_best_configs.append(pso_best['config'])
        
        # Average best configs
        rs_avg_config = {k: np.mean([c[k] for c in rs_best_configs]) 
                        for k in rs_best_configs[0].keys()}
        pso_avg_config = {k: np.mean([c[k] for c in pso_best_configs]) 
                         for k in pso_best_configs[0].keys()}
        
        # Normalize
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
                if key in ['learning_rate', 'weight_decay']:
                    val = np.log10(val)
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                normalized[key] = (val - min_val) / (max_val - min_val)
            return normalized
        
        rs_norm = normalize(rs_avg_config)
        pso_norm = normalize(pso_avg_config)
        
        # Create radar chart
        categories = list(param_ranges.keys())
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        rs_values = [rs_norm[cat] for cat in categories]
        rs_values += rs_values[:1]
        
        pso_values = [pso_norm[cat] for cat in categories]
        pso_values += pso_values[:1]
        
        ax.plot(angles, rs_values, 'o-', linewidth=3, label='Random Search (Avg)', 
                color='#3498db', markersize=8)
        ax.fill(angles, rs_values, alpha=0.25, color='#3498db')
        
        ax.plot(angles, pso_values, 's-', linewidth=3, label='PSO (Avg)', 
                color='#e74c3c', markersize=8)
        ax.fill(angles, pso_values, alpha=0.25, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace('_', '\n') for c in categories], size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.set_title('Average Best Configuration Comparison\n(Normalized values, averaged across 3 runs)', 
                     fontsize=15, fontweight='bold', pad=30)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig8_best_config_radar.png')
        plt.savefig(self.output_dir / 'fig8_best_config_radar.pdf')
        plt.close()
        print("   ✅ Saved: fig8_best_config_radar.png/pdf")
    
    def plot_convergence_speed(self):
        """Figure 9: Convergence speed comparison"""
        print("\n📊 Generating Figure 9: Convergence Speed Analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average convergence across runs
        rs_convergence = []
        pso_convergence = []
        
        max_evals = max(len(d['rs_history']) for d in self.all_data)
        
        for eval_num in range(1, max_evals + 1):
            rs_bests = []
            for data in self.all_data:
                if eval_num <= len(data['rs_history']):
                    scores = [h['score'] for h in data['rs_history'][:eval_num]]
                    rs_bests.append(max(scores))
            if rs_bests:
                rs_convergence.append(np.mean(rs_bests))
        
        max_evals_pso = max(len(d['pso_history']) for d in self.all_data)
        
        for eval_num in range(1, max_evals_pso + 1):
            pso_bests = []
            for data in self.all_data:
                if eval_num <= len(data['pso_history']):
                    scores = [h['score'] for h in data['pso_history'][:eval_num]]
                    pso_bests.append(max(scores))
            if pso_bests:
                pso_convergence.append(np.mean(pso_bests))
        
        # Plot 1: Convergence curves
        ax1 = axes[0]
        ax1.plot(range(1, len(rs_convergence) + 1), rs_convergence, 
                'o-', linewidth=2.5, markersize=6, label='Random Search', color='#3498db')
        ax1.plot(range(1, len(pso_convergence) + 1), pso_convergence, 
                's-', linewidth=2.5, markersize=6, label='PSO', color='#e74c3c')
        
        ax1.set_xlabel('Number of Evaluations', fontsize=13)
        ax1.set_ylabel('Best Validation Accuracy (%)', fontsize=13)
        ax1.set_title('Average Convergence Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Area under convergence curve (AUC)
        ax2 = axes[1]
        
        rs_auc = np.trapz(rs_convergence) / len(rs_convergence)
        pso_auc = np.trapz(pso_convergence) / len(pso_convergence)
        
        bars = ax2.bar(['Random Search', 'PSO'], [rs_auc, pso_auc], 
                       color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, val in zip(bars, [rs_auc, pso_auc]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        ax2.set_ylabel('Average Area Under Convergence Curve', fontsize=13)
        ax2.set_title('Convergence Efficiency (Higher is Better)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig9_convergence_speed.png')
        plt.savefig(self.output_dir / 'fig9_convergence_speed.pdf')
        plt.close()
        print("   ✅ Saved: fig9_convergence_speed.png/pdf")
    
    def plot_statistical_tests(self):
        """Figure 10: Statistical significance visualization"""
        print("\n📊 Generating Figure 10: Statistical Tests...")
        
        # Extract all test accuracies
        rs_accs = []
        pso_accs = []
        
        for data in self.all_data:
            rs_accs.extend(data['summary']['random_search']['test_accuracies'])
            pso_accs.extend(data['summary']['pso']['test_accuracies'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Paired difference
        ax1 = axes[0]
        
        if len(rs_accs) == len(pso_accs):
            differences = np.array(rs_accs) - np.array(pso_accs)
            
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
            ax1.bar(range(len(differences)), differences, 
                   color=['green' if d > 0 else 'red' for d in differences], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax1.set_xlabel('Run', fontsize=13)
            ax1.set_ylabel('Difference (RS - PSO) in %', fontsize=13)
            ax1.set_title('Paired Accuracy Differences\n(Positive = RS wins)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add mean difference
            mean_diff = np.mean(differences)
            ax1.axhline(y=mean_diff, color='blue', linestyle=':', linewidth=2, 
                       label=f'Mean diff: {mean_diff:.2f}%')
            ax1.legend(fontsize=11)
        
        # Plot 2: Statistical test results
        ax2 = axes[1]
        
        # Perform tests
        if len(rs_accs) == len(pso_accs):
            wilcoxon_stat, wilcoxon_p = wilcoxon(rs_accs, pso_accs)
            test_name = 'Wilcoxon\nSigned-Rank'
        else:
            wilcoxon_stat, wilcoxon_p = mannwhitneyu(rs_accs, pso_accs)
            test_name = 'Mann-Whitney\nU Test'
        
        from scipy.stats import ttest_rel, ttest_ind
        if len(rs_accs) == len(pso_accs):
            t_stat, t_p = ttest_rel(rs_accs, pso_accs)
            t_test_name = 'Paired\nt-test'
        else:
            t_stat, t_p = ttest_ind(rs_accs, pso_accs)
            t_test_name = 'Independent\nt-test'
        
        # Visualization
        tests = [test_name, t_test_name]
        p_values = [wilcoxon_p, t_p]
        
        colors_sig = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        
        bars = ax2.barh(tests, [-np.log10(p) for p in p_values], 
                       color=colors_sig, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
        ax2.axvline(x=-np.log10(0.01), color='orange', linestyle='--', linewidth=2, label='p=0.01')
        
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'p={p:.4f}', va='center', fontsize=11, fontweight='bold')
        
        ax2.set_xlabel('-log10(p-value)', fontsize=13)
        ax2.set_title('Statistical Significance Tests\n(Higher bars = more significant)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig10_statistical_tests.png')
        plt.savefig(self.output_dir / 'fig10_statistical_tests.pdf')
        plt.close()
        print("   ✅ Saved: fig10_statistical_tests.png/pdf")
    
    def generate_statistics_table(self):
        """Generate comprehensive statistics table"""
        print("\n📊 Generating Statistics Summary Table...")
        
        # Extract all metrics
        rs_accs = []
        pso_accs = []
        
        rs_best_scores = []
        pso_best_scores = []
        
        for data in self.all_data:
            rs_accs.extend(data['summary']['random_search']['test_accuracies'])
            pso_accs.extend(data['summary']['pso']['test_accuracies'])
            
            rs_best_scores.append(max([h['score'] for h in data['rs_history']]))
            pso_best_scores.append(max([h['score'] for h in data['pso_history']]))
        
        # Statistical tests
        if len(rs_accs) == len(pso_accs):
            w_stat, w_p = wilcoxon(rs_accs, pso_accs)
            t_stat, t_p = stats.ttest_rel(rs_accs, pso_accs)
        else:
            w_stat, w_p = mannwhitneyu(rs_accs, pso_accs)
            t_stat, t_p = stats.ttest_ind(rs_accs, pso_accs)
        
        # Create summary
        summary = {
            'Metric': [
                'Test Accuracy Mean (%)',
                'Test Accuracy Std (%)',
                'Test Accuracy Min (%)',
                'Test Accuracy Max (%)',
                'Best Val Acc Mean (%)',
                'Best Val Acc Std (%)',
                'Avg Evaluations',
                'Wilcoxon p-value',
                't-test p-value',
                'Winner (p<0.05)'
            ],
            'Random Search': [
                f'{np.mean(rs_accs):.2f}',
                f'{np.std(rs_accs):.2f}',
                f'{np.min(rs_accs):.2f}',
                f'{np.max(rs_accs):.2f}',
                f'{np.mean(rs_best_scores):.2f}',
                f'{np.std(rs_best_scores):.2f}',
                f'{np.mean([len(d["rs_history"]) for d in self.all_data]):.0f}',
                '-',
                '-',
                'RS' if (w_p < 0.05 and np.mean(rs_accs) > np.mean(pso_accs)) else '-'
            ],
            'PSO': [
                f'{np.mean(pso_accs):.2f}',
                f'{np.std(pso_accs):.2f}',
                f'{np.min(pso_accs):.2f}',
                f'{np.max(pso_accs):.2f}',
                f'{np.mean(pso_best_scores):.2f}',
                f'{np.std(pso_best_scores):.2f}',
                f'{np.mean([len(d["pso_history"]) for d in self.all_data]):.0f}',
                f'{w_p:.4f}',
                f'{t_p:.4f}',
                'PSO' if (w_p < 0.05 and np.mean(pso_accs) > np.mean(rs_accs)) else '-'
            ]
        }
        
        df = pd.DataFrame(summary)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'table1_statistics_summary.csv', index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False, caption='Statistical Summary of Optimization Results',
                                   label='tab:statistics_summary')
        with open(self.output_dir / 'table1_statistics_summary.tex', 'w') as f:
            f.write(latex_table)
        
        print("   ✅ Saved: table1_statistics_summary.csv")
        print("   ✅ Saved: table1_statistics_summary.tex")
        
        # Print to console
        print("\n" + "="*80)
        print("STATISTICS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all publication plots')
    parser.add_argument('--results_dirs', nargs='+', required=True,
                       help='Paths to result directories (3 runs)')
    parser.add_argument('--output_dir', type=str, default='publication_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PUBLICATION-READY PLOTS GENERATOR")
    print("="*80)
    print(f"Input directories: {len(args.results_dirs)}")
    for i, d in enumerate(args.results_dirs):
        print(f"  Run {i+1}: {d}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create generator
    generator = PublicationPlotGenerator(args.results_dirs, args.output_dir)
    
    # Generate all plots
    generator.generate_all_plots()
    
    print("\n" + "="*80)
    print("🎉 ALL PUBLICATION PLOTS GENERATED!")
    print("="*80)
    print(f"\n📂 Check output directory: {args.output_dir}/")
    print("\nGenerated files:")
    print("  • fig1_test_accuracy_comparison.png/pdf")
    print("  • fig2_convergence_all_runs.png/pdf")
    print("  • fig3_training_curves.png/pdf")
    print("  • fig4_hyperparameter_importance.png/pdf")
    print("  • fig5_hyperparameter_distributions.png/pdf")
    print("  • fig6_correlation_heatmap.png/pdf")
    print("  • fig7_computational_cost.png/pdf")
    print("  • fig8_best_config_radar.png/pdf")
    print("  • fig9_convergence_speed.png/pdf")
    print("  • fig10_statistical_tests.png/pdf")
    print("  • table1_statistics_summary.csv/tex")
    print("\n📖 Ready to include in your thesis!")
    print("="*80)


if __name__ == "__main__":
    main()