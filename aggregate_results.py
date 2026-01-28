"""
Aggregate results from multiple experiment runs for statistical analysis.
Generates summary with mean, std, confidence intervals, and error bars.

Usage:
    python aggregate_results.py --results_dir experiments/results
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class MultiRunAnalyzer:
    """Analyze and aggregate results from multiple experiment runs."""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.all_results = []
        self.summary = {}
    
    def find_experiment_dirs(self):
        """Find all experiment directories."""
        experiment_dirs = sorted([d for d in self.results_dir.glob('CIFAR10_CNN_*') if d.is_dir()])
        print(f"\n✅ Found {len(experiment_dirs)} experiment directories")
        for exp_dir in experiment_dirs:
            print(f"   - {exp_dir.name}")
        return experiment_dirs
    
    def load_experiment_results(self, exp_dir):
        """Load results from a single experiment directory."""
        summary_path = exp_dir / 'summary.json'
        
        if not summary_path.exists():
            print(f"⚠️  Warning: No summary.json in {exp_dir.name}")
            return None
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Extract key metrics
        results = {
            'experiment_dir': exp_dir.name,
            'random_search_best': summary['random_search']['best_config']['score'],
            'random_search_test': summary['random_search']['test_accuracies'][0],
            'pso_best': summary['pso']['best_config']['score'],
            'pso_test': summary['pso']['test_accuracies'][0],
            'random_search_config': summary['random_search']['best_config']['config'],
            'pso_config': summary['pso']['best_config']['config']
        }
        
        return results
    
    def aggregate_all_results(self):
        """Aggregate results from all experiments."""
        experiment_dirs = self.find_experiment_dirs()
        
        for exp_dir in experiment_dirs:
            result = self.load_experiment_results(exp_dir)
            if result:
                self.all_results.append(result)
        
        if len(self.all_results) == 0:
            print("\n❌ No valid results found!")
            return False
        
        print(f"\n✅ Loaded {len(self.all_results)} valid experiment results\n")
        return True
    
    def compute_statistics(self):
        """Compute statistical metrics."""
        df = pd.DataFrame(self.all_results)
        
        # Random Search statistics
        rs_exploration = df['random_search_best'].values
        rs_test = df['random_search_test'].values
        
        # PSO statistics
        pso_exploration = df['pso_best'].values
        pso_test = df['pso_test'].values
        
        self.summary['random_search'] = self._compute_stats(rs_exploration, rs_test, 'Random Search')
        self.summary['pso'] = self._compute_stats(pso_exploration, pso_test, 'PSO')
        
        # Statistical comparison
        self.summary['comparison'] = self._statistical_comparison(rs_test, pso_test)
        
        return self.summary
    
    def _compute_stats(self, exploration_scores, test_scores, method_name):
        """Compute statistics for a single method."""
        n = len(test_scores)
        
        stats_dict = {
            'n_runs': n,
            'exploration': {
                'mean': float(np.mean(exploration_scores)),
                'std': float(np.std(exploration_scores, ddof=1)),
                'min': float(np.min(exploration_scores)),
                'max': float(np.max(exploration_scores)),
                'median': float(np.median(exploration_scores))
            },
            'test': {
                'mean': float(np.mean(test_scores)),
                'std': float(np.std(test_scores, ddof=1)),
                'min': float(np.min(test_scores)),
                'max': float(np.max(test_scores)),
                'median': float(np.median(test_scores))
            }
        }
        
        # Confidence intervals (95%)
        if n >= 2:
            sem = stats.sem(test_scores)
            ci = stats.t.interval(0.95, n-1, loc=np.mean(test_scores), scale=sem)
            stats_dict['test']['confidence_interval_95'] = [float(ci[0]), float(ci[1])]
            stats_dict['test']['margin_of_error'] = float(ci[1] - np.mean(test_scores))
        
        return stats_dict
    
    def _statistical_comparison(self, rs_scores, pso_scores):
        """Perform statistical tests comparing RS and PSO."""
        comparison = {}
        
        n_rs = len(rs_scores)
        n_pso = len(pso_scores)
        
        # Basic comparison
        comparison['mean_difference'] = float(np.mean(rs_scores) - np.mean(pso_scores))
        comparison['winner'] = 'Random Search' if np.mean(rs_scores) > np.mean(pso_scores) else 'PSO'
        
        # Statistical tests (if enough samples)
        if n_rs >= 2 and n_pso >= 2:
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(rs_scores, pso_scores, alternative='two-sided')
            comparison['mann_whitney_u'] = {
                'statistic': float(u_stat),
                'p_value': float(u_p),
                'significant': u_p < 0.05
            }
            
            # T-test (parametric)
            t_stat, t_p = stats.ttest_ind(rs_scores, pso_scores)
            comparison['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < 0.05
            }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(rs_scores, ddof=1) + np.var(pso_scores, ddof=1)) / 2)
        if pooled_std > 0:
            comparison['cohens_d'] = float((np.mean(rs_scores) - np.mean(pso_scores)) / pooled_std)
        
        return comparison
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "="*80)
        print("MULTI-RUN STATISTICAL ANALYSIS")
        print("="*80)
        
        # Random Search
        print("\n🔵 RANDOM SEARCH (n={})".format(self.summary['random_search']['n_runs']))
        print("-" * 80)
        print(f"Exploration Phase (Best Validation Accuracy):")
        print(f"  Mean: {self.summary['random_search']['exploration']['mean']:.4f}%")
        print(f"  Std:  ±{self.summary['random_search']['exploration']['std']:.4f}%")
        print(f"  Range: [{self.summary['random_search']['exploration']['min']:.2f}, {self.summary['random_search']['exploration']['max']:.2f}]")
        
        print(f"\nTest Phase (Final Test Accuracy):")
        print(f"  Mean: {self.summary['random_search']['test']['mean']:.4f}%")
        print(f"  Std:  ±{self.summary['random_search']['test']['std']:.4f}%")
        if 'confidence_interval_95' in self.summary['random_search']['test']:
            ci = self.summary['random_search']['test']['confidence_interval_95']
            print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        # PSO
        print("\n🔴 PSO (n={})".format(self.summary['pso']['n_runs']))
        print("-" * 80)
        print(f"Exploration Phase (Best Validation Accuracy):")
        print(f"  Mean: {self.summary['pso']['exploration']['mean']:.4f}%")
        print(f"  Std:  ±{self.summary['pso']['exploration']['std']:.4f}%")
        print(f"  Range: [{self.summary['pso']['exploration']['min']:.2f}, {self.summary['pso']['exploration']['max']:.2f}]")
        
        print(f"\nTest Phase (Final Test Accuracy):")
        print(f"  Mean: {self.summary['pso']['test']['mean']:.4f}%")
        print(f"  Std:  ±{self.summary['pso']['test']['std']:.4f}%")
        if 'confidence_interval_95' in self.summary['pso']['test']:
            ci = self.summary['pso']['test']['confidence_interval_95']
            print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        # Comparison
        print("\n⚔️  STATISTICAL COMPARISON")
        print("-" * 80)
        comp = self.summary['comparison']
        print(f"Winner: {comp['winner']}")
        print(f"Mean Difference: {comp['mean_difference']:.4f}%")
        
        if 'mann_whitney_u' in comp:
            mw = comp['mann_whitney_u']
            sig = "✅ Significant" if mw['significant'] else "❌ Not Significant"
            print(f"\nMann-Whitney U Test:")
            print(f"  p-value: {mw['p_value']:.4f} {sig} (α=0.05)")
            
        if 't_test' in comp:
            tt = comp['t_test']
            sig = "✅ Significant" if tt['significant'] else "❌ Not Significant"
            print(f"\nT-Test:")
            print(f"  p-value: {tt['p_value']:.4f} {sig} (α=0.05)")
        
        if 'cohens_d' in comp:
            d = comp['cohens_d']
            effect = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            print(f"\nEffect Size (Cohen's d): {d:.4f} ({effect})")
        
        print("\n" + "="*80)
    
    def generate_comparison_plot(self, output_dir):
        """Generate publication-quality comparison plot with error bars."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        rs_mean = self.summary['random_search']['test']['mean']
        rs_std = self.summary['random_search']['test']['std']
        pso_mean = self.summary['pso']['test']['mean']
        pso_std = self.summary['pso']['test']['std']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['Random Search', 'PSO']
        means = [rs_mean, pso_mean]
        stds = [rs_std, pso_std]
        colors = ['#3498db', '#e74c3c']
        
        x_pos = np.arange(len(methods))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add individual data points
        for exp in self.all_results:
            ax.scatter([0], [exp['random_search_test']], color='blue', s=100, alpha=0.8, zorder=10)
            ax.scatter([1], [exp['pso_test']], color='red', s=100, alpha=0.8, zorder=10)
        
        ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Hyperparameter Optimization Methods Comparison (n={len(self.all_results)} runs)', 
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=12)
        ax.set_ylim([min(means) - 2*max(stds), max(means) + 2*max(stds)])
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.2, f'{mean:.2f}% ± {std:.2f}%', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save
        output_path = output_dir / 'multi_run_comparison_with_error_bars.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'multi_run_comparison_with_error_bars.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved comparison plot: {output_path}")
    
    def save_results(self, output_dir):
        """Save aggregated results."""
        output_dir = Path(output_dir)
        
        # Save summary JSON
        with open(output_dir / 'multi_run_summary.json', 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        # Save detailed CSV
        df = pd.DataFrame(self.all_results)
        df.to_csv(output_dir / 'multi_run_detailed_results.csv', index=False)
        
        # Save statistics CSV
        stats_data = []
        for method in ['random_search', 'pso']:
            stats_data.append({
                'method': method.replace('_', ' ').title(),
                'n_runs': self.summary[method]['n_runs'],
                'exploration_mean': self.summary[method]['exploration']['mean'],
                'exploration_std': self.summary[method]['exploration']['std'],
                'test_mean': self.summary[method]['test']['mean'],
                'test_std': self.summary[method]['test']['std'],
                'test_min': self.summary[method]['test']['min'],
                'test_max': self.summary[method]['test']['max']
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(output_dir / 'multi_run_statistics.csv', index=False)
        
        print(f"\n✅ Saved results to: {output_dir}")
        print(f"   - multi_run_summary.json")
        print(f"   - multi_run_detailed_results.csv")
        print(f"   - multi_run_statistics.csv")


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-run experiment results')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='multi_run_analysis',
                       help='Output directory for aggregated results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTI-RUN RESULTS AGGREGATION & STATISTICAL ANALYSIS")
    print("="*80)
    
    # Create analyzer
    analyzer = MultiRunAnalyzer(args.results_dir)
    
    # Load and aggregate results
    if not analyzer.aggregate_all_results():
        return
    
    # Compute statistics
    analyzer.compute_statistics()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate plots
    analyzer.generate_comparison_plot(args.output_dir)
    
    # Save results
    analyzer.save_results(args.output_dir)
    
    print("\n🎉 Analysis complete!")
    print(f"\n📊 For your thesis:")
    print(f"   - Use error bars from: {args.output_dir}/multi_run_comparison_with_error_bars.png")
    print(f"   - Report statistics from: {args.output_dir}/multi_run_statistics.csv")
    print(f"   - Statistical tests in: {args.output_dir}/multi_run_summary.json\n")


if __name__ == "__main__":
    main()
