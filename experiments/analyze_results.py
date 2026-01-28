"""
Statistical analysis of experiment results.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_summary(results_dir):
    """Load summary.json from results directory."""
    results_dir = Path(results_dir)
    with open(results_dir / 'summary.json', 'r') as f:
        return json.load(f)


def calculate_statistics(accuracies):
    """Calculate summary statistics."""
    if len(accuracies) == 0:
        return {}
    
    return {
        'mean': float(np.mean(accuracies)),
        'std': float(np.std(accuracies)),
        'min': float(np.min(accuracies)),
        'max': float(np.max(accuracies)),
        'median': float(np.median(accuracies)),
        'n': len(accuracies)
    }


def perform_wilcoxon_test(rs_accs, pso_accs):
    """Perform Wilcoxon signed-rank test."""
    if len(rs_accs) != len(pso_accs) or len(rs_accs) < 3:
        return None
    
    statistic, p_value = stats.wilcoxon(rs_accs, pso_accs, alternative='two-sided')
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }


def analyze_hyperparameter_correlations(results_dir):
    """Analyze correlations between hyperparameters and performance."""
    results_dir = Path(results_dir)
    
    all_configs = []
    all_scores = []
    
    for opt_name in ['random_search', 'pso']:
        opt_dir = results_dir / opt_name
        if not opt_dir.exists():
            continue
        
        for run_dir in sorted(opt_dir.iterdir()):
            if run_dir.is_dir():
                with open(run_dir / 'optimization_history.json', 'r') as f:
                    history = json.load(f)
                
                for h in history:
                    all_configs.append(h['config'])
                    all_scores.append(h['score'])
    
    if len(all_configs) == 0:
        return {}
    
    df = pd.DataFrame(all_configs)
    df['score'] = all_scores
    
    correlations = {}
    for col in df.columns:
        if col != 'score':
            corr = df[col].corr(df['score'])
            correlations[col] = {
                'correlation': float(corr),
                'abs_correlation': float(abs(corr))
            }
    
    return correlations


def generate_statistical_analysis(results_dir):
    """Generate comprehensive statistical analysis."""
    results_dir = Path(results_dir)
    
    # Load summary
    summary = load_summary(results_dir)
    
    # Create output directory
    stats_dir = results_dir / 'statistical_analysis'
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    rs_accs = summary['random_search']['test_accuracies']
    pso_accs = summary['pso']['test_accuracies']
    
    rs_stats = calculate_statistics(rs_accs)
    pso_stats = calculate_statistics(pso_accs)
    
    summary_stats = pd.DataFrame({
        'Random Search': rs_stats,
        'PSO': pso_stats
    }).T
    
    summary_stats.to_csv(stats_dir / 'summary_statistics.csv')
    print("Saved summary_statistics.csv")
    
    # Pairwise tests
    wilcoxon_result = perform_wilcoxon_test(rs_accs, pso_accs)
    
    if wilcoxon_result:
        pairwise_tests = pd.DataFrame([{
            'test': 'Wilcoxon signed-rank test',
            'statistic': wilcoxon_result['statistic'],
            'p_value': wilcoxon_result['p_value'],
            'significant': wilcoxon_result['significant']
        }])
        pairwise_tests.to_csv(stats_dir / 'pairwise_tests.csv', index=False)
        print("Saved pairwise_tests.csv")
    
    # Hyperparameter correlations
    correlations = analyze_hyperparameter_correlations(results_dir)
    if correlations:
        corr_df = pd.DataFrame(correlations).T
        corr_df.to_csv(stats_dir / 'hyperparameter_correlations.csv')
        print("Saved hyperparameter_correlations.csv")
    
    # Efficiency metrics (if timing data available)
    # This would require tracking training times in the experiment
    efficiency_df = pd.DataFrame({
        'Optimizer': ['Random Search', 'PSO'],
        'Mean Accuracy': [rs_stats['mean'], pso_stats['mean']],
        'Std Accuracy': [rs_stats['std'], pso_stats['std']]
    })
    efficiency_df.to_csv(stats_dir / 'efficiency_metrics.csv', index=False)
    print("Saved efficiency_metrics.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    print("\nSummary Statistics:")
    print(summary_stats)
    
    if wilcoxon_result:
        print(f"\nWilcoxon Test:")
        print(f"  Statistic: {wilcoxon_result['statistic']:.4f}")
        print(f"  p-value: {wilcoxon_result['p_value']:.4f}")
        print(f"  Significant: {wilcoxon_result['significant']}")
    
    print(f"\nResults saved to: {stats_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    generate_statistical_analysis(results_dir)

