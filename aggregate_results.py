"""
Aggregate Results from Multiple Experimental Runs
Combines statistics from multiple independent runs with different seeds
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind
import warnings
warnings.filterwarnings('ignore')


class MultiRunAggregator:
    """Aggregate and analyze results from multiple runs"""
    
    def __init__(self, results_dir):
        """
        Args:
            results_dir: Directory containing results from multiple runs
        """
        self.results_dir = Path(results_dir)
        self.experiment_dirs = []
        self.all_data = []
        
    def find_experiment_directories(self):
        """Find all experiment directories"""
        print("\n" + "="*80)
        print("SEARCHING FOR EXPERIMENT DIRECTORIES")
        print("="*80)
        
        # Look for directories with pattern CIFAR10_CNN_*
        base_results = Path('experiments/results')
        
        if base_results.exists():
            all_dirs = sorted([d for d in base_results.iterdir() 
                               if d.is_dir() and d.name.startswith('CIFAR10_CNN_')])
            # Only include dirs that have BOTH random_search and pso sub-dirs (full experiments)
            experiment_dirs = [d for d in all_dirs 
                               if (d / 'random_search').exists() and (d / 'pso').exists()]
            
            print(f"\n📂 Found {len(experiment_dirs)} full experiment directories:")
            for i, exp_dir in enumerate(experiment_dirs):
                print(f"   {i+1}. {exp_dir.name}")
                self.experiment_dirs.append(exp_dir)
        
        # Also check multi_run_results directory
        if self.results_dir.exists():
            config_files = list(self.results_dir.glob('config_seed_*.yaml'))
            print(f"\n📄 Found {len(config_files)} config files in {self.results_dir}")
        
        if not self.experiment_dirs:
            print("\n❌ No experiment directories found!")
            print(f"   Searched in: {base_results}")
            return False
        
        print(f"\n✅ Total experiments to aggregate: {len(self.experiment_dirs)}")
        return True
    
    def load_all_data(self):
        """Load data from all experiment directories"""
        print("\n" + "="*80)
        print("LOADING DATA FROM ALL EXPERIMENTS")
        print("="*80)
        
        for idx, exp_dir in enumerate(self.experiment_dirs):
            print(f"\n📂 Loading experiment {idx+1}/{len(self.experiment_dirs)}: {exp_dir.name}")
            
            try:
                # Load summary.json
                summary_path = exp_dir / 'summary.json'
                if not summary_path.exists():
                    print(f"   ⚠️  Warning: summary.json not found, skipping...")
                    continue
                
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                # Extract seed from directory name or summary
                seed = summary.get('seed', 42)
                
                # Load optimization histories
                rs_runs = list((exp_dir / 'random_search').glob('run_*'))
                pso_runs = list((exp_dir / 'pso').glob('run_*'))
                
                rs_history = []
                pso_history = []
                bo_history = []
                
                for rs_run in rs_runs:
                    opt_hist_path = rs_run / 'optimization_history.json'
                    if opt_hist_path.exists():
                        with open(opt_hist_path, 'r') as f:
                            rs_history.extend(json.load(f))
                
                for pso_run in pso_runs:
                    opt_hist_path = pso_run / 'optimization_history.json'
                    if opt_hist_path.exists():
                        with open(opt_hist_path, 'r') as f:
                            pso_history.extend(json.load(f))
                            
                bo_runs = list((exp_dir / 'bo').glob('run_*'))
                for bo_run in bo_runs:
                    opt_hist_path = bo_run / 'optimization_history.json'
                    if opt_hist_path.exists():
                        with open(opt_hist_path, 'r') as f:
                            bo_history.extend(json.load(f))
                
                self.all_data.append({
                    'exp_dir': exp_dir,
                    'seed': seed,
                    'summary': summary,
                    'rs_history': rs_history,
                    'pso_history': pso_history,
                    'bo_history': bo_history
                })
                
                print(f"   ✅ Loaded successfully")
                print(f"      - RS trials: {len(rs_history)}")
                print(f"      - PSO evaluations: {len(pso_history)}")
                print(f"      - BO evaluations: {len(bo_history)}")
                
            except Exception as e:
                print(f"   ❌ Error loading: {e}")
                continue
        
        print("\n" + "="*80)
        print(f"✅ Successfully loaded {len(self.all_data)} experiments")
        print("="*80)
        
        return len(self.all_data) > 0
    
    def aggregate_statistics(self):
        """Compute aggregate statistics across all runs"""
        print("\n" + "="*80)
        print("COMPUTING AGGREGATE STATISTICS")
        print("="*80)
        
        # Extract metrics
        rs_test_accs = []
        pso_test_accs = []
        bo_test_accs = []
        
        rs_val_accs = []
        pso_val_accs = []
        bo_val_accs = []
        
        rs_n_evals = []
        pso_n_evals = []
        bo_n_evals = []
        
        rs_best_configs = []
        pso_best_configs = []
        bo_best_configs = []
        
        for data in self.all_data:
            summary = data['summary']
            
            # Test accuracies
            rs_test_accs.extend(summary.get('random_search', {}).get('test_accuracies', []))
            pso_test_accs.extend(summary.get('pso', {}).get('test_accuracies', []))
            bo_test_accs.extend(summary.get('bo', {}).get('test_accuracies', []))
            
            # Validation accuracies (best found)
            if data['rs_history']:
                rs_best = max(data['rs_history'], key=lambda x: x['score'])
                rs_val_accs.append(rs_best['score'])
                rs_best_configs.append(rs_best['config'])
                rs_n_evals.append(len(data['rs_history']))
            
            if data['pso_history']:
                pso_best = max(data['pso_history'], key=lambda x: x['score'])
                pso_val_accs.append(pso_best['score'])
                pso_best_configs.append(pso_best['config'])
                pso_n_evals.append(len(data['pso_history']))
                
            if data['bo_history']:
                bo_best = max(data['bo_history'], key=lambda x: x['score'])
                bo_val_accs.append(bo_best['score'])
                bo_best_configs.append(bo_best['config'])
                bo_n_evals.append(len(data['bo_history']))
        
        # Compute statistics
        stats_dict = {
            'random_search': {
                'test_accuracy': {
                    'mean': float(np.mean(rs_test_accs)) if rs_test_accs else 0,
                    'std': float(np.std(rs_test_accs)) if rs_test_accs else 0,
                    'min': float(np.min(rs_test_accs)) if rs_test_accs else 0,
                    'max': float(np.max(rs_test_accs)) if rs_test_accs else 0,
                    'all_values': rs_test_accs
                },
                'val_accuracy': {
                    'mean': float(np.mean(rs_val_accs)) if rs_val_accs else 0,
                    'std': float(np.std(rs_val_accs)) if rs_val_accs else 0,
                    'min': float(np.min(rs_val_accs)) if rs_val_accs else 0,
                    'max': float(np.max(rs_val_accs)) if rs_val_accs else 0,
                },
                'n_evaluations': {
                    'mean': float(np.mean(rs_n_evals)) if rs_n_evals else 0,
                    'std': float(np.std(rs_n_evals)) if rs_n_evals else 0,
                },
                'n_runs': len(rs_test_accs)
            },
            'pso': {
                'test_accuracy': {
                    'mean': float(np.mean(pso_test_accs)) if pso_test_accs else 0,
                    'std': float(np.std(pso_test_accs)) if pso_test_accs else 0,
                    'min': float(np.min(pso_test_accs)) if pso_test_accs else 0,
                    'max': float(np.max(pso_test_accs)) if pso_test_accs else 0,
                    'all_values': pso_test_accs
                },
                'val_accuracy': {
                    'mean': float(np.mean(pso_val_accs)) if pso_val_accs else 0,
                    'std': float(np.std(pso_val_accs)) if pso_val_accs else 0,
                    'min': float(np.min(pso_val_accs)) if pso_val_accs else 0,
                    'max': float(np.max(pso_val_accs)) if pso_val_accs else 0,
                },
                'n_evaluations': {
                    'mean': float(np.mean(pso_n_evals)) if pso_n_evals else 0,
                    'std': float(np.std(pso_n_evals)) if pso_n_evals else 0,
                },
                'n_runs': len(pso_test_accs)
            },
            'bo': {
                'test_accuracy': {
                    'mean': float(np.mean(bo_test_accs)) if bo_test_accs else 0,
                    'std': float(np.std(bo_test_accs)) if bo_test_accs else 0,
                    'min': float(np.min(bo_test_accs)) if bo_test_accs else 0,
                    'max': float(np.max(bo_test_accs)) if bo_test_accs else 0,
                    'all_values': bo_test_accs
                },
                'val_accuracy': {
                    'mean': float(np.mean(bo_val_accs)) if bo_val_accs else 0,
                    'std': float(np.std(bo_val_accs)) if bo_val_accs else 0,
                    'min': float(np.min(bo_val_accs)) if bo_val_accs else 0,
                    'max': float(np.max(bo_val_accs)) if bo_val_accs else 0,
                },
                'n_evaluations': {
                    'mean': float(np.mean(bo_n_evals)) if bo_n_evals else 0,
                    'std': float(np.std(bo_n_evals)) if bo_n_evals else 0,
                },
                'n_runs': len(bo_test_accs)
            }
        }
        
        # Statistical tests
        if rs_test_accs and pso_test_accs and bo_test_accs:
            stats_dict['statistical_tests'] = self.perform_statistical_tests(
                rs_test_accs, pso_test_accs, bo_test_accs
            )
        
        # Average best configurations
        if rs_best_configs:
            stats_dict['random_search']['avg_best_config'] = {
                k: float(np.mean([c[k] for c in rs_best_configs]))
                for k in rs_best_configs[0].keys()
            }
        
        if pso_best_configs:
            stats_dict['pso']['avg_best_config'] = {
                k: float(np.mean([c[k] for c in pso_best_configs]))
                for k in pso_best_configs[0].keys()
            }
            
        if bo_best_configs:
            stats_dict['bo']['avg_best_config'] = {
                k: float(np.mean([c[k] for c in bo_best_configs]))
                for k in bo_best_configs[0].keys()
            }
        
        return stats_dict
    
    def perform_statistical_tests(self, rs_accs, pso_accs, bo_accs=None):
        """Perform statistical significance tests between all method pairs"""
        print("\n📊 Performing statistical tests...")
        
        tests = {}
        
        def run_pairwise(name_a, accs_a, name_b, accs_b):
            pair_tests = {}
            key = f"{name_a}_vs_{name_b}"
            print(f"\n   --- {name_a} vs {name_b} ---")
            if len(accs_a) == len(accs_b):
                try:
                    stat, p = wilcoxon(accs_a, accs_b)
                    pair_tests['wilcoxon'] = {'statistic': float(stat), 'p_value': float(p), 'significant': bool(p < 0.05)}
                    print(f"   Wilcoxon: p={p:.4f}")
                except Exception as e:
                    print(f"   ⚠️  Wilcoxon failed: {e}")
                try:
                    stat, p = ttest_rel(accs_a, accs_b)
                    pair_tests['ttest'] = {'statistic': float(stat), 'p_value': float(p), 'significant': bool(p < 0.05)}
                    print(f"   t-test (paired): p={p:.4f}")
                except Exception as e:
                    print(f"   ⚠️  t-test failed: {e}")
            else:
                try:
                    stat, p = mannwhitneyu(accs_a, accs_b)
                    pair_tests['mannwhitneyu'] = {'statistic': float(stat), 'p_value': float(p), 'significant': bool(p < 0.05)}
                    print(f"   Mann-Whitney U: p={p:.4f}")
                except Exception as e:
                    print(f"   ⚠️  Mann-Whitney failed: {e}")

            mean_diff = np.mean(accs_a) - np.mean(accs_b)
            pooled_std = np.sqrt((np.std(accs_a)**2 + np.std(accs_b)**2) / 2)
            d = mean_diff / pooled_std if pooled_std > 0 else 0
            pair_tests['effect_size'] = {'cohens_d': float(d), 'interpretation': self.interpret_cohens_d(d)}
            print(f"   Cohen's d: {d:.4f} ({pair_tests['effect_size']['interpretation']})")
            return key, pair_tests
        
        k, v = run_pairwise('random_search', rs_accs, 'pso', pso_accs)
        tests[k] = v
        if bo_accs:
            k, v = run_pairwise('bo', bo_accs, 'random_search', rs_accs)
            tests[k] = v
            k, v = run_pairwise('bo', bo_accs, 'pso', pso_accs)
            tests[k] = v
        
        return tests
    
    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def save_results(self, stats_dict, output_dir):
        """Save aggregated results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("SAVING AGGREGATED RESULTS")
        print("="*80)
        
        # Save JSON
        json_path = output_dir / 'aggregated_statistics.json'
        with open(json_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"✅ Saved: {json_path}")
        
        # Create summary table
        self.create_summary_table(stats_dict, output_dir)
        
        # Create comparison report
        self.create_comparison_report(stats_dict, output_dir)
    
    def create_summary_table(self, stats_dict, output_dir):
        """Create summary statistics table"""
        rs = stats_dict['random_search']
        pso = stats_dict['pso']
        bo = stats_dict['bo']
        
        data = {
            'Metric': [
                'Test Accuracy Mean (%)',
                'Test Accuracy Std (%)',
                'Test Accuracy Range',
                'Val Accuracy Mean (%)',
                'Val Accuracy Std (%)',
                'Avg Evaluations',
                'Number of Runs'
            ],
            'Random Search': [
                f"{rs['test_accuracy']['mean']:.2f}",
                f"{rs['test_accuracy']['std']:.2f}",
                f"{rs['test_accuracy']['min']:.2f} - {rs['test_accuracy']['max']:.2f}",
                f"{rs['val_accuracy']['mean']:.2f}",
                f"{rs['val_accuracy']['std']:.2f}",
                f"{rs['n_evaluations']['mean']:.1f} ± {rs['n_evaluations']['std']:.1f}",
                f"{rs['n_runs']}"
            ],
            'PSO': [
                f"{pso['test_accuracy']['mean']:.2f}",
                f"{pso['test_accuracy']['std']:.2f}",
                f"{pso['test_accuracy']['min']:.2f} - {pso['test_accuracy']['max']:.2f}",
                f"{pso['val_accuracy']['mean']:.2f}",
                f"{pso['val_accuracy']['std']:.2f}",
                f"{pso['n_evaluations']['mean']:.1f} ± {pso['n_evaluations']['std']:.1f}",
                f"{pso['n_runs']}"
            ],
            'Bayesian Optimization': [
                f"{bo['test_accuracy']['mean']:.2f}",
                f"{bo['test_accuracy']['std']:.2f}",
                f"{bo['test_accuracy']['min']:.2f} - {bo['test_accuracy']['max']:.2f}",
                f"{bo['val_accuracy']['mean']:.2f}",
                f"{bo['val_accuracy']['std']:.2f}",
                f"{bo['n_evaluations']['mean']:.1f} ± {bo['n_evaluations']['std']:.1f}",
                f"{bo['n_runs']}"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Save CSV
        csv_path = output_dir / 'summary_statistics.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")
        
        # Save LaTeX
        latex_path = output_dir / 'summary_statistics.tex'
        latex_table = df.to_latex(index=False, 
                                   caption='Summary Statistics Across All Runs',
                                   label='tab:summary_stats')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"✅ Saved: {latex_path}")
        
        # Print to console
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    
    def create_comparison_report(self, stats_dict, output_dir):
        """Create detailed comparison report"""
        report_path = output_dir / 'comparison_report.txt'
        
        rs = stats_dict['random_search']
        pso = stats_dict['pso']
        bo = stats_dict['bo']
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Performance comparison
            f.write("1. PERFORMANCE COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"Random Search Test Accuracy: {rs['test_accuracy']['mean']:.2f}% ± {rs['test_accuracy']['std']:.2f}%\n")
            f.write(f"PSO Test Accuracy:           {pso['test_accuracy']['mean']:.2f}% ± {pso['test_accuracy']['std']:.2f}%\n")
            f.write(f"BO Test Accuracy:            {bo['test_accuracy']['mean']:.2f}% ± {bo['test_accuracy']['std']:.2f}%\n")
            
            diff_rs_pso = rs['test_accuracy']['mean'] - pso['test_accuracy']['mean']
            diff_bo_rs = bo['test_accuracy']['mean'] - rs['test_accuracy']['mean']
            diff_bo_pso = bo['test_accuracy']['mean'] - pso['test_accuracy']['mean']
            
            f.write(f"\nDifferences:\n")
            f.write(f"RS - PSO: {diff_rs_pso:+.2f}%\n")
            f.write(f"BO - RS:  {diff_bo_rs:+.2f}%\n")
            f.write(f"BO - PSO: {diff_bo_pso:+.2f}%\n")
            
            best_val = max(rs['test_accuracy']['mean'], pso['test_accuracy']['mean'], bo['test_accuracy']['mean'])
            if best_val == bo['test_accuracy']['mean']:
                f.write("\nWinner: Bayesian Optimization\n")
            elif best_val == rs['test_accuracy']['mean']:
                f.write("\nWinner: Random Search\n")
            else:
                f.write("\nWinner: PSO\n")
            
            # Statistical significance
            if 'statistical_tests' in stats_dict:
                f.write("\n2. STATISTICAL SIGNIFICANCE\n")
                f.write("-" * 80 + "\n")
                
                tests = stats_dict['statistical_tests']
                
                if 'wilcoxon' in tests:
                    f.write(f"Wilcoxon Test: p = {tests['wilcoxon']['p_value']:.4f}")
                    if tests['wilcoxon']['significant']:
                        f.write(" (SIGNIFICANT at α=0.05)\n")
                    else:
                        f.write(" (NOT significant)\n")
                
                if 'mannwhitneyu' in tests:
                    f.write(f"Mann-Whitney U: p = {tests['mannwhitneyu']['p_value']:.4f}")
                    if tests['mannwhitneyu']['significant']:
                        f.write(" (SIGNIFICANT at α=0.05)\n")
                    else:
                        f.write(" (NOT significant)\n")
                
                if 'ttest_paired' in tests:
                    f.write(f"Paired t-test: p = {tests['ttest_paired']['p_value']:.4f}")
                    if tests['ttest_paired']['significant']:
                        f.write(" (SIGNIFICANT at α=0.05)\n")
                    else:
                        f.write(" (NOT significant)\n")
                
                if 'effect_size' in tests:
                    f.write(f"\nEffect Size (Cohen's d): {tests['effect_size']['cohens_d']:.4f}")
                    f.write(f" ({tests['effect_size']['interpretation']})\n")
            
            # Computational efficiency
            f.write("\n3. COMPUTATIONAL EFFICIENCY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Random Search Avg Evaluations: {rs['n_evaluations']['mean']:.1f} ± {rs['n_evaluations']['std']:.1f}\n")
            f.write(f"PSO Avg Evaluations:           {pso['n_evaluations']['mean']:.1f} ± {pso['n_evaluations']['std']:.1f}\n")
            f.write(f"BO Avg Evaluations:            {bo['n_evaluations']['mean']:.1f} ± {bo['n_evaluations']['std']:.1f}\n")
            
            # Best configurations
            if 'avg_best_config' in rs:
                f.write("\n4. AVERAGE BEST CONFIGURATIONS\n")
                f.write("-" * 80 + "\n")
                f.write("\nRandom Search:\n")
                for k, v in rs['avg_best_config'].items():
                    f.write(f"  {k}: {v:.6f}\n")
                
                f.write("\nPSO:\n")
                for k, v in pso['avg_best_config'].items():
                    f.write(f"  {k}: {v:.6f}\n")
                    
                f.write("\nBayesian Optimization:\n")
                for k, v in bo['avg_best_config'].items():
                    f.write(f"  {k}: {v:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✅ Saved: {report_path}")
        
        # Print report
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    def run(self, output_dir='multi_run_results'):
        """Run complete aggregation pipeline"""
        # Find experiments
        if not self.find_experiment_directories():
            return False
        
        # Load data
        if not self.load_all_data():
            return False
        
        # Aggregate statistics
        stats_dict = self.aggregate_statistics()
        
        # Save results
        self.save_results(stats_dict, output_dir)
        
        print("\n" + "="*80)
        print("✅ AGGREGATION COMPLETE!")
        print("="*80)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-run results')
    parser.add_argument('--results_dir', type=str, default='multi_run_results',
                       help='Directory to save aggregated results')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MULTI-RUN RESULTS AGGREGATION & STATISTICAL ANALYSIS")
    print("="*80)
    
    aggregator = MultiRunAggregator(args.results_dir)
    success = aggregator.run(args.results_dir)
    
    if not success:
        print("\n❌ Aggregation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())