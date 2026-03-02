"""
Statistical analysis module for Q2 HPO publication results.

Usage:
    python experiments/statistical_analysis.py --results_dir q2_results
"""
import argparse
import json
import os

import numpy as np
from scipy import stats


def compute_ci(accuracies, confidence=0.95):
    """
    Compute confidence interval for a list of accuracy values.

    Args:
        accuracies (list): List of float accuracy values.
        confidence (float): Confidence level (default 0.95).

    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    n = len(accuracies)
    mean = np.mean(accuracies)
    se = stats.sem(accuracies)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean), float(mean - h), float(mean + h)


def wilcoxon_test(accs_a, accs_b, name_a, name_b):
    """
    Perform Wilcoxon signed-rank test between two paired samples.

    Args:
        accs_a (list): Accuracies for method A.
        accs_b (list): Accuracies for method B.
        name_a (str): Name of method A.
        name_b (str): Name of method B.

    Returns:
        dict: Test statistic, p-value, and significance flag.
    """
    w_stat, p_value = stats.wilcoxon(accs_a, accs_b)
    result = {
        'comparison': f"{name_a} vs {name_b}",
        'statistic': float(w_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }
    return result


def cohens_d(accs_a, accs_b):
    """
    Compute Cohen's d effect size between two samples.

    Args:
        accs_a (list): Accuracies for method A.
        accs_b (list): Accuracies for method B.

    Returns:
        float: Cohen's d effect size.
    """
    a = np.array(accs_a)
    b = np.array(accs_b)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def friedman_test(all_accs):
    """
    Perform Friedman test across all methods (use when all 4 methods are available).

    Args:
        all_accs (dict): Dictionary mapping method name -> list of accuracies.

    Returns:
        dict: Friedman test statistic, p-value, and significance flag.
    """
    groups = list(all_accs.values())
    stat, p_value = stats.friedmanchisquare(*groups)
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }


def run_analysis(results_dir):
    """
    Main entry point for statistical analysis.

    Loads RS and Baseline results, computes statistics, and saves reports.

    Args:
        results_dir (str): Path to q2_results directory.
    """
    rs_summary_path = os.path.join(results_dir, 'random_search', 'rs_summary.json')
    baseline_summary_path = os.path.join(results_dir, 'baseline', 'summary.json')

    if not os.path.exists(rs_summary_path):
        print(f"RS summary not found at {rs_summary_path}. Run random search first.")
        return

    with open(rs_summary_path) as f:
        rs_summary = json.load(f)
    rs_accs = rs_summary['test_accuracies']

    baseline_accs = None
    if os.path.exists(baseline_summary_path):
        with open(baseline_summary_path) as f:
            baseline_summary = json.load(f)
        baseline_accs = baseline_summary['test_accuracies']

    report_lines = []
    report_data = {}

    report_lines.append("=" * 60)
    report_lines.append("Q2 STATISTICAL ANALYSIS REPORT")
    report_lines.append("=" * 60)

    # Random Search stats
    rs_mean, rs_ci_lo, rs_ci_hi = compute_ci(rs_accs)
    rs_std = float(np.std(rs_accs, ddof=1))
    report_lines.append(f"\nRandom Search (n={len(rs_accs)}):")
    report_lines.append(f"  Mean ± Std : {rs_mean:.4f}% ± {rs_std:.4f}%")
    report_lines.append(f"  95% CI     : [{rs_ci_lo:.4f}%, {rs_ci_hi:.4f}%]")
    report_data['random_search'] = {
        'mean': rs_mean, 'std': rs_std,
        'ci_95_lower': rs_ci_lo, 'ci_95_upper': rs_ci_hi,
        'accuracies': rs_accs
    }

    # Baseline stats and comparison
    if baseline_accs is not None:
        bl_mean, bl_ci_lo, bl_ci_hi = compute_ci(baseline_accs)
        bl_std = float(np.std(baseline_accs, ddof=1))
        report_lines.append(f"\nBaseline (n={len(baseline_accs)}):")
        report_lines.append(f"  Mean ± Std : {bl_mean:.4f}% ± {bl_std:.4f}%")
        report_lines.append(f"  95% CI     : [{bl_ci_lo:.4f}%, {bl_ci_hi:.4f}%]")
        report_data['baseline'] = {
            'mean': bl_mean, 'std': bl_std,
            'ci_95_lower': bl_ci_lo, 'ci_95_upper': bl_ci_hi,
            'accuracies': baseline_accs
        }

        report_lines.append("\nRS vs Baseline:")
        if len(rs_accs) == len(baseline_accs) and len(rs_accs) >= 3:
            wx = wilcoxon_test(rs_accs, baseline_accs, 'RS', 'Baseline')
            d = cohens_d(rs_accs, baseline_accs)
            sig_str = "significant" if wx['significant'] else "not significant"
            report_lines.append(f"  Wilcoxon   : W={wx['statistic']:.4f}, p={wx['p_value']:.4f} ({sig_str})")
            report_lines.append(f"  Cohen's d  : {d:.4f}")
            report_data['rs_vs_baseline'] = {**wx, 'cohens_d': d}
        else:
            report_lines.append("  (Paired Wilcoxon requires equal-length samples with n>=3)")

    report_lines.append("\n" + "=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, 'statistical_report.txt')
    json_path = os.path.join(results_dir, 'statistical_report.json')

    with open(txt_path, 'w') as f:
        f.write(report_text + "\n")

    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\nReports saved to {txt_path} and {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical analysis for Q2 HPO results')
    parser.add_argument('--results_dir', type=str, default='q2_results',
                        help='Path to q2_results directory')
    args = parser.parse_args()
    run_analysis(args.results_dir)
