# Phase 2: Multi-Run Validation - Quick Start Guide

## Overview
This guide helps you run multiple experiments (n=3) with computational cost tracking for statistical significance.

## Prerequisites
```bash
cd /home/muhammad-noman/noman/cifar10-hpo
conda activate cifar10-hpo
```

## Option 1: Run 3 Experiments Automatically (RECOMMENDED)

### Command:
```bash
python run_multiple_experiments.py --n_runs 3 --config experiments/config.yaml
```

### What it does:
- Runs 3 experiments with seeds: 42, 123, 999
- Tracks wall-clock time and GPU hours
- Saves computational cost data
- Takes ~2-3 hours total

### Custom seeds (optional):
```bash
python run_multiple_experiments.py --n_runs 3 --seeds 42 100 200
```

---

## Option 2: Run Experiments Manually (with cost tracking)

### Terminal 1 - Start Cost Monitor:
```bash
python track_computational_cost.py --output run1_cost.json --monitor_interval 60
```

### Terminal 2 - Run Experiments:
```bash
# Run 1 (seed=42)
cd experiments
python run_experiment.py --config config.yaml

# Run 2 (seed=123) - modify config first
python run_experiment.py --config config.yaml

# Run 3 (seed=999) - modify config first
python run_experiment.py --config config.yaml
```

### Stop monitor (Terminal 1):
Press Ctrl+C

---

## Step 2: Aggregate Results

After all 3 runs complete:

```bash
python aggregate_results.py --results_dir experiments/results --output_dir multi_run_analysis
```

### Outputs:
- `multi_run_summary.json` - Statistical summary
- `multi_run_statistics.csv` - Mean, std, CI for thesis table
- `multi_run_comparison_with_error_bars.png` - Publication plot
- `multi_run_detailed_results.csv` - All raw data

---

## Step 3: Generate Publication Plots with Error Bars

```bash
# Using your existing plot script (if you have one)
python generate_plots.py experiments/results/<latest_dir>

# Or use the aggregated results
python generate_publication_plots.py --results_dir multi_run_analysis
```

---

## What You Get for Thesis

### 1. Statistical Validity
✅ Mean ± std for both methods
✅ 95% confidence intervals
✅ Statistical significance tests (p-values)
✅ Effect sizes (Cohen's d)

### 2. Computational Cost Documentation
✅ Total wall-clock time (hours/days)
✅ GPU hours used
✅ Time per run
✅ Cost comparison between methods

### 3. Publication-Ready Figures
✅ Comparison plot with error bars
✅ Individual run data points
✅ Statistical annotations

### 4. Thesis Tables
Use `multi_run_statistics.csv`:

| Method | Mean Test Acc (%) | Std Dev | 95% CI | n |
|--------|-------------------|---------|--------|---|
| Random Search | 90.58 ± 0.XX | 0.XX | [XX, XX] | 3 |
| PSO | 90.52 ± 0.XX | 0.XX | [XX, XX] | 3 |

---

## Time Estimates

| Task | Time |
|------|------|
| Single run (fast config) | ~30-45 min |
| 3 runs total | ~1.5-2.5 hours |
| Aggregation & plots | ~5 minutes |
| **Total** | **~2-3 hours** |

---

## Troubleshooting

### Error: "Cannot find summary.json"
**Solution:** Wait for experiments to complete fully

### Error: "Not enough results"
**Solution:** Need at least 2 completed runs for statistics

### GPU not detected
**Solution:** `track_computational_cost.py` will still track time, just not GPU utilization

---

## Next Steps After Phase 2

After completing 3 runs:

1. ✅ **Add to thesis Chapter 5 (Results)**:
   - Table with mean ± std
   - Figure with error bars
   - Statistical significance statement

2. ⚠️ **Identify remaining gaps**:
   - Bayesian optimization comparison?
   - Additional datasets (MNIST, Fashion-MNIST)?
   - Different architectures?

3. 📝 **Document in thesis**:
   - Computational cost section
   - Statistical methodology
   - Discussion of significance

---

## Questions?

- Issues with scripts? Check error messages
- Need more runs? Change `--n_runs 3` to `--n_runs 5`
- Want different seeds? Use `--seeds 42 123 999 555 777`
