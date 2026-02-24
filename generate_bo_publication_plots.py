"""
Complete Publication-Ready Plots Generator — RS vs PSO vs Bayesian Optimization
Generates all 10 figures needed for thesis/publication
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon, pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'text.usetex': False
})

# Three distinct colors for the three methods
COLORS = {'Random Search': '#3498db', 'PSO': '#e74c3c', 'Bayesian Opt.': '#2ecc71'}
MARKERS = {'Random Search': 'o', 'PSO': 's', 'Bayesian Opt.': 'D'}

OUTPUT_DIR = Path('publication_plots_bo')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
BASE = Path('experiments/results')
RUNS = [
    BASE / 'CIFAR10_CNN_20260127_234026',
    BASE / 'CIFAR10_CNN_20260128_055854',
    BASE / 'CIFAR10_CNN_20260128_115445',
]
SEEDS = [42, 123, 999]

all_data = []
for idx, run_dir in enumerate(RUNS):
    seed = SEEDS[idx]
    with open(run_dir / 'summary.json') as f:
        summary = json.load(f)

    # RS history
    rs_history = []
    for rs_run in sorted((run_dir / 'random_search').glob('run_*')):
        p = rs_run / 'optimization_history.json'
        if p.exists():
            rs_history.extend(json.load(open(p)))

    # PSO history
    pso_history = []
    for pso_run in sorted((run_dir / 'pso').glob('run_*')):
        p = pso_run / 'optimization_history.json'
        if p.exists():
            pso_history.extend(json.load(open(p)))

    # BO history
    bo_history = []
    if (run_dir / 'bo').exists():
        for bo_run in sorted((run_dir / 'bo').glob('run_*')):
            p = bo_run / 'optimization_history.json'
            if p.exists():
                bo_history.extend(json.load(open(p)))

    # Training histories (best model)
    def load_training(subdir):
        runs = sorted((run_dir / subdir).glob('run_*'))
        if not runs:
            return None
        p = runs[0] / 'training_history.json'
        return json.load(open(p)) if p.exists() else None

    all_data.append({
        'seed': seed,
        'summary': summary,
        'rs_history': rs_history,
        'pso_history': pso_history,
        'bo_history': bo_history,
        'rs_training': load_training('random_search'),
        'pso_training': load_training('pso'),
        'bo_training': load_training('bo'),
    })

print(f"Loaded {len(all_data)} experiment runs.")

# Aggregate test accuracies
rs_test  = [acc for d in all_data for acc in d['summary'].get('random_search', {}).get('test_accuracies', [])]
pso_test = [acc for d in all_data for acc in d['summary'].get('pso', {}).get('test_accuracies', [])]
bo_test  = [acc for d in all_data for acc in d['summary'].get('bo', {}).get('test_accuracies', [])]


def save(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png')
    plt.savefig(OUTPUT_DIR / f'{name}.pdf')
    plt.close()
    print(f'  ✅ Saved {name}.png/pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Test Accuracy Comparison (boxplot)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n📊 Fig 1: Test Accuracy Comparison')
fig, ax = plt.subplots(figsize=(10, 6))
data_groups = [rs_test, pso_test, bo_test]
labels = ['Random Search', 'PSO', 'Bayesian Opt.']
bp = ax.boxplot(data_groups, positions=[1,2,3], widths=0.55, patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='gold', markersize=10, markeredgecolor='black'))
for patch, lbl in zip(bp['boxes'], labels):
    patch.set_facecolor(COLORS[lbl])
    patch.set_alpha(0.75)
for i, (vals, lbl) in enumerate(zip(data_groups, labels), 1):
    jitter = np.random.normal(i, 0.05, size=len(vals))
    ax.scatter(jitter, vals, color='black', s=60, zorder=5, edgecolors='white', linewidth=1.2)
    mean, std = np.mean(vals), np.std(vals)
    ax.text(i, max(vals)+0.15, f'{mean:.2f}±{std:.2f}%',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=COLORS[lbl], alpha=0.35))

ax.set_xticks([1,2,3])
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('CIFAR-10 Test Accuracy Comparison\n(3 Independent Runs per Method)', fontsize=16, fontweight='bold')
ax.set_ylim(min(min(rs_test), min(pso_test), min(bo_test))-1,
            max(max(rs_test), max(pso_test), max(bo_test))+1.5)
ax.grid(axis='y', alpha=0.35, linestyle='--')
plt.tight_layout()
save('fig1_test_accuracy_comparison')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Convergence Curves (Best-so-far per run)
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 2: Convergence Curves')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for idx, data in enumerate(all_data):
    ax = axes[idx]
    seed = data['seed']

    def plot_convergence(history, label):
        scores = [h['score'] for h in history]
        best_so_far = np.maximum.accumulate(scores)
        evals = list(range(1, len(scores)+1))
        ax.plot(evals, best_so_far, marker=MARKERS[label], linewidth=2.5,
                markersize=6, label=label, color=COLORS[label], markevery=max(1,len(evals)//10))

    plot_convergence(data['rs_history'],  'Random Search')
    plot_convergence(data['pso_history'], 'PSO')
    plot_convergence(data['bo_history'],  'Bayesian Opt.')

    ax.set_xlabel('Evaluation Number', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'Run {idx+1} (Seed={seed})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([65, 95])

plt.suptitle('Best-So-Far Convergence Curves — All Runs', fontsize=17, fontweight='bold')
plt.tight_layout()
save('fig2_convergence_curves')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Training Curves (best retrained model)
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 3: Training Curves')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, data in enumerate(all_data):
    seed = data['seed']
    for row, (key, lbl) in enumerate([('rs_training', 'Random Search'), ('bo_training', 'Bayesian Opt.')]):
        ax = axes[row, idx]
        hist = data[key]
        if hist is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{lbl} (Seed={seed})', fontsize=12, fontweight='bold')
            continue
        epochs = range(1, len(hist['train_loss'])+1)
        ax2 = ax.twinx()
        ax.plot(epochs, hist['train_loss'], 'b-', lw=2, label='Train Loss', alpha=0.7)
        ax.plot(epochs, hist['val_loss'],   'r-', lw=2, label='Val Loss',   alpha=0.7)
        ax2.plot(epochs, hist['train_acc'], 'b--', lw=2, label='Train Acc', alpha=0.7)
        ax2.plot(epochs, hist['val_acc'],   'r--', lw=2, label='Val Acc',   alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{lbl} (Seed={seed})', fontsize=12, fontweight='bold')
        lns = ax.get_lines() + ax2.get_lines()
        ax.legend(lns, [l.get_label() for l in lns], loc='center right', fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle('Training Curves — RS and BO Best Models', fontsize=17, fontweight='bold')
plt.tight_layout()
save('fig3_training_curves')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Hyperparameter Importance (Pearson r / combined all methods)
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 4: Hyperparameter Importance')
all_configs, all_scores = [], []
for data in all_data:
    for h in data['rs_history'] + data['pso_history'] + data['bo_history']:
        all_configs.append(h['config'])
        all_scores.append(h['score'])

df_all = pd.DataFrame(all_configs)
df_all['score'] = all_scores
hp_cols = [c for c in df_all.columns if c != 'score']

correlations, pvals = {}, {}
for col in hp_cols:
    r, p = pearsonr(df_all[col], df_all['score'])
    correlations[col] = r
    pvals[col] = p

sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
fig, ax = plt.subplots(figsize=(10, 7))
params = [k.replace('_', '\n') for k, _ in sorted_params]
values = [v for _, v in sorted_params]
colors_bar = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
bars = ax.barh(params, values, color=colors_bar, alpha=0.75, edgecolor='black', linewidth=1.2)
for bar, (k, v), p in zip(bars, sorted_params, [pvals[k] for k, _ in sorted_params]):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    xpos = bar.get_width()
    ax.text(xpos + (0.02 if xpos >= 0 else -0.02),
            bar.get_y() + bar.get_height()/2,
            f'{xpos:.3f} {sig}', va='center',
            ha='left' if xpos >= 0 else 'right', fontsize=9, fontweight='bold')

ax.axvline(0, color='black', linewidth=2)
ax.set_xlabel('Pearson Correlation with Validation Accuracy', fontsize=13)
ax.set_title('Hyperparameter Importance\n(All methods & runs combined)', fontsize=15, fontweight='bold')
ax.set_xlim([-1, 1.2])
ax.text(0.02, 0.02, '*** p<0.001  ** p<0.01  * p<0.05  n.s. not significant',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
save('fig4_hyperparameter_importance')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Hyperparameter Distributions vs Accuracy (scatter per method)
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 5: Hyperparameter Distributions')
method_data = {
    'Random Search': [(h['config'], h['score']) for d in all_data for h in d['rs_history']],
    'PSO':           [(h['config'], h['score']) for d in all_data for h in d['pso_history']],
    'Bayesian Opt.': [(h['config'], h['score']) for d in all_data for h in d['bo_history']],
}

hyperparams = ['learning_rate', 'batch_size', 'conv_channels_base',
               'num_conv_blocks', 'fc_hidden', 'dropout', 'weight_decay']

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
for idx, hp in enumerate(hyperparams):
    ax = axes[idx]
    for lbl, pairs in method_data.items():
        xs = [p[0][hp] for p in pairs]
        ys = [p[1]     for p in pairs]
        ax.scatter(xs, ys, c=COLORS[lbl], alpha=0.5, s=40, label=lbl,
                   marker=MARKERS[lbl], edgecolors='none')
    all_xs = [p[0][hp] for pairs in method_data.values() for p in pairs]
    all_ys = [p[1]     for pairs in method_data.values() for p in pairs]
    slope, intercept, r, *_ = linregress(all_xs, all_ys)
    xline = np.linspace(min(all_xs), max(all_xs), 100)
    ax.plot(xline, slope * xline + intercept, 'k--', lw=2, alpha=0.5, label=f'r={r:.2f}')
    ax.set_xlabel(hp.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Val Accuracy (%)', fontsize=11)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    if hp in ['learning_rate', 'weight_decay']:
        ax.set_xscale('log')

axes[-1].axis('off')
plt.suptitle('Hyperparameter vs Performance — All Methods Combined',
             fontsize=16, fontweight='bold')
plt.tight_layout()
save('fig5_hyperparameter_distributions')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 6: Correlation Heatmap')
fig, ax = plt.subplots(figsize=(11, 9))
corr = df_all.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.8,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'}, vmin=-1, vmax=1)
ax.set_title('Hyperparameter Correlation Matrix\n(All methods combined)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save('fig6_correlation_heatmap')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Computational Cost Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 7: Computational Cost')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
seeds_arr = [d['seed'] for d in all_data]
x = np.arange(len(seeds_arr))
width = 0.25

rs_evals  = [len(d['rs_history'])  for d in all_data]
pso_evals = [len(d['pso_history']) for d in all_data]
bo_evals  = [len(d['bo_history'])  for d in all_data]

# Sub-plot 1: Evaluations
ax1 = axes[0, 0]
ax1.bar(x - width, rs_evals,  width, label='RS',  color=COLORS['Random Search'],  alpha=0.8)
ax1.bar(x,         pso_evals, width, label='PSO', color=COLORS['PSO'],  alpha=0.8)
ax1.bar(x + width, bo_evals,  width, label='BO',  color=COLORS['Bayesian Opt.'], alpha=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(seeds_arr)
ax1.set(xlabel='Seed', ylabel='# Model Evaluations', title='Total Evaluations per Run')
ax1.legend(); ax1.grid(axis='y', alpha=0.3)

# Sub-plot 2: Estimated time (epochs ∝ evals)
ax2 = axes[0, 1]
rs_time  = [e * 30  / 60 for e in rs_evals]
pso_time = [e * 25  / 60 for e in pso_evals]
bo_time  = [e * 30  / 60 for e in bo_evals]
ax2.bar(x - width, rs_time,  width, label='RS',  color=COLORS['Random Search'],  alpha=0.8)
ax2.bar(x,         pso_time, width, label='PSO', color=COLORS['PSO'],  alpha=0.8)
ax2.bar(x + width, bo_time,  width, label='BO',  color=COLORS['Bayesian Opt.'], alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(seeds_arr)
ax2.set(xlabel='Seed', ylabel='Estimated Time (hours)', title='Estimated Compute Time')
ax2.legend(); ax2.grid(axis='y', alpha=0.3)

# Sub-plot 3: Best val acc / evaluations (efficiency)
ax3 = axes[1, 0]
for lbl, hist_key, eval_arr in [
    ('Random Search', 'rs_history', rs_evals),
    ('PSO',           'pso_history', pso_evals),
    ('Bayesian Opt.', 'bo_history', bo_evals),
]:
    eff = [max(h['score'] for h in d[hist_key])/n_e
           for d, n_e in zip(all_data, eval_arr)]
    ax3.plot(seeds_arr, eff, marker=MARKERS[lbl], linewidth=2.5,
             markersize=10, label=lbl, color=COLORS[lbl])
ax3.set(xlabel='Seed', ylabel='Best Val Acc / Evaluations', title='Optimization Efficiency')
ax3.legend(); ax3.grid(True, alpha=0.3)

# Sub-plot 4: Evaluations to reach 85 %
ax4 = axes[1, 1]
threshold = 85
for offset, (lbl, hist_key) in enumerate(
        [('Random Search', 'rs_history'), ('PSO', 'pso_history'), ('Bayesian Opt.', 'bo_history')]):
    conv_speed = []
    for d in all_data:
        scores = [h['score'] for h in d[hist_key]]
        conv_speed.append(next((i+1 for i, s in enumerate(scores) if s >= threshold), len(scores)))
    ax4.bar(x + (offset-1)*width, conv_speed, width, label=lbl, color=COLORS[lbl], alpha=0.8)
ax4.set_xticks(x); ax4.set_xticklabels(seeds_arr)
ax4.set(xlabel='Seed', ylabel=f'Evals to reach {threshold}%', title='Convergence Speed')
ax4.legend(); ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Computational Cost Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
save('fig7_computational_cost')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Radar Chart: Best Config Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 8: Radar Chart')
param_ranges = {
    'learning_rate': (0.0001, 0.01), 'batch_size': (32, 256),
    'conv_channels_base': (32, 128), 'num_conv_blocks': (2, 4),
    'fc_hidden': (64, 512), 'dropout': (0.1, 0.5), 'weight_decay': (0.00001, 0.001)
}
log_params = {'learning_rate', 'weight_decay'}

def normalize_config(cfg):
    norm = {}
    for key, (mn, mx) in param_ranges.items():
        v = cfg[key]
        if key in log_params:
            v, mn, mx = np.log10(v), np.log10(mn), np.log10(mx)
        norm[key] = np.clip((v - mn) / (mx - mn), 0, 1)
    return norm

def avg_best_config(hist_key):
    bests = [max(d[hist_key], key=lambda h: h['score'])['config'] for d in all_data if d[hist_key]]
    keys = list(param_ranges.keys())
    return {k: np.mean([b[k] for b in bests]) for k in keys}

categories = list(param_ranges.keys())
N = len(categories)
angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for lbl, hist_key in [('Random Search', 'rs_history'), ('PSO', 'pso_history'), ('Bayesian Opt.', 'bo_history')]:
    cfg_norm = normalize_config(avg_best_config(hist_key))
    vals = [cfg_norm[c] for c in categories] + [cfg_norm[categories[0]]]
    ax.plot(angles, vals, marker=MARKERS[lbl], linewidth=3, label=lbl, color=COLORS[lbl], markersize=8)
    ax.fill(angles, vals, alpha=0.15, color=COLORS[lbl])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([c.replace('_', '\n') for c in categories], size=10)
ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12)
ax.set_title('Average Best Config Comparison\n(Normalized, averaged over 3 runs)',
             fontsize=15, fontweight='bold', pad=30)
plt.tight_layout()
save('fig8_best_config_radar')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Average Convergence Speed (best-so-far mean ± std across runs)
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 9: Average Convergence Speed')
fig, ax = plt.subplots(figsize=(12, 6))

for lbl, hist_key in [('Random Search', 'rs_history'), ('PSO', 'pso_history'), ('Bayesian Opt.', 'bo_history')]:
    max_evals = max(len(d[hist_key]) for d in all_data if d[hist_key])
    matrix = []
    for d in all_data:
        if not d[hist_key]:
            continue
        scores = [h['score'] for h in d[hist_key]]
        bsf = list(np.maximum.accumulate(scores))
        # Pad to max_evals with last value
        bsf += [bsf[-1]] * (max_evals - len(bsf))
        matrix.append(bsf)
    matrix = np.array(matrix)
    mean = matrix.mean(axis=0)
    std  = matrix.std(axis=0)
    x_evals = np.arange(1, max_evals + 1)
    ax.plot(x_evals, mean, linewidth=2.5, label=lbl, color=COLORS[lbl], marker=MARKERS[lbl], markevery=max(1, max_evals//10))
    ax.fill_between(x_evals, mean - std, mean + std, alpha=0.18, color=COLORS[lbl])

ax.set_xlabel('Evaluation Number', fontsize=14)
ax.set_ylabel('Best Validation Accuracy (%)', fontsize=14)
ax.set_title('Average Convergence Speed (mean ± std across 3 runs)',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim([65, 95])
plt.tight_layout()
save('fig9_convergence_speed')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Statistical Significance Summary Panel
# ═══════════════════════════════════════════════════════════════════════════════
print('📊 Fig 10: Statistical Significance')
from scipy.stats import ttest_rel

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

pairs = [
    ('RS vs PSO',        rs_test, pso_test, 'Random Search', 'PSO'),
    ('BO vs RS',         bo_test, rs_test,  'Bayesian Opt.', 'Random Search'),
    ('BO vs PSO',        bo_test, pso_test, 'Bayesian Opt.', 'PSO'),
]

for ax, (title, accs_a, accs_b, lbl_a, lbl_b) in zip(axes, pairs):
    # Paired t-test
    try:
        stat, p_val = ttest_rel(accs_a, accs_b)
    except Exception:
        stat, p_val = 0, 1.0

    x = [1, 2]
    bp = ax.boxplot([accs_a, accs_b], positions=x, widths=0.5,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='gold', markersize=9, markeredgecolor='black'))
    for patch, lbl in zip(bp['boxes'], [lbl_a, lbl_b]):
        patch.set_facecolor(COLORS[lbl])
        patch.set_alpha(0.75)

    sig_txt = f'p = {p_val:.4f}' + (' ✓ Sig.' if p_val < 0.05 else ' n.s.')
    y_top = max(max(accs_a), max(accs_b)) + 0.4
    ax.plot([1, 1, 2, 2], [y_top, y_top+0.1, y_top+0.1, y_top], 'k-', lw=1.5)
    ax.text(1.5, y_top + 0.15, sig_txt, ha='center', fontsize=11, fontweight='bold')

    diff = np.mean(accs_a) - np.mean(accs_b)
    pooled = np.sqrt((np.std(accs_a)**2 + np.std(accs_b)**2) / 2)
    d = diff / pooled if pooled > 0 else 0
    ax.set_title(f'{title}\n(t-test paired  •  Cohen\'s d={d:.2f})', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels([lbl_a, lbl_b], fontsize=12)
    ax.set_ylabel('Test Accuracy (%)' if title.startswith('RS') else '', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Statistical Significance Tests — Pairwise Comparisons', fontsize=16, fontweight='bold')
plt.tight_layout()
save('fig10_statistical_significance')

# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX Table
# ═══════════════════════════════════════════════════════════════════════════════
print('\n📊 Generating LaTeX summary table ...')
rows = {
    'Metric': ['Mean Test Acc (%)', 'Std Test Acc (%)', 'Best Test Acc (%)',
               'Mean Val Acc (%)', '# Evaluations', 'Runs'],
    'Random Search': [
        f'{np.mean(rs_test):.2f}', f'{np.std(rs_test):.2f}', f'{max(rs_test):.2f}',
        f'{np.mean([max(h["score"] for h in d["rs_history"]) for d in all_data]):.2f}',
        f'{np.mean([len(d["rs_history"]) for d in all_data]):.0f}', str(len(rs_test))
    ],
    'PSO': [
        f'{np.mean(pso_test):.2f}', f'{np.std(pso_test):.2f}', f'{max(pso_test):.2f}',
        f'{np.mean([max(h["score"] for h in d["pso_history"]) for d in all_data]):.2f}',
        f'{np.mean([len(d["pso_history"]) for d in all_data]):.0f}', str(len(pso_test))
    ],
    'Bayesian Opt.': [
        f'{np.mean(bo_test):.2f}', f'{np.std(bo_test):.2f}', f'{max(bo_test):.2f}',
        f'{np.mean([max(h["score"] for h in d["bo_history"]) for d in all_data if d["bo_history"]]):.2f}',
        f'{np.mean([len(d["bo_history"]) for d in all_data]):.0f}', str(len(bo_test))
    ],
}
df_tbl = pd.DataFrame(rows)
latex = df_tbl.to_latex(index=False, caption='HPO Method Comparison on CIFAR-10',
                         label='tab:hpo_comparison')
(OUTPUT_DIR / 'summary_table.tex').write_text(latex)
df_tbl.to_csv(OUTPUT_DIR / 'summary_table.csv', index=False)
print('  ✅ Saved summary_table.tex and summary_table.csv')

print(f'\n{"="*60}')
print(f'✅ ALL PLOTS SAVED TO: {OUTPUT_DIR}/')
print(f'{"="*60}')
print(df_tbl.to_string(index=False))
