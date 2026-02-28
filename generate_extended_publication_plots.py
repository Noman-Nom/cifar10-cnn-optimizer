"""
Extended Publication Plots — Q1/Q2 Journal Quality
Generates 7 additional figures for RS vs PSO vs Bayesian Optimization comparison
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from models.cnn import create_cnn_from_config

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── Style ──────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 15,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'text.usetex': False,
})

COLORS = {'Random Search': '#3498db', 'PSO': '#e74c3c', 'Bayesian Opt.': '#2ecc71'}
MARKERS = {'Random Search': 'o', 'PSO': 's', 'Bayesian Opt.': 'D'}
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

OUTPUT_DIR = Path('publication_plots_bo_extended')
OUTPUT_DIR.mkdir(exist_ok=True)

BASE = Path('experiments/results')
RUNS = [
    BASE / 'CIFAR10_CNN_20260127_234026',
    BASE / 'CIFAR10_CNN_20260128_055854',
    BASE / 'CIFAR10_CNN_20260128_115445',
]
SEEDS = [42, 123, 999]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ─── Load all data ─────────────────────────────────────────────────────────────
all_data = []
for idx, run_dir in enumerate(RUNS):
    with open(run_dir / 'summary.json') as f:
        summary = json.load(f)

    def load_history(subdir):
        h = []
        d = run_dir / subdir
        if not d.exists():
            return h
        for run in sorted(d.glob('run_*')):
            p = run / 'optimization_history.json'
            if p.exists():
                h.extend(json.load(open(p)))
        return h

    def load_training(subdir):
        d = run_dir / subdir
        if not d.exists():
            return None
        runs = sorted(d.glob('run_*'))
        if not runs:
            return None
        p = runs[0] / 'training_history.json'
        return json.load(open(p)) if p.exists() else None

    def load_best_model_path(subdir):
        d = run_dir / subdir
        if not d.exists():
            return None
        runs = sorted(d.glob('run_*'))
        if not runs:
            return None
        p = runs[0] / 'best_model.pth'
        return p if p.exists() else None

    all_data.append({
        'seed': SEEDS[idx],
        'summary': summary,
        'rs_history':  load_history('random_search'),
        'pso_history': load_history('pso'),
        'bo_history':  load_history('bo'),
        'rs_training':  load_training('random_search'),
        'pso_training': load_training('pso'),
        'bo_training':  load_training('bo'),
        'rs_model_path':  load_best_model_path('random_search'),
        'pso_model_path': load_best_model_path('pso'),
        'bo_model_path':  load_best_model_path('bo'),
        'run_dir': run_dir,
    })

print(f"Loaded {len(all_data)} runs.")


def save(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png')
    plt.savefig(OUTPUT_DIR / f'{name}.pdf')
    plt.close()
    print(f'  ✅ Saved {name}.png/pdf')


# ─── CIFAR-10 test loader ──────────────────────────────────────────────────────
def get_test_loader(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ─── Helper: run inference and return predictions ──────────────────────────────
def get_predictions(model_path, config):
    model = create_cnn_from_config(config)
    ckpt = torch.load(model_path, map_location=DEVICE)
    # Some checkpoints wrap in a dict, others are raw state_dicts
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()

    loader = get_test_loader()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 1 — Aggregated Learning Curves with Confidence Bands
# ══════════════════════════════════════════════════════════════════════════════
print('\n📊 E-Fig 1: Aggregated Learning Curves with Confidence Bands')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, metric_key, ylabel, title in [
    (axes[0], 'val_loss',  'Validation Loss',          'Validation Loss (mean ± std)'),
    (axes[1], 'val_acc',   'Validation Accuracy (%)',  'Validation Accuracy (mean ± std)'),
]:
    for lbl, tk in [('Random Search', 'rs_training'), ('PSO', 'pso_training'), ('Bayesian Opt.', 'bo_training')]:
        series = [d[tk][metric_key] for d in all_data if d[tk] and metric_key in d[tk]]
        if not series:
            continue
        max_len = max(len(s) for s in series)
        padded = [s + [s[-1]] * (max_len - len(s)) for s in series]
        arr = np.array(padded)
        mean, std = arr.mean(0), arr.std(0)
        epochs = np.arange(1, max_len + 1)
        ax.plot(epochs, mean, linewidth=2.5, label=lbl, color=COLORS[lbl])
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.18, color=COLORS[lbl])
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle('Best-Model Learning Curves — Mean ± Std over 3 Seeds',
             fontsize=16, fontweight='bold')
plt.tight_layout()
save('efig1_learning_curves_confidence')


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 2 — Parallel Coordinates Plot (Search Space Exploration)
# ══════════════════════════════════════════════════════════════════════════════
print('📊 E-Fig 2: Parallel Coordinates')

rows = []
for d in all_data:
    for lbl, hk in [('Random Search','rs_history'),('PSO','pso_history'),('Bayesian Opt.','bo_history')]:
        for h in d[hk]:
            row = dict(h['config'])
            row['val_accuracy'] = h['score']
            row['method'] = lbl
            rows.append(row)

df_pc = pd.DataFrame(rows)
hp_cols = ['learning_rate', 'batch_size', 'conv_channels_base',
           'num_conv_blocks', 'fc_hidden', 'dropout', 'weight_decay', 'val_accuracy']
log_cols = {'learning_rate', 'weight_decay'}

# Normalize each column to [0,1] for plotting
df_norm = df_pc[hp_cols].copy()
for col in hp_cols:
    if col in log_cols:
        df_norm[col] = np.log10(df_norm[col])
    mn, mx = df_norm[col].min(), df_norm[col].max()
    df_norm[col] = (df_norm[col] - mn) / (mx - mn + 1e-9)

fig, ax = plt.subplots(figsize=(18, 6))
cmap = plt.cm.RdYlGn  # red = low accuracy, green = high
norm_acc = plt.Normalize(df_pc['val_accuracy'].min(), df_pc['val_accuracy'].max())

for _, row in df_pc.iterrows():
    ys = [df_norm.loc[_, col] for col in hp_cols]
    color = cmap(norm_acc(row['val_accuracy']))
    ax.plot(range(len(hp_cols)), ys, color=color, alpha=0.25, linewidth=0.8)

# Bold lines for the best config of each method
for lbl, hk in [('Random Search','rs_history'),('PSO','pso_history'),('Bayesian Opt.','bo_history')]:
    best_rows = [h for d in all_data for h in d[hk]]
    best = max(best_rows, key=lambda x: x['score'])
    idx_best = df_pc.index[(df_pc['method'] == lbl) & (df_pc['val_accuracy'] == best['score'])][0]
    ys = [df_norm.loc[idx_best, col] for col in hp_cols]
    ax.plot(range(len(hp_cols)), ys, color=COLORS[lbl], linewidth=3,
            label=f'{lbl} best', marker=MARKERS[lbl], markersize=8, zorder=5)

ax.set_xticks(range(len(hp_cols)))
ax.set_xticklabels([c.replace('_', '\n') for c in hp_cols], fontsize=10)
ax.set_ylabel('Normalized Value', fontsize=12)
ax.set_title('Parallel Coordinates — Search Space Exploration\n(Color = validation accuracy, bold = best config per method)',
             fontsize=14, fontweight='bold')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_acc)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Validation Accuracy (%)', shrink=0.8)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save('efig2_parallel_coordinates')


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 3 — Violin Plot of All Trial Accuracies
# ══════════════════════════════════════════════════════════════════════════════
print('📊 E-Fig 3: Violin Plot')
fig, ax = plt.subplots(figsize=(11, 6))
violin_data = {}
for lbl, hk in [('Random Search','rs_history'),('PSO','pso_history'),('Bayesian Opt.','bo_history')]:
    violin_data[lbl] = [h['score'] for d in all_data for h in d[hk]]

parts = ax.violinplot(list(violin_data.values()), positions=[1,2,3],
                      showmeans=True, showmedians=True, showextrema=True)
for i, (pc, lbl) in enumerate(zip(parts['bodies'], violin_data.keys())):
    pc.set_facecolor(COLORS[lbl])
    pc.set_alpha(0.7)
for part_name in ('cmeans', 'cmedians', 'cbars', 'cminima', 'cmaxima'):
    if part_name in parts:
        parts[part_name].set_color('black')
        parts[part_name].set_linewidth(1.5)

for i, (lbl, vals) in enumerate(violin_data.items(), 1):
    jitter = np.random.normal(i, 0.04, len(vals))
    ax.scatter(jitter, vals, s=20, alpha=0.5, color=COLORS[lbl], edgecolors='none', zorder=3)

ax.set_xticks([1,2,3])
ax.set_xticklabels(list(violin_data.keys()), fontsize=13)
ax.set_ylabel('Validation Accuracy (%)', fontsize=13)
ax.set_title('Distribution of ALL Trial Accuracies per Method\n(mean=◆, median=─)',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.35, linestyle='--')
plt.tight_layout()
save('efig3_violin_plot')


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 4 — 1D Sensitivity / Ablation Analysis
# ══════════════════════════════════════════════════════════════════════════════
print('📊 E-Fig 4: Hyperparameter Sensitivity')
hp_display = {
    'learning_rate': 'Learning Rate',
    'batch_size': 'Batch Size',
    'conv_channels_base': 'Conv Channels Base',
    'num_conv_blocks': 'Num Conv Blocks',
    'fc_hidden': 'FC Hidden Size',
    'dropout': 'Dropout',
    'weight_decay': 'Weight Decay',
}
log_hp = {'learning_rate', 'weight_decay'}

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for idx, (hp_key, hp_name) in enumerate(hp_display.items()):
    ax = axes[idx]
    for lbl, hk in [('Random Search','rs_history'),('PSO','pso_history'),('Bayesian Opt.','bo_history')]:
        xs = np.array([h['config'][hp_key] for d in all_data for h in d[hk]])
        ys = np.array([h['score'] for d in all_data for h in d[hk]])
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        # Binned means
        n_bins = 8
        bins = np.array_split(np.arange(len(xs)), n_bins)
        bx = [xs[b].mean() for b in bins]
        by = [ys[b].mean() for b in bins]
        be = [ys[b].std() for b in bins]
        ax.plot(bx, by, marker=MARKERS[lbl], linewidth=2, color=COLORS[lbl], label=lbl, markersize=6)
        ax.fill_between(bx, np.array(by)-np.array(be), np.array(by)+np.array(be), alpha=0.15, color=COLORS[lbl])

    ax.set_xlabel(hp_name, fontsize=11)
    ax.set_ylabel('Val Accuracy (%)', fontsize=11)
    ax.set_title(f'Sensitivity: {hp_name}', fontsize=12, fontweight='bold')
    if hp_key in log_hp:
        ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].axis('off')
plt.suptitle('Hyperparameter Sensitivity Analysis (Binned Mean ± Std)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
save('efig4_sensitivity_ablation')


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 5 — Cost-Benefit Pareto Plot (Accuracy vs #Evaluations)
# ══════════════════════════════════════════════════════════════════════════════
print('📊 E-Fig 5: Cost-Benefit Pareto Plot')
fig, ax = plt.subplots(figsize=(10, 7))

for lbl, hk in [('Random Search','rs_history'),('PSO','pso_history'),('Bayesian Opt.','bo_history')]:
    for run_idx, d in enumerate(all_data):
        hist = d[hk]
        if not hist:
            continue
        bsf = np.maximum.accumulate([h['score'] for h in hist])
        xs = np.arange(1, len(bsf)+1)
        ax.plot(xs, bsf, color=COLORS[lbl], linewidth=1.8, alpha=0.5,
                linestyle=['--', '-.', ':'][run_idx])

    # Mean trajectory
    max_e = max(len(d[hk]) for d in all_data if d[hk])
    mat = []
    for d in all_data:
        if not d[hk]:
            continue
        bsf = list(np.maximum.accumulate([h['score'] for h in d[hk]]))
        bsf += [bsf[-1]] * (max_e - len(bsf))
        mat.append(bsf)
    mean_traj = np.array(mat).mean(axis=0)
    ax.plot(np.arange(1, max_e+1), mean_traj, color=COLORS[lbl],
            linewidth=3.5, label=f'{lbl} (mean)', marker=MARKERS[lbl],
            markersize=7, markevery=max(1, max_e//8))

ax.set_xlabel('Number of Evaluations Used', fontsize=13)
ax.set_ylabel('Best Validation Accuracy Found (%)', fontsize=13)
ax.set_title('Cost-Benefit Frontier\n(Thin lines = individual seeds, thick = mean)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([65, 95])
plt.tight_layout()
save('efig5_cost_benefit_pareto')


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 6 — Confusion Matrices (Best model per method, seed=42)
# ══════════════════════════════════════════════════════════════════════════════
print('📊 E-Fig 6: Confusion Matrices (loading & inferring models, this may take a minute...)')
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

method_info = [
    ('Random Search', 'rs_history',  'rs_model_path'),
    ('PSO',           'pso_history', 'pso_model_path'),
    ('Bayesian Opt.', 'bo_history',  'bo_model_path'),
]

data0 = all_data[0]  # Use first seed (seed=42) for confusion matrices

for ax, (lbl, hk, mpk) in zip(axes, method_info):
    model_path = data0[mpk]
    best_config = max(data0[hk], key=lambda h: h['score'])['config']

    if model_path is None or not model_path.exists():
        ax.text(0.5, 0.5, f'Model not found\n({lbl})', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(lbl, fontsize=13, fontweight='bold')
        continue

    try:
        preds, labels = get_predictions(model_path, best_config)
        # Compute confusion matrix manually
        cm = np.zeros((10, 10), dtype=int)
        for p, l in zip(preds, labels):
            cm[l, p] += 1
        # Normalize per row
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax,
                    xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
                    cmap='Blues', vmin=0, vmax=1,
                    cbar_kws={'label': 'Recall', 'shrink': 0.8})
        acc = (preds == labels).mean() * 100
        ax.set_title(f'{lbl}\nTest Acc = {acc:.2f}%', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:60]}', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, wrap=True)
        ax.set_title(lbl, fontsize=13, fontweight='bold')
        print(f'   ⚠️ Could not generate confusion matrix for {lbl}: {e}')

plt.suptitle('Confusion Matrices — Best Configuration per Method (Seed 42)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save('efig6_confusion_matrices')


# ══════════════════════════════════════════════════════════════════════════════
# E-FIG 7 — Per-Class Accuracy Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print('📊 E-Fig 7: Per-Class Accuracy')
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(10)
width = 0.26

data0 = all_data[0]
method_class_acc = {}

for lbl, hk, mpk in method_info:
    model_path = data0[mpk]
    best_config = max(data0[hk], key=lambda h: h['score'])['config']
    if model_path is None or not model_path.exists():
        method_class_acc[lbl] = [0.0] * 10
        continue
    try:
        preds, labels = get_predictions(model_path, best_config)
        class_acc = []
        for c in range(10):
            mask = labels == c
            class_acc.append(preds[mask].tolist().count(c) / mask.sum() * 100)
        method_class_acc[lbl] = class_acc
    except Exception as e:
        method_class_acc[lbl] = [0.0] * 10
        print(f'   ⚠️ Could not compute per-class accuracy for {lbl}: {e}')

for i, (lbl, hk, mpk) in enumerate(method_info):
    accs = method_class_acc.get(lbl, [0.0]*10)
    ax.bar(x + (i-1)*width, accs, width, label=lbl,
           color=COLORS[lbl], alpha=0.8, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha='right', fontsize=11)
ax.set_ylabel('Per-Class Accuracy (%)', fontsize=13)
ax.set_title('Per-Class Accuracy Comparison — RS vs PSO vs Bayesian Opt. (Seed 42)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim([0, 105])
ax.grid(axis='y', alpha=0.35, linestyle='--')
ax.axhline(y=90, color='gray', linestyle=':', linewidth=1.5, label='90% line')
plt.tight_layout()
save('efig7_per_class_accuracy')


print(f'\n{"="*60}')
print(f'✅ ALL 7 EXTENDED PLOTS SAVED TO: {OUTPUT_DIR}/')
print(f'{"="*60}')
