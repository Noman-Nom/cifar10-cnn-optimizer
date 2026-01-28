import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ast


"""
Quick visualization script for partial CIFAR-10 HPO results.

It parses the combined console log stored in `run.txt` and generates
several publication-quality plots (300 DPI) from whatever is already
available (even if the full experiment did not finish).

USAGE
=====
1. Make sure you are in the project root, i.e. the directory that contains `run.txt`:

       cd /home/muhammad-noman/noman/cifar10-hpo

2. Run:

       python visualize_partial_results.py

3. Plots will be saved under:

       partial_results_plots/

Generated figures
=================
- rs_retrain_curves.png    : Train/val accuracy for best Random Search retrain
- pso_retrain_curves.png   : Train/val accuracy for best PSO retrain
- rs_vs_pso_val_boxplot.png: Distribution of validation accuracies
                             (RS trials vs PSO particle evaluations)
- pso_convergence.png      : PSO global best validation vs iteration
"""


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "run.txt"
OUT_DIR = PROJECT_ROOT / "partial_results_plots"
OUT_DIR.mkdir(exist_ok=True)


def parse_log_sections(text: str):
    """
    Parse the raw log text into:
    - RS retrain epochs (best RANDOM_SEARCH model)
    - PSO retrain epochs (best PSO model)
    - RS exploration trial validation accuracies
    - PSO particle evaluation validation accuracies
    - PSO per-iteration global best validation (for convergence)
    """
    # Generic epoch line regex
    epoch_re = re.compile(
        r"Epoch (\d+)/\d+ - Train Loss: ([0-9.]+), Train Acc: ([0-9.]+)%, "
        r"Val Loss: ([0-9.]+), Val Acc: ([0-9.]+)%"
    )

    # Context tags for where we are in the log
    current_section = None  # "RS_RETRAIN", "PSO_RETRAIN", or None

    rs_retrain_rows = []
    pso_retrain_rows = []

    # RS exploration: "Trial k/25", "Config: {...}", and "Validation Accuracy: X%"
    rs_trials = []  # dicts with config + val_acc
    # PSO exploration: "PSO Iteration i/10", "Particle p/10: {...}", "Validation Accuracy: X%"
    pso_evals = []  # dicts with iteration, particle, config + val_acc

    # PSO convergence: iteration and "Global Best Score: X%"
    pso_iteration = None
    pso_convergence_rows = []

    lines = text.splitlines()

    # First pass: retrain epochs
    for line in lines:
        stripped = line.strip()

        # Detect which retrain phase we are in
        if "Retraining best RANDOM_SEARCH model" in stripped:
            current_section = "RS_RETRAIN"
            continue
        if "Retraining best PSO model" in stripped:
            current_section = "PSO_RETRAIN"
            continue
        # End of a retrain block is implicitly when another big header appears,
        # but we don't actually need to detect that explicitly; we just keep
        # collecting epoch lines while current_section is set.

        # Parse epoch lines
        m_epoch = epoch_re.search(stripped)
        if m_epoch and current_section is not None:
            epoch, tr_loss, tr_acc, val_loss, val_acc = m_epoch.groups()
            row = {
                "epoch": int(epoch),
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
            if current_section == "RS_RETRAIN":
                rs_retrain_rows.append(row)
            elif current_section == "PSO_RETRAIN":
                pso_retrain_rows.append(row)
            continue

    # Second pass: exploration (RS/PSO) + PSO convergence
    last_nonempty = ""
    last_context = None  # "RS_TRIAL", "PSO_PARTICLE", or None
    current_rs_config = None
    current_pso_config = None
    current_pso_particle = None

    for line in lines:
        stripped = line.strip()

        # RS: mark that we are in an RS trial
        if stripped.startswith("Trial ") and "Trial " in stripped:
            # Example: "Trial 1/25"
            last_context = "RS_TRIAL"

        # RS: capture config after "Trial k/25"
        if stripped.startswith("Config: {") and "Trial" not in stripped:
            # For RS, the previous non-empty line starts with "Trial"
            if last_context == "RS_TRIAL":
                cfg_str = stripped[len("Config: ") :].strip()
                try:
                    current_rs_config = ast.literal_eval(cfg_str)
                except Exception:
                    current_rs_config = None

        # PSO: mark that we are in a PSO particle evaluation
        if stripped.startswith("Particle ") and "{" not in stripped:
            # Example: "Particle 1/10:"
            last_context = "PSO_PARTICLE"

        # PSO: capture config when line already has dict
        if "Particle " in stripped and "{" in stripped and stripped.endswith("}"):
            # Example: "Particle 1/10: {...}"
            try:
                before, cfg_str = stripped.split(":", 1)
                cfg_str = cfg_str.strip()
                # Extract particle index
                parts = before.split()
                particle_token = parts[1]  # "1/10"
                particle_idx = int(particle_token.split("/")[0])
            except Exception:
                cfg_str = None
                particle_idx = None

            if cfg_str is not None:
                try:
                    current_pso_config = ast.literal_eval(cfg_str)
                    current_pso_particle = particle_idx
                except Exception:
                    current_pso_config = None
                    current_pso_particle = None

        if stripped.startswith("PSO Iteration") and "/" in stripped:
            # Example: "PSO Iteration 5/10"
            try:
                iter_part = stripped.split()[2]  # "5/10"
                iter_idx = int(iter_part.split("/")[0])
                pso_iteration = iter_idx
            except Exception:
                pso_iteration = None

        if stripped.startswith("Global Best Score:"):
            # Example: "Global Best Score: 89.1400%"
            val_str = stripped.split(":")[1].strip().rstrip("%")
            try:
                score = float(val_str)
                if pso_iteration is not None:
                    pso_convergence_rows.append(
                        {"iteration": pso_iteration, "global_best_val_acc": score}
                    )
            except ValueError:
                pass

        if stripped.startswith("Validation Accuracy:"):
            # Example: "Validation Accuracy: 87.5400%"
            val_str = stripped.split(":")[1].strip().rstrip("%")
            try:
                acc = float(val_str)
            except ValueError:
                acc = None

            if acc is not None:
                if last_context == "RS_TRIAL":
                    # RS trial
                    row = {"optimizer": "Random Search", "val_acc": acc}
                    if isinstance(current_rs_config, dict):
                        row.update(current_rs_config)
                    rs_trials.append(row)
                    current_rs_config = None
                    # After using this trial, reset context
                    last_context = None
                elif last_context == "PSO_PARTICLE":
                    # PSO particle evaluation
                    row = {
                        "optimizer": "PSO",
                        "val_acc": acc,
                    }
                    if pso_iteration is not None:
                        row["iteration"] = pso_iteration
                    if current_pso_particle is not None:
                        row["particle"] = current_pso_particle
                    if isinstance(current_pso_config, dict):
                        row.update(current_pso_config)
                    pso_evals.append(row)
                    current_pso_config = None
                    current_pso_particle = None
                    # Keep context as PSO_PARTICLE until a new particle/trial or blank line

        if stripped:
            last_nonempty = stripped

    rs_retrain_df = pd.DataFrame(rs_retrain_rows)
    pso_retrain_df = pd.DataFrame(pso_retrain_rows)
    pso_conv_df = pd.DataFrame(pso_convergence_rows).sort_values("iteration")
    rs_trials_df = pd.DataFrame(rs_trials)
    pso_evals_df = pd.DataFrame(pso_evals)

    return {
        "rs_retrain_df": rs_retrain_df,
        "pso_retrain_df": pso_retrain_df,
        "rs_trials_df": rs_trials_df,
        "pso_evals_df": pso_evals_df,
        "pso_convergence_df": pso_conv_df,
    }


def plot_retrain_curves(df: pd.DataFrame, title: str, out_path: Path):
    if df.empty:
        print(f"[WARN] No epoch data for {title}, skipping {out_path.name}")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_val_boxplot(rs_vals: np.ndarray, pso_vals: np.ndarray, out_path: Path):
    if rs_vals.size == 0 and pso_vals.size == 0:
        print("[WARN] No validation accuracies found for RS or PSO; skipping boxplot")
        return

    data = []
    if rs_vals.size > 0:
        for v in rs_vals:
            data.append({"optimizer": "Random Search", "val_acc": v})
    if pso_vals.size > 0:
        for v in pso_vals:
            data.append({"optimizer": "PSO", "val_acc": v})

    df = pd.DataFrame(data)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="optimizer", y="val_acc")
    sns.stripplot(data=df, x="optimizer", y="val_acc", color="black", alpha=0.4, jitter=0.2)
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracies: Random Search vs PSO (partial results)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_pso_convergence(conv_df: pd.DataFrame, out_path: Path):
    if conv_df.empty:
        print("[WARN] No PSO convergence data found; skipping convergence plot")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(conv_df["iteration"], conv_df["global_best_val_acc"], marker="o")
    plt.xlabel("PSO Iteration")
    plt.ylabel("Global Best Validation Accuracy (%)")
    plt.title("PSO Convergence (Global Best Validation Accuracy)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_hparam_scatter(df: pd.DataFrame, out_path_lr: Path, out_path_channels: Path):
    """
    Scatter plots:
    - learning_rate vs val_acc (log x)
    - conv_channels_base vs val_acc
    """
    if df.empty:
        print("[WARN] No hyperparameter data available; skipping scatter plots")
        return

    # Only keep rows that have val_acc
    df = df.copy()
    df = df.dropna(subset=["val_acc"])

    # 1) learning rate vs val_acc
    if "learning_rate" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x="learning_rate",
            y="val_acc",
            hue="optimizer",
            alpha=0.7,
        )
        plt.xscale("log")
        plt.xlabel("Learning rate (log scale)")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Learning rate vs Validation Accuracy")
        plt.tight_layout()
        plt.savefig(out_path_lr, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out_path_lr}")
    else:
        print("[WARN] No learning_rate column; skipping LR scatter")

    # 2) conv_channels_base vs val_acc
    if "conv_channels_base" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df,
            x="conv_channels_base",
            y="val_acc",
            hue="optimizer",
            alpha=0.7,
        )
        plt.xlabel("Base Conv Channels")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Base Conv Channels vs Validation Accuracy")
        plt.tight_layout()
        plt.savefig(out_path_channels, dpi=300)
        plt.close()
        print(f"[INFO] Saved {out_path_channels}")
    else:
        print("[WARN] No conv_channels_base column; skipping channels scatter")


def plot_blocks_violin(df: pd.DataFrame, out_path: Path):
    """
    Violin/strip plot of validation accuracy vs num_conv_blocks,
    combining RS and PSO evaluations.
    """
    if df.empty or "num_conv_blocks" not in df.columns:
        print("[WARN] No num_conv_blocks or val_acc data; skipping violin plot")
        return

    df = df.dropna(subset=["val_acc", "num_conv_blocks"])

    plt.figure(figsize=(6, 4))
    sns.violinplot(
        data=df,
        x="num_conv_blocks",
        y="val_acc",
        inner=None,
        cut=0,
    )
    sns.stripplot(
        data=df,
        x="num_conv_blocks",
        y="val_acc",
        hue="optimizer",
        dodge=True,
        alpha=0.6,
        linewidth=0.5,
    )
    plt.xlabel("Number of Conv Blocks")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Effect of Number of Conv Blocks on Validation Accuracy")
    plt.legend(title="Optimizer", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def plot_training_dynamics(
    rs_df: pd.DataFrame,
    pso_df: pd.DataFrame,
    out_path: Path,
):
    """
    2x2 figure:
      - RS loss (train/val)
      - PSO loss (train/val)
      - RS accuracy (train/val)
      - PSO accuracy (train/val)
    """
    if rs_df.empty and pso_df.empty:
        print("[WARN] No retrain data for RS or PSO; skipping training dynamics figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)

    # Helper to plot if df not empty
    def _plot_curves(df: pd.DataFrame, ax_loss, ax_acc, title_prefix: str):
        if df.empty:
            ax_loss.set_title(f"{title_prefix}: Loss Curves (no data)")
            ax_acc.set_title(f"{title_prefix}: Accuracy Curves (no data)")
            return
        ax_loss.plot(df["epoch"], df["train_loss"], label="Train")
        ax_loss.plot(df["epoch"], df["val_loss"], label="Validation")
        ax_loss.set_title(f"{title_prefix}: Loss Curves")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()

        ax_acc.plot(df["epoch"], df["train_acc"], label="Train")
        ax_acc.plot(df["epoch"], df["val_acc"], label="Validation")
        ax_acc.set_title(f"{title_prefix}: Accuracy Curves")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend()

    _plot_curves(rs_df, axes[0, 0], axes[1, 0], "Random Search")
    _plot_curves(pso_df, axes[0, 1], axes[1, 1], "PSO")

    fig.suptitle("Training Dynamics for Best Models (RS vs PSO)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_hparam_importance(df: pd.DataFrame, out_path: Path):
    """
    Horizontal bar plot of |correlation(val_acc, hparam)| for key hyperparameters.
    """
    if df.empty or "val_acc" not in df.columns:
        print("[WARN] No hyperparameter data for importance plot; skipping")
        return

    df = df.dropna(subset=["val_acc"])

    # Map from internal column names to nicer labels
    hparam_map = {
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "weight_decay": "Weight Decay",
        "dropout": "Dropout",
        "fc_hidden": "Hidden Size",
        "num_conv_blocks": "Num Layers",
    }

    rows = []
    for col, label in hparam_map.items():
        if col in df.columns:
            try:
                corr = df[col].corr(df["val_acc"])
            except Exception:
                continue
            if pd.isna(corr):
                continue
            rows.append({"hparam": label, "abs_corr": float(abs(corr))})

    if not rows:
        print("[WARN] No usable correlations for hyperparameter importance; skipping")
        return

    imp_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=True)

    plt.figure(figsize=(8, 4))
    plt.barh(imp_df["hparam"], imp_df["abs_corr"], color=sns.color_palette("viridis", len(imp_df)))
    for idx, row in imp_df.iterrows():
        plt.text(
            row["abs_corr"] + 0.005,
            idx,
            f"{row['abs_corr']:.3f}",
            va="center",
            fontsize=9,
        )
    plt.xlabel("Absolute Correlation with Validation Accuracy")
    plt.title("Hyperparameter Importance Analysis")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {out_path}")


def main():
    if not LOG_PATH.exists():
        print(f"[ERROR] Could not find log file at {LOG_PATH}")
        print("Please ensure `run.txt` exists in the project root.")
        return

    print(f"[INFO] Reading log from {LOG_PATH}")
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")

    parsed = parse_log_sections(text)

    # Retrain curves
    plot_retrain_curves(
        parsed["rs_retrain_df"],
        title="Best Random Search Retrain (Train vs Val Accuracy)",
        out_path=OUT_DIR / "rs_retrain_curves.png",
    )
    plot_retrain_curves(
        parsed["pso_retrain_df"],
        title="Best PSO Retrain (Train vs Val Accuracy)",
        out_path=OUT_DIR / "pso_retrain_curves.png",
    )

    # RS vs PSO validation distribution
    rs_vals = parsed["rs_trials_df"]["val_acc"].to_numpy(dtype=float) if not parsed["rs_trials_df"].empty else np.array([])
    pso_vals = parsed["pso_evals_df"]["val_acc"].to_numpy(dtype=float) if not parsed["pso_evals_df"].empty else np.array([])
    plot_val_boxplot(rs_vals, pso_vals, out_path=OUT_DIR / "rs_vs_pso_val_boxplot.png")

    # PSO convergence
    plot_pso_convergence(
        parsed["pso_convergence_df"],
        out_path=OUT_DIR / "pso_convergence.png",
    )

    # Hyperparameter scatter plots (combine RS + PSO)
    all_evals = pd.concat(
        [parsed["rs_trials_df"], parsed["pso_evals_df"]],
        ignore_index=True,
    )
    plot_hparam_scatter(
        all_evals,
        out_path_lr=OUT_DIR / "hparam_lr_vs_val.png",
        out_path_channels=OUT_DIR / "hparam_channels_vs_val.png",
    )

    # Effect of num_conv_blocks
    plot_blocks_violin(
        all_evals,
        out_path=OUT_DIR / "blocks_vs_val.png",
    )

    # 2x2 training dynamics figure
    plot_training_dynamics(
        parsed["rs_retrain_df"],
        parsed["pso_retrain_df"],
        out_path=OUT_DIR / "training_dynamics.png",
    )

    # Hyperparameter importance bar chart
    plot_hparam_importance(
        all_evals,
        out_path=OUT_DIR / "hparam_importance.png",
    )

    print("\n[INFO] Done. Plots are in:", OUT_DIR)


if __name__ == "__main__":
    main()


