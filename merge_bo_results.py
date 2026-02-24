import os
import json
import shutil
from pathlib import Path

# Paths
base_dir = Path("experiments/results")

old_dirs = [
    base_dir / "CIFAR10_CNN_20260127_234026", # seed 42
    base_dir / "CIFAR10_CNN_20260128_055854", # seed 123
    base_dir / "CIFAR10_CNN_20260128_115445"  # seed 999
]

new_bo_dir = base_dir / "CIFAR10_CNN_20260224_015229"
bo_source = new_bo_dir / "bo"

# BO runs
bo_runs = sorted(list(bo_source.glob("run_*"))) # run_42, run_43, run_44

# Load new BO summary
with open(new_bo_dir / "summary.json", "r") as f:
    bo_summary = json.load(f)["bo"]

for idx, old_dir in enumerate(old_dirs):
    # Determine new run directory name (run_42, run_123, run_999) based on mapping
    seeds = ["42", "123", "999"]
    target_seed = seeds[idx]
    
    # 1. Copy the BO run folder
    src_run = bo_runs[idx]
    dst_run_dir = old_dir / "bo" / f"run_{target_seed}"
    
    if dst_run_dir.exists():
        shutil.rmtree(dst_run_dir)
    
    shutil.copytree(src_run, dst_run_dir)
    print(f"Copied {src_run} -> {dst_run_dir}")
    
    # 2. Update summary.json in old_dir
    summary_path = old_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        # Inject the BO data specific to this run
        summary["bo"] = {
            "test_accuracies": [bo_summary["test_accuracies"][idx]],
            "best_configs": [bo_summary["best_configs"][idx]]
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated summary.json in {old_dir.name}")

print("Merge completed successfully.")
