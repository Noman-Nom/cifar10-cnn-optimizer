"""
Run multiple hyperparameter optimization experiments with different seeds.
Tracks computational cost (wall-clock time, GPU hours) for thesis documentation.

Usage:
    python run_multiple_experiments.py --n_runs 3 --config experiments/config.yaml
"""

import argparse
import subprocess
import time
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import torch


class ExperimentTracker:
    """Track computational costs and results across multiple runs."""
    
    def __init__(self, base_config_path, output_dir):
        self.base_config_path = Path(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracking_data = {
            'start_time': datetime.now().isoformat(),
            'runs': [],
            'total_wall_clock_hours': 0,
            'total_gpu_hours': 0,
            'device': self._get_device_info()
        }
    
    def _get_device_info(self):
        """Get GPU/CPU information."""
        device_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            device_info['device_name'] = torch.cuda.get_device_name(0)
            device_info['device_capability'] = torch.cuda.get_device_capability(0)
        
        return device_info
    
    def run_experiment(self, seed, run_number):
        """Run a single experiment with given seed."""
        print("\n" + "="*80)
        print(f"STARTING RUN {run_number} WITH SEED={seed}")
        print("="*80 + "\n")
        
        run_data = {
            'run_number': run_number,
            'seed': seed,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        # Record start time
        start_wall_time = time.time()
        
        # Modify config with new seed
        config_path = self._create_config_with_seed(seed, run_number)
        
        # Run experiment
        try:
            result = subprocess.run(
                ['python', 'experiments/run_experiment.py', '--config', str(config_path)],
                capture_output=False,
                text=True,
                check=True
            )
            
            run_data['status'] = 'completed'
            run_data['exit_code'] = result.returncode
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ ERROR: Run {run_number} failed with exit code {e.returncode}")
            run_data['status'] = 'failed'
            run_data['exit_code'] = e.returncode
            run_data['error'] = str(e)
        
        # Record end time
        end_wall_time = time.time()
        wall_clock_hours = (end_wall_time - start_wall_time) / 3600
        
        run_data['end_time'] = datetime.now().isoformat()
        run_data['wall_clock_hours'] = wall_clock_hours
        run_data['wall_clock_formatted'] = str(timedelta(seconds=int(end_wall_time - start_wall_time)))
        
        # Estimate GPU hours (assuming single GPU usage)
        if self.tracking_data['device']['cuda_available']:
            run_data['gpu_hours'] = wall_clock_hours  # 1 GPU × wall_clock_hours
        else:
            run_data['gpu_hours'] = 0
        
        # Update totals
        self.tracking_data['total_wall_clock_hours'] += wall_clock_hours
        self.tracking_data['total_gpu_hours'] += run_data['gpu_hours']
        self.tracking_data['runs'].append(run_data)
        
        # Save tracking data
        self._save_tracking_data()
        
        print("\n" + "="*80)
        print(f"✅ RUN {run_number} COMPLETED")
        print(f"   Wall-clock time: {run_data['wall_clock_formatted']}")
        print(f"   GPU hours: {run_data['gpu_hours']:.2f}")
        print("="*80 + "\n")
        
        return run_data
    
    def _create_config_with_seed(self, seed, run_number):
        """Create a config file with modified seed."""
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update seed
        config['experiment']['base_seed'] = seed
        
        # Save to temporary config file
        temp_config_path = self.output_dir / f'config_seed_{seed}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        return temp_config_path
    
    def _save_tracking_data(self):
        """Save tracking data to JSON file."""
        tracking_file = self.output_dir / 'computational_cost_tracking.json'
        with open(tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
    
    def print_summary(self):
        """Print summary of all runs."""
        print("\n" + "="*80)
        print("MULTI-RUN EXPERIMENT SUMMARY")
        print("="*80)
        
        successful_runs = [r for r in self.tracking_data['runs'] if r['status'] == 'completed']
        failed_runs = [r for r in self.tracking_data['runs'] if r['status'] == 'failed']
        
        print(f"\n✅ Successful runs: {len(successful_runs)}/{len(self.tracking_data['runs'])}")
        if failed_runs:
            print(f"❌ Failed runs: {len(failed_runs)}")
        
        print(f"\n⏱️  Total wall-clock time: {timedelta(seconds=int(self.tracking_data['total_wall_clock_hours'] * 3600))}")
        print(f"🖥️  Total GPU hours: {self.tracking_data['total_gpu_hours']:.2f}")
        print(f"📊 Average time per run: {timedelta(seconds=int(self.tracking_data['total_wall_clock_hours'] * 3600 / len(self.tracking_data['runs'])))}")
        
        print("\n📋 Individual Run Times:")
        print("-" * 80)
        for run in self.tracking_data['runs']:
            status_icon = "✅" if run['status'] == 'completed' else "❌"
            print(f"  {status_icon} Run {run['run_number']} (seed={run['seed']}): {run.get('wall_clock_formatted', 'N/A')}")
        
        print("\n💾 Results saved to: " + str(self.output_dir))
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run multiple HPO experiments')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs with different seeds')
    parser.add_argument('--config', type=str, default='experiments/config.yaml', help='Base config file')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Custom seeds (optional)')
    parser.add_argument('--output_dir', type=str, default='multi_run_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Generate seeds
    if args.seeds:
        seeds = args.seeds
        if len(seeds) != args.n_runs:
            print(f"Warning: {len(seeds)} seeds provided but n_runs={args.n_runs}. Using first {args.n_runs} seeds.")
            seeds = seeds[:args.n_runs]
    else:
        # Default seeds: 42, 123, 999
        seeds = [42, 123, 999][:args.n_runs]
    
    print("="*80)
    print("MULTI-RUN HYPERPARAMETER OPTIMIZATION EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of runs: {args.n_runs}")
    print(f"  Seeds: {seeds}")
    print(f"  Base config: {args.config}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # Create tracker
    tracker = ExperimentTracker(args.config, args.output_dir)
    
    # Run experiments
    for i, seed in enumerate(seeds, 1):
        tracker.run_experiment(seed, i)
        
        # Small delay between runs
        if i < len(seeds):
            print("\n⏸️  Waiting 5 seconds before next run...\n")
            time.sleep(5)
    
    # Print final summary
    tracker.print_summary()
    
    print("\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"\n📂 Next steps:")
    print(f"   1. Run: python aggregate_results.py --results_dir {args.output_dir}")
    print(f"   2. Check: experiments/results/ for individual run outputs")
    print(f"   3. Generate plots with error bars for thesis\n")


if __name__ == "__main__":
    main()
