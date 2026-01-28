"""
Real-time computational cost monitoring for thesis documentation.
Tracks GPU usage, memory, and provides cost estimates.

Usage:
    python track_computational_cost.py --monitor_interval 60
"""

import argparse
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import subprocess


class ComputationalCostMonitor:
    """Monitor and log computational costs during experiments."""
    
    def __init__(self, output_file='computational_cost_log.json', monitor_interval=60):
        self.output_file = Path(output_file)
        self.monitor_interval = monitor_interval
        self.log_data = {
            'start_time': datetime.now().isoformat(),
            'snapshots': [],
            'summary': {}
        }
        
        self.has_nvidia_smi = self._check_nvidia_smi()
    
    def _check_nvidia_smi(self):
        """Check if nvidia-smi is available."""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_gpu_info(self):
        """Get current GPU utilization and memory usage."""
        if not self.has_nvidia_smi:
            return None
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            values = result.stdout.strip().split(', ')
            
            return {
                'gpu_utilization_percent': float(values[0]),
                'memory_used_mb': float(values[1]),
                'memory_total_mb': float(values[2]),
                'temperature_c': float(values[3]),
                'power_draw_w': float(values[4])
            }
        except:
            return None
    
    def take_snapshot(self):
        """Take a snapshot of current resource usage."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': (datetime.now() - datetime.fromisoformat(self.log_data['start_time'])).total_seconds()
        }
        
        gpu_info = self.get_gpu_info()
        if gpu_info:
            snapshot['gpu'] = gpu_info
        
        self.log_data['snapshots'].append(snapshot)
        self._save_log()
    
    def compute_summary(self):
        """Compute summary statistics."""
        if not self.log_data['snapshots']:
            return
        
        # Time elapsed
        start = datetime.fromisoformat(self.log_data['start_time'])
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        
        self.log_data['summary'] = {
            'end_time': end.isoformat(),
            'total_elapsed_seconds': elapsed,
            'total_elapsed_formatted': str(timedelta(seconds=int(elapsed))),
            'total_hours': elapsed / 3600
        }
        
        # GPU statistics
        if self.has_nvidia_smi and len(self.log_data['snapshots']) > 0:
            gpu_utils = [s['gpu']['gpu_utilization_percent'] for s in self.log_data['snapshots'] if 'gpu' in s]
            if gpu_utils:
                self.log_data['summary']['gpu'] = {
                    'mean_utilization_percent': sum(gpu_utils) / len(gpu_utils),
                    'max_utilization_percent': max(gpu_utils),
                    'estimated_gpu_hours': (sum(gpu_utils) / len(gpu_utils) / 100) * (elapsed / 3600)
                }
        
        self._save_log()
    
    def _save_log(self):
        """Save log to file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def monitor(self, duration_hours=None):
        """Start monitoring for specified duration."""
        print(f"🔍 Starting computational cost monitoring...")
        print(f"   Interval: {self.monitor_interval} seconds")
        print(f"   Output: {self.output_file}")
        if duration_hours:
            print(f"   Duration: {duration_hours} hours")
        print(f"   GPU monitoring: {'Enabled' if self.has_nvidia_smi else 'Disabled (nvidia-smi not found)'}")
        print("\n⏸️  Press Ctrl+C to stop monitoring\n")
        
        try:
            iteration = 0
            start_time = time.time()
            
            while True:
                iteration += 1
                self.take_snapshot()
                
                elapsed_hours = (time.time() - start_time) / 3600
                print(f"  Snapshot {iteration} at {datetime.now().strftime('%H:%M:%S')} (elapsed: {elapsed_hours:.2f}h)")
                
                if duration_hours and elapsed_hours >= duration_hours:
                    print(f"\n✅ Reached duration limit of {duration_hours} hours")
                    break
                
                time.sleep(self.monitor_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
        
        self.compute_summary()
        print(f"\n✅ Summary saved to: {self.output_file}")
        self.print_summary()
    
    def print_summary(self):
        """Print monitoring summary."""
        if 'summary' not in self.log_data:
            return
        
        summary = self.log_data['summary']
        
        print("\n" + "="*70)
        print("COMPUTATIONAL COST SUMMARY")
        print("="*70)
        print(f"Total time: {summary['total_elapsed_formatted']}")
        print(f"Total hours: {summary['total_hours']:.2f}")
        
        if 'gpu' in summary:
            print(f"\nGPU Statistics:")
            print(f"  Mean utilization: {summary['gpu']['mean_utilization_percent']:.1f}%")
            print(f"  Max utilization: {summary['gpu']['max_utilization_percent']:.1f}%")
            print(f"  Estimated GPU-hours: {summary['gpu']['estimated_gpu_hours']:.2f}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Monitor computational costs')
    parser.add_argument('--output', type=str, default='computational_cost_log.json',
                       help='Output JSON file')
    parser.add_argument('--monitor_interval', type=int, default=60,
                       help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration in hours (optional)')
    
    args = parser.parse_args()
    
    monitor = ComputationalCostMonitor(args.output, args.monitor_interval)
    monitor.monitor(args.duration)


if __name__ == "__main__":
    main()
