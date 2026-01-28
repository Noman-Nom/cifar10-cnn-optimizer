"""
Generate publication-quality plots from experiment results.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.visualization import generate_all_plots


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate publication plots from experiment results')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for plots (default: results_dir/publication_plots)')
    
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir)

