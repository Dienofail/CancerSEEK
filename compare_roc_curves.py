#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for comparing ROC curves from multiple CancerSEEK models.
"""

import os
import argparse
import matplotlib.pyplot as plt
import plotting

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare ROC curves from multiple CancerSEEK models')
    
    parser.add_argument('--data-files', nargs='+', required=True,
                        help='List of ROC curve data files (.pkl) generated from different model runs')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Optional list of labels for each model (same order as data files)')
    parser.add_argument('--output', type=str, default='comparison_roc_curve.pdf',
                        help='Output file path for the comparison plot')
    parser.add_argument('--title', type=str, default='Comparison of ROC Curves',
                        help='Title for the comparison plot')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 10],
                        help='Figure size (width height) in inches')
    
    return parser.parse_args()

def main():
    """Main function to compare ROC curves."""
    # Get command line arguments
    args = parse_arguments()
    
    # Check if all data files exist
    for file_path in args.data_files:
        if not os.path.exists(file_path):
            print(f"Error: Data file {file_path} does not exist.")
            return
    
    # Generate comparison plot
    print(f"Comparing ROC curves from {len(args.data_files)} models...")
    fig, ax = plotting.plot_multiple_roc_curves(
        data_files=args.data_files,
        labels=args.labels,
        figsize=tuple(args.figsize),
        title=args.title,
        save_path=args.output
    )
    
    print(f"Saved comparison plot to {args.output}")
    
    # Show plot if running interactively
    plt.show()

if __name__ == "__main__":
    main() 