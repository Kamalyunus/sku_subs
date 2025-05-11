#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimized SKU Substitution Analysis.
This script provides a streamlined interface to the substitution analysis pipeline.
"""

import argparse
from src.substitution_analysis import run_substitution_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Minimal SKU Substitution Analysis')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--csv', action='store_true', default=True, help='Export results to CSV format')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (INFO level)')
    args = parser.parse_args()
    
    run_substitution_analysis(args.config, export_csv=args.csv, verbose=args.verbose)