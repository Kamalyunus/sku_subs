#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple CLI tool to visualize the relationship between two SKUs.
"""

import argparse
from src.visualize_relationship import main as visualize_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize relationship between two SKUs')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--item-a', required=True, help='Primary item ID')
    parser.add_argument('--item-b', required=True, help='Secondary item (potential substitute) ID')
    parser.add_argument('--output-dir', help='Directory to save visualization files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (INFO level)')
    args = parser.parse_args()
    
    visualize_main(args.config, args.item_a, args.item_b, args.output_dir, args.verbose)