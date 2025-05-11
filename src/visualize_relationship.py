#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal visualization tool for SKU substitution relationships.
Creates visualizations to show the relationship between a pair of products.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta
import argparse
import logging

from src.substitution_analysis import (
    load_transaction_data, load_config, validate_and_preprocess,
    ensure_dir, create_feature_set, setup_logging
)

def visualize_sku_relationship(transactions_df, item_a, item_b, output_dir='data/results/figures'):
    """
    Create visualizations showing the relationship between two SKUs
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Preprocessed transaction data
    item_a : str
        Primary item ID
    item_b : str
        Secondary item (potential substitute) ID
    output_dir : str
        Directory to save visualization files
    
    Returns:
    --------
    tuple
        (success_flag, list_of_created_files)
    """
    # Ensure both items exist in the data
    if item_a not in transactions_df['item_id'].unique() or item_b not in transactions_df['item_id'].unique():
        print(f"Error: One or both items ({item_a}, {item_b}) not found in transaction data")
        return False, []
        
    # Create output directory if it doesn't exist
    ensure_dir(output_dir)
    
    # Filter data for the two items
    item_a_data = transactions_df[transactions_df['item_id'] == item_a]
    item_b_data = transactions_df[transactions_df['item_id'] == item_b]
    
    # Create pivot tables for analysis
    # Create common date range for all items
    all_dates = sorted(pd.unique(transactions_df['date']))
    date_range = pd.DataFrame({'date': all_dates})
    
    # Merge with item data
    item_a_daily = pd.merge(date_range, item_a_data, on='date', how='left').fillna({
        'sales': 0, 
        'is_out_of_stock': 0, 
        'is_on_promotion': 0
    })
    
    item_b_daily = pd.merge(date_range, item_b_data, on='date', how='left').fillna({
        'sales': 0, 
        'is_out_of_stock': 0, 
        'is_on_promotion': 0
    })
    
    # Set date as index
    item_a_daily.set_index('date', inplace=True)
    item_b_daily.set_index('date', inplace=True)
    
    # Create figures list to return
    created_files = []
    
    # Plot 1: Sales of A with OOS periods of B highlighted
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Plot sales of A
    plt.plot(item_a_daily.index, item_a_daily['sales'], label=f'Sales of {item_a}', color='blue', linewidth=2)
    
    # Highlight OOS periods of B
    oos_periods = item_b_daily[item_b_daily['is_out_of_stock'] == 1].index
    if len(oos_periods) > 0:
        for date in oos_periods:
            plt.axvspan(date - timedelta(days=0.5), date + timedelta(days=0.5), color='red', alpha=0.2)
    
    # Add OOS indicator on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(item_b_daily.index, item_b_daily['is_out_of_stock'], color='red', linestyle='--', 
             drawstyle='steps-post', alpha=0.7, label=f'{item_b} Out of Stock')
    ax2.set_ylabel(f'{item_b} OOS Status', color='red')
    ax2.set_ylim(-0.1, 1.1)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Format x-axis to show dates better
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()
    
    plt.title(f'Sales of {item_a} with {item_b} Out-of-Stock Periods')
    plt.xlabel('Date')
    ax.set_ylabel(f'Sales of {item_a}', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    oos_fig_path = os.path.join(output_dir, f'{item_a}_{item_b}_oos_relationship.png')
    plt.savefig(oos_fig_path)
    created_files.append(oos_fig_path)
    plt.close()
    
    # Plot 2: Sales of A with Price of B
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Plot sales of A
    ax.plot(item_a_daily.index, item_a_daily['sales'], label=f'Sales of {item_a}', color='blue', linewidth=2)
    
    # Plot price of B on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(item_b_daily.index, item_b_daily['price'], color='green', linewidth=2, label=f'Price of {item_b}')
    ax2.set_ylabel(f'Price of {item_b}', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()
    
    plt.title(f'Sales of {item_a} vs Price of {item_b}')
    plt.xlabel('Date')
    ax.set_ylabel(f'Sales of {item_a}', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    price_fig_path = os.path.join(output_dir, f'{item_a}_{item_b}_price_relationship.png')
    plt.savefig(price_fig_path)
    created_files.append(price_fig_path)
    plt.close()
    
    # Plot 3: Sales of A with Promotion periods of B highlighted
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Plot sales of A
    plt.plot(item_a_daily.index, item_a_daily['sales'], label=f'Sales of {item_a}', color='blue', linewidth=2)
    
    # Highlight promotion periods of B
    promo_periods = item_b_daily[item_b_daily['is_on_promotion'] == 1].index
    if len(promo_periods) > 0:
        for date in promo_periods:
            plt.axvspan(date - timedelta(days=0.5), date + timedelta(days=0.5), color='purple', alpha=0.2)
    
    # Add promotion indicator on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(item_b_daily.index, item_b_daily['is_on_promotion'], color='purple', linestyle='--', 
             drawstyle='steps-post', alpha=0.7, label=f'{item_b} On Promotion')
    ax2.set_ylabel(f'{item_b} Promotion Status', color='purple')
    ax2.set_ylim(-0.1, 1.1)
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()
    
    plt.title(f'Sales of {item_a} with {item_b} Promotion Periods')
    plt.xlabel('Date')
    ax.set_ylabel(f'Sales of {item_a}', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    promo_fig_path = os.path.join(output_dir, f'{item_a}_{item_b}_promo_relationship.png')
    plt.savefig(promo_fig_path)
    created_files.append(promo_fig_path)
    plt.close()
    
    print(f"Created {len(created_files)} visualization files in {output_dir}")
    return True, created_files

def main(config_path, item_a, item_b, output_dir=None, verbose=False):
    """
    Main function to load data and create visualizations for SKU relationships
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    item_a : str
        Primary item ID
    item_b : str
        Secondary item (potential substitute) ID
    output_dir : str
        Directory to save visualization files
    verbose : bool
        Enable verbose logging if True
    """
    # Set up logging
    setup_logging(verbose=verbose)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting visualization for items {item_a} and {item_b}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(config['data']['results_dir'], 'figures')
    
    # Load data
    logger.info("Loading transaction data")
    raw_transactions_df = load_transaction_data(config['data']['input_file'])
    
    # Run preprocessing
    logger.info("Running preprocessing and validation")
    transactions_df, anomalies = validate_and_preprocess(raw_transactions_df)
    
    # Create visualizations
    success, files = visualize_sku_relationship(transactions_df, item_a, item_b, output_dir)
    
    if success:
        print(f"Successfully created visualizations for {item_a} and {item_b}:")
        for file in files:
            print(f"  - {file}")
    else:
        print(f"Failed to create visualizations for {item_a} and {item_b}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize relationship between two SKUs')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--item-a', required=True, help='Primary item ID')
    parser.add_argument('--item-b', required=True, help='Secondary item (potential substitute) ID')
    parser.add_argument('--output-dir', help='Directory to save visualization files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (INFO level)')
    args = parser.parse_args()
    
    main(args.config, args.item_a, args.item_b, args.output_dir, args.verbose)