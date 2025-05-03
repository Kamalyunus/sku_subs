#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for the project.
"""

import pandas as pd
import numpy as np
import os
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to config file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def ensure_dir(directory):
    """
    Ensure directory exists, create if not
    
    Parameters:
    -----------
    directory : str
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def save_results(results, filepath):
    """
    Save results to pickle file
    
    Parameters:
    -----------
    results : object
        Results to save
    filepath : str
        Output filepath
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        pd.to_pickle(results, filepath)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def load_results(filepath):
    """
    Load results from pickle file
    
    Parameters:
    -----------
    filepath : str
        Input filepath
        
    Returns:
    --------
    object
        Loaded results
    """
    try:
        results = pd.read_pickle(filepath)
        logger.info(f"Results loaded from {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise

def filter_sparse_items(transactions_df, min_days=30):
    """
    Filter out items with sparse data
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Transaction data
    min_days : int
        Minimum number of days with data
        
    Returns:
    --------
    DataFrame
        Filtered transaction data
    list
        List of items with sufficient data
    """
    # Count days per item
    item_counts = transactions_df.groupby('item_id')['date'].nunique().rename('days_count')
    
    # Filter items with sufficient data
    valid_items = item_counts[item_counts >= min_days].index.tolist()
    
    # Filter transactions
    filtered_df = transactions_df[transactions_df['item_id'].isin(valid_items)]
    
    logger.info(f"Filtered from {transactions_df['item_id'].nunique()} to {len(valid_items)} items")
    logger.info(f"Retained {len(filtered_df)/len(transactions_df):.1%} of transaction records")
    
    return filtered_df, valid_items