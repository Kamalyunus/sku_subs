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
    
    return filtered_df

def check_substitution_scope(item_a, item_b, product_attributes, substitution_scope="category"):
    """
    Check if two items are within the same substitution scope (category or subcategory)
    
    Parameters:
    -----------
    item_a : str
        First item ID
    item_b : str
        Second item ID
    product_attributes : DataFrame
        DataFrame with product attribute data, must have 'item_id' column
    substitution_scope : str
        Scope for substitution filtering: "category", "sub_category", or "all"
        
    Returns:
    --------
    bool
        True if items are in same category/subcategory (based on scope), False otherwise
    """
    # If no filtering or no attributes data, all pairs are valid
    if substitution_scope == "all" or product_attributes is None:
        return True
    
    # Make sure both items exist in attributes
    item_a_exists = item_a in product_attributes['item_id'].values
    item_b_exists = item_b in product_attributes['item_id'].values
    
    if not (item_a_exists and item_b_exists):
        return False
    
    # Get attributes for each item
    item_a_attr = product_attributes[product_attributes['item_id'] == item_a].iloc[0]
    item_b_attr = product_attributes[product_attributes['item_id'] == item_b].iloc[0]
    
    # Check if they're in the same category/subcategory
    if substitution_scope == "category":
        if 'category' not in item_a_attr or 'category' not in item_b_attr:
            return False
        return item_a_attr['category'] == item_b_attr['category']
    
    elif substitution_scope == "sub_category":
        if 'sub_category' not in item_a_attr or 'sub_category' not in item_b_attr:
            return False
        return item_a_attr['sub_category'] == item_b_attr['sub_category']
    
    return True