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

def calculate_timestamp_features(df, date_col='date'):
    """
    Calculate timestamp features from date column
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    date_col : str
        Name of date column
        
    Returns:
    --------
    DataFrame
        Dataframe with timestamp features
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['weekday'] = df[date_col].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # Quarter and month names
    df['quarter'] = df[date_col].dt.quarter
    df['month_name'] = df[date_col].dt.month_name()
    
    logger.info(f"Added timestamp features to dataframe")
    return df

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

def detect_data_anomalies(transactions_df):
    """
    Detect anomalies in transaction data
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Transaction data
        
    Returns:
    --------
    dict
        Dictionary with anomaly reports
    """
    anomalies = {}
    
    # Check for negative prices
    neg_prices = transactions_df[transactions_df['price'] < 0]
    if len(neg_prices) > 0:
        anomalies['negative_prices'] = {
            'count': len(neg_prices),
            'items': neg_prices['item_id'].unique().tolist(),
            'sample': neg_prices.head(5).to_dict(orient='records')
        }
    
    # Check for negative sales
    neg_sales = transactions_df[transactions_df['sales'] < 0]
    if len(neg_sales) > 0:
        anomalies['negative_sales'] = {
            'count': len(neg_sales),
            'items': neg_sales['item_id'].unique().tolist(),
            'sample': neg_sales.head(5).to_dict(orient='records')
        }
    
    # Check for extreme price changes (>100%)
    price_changes = transactions_df.sort_values(['item_id', 'date'])
    price_changes['prev_price'] = price_changes.groupby(['item_id'])['price'].shift(1)
    price_changes['price_change_pct'] = (price_changes['price'] / price_changes['prev_price'] - 1) * 100
    
    extreme_changes = price_changes[abs(price_changes['price_change_pct']) > 100].dropna()
    if len(extreme_changes) > 0:
        anomalies['extreme_price_changes'] = {
            'count': len(extreme_changes),
            'items': extreme_changes['item_id'].unique().tolist(),
            'sample': extreme_changes.head(5).to_dict(orient='records')
        }
    
    # Check for unusually high sales
    item_stats = transactions_df.groupby('item_id')['sales'].agg(['mean', 'std']).reset_index()
    for _, row in item_stats.iterrows():
        if row['std'] > 0:  # Avoid division by zero
            threshold = row['mean'] + 5 * row['std']  # 5 sigma
            high_sales = transactions_df[
                (transactions_df['item_id'] == row['item_id']) & 
                (transactions_df['sales'] > threshold)
            ]
            
            if len(high_sales) > 0:
                if 'unusually_high_sales' not in anomalies:
                    anomalies['unusually_high_sales'] = {'items': {}}
                
                anomalies['unusually_high_sales']['items'][row['item_id']] = {
                    'count': len(high_sales),
                    'mean': row['mean'],
                    'threshold': threshold,
                    'max_observed': high_sales['sales'].max(),
                    'sample': high_sales.head(3).to_dict(orient='records')
                }
    
    logger.info(f"Detected {len(anomalies)} types of anomalies in the data")
    return anomalies