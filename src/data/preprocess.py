#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for preprocessing transaction data and detecting anomalies.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_and_preprocess(df):
    """
    Consolidated function for data validation, preprocessing, and anomaly detection
    
    Parameters:
    -----------
    df : DataFrame
        Raw transaction data
        
    Returns:
    --------
    tuple
        (Preprocessed DataFrame, Dictionary with anomaly reports)
    """
    logger.info("Starting data validation and preprocessing")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Initialize anomalies dictionary
    anomalies = {}
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.info("Converting date column to datetime")
        df['date'] = pd.to_datetime(df['date'])
    
    # Add all time-based features at once
    logger.info("Adding time-based features")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['month_name'] = df['date'].dt.month_name()
    
    # Handle missing values
    logger.info("Handling missing values")
    
    # For is_on_promotion and is_out_of_stock, missing usually means False
    df['is_on_promotion'] = df['is_on_promotion'].fillna(0).astype(int)
    df['is_out_of_stock'] = df['is_out_of_stock'].fillna(0).astype(int)
    
    # Detect and handle negative prices
    neg_prices = df[df['price'] <= 0]
    if len(neg_prices) > 0:
        logger.warning(f"Found {len(neg_prices)} records with invalid prices (<=0)")
        anomalies['negative_prices'] = {
            'count': len(neg_prices),
            'items': neg_prices['item_id'].unique().tolist(),
            'sample': neg_prices.head(5).to_dict(orient='records')
        }
        # Remove negative prices
        df = df[df['price'] > 0]
    
    # Detect and handle negative sales
    neg_sales = df[df['sales'] < 0]
    if len(neg_sales) > 0:
        logger.warning(f"Found {len(neg_sales)} records with negative sales")
        anomalies['negative_sales'] = {
            'count': len(neg_sales),
            'items': neg_sales['item_id'].unique().tolist(),
            'sample': neg_sales.head(5).to_dict(orient='records')
        }
        # Remove negative sales
        df = df[df['sales'] >= 0]
    
    # Check for extreme price changes (>100%)
    logger.info("Checking for extreme price changes")
    df_sorted = df.sort_values(['item_id', 'date'])
    df_sorted['prev_price'] = df_sorted.groupby(['item_id'])['price'].shift(1)
    df_sorted['price_change_pct'] = (df_sorted['price'] / df_sorted['prev_price'] - 1) * 100
    
    extreme_changes = df_sorted[abs(df_sorted['price_change_pct']) > 100].dropna()
    if len(extreme_changes) > 0:
        logger.warning(f"Found {len(extreme_changes)} records with extreme price changes (>100%)")
        anomalies['extreme_price_changes'] = {
            'count': len(extreme_changes),
            'items': extreme_changes['item_id'].unique().tolist(),
            'sample': extreme_changes.head(5).to_dict(orient='records')
        }
    
    # Add price features
    logger.info("Adding price features")
    # Retain the price change percentage as a feature
    df['price_change_pct'] = df_sorted['price_change_pct']
    
    # Detect unusually high sales (outliers)
    logger.info("Detecting sales outliers")
    sales_outliers = 0
    
    item_stats = df.groupby('item_id')['sales'].agg(['mean', 'std']).reset_index()
    high_sales_items = {}
    
    for _, row in item_stats.iterrows():
        if row['std'] > 0:  # Avoid division by zero
            threshold = row['mean'] + 5 * row['std']  # 5 sigma
            high_sales = df[
                (df['item_id'] == row['item_id']) & 
                (df['sales'] > threshold)
            ]
            
            if len(high_sales) > 0:
                sales_outliers += len(high_sales)
                high_sales_items[row['item_id']] = {
                    'count': len(high_sales),
                    'mean': row['mean'],
                    'threshold': threshold,
                    'max_observed': high_sales['sales'].max(),
                    'sample': high_sales.head(3).to_dict(orient='records')
                }
                
                # Mark outliers (optionally, could remove them)
                outlier_idx = high_sales.index
                df.loc[outlier_idx, 'is_sales_outlier'] = 1
    
    if sales_outliers > 0:
        logger.warning(f"Found {sales_outliers} unusually high sales across {len(high_sales_items)} items")
        anomalies['unusually_high_sales'] = {'items': high_sales_items}
    
    # Make sure is_sales_outlier exists for all rows
    if 'is_sales_outlier' not in df.columns:
        df['is_sales_outlier'] = 0
    
    # Drop rows where sales is NaN
    df = df.dropna(subset=['sales'])
    
    logger.info(f"Preprocessing complete. Found {len(anomalies)} types of anomalies.")
    logger.info(f"Returning {len(df)} records")
    return df, anomalies

# Keep original function for backward compatibility, but use the new one internally
def preprocess_transactions(df):
    """
    Preprocess transaction data
    
    Parameters:
    -----------
    df : DataFrame
        Raw transaction data
        
    Returns:
    --------
    DataFrame
        Preprocessed transaction data
    """
    df_processed, _ = validate_and_preprocess(df)
    return df_processed