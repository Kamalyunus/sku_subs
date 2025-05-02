#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for preprocessing transaction data.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
    logger.info("Starting data preprocessing")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.info("Converting date column to datetime")
        df['date'] = pd.to_datetime(df['date'])
    
    # Add time-based features
    logger.info("Adding time-based features")
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Handle missing values
    logger.info("Handling missing values")
    
    # For is_on_promotion and is_out_of_stock, missing usually means False
    df['is_on_promotion'] = df['is_on_promotion'].fillna(0).astype(int)
    df['is_out_of_stock'] = df['is_out_of_stock'].fillna(0).astype(int)
    
    # Check for negative or zero prices
    invalid_prices = (df['price'] <= 0).sum()
    if invalid_prices > 0:
        logger.warning(f"Found {invalid_prices} records with invalid prices (<=0)")
        df = df[df['price'] > 0]
    
    # Check for negative sales
    invalid_sales = (df['sales'] < 0).sum()
    if invalid_sales > 0:
        logger.warning(f"Found {invalid_sales} records with negative sales")
        df = df[df['sales'] >= 0]
    
    # Remove extreme outliers in sales
    for item in df['item_id'].unique():
        item_sales = df[df['item_id'] == item]['sales']
        if len(item_sales) > 10:  # Only if we have enough data
            q1, q3 = item_sales.quantile([0.25, 0.75])
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr
            
            outliers = (df['item_id'] == item) & (df['sales'] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"Removing {outlier_count} sales outliers for item {item}")
                df.loc[outliers, 'sales'] = np.nan
    
    # Drop rows where sales is still NaN after outlier removal
    df = df.dropna(subset=['sales'])
    
    logger.info(f"Preprocessing complete. Returning {len(df)} records")
    return df