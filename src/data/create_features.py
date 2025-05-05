#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for creating features and pivot tables.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_feature_set(transactions_df, items_list, baseline_window=30, min_periods=7, discount_threshold=0.05):
    """
    Consolidated function to create all features and pivots in one pass
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Preprocessed transaction data (at date-item level)
    items_list : list
        List of items to analyze
    baseline_window : int
        Window size for baseline price calculation
    min_periods : int
        Minimum periods required for baseline calculation
    discount_threshold : float
        Minimum discount percentage to consider significant
        
    Returns:
    --------
    dict
        Dictionary containing all created features and pivots:
        - sales_pivot: Sales pivot table
        - price_pivot: Price pivot table
        - promo_pivot: Promotion pivot table
        - oos_pivot: Out-of-stock pivot table
        - control_pivot: Control variables pivot table
        - price_change_types: Price change categorization
        - discount_df: Discount percentages from baseline
    """
    logger.info("Creating all features and pivots in one pass")
    results = {}

    # Create pivot tables
    daily_data = transactions_df

    # Create sales pivot
    logger.info("Creating sales pivot")
    sales_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='sales',
        fill_value=0
    )
    results['sales_pivot'] = sales_pivot
    
    # Create price pivot
    logger.info("Creating price pivot")
    price_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='price'
    )
    results['price_pivot'] = price_pivot
    
    # Create promotion pivot
    logger.info("Creating promotion pivot")
    promo_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='is_on_promotion',
        fill_value=0
    )
    results['promo_pivot'] = promo_pivot
    
    # Create OOS pivot
    logger.info("Creating OOS pivot")
    oos_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='is_out_of_stock',
        fill_value=0
    )
    results['oos_pivot'] = oos_pivot
    
    # Create control variables pivot using existing calculated fields
    logger.info("Creating control variables pivot")
    control_pivot = pd.DataFrame(index=sales_pivot.index)
    
    # Add is_weekend
    control_pivot['is_weekend'] = daily_data.groupby('date')['is_weekend'].first()
    
    # Add weekday dummies
    for day in range(7):
        control_pivot[f'weekday_{day}'] = (daily_data.groupby('date')['weekday'].first() == day).astype(int)
        
    # Add month dummies
    for month in range(1, 13):
        control_pivot[f'month_{month}'] = (daily_data.groupby('date')['month'].first() == month).astype(int)
    
    results['control_pivot'] = control_pivot
    
    # Process price changes - use precalculated price changes if available
    logger.info("Processing price changes and calculating baselines")
    price_wide = price_pivot.copy()
    
    # Create dictionaries for results
    price_change_types = {}
    
    for item in items_list:
        if item not in price_wide.columns:
            continue
            
        item_prices = price_wide[item].dropna()
        
        if item not in promo_pivot.columns:
            item_promos = pd.Series(0, index=item_prices.index)
        else:
            item_promos = promo_pivot[item].reindex(item_prices.index).fillna(0)
        
        # Calculate baseline from non-promotional periods
        non_promo_prices = item_prices[item_promos == 0]
        
        if len(non_promo_prices) < min_periods:
            logger.warning(f"Not enough non-promotional periods for item {item}")
            continue
            
        # Calculate rolling median for baseline
        baseline = non_promo_prices.rolling(window=baseline_window, min_periods=min_periods).median()
        
        # Forward fill to get baseline for all days
        baseline = baseline.reindex(item_prices.index).ffill().bfill()
        
        # Calculate percent discount from baseline
        discount = (baseline - item_prices) / baseline
        
        # Simplify price change categorization - we now rely on elasticity instead
        change_type = pd.Series("normal", index=item_prices.index)
        
        # Mark any significant price change, regardless of whether it's a promotion or not
        discount_mask = discount >= discount_threshold
        change_type[discount_mask] = "discount"
        
        # If promotion flag is set, mark as promotion
        promo_mask = item_promos == 1
        change_type[promo_mask] = "promotion"
        
        # Store results
        price_change_types[item] = change_type
    
    # Combine into dataframes
    price_types_df = pd.DataFrame(price_change_types)
    
    # Calculate discount percentages
    discount_df = pd.DataFrame(index=price_wide.index, columns=price_wide.columns)
    
    for item in items_list:
        if item not in price_wide.columns or item not in price_change_types:
            continue
        
        item_prices = price_wide[item].dropna()
        change_type = price_change_types[item]
        
        if item not in promo_pivot.columns:
            item_promos = pd.Series(0, index=item_prices.index)
        else:
            item_promos = promo_pivot[item].reindex(item_prices.index).fillna(0)
        
        # Calculate baseline from non-promotional periods
        non_promo_prices = item_prices[item_promos == 0]
        
        if len(non_promo_prices) < min_periods:
            continue
            
        # Calculate rolling median for baseline
        baseline = non_promo_prices.rolling(window=baseline_window, min_periods=min_periods).median()
        
        # Forward fill to get baseline for all days
        baseline = baseline.reindex(item_prices.index).ffill().bfill()
        
        # Calculate and store discount percentages
        discount = (baseline - item_prices) / baseline
        discount_df[item] = discount
    
    results['price_change_types'] = price_types_df
    results['discount_df'] = discount_df
    
    logger.info("All features and pivots created successfully")
    return results