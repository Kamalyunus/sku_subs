#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for creating features and pivot tables.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_pivots(transactions_df):
    """
    Create pivot tables from transaction data
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Preprocessed transaction data
        
    Returns:
    --------
    tuple
        (sales_pivot, price_pivot, promo_pivot, oos_pivot, control_pivot)
    """
    logger.info("Creating pivot tables")
    
    # Group by date and item
    daily_data = transactions_df.groupby(['date', 'item_id']).agg({
        'sales': 'sum',
        'price': 'mean',
        'is_on_promotion': 'max',
        'is_out_of_stock': 'max',
        'is_weekend': 'first',
        'weekday': 'first',
        'month': 'first'
    }).reset_index()
    
    # Create sales pivot (date is the only index)
    logger.info("Creating sales pivot")
    sales_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='sales',
        fill_value=0
    )
    
    # Create price pivot
    logger.info("Creating price pivot")
    price_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='price'
    )
    
    # Create promotion pivot
    logger.info("Creating promotion pivot")
    promo_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='is_on_promotion',
        fill_value=0
    )
    
    # Create OOS pivot
    logger.info("Creating OOS pivot")
    oos_pivot = daily_data.pivot_table(
        index='date', 
        columns='item_id', 
        values='is_out_of_stock',
        fill_value=0
    )
    
    # Create control variables pivot
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
    
    logger.info("Pivot tables created successfully")
    return sales_pivot, price_pivot, promo_pivot, oos_pivot, control_pivot

def separate_price_changes(transactions_df, items_list, baseline_window=30, min_periods=7, discount_threshold=0.05):
    """
    Separate price changes into promotional discounts vs. competitive price matching
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Transaction data
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
    DataFrame
        Price change categorization by date and item
    DataFrame
        Discount percentages from baseline
    """
    logger.info("Separating price changes into promotion vs. price matching")
    
    # Group by date and item to get daily prices
    daily_prices = transactions_df.groupby(['date', 'item_id']).agg({
        'price': 'mean',
        'is_on_promotion': 'max'
    }).reset_index()
    
    # Create wide format
    price_wide = daily_prices.pivot(index='date', columns='item_id', values='price')
    promo_wide = daily_prices.pivot(index='date', columns='item_id', values='is_on_promotion')
    
    # Calculate baseline prices and price change types
    baseline_prices = {}
    price_change_types = {}
    
    for item in items_list:
        if item not in price_wide.columns:
            continue
            
        item_prices = price_wide[item].dropna()
        
        if item not in promo_wide.columns:
            item_promos = pd.Series(0, index=item_prices.index)
        else:
            item_promos = promo_wide[item].reindex(item_prices.index).fillna(0)
        
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
        
        # Categorize price changes
        change_type = pd.Series("normal", index=item_prices.index)
        
        # Promotion: on promotion flag and at least discount_threshold discount
        promo_mask = (item_promos == 1) & (discount >= discount_threshold)
        change_type[promo_mask] = "promotion"
        
        # Price matching: not on promotion but at least discount_threshold discount
        matching_mask = (item_promos == 0) & (discount >= discount_threshold)
        change_type[matching_mask] = "price_matching"
        
        # Store results
        price_change_types[item] = change_type
        baseline_prices[item] = baseline
    
    # Combine into dataframes
    result = pd.DataFrame(price_change_types)
    baseline_df = pd.DataFrame(baseline_prices)
    
    # Calculate discount percentages
    discount_df = (baseline_df - price_wide) / baseline_df
    
    logger.info("Price change separation complete")
    return result, discount_df