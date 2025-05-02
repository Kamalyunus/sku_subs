#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for calculating cross-price elasticity between products.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

def calculate_cross_price_elasticity(sales_df, price_df, promo_df, item_a, item_b, control_vars=None, oos_df=None):
    """
    Calculate cross-price elasticity between two items
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with sales data
    price_df : DataFrame
        Pivot table with price data
    promo_df : DataFrame
        Pivot table with promotion flags
    item_a : str
        Item ID for which price changes are analyzed
    item_b : str
        Item ID for which sales changes are analyzed
    control_vars : DataFrame
        Control variables (e.g., weekday, month)
    oos_df : DataFrame, optional
        Pivot table with OOS flags (0/1), used to filter out and control for OOS periods
        
    Returns:
    --------
    dict
        Dictionary with elasticity results
    """
    logger.info(f"Calculating cross-price elasticity for {item_a} -> {item_b}")
    
    # Check if items exist in the data
    if item_a not in price_df.columns or item_b not in sales_df.columns:
        logger.warning(f"Missing data for items {item_a} or {item_b}")
        return {
            'elasticity': 0,
            'significant': False,
            'p_value': 1,
            'r_squared': 0,
            'sample_size': 0,
            'status': 'missing_data'
        }
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'sales_b': sales_df[item_b],
        'price_a': price_df[item_a]
    })
    
    # Add control variables
    if item_b in price_df.columns:
        analysis_df['price_b'] = price_df[item_b]
        
    if item_a in promo_df.columns:
        analysis_df['promo_a'] = promo_df[item_a]
        
    if item_b in promo_df.columns:
        analysis_df['promo_b'] = promo_df[item_b]
    
    # Add OOS status and use it for filtering
    oos_a_filter = True
    oos_b_filter = True
    
    if oos_df is not None:
        if item_a in oos_df.columns:
            analysis_df['oos_a'] = oos_df[item_a]
            # Filter out periods where item A is out of stock for elasticity calculation
            oos_a_filter = (analysis_df['oos_a'] == 0)
            
            # Add interaction term between price and OOS
            if 'price_a' in analysis_df.columns:
                analysis_df['price_a_x_oos_a'] = analysis_df['price_a'] * analysis_df['oos_a']
            
        if item_b in oos_df.columns:
            analysis_df['oos_b'] = oos_df[item_b]
            # Filter out periods where item B is out of stock for elasticity calculation
            oos_b_filter = (analysis_df['oos_b'] == 0)
            
            # Add interaction term between price and OOS
            if 'price_b' in analysis_df.columns:
                analysis_df['price_b_x_oos_b'] = analysis_df['price_b'] * analysis_df['oos_b']
    
    if control_vars is not None:
        analysis_df = pd.concat([analysis_df, control_vars], axis=1)
    
    # Filter out OOS periods for both items
    analysis_df_filtered = analysis_df[oos_a_filter & oos_b_filter].copy()
    
    # Log how many periods were filtered out
    filtered_rows = len(analysis_df) - len(analysis_df_filtered)
    if filtered_rows > 0:
        logger.info(f"Filtered out {filtered_rows} OOS periods from elasticity calculation for {item_a}->{item_b}")
    
    # Drop rows with missing values
    analysis_df_filtered = analysis_df_filtered.dropna(subset=['sales_b', 'price_a'])
    
    # Replace original dataframe with filtered version
    analysis_df = analysis_df_filtered
    
    if len(analysis_df) < 30:
        logger.warning(f"Not enough data points for elasticity calculation: {len(analysis_df)}")
        return {
            'elasticity': 0,
            'significant': False,
            'p_value': 1,
            'r_squared': 0,
            'sample_size': len(analysis_df),
            'status': 'insufficient_data'
        }
    
    try:
        # Create log values for elasticity calculation, handling zeros
        analysis_df['log_sales_b'] = np.log(analysis_df['sales_b'].replace(0, 0.01))
        analysis_df['log_price_a'] = np.log(analysis_df['price_a'])
        
        # Prepare regression formula
        formula_parts = ['log_sales_b ~ log_price_a']
        
        if 'price_b' in analysis_df.columns:
            analysis_df['log_price_b'] = np.log(analysis_df['price_b'])
            formula_parts.append('log_price_b')
            
        if 'promo_a' in analysis_df.columns:
            formula_parts.append('promo_a')
            
        if 'promo_b' in analysis_df.columns:
            formula_parts.append('promo_b')
        
        # Add OOS controls if still present after filtering
        # Though we filter out OOS periods, there might still be partial OOS days
        # or we might choose not to filter but only control in some cases
        if 'oos_a' in analysis_df.columns:
            formula_parts.append('oos_a')
            # Add interaction term
            if 'price_a_x_oos_a' in analysis_df.columns:
                formula_parts.append('price_a_x_oos_a')
            
        if 'oos_b' in analysis_df.columns:
            formula_parts.append('oos_b')
            # Add interaction term
            if 'price_b_x_oos_b' in analysis_df.columns:
                formula_parts.append('price_b_x_oos_b')
        
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            formula_parts.extend(control_cols)
        
        # Create formula
        formula = ' + '.join(formula_parts)
        
        # Run regression
        model = sm.formula.ols(formula=formula, data=analysis_df).fit()
        
        # Extract elasticity
        elasticity = model.params.get('log_price_a', 0)
        p_value = model.pvalues.get('log_price_a', 1)
        
        # Cross-price elasticity interpretation:
        # Positive elasticity = substitute (as price of A increases, sales of B increase)
        # Negative elasticity = complement (as price of A increases, sales of B decrease)
        
        return {
            'elasticity': elasticity,
            'significant': p_value < 0.05,
            'p_value': p_value,
            'r_squared': model.rsquared,
            'sample_size': len(analysis_df),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error calculating elasticity for {item_a}->{item_b}: {str(e)}")
        return {
            'elasticity': 0,
            'significant': False,
            'p_value': 1,
            'r_squared': 0,
            'sample_size': len(analysis_df),
            'status': 'error',
            'error_message': str(e)
        }

def calculate_elasticity_matrix(sales_df, price_df, promo_df, items_list, control_vars=None, oos_df=None):
    """
    Calculate cross-price elasticity matrix for multiple items
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with sales data
    price_df : DataFrame
        Pivot table with price data
    promo_df : DataFrame
        Pivot table with promotion flags
    items_list : list
        List of items to analyze
    control_vars : DataFrame
        Control variables
    oos_df : DataFrame, optional
        Pivot table with OOS flags (0/1), used to filter out OOS periods
        
    Returns:
    --------
    DataFrame
        Matrix of cross-price elasticities
    DataFrame
        Matrix indicating significance of elasticities
    """
    logger.info(f"Calculating elasticity matrix for {len(items_list)} items")
    
    # Create matrices with explicit data types
    elasticity_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    significance_matrix = pd.DataFrame(False, index=items_list, columns=items_list, dtype=bool)
    
    item_count = len(items_list)
    processed = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 5 == 0:
            logger.info(f"Elasticity Analysis: Processed {processed}/{item_count} items")
            
        for item_b in items_list:
            if item_a == item_b:
                continue
                
            result = calculate_cross_price_elasticity(
                sales_df, price_df, promo_df,
                item_a, item_b, control_vars, oos_df
            )
            
            if result['status'] == 'success':
                # Ensure we're explicitly setting float values for the elasticity
                elasticity_matrix.loc[item_a, item_b] = float(result['elasticity'])
                # Ensure we're explicitly setting boolean values for significance
                significance_matrix.loc[item_a, item_b] = bool(result['significant'])
    
    logger.info("Elasticity matrix calculation complete")
    return elasticity_matrix, significance_matrix