#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of price effects, including promotions, price matching, and cross-price elasticity.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from src.analysis.elasticity import calculate_cross_price_elasticity, calculate_elasticity_matrix

logger = logging.getLogger(__name__)

def calculate_price_effects(sales_df, price_df, promo_df, price_change_types, items_list, 
                            min_price_changes=5, control_vars=None, use_elasticity=True, oos_df=None):
    """
    Calculate price effects using cross-price elasticity
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    promo_df : DataFrame
        Pivot table with dates as index and items as columns, values are promo flags (0/1)
    price_change_types : DataFrame
        Price change categorization from separate_price_changes (not used in elasticity mode)
    items_list : list
        List of item IDs to analyze
    min_price_changes : int
        Minimum number of price change periods required for analysis (not used in elasticity mode)
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
    use_elasticity : bool, optional
        Parameter kept for backward compatibility (ignored, always True)
    oos_df : DataFrame, optional
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
        Used as a control variable to isolate price effects from availability effects
        
    Returns:
    --------
    tuple
        (price_effect_matrix, price_effect_type, significance_matrix)
    """
    logger.info("Calculating price effects using elasticity-based method")
    
    # Directly call the elasticity-based calculation
    return calculate_price_effects_with_elasticity(
        sales_df, price_df, promo_df, items_list, control_vars, oos_df
    )

def calculate_price_effects_with_elasticity(sales_df, price_df, promo_df, items_list, control_vars=None, oos_df=None):
    """
    Calculate price effects using cross-price elasticity from elasticity.py
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    promo_df : DataFrame
        Pivot table with dates as index and items as columns, values are promo flags (0/1)
    items_list : list
        List of item IDs to analyze
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
    oos_df : DataFrame, optional
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
        Used to control for and filter out OOS periods in elasticity calculations
        
    Returns:
    --------
    tuple
        (elasticity_matrix, elasticity_type, significance_matrix)
    """
    logger.info("Calculating cross-price elasticity between items")
    
    # Calculate elasticity matrix using elasticity.py function
    elasticity_matrix, significance_matrix = calculate_elasticity_matrix(
        sales_df, price_df, promo_df, items_list, control_vars, oos_df
    )
    
    # Create a matrix for elasticity type (substitutes vs complements)
    elasticity_type = pd.DataFrame("none", index=items_list, columns=items_list)
    
    # Positive elasticity = substitutes, negative = complements
    for item_a in items_list:
        for item_b in items_list:
            if item_a == item_b:
                continue
                
            if elasticity_matrix.loc[item_a, item_b] > 0:
                elasticity_type.loc[item_a, item_b] = "substitute"
            elif elasticity_matrix.loc[item_a, item_b] < 0:
                elasticity_type.loc[item_a, item_b] = "complement"
    
    logger.info("Elasticity-based price effect analysis complete")
    return elasticity_matrix, elasticity_type, significance_matrix

def analyze_individual_elasticity(sales_df, price_df, promo_df, item_a, item_b, control_vars=None, oos_df=None):
    """
    Perform detailed elasticity analysis for a specific item pair
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with sales data
    price_df : DataFrame
        Pivot table with price data
    promo_df : DataFrame
        Pivot table with promotion flags
    item_a : str
        Primary item
    item_b : str
        Potential substitute item
    control_vars : DataFrame, optional
        Control variables
    oos_df : DataFrame, optional
        Pivot table with OOS flags to control for availability
        
    Returns:
    --------
    dict
        Detailed elasticity results
    """
    logger.info(f"Analyzing elasticity relationship between {item_a} and {item_b}")
    
    result = calculate_cross_price_elasticity(
        sales_df, price_df, promo_df, item_a, item_b, control_vars, oos_df
    )
    
    # Add interpretation
    if result['status'] == 'success':
        elasticity = result['elasticity']
        if elasticity > 0 and result['significant']:
            result['interpretation'] = "Substitute relationship (significant)"
        elif elasticity > 0 and not result['significant']:
            result['interpretation'] = "Potential substitute (not significant)"
        elif elasticity < 0 and result['significant']:
            result['interpretation'] = "Complement relationship (significant)"
        elif elasticity < 0 and not result['significant']:
            result['interpretation'] = "Potential complement (not significant)"
        else:
            result['interpretation'] = "No relationship detected"
    
    return result