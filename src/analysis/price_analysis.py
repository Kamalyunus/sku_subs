#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of price effects using cross-price elasticity.
"""

import pandas as pd
import logging
from src.analysis.elasticity import calculate_cross_price_elasticity

logger = logging.getLogger(__name__)

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