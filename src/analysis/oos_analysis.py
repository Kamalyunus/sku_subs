#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of out-of-stock substitution effects.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from src.utils.validation import validate_substitution_with_controls

logger = logging.getLogger(__name__)


def calculate_oos_substitution_with_validation(sales_df, oos_df, price_df, promo_df, items_list, 
                                     min_oos_days=5, control_vars=None, product_attributes=None, 
                                     substitution_scope="category"):
    """
    Calculate substitution effect using the enhanced validation framework
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    oos_df : DataFrame
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    promo_df : DataFrame
        Pivot table with promotion flags
    items_list : list
        List of item IDs to analyze
    min_oos_days : int
        Minimum number of OOS days required to consider a valid signal
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
    product_attributes : DataFrame, optional
        Product attributes data with category/subcategory information
    substitution_scope : str, optional
        Scope for substitution filtering: "category", "sub_category", or "all"
        
    Returns:
    --------
    tuple
        (substitution_matrix, significance_matrix, detailed_results)
    """
    logger.info(f"Calculating OOS substitution with validation (min_oos_days={min_oos_days}, scope={substitution_scope})")
    
    from src.utils.helpers import check_substitution_scope
    
    substitution_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    significance_matrix = pd.DataFrame(False, index=items_list, columns=items_list)
    detailed_results = {}
    
    item_count = len(items_list)
    processed = 0
    pairs_evaluated = 0
    pairs_skipped = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 10 == 0:
            logger.info(f"OOS Validation: Processed {processed}/{item_count} items")
        
        # Skip if item not in OOS data
        if item_a not in oos_df.columns:
            continue
            
        # Days when item A was OOS
        oos_days = oos_df[oos_df[item_a] == 1].index
        
        if len(oos_days) < min_oos_days:
            continue
        
        detailed_results[item_a] = {}
        
        for item_b in items_list:
            if item_a == item_b or item_b not in sales_df.columns:
                continue
            
            # Skip if not in same category/subcategory based on substitution scope
            if not check_substitution_scope(item_a, item_b, product_attributes, substitution_scope):
                pairs_skipped += 1
                continue
                
            pairs_evaluated += 1
            
            # Use validation function from validation.py
            result = validate_substitution_with_controls(
                sales_df, oos_df, price_df, promo_df, 
                item_a, item_b, control_vars
            )
            
            detailed_results[item_a][item_b] = result
            
            if result['validation_successful'] and result['oos_significant']:
                substitution_matrix.loc[item_a, item_b] = result['oos_effect']
                significance_matrix.loc[item_a, item_b] = True
    
    logger.info(f"OOS substitution validation complete: evaluated {pairs_evaluated} pairs, skipped {pairs_skipped} pairs")
    return substitution_matrix, significance_matrix, detailed_results