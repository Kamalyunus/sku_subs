#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of out-of-stock substitution effects.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from src.utils.validation import validate_substitution

logger = logging.getLogger(__name__)


def calculate_substitution_effects(sales_df, oos_df, price_df, promo_df, items_list, 
                                   min_oos_days=5, control_vars=None, product_attributes=None, 
                                   substitution_scope="category"):
    """
    Calculate substitution effects (OOS, price, and promo) using a combined model
    
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
        (oos_matrix, oos_significance, price_matrix, price_significance,
         promo_matrix, promo_significance, detailed_results)
    """
    logger.info(f"Calculating substitution effects with combined model (min_oos_days={min_oos_days}, scope={substitution_scope})")
    
    from src.utils.helpers import check_substitution_scope
    
    # Create matrices for different effects
    oos_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    oos_significance = pd.DataFrame(False, index=items_list, columns=items_list)
    
    price_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    price_significance = pd.DataFrame(False, index=items_list, columns=items_list)
    
    promo_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    promo_significance = pd.DataFrame(False, index=items_list, columns=items_list)
    
    detailed_results = {}
    
    item_count = len(items_list)
    processed = 0
    pairs_evaluated = 0
    pairs_skipped = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 10 == 0:
            logger.info(f"Substitution Analysis: Processed {processed}/{item_count} items")
        
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
            
            # Use combined validation function
            result = validate_substitution(
                sales_df, oos_df, price_df, promo_df, 
                item_a, item_b, control_vars
            )
            
            detailed_results[item_a][item_b] = result
            
            if result['validation_successful']:
                # Store OOS effects
                if result['oos_significant']:
                    oos_matrix.loc[item_a, item_b] = result['oos_effect']
                    oos_significance.loc[item_a, item_b] = True
                
                # Store Price effects (elasticity)
                if result['price_significant']:
                    price_matrix.loc[item_a, item_b] = result['price_effect']
                    price_significance.loc[item_a, item_b] = True
                
                # Store Promo effects
                if result['promo_significant']:
                    promo_matrix.loc[item_a, item_b] = result['promo_effect']
                    promo_significance.loc[item_a, item_b] = True
    
    logger.info(f"Substitution analysis complete: evaluated {pairs_evaluated} pairs, skipped {pairs_skipped} pairs")
    return (oos_matrix, oos_significance, 
            price_matrix, price_significance,
            promo_matrix, promo_significance,
            detailed_results)